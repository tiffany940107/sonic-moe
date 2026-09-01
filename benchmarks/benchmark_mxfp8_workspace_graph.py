#!/usr/bin/env python3
"""K0 allocation reuse and CUDA Graph feasibility benchmark for SM120 MXFP8."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sonicmoe.functional.mxfp8 import (
    Mxfp8MoEKernelConfig,
    allocate_mxfp8_moe_workspace,
    allocate_mxfp8_weights,
    moe_mxfp8_grouped_forward,
    quantize_varlen_m_operand,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8192 * 32)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1280)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def bench(fn, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    values = torch.tensor(samples, dtype=torch.float64)
    return {
        "p50_ms": float(values.quantile(0.50)),
        "p95_ms": float(values.quantile(0.95)),
        "mean_ms": float(values.mean()),
    }


def main() -> int:
    args = parse_args()
    if torch.cuda.get_device_capability()[0] != 12:
        raise RuntimeError("physical SM120/SM121 is required")
    if args.rows < args.experts:
        raise ValueError("rows must be at least experts for this graph benchmark")
    base, remainder = divmod(args.rows, args.experts)
    lengths = torch.full(
        (args.experts,), base, dtype=torch.int32, device="cuda"
    )
    lengths[:remainder] += 1
    cu = torch.empty(args.experts + 1, dtype=torch.int32, device="cuda")
    cu[0] = 0
    torch.cumsum(lengths, dim=0, out=cu[1:])
    torch.manual_seed(902)
    x_hp = torch.randn(
        args.rows, args.hidden, dtype=torch.bfloat16, device="cuda"
    ) * 0.02
    x = quantize_varlen_m_operand(x_hp, cu)
    del x_hp
    w1 = allocate_mxfp8_weights(
        args.experts,
        2 * args.intermediate,
        args.hidden,
        device="cuda",
        seed=1902,
    )
    w2 = allocate_mxfp8_weights(
        args.experts,
        args.hidden,
        args.intermediate,
        device="cuda",
        seed=2902,
    )
    workspace = allocate_mxfp8_moe_workspace(
        args.experts,
        args.rows,
        args.hidden,
        args.intermediate,
        device="cuda",
    )
    config = Mxfp8MoEKernelConfig()

    def eager():
        return moe_mxfp8_grouped_forward(
            x, w1, w2, cu, config=config, workspace=workspace
        )[0]

    eager()
    torch.cuda.synchronize()
    eager_reference = eager().clone()
    eager_stats = bench(eager, args.warmup, args.iters)
    graph_status = "supported"
    graph_error = None
    graph_stats = None
    numeric = None
    try:
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            graph_output = eager()
        graph.replay()
        torch.cuda.synchronize()
        difference = graph_output.float() - eager_reference.float()
        numeric = {
            "max_abs": float(difference.abs().max()),
            "relative_l2": float(
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(eager_reference.float()).clamp_min(1e-20)
            ),
            "byte_exact": bool(torch.equal(graph_output, eager_reference)),
        }
        graph_stats = bench(graph.replay, args.warmup, args.iters)
    except Exception as error:  # Capture support is an audited capability gate.
        graph_status = "unsupported"
        graph_error = f"{type(error).__name__}: {error}"[:1000]

    record = {
        "schema_version": 1,
        "benchmark": "sonic_mxfp8_workspace_cuda_graph_k0",
        "timestamp_unix": time.time(),
        "shape": {
            "rows": args.rows,
            "experts": args.experts,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
        },
        "workspace_bytes": workspace.nbytes,
        "eager": eager_stats,
        "cuda_graph_status": graph_status,
        "cuda_graph_error": graph_error,
        "cuda_graph": graph_stats,
        "correctness": numeric,
    }
    if graph_stats is not None:
        record["graph_speedup"] = eager_stats["p50_ms"] / graph_stats["p50_ms"]
    payload = json.dumps(record, sort_keys=True)
    print(payload, flush=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
