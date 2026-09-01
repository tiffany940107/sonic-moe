#!/usr/bin/env python3
"""Measure customer Segment-M boundaries with one active and one empty expert."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from sonicmoe.functional.mxfp8 import (
    Mxfp8MoEKernelConfig,
    allocate_mxfp8_moe_workspace,
    allocate_mxfp8_weights,
    dequantize_varlen_m_operand,
    moe_mxfp8_grouped_forward,
    quantize_varlen_m_operand,
)


SEGMENT_M = (1, 127, 128, 129, 517, 2528, 16384)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1280)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--restart", type=int, required=True)
    parser.add_argument("--node-label", choices=["PRO5000-A", "PRO5000-B"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize(samples: torch.Tensor) -> dict[str, float | int]:
    quantiles = torch.quantile(
        samples, torch.tensor([0.50, 0.90, 0.95, 0.99], dtype=torch.float64)
    )
    return {
        "count": int(samples.numel()),
        "min_ms": float(samples.min()),
        "mean_ms": float(samples.mean()),
        "p50_ms": float(quantiles[0]),
        "p90_ms": float(quantiles[1]),
        "p95_ms": float(quantiles[2]),
        "p99_ms": float(quantiles[3]),
        "max_ms": float(samples.max()),
    }


@torch.inference_mode()
def run_case(segment_m: int, args: argparse.Namespace) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(90200 + segment_m)
    # Expert 0 is deliberately empty. Expert 1 covers the requested boundary.
    cu = torch.tensor([0, 0, segment_m], dtype=torch.int32, device="cuda")
    x_hp = torch.randn(
        segment_m, args.hidden, dtype=torch.bfloat16, device="cuda"
    ) * 0.015625
    x = quantize_varlen_m_operand(x_hp, cu)
    del x_hp
    w1 = allocate_mxfp8_weights(
        2, 2 * args.intermediate, args.hidden, device="cuda", seed=segment_m + 1
    )
    w2 = allocate_mxfp8_weights(
        2, args.hidden, args.intermediate, device="cuda", seed=segment_m + 2
    )
    workspace = allocate_mxfp8_moe_workspace(
        2, segment_m, args.hidden, args.intermediate, device="cuda"
    )
    config = Mxfp8MoEKernelConfig()

    out, postact = moe_mxfp8_grouped_forward(
        x, w1, w2, cu, config=config, workspace=workspace
    )
    torch.cuda.synchronize()

    rows = min(4, segment_m)
    x_dq = dequantize_varlen_m_operand(x, cu)[-rows:]
    postact_dq = dequantize_varlen_m_operand(postact, cu)[-rows:]
    w1_dq = w1.dequantize(torch.float32)[1]
    w2_dq = w2.dequantize(torch.float32)[1]
    preact = x_dq @ w1_dq.T
    postact_ref = torch.nn.functional.silu(preact[:, 0::2]) * preact[:, 1::2]
    output_ref = postact_dq @ w2_dq.T
    postact_rel_l2 = float(
        torch.linalg.vector_norm(postact_dq - postact_ref)
        / torch.linalg.vector_norm(postact_ref).clamp_min(1e-20)
    )
    difference = out[-rows:].float() - output_ref
    bad = difference.abs() > 0.05 + 0.05 * output_ref.abs()
    output_rel_l2 = float(
        torch.linalg.vector_norm(difference)
        / torch.linalg.vector_norm(output_ref).clamp_min(1e-20)
    )
    if int(bad.sum()) or output_rel_l2 > 5e-3 or not torch.isfinite(out).all():
        raise AssertionError(
            f"Segment-M={segment_m} correctness failed: bad={int(bad.sum())}, "
            f"relative_l2={output_rel_l2}"
        )

    for _ in range(args.warmup):
        moe_mxfp8_grouped_forward(x, w1, w2, cu, config=config, workspace=workspace)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    for start, end in zip(starts, ends):
        start.record()
        moe_mxfp8_grouped_forward(x, w1, w2, cu, config=config, workspace=workspace)
        end.record()
    torch.cuda.synchronize()
    samples = torch.tensor(
        [start.elapsed_time(end) for start, end in zip(starts, ends)],
        dtype=torch.float64,
    )
    return {
        "schema_version": 1,
        "benchmark": "sonic_mxfp8_segment_edge_sweep",
        "source_commit": os.environ.get("SONIC_COMMIT", "unknown"),
        "quack_commit": os.environ.get("QUACK_COMMIT", "unknown"),
        "node_label": args.node_label,
        "restart": args.restart,
        "segment_m": segment_m,
        "physical_experts": 2,
        "active_experts": 1,
        "empty_experts": 1,
        "hidden": args.hidden,
        "intermediate_after_swiglu": args.intermediate,
        "dtype": "OCP MXFP8 E4M3 values + E8M0 K32 scales",
        "scope": "prequantized local FC1+SwiGLU+requant+FC2",
        "latency": summarize(samples),
        "correctness": {
            "checked_rows": rows,
            "bad_count": int(bad.sum()),
            "output_relative_l2": output_rel_l2,
            "postact_relative_l2": postact_rel_l2,
        },
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        raise RuntimeError("physical SM120/SM121 hardware is required")
    records = [run_case(segment_m, args) for segment_m in SEGMENT_M]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps({"cases": len(records), "all_correct": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
