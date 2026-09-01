#!/usr/bin/env python3
"""Formal K1 benchmark: row-linear MXFP8 routes to local reduced output."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from ep4_support.transport import Dispatched, expand_and_sort
from quack.blockscaled import BlockScaledOperand

from sonicmoe.functional.mxfp8 import (
    Mxfp8MoEKernelConfig,
    allocate_mxfp8_moe_workspace,
    allocate_mxfp8_weights,
    make_varlen_m_operand,
    mxfp8_down_grouped,
    mxfp8_swiglu_grouped,
    mxfp8_swiglu_grouped_out,
    quantize_mxfp8_rows,
)
from sonicmoe.functional.mxfp8_route_pack import (
    allocate_mxfp8_route_pack_workspace,
    route_pack_mxfp8,
)
from sonicmoe.functional.mxfp8_weighted_reduce import (
    allocate_mxfp8_weighted_reduce_workspace,
    segmented_weighted_reduce_mxfp8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recv-tokens", type=int, default=32768)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--local-routes", type=int, default=8)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1280)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--node-label", default="anonymous")
    parser.add_argument("--run-label", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.node_label):
        parser.error("--node-label must be a privacy-safe alias")
    if not 0 < args.local_routes <= args.top_k:
        parser.error("local-routes must be in [1, top-k]")
    if args.recv_tokens <= 0 or args.experts <= 0:
        parser.error("recv-tokens and experts must be positive")
    return args


def summarize(values: list[float]) -> dict[str, float | int]:
    samples = torch.tensor(values, dtype=torch.float64)
    return {
        "count": len(values),
        "min_ms": float(samples.min()),
        "mean_ms": float(samples.mean()),
        "p50_ms": float(samples.quantile(0.50)),
        "p95_ms": float(samples.quantile(0.95)),
        "p99_ms": float(samples.quantile(0.99)),
        "max_ms": float(samples.max()),
    }


def main() -> int:
    args = parse_args()
    if torch.cuda.get_device_capability()[0] != 12:
        raise RuntimeError("physical SM120/SM121 is required")
    device = torch.device("cuda")
    torch.manual_seed(902)
    source = torch.randn(
        (args.recv_tokens, args.hidden), dtype=torch.bfloat16, device=device
    ) * 0.02
    source_q, source_scale = quantize_mxfp8_rows(source)
    del source

    slots = torch.arange(args.top_k, dtype=torch.int64, device=device)[None, :]
    tokens = torch.arange(args.recv_tokens, dtype=torch.int64, device=device)[:, None]
    valid = slots < args.local_routes
    expert_ids = ((tokens * args.local_routes + slots) % args.experts).to(torch.int32)
    expert_ids = torch.where(valid, expert_ids, torch.full_like(expert_ids, -1))
    weights = torch.where(
        valid,
        torch.full_like(expert_ids, 1.0 / args.top_k, dtype=torch.float32),
        torch.zeros_like(expert_ids, dtype=torch.float32),
    )
    total_pairs = args.recv_tokens * args.local_routes

    w1 = allocate_mxfp8_weights(
        args.experts,
        2 * args.intermediate,
        args.hidden,
        device=device,
        seed=1902,
    )
    w2 = allocate_mxfp8_weights(
        args.experts,
        args.hidden,
        args.intermediate,
        device=device,
        seed=2902,
    )
    route_workspace = allocate_mxfp8_route_pack_workspace(
        args.recv_tokens,
        total_pairs,
        args.top_k,
        args.experts,
        args.hidden,
        device=device,
    )
    mlp_workspace = allocate_mxfp8_moe_workspace(
        args.experts,
        total_pairs,
        args.hidden,
        args.intermediate,
        device=device,
    )
    reduce_workspace = allocate_mxfp8_weighted_reduce_workspace(
        args.recv_tokens, args.hidden, device=device, include_fp32=False
    )
    config = Mxfp8MoEKernelConfig()

    dispatched = Dispatched(
        source_q, expert_ids, weights, source_scale.view(torch.uint8)
    )
    pair_q, _, recv_token, pair_weight, reference_indptr = expand_and_sort(
        dispatched, args.experts
    )
    reference_scale = (
        source_scale.index_select(0, recv_token)
        .contiguous()
        .view(torch.float8_e8m0fnu)
    )
    reference_operand = make_varlen_m_operand(
        pair_q, reference_scale, reference_indptr
    )
    reference_postact = mxfp8_swiglu_grouped(
        reference_operand, w1, reference_indptr, config=config
    )
    reference_pairs = mxfp8_down_grouped(
        reference_postact, w2, reference_indptr, config=config
    )
    reference = torch.zeros(
        (args.recv_tokens, args.hidden), dtype=torch.float32, device=device
    )
    chunk = 65536
    for begin in range(0, total_pairs, chunk):
        end = min(begin + chunk, total_pairs)
        reference.index_add_(
            0,
            recv_token[begin:end],
            reference_pairs[begin:end].float() * pair_weight[begin:end, None],
        )
    reference = reference.to(torch.bfloat16)
    del reference_postact, reference_pairs, reference_operand, reference_scale
    del pair_q, recv_token, pair_weight

    def invoke() -> torch.Tensor:
        packed = route_pack_mxfp8(
            source_q,
            source_scale,
            expert_ids,
            weights,
            total_pairs,
            route_workspace,
        )
        postact_out = mlp_workspace.active_postact(total_pairs)
        postact: BlockScaledOperand = mxfp8_swiglu_grouped_out(
            packed.operand,
            w1,
            packed.indptr,
            postact_out,
            config=config,
        )
        pair_out = mxfp8_down_grouped(
            postact,
            w2,
            packed.indptr,
            config=config,
            out=mlp_workspace.active_fc2_output(total_pairs),
        )
        return segmented_weighted_reduce_mxfp8(
            pair_out,
            route_workspace.scatter_pos,
            packed.weights,
            args.recv_tokens,
            args.top_k,
            reduce_workspace,
        )

    actual = invoke()
    torch.cuda.synchronize()
    difference = actual.float() - reference.float()
    bad = difference.abs() > 0.05 + 0.05 * reference.float().abs()
    correctness = {
        "bad_count": int(bad.sum()),
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "relative_l2": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference.float()).clamp_min(1e-20)
        ),
    }
    if correctness["bad_count"] or correctness["relative_l2"] > 5e-3:
        raise AssertionError(f"K1 C4 failed: {correctness}")

    for _ in range(args.warmup):
        invoke()
    torch.cuda.synchronize()
    allocator_keys = (
        "allocation.all.allocated",
        "allocation.all.freed",
        "segment.all.allocated",
        "segment.all.freed",
        "num_alloc_retries",
        "num_ooms",
    )
    allocator_before = torch.cuda.memory_stats()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(args.iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        invoke()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    allocator_after = torch.cuda.memory_stats()
    record = {
        "schema_version": 1,
        "benchmark": "sonic_mxfp8_k1_local_route_compute_reduce",
        "timestamp_unix": time.time(),
        "node_label": args.node_label,
        "run_label": args.run_label,
        "timing_level": "K1_row_linear_mxfp8_route_to_local_reduced_output",
        "sonic_commit": os.environ.get("SONIC_COMMIT", "unknown"),
        "quack_commit": os.environ.get("QUACK_COMMIT", "unknown"),
        "shape": {
            "recv_tokens": args.recv_tokens,
            "top_k": args.top_k,
            "local_routes_per_token": args.local_routes,
            "routed_pairs": total_pairs,
            "experts": args.experts,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
        },
        "warmup": args.warmup,
        "iterations": args.iters,
        "latency": summarize(samples),
        "correctness": correctness,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "timed_allocator_delta": {
            key: int(allocator_after.get(key, 0) - allocator_before.get(key, 0))
            for key in allocator_keys
        },
        "workspace_bytes": {
            "route_pack": route_workspace.nbytes,
            "local_mlp": mlp_workspace.nbytes,
            "weighted_reduce": reduce_workspace.nbytes,
        },
    }
    payload = json.dumps(record, sort_keys=True)
    print(payload, flush=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
