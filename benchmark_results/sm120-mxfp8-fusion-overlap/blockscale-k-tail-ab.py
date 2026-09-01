#!/usr/bin/env python3
"""Commit-parametric regression for the block-scale MXFP8 MoE K tail."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--variant", choices=("old", "new"), required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 0:
        parser.error("warmup and iters must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "test"))
    from common import _init_op
    from utils.layout import per_token_cast_to_mxfp8_for_moe_gemm

    _init_op()
    counts = [0, 1, 127, 128, 129, 517]
    offsets = torch.tensor(
        [0, *torch.tensor(counts).cumsum(0).tolist()],
        dtype=torch.int32,
        device="cuda",
    )
    torch.manual_seed(902)
    x = torch.randn(sum(counts), 1280, dtype=torch.bfloat16, device="cuda")

    def invoke() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.custom_ops.fp8_quant_and_transform_for_moe(x, offsets, 32)

    actual_q, actual_sf = invoke()
    expected_q, expected_sf = per_token_cast_to_mxfp8_for_moe_gemm(
        x, offsets, gran_k=32
    )
    torch.cuda.synchronize()
    q_bad = int(
        (actual_q.view(torch.uint8) != expected_q.view(torch.uint8)).sum().item()
    )
    sf_bad = int(
        (
            actual_sf.contiguous().view(torch.uint8)
            != expected_sf.contiguous().view(torch.uint8)
        ).sum().item()
    )

    latency = None
    if args.iters:
        for _ in range(args.warmup):
            invoke()
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            invoke()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
        latency = {
            "count": len(samples),
            "min_ms": min(samples),
            "mean_ms": sum(samples) / len(samples),
            "p50_ms": percentile(samples, 0.50),
            "p95_ms": percentile(samples, 0.95),
            "p99_ms": percentile(samples, 0.99),
            "max_ms": max(samples),
        }

    record = {
        "schema_version": 1,
        "benchmark": "blockscale_mxfp8_moe_k1280_tail_commit_ab",
        "source_commit": args.source_commit,
        "variant": args.variant,
        "shape": {
            "expert_segments": counts,
            "k": 1280,
            "gran_k": 32,
        },
        "qdata_bad_bytes": q_bad,
        "scale_bad_bytes": sf_bad,
        "finite": bool(torch.isfinite(actual_q.float()).all()),
        "warmup": args.warmup,
        "iterations": args.iters,
        "latency": latency,
        "status": "passed" if not q_bad and not sf_bad else "failed",
        "timestamp_unix": time.time(),
    }
    print(json.dumps(record, sort_keys=True), flush=True)
    if q_bad or sf_bad:
        raise AssertionError(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
