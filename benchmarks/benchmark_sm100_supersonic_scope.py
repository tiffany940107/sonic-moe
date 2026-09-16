# Copyright (c) 2026, SonicMoE contributors.
"""Benchmark the local-expert scope used by public SuperSonic-MoE results.

This benchmark starts from fixed, already-dispatched top-k routes. It includes
route-metadata construction, local expert forward/backward, input and router-
score gradients, and expert wgrad accumulation. It excludes router projection,
top-k selection, auxiliary loss, optimizer step, and communication.

The default shape and warmup/iteration counts match
``PFCCLab/supersonic-moe@76b4f4f``'s
``tests/ops/bench_mlpnode_topk_nsys.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
from sonicmoe.enums import ActivationType
from sonicmoe.functional.mxfp8 import (
    Mxfp8TrainingPolicy,
    Mxfp8Workspace,
    mxfp8_experts,
)
from sonicmoe.functional.triton_kernels import (
    TC_topk_router_metadata_triton_workspace,
    topk_router_workspace_shape,
)


@dataclass(frozen=True)
class _Metadata:
    expert_offsets: torch.Tensor
    x_gather_idx: torch.Tensor
    s_scatter_idx: torch.Tensor
    s_reverse_scatter_idx: torch.Tensor


def _percentile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    return ordered[min(round(q * (len(ordered) - 1)), len(ordered) - 1)]


def _gpu_projection_us(sqlite_path: Path, iterations: int) -> float:
    """Return merged GPU-busy time per iteration within the BENCH NVTX range."""
    with sqlite3.connect(sqlite_path) as connection:
        ranges = connection.execute(
            "SELECT start, end FROM NVTX_EVENTS WHERE text = 'BENCH'"
        ).fetchall()
        if not ranges:
            raise RuntimeError("the nsys sqlite file has no BENCH NVTX range")
        bench_start, bench_end = ranges[0]
        intervals = connection.execute(
            "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "WHERE start >= ? AND end <= ? ORDER BY start",
            (bench_start, bench_end),
        ).fetchall()
    if not intervals:
        raise RuntimeError("the BENCH NVTX range contains no CUDA kernels")

    merged = []
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return sum(end - start for start, end in merged) / iterations / 1000.0


def _make_routes(
    tokens: int,
    experts: int,
    top_k: int,
    imbalance: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if imbalance == "extreme":
        indices = torch.arange(top_k, dtype=torch.int32, device=device)
        indices = indices.unsqueeze(0).expand(tokens, top_k).contiguous()
    else:
        route_logits = torch.randn(tokens, experts, device=device)
        if imbalance == "skew":
            hot = torch.rand(tokens, device=device) < 0.8
            route_logits[hot, 0] += 100.0
        indices = route_logits.topk(top_k, dim=-1).indices.int()
    scores = torch.rand(tokens, top_k, dtype=torch.float32, device=device)
    scores = scores * 0.5 + 0.5
    scores /= scores.sum(dim=-1, keepdim=True)
    return indices, scores.requires_grad_()


def _metadata(
    indices: torch.Tensor,
    experts: int,
    scratch: torch.Tensor,
) -> _Metadata:
    tokens, top_k = indices.shape
    routes = tokens * top_k
    frequency = torch.empty(experts, dtype=torch.int32, device=indices.device)
    offsets = torch.empty(experts + 1, dtype=torch.int32, device=indices.device)
    gather = torch.empty(routes, dtype=torch.int32, device=indices.device)
    scatter = torch.empty(routes, dtype=torch.int32, device=indices.device)
    reverse = torch.empty(routes, dtype=torch.int32, device=indices.device)
    TC_topk_router_metadata_triton_workspace(
        indices,
        experts,
        frequency,
        offsets,
        gather,
        scatter,
        reverse,
        scratch,
    )
    return _Metadata(offsets, gather, scatter, reverse)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=3072)
    parser.add_argument("--intermediate", type=int, default=1536)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--imbalance", choices=("none", "skew", "extreme"), default="none"
    )
    parser.add_argument(
        "--metadata",
        choices=("included", "precomputed"),
        default="included",
        help="construct route metadata inside or before the timed region",
    )
    parser.add_argument(
        "--wgrad-accum",
        choices=("fp32", "bf16"),
        default="fp32",
        help="persistent TMA FP32 accumulator matches the SuperSonic scope",
    )
    parser.add_argument(
        "--extract",
        type=Path,
        help="extract merged GPU-projection time from an nsys sqlite export",
    )
    args = parser.parse_args()
    if args.extract is not None:
        print(f"{_gpu_projection_us(args.extract, args.iterations):.3f}")
        return
    if args.iterations < 1 or args.warmup < 1:
        parser.error("--warmup and --iterations must be positive")
    if not 1 <= args.top_k <= args.experts:
        parser.error("--top-k must be between 1 and --experts")
    if torch.cuda.get_device_properties(0).major != 10:
        raise RuntimeError("this benchmark requires SM100")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    indices, scores = _make_routes(
        args.tokens, args.experts, args.top_k, args.imbalance, device
    )
    x = (0.02 * torch.randn(args.tokens, args.hidden, device=device)).to(torch.bfloat16)
    x.requires_grad_()
    output_grad = (0.01 * torch.randn_like(x)).to(torch.bfloat16)
    w1_storage = (
        torch.randn(
            args.experts,
            2 * args.intermediate,
            args.hidden,
            dtype=torch.bfloat16,
            device=device,
        )
        / args.hidden**0.5
    )
    w2_storage = (
        torch.randn(
            args.experts,
            args.hidden,
            args.intermediate,
            dtype=torch.bfloat16,
            device=device,
        )
        / args.intermediate**0.5
    )
    w1 = w1_storage.permute(1, 2, 0)
    w2 = w2_storage.permute(1, 2, 0)
    workspace = Mxfp8Workspace(training_policy=Mxfp8TrainingPolicy())
    stream_key = workspace._stream_key(device)
    if args.wgrad_accum == "fp32":
        workspace.enable_fp32_wgrad_accumulation(stream_key=stream_key)
    scratch = torch.empty(
        topk_router_workspace_shape(args.tokens, args.experts, args.top_k),
        dtype=torch.int32,
        device=device,
    )
    static_metadata = (
        _metadata(indices, args.experts, scratch)
        if args.metadata == "precomputed"
        else None
    )

    def run() -> None:
        route_metadata = (
            static_metadata
            if static_metadata is not None
            else _metadata(indices, args.experts, scratch)
        )
        output = mxfp8_experts(
            x,
            w1,
            None,
            w2,
            None,
            scores,
            route_metadata.expert_offsets,
            route_metadata.x_gather_idx,
            route_metadata.s_scatter_idx,
            route_metadata.s_reverse_scatter_idx,
            workspace,
            stream_key,
            args.tokens,
            args.top_k,
            ActivationType.SWIGLU,
            False,
        )
        output.backward(output_grad)

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
    torch.cuda.nvtx.range_push("BENCH")
    for start, end in zip(starts, ends):
        start.record()
        run()
        end.record()
    ends[-1].synchronize()
    torch.cuda.nvtx.range_pop()
    samples_us = [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]

    counts = torch.bincount(indices.flatten().long(), minlength=args.experts).cpu()
    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "reference": "PFCCLab/supersonic-moe@76b4f4f",
        "configuration": {
            "dz_iso32": os.environ.get("SONICMOE_MXFP8_DZ_ISO32", "0"),
            "all_iso32": os.environ.get("SONICMOE_MXFP8_ALL_ISO32", "0"),
            "fast_finite_bf16_quant": os.environ.get(
                "SONICMOE_MXFP8_FAST_BF16_QUANT", "0"
            ),
            "varlen_k_block_n": os.environ.get("SONICMOE_MXFP8_VARLEN_K_BLOCK_N", "64"),
            "varlen_k_warps": os.environ.get("SONICMOE_MXFP8_VARLEN_K_WARPS", "4"),
            "fc1_cluster_m": os.environ.get("SONICMOE_MXFP8_FC1_CLUSTER_M", "1"),
        },
        "scope": {
            "metadata": args.metadata,
            "expert_forward_backward": True,
            "input_gradient": True,
            "router_score_gradient": True,
            "expert_wgrad": args.wgrad_accum,
            "router_projection_topk": False,
            "aux_loss": False,
            "optimizer_step": False,
            "communication": False,
            "route_padding": False,
        },
        "shape": {
            "tokens": args.tokens,
            "experts": args.experts,
            "top_k": args.top_k,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "routes": args.tokens * args.top_k,
            "expert_count_min": int(counts.min()),
            "expert_count_max": int(counts.max()),
        },
        "imbalance": args.imbalance,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "latency_us": {
            "mean": statistics.fmean(samples_us),
            "p50": _percentile(samples_us, 0.50),
            "p95": _percentile(samples_us, 0.95),
            "p99": _percentile(samples_us, 0.99),
        },
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
