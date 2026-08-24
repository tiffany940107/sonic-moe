#!/usr/bin/env python3
# ********************************************************************************
# Copyright (c) 2026 SonicMoE contributors
# ********************************************************************************
"""Benchmark pre-quantized SM120 MXFP8 variable-M grouped GEMM.

M is the total number of rows across groups, not the per-group maximum.  The
script prints the exact M_e split so balanced and intentionally ragged runs are
never accidentally compared as if they were the same workload.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from quack.blockscaled import unpack_scale_blocked_to_2d

from sonicmoe.functional.mxfp8 import (
    Mxfp8MoEKernelConfig,
    allocate_mxfp8_weights,
    dequantize_varlen_m_operand,
    mxfp8_grouped_gemm,
    quantize_varlen_m_operand,
)

WORKLOADS = {
    "mnk_8k_1280_2k": (8192, 1280, 2048),
    "mnk_8k_2k_1280": (8192, 2048, 1280),
    "mnk_16k_1280_2k": (16384, 1280, 2048),
    "mnk_16k_2k_1280": (16384, 2048, 1280),
    "mnk_32k_1280_2k": (32768, 1280, 2048),
    "mnk_32k_2k_1280": (32768, 2048, 1280),
}


def _comma_ints(value: str) -> list[int]:
    try:
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("group row counts must be non-negative")
    return result


def _parse_workloads(value: str) -> list[str]:
    if value == "all":
        return list(WORKLOADS)
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in WORKLOADS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown workloads {unknown}; choose from {list(WORKLOADS)}"
        )
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workloads",
        type=_parse_workloads,
        default=list(WORKLOADS),
        help="comma-separated workload names, or 'all' (default: all)",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=8,
        help="number of independent GEMMs/experts (default: 8)",
    )
    parser.add_argument(
        "--distribution",
        choices=("balanced", "ragged"),
        default="balanced",
        help="how total M is divided across groups",
    )
    parser.add_argument(
        "--group-ms",
        type=_comma_ints,
        help="exact M_e list; valid only when one workload is selected",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--check-rows", type=int, default=4)
    parser.add_argument("--check-groups", type=int, default=8)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--tile-m", type=int, choices=(128, 256), default=128)
    parser.add_argument("--tile-n", type=int, choices=(128, 256), default=128)
    parser.add_argument("--no-pingpong", action="store_true")
    parser.add_argument("--static-persistent", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--jsonl", type=Path, help="append one JSON object per workload"
    )
    args = parser.parse_args()
    if args.groups <= 0:
        parser.error("--groups must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be non-negative and --iterations must be positive")
    if args.check_rows <= 0 or args.check_groups < 0:
        parser.error("--check-rows must be positive and --check-groups non-negative")
    if args.group_ms is not None:
        if len(args.workloads) != 1:
            parser.error("--group-ms requires exactly one selected workload")
        if len(args.group_ms) != args.groups:
            parser.error("--group-ms length must equal --groups")
    return args


def _group_rows(total_m: int, groups: int, distribution: str) -> list[int]:
    if total_m < groups:
        raise ValueError(f"M={total_m} must be at least groups={groups}")
    if distribution == "balanced":
        quotient, remainder = divmod(total_m, groups)
        return [quotient + (index < remainder) for index in range(groups)]

    # A deterministic long-tail distribution.  Keep every group non-empty,
    # then distribute remaining rows in proportion to weights 1..E.
    remaining = total_m - groups
    weights = list(range(1, groups + 1))
    denominator = sum(weights)
    extras = [remaining * weight // denominator for weight in weights]
    leftover = remaining - sum(extras)
    for index in range(groups - leftover, groups):
        extras[index] += 1
    return [extra + 1 for extra in extras]


def _indptr(group_rows: list[int]) -> torch.Tensor:
    prefix = [0]
    for rows in group_rows:
        prefix.append(prefix[-1] + rows)
    return torch.tensor(prefix, dtype=torch.int32, device="cuda")


def _selected_groups(group_rows: list[int], limit: int) -> list[int]:
    nonempty = [index for index, rows in enumerate(group_rows) if rows]
    if limit == 0 or len(nonempty) <= limit:
        return nonempty
    if limit == 1:
        return [nonempty[len(nonempty) // 2]]
    selected = {
        nonempty[round(index * (len(nonempty) - 1) / (limit - 1))]
        for index in range(limit)
    }
    return sorted(selected)


@torch.inference_mode()
def _check_result(
    out: torch.Tensor,
    x,
    weight,
    cu: torch.Tensor,
    group_rows: list[int],
    rows_per_group: int,
    group_limit: int,
) -> dict[str, float | int]:
    """Compare sampled rows against FP32 math on the exact quantized inputs."""

    x_dq = dequantize_varlen_m_operand(x, cu)
    n, k = weight.shape[1:]
    weight_sf = unpack_scale_blocked_to_2d(weight.scale, n, k // 32)
    cu_host = cu.cpu().tolist()
    diff_squares = []
    ref_squares = []
    max_errors = []
    checked_rows = 0
    selected = _selected_groups(group_rows, group_limit)
    for expert in selected:
        lo, hi = cu_host[expert], cu_host[expert + 1]
        hi = min(hi, lo + rows_per_group)
        if hi == lo:
            continue
        w_dq = weight.qdata[expert].float() * weight_sf[
            expert
        ].float().repeat_interleave(32, dim=-1)
        ref = x_dq[lo:hi] @ w_dq.T
        diff = out[lo:hi].float() - ref
        diff_squares.append(diff.square().sum())
        ref_squares.append(ref.square().sum())
        max_errors.append(diff.abs().max())
        checked_rows += hi - lo

    rel_l2 = math.sqrt(
        float(torch.stack(diff_squares).sum() / torch.stack(ref_squares).sum())
    )
    max_abs = float(torch.stack(max_errors).max())
    if not torch.isfinite(out).all() or rel_l2 > 0.02:
        raise AssertionError(
            f"grouped GEMM correctness failed: rel_l2={rel_l2:.6g}, "
            f"max_abs={max_abs:.6g}"
        )
    return {
        "checked_groups": len(selected),
        "checked_rows": checked_rows,
        "sample_rel_l2": rel_l2,
        "sample_max_abs": max_abs,
    }


@torch.inference_mode()
def run_workload(
    name: str,
    shape: tuple[int, int, int],
    group_rows: list[int],
    args: argparse.Namespace,
) -> dict:
    m, n, k = shape
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(args.seed)
    cu = _indptr(group_rows)
    x_hp = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.015625
    x = quantize_varlen_m_operand(x_hp, cu)
    del x_hp
    weight = allocate_mxfp8_weights(
        len(group_rows), n, k, device="cuda", seed=args.seed + 1
    )
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    config = Mxfp8MoEKernelConfig(
        fc2_tile_m=args.tile_m,
        fc2_tile_n=args.tile_n,
        pingpong=not args.no_pingpong,
        dynamic_persistent=not args.static_persistent,
    )

    # First call performs any cold CuTe-DSL compilation.  Quantization, scale
    # packing, allocation, and compilation are outside the reported latency.
    mxfp8_grouped_gemm(x, weight, cu, config=config, out=out)
    torch.cuda.synchronize()
    correctness = (
        {}
        if args.skip_check
        else _check_result(
            out,
            x,
            weight,
            cu,
            group_rows,
            args.check_rows,
            args.check_groups,
        )
    )

    for _ in range(args.warmup):
        mxfp8_grouped_gemm(x, weight, cu, config=config, out=out)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
    for start, end in zip(starts, ends):
        start.record()
        mxfp8_grouped_gemm(x, weight, cu, config=config, out=out)
        end.record()
    torch.cuda.synchronize()
    samples_ms = torch.tensor(
        [start.elapsed_time(end) for start, end in zip(starts, ends)],
        dtype=torch.float64,
    )
    p50_ms, p90_ms, p99_ms = torch.quantile(
        samples_ms, torch.tensor([0.50, 0.90, 0.99], dtype=torch.float64)
    ).tolist()
    mean_ms = float(samples_ms.mean())
    tflops = 2.0 * m * n * k / (p50_ms / 1e3) / 1e12

    return {
        "workload": name,
        "m_total": m,
        "n": n,
        "k": k,
        "groups": len(group_rows),
        "group_rows": group_rows,
        "distribution": "explicit" if args.group_ms is not None else args.distribution,
        "dtype": "OCP MXFP8 E4M3 + E8M0",
        "semantic_scale_granularity": "1x32 along reduction K",
        "output_dtype": "bfloat16",
        "scope": "prequantized grouped GEMM kernel only",
        "tile_m": args.tile_m,
        "tile_n": args.tile_n,
        "pingpong": not args.no_pingpong,
        "dynamic_persistent": not args.static_persistent,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "mean_ms": mean_ms,
        "min_ms": float(samples_ms.min()),
        "p50_ms": p50_ms,
        "p90_ms": p90_ms,
        "p99_ms": p99_ms,
        "max_ms": float(samples_ms.max()),
        "tflops_at_p50": tflops,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        **correctness,
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        raise RuntimeError("this benchmark requires a physical SM120/SM121 GPU")

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "capability": torch.cuda.get_device_capability(),
                "torch": torch.__version__,
                "workloads": args.workloads,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for name in args.workloads:
        shape = WORKLOADS[name]
        if args.group_ms is not None:
            if sum(args.group_ms) != shape[0]:
                raise ValueError(
                    f"sum(--group-ms)={sum(args.group_ms)} does not match M={shape[0]}"
                )
            group_rows = args.group_ms
        else:
            group_rows = _group_rows(shape[0], args.groups, args.distribution)
        result = run_workload(name, shape, group_rows, args)
        line = json.dumps(result, sort_keys=True)
        print(line, flush=True)
        if args.jsonl is not None:
            args.jsonl.parent.mkdir(parents=True, exist_ok=True)
            with args.jsonl.open("a", encoding="utf-8") as output_file:
                output_file.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
