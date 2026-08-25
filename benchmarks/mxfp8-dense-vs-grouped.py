#!/usr/bin/env python3
# ********************************************************************************
# Copyright (c) 2026 SonicMoE contributors
# ********************************************************************************
"""Compare SM120 MXFP8 grouped GEMM with two same-MNK dense baselines.

The three timed paths use the same quantized activation bytes, total FLOPs,
tile configuration, BF16 output dtype, and semantic 1x32 E8M0 scales:

* ``grouped``: one variable-M launch, with a different weight per group;
* ``dense_loop``: one dense launch per group, with identical MoE semantics;
* ``one_big_dense``: one launch over total M using a single shared weight.

Only ``dense_loop / grouped`` is an MoE acceleration ratio.  ``one_big_dense``
is deliberately included as a hardware-efficiency ceiling; it is not the same
model because all rows use W_0.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import torch
from quack.blockscaled import BlockScaledOperand, pack_scale_2d_to_blocked_contig
from quack.epilogue.library import identity_epi

from sonicmoe.functional.mxfp8 import (
    MXFP8_FORMAT,
    Mxfp8MoEKernelConfig,
    allocate_mxfp8_weights,
    make_varlen_m_operand,
    mxfp8_grouped_gemm,
    quantize_mxfp8_rows,
)

WORKLOADS = {
    "mnk_8k_1280_2k": (8192, 1280, 2048),
    "mnk_8k_2k_1280": (8192, 2048, 1280),
    "mnk_16k_1280_2k": (16384, 1280, 2048),
    "mnk_16k_2k_1280": (16384, 2048, 1280),
    "mnk_32k_1280_2k": (32768, 1280, 2048),
    "mnk_32k_2k_1280": (32768, 2048, 1280),
}
METHODS = ("grouped", "dense_loop", "one_big_dense")


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
    if not names or unknown:
        raise argparse.ArgumentTypeError(
            f"choose one or more of {list(WORKLOADS)}, or 'all'; unknown={unknown}"
        )
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workloads",
        type=_parse_workloads,
        default=list(WORKLOADS),
        help="comma-separated names, or 'all' (default: all)",
    )
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument(
        "--distribution", choices=("balanced", "ragged"), default="balanced"
    )
    parser.add_argument(
        "--group-ms",
        type=_comma_ints,
        help="exact M_e list; requires one workload and overrides --distribution",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tile-m", type=int, choices=(128, 256), default=128)
    parser.add_argument("--tile-n", type=int, choices=(128, 256), default=128)
    parser.add_argument("--no-pingpong", action="store_true")
    parser.add_argument("--static-persistent", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--node-label",
        default="anonymous",
        help="public alias stored in results; hostnames are never collected",
    )
    parser.add_argument(
        "--output", type=Path, help="write the full result document as JSON"
    )
    args = parser.parse_args()
    if args.groups <= 0:
        parser.error("--groups must be positive")
    if args.warmup < 0 or args.iterations <= 0 or args.repeats <= 0:
        parser.error("warmup must be non-negative; iterations/repeats must be positive")
    if args.group_ms is not None:
        if len(args.workloads) != 1:
            parser.error("--group-ms requires exactly one workload")
        if len(args.group_ms) != args.groups:
            parser.error("--group-ms length must equal --groups")
    return args


def _group_rows(total_m: int, groups: int, distribution: str) -> list[int]:
    if total_m < groups:
        raise ValueError(f"M={total_m} must be at least groups={groups}")
    if distribution == "balanced":
        quotient, remainder = divmod(total_m, groups)
        return [quotient + (index < remainder) for index in range(groups)]

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


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={path}",
                "-C",
                str(path),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _nvidia_smi_metadata(device_uuid: str) -> dict[str, str]:
    # The UUID is used only to select the same physical device PyTorch sees. It
    # is deliberately excluded from the returned, publishable metadata.
    query = "driver_version,power.limit"
    try:
        line = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=" + query,
                "--format=csv,noheader,nounits",
                "--id=" + device_uuid,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.splitlines()[0]
        values = [part.strip() for part in line.split(",")]
        return dict(zip(query.split(","), values))
    except (OSError, subprocess.SubprocessError, IndexError):
        return {}


def _environment(node_label: str) -> dict:
    import quack

    import sonicmoe

    props = torch.cuda.get_device_properties(0)
    device_uuid = str(props.uuid)
    if not device_uuid.startswith("GPU-"):
        device_uuid = "GPU-" + device_uuid
    return {
        "node_label": node_label,
        "gpu_name": props.name,
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_total_memory_gib": props.total_memory / 2**30,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "sonic_commit": _git_commit(Path(sonicmoe.__file__).resolve().parents[1]),
        "quack_commit": _git_commit(Path(quack.__file__).resolve().parents[1]),
        **_nvidia_smi_metadata(device_uuid),
    }


def _dense_call(
    activation: BlockScaledOperand,
    weight: BlockScaledOperand,
    out: torch.Tensor,
    config: Mxfp8MoEKernelConfig,
) -> None:
    identity_epi.gemm(
        activation.qdata,
        weight.qdata,
        out,
        epi_args={},
        tile_M=config.fc2_tile_m,
        tile_N=config.fc2_tile_n,
        cluster_M=1,
        cluster_N=1,
        pingpong=config.pingpong,
        persistent=True,
        is_dynamic_persistent=config.dynamic_persistent,
        SFA=activation.scale,
        SFB=weight.scale,
        bs_format_a=activation.format.name,
        bs_format_b=weight.format.name,
    )


@torch.inference_mode()
def _benchmark_call(fn, warmup: int, iterations: int) -> dict[str, float]:
    # Compile before both warmup and timing.  The returned metrics are kernel-only.
    fn()
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    samples = torch.tensor(
        [start.elapsed_time(end) for start, end in zip(starts, ends)],
        dtype=torch.float64,
    )
    p50, p90, p99 = torch.quantile(
        samples, torch.tensor([0.50, 0.90, 0.99], dtype=torch.float64)
    ).tolist()
    return {
        "mean_ms": float(samples.mean()),
        "min_ms": float(samples.min()),
        "p50_ms": p50,
        "p90_ms": p90,
        "p99_ms": p99,
        "max_ms": float(samples.max()),
    }


def _aggregate(
    repeats: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    result = {}
    for method in METHODS:
        result[method] = {
            metric: statistics.median(repeat[method][metric] for repeat in repeats)
            for metric in ("mean_ms", "min_ms", "p50_ms", "p90_ms", "p99_ms", "max_ms")
        }
    return result


@torch.inference_mode()
def run_workload(
    name: str,
    shape: tuple[int, int, int],
    group_rows: list[int],
    args: argparse.Namespace,
    workload_index: int,
) -> dict:
    m, n, k = shape
    if sum(group_rows) != m:
        raise ValueError(f"sum(M_e)={sum(group_rows)} does not match workload M={m}")

    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    source = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.015625
    qdata, linear_sf = quantize_mxfp8_rows(source)
    del source
    cu = _indptr(group_rows)
    grouped_a = make_varlen_m_operand(qdata, linear_sf, cu)
    dense_a = BlockScaledOperand.from_parts(
        qdata, pack_scale_2d_to_blocked_contig(linear_sf), MXFP8_FORMAT
    )
    weight = allocate_mxfp8_weights(
        len(group_rows), n, k, device="cuda", seed=args.seed + 1
    )
    one_dense_w = BlockScaledOperand.from_parts(
        weight.qdata[0], weight.scale[0:1], MXFP8_FORMAT
    )

    expert_as = []
    expert_ws = []
    cu_host = cu.cpu().tolist()
    for expert, (lo, hi) in enumerate(pairwise(cu_host)):
        expert_as.append(
            BlockScaledOperand.from_parts(
                qdata[lo:hi],
                pack_scale_2d_to_blocked_contig(linear_sf[lo:hi]),
                MXFP8_FORMAT,
            )
        )
        expert_ws.append(
            BlockScaledOperand.from_parts(
                weight.qdata[expert], weight.scale[expert : expert + 1], MXFP8_FORMAT
            )
        )

    grouped_out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    loop_out = torch.empty_like(grouped_out)
    one_dense_out = torch.empty_like(grouped_out)
    config = Mxfp8MoEKernelConfig(
        fc2_tile_m=args.tile_m,
        fc2_tile_n=args.tile_n,
        pingpong=not args.no_pingpong,
        dynamic_persistent=not args.static_persistent,
    )

    def grouped_fn() -> None:
        mxfp8_grouped_gemm(grouped_a, weight, cu, config=config, out=grouped_out)

    def dense_loop_fn() -> None:
        for expert, (lo, hi) in enumerate(pairwise(cu_host)):
            if hi != lo:
                _dense_call(
                    expert_as[expert], expert_ws[expert], loop_out[lo:hi], config
                )

    def one_big_dense_fn() -> None:
        _dense_call(dense_a, one_dense_w, one_dense_out, config)

    calls = {
        "grouped": grouped_fn,
        "dense_loop": dense_loop_fn,
        "one_big_dense": one_big_dense_fn,
    }
    # Compile all methods, then prove the exact-semantics baselines agree.
    for fn in calls.values():
        fn()
    torch.cuda.synchronize()
    diff = grouped_out.float() - loop_out.float()
    correctness = {
        "grouped_vs_dense_loop_rel_l2": float(
            torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(loop_out.float())
        ),
        "grouped_vs_dense_loop_max_abs": float(diff.abs().max()),
    }
    if not args.skip_correctness and (
        not torch.isfinite(grouped_out).all()
        or correctness["grouped_vs_dense_loop_rel_l2"] > 0.005
    ):
        raise AssertionError(f"grouped/dense-loop correctness failed: {correctness}")
    del diff

    repeat_results = []
    for repeat_index in range(args.repeats):
        offset = (workload_index + repeat_index) % len(METHODS)
        order = METHODS[offset:] + METHODS[:offset]
        measured = {
            method: _benchmark_call(calls[method], args.warmup, args.iterations)
            for method in order
        }
        repeat_results.append({method: measured[method] for method in METHODS})
    aggregate = _aggregate(repeat_results)
    grouped_ms = aggregate["grouped"]["p50_ms"]
    loop_ms = aggregate["dense_loop"]["p50_ms"]
    one_dense_ms = aggregate["one_big_dense"]["p50_ms"]
    return {
        "workload": name,
        "m_total": m,
        "n": n,
        "k": k,
        "groups": len(group_rows),
        "group_rows": group_rows,
        "distribution": "explicit" if args.group_ms is not None else args.distribution,
        "aggregate": aggregate,
        "repeat_metrics": repeat_results,
        "speedup_grouped_vs_dense_loop": loop_ms / grouped_ms,
        "grouped_efficiency_vs_one_big_dense": one_dense_ms / grouped_ms,
        "grouped_slowdown_vs_one_big_dense": grouped_ms / one_dense_ms,
        "grouped_tflops_at_p50": 2.0 * m * n * k / (grouped_ms / 1e3) / 1e12,
        "correctness": correctness,
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        raise RuntimeError("this benchmark requires physical SM120/SM121 hardware")

    document = {
        "schema_version": 1,
        "benchmark": "sonicmoe_sm120_mxfp8_dense_vs_grouped",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "scope": "prequantized kernel-only; allocation, packing, correctness, and JIT excluded",
        "dtype": "OCP MXFP8 E4M3 values + E8M0 scales; BF16 output",
        "semantic_scale_granularity": "1x32 along reduction K",
        "config": {
            "groups": args.groups,
            "distribution": args.distribution,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "tile_m": args.tile_m,
            "tile_n": args.tile_n,
            "pingpong": not args.no_pingpong,
            "dynamic_persistent": not args.static_persistent,
            "seed": args.seed,
        },
        "environment": _environment(args.node_label),
        "results": [],
    }
    print(
        json.dumps(
            {"environment": document["environment"], "config": document["config"]}
        )
    )
    for index, name in enumerate(args.workloads):
        shape = WORKLOADS[name]
        rows = (
            args.group_ms
            if args.group_ms is not None
            else _group_rows(shape[0], args.groups, args.distribution)
        )
        result = run_workload(name, shape, rows, args, index)
        document["results"].append(result)
        print(
            json.dumps(
                {
                    "workload": name,
                    "grouped_p50_ms": result["aggregate"]["grouped"]["p50_ms"],
                    "dense_loop_p50_ms": result["aggregate"]["dense_loop"]["p50_ms"],
                    "one_big_dense_p50_ms": result["aggregate"]["one_big_dense"][
                        "p50_ms"
                    ],
                    "speedup": result["speedup_grouped_vs_dense_loop"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if args.output is None:
        print(json.dumps(document, indent=2, sort_keys=True))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
