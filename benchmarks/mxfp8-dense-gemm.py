#!/usr/bin/env python3
# ********************************************************************************
# Copyright (c) 2026 SonicMoE contributors
# ********************************************************************************
"""Benchmark the requested SM120 dense MXFP8 GEMM workloads.

The timed path is one QuACK SM120 block-scaled GEMM with prequantized OCP
E4M3 values, E8M0 scales at semantic 1x32 granularity along K, and BF16
output. Allocation, random input generation, quantization, scale packing,
correctness checks, and cold JIT compilation are outside the measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch
from quack.blockscaled import BlockScaledOperand, pack_scale_2d_to_blocked_contig
from quack.epilogue.library import identity_epi

from sonicmoe.functional.mxfp8 import (
    MXFP8_FORMAT,
    Mxfp8MoEKernelConfig,
    allocate_mxfp8_weights,
    quantize_mxfp8_rows,
)

DENSE_WORKLOADS = {
    "mnk_8k4k4k": (8192, 4096, 4096),
    "mnk_16k4k4k": (16384, 4096, 4096),
    "mnk_32k4k4k": (32768, 4096, 4096),
}


def _parse_workloads(value: str) -> list[str]:
    if value == "all":
        return list(DENSE_WORKLOADS)
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in DENSE_WORKLOADS]
    if not names or unknown:
        raise argparse.ArgumentTypeError(
            f"choose one or more of {list(DENSE_WORKLOADS)}, or 'all'; unknown={unknown}"
        )
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workloads",
        type=_parse_workloads,
        default=list(DENSE_WORKLOADS),
        help="comma-separated names, or 'all' (default: all)",
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
    parser.add_argument("--node-label", default=socket.gethostname())
    parser.add_argument(
        "--output", type=Path, help="write the full result document as JSON"
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.repeats <= 0:
        parser.error("warmup must be non-negative; iterations/repeats must be positive")
    return args


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
    query = "uuid,pci.bus_id,driver_version,power.limit"
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
        "hostname": socket.gethostname(),
        "gpu_name": props.name,
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_total_memory_gib": props.total_memory / 2**30,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
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


def _aggregate(repeats: list[dict[str, float]]) -> dict[str, float]:
    return {
        metric: statistics.median(repeat[metric] for repeat in repeats)
        for metric in ("mean_ms", "min_ms", "p50_ms", "p90_ms", "p99_ms", "max_ms")
    }


@torch.inference_mode()
def run_workload(
    name: str,
    shape: tuple[int, int, int],
    args: argparse.Namespace,
) -> dict:
    m, n, k = shape
    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    source = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.015625
    qdata, linear_sf = quantize_mxfp8_rows(source)
    del source
    activation = BlockScaledOperand.from_parts(
        qdata, pack_scale_2d_to_blocked_contig(linear_sf), MXFP8_FORMAT
    )
    weights = allocate_mxfp8_weights(1, n, k, device="cuda", seed=args.seed + 1)
    weight = BlockScaledOperand.from_parts(
        weights.qdata[0], weights.scale[0:1], MXFP8_FORMAT
    )
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    config = Mxfp8MoEKernelConfig(
        fc2_tile_m=args.tile_m,
        fc2_tile_n=args.tile_n,
        pingpong=not args.no_pingpong,
        dynamic_persistent=not args.static_persistent,
    )

    def dense_fn() -> None:
        _dense_call(activation, weight, out, config)

    dense_fn()
    torch.cuda.synchronize()
    all_finite = bool(torch.isfinite(out).all())
    output_abs_max = float(out.abs().max())
    if not args.skip_correctness and not all_finite:
        raise AssertionError(f"non-finite output for {name}")

    repeats = [
        _benchmark_call(dense_fn, args.warmup, args.iterations)
        for _ in range(args.repeats)
    ]
    aggregate = _aggregate(repeats)
    p50_ms = aggregate["p50_ms"]
    return {
        "workload": name,
        "m": m,
        "n": n,
        "k": k,
        "aggregate": aggregate,
        "repeat_metrics": repeats,
        "dense_tflops_at_p50": 2.0 * m * n * k / (p50_ms / 1e3) / 1e12,
        "correctness": {
            "all_finite": all_finite,
            "output_abs_max": output_abs_max,
        },
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        raise RuntimeError("this benchmark requires physical SM120/SM121 hardware")

    document = {
        "schema_version": 1,
        "benchmark": "sonicmoe_sm120_mxfp8_dense_gemm",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "scope": "prequantized kernel-only; allocation, packing, checks, and JIT excluded",
        "dtype": "OCP MXFP8 E4M3 values + E8M0 scales; BF16 output",
        "semantic_scale_granularity": "1x32 along reduction K",
        "config": {
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
    for name in args.workloads:
        result = run_workload(name, DENSE_WORKLOADS[name], args)
        document["results"].append(result)
        print(
            json.dumps(
                {
                    "workload": name,
                    "p50_ms": result["aggregate"]["p50_ms"],
                    "tflops": result["dense_tflops_at_p50"],
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
