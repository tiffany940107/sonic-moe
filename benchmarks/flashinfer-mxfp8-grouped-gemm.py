#!/usr/bin/env python3
# ********************************************************************************
# Copyright (c) 2026 SonicMoE contributors
# ********************************************************************************
"""Benchmark FlashInfer SM120 MXFP8 grouped GEMM on the SonicMoE workloads.

The default protocol is intentionally aligned with ``mxfp8-dense-vs-grouped.py``:

* six total-M workloads, with eight balanced expert groups;
* OCP E4M3 values, UE8M0 scales, semantic ``(1, 1, 32)`` granularity;
* BF16 output; and
* 20 warmups, 100 CUDA-event iterations, and three repeats.

Inputs and weights are quantized and packed before timing.  Allocation,
quantization, scale packing, dequantized-reference validation, cold CuTe-DSL
compilation, and FlashInfer autotuning are excluded from reported latency.

Set ``FLASHINFER_REPO`` to the exact FlashInfer checkout installed in the
environment.  This script imports layout helpers from that checkout's
``tests/grouped_mm/test_cute_sm120_mxfp8.py`` and rejects a different imported
FlashInfer package.  Set ``FLASHINFER_COMMIT`` as an additional revision check
when the checkout does not contain Git metadata (for example, in a container).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import torch


WORKLOADS = {
    "mnk_8k_1280_2k": (8192, 1280, 2048),
    "mnk_8k_2k_1280": (8192, 2048, 1280),
    "mnk_16k_1280_2k": (16384, 1280, 2048),
    "mnk_16k_2k_1280": (16384, 2048, 1280),
    "mnk_32k_1280_2k": (32768, 1280, 2048),
    "mnk_32k_2k_1280": (32768, 2048, 1280),
}
METRICS = ("mean_ms", "min_ms", "p50_ms", "p90_ms", "p99_ms", "max_ms")
GRAN_K = 32
COSINE_THRESHOLD = 0.99
RELATIVE_L2_THRESHOLD = 0.02
HELPER_RELATIVE_PATH = Path("tests/grouped_mm/test_cute_sm120_mxfp8.py")
REQUIRED_HELPERS = (
    "per_token_cast_to_mxfp8_for_moe_gemm",
    "per_token_cast_to_fp8",
    "per_token_dequant_from_fp8",
    "transform_sf_into_required_layout",
)


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
        help="comma-separated workload names, or 'all' (default: all)",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=8,
        help="balanced expert groups (default: 8, matching the SonicMoE suite)",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="skip the full dequantized-reference check (not for published results)",
    )
    parser.add_argument(
        "--node-label",
        default="anonymous",
        help="public alias stored in results; hostnames and device IDs are not collected",
    )
    parser.add_argument(
        "--output", type=Path, help="write the full result document as JSON"
    )
    args = parser.parse_args()
    if args.groups <= 0:
        parser.error("--groups must be positive")
    if args.warmup < 0 or args.iterations <= 0 or args.repeats <= 0:
        parser.error("warmup must be non-negative; iterations/repeats must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.node_label):
        parser.error("--node-label must be a privacy-safe public alias")
    return args


def _commits_match(left: str, right: str) -> bool:
    return left == right or left.startswith(right) or right.startswith(left)


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


def _source_revision(source_root: Path) -> str | None:
    marker = source_root / ".git_commit"
    marker_revision = None
    try:
        marker_revision = marker.read_text(encoding="utf-8").strip() or None
    except OSError:
        pass

    git_revision = _git_commit(source_root)
    if marker_revision and git_revision and not _commits_match(
        marker_revision, git_revision
    ):
        raise RuntimeError(
            "FlashInfer source revision mismatch between .git_commit and Git metadata"
        )
    return git_revision or marker_revision


def _load_layout_helpers() -> tuple[ModuleType, Path, str]:
    source_value = os.environ.get("FLASHINFER_REPO")
    if not source_value:
        raise RuntimeError(
            "FLASHINFER_REPO must point to the exact installed FlashInfer checkout"
        )
    source_root = Path(source_value).expanduser().resolve()
    helper_path = source_root / HELPER_RELATIVE_PATH
    if not helper_path.is_file():
        raise RuntimeError(f"FlashInfer MXFP8 helper is missing: {helper_path}")

    # Import the package before the test helper, then prove that the active
    # package and helper file come from the same source checkout.
    import flashinfer

    package_path = Path(flashinfer.__file__).resolve()
    if not package_path.is_relative_to(source_root):
        raise RuntimeError(
            "imported FlashInfer package is not from FLASHINFER_REPO: "
            f"package={package_path}, source={source_root}"
        )

    revision = _source_revision(source_root)
    expected_revision = os.environ.get("FLASHINFER_COMMIT")
    if expected_revision and revision and not _commits_match(
        expected_revision, revision
    ):
        raise RuntimeError(
            "FLASHINFER_COMMIT does not match the FLASHINFER_REPO revision: "
            f"expected={expected_revision}, source={revision}"
        )
    revision = revision or expected_revision or "unknown"

    spec = importlib.util.spec_from_file_location(
        "flashinfer_mxfp8_benchmark_helpers", helper_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import FlashInfer helper: {helper_path}")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    missing = [name for name in REQUIRED_HELPERS if not hasattr(helpers, name)]
    if missing:
        raise RuntimeError(f"FlashInfer helper is missing functions: {missing}")
    return helpers, source_root, revision


def _environment(node_label: str, revision: str) -> dict:
    props = torch.cuda.get_device_properties(0)
    return {
        "node_label": node_label,
        "gpu_name": props.name,
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_total_memory_gib": props.total_memory / 2**30,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "flashinfer_commit": revision,
    }


def _group_rows(total_m: int, groups: int) -> list[int]:
    if total_m < groups:
        raise ValueError(f"M={total_m} must be at least groups={groups}")
    quotient, remainder = divmod(total_m, groups)
    return [quotient + (index < remainder) for index in range(groups)]


def _indptr(group_rows: list[int]) -> torch.Tensor:
    prefix = [0]
    for rows in group_rows:
        prefix.append(prefix[-1] + rows)
    return torch.tensor(prefix, dtype=torch.int32, device="cuda")


def _summarize(samples: list[float]) -> dict[str, float]:
    values = torch.tensor(samples, dtype=torch.float64)
    p50, p90, p99 = torch.quantile(
        values, torch.tensor([0.50, 0.90, 0.99], dtype=torch.float64)
    ).tolist()
    return {
        "mean_ms": float(values.mean()),
        "min_ms": float(values.min()),
        "p50_ms": p50,
        "p90_ms": p90,
        "p99_ms": p99,
        "max_ms": float(values.max()),
    }


@torch.inference_mode()
def _benchmark_call(invoke, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        invoke()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        invoke()
        end.record()
    torch.cuda.synchronize()
    return _summarize(
        [start.elapsed_time(end) for start, end in zip(starts, ends)]
    )


@torch.inference_mode()
def _check_dequantized_reference(
    out: torch.Tensor,
    source_a: torch.Tensor,
    a_fp8: torch.Tensor,
    b_fp8: torch.Tensor,
    b_sf_ue8m0: torch.Tensor,
    m_indptr: torch.Tensor,
    helpers: ModuleType,
) -> dict[str, float | int | bool | str]:
    """Validate every output element against GEMM on dequantized operands."""

    offsets = m_indptr.cpu().tolist()
    diff_squares = 0.0
    reference_squares = 0.0
    output_squares = 0.0
    output_reference_dot = 0.0
    max_abs = 0.0
    activation_qdata_matches_reference = True

    for expert, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        expert_fp8, expert_sf = helpers.per_token_cast_to_fp8(
            source_a[start:end], use_ue8m0=True, gran_k=GRAN_K
        )
        activation_qdata_matches_reference &= bool(
            torch.equal(a_fp8[start:end], expert_fp8)
        )
        activation_dequantized = helpers.per_token_dequant_from_fp8(
            expert_fp8, expert_sf, gran_k=GRAN_K, dtype=torch.bfloat16
        )
        weight_dequantized = helpers.per_token_dequant_from_fp8(
            b_fp8[expert],
            b_sf_ue8m0[expert],
            gran_k=GRAN_K,
            dtype=torch.bfloat16,
        )
        reference = (
            activation_dequantized @ weight_dequantized.transpose(0, 1)
        ).to(torch.bfloat16)
        actual_float = out[start:end].float()
        reference_float = reference.float()
        difference = actual_float - reference_float
        diff_squares += float(difference.square().sum())
        reference_squares += float(reference_float.square().sum())
        output_squares += float(actual_float.square().sum())
        output_reference_dot += float((actual_float * reference_float).sum())
        max_abs = max(max_abs, float(difference.abs().max()))

    relative_l2 = math.sqrt(diff_squares / reference_squares)
    cosine_similarity = output_reference_dot / math.sqrt(
        output_squares * reference_squares
    )
    all_finite = bool(torch.isfinite(out).all())
    passed = (
        all_finite
        and activation_qdata_matches_reference
        and cosine_similarity >= COSINE_THRESHOLD
        and relative_l2 <= RELATIVE_L2_THRESHOLD
    )
    result: dict[str, float | int | bool | str] = {
        "status": "passed" if passed else "failed",
        "all_finite": all_finite,
        "checked_groups": len(offsets) - 1,
        "checked_rows": out.shape[0],
        "checked_elements": out.numel(),
        "activation_qdata_matches_reference": activation_qdata_matches_reference,
        "cosine_similarity_vs_dequantized_reference": cosine_similarity,
        "cosine_similarity_threshold": COSINE_THRESHOLD,
        "relative_l2_vs_dequantized_reference": relative_l2,
        "relative_l2_threshold": RELATIVE_L2_THRESHOLD,
        "max_abs_vs_dequantized_reference": max_abs,
    }
    if not passed:
        raise AssertionError(f"FlashInfer grouped GEMM correctness failed: {result}")
    return result


@torch.inference_mode()
def run_workload(
    name: str,
    shape: tuple[int, int, int],
    group_rows: list[int],
    args: argparse.Namespace,
    helpers: ModuleType,
) -> dict:
    from flashinfer.grouped_mm import moe_gemm_mxfp8_nt_groupwise

    m, n, k = shape
    if sum(group_rows) != m:
        raise ValueError(f"sum(M_e)={sum(group_rows)} does not match workload M={m}")
    if k % GRAN_K or n % GRAN_K:
        raise ValueError(f"N={n} and K={k} must be divisible by {GRAN_K}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(args.seed)
    m_indptr = _indptr(group_rows)

    source_a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.015625
    a_fp8, a_scale = helpers.per_token_cast_to_mxfp8_for_moe_gemm(
        source_a, m_indptr, gran_k=GRAN_K
    )

    source_b = (
        torch.randn(
            (len(group_rows), n, k), dtype=torch.bfloat16, device="cuda"
        )
        / math.sqrt(k)
    )
    b_fp8_parts = []
    b_sf_parts = []
    for expert in range(len(group_rows)):
        # B uses the same per-output-row 1x32 semantic granularity as SonicMoE,
        # not an N32xK32 block scale broadcast across 32 output rows.
        expert_fp8, expert_sf = helpers.per_token_cast_to_fp8(
            source_b[expert], use_ue8m0=True, gran_k=GRAN_K
        )
        b_fp8_parts.append(expert_fp8)
        b_sf_parts.append(expert_sf)
    b_fp8 = torch.stack(b_fp8_parts, dim=0)
    b_sf_ue8m0 = torch.stack(b_sf_parts, dim=0)
    del source_b, b_fp8_parts, b_sf_parts
    b_scale = helpers.transform_sf_into_required_layout(
        b_sf_ue8m0,
        mn=n,
        k=k,
        recipe=(1, GRAN_K),
        num_groups=len(group_rows),
        is_sfa=False,
    )
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

    def invoke() -> None:
        moe_gemm_mxfp8_nt_groupwise(
            a_fp8,
            b_fp8,
            a_scale,
            b_scale,
            m_indptr,
            scale_granularity_mnk=(1, 1, GRAN_K),
            out=out,
            out_dtype=torch.bfloat16,
            is_gated=False,
        )

    # Trigger cold compilation and autotuning before correctness and timing.
    invoke()
    torch.cuda.synchronize()
    if args.skip_correctness:
        correctness: dict[str, float | int | bool | str] = {
            "status": "skipped",
            "all_finite": bool(torch.isfinite(out).all()),
        }
    else:
        correctness = _check_dequantized_reference(
            out,
            source_a,
            a_fp8,
            b_fp8,
            b_sf_ue8m0,
            m_indptr,
            helpers,
        )
    del source_a, b_sf_ue8m0

    repeat_metrics = []
    for _ in range(args.repeats):
        repeat_metrics.append(
            {"grouped": _benchmark_call(invoke, args.warmup, args.iterations)}
        )
    aggregate_grouped = {
        metric: statistics.median(
            repeat["grouped"][metric] for repeat in repeat_metrics
        )
        for metric in METRICS
    }
    p50_ms = aggregate_grouped["p50_ms"]
    return {
        "workload": name,
        "m_total": m,
        "n": n,
        "k": k,
        "groups": len(group_rows),
        "group_rows": group_rows,
        "distribution": "balanced",
        "aggregate": {"grouped": aggregate_grouped},
        "repeat_metrics": repeat_metrics,
        "grouped_tflops_at_p50": 2.0 * m * n * k / (p50_ms / 1e3) / 1e12,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "correctness": correctness,
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 12:
        raise RuntimeError("this benchmark requires a physical SM120/SM121 GPU")

    helpers, _source_root, revision = _load_layout_helpers()
    document = {
        "schema_version": 1,
        "benchmark": "flashinfer_sm120_mxfp8_grouped_gemm",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "scope": (
            "prequantized kernel-only; allocation, quantization, scale packing, "
            "dequantized-reference validation, cold JIT, and autotuning excluded"
        ),
        "dtype": "OCP MXFP8 E4M3 values + E8M0 scales; BF16 output",
        "semantic_scale_granularity": "1x32 along reduction K",
        "config": {
            "groups": args.groups,
            "distribution": "balanced",
            "api_scale_granularity_mnk": [1, 1, GRAN_K],
            "scale_encoding": "UE8M0",
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "seed": args.seed,
            "is_gated": False,
        },
        "environment": _environment(args.node_label, revision),
        "source_validation": {
            "helper": str(HELPER_RELATIVE_PATH),
            "helper_and_imported_package_share_flashinfer_repo": True,
        },
        "results": [],
    }
    print(
        json.dumps(
            {"environment": document["environment"], "config": document["config"]}
        ),
        flush=True,
    )
    for name in args.workloads:
        shape = WORKLOADS[name]
        rows = _group_rows(shape[0], args.groups)
        result = run_workload(name, shape, rows, args, helpers)
        document["results"].append(result)
        print(
            json.dumps(
                {
                    "workload": name,
                    "grouped_p50_ms": result["aggregate"]["grouped"]["p50_ms"],
                    "grouped_tflops_at_p50": result["grouped_tflops_at_p50"],
                    "correctness": result["correctness"]["status"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        torch.cuda.empty_cache()

    encoded = json.dumps(document, indent=2, sort_keys=True)
    if args.output is None:
        print(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
