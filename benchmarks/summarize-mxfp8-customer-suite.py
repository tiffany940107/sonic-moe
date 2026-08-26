#!/usr/bin/env python3
"""Build a Markdown comparison from MXFP8 customer-suite JSON results."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def _suite_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=RESULT_DIRECTORY")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("expected non-empty LABEL=RESULT_DIRECTORY")
    return label, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=_suite_arg,
        action="append",
        required=True,
        help="repeatable LABEL=directory containing dense.json and grouped-vs-dense.json",
    )
    parser.add_argument(
        "--baseline",
        help="label used as the speedup numerator (default: first --suite label)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = [label for label, _ in args.suite]
    if len(set(labels)) != len(labels):
        parser.error("--suite labels must be unique")
    if args.baseline is None:
        args.baseline = labels[0]
    if args.baseline not in labels:
        parser.error("--baseline must match one --suite label")
    return args


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load {path}: {error}") from error


def _index_results(document: dict) -> dict[str, dict]:
    return {result["workload"]: result for result in document["results"]}


def _load_suite(label: str, directory: Path) -> dict:
    dense_path = directory / "dense.json"
    grouped_path = directory / "grouped-vs-dense.json"
    dense = _load_json(dense_path)
    grouped = _load_json(grouped_path)
    if dense.get("benchmark") != "sonicmoe_sm120_mxfp8_dense_gemm":
        raise RuntimeError(f"{dense_path} has an unexpected benchmark tag")
    if grouped.get("benchmark") != "sonicmoe_sm120_mxfp8_dense_vs_grouped":
        raise RuntimeError(f"{grouped_path} has an unexpected benchmark tag")
    if grouped["config"]["groups"] != 8:
        raise RuntimeError(f"{grouped_path} must contain exactly eight groups")
    if dense["semantic_scale_granularity"] != "1x32 along reduction K":
        raise RuntimeError(f"{dense_path} does not use semantic 1x32 scales")
    if grouped["semantic_scale_granularity"] != "1x32 along reduction K":
        raise RuntimeError(f"{grouped_path} does not use semantic 1x32 scales")
    if not all(result["correctness"]["all_finite"] for result in dense["results"]):
        raise RuntimeError(f"{dense_path} contains non-finite output")
    if not all(
        result["correctness"]["relative_l2_vs_dequantized_reference"]
        <= result["correctness"]["relative_l2_threshold"]
        for result in dense["results"]
    ):
        raise RuntimeError(f"{dense_path} failed its dequantized reference check")
    if not all(
        result["correctness"]["grouped_vs_dense_loop_max_abs"] == 0.0
        for result in grouped["results"]
    ):
        raise RuntimeError(f"{grouped_path} failed grouped/dense-loop equality")
    return {
        "label": label,
        "directory": directory,
        "dense_path": dense_path,
        "grouped_path": grouped_path,
        "dense": dense,
        "grouped": grouped,
        "dense_results": _index_results(dense),
        "grouped_results": _index_results(grouped),
    }


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    def escape(value: str) -> str:
        return value.replace("|", "\\|")

    result = [
        "| " + " | ".join(escape(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    result.extend(
        "| " + " | ".join(escape(value) for value in row) + " |" for row in rows
    )
    return result


def _assert_matching_workloads(suites: list[dict], key: str) -> list[str]:
    reference = list(suites[0][key])
    reference_set = set(reference)
    for suite in suites[1:]:
        if set(suite[key]) != reference_set:
            raise RuntimeError(f"workload mismatch for {key}: {suite['label']}")
    return reference


def _relative_link(path: Path, output: Path) -> str:
    return os.path.relpath(path.resolve(), output.parent.resolve())


def build_report(suites: list[dict], baseline_label: str, output: Path) -> str:
    baseline = next(suite for suite in suites if suite["label"] == baseline_label)
    dense_workloads = _assert_matching_workloads(suites, "dense_results")
    grouped_workloads = _assert_matching_workloads(suites, "grouped_results")

    first_dense = suites[0]["dense"]
    first_grouped = suites[0]["grouped"]
    lines = [
        "# SM120 MXFP8 dense and grouped GEMM comparison",
        "",
        (
            "For the PRO5000-only single-GPU scope, timing boundaries, scripts, and a "
            "Chinese explanation, see "
            "[PRO5000_SINGLE_GPU_RESULTS.md](PRO5000_SINGLE_GPU_RESULTS.md)."
        ),
        "",
        "All numbers are GPU-event P50 kernel latency. Inputs and weights are already",
        "quantized as OCP E4M3 plus E8M0 scales with semantic 1x32 granularity along",
        "K; allocation, quantization, scale packing, checks, and cold JIT are excluded.",
        "",
        "## Protocol",
        "",
        (
            f"- Dense: `{first_dense['config']['warmup']}` warmups, "
            f"`{first_dense['config']['iterations']}` iterations, "
            f"`{first_dense['config']['repeats']}` repeats."
        ),
        (
            f"- Grouped: eight balanced groups, "
            f"`{first_grouped['config']['warmup']}` warmups, "
            f"`{first_grouped['config']['iterations']}` iterations, "
            f"`{first_grouped['config']['repeats']}` repeats."
        ),
        "- Each displayed metric is the median of the per-repeat P50 values.",
        (
            "- `dense_loop` uses eight distinct expert weights and preserves grouped-MoE "
            "semantics. `one_big_dense` shares one weight and is only an efficiency ceiling."
        ),
        (
            "- The discarded Bash `GROUPS` collision and final eight-group rerun are "
            "documented in the [harness audit](AUDIT.md)."
        ),
        "",
        "## Environments",
        "",
        (
            "Hostnames, IP addresses, GPU UUIDs, PCI bus IDs, and visible-device indices "
            "are intentionally omitted. Node labels must be anonymous public aliases."
        ),
        "",
    ]

    environment_rows = []
    for suite in suites:
        env = suite["grouped"]["environment"]
        environment_rows.append(
            [
                suite["label"],
                env["gpu_name"],
                f"{env['gpu_total_memory_gib']:.1f}",
                env.get("power.limit", "unknown"),
                env.get("driver_version", "unknown"),
                (
                    suite["dense"]["environment"]["sonic_commit"][:8]
                    + "/"
                    + env["sonic_commit"][:8]
                ),
                env["quack_commit"][:8],
            ]
        )
    lines.extend(
        _table(
            [
                "label",
                "GPU",
                "GiB",
                "power W",
                "driver",
                "Sonic dense/grouped",
                "QuACK",
            ],
            environment_rows,
        )
    )

    lines.extend(["", "## Dense GEMM", ""])
    dense_rows = []
    for workload in dense_workloads:
        row = [workload]
        for suite in suites:
            result = suite["dense_results"][workload]
            row.append(
                f"{result['aggregate']['p50_ms']:.6f} / "
                f"{result['dense_tflops_at_p50']:.1f}"
            )
        dense_rows.append(row)
    lines.extend(
        _table(
            ["workload"] + [f"{suite['label']} ms / TFLOP/s" for suite in suites],
            dense_rows,
        )
    )

    lines.extend(["", "## Grouped GEMM", ""])
    grouped_rows = []
    for workload in grouped_workloads:
        row = [workload]
        for suite in suites:
            result = suite["grouped_results"][workload]
            row.append(
                f"{result['aggregate']['grouped']['p50_ms']:.6f} / "
                f"{result['grouped_tflops_at_p50']:.1f}"
            )
        grouped_rows.append(row)
    lines.extend(
        _table(
            ["workload"] + [f"{suite['label']} ms / TFLOP/s" for suite in suites],
            grouped_rows,
        )
    )

    lines.extend(["", "## Same-MNK grouped acceleration", ""])
    acceleration_rows = []
    for workload in grouped_workloads:
        row = [workload]
        for suite in suites:
            speedup = suite["grouped_results"][workload][
                "speedup_grouped_vs_dense_loop"
            ]
            row.append(f"{speedup:.3f}x")
        acceleration_rows.append(row)
    lines.extend(
        _table(
            ["workload"] + [f"{suite['label']} grouped speedup" for suite in suites],
            acceleration_rows,
        )
    )

    lines.extend(["", f"## Cross-GPU summary (baseline: {baseline_label})", ""])
    for suite in suites:
        if suite is baseline:
            continue
        dense_speedups = [
            suite["dense_results"][name]["aggregate"]["p50_ms"]
            / baseline["dense_results"][name]["aggregate"]["p50_ms"]
            for name in dense_workloads
        ]
        grouped_speedups = [
            suite["grouped_results"][name]["aggregate"]["grouped"]["p50_ms"]
            / baseline["grouped_results"][name]["aggregate"]["grouped"]["p50_ms"]
            for name in grouped_workloads
        ]
        lines.append(
            f"- `{baseline_label}` is `{_geomean(dense_speedups):.3f}x` faster than "
            f"`{suite['label']}` on dense and `{_geomean(grouped_speedups):.3f}x` "
            "faster on grouped GEMM (geometric mean)."
        )
    for suite in suites:
        grouped_speedups = [
            suite["grouped_results"][name]["speedup_grouped_vs_dense_loop"]
            for name in grouped_workloads
        ]
        ceiling_efficiency = [
            suite["grouped_results"][name]["grouped_efficiency_vs_one_big_dense"]
            for name in grouped_workloads
        ]
        lines.append(
            f"- `{suite['label']}` grouped vs eight dense launches: "
            f"`{_geomean(grouped_speedups):.3f}x`; efficiency vs one-big-dense "
            f"ceiling: `{100.0 * _geomean(ceiling_efficiency):.1f}%` (geometric mean)."
        )

    lines.extend(
        [
            "",
            "## Validation",
            "",
            (
                "Every dense output was finite and had relative L2 error <= 0.005 vs a "
                "dequantized FP32 reference. Every grouped output was bit-identical to the "
                "eight-launch dense-loop baseline (`max_abs = 0`, `rel_l2 = 0`)."
            ),
            "",
            "## Raw data",
            "",
        ]
    )
    for suite in suites:
        dense_link = _relative_link(suite["dense_path"], output)
        grouped_link = _relative_link(suite["grouped_path"], output)
        lines.append(
            f"- `{suite['label']}`: [dense]({dense_link}), "
            f"[grouped-vs-dense]({grouped_link})"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    suites = [_load_suite(label, path) for label, path in args.suite]
    report = build_report(suites, args.baseline, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
