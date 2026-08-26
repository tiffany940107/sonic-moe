#!/usr/bin/env python3
"""Validate and summarize aligned single-GPU FlashInfer/Sonic MXFP8 results."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


WORKLOADS = {
    "mnk_8k_1280_2k": (8192, 1280, 2048),
    "mnk_8k_2k_1280": (8192, 2048, 1280),
    "mnk_16k_1280_2k": (16384, 1280, 2048),
    "mnk_16k_2k_1280": (16384, 2048, 1280),
    "mnk_32k_1280_2k": (32768, 1280, 2048),
    "mnk_32k_2k_1280": (32768, 2048, 1280),
}
OLD_COMMIT = "05e5d927399d62a2479c430ad3e167738254d760"
NEW_COMMIT = "ff22228d2fa144e9ac6a0d841f2e9ba767ba0f0a"


def suite_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=RESULT_DIRECTORY")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("expected non-empty LABEL=RESULT_DIRECTORY")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=suite_arg, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load {path}: {error}") from error


def index_results(document: dict) -> dict[str, dict]:
    return {record["workload"]: record for record in document["results"]}


def validate_common(document: dict, path: Path) -> None:
    config = document.get("config", {})
    expected = {
        "groups": 8,
        "distribution": "balanced",
        "warmup": 20,
        "iterations": 100,
        "repeats": 3,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"{path}: {key}={config.get(key)!r}, expected {value!r}")
    if document.get("semantic_scale_granularity") != "1x32 along reduction K":
        raise RuntimeError(f"{path}: expected semantic K32 scales")
    records = index_results(document)
    if set(records) != set(WORKLOADS):
        raise RuntimeError(f"{path}: workload set does not match the requested suite")
    for name, (m, n, k) in WORKLOADS.items():
        record = records[name]
        if (record["m_total"], record["n"], record["k"]) != (m, n, k):
            raise RuntimeError(f"{path}: {name} has an incorrect shape")
        if record["groups"] != 8 or record["group_rows"] != [m // 8] * 8:
            raise RuntimeError(f"{path}: {name} is not eight balanced groups")


def validate_sonic(document: dict, path: Path) -> None:
    if document.get("benchmark") != "sonicmoe_sm120_mxfp8_dense_vs_grouped":
        raise RuntimeError(f"{path}: unexpected Sonic benchmark tag")
    validate_common(document, path)
    for record in document["results"]:
        correctness = record["correctness"]
        if correctness["grouped_vs_dense_loop_max_abs"] != 0.0:
            raise RuntimeError(f"{path}: Sonic grouped/dense-loop check failed")
        if correctness["grouped_vs_dense_loop_rel_l2"] != 0.0:
            raise RuntimeError(f"{path}: Sonic grouped/dense-loop check failed")


def validate_flashinfer(document: dict, path: Path, commit: str) -> None:
    if document.get("benchmark") != "flashinfer_sm120_mxfp8_grouped_gemm":
        raise RuntimeError(f"{path}: unexpected FlashInfer benchmark tag")
    validate_common(document, path)
    config = document["config"]
    if config.get("api_scale_granularity_mnk") != [1, 1, 32]:
        raise RuntimeError(f"{path}: FlashInfer API granularity is not [1,1,32]")
    if config.get("is_gated") is not False or config.get("scale_encoding") != "UE8M0":
        raise RuntimeError(f"{path}: unexpected FlashInfer dtype/configuration")
    if document["environment"].get("flashinfer_commit") != commit:
        raise RuntimeError(f"{path}: FlashInfer commit mismatch")
    if not document.get("source_validation", {}).get(
        "helper_and_imported_package_share_flashinfer_repo"
    ):
        raise RuntimeError(f"{path}: source checkout validation is missing")
    for record in document["results"]:
        correctness = record["correctness"]
        if correctness.get("status") != "passed" or not correctness.get("all_finite"):
            raise RuntimeError(f"{path}: FlashInfer correctness failed")
        if correctness["relative_l2_vs_dequantized_reference"] > correctness[
            "relative_l2_threshold"
        ]:
            raise RuntimeError(f"{path}: FlashInfer relative-L2 check failed")


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def relative_link(path: Path, output: Path) -> str:
    return os.path.relpath(path.resolve(), output.parent.resolve())


def main() -> int:
    args = parse_args()
    suites = []
    for label, directory in args.suite:
        paths = {
            "sonic": directory / "grouped-vs-dense.json",
            "old": directory / "flashinfer-old-grouped.json",
            "new": directory / "flashinfer-pr4660-grouped.json",
        }
        docs = {key: load(path) for key, path in paths.items()}
        validate_sonic(docs["sonic"], paths["sonic"])
        validate_flashinfer(docs["old"], paths["old"], OLD_COMMIT)
        validate_flashinfer(docs["new"], paths["new"], NEW_COMMIT)
        node_labels = {
            docs["sonic"]["environment"]["node_label"],
            docs["old"]["environment"]["node_label"],
            docs["new"]["environment"]["node_label"],
        }
        if node_labels != {label}:
            raise RuntimeError(f"{directory}: node aliases do not all equal {label}")
        suites.append(
            {
                "label": label,
                "paths": paths,
                "docs": docs,
                "results": {key: index_results(doc) for key, doc in docs.items()},
            }
        )

    lines = [
        "# PRO5000 single-GPU MXFP8 grouped GEMM: Sonic vs FlashInfer",
        "",
        "All three columns use eight balanced groups, semantic K32 UE8M0 scales, "
        "BF16 output, prequantized kernel-only timing, 20 warmups, 100 timed CUDA-event "
        "iterations, and three repeats. The displayed value is the median repeat P50.",
        "",
        "| node | workload | Sonic (ms) | FlashInfer old (ms) | FlashInfer PR #4660 (ms) | PR vs old | Sonic speedup vs PR |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for suite in suites:
        for name in WORKLOADS:
            sonic = suite["results"]["sonic"][name]["aggregate"]["grouped"]["p50_ms"]
            old = suite["results"]["old"][name]["aggregate"]["grouped"]["p50_ms"]
            new = suite["results"]["new"][name]["aggregate"]["grouped"]["p50_ms"]
            gain = 100.0 * (old - new) / old
            lines.append(
                f"| {suite['label']} | {name} | {sonic:.6f} | {old:.6f} | "
                f"{new:.6f} | {gain:+.2f}% | {new / sonic:.3f}x |"
            )

    lines.extend(["", "## Geometric-mean summary", ""])
    for suite in suites:
        old_values = []
        new_values = []
        sonic_values = []
        for name in WORKLOADS:
            sonic_values.append(
                suite["results"]["sonic"][name]["aggregate"]["grouped"]["p50_ms"]
            )
            old_values.append(
                suite["results"]["old"][name]["aggregate"]["grouped"]["p50_ms"]
            )
            new_values.append(
                suite["results"]["new"][name]["aggregate"]["grouped"]["p50_ms"]
            )
        pr_gain = 100.0 * (1.0 - geomean(new_values) / geomean(old_values))
        sonic_speedup = geomean(
            [new / sonic for new, sonic in zip(new_values, sonic_values)]
        )
        repeat_spreads = []
        for implementation in ("old", "new"):
            for name in WORKLOADS:
                repeat_p50 = [
                    repeat["grouped"]["p50_ms"]
                    for repeat in suite["results"][implementation][name][
                        "repeat_metrics"
                    ]
                ]
                repeat_spreads.append(
                    100.0
                    * (max(repeat_p50) - min(repeat_p50))
                    / sorted(repeat_p50)[len(repeat_p50) // 2]
                )
        lines.append(
            f"- {suite['label']}: PR #4660 vs old {pr_gain:+.2f}%; "
            f"Sonic vs PR #4660 {sonic_speedup:.3f}x; largest within-result "
            f"repeat-P50 range {max(repeat_spreads):.2f}%."
        )

    lines.extend(
        [
            "",
            "Positive PR vs old means lower latency. Sonic speedup vs PR is "
            "FlashInfer-PR latency divided by Sonic latency, so values above one favor Sonic.",
            "",
            "This is a single-GPU grouped-kernel comparison. It contains no EP, routing, "
            "NCCL, dispatch/combine, or NUMA traffic.",
            "",
            "## Raw data",
            "",
        ]
    )
    for suite in suites:
        links = ", ".join(
            f"[{key}]({relative_link(path, args.output)})"
            for key, path in suite["paths"].items()
        )
        lines.append(f"- {suite['label']}: {links}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
