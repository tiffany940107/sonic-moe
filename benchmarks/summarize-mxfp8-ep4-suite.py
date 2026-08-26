#!/usr/bin/env python3
"""Validate and summarize the public Sonic-MXFP8 EP4 customer-case results."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


CASES = {
    "balanced": ("uniform", "contiguous"),
    "zipf-contiguous": ("zipf", "contiguous"),
    "zipf-greedy": ("zipf", "greedy"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--expected-restarts", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{path}:{line_number}: invalid JSON: {error}") from error
    return records


def validate_public_label(label: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise RuntimeError("node_label must be an anonymous alphanumeric alias")
    if re.fullmatch(r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}", label):
        raise RuntimeError("node_label must not be an IP address")


def validate_record(record: dict, case_name: str) -> None:
    routing, placement = CASES[case_name]
    expected = {
        "benchmark": "sonic_mxfp8_ep_e2e",
        "world_size": 4,
        "global_tokens": 16384,
        "source_tokens_per_rank": 4096,
        "top_k": 24,
        "global_experts": 768,
        "local_experts": 192,
        "hidden": 2560,
        "intermediate_after_swiglu": 1024,
        "fc1_physical_output": 2048,
        "routing": routing,
        "placement": placement,
        "warmup": 20,
        "iterations": 100,
        "activation_transport": "mxfp8",
        "placement_migration_in_timed_forward": False,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(
                f"{case_name}: {key}={record.get(key)!r}, expected {value!r}"
            )
    if "host" in record:
        raise RuntimeError(f"{case_name}: public result must not contain host")
    validate_public_label(record.get("node_label", ""))
    if record.get("topology") not in {
        "same_numa",
        "cross_numa_2plus2",
        "unspecified",
    }:
        raise RuntimeError(f"{case_name}: invalid topology label")
    quant = record.get("quantization", {})
    if quant.get("values") != "OCP MXFP8 E4M3":
        raise RuntimeError(f"{case_name}: unexpected value format")
    if quant.get("scales") != "E8M0" or quant.get("group_size_k") != 32:
        raise RuntimeError(f"{case_name}: expected E8M0 K32 scale factors")
    correctness = (
        record.get("transport_identity_max_abs", math.inf),
        record.get("transport_identity_rel_l2", math.inf),
        record.get("mxfp8_transport_vs_bf16_rel_l2", math.inf),
        record.get("mxfp8_transport_vs_bf16_cosine", -math.inf),
    )
    if not all(math.isfinite(value) for value in correctness):
        raise RuntimeError(f"{case_name}: non-finite correctness metric")
    if correctness[0] > 0.05 or correctness[1] > 0.01:
        raise RuntimeError(f"{case_name}: dispatch/combine identity check failed")
    if correctness[2] > 0.01 or correctness[3] < 0.99:
        raise RuntimeError(f"{case_name}: MXFP8 transport comparison failed")


def main() -> int:
    args = parse_args()
    suites: dict[str, list[dict]] = {}
    for case_name in CASES:
        path = args.result_dir / f"{case_name}.jsonl"
        records = load_jsonl(path)
        if len(records) != args.expected_restarts:
            raise RuntimeError(
                f"{path}: found {len(records)} records, expected {args.expected_restarts}"
            )
        for record in records:
            validate_record(record, case_name)
        suites[case_name] = records

    labels = {record["node_label"] for records in suites.values() for record in records}
    topologies = {record["topology"] for records in suites.values() for record in records}
    if len(labels) != 1 or len(topologies) != 1:
        raise RuntimeError("all records must use one node alias and one topology label")

    medians = {
        case_name: statistics.median(
            record["e2e_rank_max"]["p50_ms"] for record in records
        )
        for case_name, records in suites.items()
    }
    improvement = 100.0 * (
        medians["zipf-contiguous"] - medians["zipf-greedy"]
    ) / medians["zipf-contiguous"]
    lines = [
        "# Sonic-MXFP8 EP4 result",
        "",
        f"- Node: `{next(iter(labels))}`",
        f"- Topology: `{next(iter(topologies))}`",
        "- Shape: global tokens 16384, 4096/rank, top-k 24, hidden 2560, "
        "post-SwiGLU intermediate 1024, 768 global / 192 local experts",
        "- Timing: 20 warmups, 100 iterations, 3 process restarts; displayed value "
        "is the median of restart P50 rank-critical-path CUDA-event latency",
        "- Boundary: routing setup and static EPLB weight migration are excluded; "
        "dispatch, NCCL all-to-all, grouped FC1/FC2, local reduction, and combine are included",
        "",
        "| case | latency (ms) |",
        "| --- | ---: |",
        f"| balanced | {medians['balanced']:.3f} |",
        f"| Zipf contiguous | {medians['zipf-contiguous']:.3f} |",
        f"| Zipf static greedy | {medians['zipf-greedy']:.3f} |",
        "",
        f"Static greedy reduces Zipf steady-state latency by **{improvement:.2f}%**. "
        "This percentage excludes its one-time planning and weight-migration cost.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
