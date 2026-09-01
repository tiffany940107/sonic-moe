#!/usr/bin/env python3
"""Fail closed when the publishable 0902 result package is incomplete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TOKENS = (8192, 16384, 32768, 46080)
NODES = ("PRO5000-A", "PRO5000-B")


def key(item: dict) -> tuple:
    return (
        item["node_label"],
        item["topology"],
        item["backend"],
        item["variant"],
        item["source_tokens_per_rank"],
        item["trace"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    results = {key(item): item for item in summary["supported_results"]}
    statuses = summary["status_records"]
    checks: list[dict] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    check(
        "all_admitted_groups_have_three_restarts",
        all(item["restarts"] == 3 for item in results.values()),
        f"groups={len(results)}",
    )
    check(
        "all_inline_correctness_gates_pass",
        all(item["all_inline_correct"] for item in results.values()),
        "no admitted Sonic/block-scale record has a failed C4 gate",
    )

    for node in NODES:
        for tokens in TOKENS:
            trace = f"m{tokens}_balanced"
            for backend, variant in (
                ("sonic", "baseline_contiguous"),
                ("sonic", "fused_segmented_contiguous"),
                ("sonic", "fused_segmented_transport_workspace_contiguous"),
                ("blockscale_groupwise", "contiguous"),
            ):
                check(
                    f"balanced_{node}_{backend}_{variant}_{tokens}",
                    (node, "same_numa", backend, variant, tokens, trace) in results,
                    "required balanced same-NUMA result",
                )
        for tokens in (8192, 16384):
            trace = f"m{tokens}_balanced"
            item = results.get(
                (
                    node,
                    "same_numa",
                    "megamoe_native",
                    "p2p_direct_p95_p99",
                    tokens,
                    trace,
                )
            )
            check(
                f"megamoe_tail_percentiles_{node}_{tokens}",
                item is not None
                and len(item["restart_p95_ms"]) == 3
                and len(item["restart_p99_ms"]) == 3,
                "reporting-only p95/p99 patch, three restarts",
            )
        for tokens in (32768, 46080):
            check(
                f"megamoe_signed_extent_status_{node}_{tokens}",
                any(
                    item.get("node_label") == node
                    and item.get("backend") == "megamoe_native"
                    and item.get("source_tokens_per_rank") == tokens
                    and item.get("status") == "unsupported_signed_int32_extent"
                    for item in statuses
                ),
                "unsupported is explicit and has no admitted latency",
            )
        check(
            f"cross_numa_resource_status_{node}",
            any(
                item.get("node_label") == node
                and item.get("topology") == "cross_numa_2plus2"
                and item.get("status") == "resource_blocked_external_gpu_memory"
                for item in statuses
            ),
            "no same-NUMA substitution",
        )

    local = summary["local_timing_results"]
    check(
        "k0_k1_both_nodes_three_restarts",
        len(local) == 4
        and all(item["restarts"] == 3 and item["all_correct"] for item in local),
        f"local_groups={len(local)}",
    )
    k1 = [item for item in local if item["timing_level"].startswith("K1_")]
    check(
        "k1_zero_timed_allocator_activity",
        len(k1) == 2
        and all(
            max(item["max_timed_allocator_delta"].values(), default=0) == 0
            for item in k1
        ),
        "caller-owned local workspaces",
    )

    single_hot_variants = (
        "fused_segmented_transport_workspace_contiguous",
        "fused_segmented_transport_workspace_greedy",
        "fused_segmented_transport_workspace_contiguous_replica1",
    )
    check(
        "single_hot_sonic_control_matrix",
        all(
            (
                "PRO5000-B",
                "same_numa",
                "sonic",
                variant,
                16384,
                "m16384_persistent_single_hot",
            )
            in results
            for variant in single_hot_variants
        ),
        "contiguous, whole-expert migration, and replica split",
    )
    check(
        "single_hot_megamoe_three_restarts",
        (
            "PRO5000-B",
            "same_numa",
            "megamoe_native",
            "p2p_direct_p95_p99",
            16384,
            "m16384_persistent_single_hot",
        )
        in results,
        "same common trace",
    )

    hashes: dict[str, set[str]] = {}
    for item in results.values():
        if item.get("trace_sha256"):
            hashes.setdefault(item["trace"], set()).add(item["trace_sha256"])
    mismatched = sorted(name for name, values in hashes.items() if len(values) != 1)
    check(
        "common_trace_hashes_match_across_backends",
        not mismatched,
        f"mismatched={mismatched}",
    )

    failed = [item for item in checks if not item["passed"]]
    record = {
        "schema_version": 1,
        "audit": "0902_completion",
        "passed": not failed,
        "checks": checks,
        "failed_checks": [item["name"] for item in failed],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"checks": len(checks), "failed": len(failed)}, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
