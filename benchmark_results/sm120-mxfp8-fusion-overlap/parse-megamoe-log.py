#!/usr/bin/env python3
"""Normalize one valid MegaMoE native runner log into the 0902 schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time


PROBLEM = re.compile(
    r"tokens_per_rank=(?P<m>\d+) topk=(?P<topk>\d+) experts\(total=(?P<experts>\d+), "
    r"per_rank=(?P<local>\d+)\) \| hidden=(?P<hidden>\d+) intermediate=(?P<intermediate>\d+)"
)
TIMING = re.compile(
    r"iters=(?P<count>\d+) min=(?P<min>[0-9.]+) us p10=(?P<p10>[0-9.]+) us "
    r"p50=(?P<p50>[0-9.]+) us p90=(?P<p90>[0-9.]+) us "
    r"p95=(?P<p95>[0-9.]+) us p99=(?P<p99>[0-9.]+) us "
    r"max=(?P<max>[0-9.]+) us"
)
TRACE = re.compile(r"\[routing trace\] file=(?P<file>\S+) sha256=(?P<sha>[0-9a-f]{64})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node-label", choices=["PRO5000-A", "PRO5000-B"], required=True)
    parser.add_argument("--topology", choices=["same_numa", "cross_numa_2plus2"], required=True)
    parser.add_argument("--transport", choices=["p2p_direct", "nvshmem_hybrid"], required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--restart", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--reporting-patch", type=Path, required=True)
    parser.add_argument("--implementation-label", required=True)
    args = parser.parse_args()
    value = args.log.read_text(encoding="utf-8", errors="replace")
    problem = PROBLEM.search(value)
    trace = TRACE.search(value)
    if not problem or not trace:
        raise ValueError("log lacks problem or common trace provenance")
    marker = "---- CUDA-event target pipeline critical-path time"
    position = value.rfind(marker)
    timing = TIMING.search(value, position if position >= 0 else 0)
    if not timing:
        raise ValueError("log lacks CUDA-event target timing")
    lowered = value.lower()
    if any(marker in lowered for marker in ("validation failed", "illegal memory access", "segfault")):
        raise ValueError("invalid MegaMoE run cannot enter latency results")
    p = {key: int(item) for key, item in problem.groupdict().items()}
    t = {key: float(item) for key, item in timing.groupdict().items() if key != "count"}
    latency = {
        "count": int(timing.group("count")),
        "min_ms": t["min"] / 1000.0,
        "p10_ms": t["p10"] / 1000.0,
        "p50_ms": t["p50"] / 1000.0,
        "p90_ms": t["p90"] / 1000.0,
        "p95_ms": t["p95"] / 1000.0,
        "p99_ms": t["p99"] / 1000.0,
        "max_ms": t["max"] / 1000.0,
    }
    record = {
        "schema_version": 1,
        "benchmark": "megamoe_native_ep4",
        "backend": "megamoe_native",
        "source_commit": args.source_commit,
        "adapter_patch_sha256": hashlib.sha256(args.patch.read_bytes()).hexdigest(),
        "reporting_patch_sha256": hashlib.sha256(
            args.reporting_patch.read_bytes()
        ).hexdigest(),
        "implementation_label": args.implementation_label,
        "timestamp_unix": time.time(),
        "node_label": args.node_label,
        "topology": args.topology,
        "transport": args.transport,
        "run_label": args.run_label,
        "restart": args.restart,
        "world_size": 4,
        "source_tokens_per_rank": p["m"],
        "global_tokens": 4 * p["m"],
        "top_k": p["topk"],
        "global_experts": p["experts"],
        "local_experts": p["local"],
        "hidden": p["hidden"],
        "intermediate_after_swiglu": p["intermediate"] // 2,
        "fc1_physical_output": p["intermediate"],
        "routing_trace_file": trace.group("file"),
        "routing_trace_sha256": trace.group("sha"),
        "timing_level": "native_E0_prequantized",
        "timing_boundary": "native_fused_dispatch_fc1_swiglu_requant_fc2_combine_reduce",
        "e2e_rank_max": latency,
        "global_tokens_per_second": 4 * p["m"] / (latency["p50_ms"] / 1e3),
        "global_pairs_per_second": 4 * p["m"] * p["topk"] / (latency["p50_ms"] / 1e3),
        "correctness_gate": "strict reduced-token common-trace test recorded separately",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
