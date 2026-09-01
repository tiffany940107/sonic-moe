#!/usr/bin/env python3
"""Generate public EP4 workloads for the 512-expert MXFP8 EPLB campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from ep4_support.workloads import (
    TOKEN_WORKLOADS,
    make_trace,
    scenarios_for,
    validate_trace,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--m", type=int, nargs="+", default=list(TOKEN_WORKLOADS))
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for tokens in args.m:
        for scenario in scenarios_for(tokens):
            path = args.output_dir / f"m{tokens}_{scenario}.pt"
            sidecar = path.with_suffix(".json")
            if (path.exists() or sidecar.exists()) and not args.overwrite:
                raise FileExistsError(f"refusing to overwrite {path} or {sidecar}")
            ids, weights, metadata = make_trace(tokens, scenario, args.seed + tokens)
            validate_trace(ids, weights, metadata)
            torch.save({"expert_ids": ids, "weights": weights}, path)
            metadata["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            metadata["file"] = path.name
            sidecar.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            manifest.append(metadata)
            print(
                json.dumps(
                    {
                        "file": path.name,
                        "m": tokens,
                        "rank_ratio": metadata["actual_rank_max_over_mean"],
                        "scenario": scenario,
                        "sha256": metadata["sha256"],
                        "source_segment": metadata["source_segment_metrics"][0],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    (args.output_dir / "trace_manifest.json").write_text(
        json.dumps({"schema_version": 1, "traces": manifest}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
