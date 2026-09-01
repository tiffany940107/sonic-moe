#!/usr/bin/env python3
"""Replay workload histograms through the versioned windowed EPLB policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ep4_support.placement.windowed_controller import WindowedEPLB
from ep4_support.routing import contiguous_placement


def parse_profile(value: str) -> tuple[str, Path, int]:
    try:
        label, remainder = value.split("=", 1)
        path, steps = remainder.rsplit(":", 1)
        return label, Path(path), int(steps)
    except (ValueError, IndexError) as error:
        raise argparse.ArgumentTypeError(
            "profile must be LABEL=SIDECAR.json:STEPS"
        ) from error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="append", type=parse_profile, required=True)
    parser.add_argument("--unoptimized-step-ms", type=float, required=True)
    parser.add_argument("--optimized-step-ms", type=float, required=True)
    parser.add_argument("--reconfiguration-ms", type=float, required=True)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--update-interval", type=int, default=16)
    parser.add_argument("--persistence-windows", type=int, default=2)
    parser.add_argument("--migration-limit", type=int, default=32)
    parser.add_argument("--observe-threshold", type=float, default=1.15)
    parser.add_argument("--apply-threshold", type=float, default=1.30)
    parser.add_argument("--replica-threshold", type=float, default=1.50)
    parser.add_argument("--replica-hot-expert-ratio", type=float, default=4.0)
    parser.add_argument("--replica-slots-per-rank", type=int, default=8)
    parser.add_argument("--replica-max-copies", type=int, default=4)
    parser.add_argument("--minimum-saving-ms", type=float, default=0.05)
    parser.add_argument(
        "--experimental-replica",
        action="store_true",
        help="allow experimental replica recommendations; disabled by default",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    controller = WindowedEPLB(
        contiguous_placement(512, 4),
        4,
        window=args.window,
        update_interval=args.update_interval,
        persistence_windows=args.persistence_windows,
        migration_limit=args.migration_limit,
        observe_threshold=args.observe_threshold,
        apply_threshold=args.apply_threshold,
        replica_threshold=args.replica_threshold,
        replica_hot_expert_ratio=args.replica_hot_expert_ratio,
        replica_slots_per_rank=args.replica_slots_per_rank,
        replica_max_copies=args.replica_max_copies,
        minimum_saving_ms=args.minimum_saving_ms,
        enable_experimental_replica=args.experimental_replica,
    )
    timeline = []
    step = 0
    for label, sidecar, profile_steps in args.profile:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("experts") != 512 or metadata.get("ep_size") != 4:
            raise ValueError(f"unsupported sidecar shape: {sidecar}")
        loads = torch.tensor(metadata["post_ep_global_expert_counts"], dtype=torch.int64)
        for _ in range(profile_steps):
            step += 1
            decision = controller.observe(
                loads,
                unoptimized_step_ms=args.unoptimized_step_ms,
                reconfiguration_ms=args.reconfiguration_ms,
            )
            timeline.append(
                {
                    "step": step,
                    "profile": label,
                    "applied": decision.applied,
                    "action": decision.action,
                    "reason": decision.reason,
                    "experimental": decision.experimental,
                    "route_map_version": decision.route_map_version,
                    "rank_ratio": decision.rank_ratio,
                    "hot_expert_ratio": decision.hot_expert_ratio,
                    "changed_expert_count": len(decision.changed_experts),
                    "changed_experts": decision.changed_experts,
                    "predicted_before": decision.predicted_before,
                    "predicted_after": decision.predicted_after,
                    "predicted_saving_ms": decision.predicted_saving_ms,
                    "break_even_steps": decision.break_even_steps,
                }
            )

    horizons = [1, 2, 4, 8, 16, 32, 64, 128]
    amortized = [
        {
            "stable_steps": horizon,
            "amortized_ms": args.optimized_step_ms
            + args.reconfiguration_ms / horizon,
            "beats_unoptimized": args.optimized_step_ms
            + args.reconfiguration_ms / horizon
            < args.unoptimized_step_ms,
        }
        for horizon in horizons
    ]
    record = {
        "schema_version": 1,
        "analysis": "windowed_static_eplb",
        "experimental_replica_enabled": args.experimental_replica,
        "controller": {
            "window": args.window,
            "update_interval": args.update_interval,
            "persistence_windows": args.persistence_windows,
            "migration_limit": args.migration_limit,
            "observe_threshold": controller.observe_threshold,
            "apply_threshold": controller.apply_threshold,
            "replica_threshold": controller.replica_threshold,
            "replica_hot_expert_ratio": controller.replica_hot_expert_ratio,
            "replica_slots_per_rank": controller.replica_slots_per_rank,
            "replica_max_copies": controller.replica_max_copies,
            "minimum_saving_ms": controller.minimum_saving_ms,
        },
        "measured_inputs": {
            "unoptimized_step_ms": args.unoptimized_step_ms,
            "optimized_step_ms": args.optimized_step_ms,
            "reconfiguration_ms": args.reconfiguration_ms,
        },
        "measured_break_even_steps": (
            args.reconfiguration_ms
            / (args.unoptimized_step_ms - args.optimized_step_ms)
            if args.unoptimized_step_ms > args.optimized_step_ms
            else None
        ),
        "timeline": timeline,
        "amortized": amortized,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "applied_steps": [item["step"] for item in timeline if item["applied"]],
                "experimental_steps": [
                    item["step"] for item in timeline if item["experimental"]
                ],
                "measured_break_even_steps": record["measured_break_even_steps"],
                "reasons": sorted({item["reason"] for item in timeline}),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
