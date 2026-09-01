"""Windowed EPLB policy for the Sonic EP4 benchmark/runtime prototype."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import torch

from ep4_support.routing import greedy_placement

from .static_greedy import limit_migration, placement_rank_loads

EPLBAction = Literal["none", "migrate", "experimental_replica"]


@dataclass(frozen=True)
class EPLBDecision:
    applied: bool
    action: EPLBAction
    reason: str
    route_map_version: int
    rank_ratio: float
    hot_expert_ratio: float
    changed_experts: list[int]
    predicted_before: int
    predicted_after: int
    remote_records_before: int | None
    remote_records_after: int | None
    predicted_candidate_ms: float
    predicted_saving_ms: float
    reconfiguration_ms: float
    break_even_steps: float | None
    experimental: bool = False


class WindowedEPLB:
    """Versioned equal-capacity EPLB controller with persistence and hysteresis.

    The stable path only emits whole-expert migration decisions.  Hot-expert
    replication is opt-in and is returned as an experimental recommendation;
    it never silently changes placement or the route-map version.
    """

    def __init__(
        self,
        placement: torch.Tensor,
        ep_size: int,
        *,
        window: int = 16,
        update_interval: int = 16,
        migration_limit: int = 32,
        observe_threshold: float = 1.15,
        apply_threshold: float = 1.30,
        replica_threshold: float = 1.50,
        replica_hot_expert_ratio: float = 4.0,
        replica_slots_per_rank: int = 8,
        replica_max_copies: int = 4,
        persistence_windows: int = 2,
        minimum_saving_ms: float = 0.05,
        enable_experimental_replica: bool = False,
    ) -> None:
        if window <= 0 or update_interval <= 0 or persistence_windows <= 0:
            raise ValueError("window, update_interval and persistence_windows must be positive")
        if migration_limit < 0:
            raise ValueError("migration_limit must be non-negative")
        if not 1.0 <= observe_threshold <= apply_threshold:
            raise ValueError("thresholds must satisfy 1 <= observe <= apply")
        if replica_threshold < apply_threshold:
            raise ValueError("replica_threshold must be at least apply_threshold")
        if ep_size <= 0:
            raise ValueError("ep_size must be positive")
        placement = placement.to(torch.int64).cpu().clone()
        if placement.numel() == 0 or int(placement.min()) < 0 or int(placement.max()) >= ep_size:
            raise ValueError("placement contains an invalid rank")
        capacity = torch.bincount(placement, minlength=ep_size)
        if not bool((capacity == capacity[0]).all()):
            raise ValueError("windowed EPLB requires equal expert capacity per rank")

        self.placement = placement
        self.ep_size = ep_size
        self.window = window
        self.update_interval = update_interval
        self.migration_limit = migration_limit
        self.observe_threshold = observe_threshold
        self.apply_threshold = apply_threshold
        self.replica_threshold = replica_threshold
        self.replica_hot_expert_ratio = replica_hot_expert_ratio
        self.replica_slots_per_rank = replica_slots_per_rank
        self.replica_max_copies = replica_max_copies
        self.persistence_windows = persistence_windows
        self.minimum_saving_ms = minimum_saving_ms
        self.enable_experimental_replica = enable_experimental_replica
        self.history: deque[torch.Tensor] = deque(maxlen=window)
        self.step = 0
        self.route_map_version = 0
        self._persistent_windows = 0

    def observe(
        self,
        expert_loads: torch.Tensor,
        *,
        unoptimized_step_ms: float,
        reconfiguration_ms: float,
        source_token_experts: torch.Tensor | None = None,
        remote_record_cost_ms: float = 0.0,
        amortization_horizon_steps: int | None = None,
    ) -> EPLBDecision:
        """Observe one histogram and optionally apply a new static placement.

        ``source_token_experts`` is an optional global routing sample with shape
        ``[ep_size, tokens_per_rank, top_k]`` (or the flattened two-dimensional
        equivalent).  When supplied, the controller predicts the same
        source-token/destination-rank records used by deduplicated EP dispatch.
        This is intentionally not a source/expert pair histogram: multiple
        routes from one token to the same destination rank share one activation
        record on the wire.
        """
        if min(unoptimized_step_ms, reconfiguration_ms, remote_record_cost_ms) < 0:
            raise ValueError("latency inputs must be non-negative")
        if amortization_horizon_steps is not None and amortization_horizon_steps <= 0:
            raise ValueError("amortization_horizon_steps must be positive")
        loads = expert_loads.to(torch.int64).cpu()
        if loads.shape != self.placement.shape or bool((loads < 0).any()):
            raise ValueError("expert_loads must be a non-negative vector matching placement")
        self.history.append(loads)
        self.step += 1
        aggregate = torch.stack(tuple(self.history)).sum(0)
        before_loads = placement_rank_loads(aggregate, self.placement, self.ep_size)
        mean_rank = before_loads.to(torch.float64).mean().clamp_min(1.0)
        rank_ratio = float(before_loads.max() / mean_rank)
        mean_expert = aggregate.to(torch.float64).mean().clamp_min(1.0)
        hot_expert_ratio = float(aggregate.max() / mean_expert)
        predicted_before = int(before_loads.max())
        routes = None
        if source_token_experts is not None:
            routes = source_token_experts.to(torch.int64).cpu()
            if routes.ndim == 2:
                if routes.shape[0] % self.ep_size:
                    raise ValueError(
                        "flattened source_token_experts rows must divide ep_size"
                    )
                routes = routes.view(self.ep_size, -1, routes.shape[1])
            if routes.ndim != 3 or routes.shape[0] != self.ep_size:
                raise ValueError(
                    "source_token_experts must have shape (ep_size, tokens, top_k)"
                )
            if routes.numel() and (
                int(routes.min()) < 0 or int(routes.max()) >= self.placement.numel()
            ):
                raise ValueError("source_token_experts contains an invalid expert")
            route_loads = torch.bincount(
                routes.reshape(-1), minlength=self.placement.numel()
            )
            if not torch.equal(route_loads, loads):
                raise ValueError("source_token_experts do not match expert_loads")

        def remote_records(candidate: torch.Tensor) -> int | None:
            if routes is None:
                return None
            owners = candidate.to(torch.int64)[routes]
            count = 0
            for source in range(self.ep_size):
                for destination in range(self.ep_size):
                    if destination != source:
                        count += int((owners[source] == destination).any(dim=1).sum())
            return count

        before_remote_records = remote_records(self.placement)

        def candidate_cost(
            candidate_load: int, candidate_remote_records: int | None
        ) -> float:
            compute_ms = unoptimized_step_ms * candidate_load / max(1, predicted_before)
            communication_delta_ms = 0.0
            if (
                before_remote_records is not None
                and candidate_remote_records is not None
            ):
                communication_delta_ms = (
                    candidate_remote_records - before_remote_records
                ) * remote_record_cost_ms
            return compute_ms + communication_delta_ms

        def no_change(
            reason: str,
            *,
            action: EPLBAction = "none",
            predicted_after: int | None = None,
            candidate_remote_records: int | None = None,
            candidate_ms: float | None = None,
            saving_ms: float = 0.0,
            experimental: bool = False,
        ) -> EPLBDecision:
            return EPLBDecision(
                applied=False,
                action=action,
                reason=reason,
                route_map_version=self.route_map_version,
                rank_ratio=rank_ratio,
                hot_expert_ratio=hot_expert_ratio,
                changed_experts=[],
                predicted_before=predicted_before,
                predicted_after=(predicted_before if predicted_after is None else predicted_after),
                remote_records_before=before_remote_records,
                remote_records_after=(
                    before_remote_records
                    if candidate_remote_records is None
                    else candidate_remote_records
                ),
                predicted_candidate_ms=(
                    unoptimized_step_ms if candidate_ms is None else candidate_ms
                ),
                predicted_saving_ms=saving_ms,
                reconfiguration_ms=reconfiguration_ms,
                break_even_steps=(reconfiguration_ms / saving_ms if saving_ms > 0 else None),
                experimental=experimental,
            )

        if len(self.history) < self.window or self.step % self.update_interval:
            return no_change("window_or_interval")
        replica_signal = self.enable_experimental_replica and (
            rank_ratio >= self.replica_threshold
            or hot_expert_ratio >= self.replica_hot_expert_ratio
        )
        if rank_ratio < self.observe_threshold and not replica_signal:
            self._persistent_windows = 0
            return no_change("below_observe_threshold")
        if rank_ratio < self.apply_threshold and not replica_signal:
            self._persistent_windows = 0
            return no_change("ema_only")

        self._persistent_windows += 1
        full_target = greedy_placement(aggregate, self.ep_size)
        full_after = int(placement_rank_loads(aggregate, full_target, self.ep_size).max())
        full_remote_records = remote_records(full_target)
        full_candidate_ms = candidate_cost(full_after, full_remote_records)

        if self._persistent_windows < self.persistence_windows:
            return no_change("await_persistence", predicted_after=full_after)

        if replica_signal and hot_expert_ratio >= self.replica_hot_expert_ratio:
            # Importing here makes the stable controller independent from the
            # experimental replica implementation and its evolving API.
            from .experimental_sonic_replica import plan_replicas

            replica = plan_replicas(
                aggregate,
                self.placement,
                self.ep_size,
                self.replica_slots_per_rank,
                max_copies_per_expert=self.replica_max_copies,
                minimum_hot_expert_ratio=self.replica_hot_expert_ratio,
            )
            replica_after = max(replica.estimated_rank_loads)
            if replica.count and replica_after < full_after:
                saving = unoptimized_step_ms * (
                    predicted_before - replica_after
                ) / max(1, predicted_before)
                if saving < self.minimum_saving_ms:
                    return no_change(
                        "replica_benefit_too_small",
                        action="experimental_replica",
                        predicted_after=replica_after,
                        saving_ms=saving,
                        experimental=True,
                    )
                break_even_steps = reconfiguration_ms / saving
                if (
                    amortization_horizon_steps is not None
                    and break_even_steps > amortization_horizon_steps
                ):
                    return no_change(
                        "replica_amortization_horizon_exceeded",
                        action="experimental_replica",
                        predicted_after=replica_after,
                        saving_ms=saving,
                        experimental=True,
                    )
                return no_change(
                    "prefer_experimental_replica",
                    action="experimental_replica",
                    predicted_after=replica_after,
                    saving_ms=saving,
                    experimental=True,
                )

        # A hot expert can be severe while aggregate rank load still looks
        # balanced.  If replication cannot beat whole-expert placement, do not
        # fall through and migrate merely because the hot signal bypassed the
        # rank-ratio observation gate.
        if rank_ratio < self.observe_threshold:
            return no_change("hot_expert_replica_not_beneficial")
        if rank_ratio < self.apply_threshold:
            return no_change("ema_only")

        full_saving = unoptimized_step_ms - full_candidate_ms
        if full_saving < self.minimum_saving_ms:
            return no_change(
                "benefit_too_small",
                predicted_after=full_after,
                candidate_remote_records=full_remote_records,
                candidate_ms=full_candidate_ms,
                saving_ms=full_saving,
            )

        candidate = limit_migration(
            self.placement, full_target, self.migration_limit
        )
        changed = torch.where(candidate != self.placement)[0].tolist()
        if not changed:
            return no_change("migration_limit", predicted_after=full_after, saving_ms=full_saving)
        predicted_after = int(
            placement_rank_loads(aggregate, candidate, self.ep_size).max()
        )
        candidate_remote_records = remote_records(candidate)
        predicted_candidate_ms = candidate_cost(
            predicted_after, candidate_remote_records
        )
        if predicted_after >= predicted_before:
            return no_change(
                "partial_not_beneficial",
                predicted_after=predicted_after,
                saving_ms=full_saving,
            )

        actual_saving = unoptimized_step_ms - predicted_candidate_ms
        if actual_saving < self.minimum_saving_ms:
            return no_change(
                "communication_cost_outweighs_compute",
                predicted_after=predicted_after,
                candidate_remote_records=candidate_remote_records,
                candidate_ms=predicted_candidate_ms,
                saving_ms=actual_saving,
            )
        break_even_steps = reconfiguration_ms / actual_saving
        if (
            amortization_horizon_steps is not None
            and break_even_steps > amortization_horizon_steps
        ):
            return no_change(
                "amortization_horizon_exceeded",
                predicted_after=predicted_after,
                candidate_remote_records=candidate_remote_records,
                candidate_ms=predicted_candidate_ms,
                saving_ms=actual_saving,
            )
        self.placement = candidate
        self.route_map_version += 1
        self._persistent_windows = 0
        return EPLBDecision(
            applied=True,
            action="migrate",
            reason="applied",
            route_map_version=self.route_map_version,
            rank_ratio=rank_ratio,
            hot_expert_ratio=hot_expert_ratio,
            changed_experts=changed,
            predicted_before=predicted_before,
            predicted_after=predicted_after,
            remote_records_before=before_remote_records,
            remote_records_after=candidate_remote_records,
            predicted_candidate_ms=predicted_candidate_ms,
            predicted_saving_ms=actual_saving,
            reconfiguration_ms=reconfiguration_ms,
            break_even_steps=break_even_steps,
        )


__all__ = ["EPLBAction", "EPLBDecision", "WindowedEPLB"]
