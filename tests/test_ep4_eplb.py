from __future__ import annotations

import sys
from pathlib import Path

import torch

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

from ep4_support.placement.experimental_sonic_replica import (
    allocate_quota,
    materialize_comm_aware,
    plan_replicas,
)
from ep4_support.placement.sonic_migration import migration_moves
from ep4_support.placement.static_greedy import (
    capacity_preserving_cycles,
    limit_migration,
    placement_rank_loads,
)
from ep4_support.placement.windowed_controller import WindowedEPLB
from ep4_support.routing import contiguous_placement, greedy_placement
from ep4_support.workloads import (
    BASE_TOKENS,
    TOKEN_WORKLOADS,
    TOP_K,
    base_segment_profile,
    scaled_segment_profile,
)


def test_customer_segment_profile_anchors_and_conserves_pairs():
    profile = base_segment_profile()
    ordered = profile.sort().values
    assert profile.shape == (512,)
    assert int(profile.sum()) == BASE_TOKENS * TOP_K
    assert int(profile.max()) == 16_384
    assert float(torch.quantile(ordered.float(), 0.5)) == 517
    assert float(torch.quantile(ordered.float(), 0.9)) == 2528


def test_every_requested_token_length_has_a_conserved_scaled_profile():
    for tokens in TOKEN_WORKLOADS:
        profile = scaled_segment_profile(tokens)
        assert profile.shape == (512,)
        assert int(profile.sum()) == tokens * TOP_K
        assert int(profile.max()) == tokens
        assert int(profile.min()) >= 0


def test_lpt_is_equal_capacity_and_reduces_straggler():
    loads = torch.arange(1, 513, dtype=torch.int64)
    initial = contiguous_placement(512, 4)
    target = greedy_placement(loads, 4)
    assert torch.bincount(target, minlength=4).tolist() == [128] * 4
    assert int(placement_rank_loads(loads, target, 4).max()) < int(
        placement_rank_loads(loads, initial, 4).max()
    )


def test_migration_limit_only_applies_capacity_preserving_cycles():
    current = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    target = torch.tensor([1, 0, 2, 1, 3, 2, 0, 3])
    cycles = capacity_preserving_cycles(current, target)
    assert sorted(expert for cycle in cycles for expert in cycle) == [0, 2, 4, 6]
    limited = limit_migration(current, target, 4)
    assert torch.equal(limited, target)
    assert torch.bincount(limited, minlength=4).tolist() == [2, 2, 2, 2]
    unchanged = limit_migration(current, target, 3)
    assert torch.equal(unchanged, current)


def test_migration_move_lists_match_on_senders_and_receivers():
    current = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    target = torch.tensor([1, 0, 2, 1, 3, 2, 0, 3])
    sends = []
    receives = []
    for rank in range(4):
        send, receive = migration_moves(current, target, rank, 4)
        sends.append(send)
        receives.append(receive)
    for source in range(4):
        for destination in range(4):
            assert sends[source][destination] == receives[destination][source]


def test_windowed_controller_uses_hysteresis_and_versions_routes():
    placement = contiguous_placement(512, 4)
    loads = torch.ones(512, dtype=torch.int64) * 100
    loads[:128] *= 2
    controller = WindowedEPLB(
        placement,
        4,
        window=2,
        update_interval=2,
        persistence_windows=2,
        observe_threshold=1.05,
        apply_threshold=1.1,
        replica_threshold=2.0,
        minimum_saving_ms=0.0,
        migration_limit=512,
    )
    assert controller.observe(
        loads, unoptimized_step_ms=10, reconfiguration_ms=60
    ).reason == "window_or_interval"
    assert controller.observe(
        loads, unoptimized_step_ms=10, reconfiguration_ms=60
    ).reason == "await_persistence"
    assert controller.observe(
        loads, unoptimized_step_ms=10, reconfiguration_ms=60
    ).reason == "window_or_interval"
    decision = controller.observe(loads, unoptimized_step_ms=10, reconfiguration_ms=60)
    assert decision.applied
    assert decision.action == "migrate"
    assert decision.route_map_version == 1
    assert decision.predicted_after < decision.predicted_before


def test_replica_recommendation_is_explicitly_opt_in():
    placement = contiguous_placement(512, 4)
    loads = torch.ones(512, dtype=torch.int64) * 100
    loads[0] = 100_000
    stable = WindowedEPLB(
        placement,
        4,
        window=1,
        update_interval=1,
        persistence_windows=1,
        migration_limit=512,
        minimum_saving_ms=0.0,
    )
    stable_decision = stable.observe(
        loads, unoptimized_step_ms=120.0, reconfiguration_ms=190.0
    )
    assert stable_decision.action != "experimental_replica"
    assert not stable_decision.experimental

    experimental = WindowedEPLB(
        placement,
        4,
        window=1,
        update_interval=1,
        persistence_windows=1,
        migration_limit=512,
        minimum_saving_ms=0.0,
        enable_experimental_replica=True,
    )
    decision = experimental.observe(
        loads, unoptimized_step_ms=120.0, reconfiguration_ms=190.0
    )
    assert decision.action == "experimental_replica"
    assert decision.experimental and not decision.applied
    assert decision.route_map_version == 0


def test_experimental_replica_quota_conserves_every_pair():
    placement = contiguous_placement(512, 4)
    ids = torch.arange(32).repeat(4 * 128, 1)
    ids[:, 0] = 0
    ids[:, 1:] += 1
    loads = torch.bincount(ids.reshape(-1), minlength=512)
    plan = plan_replicas(loads, placement, 4, 2)
    rank_loads, quotas = allocate_quota(loads, placement, plan.replicas, 4)
    destinations, local_ids, _ = materialize_comm_aware(
        ids, placement, plan.replicas, quotas, 4, 2, 128
    )
    assert destinations.shape == ids.shape and local_ids.shape == ids.shape
    assert int((destinations < 0).sum()) == 0
    assert int((local_ids < 0).sum()) == 0
    assert int(rank_loads.sum()) == ids.numel()
    for expert, quota in quotas.items():
        assert sum(quota.values()) == int(loads[expert])
