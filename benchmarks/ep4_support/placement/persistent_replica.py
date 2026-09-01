"""EXPERIMENTAL replica planner used by comparison wrappers.

This older generic path is retained for reproducibility.  It is not part of
the stable Sonic EPLB suite; new Sonic experiments use the explicitly named
``experimental_sonic_replica`` module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ep4_support.transport import local_expert_indices


@dataclass
class ReplicaPlan:
    replicas: dict[int, list[int]]
    slots_per_rank: int
    estimated_rank_loads: list[int]

    @property
    def physical_replica_count(self) -> int:
        return sum(len(ranks) for ranks in self.replicas.values())


def allocate_replica_quota(
    expert_loads: torch.Tensor,
    placement: torch.Tensor,
    replicas: dict[int, list[int]],
    ep_size: int,
) -> tuple[torch.Tensor, dict[int, dict[int, int]]]:
    """Assign every logical pair exactly once to the least-loaded valid replica."""
    rank_loads = [0] * ep_size
    quotas: dict[int, dict[int, int]] = {}
    order = sorted(range(expert_loads.numel()), key=lambda e: (-int(expert_loads[e]), e))
    for expert in order:
        count = int(expert_loads[expert])
        candidates = sorted({int(placement[expert]), *replicas.get(expert, [])})
        quota = {rank: 0 for rank in candidates}
        # Integer water filling over at most `ep_size` candidates. This is
        # exactly equivalent to one-pair-at-a-time least-loaded assignment but
        # avoids a Python loop over hundreds of thousands of pairs.
        remaining = count
        active: list[int] = []
        ordered = sorted(candidates, key=lambda rank: (rank_loads[rank], rank))
        for index, rank in enumerate(ordered):
            active.append(rank)
            if index + 1 == len(ordered):
                break
            level = rank_loads[rank]
            next_level = rank_loads[ordered[index + 1]]
            needed = (next_level - level) * len(active)
            if remaining < needed:
                q, rem = divmod(remaining, len(active))
                for pos, active_rank in enumerate(active):
                    add = q + int(pos < rem)
                    quota[active_rank] += add
                    rank_loads[active_rank] += add
                remaining = 0
                break
            for active_rank in active:
                add = next_level - rank_loads[active_rank]
                quota[active_rank] += add
                rank_loads[active_rank] = next_level
            remaining -= needed
        if remaining:
            q, rem = divmod(remaining, len(active))
            for pos, active_rank in enumerate(active):
                add = q + int(pos < rem)
                quota[active_rank] += add
                rank_loads[active_rank] += add
        quotas[expert] = {rank: value for rank, value in quota.items() if value}
    return torch.tensor(rank_loads, dtype=torch.int64), quotas


def plan_persistent_replicas(
    expert_loads: torch.Tensor,
    placement: torch.Tensor,
    ep_size: int,
    slots_per_rank: int,
) -> ReplicaPlan:
    if slots_per_rank < 0:
        raise ValueError("slots_per_rank must be non-negative")
    available = [slots_per_rank] * ep_size
    replicas: dict[int, list[int]] = {}
    current, _ = allocate_replica_quota(expert_loads, placement, replicas, ep_size)
    for _ in range(slots_per_rank * ep_size):
        best: tuple[int, int, int, torch.Tensor] | None = None
        hot_candidates = sorted(
            range(expert_loads.numel()), key=lambda e: (-int(expert_loads[e]), e)
        )[: max(32, slots_per_rank * ep_size * 4)]
        for expert in hot_candidates:
            existing = {int(placement[expert]), *replicas.get(expert, [])}
            for rank in range(ep_size):
                if rank in existing or available[rank] == 0:
                    continue
                candidate = {k: list(v) for k, v in replicas.items()}
                candidate.setdefault(expert, []).append(rank)
                loads, _ = allocate_replica_quota(expert_loads, placement, candidate, ep_size)
                score = int(loads.max())
                key = (score, expert, rank, loads)
                if best is None or key[:3] < best[:3]:
                    best = key
        if best is None or best[0] >= int(current.max()):
            break
        _, expert, rank, current = best
        replicas.setdefault(expert, []).append(rank)
        available[rank] -= 1
    return ReplicaPlan(replicas, slots_per_rank, current.tolist())


def assign_global_pairs_to_replica_quota(
    expert_ids: torch.Tensor,
    placement: torch.Tensor,
    replicas: dict[int, list[int]],
    quotas: dict[int, dict[int, int]],
    ep_size: int,
    slots_per_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize deterministic physical `(rank,slot)` for every logical pair.

    The input may have any leading shape. Its flattened order is the global
    rank-major/token-major/top-k order used to consume each expert's quota.
    """
    ids = expert_ids.to(torch.int64).cpu()
    flat = ids.reshape(-1)
    experts = placement.numel()
    physical_slot = _replica_physical_slots(
        placement, replicas, ep_size, slots_per_rank
    )

    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=experts)
    offsets = torch.empty(experts + 1, dtype=torch.int64)
    offsets[0] = 0
    torch.cumsum(counts, 0, out=offsets[1:])
    destinations = torch.empty_like(flat)
    local_ids = torch.empty_like(flat)
    for expert in range(experts):
        positions = order[offsets[expert] : offsets[expert + 1]]
        cursor = 0
        for rank, count in sorted(quotas[expert].items()):
            end = cursor + count
            selected = positions[cursor:end]
            destinations[selected] = rank
            slot = int(physical_slot[expert, rank])
            if slot < 0:
                raise RuntimeError(f"expert {expert} has quota on rank {rank} without a replica")
            local_ids[selected] = slot
            cursor = end
        if cursor != positions.numel():
            raise RuntimeError(f"expert {expert} quota does not conserve pairs")
    return destinations.view_as(ids), local_ids.view_as(ids), physical_slot


def _replica_physical_slots(
    placement: torch.Tensor,
    replicas: dict[int, list[int]],
    ep_size: int,
    slots_per_rank: int,
) -> torch.Tensor:
    experts = placement.numel()
    base_slots = local_expert_indices(placement, ep_size)
    physical_slot = torch.full((experts, ep_size), -1, dtype=torch.int64)
    for expert in range(experts):
        physical_slot[expert, int(placement[expert])] = base_slots[expert]
    by_rank: list[list[int]] = [[] for _ in range(ep_size)]
    for expert, ranks in replicas.items():
        for rank in ranks:
            by_rank[rank].append(expert)
    for rank, replica_experts in enumerate(by_rank):
        if len(replica_experts) > slots_per_rank:
            raise ValueError("replica plan exceeds physical slot budget")
        for offset, expert in enumerate(sorted(replica_experts)):
            physical_slot[expert, rank] = int((placement == rank).sum()) + offset
    return physical_slot


def assign_global_pairs_comm_aware(
    expert_ids: torch.Tensor,
    placement: torch.Tensor,
    replicas: dict[int, list[int]],
    quotas: dict[int, dict[int, int]],
    ep_size: int,
    slots_per_rank: int,
    source_tokens_per_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize fixed compute quotas while minimizing remote pair placement.

    Each expert keeps exactly the same per-rank quota produced by
    :func:`allocate_replica_quota`. Within that constraint, pairs originating
    on a rank that owns a copy are consumed first by that local copy. This
    cannot worsen pair balance and reduces remote pair traffic; the final
    transport metric still uses token/destination deduplication.
    """
    ids = expert_ids.to(torch.int64).cpu()
    if ids.ndim != 2:
        raise ValueError("expert_ids must have [tokens, top_k] shape")
    if ids.shape[0] != ep_size * source_tokens_per_rank:
        raise ValueError("global token rows do not match ep_size * source_tokens_per_rank")
    flat = ids.reshape(-1)
    experts = placement.numel()
    physical_slot = _replica_physical_slots(
        placement, replicas, ep_size, slots_per_rank
    )
    source_token_rank = torch.arange(ids.shape[0], dtype=torch.int64) // source_tokens_per_rank
    source_rank = source_token_rank[:, None].expand_as(ids).reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=experts)
    offsets = torch.empty(experts + 1, dtype=torch.int64)
    offsets[0] = 0
    torch.cumsum(counts, 0, out=offsets[1:])
    destinations = torch.full_like(flat, -1)
    local_ids = torch.full_like(flat, -1)
    for expert in range(experts):
        positions = order[offsets[expert] : offsets[expert + 1]]
        remaining = {int(rank): int(count) for rank, count in quotas[expert].items()}
        for target_rank in sorted(remaining):
            local_positions = positions[source_rank[positions] == target_rank]
            take = min(remaining[target_rank], local_positions.numel())
            selected = local_positions[:take]
            destinations[selected] = target_rank
            local_ids[selected] = int(physical_slot[expert, target_rank])
            remaining[target_rank] -= take
        unassigned = positions[destinations[positions] < 0]
        cursor = 0
        for target_rank in sorted(remaining):
            count = remaining[target_rank]
            selected = unassigned[cursor : cursor + count]
            destinations[selected] = target_rank
            local_ids[selected] = int(physical_slot[expert, target_rank])
            cursor += count
        if cursor != unassigned.numel() or bool((destinations[positions] < 0).any()):
            raise RuntimeError(f"expert {expert} communication-aware quota is incomplete")
    return destinations.view_as(ids), local_ids.view_as(ids), physical_slot
