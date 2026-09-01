"""EXPERIMENTAL hot-expert replica planning for the Sonic EP4 prototype.

This API is intentionally separate from stable static EPLB.  It may change,
is disabled by default, and is not a production consistency protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from quack.blockscaled import BlockScaledOperand


@dataclass(frozen=True)
class ExperimentalReplicaPlan:
    replicas: dict[int, list[int]]
    slots_per_rank: int
    estimated_rank_loads: list[int]

    @property
    def count(self) -> int:
        return sum(len(ranks) for ranks in self.replicas.values())


def allocate_quota(
    expert_loads: torch.Tensor,
    placement: torch.Tensor,
    replicas: dict[int, list[int]],
    ep_size: int,
) -> tuple[torch.Tensor, dict[int, dict[int, int]]]:
    """Water-fill each expert across its owner and experimental replicas."""
    rank_loads = [0] * ep_size
    quotas: dict[int, dict[int, int]] = {}
    order = sorted(
        range(expert_loads.numel()), key=lambda expert: (-int(expert_loads[expert]), expert)
    )
    for expert in order:
        remaining = int(expert_loads[expert])
        candidates = sorted({int(placement[expert]), *replicas.get(expert, [])})
        quota = {rank: 0 for rank in candidates}
        while remaining:
            minimum = min(rank_loads[rank] for rank in candidates)
            active = [rank for rank in candidates if rank_loads[rank] == minimum]
            higher = [rank_loads[rank] for rank in candidates if rank_loads[rank] > minimum]
            increment = (
                remaining
                if not higher
                else min(remaining, (min(higher) - minimum) * len(active))
            )
            each, extra = divmod(increment, len(active))
            for index, rank in enumerate(active):
                value = each + int(index < extra)
                quota[rank] += value
                rank_loads[rank] += value
            remaining -= increment
        quotas[expert] = {rank: count for rank, count in quota.items() if count}
    return torch.tensor(rank_loads, dtype=torch.int64), quotas


def plan_replicas(
    expert_loads: torch.Tensor,
    placement: torch.Tensor,
    ep_size: int,
    slots_per_rank: int,
    *,
    max_copies_per_expert: int = 4,
    minimum_hot_expert_ratio: float = 4.0,
) -> ExperimentalReplicaPlan:
    """Greedily add bounded copies only for experts above a hotness floor."""
    if slots_per_rank not in (1, 2, 4, 8, 16):
        raise ValueError("slots_per_rank must be one of 1, 2, 4, 8, 16")
    if not 2 <= max_copies_per_expert <= ep_size:
        raise ValueError("max_copies_per_expert must be in [2, ep_size]")
    if minimum_hot_expert_ratio < 1.0:
        raise ValueError("minimum_hot_expert_ratio must be at least one")

    replicas: dict[int, list[int]] = {}
    available = [slots_per_rank] * ep_size
    current, _ = allocate_quota(expert_loads, placement, replicas, ep_size)
    mean = float(expert_loads.to(torch.float64).mean().clamp_min(1.0))
    hot = [
        expert
        for expert in sorted(
            range(expert_loads.numel()),
            key=lambda item: (-int(expert_loads[item]), item),
        )
        if float(expert_loads[expert]) >= mean * minimum_hot_expert_ratio
    ][: max(32, slots_per_rank * ep_size * 4)]

    for _ in range(slots_per_rank * ep_size):
        best: tuple[int, int, int, torch.Tensor] | None = None
        for expert in hot:
            existing = {int(placement[expert]), *replicas.get(expert, [])}
            if len(existing) >= max_copies_per_expert:
                continue
            for rank in range(ep_size):
                if rank in existing or available[rank] == 0:
                    continue
                candidate = {key: list(value) for key, value in replicas.items()}
                candidate.setdefault(expert, []).append(rank)
                loads, _ = allocate_quota(expert_loads, placement, candidate, ep_size)
                key = (int(loads.max()), expert, rank, loads)
                if best is None or key[:3] < best[:3]:
                    best = key
        if best is None or best[0] >= int(current.max()):
            break
        _, expert, rank, current = best
        replicas.setdefault(expert, []).append(rank)
        available[rank] -= 1
    return ExperimentalReplicaPlan(replicas, slots_per_rank, current.tolist())


def physical_slots(
    placement: torch.Tensor,
    replicas: dict[int, list[int]],
    ep_size: int,
    slots_per_rank: int,
) -> torch.Tensor:
    """Map logical expert/rank pairs to owner-bank or shadow-bank slots."""
    experts = placement.numel()
    result = torch.full((experts, ep_size), -1, dtype=torch.int64)
    for rank in range(ep_size):
        owned = torch.where(placement == rank)[0].tolist()
        for slot, expert in enumerate(owned):
            result[expert, rank] = slot
        replicated = sorted(expert for expert, ranks in replicas.items() if rank in ranks)
        if len(replicated) > slots_per_rank:
            raise ValueError("replica slot budget exceeded")
        for offset, expert in enumerate(replicated):
            result[expert, rank] = len(owned) + offset
    return result


def materialize_comm_aware(
    expert_ids: torch.Tensor,
    placement: torch.Tensor,
    replicas: dict[int, list[int]],
    quotas: dict[int, dict[int, int]],
    ep_size: int,
    slots_per_rank: int,
    source_tokens_per_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign every route to one physical copy, preferring source-local work."""
    ids = expert_ids.to(torch.int64).cpu()
    if ids.ndim != 2 or ids.shape[0] != ep_size * source_tokens_per_rank:
        raise ValueError("expert_ids must be global rank-major [EP*m, top_k]")
    flat = ids.reshape(-1)
    source = (
        torch.arange(ids.shape[0], dtype=torch.int64) // source_tokens_per_rank
    )[:, None].expand_as(ids).reshape(-1)
    slots = physical_slots(placement, replicas, ep_size, slots_per_rank)
    destination = torch.full_like(flat, -1)
    local_slot = torch.full_like(flat, -1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=placement.numel())
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))

    for expert in range(placement.numel()):
        positions = order[offsets[expert] : offsets[expert + 1]]
        remaining = dict(quotas[expert])
        for rank in sorted(remaining):
            local = positions[(source[positions] == rank) & (destination[positions] < 0)]
            take = min(remaining[rank], local.numel())
            selected = local[:take]
            destination[selected] = rank
            local_slot[selected] = slots[expert, rank]
            remaining[rank] -= take
        unassigned = positions[destination[positions] < 0]
        cursor = 0
        for rank in sorted(remaining):
            selected = unassigned[cursor : cursor + remaining[rank]]
            destination[selected] = rank
            local_slot[selected] = slots[expert, rank]
            cursor += remaining[rank]
        if cursor != unassigned.numel() or bool((destination[positions] < 0).any()):
            raise RuntimeError(f"expert {expert} quota conservation failed")
    return destination.view_as(ids), local_slot.view_as(ids), slots


def copy_replica_bank(
    operand: BlockScaledOperand,
    placement: torch.Tensor,
    replicas: dict[int, list[int]],
    slot_map: torch.Tensor,
    rank: int,
    pair_groups: dict[tuple[int, int], dist.ProcessGroup],
) -> int:
    """EXPERIMENTAL: copy owner values/scales into preallocated shadow slots.

    Every rank must call this function in the same order.  Two-rank process
    groups must be created collectively before the timed preload region.
    """
    base_slot = torch.empty(placement.numel(), dtype=torch.int64)
    for owner in range(int(placement.max()) + 1):
        for slot, expert in enumerate(torch.where(placement == owner)[0].tolist()):
            base_slot[expert] = slot

    transfers = []
    for expert, targets in sorted(replicas.items()):
        owner = int(placement[expert])
        source_slot = int(base_slot[expert])
        for target in sorted(targets):
            target_slot = int(slot_map[expert, target])
            members = tuple(sorted((owner, target)))
            transfers.append(
                (members, expert, owner, target, source_slot, target_slot)
            )

    copied = 0
    for members in sorted(pair_groups):
        group = pair_groups[members]
        matching = (transfer for transfer in transfers if transfer[0] == members)
        for _, _, owner, target, source_slot, target_slot in matching:
            for storage in (operand.qdata, operand.scale):
                if rank == owner:
                    buffer = storage[source_slot]
                elif rank == target:
                    buffer = storage[target_slot]
                else:
                    continue
                dist.broadcast(buffer.view(torch.uint8), src=owner, group=group)
                if rank == owner:
                    copied += storage[0].numel() * storage[0].element_size()
    return copied


__all__ = [
    "ExperimentalReplicaPlan",
    "allocate_quota",
    "copy_replica_bank",
    "materialize_comm_aware",
    "physical_slots",
    "plan_replicas",
]
