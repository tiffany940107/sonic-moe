from __future__ import annotations

import torch

from ep4_support.routing import greedy_placement


def placement_rank_loads(
    expert_loads: torch.Tensor, placement: torch.Tensor, ep_size: int
) -> torch.Tensor:
    out = torch.zeros(ep_size, dtype=torch.int64)
    out.scatter_add_(0, placement.to(torch.int64).cpu(), expert_loads.to(torch.int64).cpu())
    return out


def make_static_greedy(expert_loads: torch.Tensor, ep_size: int) -> torch.Tensor:
    return greedy_placement(expert_loads, ep_size)


def capacity_preserving_cycles(
    current: torch.Tensor, target: torch.Tensor
) -> list[list[int]]:
    """Decompose an equal-capacity placement change into rank cycles.

    Applying a complete cycle preserves the number of experts on every rank.
    This lets a controller enforce a migration budget without ever producing
    an invalid intermediate placement.
    """
    current = current.to(torch.int64).cpu()
    target = target.to(torch.int64).cpu()
    if current.shape != target.shape:
        raise ValueError("current and target placements must have the same shape")
    if not torch.equal(torch.bincount(current), torch.bincount(target)):
        raise ValueError("current and target placements must have equal rank capacities")

    remaining: dict[tuple[int, int], list[int]] = {}
    for expert in torch.where(current != target)[0].tolist():
        edge = (int(current[expert]), int(target[expert]))
        remaining.setdefault(edge, []).append(expert)

    cycles: list[list[int]] = []
    while remaining:
        start = min(remaining)[0]
        rank = start
        cycle: list[int] = []
        while True:
            candidates = [edge for edge, values in remaining.items() if edge[0] == rank and values]
            if not candidates:
                raise ValueError("placement delta is not capacity preserving")
            edge = min(candidates)
            expert = remaining[edge].pop()
            if not remaining[edge]:
                del remaining[edge]
            cycle.append(expert)
            rank = edge[1]
            if rank == start:
                break
        cycles.append(cycle)
    return cycles


def limit_migration(
    current: torch.Tensor, target: torch.Tensor, maximum_changed_experts: int
) -> torch.Tensor:
    """Apply whole placement cycles until an expert-migration budget is full."""
    if maximum_changed_experts < 0:
        raise ValueError("maximum_changed_experts must be non-negative")
    if maximum_changed_experts == 0:
        return target.to(torch.int64).cpu().clone()

    current = current.to(torch.int64).cpu()
    candidate = current.clone()
    changed = 0
    for cycle in sorted(
        capacity_preserving_cycles(current, target), key=lambda value: (len(value), value)
    ):
        if changed + len(cycle) > maximum_changed_experts:
            continue
        for expert in cycle:
            candidate[expert] = int(target[expert])
        changed += len(cycle)
    return candidate


__all__ = [
    "capacity_preserving_cycles",
    "limit_migration",
    "make_static_greedy",
    "placement_rank_loads",
]
