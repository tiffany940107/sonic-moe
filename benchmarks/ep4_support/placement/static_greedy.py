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
