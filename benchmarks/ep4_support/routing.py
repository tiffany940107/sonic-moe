from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Iterable, Literal

import torch


@dataclass
class RoutingBatch:
    expert_ids: torch.Tensor
    weights: torch.Tensor
    placement: torch.Tensor
    scenario: str
    seed: int

    def validate(self, experts: int, top_k: int) -> None:
        if self.expert_ids.ndim != 2 or self.expert_ids.shape[1] != top_k:
            raise ValueError("invalid top-k shape")
        if self.weights.shape != self.expert_ids.shape:
            raise ValueError("weights and expert IDs differ in shape")
        if self.expert_ids.min().item() < 0 or self.expert_ids.max().item() >= experts:
            raise ValueError("expert ID out of range")
        sorted_ids = self.expert_ids.sort(dim=1).values
        if bool((sorted_ids[:, 1:] == sorted_ids[:, :-1]).any()):
            raise ValueError("duplicate expert within a token")
        if not torch.allclose(
            self.weights.sum(dim=1),
            torch.ones(self.weights.shape[0], device=self.weights.device),
            atol=2e-6,
            rtol=0,
        ):
            raise ValueError("router weights are not normalized")


def contiguous_placement(experts: int, ep_size: int) -> torch.Tensor:
    if experts % ep_size:
        raise ValueError("experts must divide ep_size")
    return torch.arange(experts, dtype=torch.int64) // (experts // ep_size)


def interleaved_placement(experts: int, ep_size: int) -> torch.Tensor:
    return torch.arange(experts, dtype=torch.int64) % ep_size


def greedy_placement(expert_loads: torch.Tensor, ep_size: int) -> torch.Tensor:
    """LPT placement with an exact equal expert-count capacity per rank."""
    loads = expert_loads.to(torch.float64).cpu()
    experts = loads.numel()
    if experts % ep_size:
        raise ValueError("experts must divide ep_size")
    capacity = experts // ep_size
    rank_load = [0.0] * ep_size
    rank_count = [0] * ep_size
    out = torch.empty(experts, dtype=torch.int64)
    order = sorted(range(experts), key=lambda e: (-float(loads[e]), e))
    for expert in order:
        candidates = [r for r in range(ep_size) if rank_count[r] < capacity]
        rank = min(candidates, key=lambda r: (rank_load[r], rank_count[r], r))
        out[expert] = rank
        rank_load[rank] += float(loads[expert])
        rank_count[rank] += 1
    return out


def _largest_remainder(total: int, proportions: Iterable[float]) -> list[int]:
    p = [float(x) for x in proportions]
    norm = sum(p)
    raw = [total * x / norm for x in p]
    out = [int(math.floor(x)) for x in raw]
    for i in sorted(range(len(p)), key=lambda j: (-(raw[j] - out[j]), j))[
        : total - sum(out)
    ]:
        out[i] += 1
    return out


def rank_pair_targets(total_pairs: int, ep_size: int, max_over_mean: float) -> list[int]:
    if not 1.0 <= max_over_mean <= ep_size:
        raise ValueError("max_over_mean must be in [1, ep_size]")
    mean = total_pairs / ep_size
    hot = int(round(max_over_mean * mean))
    rest = total_pairs - hot
    tails = _largest_remainder(rest, [1.0] * (ep_size - 1))
    return [hot, *tails]


def generate_rank_skew_routing(
    tokens: int,
    top_k: int,
    experts: int,
    ep_size: int,
    max_over_mean: float,
    *,
    seed: int = 42,
    placement: torch.Tensor | None = None,
) -> RoutingBatch:
    """Generate exact rank totals while keeping 24 distinct experts per token."""
    if experts % ep_size or top_k > experts // ep_size:
        raise ValueError("generator requires top_k <= experts per rank")
    placement = contiguous_placement(experts, ep_size) if placement is None else placement.cpu()
    members = [torch.where(placement == r)[0] for r in range(ep_size)]
    if any(len(m) < top_k for m in members):
        raise ValueError("each rank needs at least top_k experts")
    total_pairs = tokens * top_k
    targets = rank_pair_targets(total_pairs, ep_size, max_over_mean)
    rank_labels = torch.repeat_interleave(
        torch.arange(ep_size, dtype=torch.int64), torch.tensor(targets)
    )
    gen = torch.Generator().manual_seed(seed)
    rank_labels = rank_labels[torch.randperm(total_pairs, generator=gen)].view(tokens, top_k)
    ids = torch.empty_like(rank_labels)
    # Per-token offsets make repeats of a destination rank map to distinct experts.
    for token in range(tokens):
        used: set[int] = set()
        per_rank_seen = [0] * ep_size
        for slot in range(top_k):
            rank = int(rank_labels[token, slot])
            pool = members[rank]
            base = (token * 37 + rank * 17) % len(pool)
            candidate = int(pool[(base + per_rank_seen[rank]) % len(pool)])
            while candidate in used:
                per_rank_seen[rank] += 1
                candidate = int(pool[(base + per_rank_seen[rank]) % len(pool)])
            ids[token, slot] = candidate
            used.add(candidate)
            per_rank_seen[rank] += 1
    logits = torch.randn((tokens, top_k), generator=gen)
    weights = torch.softmax(logits, dim=1)
    batch = RoutingBatch(ids, weights, placement, f"rank_skew_{max_over_mean:g}", seed)
    batch.validate(experts, top_k)
    return batch


def generate_weighted_routing(
    tokens: int,
    top_k: int,
    experts: int,
    distribution: Literal["uniform", "zipf", "hotset", "empty"],
    *,
    seed: int = 42,
    zipf_alpha: float = 1.2,
    hot_experts: int = 24,
    hot_mass: float = 0.8,
    active_experts: int | None = None,
    placement: torch.Tensor | None = None,
    ep_size: int = 4,
) -> RoutingBatch:
    gen = torch.Generator().manual_seed(seed)
    if distribution == "uniform":
        probabilities = torch.ones(experts)
    elif distribution == "zipf":
        probabilities = torch.arange(1, experts + 1, dtype=torch.float64).pow(-zipf_alpha)
    elif distribution == "hotset":
        if hot_experts < top_k:
            raise ValueError("hot_experts must be >= top_k")
        probabilities = torch.full((experts,), (1.0 - hot_mass) / (experts - hot_experts))
        probabilities[:hot_experts] = hot_mass / hot_experts
    elif distribution == "empty":
        active = active_experts or max(top_k, experts // 4)
        probabilities = torch.zeros(experts)
        probabilities[:active] = 1.0
    else:
        raise ValueError(distribution)
    ids = torch.multinomial(
        probabilities.expand(tokens, -1), top_k, replacement=False, generator=gen
    ).to(torch.int64)
    weights = torch.softmax(torch.randn((tokens, top_k), generator=gen), dim=1)
    placement = contiguous_placement(experts, ep_size) if placement is None else placement.cpu()
    batch = RoutingBatch(ids, weights, placement, distribution, seed)
    batch.validate(experts, top_k)
    return batch


def load_global_routing_trace(
    path: str | Path,
    *,
    source_tokens_per_rank: int,
    top_k: int,
    experts: int,
    ep_size: int,
    rank: int,
) -> tuple[RoutingBatch, str]:
    """Load a global rank-major ``.pt`` routing trace and select one rank.

    The file must contain a dictionary with ``expert_ids`` shaped
    ``[ep_size * source_tokens_per_rank, top_k]``. ``weights`` is optional;
    when absent, equal top-k weights are used. The SHA-256 is returned for
    benchmark provenance.
    """
    trace_path = Path(path)
    digest = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    payload = torch.load(trace_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "expert_ids" not in payload:
        raise ValueError("routing trace must be a dict containing expert_ids")
    global_ids = torch.as_tensor(payload["expert_ids"], dtype=torch.int64)
    expected = (ep_size * source_tokens_per_rank, top_k)
    if tuple(global_ids.shape) != expected:
        raise ValueError(f"expert_ids shape {tuple(global_ids.shape)} != expected {expected}")
    if "weights" in payload:
        global_weights = torch.as_tensor(payload["weights"], dtype=torch.float32)
        if tuple(global_weights.shape) != expected:
            raise ValueError(f"weights shape {tuple(global_weights.shape)} != expected {expected}")
    else:
        global_weights = torch.full(expected, 1.0 / top_k, dtype=torch.float32)
    lo = rank * source_tokens_per_rank
    hi = (rank + 1) * source_tokens_per_rank
    placement = contiguous_placement(experts, ep_size)
    batch = RoutingBatch(
        global_ids[lo:hi].contiguous(),
        global_weights[lo:hi].contiguous(),
        placement,
        f"trace:{trace_path.name}",
        -1,
    )
    batch.validate(experts, top_k)
    return batch, digest


def routing_metrics(batch: RoutingBatch, experts: int, ep_size: int) -> dict:
    flat = batch.expert_ids.reshape(-1).cpu()
    expert = torch.bincount(flat, minlength=experts).to(torch.float64)
    rank_ids = batch.placement[flat]
    rank = torch.bincount(rank_ids, minlength=ep_size).to(torch.float64)
    mean_rank = float(rank.mean())
    mean_expert = float(expert.mean())
    token_destinations = batch.placement[batch.expert_ids].unique(dim=1) if False else None
    # `unique(dim=1)` is not row-wise; count destination fanout explicitly.
    fanout = torch.stack(
        [(batch.placement[batch.expert_ids] == r).any(dim=1) for r in range(ep_size)], dim=1
    ).sum(dim=1).to(torch.float64)
    return {
        "scenario": batch.scenario,
        "total_pairs": int(flat.numel()),
        "expert_counts": expert.to(torch.int64).tolist(),
        "rank_pair_counts": rank.to(torch.int64).tolist(),
        "rank_max_over_mean": float(rank.max()) / mean_rank,
        "rank_cv": float(rank.std(unbiased=False)) / mean_rank,
        "expert_max_over_mean": float(expert.max()) / mean_expert,
        "empty_experts": int((expert == 0).sum()),
        "destination_ranks_per_token_mean": float(fanout.mean()),
    }
