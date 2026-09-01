"""Synthetic, quantile-matched workloads for the 512-expert customer case.

These traces model the supplied aggregate statistics.  They are not customer
route captures and must not be presented as such.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable

import torch

EP_SIZE = 4
EXPERTS = 512
TOP_K = 32
BASE_TOKENS = 16_384
TOKEN_WORKLOADS = (8_192, 16_384, 32_768, 46_080)
RANK_RATIOS = (1.043, 1.106, 1.227, 1.698)


def linear_quantile(values: torch.Tensor, q: float) -> float:
    ordered = values.to(torch.float64).sort().values
    position = q * (ordered.numel() - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def largest_remainder(total: int, weights: Iterable[float]) -> list[int]:
    values = [float(value) for value in weights]
    if not values or sum(values) <= 0:
        raise ValueError("weights must have a positive sum")
    raw = [total * value / sum(values) for value in values]
    result = [math.floor(value) for value in raw]
    remainder = total - sum(result)
    for index in sorted(
        range(len(values)), key=lambda item: (-(raw[item] - result[item]), item)
    )[:remainder]:
        result[index] += 1
    return result


def base_segment_profile() -> torch.Tensor:
    """Build the exact 16K p50=517, p90=2528, max=16384 profile."""
    counts: list[int] = []
    counts.extend(round(24 + (517 - 24) * index / 255) for index in range(256))
    counts.extend(
        round(517 + (2528 - 517) * (index - 256) / (460 - 256))
        for index in range(256, 461)
    )
    counts.extend(
        round(2529 + (2533 - 2529) * (index - 461) / (510 - 461))
        for index in range(461, 511)
    )
    counts.append(BASE_TOKENS)
    counts[459] = 2528

    delta = sum(counts) - BASE_TOKENS * TOP_K
    for index in range(255):
        if delta == 0:
            break
        lower = counts[index - 1] if index else 0
        take = min(delta, max(0, counts[index] - lower))
        counts[index] -= take
        delta -= take
    if delta:
        raise RuntimeError(f"could not normalize base profile, residual={delta}")

    profile = torch.tensor(counts, dtype=torch.int64)
    if int(profile.sum()) != BASE_TOKENS * TOP_K:
        raise AssertionError("base profile pair conservation failed")
    if linear_quantile(profile, 0.5) != 517 or linear_quantile(profile, 0.9) != 2528:
        raise AssertionError("base profile quantile anchors failed")
    if int(profile.max()) != BASE_TOKENS:
        raise AssertionError("base profile max anchor failed")
    return profile


def scaled_segment_profile(tokens: int) -> torch.Tensor:
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    base = base_segment_profile().to(torch.float64)
    raw = base * (tokens / BASE_TOKENS)
    counts = torch.floor(raw).to(torch.int64)
    counts[-1] = tokens
    residual = tokens * TOP_K - int(counts.sum())
    fractions = raw - torch.floor(raw)
    order = sorted(range(EXPERTS - 1), key=lambda i: (-float(fractions[i]), i))
    direction = 1 if residual >= 0 else -1
    for index in range(abs(residual)):
        counts[order[index % len(order)]] += direction
    if int(counts.sum()) != tokens * TOP_K or int(counts.max()) > tokens:
        raise AssertionError("scaled segment profile is invalid")
    return counts


def realize_histogram(counts: torch.Tensor, tokens: int, seed: int) -> torch.Tensor:
    """Realize expert degrees as a simple token/expert bipartite graph."""
    counts = counts.to(torch.int64).cpu()
    if counts.numel() != EXPERTS or int(counts.sum()) != tokens * TOP_K:
        raise ValueError("histogram shape or sum is invalid")
    if int(counts.max()) > tokens or int(counts.min()) < 0:
        raise ValueError("an expert degree must be in [0, tokens]")

    heap = [(-int(count), expert) for expert, count in enumerate(counts) if count]
    heapq.heapify(heap)
    rows = torch.empty((tokens, TOP_K), dtype=torch.int64)
    for token in range(tokens):
        selected: list[tuple[int, int]] = []
        for slot in range(TOP_K):
            if not heap:
                raise RuntimeError(f"degree sequence exhausted at token={token}, slot={slot}")
            negative, expert = heapq.heappop(heap)
            if negative >= 0:
                raise RuntimeError("non-positive degree reached")
            rows[token, slot] = expert
            selected.append((negative + 1, expert))
        for item in selected:
            if item[0] < 0:
                heapq.heappush(heap, item)
    if heap:
        raise RuntimeError("degree sequence was not fully consumed")

    generator = torch.Generator().manual_seed(seed)
    rows = rows[torch.randperm(tokens, generator=generator)]
    slot_order = torch.rand((tokens, TOP_K), generator=generator).argsort(dim=1)
    rows = rows.gather(1, slot_order)
    ordered = rows.sort(dim=1).values
    if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
        raise AssertionError("realization produced a duplicate within a token")
    if not torch.equal(torch.bincount(rows.reshape(-1), minlength=EXPERTS), counts):
        raise AssertionError("realized histogram differs from target")
    return rows


def balanced_counts(tokens: int) -> torch.Tensor:
    return torch.tensor(
        largest_remainder(tokens * TOP_K, [1.0] * EXPERTS), dtype=torch.int64
    )


def rank_skew_counts(tokens: int, ratio: float) -> torch.Tensor:
    if not 1.0 <= ratio <= EP_SIZE:
        raise ValueError("rank max/mean ratio must be in [1, ep_size]")
    total = tokens * TOP_K
    hot = round(ratio * total / EP_SIZE)
    owner_totals = [
        hot, *largest_remainder(total - hot, [1.0] * (EP_SIZE - 1))
    ]
    counts: list[int] = []
    for owner_total in owner_totals:
        counts.extend(largest_remainder(owner_total, [1.0] * (EXPERTS // EP_SIZE)))
    return torch.tensor(counts, dtype=torch.int64)


def assign_segment_to_rank_targets(
    sorted_counts: torch.Tensor, ratio: float
) -> torch.Tensor:
    total = int(sorted_counts.sum())
    hot = round(ratio * total / EP_SIZE)
    targets = [hot, *largest_remainder(total - hot, [1.0] * (EP_SIZE - 1))]
    bins: list[list[int]] = [[] for _ in range(EP_SIZE)]
    loads = [0] * EP_SIZE
    for count in sorted((int(value) for value in sorted_counts), reverse=True):
        choices = [
            rank for rank in range(EP_SIZE) if len(bins[rank]) < EXPERTS // EP_SIZE
        ]
        rank = max(
            choices, key=lambda item: (targets[item] - loads[item], -loads[item], -item)
        )
        bins[rank].append(count)
        loads[rank] += count
    output = torch.empty(EXPERTS, dtype=torch.int64)
    for rank, values in enumerate(bins):
        begin = rank * (EXPERTS // EP_SIZE)
        output[begin : begin + EXPERTS // EP_SIZE] = torch.tensor(values)
    return output


def make_trace(
    tokens: int, scenario: str, seed: int
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Create one global rank-major trace and its auditable metadata."""
    source_rows: list[torch.Tensor] = []
    source_counts: list[torch.Tensor] = []
    segment = scaled_segment_profile(tokens)
    for source_rank in range(EP_SIZE):
        if scenario == "balanced":
            counts = balanced_counts(tokens)
        elif scenario == "segment_aligned":
            counts = segment.clone()
        elif scenario == "segment_permuted":
            generator = torch.Generator().manual_seed(seed + source_rank * 1009)
            permutation = torch.randperm(EXPERTS, generator=generator)
            counts = torch.empty_like(segment)
            counts[permutation] = segment
        elif scenario.startswith("rank_r"):
            counts = rank_skew_counts(tokens, float(scenario.removeprefix("rank_r")))
        elif scenario == "joint_aligned_max":
            counts = assign_segment_to_rank_targets(segment, 1.698)
        else:
            raise ValueError(f"unknown workload scenario: {scenario}")
        source_counts.append(counts)
        source_rows.append(realize_histogram(counts, tokens, seed + source_rank * 7919))

    expert_ids = torch.cat(source_rows, dim=0)
    generator = torch.Generator().manual_seed(seed + 1_000_003)
    weights = torch.softmax(torch.randn(expert_ids.shape, generator=generator), dim=1).float()
    placement = torch.arange(EXPERTS, dtype=torch.int64) // (EXPERTS // EP_SIZE)
    global_counts = torch.stack(source_counts).sum(dim=0)
    owner_counts = torch.bincount(
        placement, weights=global_counts.to(torch.float64), minlength=EP_SIZE
    ).to(torch.int64)
    metadata = {
        "schema_version": 1,
        "synthetic_quantile_matched": scenario != "balanced",
        "synthetic_scaled_from_16k": tokens != BASE_TOKENS and scenario != "balanced",
        "scenario": scenario,
        "exact_m": tokens,
        "ep_size": EP_SIZE,
        "experts": EXPERTS,
        "top_k": TOP_K,
        "segment_scope": "per_source_rank",
        "seed": seed,
        "source_expert_counts": [value.tolist() for value in source_counts],
        "post_ep_global_expert_counts": global_counts.tolist(),
        "post_ep_owner_pair_counts": owner_counts.tolist(),
        "source_segment_metrics": [
            {
                "p50": linear_quantile(value, 0.5),
                "p90": linear_quantile(value, 0.9),
                "max": int(value.max()),
            }
            for value in source_counts
        ],
        "post_ep_segment_metrics": {
            "p50": linear_quantile(global_counts, 0.5),
            "p90": linear_quantile(global_counts, 0.9),
            "max": int(global_counts.max()),
        },
        "actual_rank_max_over_mean": float(owner_counts.max())
        / float(owner_counts.to(torch.float64).mean()),
        "target_rank_max_over_mean": (
            float(scenario.removeprefix("rank_r"))
            if scenario.startswith("rank_r")
            else (1.698 if scenario == "joint_aligned_max" else None)
        ),
        "target_16k_segment": (
            {"p50": 517, "p90": 2528, "max": 16384}
            if scenario.startswith("segment_") or scenario == "joint_aligned_max"
            else None
        ),
    }
    return expert_ids, weights, metadata


def validate_trace(ids: torch.Tensor, weights: torch.Tensor, metadata: dict) -> None:
    tokens = int(metadata["exact_m"])
    expected = (EP_SIZE * tokens, TOP_K)
    if tuple(ids.shape) != expected or tuple(weights.shape) != expected:
        raise AssertionError(f"trace shapes differ from {expected}")
    if int(ids.min()) < 0 or int(ids.max()) >= EXPERTS:
        raise AssertionError("expert ID out of range")
    ordered = ids.sort(dim=1).values
    if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
        raise AssertionError("duplicate expert within token")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise AssertionError("invalid top-k weights")
    if not torch.allclose(
        weights.sum(dim=1), torch.ones(ids.shape[0]), atol=2e-6, rtol=0
    ):
        raise AssertionError("top-k weights do not sum to one")
    if float(weights.var(dim=1).mean()) <= 0:
        raise AssertionError("top-k weights unexpectedly uniform")


def scenarios_for(tokens: int) -> list[str]:
    if tokens == BASE_TOKENS:
        return [
            "balanced",
            "segment_aligned",
            "segment_permuted",
            *(f"rank_r{ratio:g}" for ratio in RANK_RATIOS),
            "joint_aligned_max",
        ]
    return ["balanced", "segment_aligned", "rank_r1.227"]


__all__ = [
    "BASE_TOKENS",
    "EP_SIZE",
    "EXPERTS",
    "RANK_RATIOS",
    "TOKEN_WORKLOADS",
    "TOP_K",
    "base_segment_profile",
    "make_trace",
    "scaled_segment_profile",
    "scenarios_for",
    "validate_trace",
]
