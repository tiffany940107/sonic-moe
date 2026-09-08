# Copyright (c) 2026, SonicMoE contributors.
"""Tests for single-stage small-expert route metadata."""

import pytest
import torch

from sonicmoe.functional.triton_kernels import (
    TC_topk_router_metadata_switch_aux_triton_fused,
    TC_topk_router_metadata_triton_fused,
    topk_router_workspace_shape,
)


@pytest.mark.parametrize("tokens,experts,top_k", [(1024, 8, 2), (197, 16, 4)])
def test_fused_router_metadata_matches_stable_sort(tokens, experts, top_k):
    if torch.cuda.get_device_properties(0).major != 10:
        pytest.skip("SM100 required")
    torch.manual_seed(tokens + experts + top_k)
    indices = torch.randint(
        0,
        experts,
        (tokens, top_k),
        dtype=torch.int32,
        device="cuda",
    )
    # Force an empty expert and repeated offsets in the second case.
    indices[indices == experts - 1] = 0
    total = tokens * top_k
    frequency = torch.empty(experts, dtype=torch.int32, device="cuda")
    offsets = torch.empty(experts + 1, dtype=torch.int32, device="cuda")
    gather = torch.empty(total, dtype=torch.int32, device="cuda")
    scatter = torch.empty(total, dtype=torch.int32, device="cuda")
    reverse = torch.empty(total, dtype=torch.int32, device="cuda")
    scratch = torch.empty(
        topk_router_workspace_shape(tokens, experts, top_k),
        dtype=torch.int32,
        device="cuda",
    )

    TC_topk_router_metadata_triton_fused(
        indices,
        experts,
        frequency,
        offsets,
        gather,
        scatter,
        reverse,
        scratch,
    )
    expected_scatter = torch.argsort(indices.flatten().long(), stable=True).int()
    expected_frequency = torch.bincount(
        indices.flatten().long(), minlength=experts
    ).int()
    expected_offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            expected_frequency.cumsum(0),
        )
    )
    expected_reverse = torch.empty_like(reverse)
    expected_reverse[expected_scatter.long()] = torch.arange(
        total, dtype=torch.int32, device="cuda"
    )

    assert torch.equal(frequency, expected_frequency)
    assert torch.equal(offsets, expected_offsets)
    assert torch.equal(scatter, expected_scatter)
    assert torch.equal(reverse, expected_reverse)
    assert torch.equal(gather, expected_scatter // top_k)


@pytest.mark.parametrize("tokens,experts,top_k", [(1024, 8, 2), (197, 16, 4)])
def test_fused_router_metadata_switch_aux_matches_pytorch(tokens, experts, top_k):
    if torch.cuda.get_device_properties(0).major != 10:
        pytest.skip("SM100 required")
    torch.manual_seed(91 + tokens)
    logits = torch.randn(tokens, experts, dtype=torch.bfloat16, device="cuda")
    indices = torch.topk(logits, top_k, dim=-1).indices.int()
    scores = torch.softmax(torch.topk(logits, top_k, dim=-1).values.float(), dim=-1)
    grouped_scores = torch.empty_like(scores)
    total = tokens * top_k
    frequency = torch.empty(experts, dtype=torch.int32, device="cuda")
    offsets = torch.empty(experts + 1, dtype=torch.int32, device="cuda")
    gather = torch.empty(total, dtype=torch.int32, device="cuda")
    scatter = torch.empty(total, dtype=torch.int32, device="cuda")
    reverse = torch.empty(total, dtype=torch.int32, device="cuda")
    scratch = torch.empty(
        topk_router_workspace_shape(tokens, experts, top_k),
        dtype=torch.int32,
        device="cuda",
    )

    actual = TC_topk_router_metadata_switch_aux_triton_fused(
        indices,
        scores,
        grouped_scores,
        logits,
        experts,
        frequency,
        offsets,
        gather,
        scatter,
        reverse,
        scratch,
    )
    assert actual is not None
    expected_frequency = torch.bincount(
        indices.flatten().long(), minlength=experts
    ).int()
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
    expected = (
        experts
        * (
            probabilities.sum(0)
            / probabilities.sum()
            * expected_frequency.float()
            / expected_frequency.sum()
        ).sum()
    )
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    assert torch.equal(frequency, expected_frequency)
    expected_scatter = torch.argsort(indices.flatten().long(), stable=True)
    torch.testing.assert_close(
        grouped_scores.flatten(), scores.flatten()[expected_scatter]
    )
