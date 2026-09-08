# Copyright (c) 2026, SonicMoE contributors.
"""Tests for the small-expert fused MXFP8 router projection."""

import pytest
import torch
import torch.nn.functional as F

from sonicmoe.functional.router_aux import mxfp8_switch_aux_loss
from sonicmoe.functional.router_mxfp8 import (
    mxfp8_router_attach_switch_aux,
    mxfp8_router_linear_topk,
    mxfp8_router_linear_topk_raw,
)
from sonicmoe.functional.triton_kernels import (
    TC_topk_router_metadata_switch_aux_triton_fused,
    topk_router_workspace_shape,
)


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_mxfp8_router_linear_topk_matches_pytorch(top_k):
    if torch.cuda.get_device_properties(0).major != 10:
        pytest.skip("SM100 required")
    torch.manual_seed(41 + top_k)
    x_ref = torch.randn(
        129, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    weight_ref = torch.randn(
        8, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    x_actual = x_ref.detach().clone().requires_grad_()
    weight_actual = weight_ref.detach().clone().requires_grad_()

    logits_ref = F.linear(x_ref, weight_ref)
    values_ref, indices_ref = torch.topk(logits_ref, top_k, dim=-1)
    scores_ref = F.softmax(values_ref.float(), dim=-1)
    logits_actual, scores_actual, indices_actual = mxfp8_router_linear_topk(
        x_actual, weight_actual, top_k
    )

    torch.testing.assert_close(logits_actual, logits_ref, rtol=2e-2, atol=0.125)
    assert torch.equal(
        torch.sort(indices_actual.long(), dim=-1).values,
        torch.sort(indices_ref, dim=-1).values,
    )
    torch.testing.assert_close(scores_actual, scores_ref, rtol=2e-3, atol=2e-3)

    logits_grad = torch.randn_like(logits_ref)
    # Weight selected scores by expert id so equal-logit tie ordering does not
    # change the mathematical objective.
    scores_grad_ref = indices_ref.float() + 1.0
    scores_grad_actual = indices_actual.float() + 1.0
    grads_ref = torch.autograd.grad(
        (logits_ref * logits_grad).sum() + (scores_ref * scores_grad_ref).sum(),
        (x_ref, weight_ref),
    )
    grads_actual = torch.autograd.grad(
        (logits_actual * logits_grad).sum()
        + (scores_actual * scores_grad_actual).sum(),
        (x_actual, weight_actual),
    )
    for actual, expected in zip(grads_actual, grads_ref):
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=0.5)


def test_mxfp8_router_combines_score_and_switch_loss_backward():
    if torch.cuda.get_device_properties(0).major != 10:
        pytest.skip("SM100 required")
    torch.manual_seed(73)
    x_ref = torch.randn(
        197, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    weight_ref = torch.randn(
        8, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    x_actual = x_ref.detach().clone().requires_grad_()
    weight_actual = weight_ref.detach().clone().requires_grad_()

    logits_ref, scores_ref, indices_ref = mxfp8_router_linear_topk(x_ref, weight_ref, 2)
    frequency_ref = torch.bincount(
        indices_ref.flatten().long(), minlength=weight_ref.shape[0]
    ).to(torch.int32)
    aux_ref = mxfp8_switch_aux_loss(logits_ref, frequency_ref)

    logits_raw, scores_raw, indices_actual = mxfp8_router_linear_topk_raw(
        x_actual, weight_actual, 2
    )
    total = indices_actual.numel()
    frequency_actual = torch.empty(8, dtype=torch.int32, device="cuda")
    offsets = torch.empty(9, dtype=torch.int32, device="cuda")
    gather = torch.empty(total, dtype=torch.int32, device="cuda")
    scatter = torch.empty(total, dtype=torch.int32, device="cuda")
    reverse = torch.empty(total, dtype=torch.int32, device="cuda")
    grouped_scores = torch.empty_like(scores_raw)
    scratch = torch.empty(
        topk_router_workspace_shape(197, 8, 2),
        dtype=torch.int32,
        device="cuda",
    )
    precomputed_aux = TC_topk_router_metadata_switch_aux_triton_fused(
        indices_actual,
        scores_raw,
        grouped_scores,
        logits_raw,
        8,
        frequency_actual,
        offsets,
        gather,
        scatter,
        reverse,
        scratch,
    )
    assert precomputed_aux is not None
    logits_actual, scores_actual, aux_actual = mxfp8_router_attach_switch_aux(
        x_actual,
        weight_actual,
        logits_raw,
        scores_raw,
        indices_actual,
        frequency_actual,
        precomputed_aux,
        grouped_scores,
        reverse,
    )

    torch.testing.assert_close(aux_actual, aux_ref, rtol=2e-3, atol=2e-3)
    logits_grad = torch.randn_like(logits_ref)
    scores_grad_ref = indices_ref.float() + 1.0
    scores_grad_actual = (
        (indices_actual.float() + 1.0).flatten()[scatter.long()].view_as(scores_actual)
    )
    objective_ref = (
        (logits_ref * logits_grad).sum()
        + (scores_ref * scores_grad_ref).sum()
        + 0.03 * aux_ref
    )
    objective_actual = (
        (logits_actual * logits_grad).sum()
        + (scores_actual * scores_grad_actual).sum()
        + 0.03 * aux_actual
    )
    grads_ref = torch.autograd.grad(objective_ref, (x_ref, weight_ref))
    grads_actual = torch.autograd.grad(objective_actual, (x_actual, weight_actual))
    for actual, expected in zip(grads_actual, grads_ref):
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=0.5)
