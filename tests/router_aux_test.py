# Copyright (c) 2026, SonicMoE contributors.
"""Tests for the low-launch MXFP8 router auxiliary loss."""

import pytest
import torch
import torch.nn.functional as F

from sonicmoe.functional.router_aux import mxfp8_switch_aux_loss


@pytest.mark.parametrize("tokens,experts,top_k", [(1024, 8, 2), (197, 16, 4)])
def test_mxfp8_switch_aux_loss_matches_pytorch(tokens, experts, top_k):
    if torch.cuda.get_device_properties(0).major != 10:
        pytest.skip("SM100 required")
    torch.manual_seed(tokens + experts)
    logits_ref = torch.randn(
        tokens, experts, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    logits_actual = logits_ref.detach().clone().requires_grad_()
    frequency = torch.randint(
        1,
        max(2, tokens * top_k // experts),
        (experts,),
        dtype=torch.int32,
        device="cuda",
    )

    probabilities = F.softmax(logits_ref, dim=-1, dtype=torch.float32)
    accumulated = probabilities.sum(dim=0)
    expected = (
        experts
        * (
            F.normalize(accumulated, p=1, dim=0)
            * F.normalize(frequency.float(), p=1, dim=0)
        ).sum()
    )
    actual = mxfp8_switch_aux_loss(logits_actual, frequency)
    expected_grad = torch.autograd.grad(expected, logits_ref)[0]
    actual_grad = torch.autograd.grad(actual, logits_actual)[0]

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-2, atol=2e-5)
