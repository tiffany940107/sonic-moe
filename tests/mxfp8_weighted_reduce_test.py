# Copyright (c) 2026, SonicMoE contributors.
"""Numerical and gradient tests for deterministic routed reduction."""

import pytest
import torch
from sonicmoe.functional.mxfp8_weighted_reduce import (
    allocate_mxfp8_weighted_reduce_workspace,
    segmented_weighted_reduce_mxfp8,
)


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


def test_segmented_weighted_reduce_matches_reference_and_gradients():
    _skip_if_no_cuda()
    recv_tokens, top_k, hidden = 29, 8, 256
    torch.manual_seed(903)
    valid = torch.rand(recv_tokens, top_k, device="cuda") > 0.35
    pair_count = int(valid.sum().item())
    scatter = torch.full((recv_tokens, top_k), -1, dtype=torch.int32, device="cuda")
    scatter[valid] = torch.arange(pair_count, dtype=torch.int32, device="cuda")
    recv_token = (
        torch.arange(recv_tokens, dtype=torch.int32, device="cuda")[:, None]
        .expand(-1, top_k)[valid]
        .contiguous()
    )
    pair_actual = torch.randn(
        pair_count, hidden, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    weight_actual = torch.rand(
        pair_count, dtype=torch.float32, device="cuda", requires_grad=True
    )
    pair_ref = pair_actual.detach().clone().requires_grad_()
    weight_ref = weight_actual.detach().clone().requires_grad_()
    workspace = allocate_mxfp8_weighted_reduce_workspace(
        recv_tokens, hidden, device="cuda"
    )
    actual = segmented_weighted_reduce_mxfp8(
        pair_actual,
        weight_actual,
        scatter.reshape(-1),
        recv_token,
        recv_tokens,
        top_k,
        workspace,
    )
    reference = torch.zeros(recv_tokens, hidden, dtype=torch.float32, device="cuda")
    reference.index_add_(0, recv_token, pair_ref.float() * weight_ref.unsqueeze(1))
    reference = reference.to(torch.bfloat16)
    # The fused kernel contracts multiply/add into FMA while ``index_add_``
    # materializes the product first.  Accept one BF16 rounding step while
    # keeping the comparison much tighter than the MXFP8 expert error budget.
    torch.testing.assert_close(actual, reference, rtol=2**-7, atol=2**-12)

    grad = torch.randn_like(actual)
    grads_actual = torch.autograd.grad(actual, (pair_actual, weight_actual), grad)
    grads_ref = torch.autograd.grad(reference, (pair_ref, weight_ref), grad)
    torch.testing.assert_close(grads_actual[0], grads_ref[0], rtol=0, atol=0)
    torch.testing.assert_close(grads_actual[1], grads_ref[1], rtol=1e-6, atol=1e-6)


def test_segmented_weighted_reduce_reuses_output_storage():
    _skip_if_no_cuda()
    workspace = allocate_mxfp8_weighted_reduce_workspace(16, 128, device="cuda")
    pair = torch.randn(16, 128, dtype=torch.bfloat16, device="cuda")
    weights = torch.ones(16, dtype=torch.float32, device="cuda")
    scatter = torch.arange(16, dtype=torch.int32, device="cuda")
    recv_token = torch.arange(16, dtype=torch.int32, device="cuda")
    pointer = workspace.output.data_ptr()
    first = segmented_weighted_reduce_mxfp8(
        pair, weights, scatter, recv_token, 16, 1, workspace
    ).clone()
    second = segmented_weighted_reduce_mxfp8(
        pair, weights, scatter, recv_token, 16, 1, workspace
    )
    assert second.data_ptr() == pointer
    assert torch.equal(first, second)


def test_segmented_weighted_reduce_reuses_storage_across_backward_steps():
    _skip_if_no_cuda()
    workspace = allocate_mxfp8_weighted_reduce_workspace(16, 128, device="cuda")
    weights = torch.ones(16, dtype=torch.float32, device="cuda", requires_grad=True)
    scatter = torch.arange(16, dtype=torch.int32, device="cuda")
    recv_token = torch.arange(16, dtype=torch.int32, device="cuda")
    pointer = workspace.output.data_ptr()
    for _ in range(3):
        pair = torch.randn(
            16, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        output = segmented_weighted_reduce_mxfp8(
            pair, weights, scatter, recv_token, 16, 1, workspace
        )
        assert output.data_ptr() == pointer
        output.float().sum().backward()
        assert pair.grad is not None
        weights.grad = None
