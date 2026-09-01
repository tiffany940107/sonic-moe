# ********************************************************************************
# Copyright (c) 2026 SonicMoE contributors
# ********************************************************************************
"""Correctness and allocation gates for the fused local weighted reduce."""

import pytest
import torch

from sonicmoe.functional.mxfp8_weighted_reduce import (
    allocate_mxfp8_weighted_reduce_workspace,
    segmented_weighted_reduce_mxfp8,
    weighted_reduce_mxfp8,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="weighted reduce requires CUDA"
)


@pytest.mark.parametrize("recv_tokens,num_pairs,hidden", [(19, 71, 256), (7, 0, 128)])
def test_weighted_reduce_matches_fp32_index_add(recv_tokens, num_pairs, hidden):
    torch.manual_seed(902)
    pair_output = torch.randn(
        num_pairs, hidden, dtype=torch.bfloat16, device="cuda"
    )
    token = torch.randint(
        recv_tokens, (num_pairs,), dtype=torch.int32, device="cuda"
    )
    weights = torch.rand(num_pairs, dtype=torch.float32, device="cuda")
    workspace = allocate_mxfp8_weighted_reduce_workspace(
        recv_tokens, hidden, device="cuda"
    )
    actual = weighted_reduce_mxfp8(
        pair_output, token, weights, recv_tokens, workspace, out_dtype=torch.float32
    )
    reference = torch.zeros(
        recv_tokens, hidden, dtype=torch.float32, device="cuda"
    )
    reference.index_add_(0, token.to(torch.int64), pair_output.float() * weights[:, None])
    torch.testing.assert_close(actual, reference, rtol=2e-6, atol=2e-6)


def test_weighted_reduce_reuses_fp32_destination_and_is_repeatable():
    recv_tokens, num_pairs, hidden = 23, 193, 256
    pair_output = torch.randn(
        num_pairs, hidden, dtype=torch.bfloat16, device="cuda"
    )
    token = torch.arange(num_pairs, dtype=torch.int32, device="cuda") % recv_tokens
    weights = torch.rand(num_pairs, dtype=torch.float32, device="cuda")
    workspace = allocate_mxfp8_weighted_reduce_workspace(
        recv_tokens, hidden, device="cuda"
    )
    pointer = workspace.reduced.data_ptr()
    first = weighted_reduce_mxfp8(
        pair_output, token, weights, recv_tokens, workspace, out_dtype=torch.float32
    ).clone()
    second = weighted_reduce_mxfp8(
        pair_output, token, weights, recv_tokens, workspace, out_dtype=torch.float32
    )
    assert workspace.reduced.data_ptr() == pointer
    torch.testing.assert_close(second, first, rtol=2e-6, atol=2e-6)


def test_segmented_reduce_uses_inverse_route_map_without_atomics():
    recv_tokens, top_k, hidden = 29, 8, 256
    torch.manual_seed(903)
    valid = torch.rand(recv_tokens, top_k, device="cuda") > 0.35
    num_pairs = int(valid.sum().item())
    scatter = torch.full(
        (recv_tokens, top_k), -1, dtype=torch.int32, device="cuda"
    )
    scatter[valid] = torch.arange(num_pairs, dtype=torch.int32, device="cuda")
    token = (
        torch.arange(recv_tokens, dtype=torch.int32, device="cuda")[:, None]
        .expand(-1, top_k)[valid]
        .contiguous()
    )
    pair_output = torch.randn(
        num_pairs, hidden, dtype=torch.bfloat16, device="cuda"
    )
    weights = torch.rand(num_pairs, dtype=torch.float32, device="cuda")
    workspace = allocate_mxfp8_weighted_reduce_workspace(
        recv_tokens, hidden, device="cuda"
    )
    actual = segmented_weighted_reduce_mxfp8(
        pair_output,
        scatter.reshape(-1),
        weights,
        recv_tokens,
        top_k,
        workspace,
        out_dtype=torch.float32,
    )
    reference = torch.zeros(
        recv_tokens, hidden, dtype=torch.float32, device="cuda"
    )
    reference.index_add_(0, token, pair_output.float() * weights[:, None])
    torch.testing.assert_close(actual, reference, rtol=2e-6, atol=2e-6)
