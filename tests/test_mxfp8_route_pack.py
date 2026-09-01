# ********************************************************************************
# Copyright (c) 2026 SonicMoE contributors
# ********************************************************************************
"""Byte-exact gates for the counting MXFP8 route-pack path."""

import pytest
import torch

from sonicmoe.functional.mxfp8 import (
    quantize_mxfp8_rows,
    unpack_varlen_m_scales,
)
from sonicmoe.functional.mxfp8_route_pack import (
    allocate_mxfp8_route_pack_workspace,
    route_pack_mxfp8,
)

_PHYSICAL_ARCH = (
    torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
)
pytestmark = pytest.mark.skipif(
    _PHYSICAL_ARCH != 12,
    reason="MXFP8 route-pack validation requires physical SM120/SM121 hardware",
)


def test_route_pack_preserves_route_activation_and_scale_bytes():
    recv_tokens, top_k = 37, 4
    local_experts, hidden = 7, 256
    torch.manual_seed(2026)
    source = torch.randn(
        recv_tokens, hidden, dtype=torch.bfloat16, device="cuda"
    )
    qdata, linear_scale = quantize_mxfp8_rows(source)

    rows = torch.arange(recv_tokens, device="cuda")[:, None]
    slots = torch.arange(top_k, device="cuda")[None, :]
    expert_ids = ((rows + 2 * slots) % local_experts).to(torch.int32)
    valid = (rows + slots) % 5 != 0
    expert_ids = torch.where(valid, expert_ids, torch.full_like(expert_ids, -1))
    weights = torch.rand(
        recv_tokens, top_k, dtype=torch.float32, device="cuda"
    )
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    total_pairs = int(valid.sum().item())

    workspace = allocate_mxfp8_route_pack_workspace(
        recv_tokens,
        total_pairs,
        top_k,
        local_experts,
        hidden,
        device="cuda",
    )
    result = route_pack_mxfp8(
        qdata,
        linear_scale,
        expert_ids,
        weights,
        total_pairs,
        workspace,
    )
    torch.cuda.synchronize()

    reference_counts = torch.bincount(
        expert_ids[expert_ids >= 0].to(torch.int64), minlength=local_experts
    ).to(torch.int32)
    assert torch.equal(result.indptr[1:] - result.indptr[:-1], reference_counts)
    assert int(result.indptr[-1]) == total_pairs
    assert torch.all(result.expert[:-1] <= result.expert[1:])

    recv = result.recv_token.to(torch.int64)
    assert torch.equal(result.operand.qdata, qdata.index_select(0, recv))
    unpacked_scale = unpack_varlen_m_scales(
        result.operand.scale,
        result.indptr,
        sf_k=hidden // 32,
    )
    expected_scale = linear_scale.index_select(0, recv)
    assert torch.equal(
        unpacked_scale.view(torch.uint8), expected_scale.view(torch.uint8)
    )

    route_match = expert_ids.index_select(0, recv) == result.expert[:, None]
    assert torch.equal(
        route_match.sum(dim=1),
        torch.ones(total_pairs, dtype=torch.int64, device="cuda"),
    )
    expected_weights = weights.index_select(0, recv)[route_match]
    assert torch.equal(result.weights.view(torch.int32), expected_weights.view(torch.int32))


def test_route_pack_reuses_all_capacity_buffers():
    recv_tokens, top_k, local_experts, hidden = 19, 4, 8, 256
    source = torch.randn(
        recv_tokens, hidden, dtype=torch.bfloat16, device="cuda"
    )
    qdata, linear_scale = quantize_mxfp8_rows(source)
    expert_ids = torch.arange(
        recv_tokens * top_k, dtype=torch.int32, device="cuda"
    ).reshape(recv_tokens, top_k) % local_experts
    weights = torch.full(
        (recv_tokens, top_k), 0.25, dtype=torch.float32, device="cuda"
    )
    total_pairs = recv_tokens * top_k
    workspace = allocate_mxfp8_route_pack_workspace(
        recv_tokens,
        total_pairs,
        top_k,
        local_experts,
        hidden,
        device="cuda",
    )
    pointers = (
        workspace.qdata.data_ptr(),
        workspace.scale.data_ptr(),
        workspace.recv_token.data_ptr(),
        workspace.weights.data_ptr(),
    )
    first = route_pack_mxfp8(
        qdata, linear_scale, expert_ids, weights, total_pairs, workspace
    )
    second = route_pack_mxfp8(
        qdata, linear_scale, expert_ids, weights, total_pairs, workspace
    )
    torch.cuda.synchronize()
    assert pointers == (
        workspace.qdata.data_ptr(),
        workspace.scale.data_ptr(),
        workspace.recv_token.data_ptr(),
        workspace.weights.data_ptr(),
    )
    assert first.operand.qdata.data_ptr() == workspace.qdata.data_ptr()
    assert second.operand.qdata.data_ptr() == workspace.qdata.data_ptr()
