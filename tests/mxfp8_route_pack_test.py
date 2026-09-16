# Copyright (c) 2026, SonicMoE contributors.
"""Byte-exact SM100 tests for counting MXFP8 route-pack."""

import pytest
import torch
from quack.blockscaled import unpack_scale_blocked_to_2d
from sonicmoe.functional.mxfp8_route_pack import (
    allocate_mxfp8_route_pack_workspace,
    route_pack_mxfp8,
)
from sonicmoe.functional.triton_kernels.mxfp8_quant import quantize_mxfp8_rows


def _skip_if_not_sm100():
    if torch.cuda.get_device_properties(0).major != 10:
        pytest.skip("SM100 required")


def _unpack_active_scales(blocked, indptr, sf_k):
    linear = unpack_scale_blocked_to_2d(blocked, blocked.shape[1] * 128, sf_k)[0]
    offsets = indptr.tolist()
    return torch.cat(
        [
            linear[
                (offsets[expert] // 128 + expert) * 128 : (
                    offsets[expert] // 128 + expert
                )
                * 128
                + offsets[expert + 1]
                - offsets[expert]
            ]
            for expert in range(indptr.numel() - 1)
        ]
    )


def test_route_pack_preserves_value_scale_and_metadata_bytes():
    _skip_if_not_sm100()
    recv_tokens, top_k, local_experts, hidden = 37, 4, 7, 256
    torch.manual_seed(2026)
    source = torch.randn(recv_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    qdata, linear_scale = quantize_mxfp8_rows(source)
    rows = torch.arange(recv_tokens, device="cuda")[:, None]
    slots = torch.arange(top_k, device="cuda")[None, :]
    expert_ids = ((rows + 2 * slots) % local_experts).to(torch.int32)
    valid = (rows + slots) % 5 != 0
    expert_ids = torch.where(valid, expert_ids, torch.full_like(expert_ids, -1))
    weights = torch.rand(recv_tokens, top_k, dtype=torch.float32, device="cuda")
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    total_pairs = int(valid.sum().item())
    workspace = allocate_mxfp8_route_pack_workspace(
        recv_tokens,
        recv_tokens * top_k,
        top_k,
        local_experts,
        hidden,
        device="cuda",
    )
    result = route_pack_mxfp8(
        qdata, linear_scale, expert_ids, weights, total_pairs, workspace
    )

    counts = torch.bincount(
        expert_ids[valid].to(torch.int64), minlength=local_experts
    ).to(torch.int32)
    assert torch.equal(result.indptr[1:] - result.indptr[:-1], counts)
    assert int(result.indptr[-1]) == total_pairs
    assert torch.all(result.expert[:-1] <= result.expert[1:])
    recv = result.recv_token.to(torch.int64)
    assert result.operand.qdata.data_ptr() == qdata.data_ptr()
    assert not result.physical_qdata_copied
    assert torch.equal(result.operand.qdata, qdata)
    assert torch.equal(result.a_idx.to(torch.int64), recv)
    scales = _unpack_active_scales(result.operand.scale, result.indptr, hidden // 32)
    assert torch.equal(
        scales.view(torch.uint8),
        linear_scale.index_select(0, recv).view(torch.uint8),
    )
    route_match = expert_ids.index_select(0, recv) == result.expert[:, None]
    assert torch.equal(
        route_match.sum(dim=1),
        torch.ones(total_pairs, dtype=torch.int64, device="cuda"),
    )
    expected_weights = weights.index_select(0, recv)[route_match]
    assert torch.equal(
        result.weights.view(torch.int32), expected_weights.view(torch.int32)
    )
    flat_valid = valid.reshape(-1)
    assert torch.equal(
        result.scatter_pos[flat_valid].sort().values,
        torch.arange(total_pairs, dtype=torch.int32, device="cuda"),
    )


def test_route_pack_reuses_capacity_and_handles_empty_receive_rank():
    _skip_if_not_sm100()
    workspace = allocate_mxfp8_route_pack_workspace(32, 128, 4, 8, 256, device="cuda")
    pointers = tuple(
        tensor.data_ptr()
        for tensor in (
            workspace.qdata,
            workspace.scale,
            workspace.recv_token,
            workspace.weights,
            workspace.indptr,
        )
    )
    source = torch.randn(19, 256, dtype=torch.bfloat16, device="cuda")
    contiguous_qdata, scale = quantize_mxfp8_rows(source)
    payload = torch.empty(19, 260, dtype=torch.uint8, device="cuda")
    qdata = payload[:, :256].view(contiguous_qdata.dtype)
    qdata.copy_(contiguous_qdata)
    expert_ids = torch.arange(76, dtype=torch.int32, device="cuda").view(19, 4) % 8
    weights = torch.full((19, 4), 0.25, dtype=torch.float32, device="cuda")
    for _ in range(2):
        result = route_pack_mxfp8(qdata, scale, expert_ids, weights, 76, workspace)
        assert result.operand.qdata.data_ptr() == workspace.qdata.data_ptr()
        assert result.physical_qdata_copied
    assert pointers == tuple(
        tensor.data_ptr()
        for tensor in (
            workspace.qdata,
            workspace.scale,
            workspace.recv_token,
            workspace.weights,
            workspace.indptr,
        )
    )

    empty_q = qdata[:0]
    empty_scale = scale[:0]
    empty_ids = expert_ids[:0]
    empty_weights = weights[:0]
    empty = route_pack_mxfp8(
        empty_q, empty_scale, empty_ids, empty_weights, 0, workspace
    )
    assert empty.total_pairs == 0
    assert torch.equal(empty.indptr, torch.zeros(9, dtype=torch.int32, device="cuda"))
