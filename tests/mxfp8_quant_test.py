# Copyright (c) 2026, SonicMoE contributors.
"""Numerical tests for routed and segmented MXFP8 quantization fusions."""

import pytest
import torch
from quack.blockscaled import MXFP8_E4M3, BlockScaledOperand, unpack_scale_blocked_to_2d
from quack.blockscaled import (
    quantize_mxfp8_varlen_k as quantize_mxfp8_varlen_k_ref,
)
from quack.blockscaled import (
    quantize_mxfp8_varlen_m as quantize_mxfp8_varlen_m_ref,
)
from sonicmoe.functional.triton_kernels import (
    TC_topk_router_metadata_triton,
    TC_topk_router_metadata_triton_workspace,
    topk_router_workspace_shape,
)
from sonicmoe.functional.triton_kernels.mxfp8_quant import (
    quantize_mxfp8_varlen_k,
    quantize_mxfp8_varlen_m,
    quantize_mxfp8_weight,
)


def _skip_if_not_sm100():
    if torch.cuda.get_device_properties(0).major != 10:
        pytest.skip("SM100 required")


def _active_varlen_m_scales(operand, cu):
    experts = cu.numel() - 1
    sf_k = operand.shape[1] // 32
    linear = unpack_scale_blocked_to_2d(
        operand.scale, operand.scale.shape[1] * 128, sf_k
    )[0]
    offsets = cu.tolist()
    return torch.cat(
        [
            linear[
                (offsets[e] // 128 + e) * 128 : (offsets[e] // 128 + e) * 128
                + offsets[e + 1]
                - offsets[e]
            ]
            for e in range(experts)
        ]
    )


def _active_varlen_k_scales(operand, cu):
    sf_k = operand.scale.shape[2] * 4
    linear = unpack_scale_blocked_to_2d(operand.scale, operand.shape[1], sf_k)[0].view(
        torch.uint8
    )
    offsets = cu.tolist()
    return torch.cat(
        [
            linear[
                :,
                (offsets[e] // 128 + e) * 4 : (offsets[e] // 128 + e) * 4
                + (offsets[e + 1] - offsets[e] + 31) // 32,
            ].flatten()
            for e in range(cu.numel() - 1)
        ]
    )


def test_routed_varlen_m_quantization_matches_quack():
    _skip_if_not_sm100()
    torch.manual_seed(0)
    x = torch.randn(197, 256, dtype=torch.bfloat16, device="cuda")
    gather = torch.randperm(197, dtype=torch.int64, device="cuda")[:163].to(torch.int32)
    cu = torch.tensor([0, 0, 1, 34, 34, 163], dtype=torch.int32, device="cuda")
    expected = quantize_mxfp8_varlen_m_ref(x[gather.long()].contiguous(), cu)
    actual = quantize_mxfp8_varlen_m(x, cu, gather_idx=gather)
    assert torch.equal(actual.qdata, expected.qdata)
    assert torch.equal(
        _active_varlen_m_scales(actual, cu).view(torch.uint8),
        _active_varlen_m_scales(expected, cu).view(torch.uint8),
    )


def test_routed_varlen_k_quantization_matches_quack():
    _skip_if_not_sm100()
    torch.manual_seed(1)
    x = torch.randn(257, 192, dtype=torch.bfloat16, device="cuda")
    gather = torch.randperm(257, dtype=torch.int64, device="cuda")[:226].to(torch.int32)
    cu = torch.tensor([0, 0, 33, 97, 97, 226], dtype=torch.int32, device="cuda")
    expected = quantize_mxfp8_varlen_k_ref(x[gather.long()].contiguous(), cu)
    actual = quantize_mxfp8_varlen_k(x, cu, gather_idx=gather)
    assert torch.equal(actual.qdata, expected.qdata)
    assert torch.equal(
        _active_varlen_k_scales(actual, cu),
        _active_varlen_k_scales(expected, cu),
    )


def test_quantization_uses_caller_owned_outputs():
    _skip_if_not_sm100()
    x = torch.randn(65, 128, dtype=torch.bfloat16, device="cuda")
    cu = torch.tensor([0, 1, 1, 65], dtype=torch.int32, device="cuda")
    initial = quantize_mxfp8_varlen_m(x, cu)
    qdata = torch.empty_like(initial.qdata)
    scale = torch.empty_like(initial.scale)
    result = quantize_mxfp8_varlen_m(x, cu, qdata_out=qdata, scale_out=scale)
    assert result.qdata is qdata and result.scale is scale
    assert torch.equal(result.qdata, initial.qdata)
    assert torch.equal(
        _active_varlen_m_scales(result, cu).view(torch.uint8),
        _active_varlen_m_scales(initial, cu).view(torch.uint8),
    )


@pytest.mark.parametrize("dim", [-1, -2])
def test_dense_weight_quantization_matches_quack(dim):
    _skip_if_not_sm100()
    torch.manual_seed(2)
    weight = torch.randn(3, 256, 128, dtype=torch.bfloat16, device="cuda")
    expected = BlockScaledOperand.quantize(weight, MXFP8_E4M3, dim=dim)
    actual = quantize_mxfp8_weight(weight, dim=dim)
    assert actual.quant_dim == dim
    assert torch.equal(actual.qdata, expected.qdata)
    assert torch.equal(actual.scale.view(torch.uint8), expected.scale.view(torch.uint8))


def test_caller_owned_router_workspace_matches_official_path():
    _skip_if_not_sm100()
    torch.manual_seed(3)
    tokens, experts, top_k = 777, 16, 4
    indices = (
        torch.randn(tokens, experts, device="cuda").topk(top_k, dim=-1).indices.int()
    )

    def outputs():
        routed = tokens * top_k
        return (
            torch.empty(experts, dtype=torch.int32, device="cuda"),
            torch.empty(experts + 1, dtype=torch.int32, device="cuda"),
            torch.empty(routed, dtype=torch.int32, device="cuda"),
            torch.empty(routed, dtype=torch.int32, device="cuda"),
            torch.empty(routed, dtype=torch.int32, device="cuda"),
        )

    expected = outputs()
    TC_topk_router_metadata_triton(indices, experts, *expected)
    actual = outputs()
    scratch = torch.empty(
        topk_router_workspace_shape(tokens, experts, top_k),
        dtype=torch.int32,
        device="cuda",
    )
    TC_topk_router_metadata_triton_workspace(indices, experts, *actual, scratch)
    for result, reference in zip(actual, expected):
        assert torch.equal(result, reference)
