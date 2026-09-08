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
    quantize_mxfp8_varlen_dual,
    quantize_mxfp8_varlen_k,
    quantize_mxfp8_varlen_k_pair,
    quantize_mxfp8_varlen_m,
    quantize_mxfp8_weight,
    quantize_mxfp8_weight_dual,
    sgd_update_and_quantize_mxfp8_weight,
    sgd_update_and_quantize_mxfp8_weight_dual,
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


def test_routed_varlen_dual_quantization_matches_separate_paths():
    _skip_if_not_sm100()
    torch.manual_seed(23)
    x = torch.randn(257, 256, dtype=torch.bfloat16, device="cuda")
    gather = torch.randperm(257, dtype=torch.int64, device="cuda")[:226].to(torch.int32)
    cu = torch.tensor([0, 0, 33, 97, 97, 226], dtype=torch.int32, device="cuda")
    expected_row = quantize_mxfp8_varlen_m(x, cu, gather_idx=gather)
    expected_col = quantize_mxfp8_varlen_k(x, cu, gather_idx=gather)
    actual_row, actual_col = quantize_mxfp8_varlen_dual(x, cu, gather_idx=gather)

    assert torch.equal(actual_row.qdata, expected_row.qdata)
    assert torch.equal(actual_col.qdata, expected_col.qdata)
    assert torch.equal(
        _active_varlen_m_scales(actual_row, cu).view(torch.uint8),
        _active_varlen_m_scales(expected_row, cu).view(torch.uint8),
    )
    assert torch.equal(
        _active_varlen_k_scales(actual_col, cu),
        _active_varlen_k_scales(expected_col, cu),
    )


def test_varlen_k_pair_quantization_matches_separate_paths():
    _skip_if_not_sm100()
    torch.manual_seed(29)
    first = torch.randn(226, 192, dtype=torch.bfloat16, device="cuda")
    second = torch.randn(226, 320, dtype=torch.bfloat16, device="cuda")
    cu = torch.tensor([0, 0, 33, 97, 97, 226], dtype=torch.int32, device="cuda")
    expected_first = quantize_mxfp8_varlen_k(first, cu)
    expected_second = quantize_mxfp8_varlen_k(second, cu)
    actual_first, actual_second = quantize_mxfp8_varlen_k_pair(first, second, cu)

    assert torch.equal(actual_first.qdata, expected_first.qdata)
    assert torch.equal(actual_second.qdata, expected_second.qdata)
    assert torch.equal(
        _active_varlen_k_scales(actual_first, cu),
        _active_varlen_k_scales(expected_first, cu),
    )
    assert torch.equal(
        _active_varlen_k_scales(actual_second, cu),
        _active_varlen_k_scales(expected_second, cu),
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


@pytest.mark.parametrize("shape", [(3, 256, 128), (2, 128, 256)])
def test_dense_dual_weight_quantization_matches_quack(shape):
    _skip_if_not_sm100()
    torch.manual_seed(21)
    weight = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
    expected_row = BlockScaledOperand.quantize(weight, MXFP8_E4M3, dim=-1)
    expected_col = BlockScaledOperand.quantize(weight, MXFP8_E4M3, dim=-2)
    actual_row, actual_col = quantize_mxfp8_weight_dual(weight)

    for actual, expected, dim in (
        (actual_row, expected_row, -1),
        (actual_col, expected_col, -2),
    ):
        assert actual.quant_dim == dim
        assert torch.equal(actual.qdata, expected.qdata)
        assert torch.equal(
            actual.scale.view(torch.uint8), expected.scale.view(torch.uint8)
        )


def test_fused_sgd_weight_refresh_matches_separate_operations():
    _skip_if_not_sm100()
    torch.manual_seed(22)
    weight = torch.randn(3, 256, 128, dtype=torch.bfloat16, device="cuda")
    expected_weight = weight.clone()
    grad = torch.randn_like(weight)
    learning_rate = 0.0125
    initial_version = weight._version
    expected_weight.add_(grad, alpha=-learning_rate)
    expected = quantize_mxfp8_weight(expected_weight)

    actual = sgd_update_and_quantize_mxfp8_weight(weight, grad, learning_rate)
    assert torch.equal(weight, expected_weight)
    assert weight._version == initial_version + 1
    assert torch.equal(actual.qdata, expected.qdata)
    assert torch.equal(actual.scale.view(torch.uint8), expected.scale.view(torch.uint8))


def test_fused_sgd_dual_weight_refresh_matches_separate_operations():
    _skip_if_not_sm100()
    torch.manual_seed(24)
    weight = torch.randn(3, 256, 128, dtype=torch.bfloat16, device="cuda")
    expected_weight = weight.clone()
    grad = torch.randn_like(weight)
    aux = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    expected_aux = aux.clone()
    aux_grad = torch.randn_like(aux)
    learning_rate = 0.0125
    initial_versions = weight._version, aux._version
    expected_weight.add_(grad, alpha=-learning_rate)
    expected_aux.add_(aux_grad, alpha=-learning_rate)
    expected_row = quantize_mxfp8_weight(expected_weight, dim=-1)
    expected_col = quantize_mxfp8_weight(expected_weight, dim=-2)

    actual_row, actual_col = sgd_update_and_quantize_mxfp8_weight_dual(
        weight,
        grad,
        learning_rate,
        aux_updates=((aux, aux_grad),),
    )
    assert torch.equal(weight, expected_weight)
    assert torch.equal(aux, expected_aux)
    assert weight._version == initial_versions[0] + 1
    assert aux._version == initial_versions[1] + 1
    for actual, expected in (
        (actual_row, expected_row),
        (actual_col, expected_col),
    ):
        assert torch.equal(actual.qdata, expected.qdata)
        assert torch.equal(
            actual.scale.view(torch.uint8), expected.scale.view(torch.uint8)
        )


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
