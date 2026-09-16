# Copyright (c) 2026, SonicMoE contributors.
"""Unit tests for packed expert-parallel MXFP8 transport records."""

import pytest
import torch
from sonicmoe.functional.mxfp8_transport import (
    pack_mxfp8_transport,
    unpack_mxfp8_transport,
)
from sonicmoe.functional.triton_kernels.mxfp8_quant import (
    dequantize_mxfp8_rows,
    quantize_mxfp8_rows,
)


def test_pack_mxfp8_transport_round_trip_and_storage_views():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(908)
    source = torch.randn(11, 256, dtype=torch.bfloat16, device="cuda")
    qdata, scale = quantize_mxfp8_rows(source)
    send_indices = torch.tensor([8, 0, 10, 3, 3, 6], device="cuda")
    expert_ids = torch.tensor(
        [[0, -1], [1, 0], [0, 1], [-1, 1], [1, -1], [0, -1]],
        dtype=torch.int32,
        device="cuda",
    )
    weights = torch.randn(6, 2, dtype=torch.float32, device="cuda")

    payload, layout = pack_mxfp8_transport(
        qdata,
        scale,
        send_indices,
        expert_ids,
        weights,
    )
    actual_q, actual_scale, actual_ids, actual_weights = unpack_mxfp8_transport(
        payload, layout
    )

    assert actual_q.untyped_storage().data_ptr() == payload.untyped_storage().data_ptr()
    assert actual_scale.stride(0) == layout.record_bytes
    assert actual_ids.stride(0) == layout.record_bytes // 4
    assert actual_weights.stride(0) == layout.record_bytes // 4
    assert torch.equal(actual_q, qdata.index_select(0, send_indices))
    assert torch.equal(actual_scale, scale.index_select(0, send_indices))
    assert torch.equal(actual_ids, expert_ids)
    assert torch.equal(actual_weights, weights)
    expected_dequant = dequantize_mxfp8_rows(
        qdata.index_select(0, send_indices),
        scale.index_select(0, send_indices),
    )
    actual_dequant = dequantize_mxfp8_rows(actual_q, actual_scale)
    assert torch.equal(actual_dequant, expected_dequant)


def test_pack_mxfp8_transport_reuses_caller_output():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    source = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
    qdata, scale = quantize_mxfp8_rows(source)
    send_indices = torch.tensor([3, 1], dtype=torch.int64, device="cuda")
    expert_ids = torch.tensor([[0], [1]], dtype=torch.int32, device="cuda")
    weights = torch.ones(2, 1, dtype=torch.float32, device="cuda")
    _, layout = pack_mxfp8_transport(qdata, scale, send_indices, expert_ids, weights)
    output = torch.empty((2, layout.record_bytes), dtype=torch.uint8, device="cuda")
    pointer = output.data_ptr()
    payload, _ = pack_mxfp8_transport(
        qdata,
        scale,
        send_indices,
        expert_ids,
        weights,
        out=output,
    )
    assert payload.data_ptr() == pointer
