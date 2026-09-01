# ********************************************************************************
# Copyright (c) 2026 SonicMoE contributors
# ********************************************************************************
"""Correctness gates for the experimental SM120 MXFP8 grouped path."""

import pytest
import torch
import torch.nn.functional as F
from quack.blockscaled import (
    BlockScaledOperand,
    pack_scale_2d_to_blocked_contig,
    to_mx_compiled,
)

from sonicmoe.functional.mxfp8 import (
    MXFP8_FORMAT,
    Mxfp8MoEKernelConfig,
    allocate_mxfp8_moe_workspace,
    dequantize_varlen_m_operand,
    moe_mxfp8_grouped_forward,
    mxfp8_down_grouped_weighted_reduce,
    pack_varlen_m_scales,
    quantize_mxfp8_rows,
    quantize_varlen_m_operand,
    unpack_varlen_m_scales,
)

_PHYSICAL_ARCH = (
    torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
)
pytestmark = pytest.mark.skipif(
    _PHYSICAL_ARCH != 12,
    reason="MXFP8 variable-M warp-MMA kernels require physical SM120/SM121 hardware",
)


def _weight_operand(x: torch.Tensor) -> BlockScaledOperand:
    qdata, scales = to_mx_compiled(x.contiguous(), 32)
    return BlockScaledOperand.from_parts(
        qdata, pack_scale_2d_to_blocked_contig(scales), MXFP8_FORMAT
    )


def _ragged_indptr(seqlens: list[int]) -> torch.Tensor:
    return torch.tensor(
        [0, *torch.tensor(seqlens, dtype=torch.int64).cumsum(0).tolist()],
        dtype=torch.int32,
        device="cuda",
    )


def test_mxfp8_varlen_scale_bytes_round_trip():
    """Expert padding must not alter any active 1x32 E8M0 scale byte."""

    seqlens = [0, 1, 128, 127, 129]
    cu = _ragged_indptr(seqlens)
    torch.manual_seed(123)
    x = torch.randn(sum(seqlens), 256, dtype=torch.bfloat16, device="cuda")
    _, linear_scales = quantize_mxfp8_rows(x)
    blocked = pack_varlen_m_scales(linear_scales, cu)
    unpacked = unpack_varlen_m_scales(blocked, cu, sf_k=x.shape[1] // 32)

    assert torch.equal(unpacked.view(torch.uint8), linear_scales.view(torch.uint8))


def test_mxfp8_grouped_swiglu_forward_ragged():
    """Exercise the FC1-quantize-FC2 chain across 128-row boundaries."""

    seqlens = [0, 1, 128, 127, 129]
    experts = len(seqlens)
    hidden, intermediate = 256, 128
    cu = _ragged_indptr(seqlens)
    torch.manual_seed(123)

    x_hp = torch.randn(sum(seqlens), hidden, dtype=torch.bfloat16, device="cuda") * 0.04
    w1_hp = (
        torch.randn(
            experts,
            2 * intermediate,
            hidden,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.04
    )
    w2_hp = (
        torch.randn(
            experts,
            hidden,
            intermediate,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.04
    )
    x = quantize_varlen_m_operand(x_hp, cu)
    w1 = _weight_operand(w1_hp)
    w2 = _weight_operand(w2_hp)

    out, postact = moe_mxfp8_grouped_forward(
        x, w1, w2, cu, config=Mxfp8MoEKernelConfig()
    )
    x_dq = dequantize_varlen_m_operand(x, cu)
    w1_dq = w1.dequantize(torch.float32)
    w2_dq = w2.dequantize(torch.float32)
    postact_dq = dequantize_varlen_m_operand(postact, cu)

    post_refs = []
    out_refs = []
    cu_host = cu.cpu().tolist()
    for expert in range(experts):
        lo, hi = cu_host[expert], cu_host[expert + 1]
        preact = x_dq[lo:hi] @ w1_dq[expert].T
        post_refs.append(F.silu(preact[:, 0::2]) * preact[:, 1::2])
        # FC2 consumes the quantized post-activation, so its isolated reference
        # must consume that exact dequantized operand as well.
        out_refs.append(postact_dq[lo:hi] @ w2_dq[expert].T)

    post_ref = torch.cat(post_refs)
    out_ref = torch.cat(out_refs)
    post_rel_l2 = torch.linalg.vector_norm(
        postact_dq - post_ref
    ) / torch.linalg.vector_norm(post_ref)
    out_rel_l2 = torch.linalg.vector_norm(
        out.float() - out_ref
    ) / torch.linalg.vector_norm(out_ref)
    assert torch.isfinite(out).all()
    assert post_rel_l2 < 0.12
    assert out_rel_l2 < 0.02


def test_mxfp8_workspace_reuses_outputs_without_changing_results():
    """Capacity views must preserve active value/scale bytes and FC2 output."""

    seqlens = [0, 1, 127, 128, 129]
    experts = len(seqlens)
    hidden, intermediate = 256, 128
    total_m = sum(seqlens)
    cu = _ragged_indptr(seqlens)
    torch.manual_seed(321)
    x_hp = torch.randn(total_m, hidden, dtype=torch.bfloat16, device="cuda") * 0.04
    w1_hp = (
        torch.randn(
            experts, 2 * intermediate, hidden, dtype=torch.bfloat16, device="cuda"
        )
        * 0.04
    )
    w2_hp = (
        torch.randn(
            experts, hidden, intermediate, dtype=torch.bfloat16, device="cuda"
        )
        * 0.04
    )
    x = quantize_varlen_m_operand(x_hp, cu)
    w1 = _weight_operand(w1_hp)
    w2 = _weight_operand(w2_hp)
    config = Mxfp8MoEKernelConfig()

    reference_out, reference_postact = moe_mxfp8_grouped_forward(
        x, w1, w2, cu, config=config
    )
    workspace = allocate_mxfp8_moe_workspace(
        experts,
        total_m + 128,
        hidden,
        intermediate,
        device="cuda",
    )
    q_ptr = workspace.postact_qdata.data_ptr()
    sf_ptr = workspace.postact_scale.data_ptr()
    out_ptr = workspace.fc2_output.data_ptr()
    candidate_out, candidate_postact = moe_mxfp8_grouped_forward(
        x, w1, w2, cu, config=config, workspace=workspace
    )
    torch.cuda.synchronize()

    assert workspace.postact_qdata.data_ptr() == q_ptr
    assert workspace.postact_scale.data_ptr() == sf_ptr
    assert workspace.fc2_output.data_ptr() == out_ptr
    assert candidate_out.data_ptr() == out_ptr
    assert candidate_postact.qdata.data_ptr() == q_ptr
    assert torch.equal(candidate_out, reference_out)
    assert torch.equal(candidate_postact.qdata, reference_postact.qdata)
    candidate_sf = unpack_varlen_m_scales(
        candidate_postact.scale, cu, sf_k=intermediate // 32
    )
    reference_sf = unpack_varlen_m_scales(
        reference_postact.scale, cu, sf_k=intermediate // 32
    )
    assert torch.equal(candidate_sf.view(torch.uint8), reference_sf.view(torch.uint8))

    # Scheduler permutation changes task issue order only; physical expert
    # layout, row order, and every output byte remain unchanged.
    heavy_first = torch.argsort(
        torch.tensor(seqlens, dtype=torch.int32, device="cuda"),
        descending=True,
        stable=True,
    ).to(torch.int32)
    ordered_out, ordered_postact = moe_mxfp8_grouped_forward(
        x,
        w1,
        w2,
        cu,
        config=config,
        workspace=workspace,
        expert_order=heavy_first,
    )
    torch.cuda.synchronize()
    assert torch.equal(ordered_out, reference_out)
    assert torch.equal(ordered_postact.qdata, reference_postact.qdata)
    ordered_sf = unpack_varlen_m_scales(
        ordered_postact.scale, cu, sf_k=intermediate // 32
    )
    assert torch.equal(ordered_sf.view(torch.uint8), reference_sf.view(torch.uint8))

    recv_tokens = 37
    recv_token = (
        torch.arange(total_m, dtype=torch.int32, device="cuda") % recv_tokens
    )
    scores = torch.rand(total_m, dtype=torch.float32, device="cuda")
    reduced = torch.empty(
        recv_tokens, hidden, dtype=torch.float32, device="cuda"
    )
    direct = mxfp8_down_grouped_weighted_reduce(
        ordered_postact,
        w2,
        cu,
        recv_token,
        scores,
        reduced,
        config=config,
        expert_order=heavy_first,
    )
    reference_reduced = torch.zeros_like(reduced)
    reference_reduced.index_add_(
        0, recv_token, reference_out.float() * scores[:, None]
    )
    difference = direct - reference_reduced
    bad = difference.abs() > 0.05 + 0.05 * reference_reduced.abs()
    assert int(bad.sum()) == 0
    assert (
        torch.linalg.vector_norm(difference)
        / torch.linalg.vector_norm(reference_reduced).clamp_min(1e-20)
        < 5e-3
    )
