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
    dequantize_varlen_m_operand,
    moe_mxfp8_grouped_forward,
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
