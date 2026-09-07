# ********************************************************************************
# Copyright (c) 2026, SonicMoE contributors
# ********************************************************************************
"""Fused routed/segmented OCP MXFP8 quantization for SM100 training."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from quack.blockscaled import MXFP8_E4M3, BlockScaledOperand

_SF_VEC = 32
_SF_ATOM = 128


@triton.jit
def _rceil_e8m0(amax):
    """Bit-exact form of Quack's RCEIL E8M0 scale conversion."""
    ratio = amax / 448.0
    bits = ratio.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) & 0xFF
    has_fraction = (bits & 0x7FFFFF) != 0
    biased = tl.minimum(exponent + has_fraction.to(tl.int32), 255)
    # E8M0 byte 0 is reconstructed as 2^-127 by the MMA.  Triton flushes that
    # denormal, so use the equivalent finite divisor used by Quack's quantizer.
    scale_bits = tl.maximum(biased, 1) << 23
    scale = scale_bits.to(tl.float32, bitcast=True)
    return biased, scale


@triton.jit
def _find_expert_for_row(cu_ptr, row, E: tl.constexpr, N_ITERS: tl.constexpr):
    lo = tl.zeros((), dtype=tl.int32)
    hi = tl.full((), E, dtype=tl.int32)
    for _ in tl.static_range(N_ITERS):
        mid = (lo + hi) >> 1
        valid = mid < E
        end = tl.load(cu_ptr + tl.minimum(mid + 1, E), mask=valid, other=0x7FFFFFFF)
        go_right = valid & (end <= row)
        lo = tl.where(go_right, mid + 1, lo)
        hi = tl.where(go_right, hi, mid)
    return lo


@triton.jit
def _find_expert_for_padded_block(
    cu_ptr,
    padded_block,
    E: tl.constexpr,
    N_ITERS: tl.constexpr,
):
    lo = tl.zeros((), dtype=tl.int32)
    hi = tl.full((), E, dtype=tl.int32)
    for _ in tl.static_range(N_ITERS):
        mid = (lo + hi) >> 1
        valid = mid < E
        start = tl.load(cu_ptr + tl.minimum(mid, E - 1), mask=valid, other=0)
        padded_start = (start // 128 + mid) * 4
        go_right = valid & (padded_start <= padded_block)
        lo = tl.where(go_right, mid + 1, lo)
        hi = tl.where(go_right, hi, mid)
    return tl.maximum(lo - 1, 0)


@triton.jit
def _rowwise_varlen_m_kernel(
    x_ptr,
    gather_ptr,
    q_ptr,
    sf_ptr,
    cu_ptr,
    K: tl.constexpr,
    E: tl.constexpr,
    RK: tl.constexpr,
    HAS_GATHER: tl.constexpr,
    N_SEARCH_ITERS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    k_tile = tl.program_id(1)
    source_row = tl.load(gather_ptr + row) if HAS_GATHER else row
    offsets = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
    values = tl.load(x_ptr + source_row * K + offsets).to(tl.float32)
    groups = tl.reshape(values, [BLOCK_K // 32, 32])
    amax = tl.max(tl.abs(groups), axis=1)
    scale_byte, scale = _rceil_e8m0(amax)
    scaled = tl.maximum(
        tl.minimum(groups / scale[:, None], 448.0),
        -448.0,
    )
    tl.store(
        q_ptr + row * K + offsets,
        tl.reshape(scaled, [BLOCK_K]).to(tl.float8e4nv),
    )

    expert = _find_expert_for_row(cu_ptr, row, E, N_SEARCH_ITERS)
    expert_start = tl.load(cu_ptr + expert)
    padded_row = (expert_start // 128 + expert) * 128 + row - expert_start
    k_blocks = k_tile * (BLOCK_K // 32) + tl.arange(0, BLOCK_K // 32)
    rm = padded_row // 128
    row_inner = padded_row % 32
    row_outer = (padded_row % 128) // 32
    rk = k_blocks // 4
    k_inner = k_blocks % 4
    sf_offset = (((rm * RK + rk) * 32 + row_inner) * 4 + row_outer) * 4 + k_inner
    tl.store(sf_ptr + sf_offset, scale_byte.to(tl.uint8))


@triton.jit
def _segmented_varlen_k_kernel(
    x_ptr,
    gather_ptr,
    q_ptr,
    sf_ptr,
    cu_ptr,
    N: tl.constexpr,
    E: tl.constexpr,
    PADDED_BLOCKS: tl.constexpr,
    RK: tl.constexpr,
    HAS_GATHER: tl.constexpr,
    N_SEARCH_ITERS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    padded_block = tl.program_id(0)
    n_tile = tl.program_id(1)
    expert = _find_expert_for_padded_block(cu_ptr, padded_block, E, N_SEARCH_ITERS)
    start = tl.load(cu_ptr + expert)
    end = tl.load(cu_ptr + expert + 1)
    padded_start = (start // 128 + expert) * 4
    active_block = padded_block - padded_start
    active_blocks = (end - start + 31) // 32
    block_valid = (active_block >= 0) & (active_block < active_blocks)

    rows = start + active_block * 32 + tl.arange(0, 32)
    cols = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    value_mask = block_valid & (rows[:, None] < end) & (cols[None, :] < N)
    source_rows = (
        tl.load(gather_ptr + rows, mask=block_valid & (rows < end), other=0)
        if HAS_GATHER
        else rows
    )
    values = tl.load(
        x_ptr + source_rows[:, None] * N + cols[None, :],
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(values), axis=0)
    scale_byte, scale = _rceil_e8m0(amax)
    scaled = tl.maximum(tl.minimum(values / scale[None, :], 448.0), -448.0)
    tl.store(
        q_ptr + rows[:, None] * N + cols[None, :],
        scaled.to(tl.float8e4nv),
        mask=value_mask,
    )

    scale_mask = block_valid & (cols < N)
    rm = cols // 128
    row_inner = cols % 32
    row_outer = (cols % 128) // 32
    rk = padded_block // 4
    k_inner = padded_block % 4
    sf_offset = (((rm * RK + rk) * 32 + row_inner) * 4 + row_outer) * 4 + k_inner
    tl.store(sf_ptr + sf_offset, scale_byte.to(tl.uint8), mask=scale_mask)


@triton.jit
def _dense_rowwise_kernel(
    x_ptr,
    q_ptr,
    sf_ptr,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    RM: tl.constexpr,
    RK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    flat_row = tl.program_id(0)
    k_tile = tl.program_id(1)
    batch = flat_row // ROWS
    row = flat_row % ROWS
    offsets = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
    values = tl.load(x_ptr + flat_row * K + offsets).to(tl.float32)
    groups = tl.reshape(values, [BLOCK_K // 32, 32])
    amax = tl.max(tl.abs(groups), axis=1)
    scale_byte, scale = _rceil_e8m0(amax)
    scaled = tl.maximum(tl.minimum(groups / scale[:, None], 448.0), -448.0)
    tl.store(
        q_ptr + flat_row * K + offsets,
        tl.reshape(scaled, [BLOCK_K]).to(tl.float8e4nv),
    )

    k_blocks = k_tile * (BLOCK_K // 32) + tl.arange(0, BLOCK_K // 32)
    rm = row // 128
    row_inner = row % 32
    row_outer = (row % 128) // 32
    rk = k_blocks // 4
    k_inner = k_blocks % 4
    sf_offset = (
        ((((batch * RM + rm) * RK + rk) * 32 + row_inner) * 4 + row_outer) * 4
    ) + k_inner
    tl.store(sf_ptr + sf_offset, scale_byte.to(tl.uint8))


@triton.jit
def _dense_dim0_kernel(
    x_ptr,
    q_ptr,
    sf_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    RN: tl.constexpr,
    RK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0)
    row_block = tl.program_id(1)
    col_tile = tl.program_id(2)
    rows = row_block * 32 + tl.arange(0, 32)
    cols = col_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    values = tl.load(x_ptr + (batch * ROWS + rows[:, None]) * COLS + cols[None, :]).to(
        tl.float32
    )
    amax = tl.max(tl.abs(values), axis=0)
    scale_byte, scale = _rceil_e8m0(amax)
    scaled = tl.maximum(tl.minimum(values / scale[None, :], 448.0), -448.0)
    tl.store(
        q_ptr + (batch * ROWS + rows[:, None]) * COLS + cols[None, :],
        scaled.to(tl.float8e4nv),
    )

    rm = cols // 128
    row_inner = cols % 32
    row_outer = (cols % 128) // 32
    rk = row_block // 4
    k_inner = row_block % 4
    sf_offset = (
        ((((batch * RN + rm) * RK + rk) * 32 + row_inner) * 4 + row_outer) * 4
    ) + k_inner
    tl.store(sf_ptr + sf_offset, scale_byte.to(tl.uint8))


def _validate(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    gather_idx: torch.Tensor | None,
) -> tuple[int, int]:
    if x.ndim != 2 or x.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("x must be a two-dimensional BF16/FP32 tensor")
    if not x.is_cuda or not x.is_contiguous():
        raise ValueError("x must be a contiguous CUDA tensor")
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype != torch.int32:
        raise TypeError("cu_seqlens must be a one-dimensional int32 tensor")
    if cu_seqlens.device != x.device:
        raise ValueError("x and cu_seqlens must be on the same device")
    experts = cu_seqlens.numel() - 1
    if experts <= 0:
        raise ValueError("cu_seqlens must describe at least one expert")
    total = x.shape[0] if gather_idx is None else gather_idx.numel()
    if total == 0:
        raise ValueError("fully empty routed operands are not supported")
    if gather_idx is not None:
        if gather_idx.ndim != 1 or gather_idx.dtype != torch.int32:
            raise TypeError("gather_idx must be a one-dimensional int32 tensor")
        if gather_idx.device != x.device:
            raise ValueError("gather_idx and x must be on the same device")
    return experts, total


def _check_outputs(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    q_shape: tuple[int, ...],
    sf_shape: tuple[int, ...],
) -> None:
    if qdata.shape != q_shape or qdata.dtype != MXFP8_E4M3.qdata_dtype:
        raise ValueError(f"qdata must have shape {q_shape} and E4M3 dtype")
    if scale.shape != sf_shape or scale.dtype != MXFP8_E4M3.scale_dtype:
        raise ValueError(f"scale must have shape {sf_shape} and E8M0 dtype")
    if not qdata.is_contiguous() or not scale.is_contiguous():
        raise ValueError("qdata and scale outputs must be contiguous")


def quantize_mxfp8_weight(
    x: torch.Tensor,
    *,
    dim: int = -1,
    qdata_out: torch.Tensor | None = None,
    scale_out: torch.Tensor | None = None,
) -> BlockScaledOperand:
    """Direct-to-blocked MXFP8 cast for contiguous ``(E, M, K)`` weights."""
    if (
        x.ndim != 3
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
    ):
        raise ValueError(
            "weight must be a contiguous three-dimensional CUDA BF16 tensor"
        )
    experts, rows, cols = x.shape
    if rows % _SF_ATOM or cols % _SF_ATOM:
        raise ValueError("both weight matrix dimensions must be divisible by 128")
    if dim not in (-1, 2, -2, 1):
        raise ValueError("dim must select one of the final two weight dimensions")
    quant_dim = -1 if dim in (-1, 2) else -2
    q_shape = tuple(x.shape)
    if quant_dim == -1:
        rm, rk = rows // _SF_ATOM, cols // _SF_ATOM
        sf_shape = (experts, rm, rk, 32, 4, 4)
    else:
        rn, rk = cols // _SF_ATOM, rows // _SF_ATOM
        sf_shape = (experts, rn, rk, 32, 4, 4)
    qdata = (
        torch.empty(q_shape, dtype=MXFP8_E4M3.qdata_dtype, device=x.device)
        if qdata_out is None
        else qdata_out
    )
    scale = (
        torch.empty(sf_shape, dtype=MXFP8_E4M3.scale_dtype, device=x.device)
        if scale_out is None
        else scale_out
    )
    _check_outputs(qdata, scale, q_shape, sf_shape)
    if quant_dim == -1:
        _dense_rowwise_kernel[(experts * rows, cols // _SF_ATOM)](
            x,
            qdata,
            scale.view(torch.uint8),
            ROWS=rows,
            K=cols,
            RM=rm,
            RK=rk,
            BLOCK_K=_SF_ATOM,
            num_warps=4,
        )
    else:
        block_n = 64
        _dense_dim0_kernel[(experts, rows // _SF_VEC, cols // block_n)](
            x,
            qdata,
            scale.view(torch.uint8),
            ROWS=rows,
            COLS=cols,
            RN=rn,
            RK=rk,
            BLOCK_N=block_n,
            num_warps=4,
        )
    return BlockScaledOperand.from_parts(
        qdata,
        scale,
        MXFP8_E4M3,
        orig_dtype=x.dtype,
        quant_dim=quant_dim,
    )


def quantize_mxfp8_varlen_m(
    x: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    *,
    gather_idx: torch.Tensor | None = None,
    qdata_out: torch.Tensor | None = None,
    scale_out: torch.Tensor | None = None,
) -> BlockScaledOperand:
    """Fused optional gather + rowwise quantization + blocked scale store."""
    experts, total = _validate(x, cu_seqlens_m, gather_idx)
    k = x.shape[1]
    if k % _SF_ATOM:
        raise ValueError(f"K={k} must be divisible by {_SF_ATOM}")
    padded_rm = (total + _SF_ATOM - 1) // _SF_ATOM + experts - 1
    rk = (k + _SF_ATOM - 1) // _SF_ATOM
    q_shape = (total, k)
    sf_shape = (1, padded_rm, rk, 32, 4, 4)
    qdata = (
        torch.empty(q_shape, dtype=MXFP8_E4M3.qdata_dtype, device=x.device)
        if qdata_out is None
        else qdata_out
    )
    scale = (
        torch.empty(sf_shape, dtype=MXFP8_E4M3.scale_dtype, device=x.device)
        if scale_out is None
        else scale_out
    )
    _check_outputs(qdata, scale, q_shape, sf_shape)
    _rowwise_varlen_m_kernel[(total, k // _SF_ATOM)](
        x,
        gather_idx,
        qdata,
        scale.view(torch.uint8),
        cu_seqlens_m,
        K=k,
        E=experts,
        RK=rk,
        HAS_GATHER=gather_idx is not None,
        N_SEARCH_ITERS=experts.bit_length(),
        BLOCK_K=_SF_ATOM,
        num_warps=4,
    )
    return BlockScaledOperand.from_parts(qdata, scale, MXFP8_E4M3, orig_dtype=x.dtype)


def quantize_mxfp8_varlen_k(
    x: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    gather_idx: torch.Tensor | None = None,
    qdata_out: torch.Tensor | None = None,
    scale_out: torch.Tensor | None = None,
) -> BlockScaledOperand:
    """Fused optional gather + expert-segmented dim-0 MXFP8 quantization."""
    experts, total = _validate(x, cu_seqlens_k, gather_idx)
    n = x.shape[1]
    padded_rk = (total + _SF_ATOM - 1) // _SF_ATOM + experts - 1
    padded_blocks = padded_rk * (_SF_ATOM // _SF_VEC)
    rn = (n + _SF_ATOM - 1) // _SF_ATOM
    q_shape = (total, n)
    sf_shape = (1, rn, padded_rk, 32, 4, 4)
    qdata = (
        torch.empty(q_shape, dtype=MXFP8_E4M3.qdata_dtype, device=x.device)
        if qdata_out is None
        else qdata_out
    )
    scale = (
        torch.empty(sf_shape, dtype=MXFP8_E4M3.scale_dtype, device=x.device)
        if scale_out is None
        else scale_out
    )
    _check_outputs(qdata, scale, q_shape, sf_shape)
    block_n = 64
    _segmented_varlen_k_kernel[(padded_blocks, triton.cdiv(n, block_n))](
        x,
        gather_idx,
        qdata,
        scale.view(torch.uint8),
        cu_seqlens_k,
        N=n,
        E=experts,
        PADDED_BLOCKS=padded_blocks,
        RK=padded_rk,
        HAS_GATHER=gather_idx is not None,
        N_SEARCH_ITERS=experts.bit_length(),
        BLOCK_N=block_n,
        num_warps=4,
    )
    return BlockScaledOperand.from_parts(
        qdata,
        scale,
        MXFP8_E4M3,
        orig_dtype=x.dtype,
        quant_dim=-2,
    )


__all__ = [
    "quantize_mxfp8_varlen_k",
    "quantize_mxfp8_varlen_m",
    "quantize_mxfp8_weight",
]
