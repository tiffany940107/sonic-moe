# ********************************************************************************
# Copyright (c) 2026, SonicMoE contributors
# ********************************************************************************
"""Fused routed/segmented OCP MXFP8 quantization for SM100 training."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from quack.blockscaled import MXFP8_E4M3, BlockScaledOperand

_SF_VEC = 32
_SF_ATOM = 128
_VARLEN_BLOCK_M = int(os.environ.get("SONICMOE_MXFP8_VARLEN_BLOCK_M", "8"))
if _VARLEN_BLOCK_M not in (1, 2, 4, 8, 16, 32, 64):
    raise ValueError("SONICMOE_MXFP8_VARLEN_BLOCK_M must be 1, 2, 4, 8, 16, 32, or 64")
_PHYSICAL_ROW_NUM_WARPS = int(os.environ.get("SONICMOE_MXFP8_PHYSICAL_ROW_WARPS", "4"))
if _PHYSICAL_ROW_NUM_WARPS not in (1, 2, 4, 8):
    raise ValueError("SONICMOE_MXFP8_PHYSICAL_ROW_WARPS must be 1, 2, 4, or 8")
_VARLEN_K_BLOCK_N = int(os.environ.get("SONICMOE_MXFP8_VARLEN_K_BLOCK_N", "64"))
if _VARLEN_K_BLOCK_N not in (32, 64, 128):
    raise ValueError("SONICMOE_MXFP8_VARLEN_K_BLOCK_N must be 32, 64, or 128")
_VARLEN_K_NUM_WARPS = int(os.environ.get("SONICMOE_MXFP8_VARLEN_K_WARPS", "4"))
if _VARLEN_K_NUM_WARPS not in (1, 2, 4, 8):
    raise ValueError("SONICMOE_MXFP8_VARLEN_K_WARPS must be 1, 2, 4, or 8")
_VARLEN_K_PAIR_BLOCK_N = int(
    os.environ.get("SONICMOE_MXFP8_VARLEN_K_PAIR_BLOCK_N", "32")
)
if _VARLEN_K_PAIR_BLOCK_N not in (32, 64, 128):
    raise ValueError("SONICMOE_MXFP8_VARLEN_K_PAIR_BLOCK_N must be 32, 64, or 128")
_VARLEN_K_PAIR_NUM_WARPS = int(
    os.environ.get("SONICMOE_MXFP8_VARLEN_K_PAIR_WARPS", "4")
)
if _VARLEN_K_PAIR_NUM_WARPS not in (1, 2, 4, 8):
    raise ValueError("SONICMOE_MXFP8_VARLEN_K_PAIR_WARPS must be 1, 2, 4, or 8")
_VARLEN_DUAL_BLOCK_N = int(os.environ.get("SONICMOE_MXFP8_VARLEN_DUAL_BLOCK_N", "128"))
if _VARLEN_DUAL_BLOCK_N not in (32, 64, 128):
    raise ValueError("SONICMOE_MXFP8_VARLEN_DUAL_BLOCK_N must be 32, 64, or 128")
_VARLEN_DUAL_NUM_WARPS = int(os.environ.get("SONICMOE_MXFP8_VARLEN_DUAL_WARPS", "4"))
if _VARLEN_DUAL_NUM_WARPS not in (1, 2, 4, 8):
    raise ValueError("SONICMOE_MXFP8_VARLEN_DUAL_WARPS must be 1, 2, 4, or 8")
_GATHER_SF_BLOCK_M = int(os.environ.get("SONICMOE_MXFP8_GATHER_SF_BLOCK_M", "32"))
if _GATHER_SF_BLOCK_M not in (8, 16, 32, 64, 128):
    raise ValueError("SONICMOE_MXFP8_GATHER_SF_BLOCK_M must be 8, 16, 32, 64, or 128")
_FAST_BF16_QUANT = os.environ.get("SONICMOE_MXFP8_FAST_BF16_QUANT", "0") == "1"


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
def _rceil_e8m0_bf16_quant_scale(amax):
    """Return bit-exact BF16 RCEIL scale bytes and reciprocal powers of two."""
    bits = amax.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) & 0xFF
    carry = (bits & 0x7FFFFF) > 0x600000
    raw_scale_byte = exponent + carry.to(tl.int32) - 8
    scale_byte = tl.where(amax != 0.0, tl.maximum(raw_scale_byte, 1), 0)
    special = exponent == 0xFF
    scale_byte = tl.where(special, 0xFF, scale_byte)
    # E8M0 byte 255 reconstructs as +inf: finite / inf -> 0, while inf / inf
    # and NaN / inf stay NaN. Multiplication by zero has the same behavior.
    quant_bits = tl.where(special, 0, (254 - scale_byte) << 23)
    quant_scale = quant_bits.to(tl.float32, bitcast=True)
    return scale_byte, quant_scale


@triton.jit
def _find_expert_for_row(cu_ptr, row, E: tl.constexpr, N_ITERS: tl.constexpr):
    # ``row`` may be a scalar or a vector when one CTA quantizes several
    # adjacent routed rows.
    lo = row * 0
    hi = lo + E
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
    TOTAL_M: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    RK: tl.constexpr,
    HAS_GATHER: tl.constexpr,
    N_SEARCH_ITERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FAST_BF16: tl.constexpr,
):
    row_tile = tl.program_id(0)
    k_tile = tl.program_id(1)
    rows = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < TOTAL_M
    source_rows = (
        tl.load(gather_ptr + rows, mask=row_mask, other=0) if HAS_GATHER else rows
    )
    offsets = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
    values = tl.load(
        x_ptr + source_rows[:, None] * K + offsets[None, :],
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    groups = tl.reshape(values, [BLOCK_M * (BLOCK_K // 32), 32])
    amax = tl.max(tl.abs(groups), axis=1)
    if FAST_BF16:
        scale_byte, quant_scale = _rceil_e8m0_bf16_quant_scale(amax)
        scaled = groups * quant_scale[:, None]
    else:
        scale_byte, scale = _rceil_e8m0(amax)
        scaled = tl.maximum(
            tl.minimum(groups / scale[:, None], 448.0),
            -448.0,
        )
    tl.store(
        q_ptr + rows[:, None] * K + offsets[None, :],
        tl.reshape(scaled, [BLOCK_M, BLOCK_K]).to(tl.float8e4nv),
        mask=row_mask[:, None],
    )

    expert = _find_expert_for_row(cu_ptr, rows, E, N_SEARCH_ITERS)
    expert_start = tl.load(cu_ptr + expert)
    padded_row = (expert_start // 128 + expert) * 128 + rows - expert_start
    k_blocks = k_tile * (BLOCK_K // 32) + tl.arange(0, BLOCK_K // 32)
    rm = padded_row[:, None] // 128
    row_inner = padded_row[:, None] % 32
    row_outer = (padded_row[:, None] % 128) // 32
    rk = k_blocks[None, :] // 4
    k_inner = k_blocks[None, :] % 4
    sf_offset = (((rm * RK + rk) * 32 + row_inner) * 4 + row_outer) * 4 + k_inner
    tl.store(
        sf_ptr + sf_offset,
        tl.reshape(scale_byte, [BLOCK_M, BLOCK_K // 32]).to(tl.uint8),
        mask=row_mask[:, None],
    )


@triton.jit
def _physical_rowwise_linear_sf_kernel(
    x_ptr,
    q_ptr,
    linear_sf_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FAST_BF16: tl.constexpr,
):
    """Quantize physical rows once and retain their scales in a linear scratch."""
    row_tile = tl.program_id(0)
    k_tile = tl.program_id(1)
    rows = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    for group in tl.static_range(BLOCK_K // 32):
        offsets = k_tile * BLOCK_K + group * 32 + tl.arange(0, 32)
        values = tl.load(
            x_ptr + rows[:, None] * K + offsets[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        amax = tl.max(tl.abs(values), axis=1)
        if FAST_BF16:
            scale_byte, quant_scale = _rceil_e8m0_bf16_quant_scale(amax)
            scaled = values * quant_scale[:, None]
        else:
            scale_byte, scale = _rceil_e8m0(amax)
            scaled = tl.maximum(tl.minimum(values / scale[:, None], 448.0), -448.0)
        tl.store(
            q_ptr + rows[:, None] * K + offsets[None, :],
            scaled.to(tl.float8e4nv),
            mask=row_mask[:, None],
        )
        k_block = k_tile * (BLOCK_K // 32) + group
        tl.store(
            linear_sf_ptr + rows * (K // 32) + k_block,
            scale_byte.to(tl.uint8),
            mask=row_mask,
        )


@triton.jit
def _physical_rowwise_route_sf_kernel(
    x_ptr,
    reverse_ptr,
    q_ptr,
    sf_ptr,
    cu_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    RK: tl.constexpr,
    TOP_K: tl.constexpr,
    N_SEARCH_ITERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FAST_BF16: tl.constexpr,
):
    """Quantize physical tokens once and scatter scales to all routed rows."""
    row_tile = tl.program_id(0)
    k_tile = tl.program_id(1)
    rows = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    offsets = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
    values = tl.load(
        x_ptr + rows[:, None] * K + offsets[None, :],
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    groups = tl.reshape(values, [BLOCK_M * (BLOCK_K // 32), 32])
    amax = tl.max(tl.abs(groups), axis=1)
    if FAST_BF16:
        scale_byte, quant_scale = _rceil_e8m0_bf16_quant_scale(amax)
        scaled = groups * quant_scale[:, None]
    else:
        scale_byte, scale = _rceil_e8m0(amax)
        scaled = tl.maximum(tl.minimum(groups / scale[:, None], 448.0), -448.0)
    tl.store(
        q_ptr + rows[:, None] * K + offsets[None, :],
        tl.reshape(scaled, [BLOCK_M, BLOCK_K]).to(tl.float8e4nv),
        mask=row_mask[:, None],
    )

    k_blocks = k_tile * (BLOCK_K // 32) + tl.arange(0, BLOCK_K // 32)
    scale_byte = tl.reshape(scale_byte, [BLOCK_M, BLOCK_K // 32]).to(tl.uint8)
    for route_slot in tl.static_range(TOP_K):
        original_route = rows * TOP_K + route_slot
        routed_row = tl.load(reverse_ptr + original_route, mask=row_mask, other=0)
        expert = _find_expert_for_row(cu_ptr, routed_row, E, N_SEARCH_ITERS)
        expert_start = tl.load(cu_ptr + expert)
        padded_row = (expert_start // 128 + expert) * 128 + routed_row - expert_start
        rm = padded_row[:, None] // 128
        row_inner = padded_row[:, None] % 32
        row_outer = (padded_row[:, None] % 128) // 32
        rk = k_blocks[None, :] // 4
        k_inner = k_blocks[None, :] % 4
        sf_offset = (((rm * RK + rk) * 32 + row_inner) * 4 + row_outer) * 4 + k_inner
        tl.store(sf_ptr + sf_offset, scale_byte, mask=row_mask[:, None])


@triton.jit
def _gather_varlen_m_scale_kernel(
    linear_sf_ptr,
    gather_ptr,
    sf_ptr,
    cu_ptr,
    TOTAL_M: tl.constexpr,
    E: tl.constexpr,
    RK: tl.constexpr,
    N_SEARCH_ITERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Gather only scale bytes and store the expert-padded blocked layout."""
    row_tile = tl.program_id(0)
    rk = tl.program_id(1)
    rows = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < TOTAL_M
    source_rows = tl.load(gather_ptr + rows, mask=row_mask, other=0)
    k_inner = tl.arange(0, 4)
    scale_byte = tl.load(
        linear_sf_ptr + source_rows[:, None] * (RK * 4) + rk * 4 + k_inner[None, :],
        mask=row_mask[:, None],
        other=0,
    )

    expert = _find_expert_for_row(cu_ptr, rows, E, N_SEARCH_ITERS)
    expert_start = tl.load(cu_ptr + expert)
    padded_row = (expert_start // 128 + expert) * 128 + rows - expert_start
    rm = padded_row[:, None] // 128
    row_inner = padded_row[:, None] % 32
    row_outer = (padded_row[:, None] % 128) // 32
    sf_offset = (((rm * RK + rk) * 32 + row_inner) * 4 + row_outer) * 4 + k_inner
    tl.store(sf_ptr + sf_offset, scale_byte, mask=row_mask[:, None])


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
    FAST_BF16: tl.constexpr,
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
    if FAST_BF16:
        scale_byte, quant_scale = _rceil_e8m0_bf16_quant_scale(amax)
        scaled = values * quant_scale[None, :]
    else:
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
def _segmented_varlen_k_pair_kernel(
    x0_ptr,
    x1_ptr,
    q0_ptr,
    sf0_ptr,
    q1_ptr,
    sf1_ptr,
    cu_ptr,
    N0: tl.constexpr,
    N1: tl.constexpr,
    E: tl.constexpr,
    RK: tl.constexpr,
    N_SEARCH_ITERS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FAST_BF16: tl.constexpr,
):
    """Quantize two routed operands with one launch and shared metadata."""
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
    row_mask = block_valid & (rows < end)

    value0_mask = row_mask[:, None] & (cols[None, :] < N0)
    values0 = tl.load(
        x0_ptr + rows[:, None] * N0 + cols[None, :],
        mask=value0_mask,
        other=0.0,
    ).to(tl.float32)
    amax0 = tl.max(tl.abs(values0), axis=0)
    if FAST_BF16:
        scale0_byte, quant_scale0 = _rceil_e8m0_bf16_quant_scale(amax0)
        scaled0 = values0 * quant_scale0[None, :]
    else:
        scale0_byte, scale0 = _rceil_e8m0(amax0)
        scaled0 = tl.maximum(tl.minimum(values0 / scale0[None, :], 448.0), -448.0)
    tl.store(
        q0_ptr + rows[:, None] * N0 + cols[None, :],
        scaled0.to(tl.float8e4nv),
        mask=value0_mask,
    )

    value1_mask = row_mask[:, None] & (cols[None, :] < N1)
    values1 = tl.load(
        x1_ptr + rows[:, None] * N1 + cols[None, :],
        mask=value1_mask,
        other=0.0,
    ).to(tl.float32)
    amax1 = tl.max(tl.abs(values1), axis=0)
    if FAST_BF16:
        scale1_byte, quant_scale1 = _rceil_e8m0_bf16_quant_scale(amax1)
        scaled1 = values1 * quant_scale1[None, :]
    else:
        scale1_byte, scale1 = _rceil_e8m0(amax1)
        scaled1 = tl.maximum(tl.minimum(values1 / scale1[None, :], 448.0), -448.0)
    tl.store(
        q1_ptr + rows[:, None] * N1 + cols[None, :],
        scaled1.to(tl.float8e4nv),
        mask=value1_mask,
    )

    rm = cols // 128
    row_inner = cols % 32
    row_outer = (cols % 128) // 32
    rk = padded_block // 4
    k_inner = padded_block % 4
    sf_offset = (((rm * RK + rk) * 32 + row_inner) * 4 + row_outer) * 4 + k_inner
    tl.store(
        sf0_ptr + sf_offset,
        scale0_byte.to(tl.uint8),
        mask=block_valid & (cols < N0),
    )
    tl.store(
        sf1_ptr + sf_offset,
        scale1_byte.to(tl.uint8),
        mask=block_valid & (cols < N1),
    )


@triton.jit
def _varlen_dual_kernel(
    x_ptr,
    gather_ptr,
    row_q_ptr,
    row_sf_ptr,
    col_q_ptr,
    col_sf_ptr,
    cu_ptr,
    N: tl.constexpr,
    E: tl.constexpr,
    PADDED_BLOCKS: tl.constexpr,
    ROW_RK: tl.constexpr,
    COL_RK: tl.constexpr,
    HAS_GATHER: tl.constexpr,
    N_SEARCH_ITERS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FAST_BF16: tl.constexpr,
):
    """One routed 32xBLOCK_N read produces rowwise and segmented-K views."""
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
    row_valid = block_valid & (rows < end)
    value_mask = row_valid[:, None] & (cols[None, :] < N)
    source_rows = (
        tl.load(gather_ptr + rows, mask=row_valid, other=0) if HAS_GATHER else rows
    )
    offsets = rows[:, None] * N + cols[None, :]
    values = tl.load(
        x_ptr + source_rows[:, None] * N + cols[None, :],
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)

    # Rowwise view: 32 adjacent columns per scale, independently per row.
    row_groups = tl.reshape(values, [32 * (BLOCK_N // 32), 32])
    row_amax = tl.max(tl.abs(row_groups), axis=1)
    if FAST_BF16:
        row_scale_byte, row_quant_scale = _rceil_e8m0_bf16_quant_scale(row_amax)
        row_scaled = row_groups * row_quant_scale[:, None]
    else:
        row_scale_byte, row_scale = _rceil_e8m0(row_amax)
        row_scaled = tl.maximum(
            tl.minimum(row_groups / row_scale[:, None], 448.0),
            -448.0,
        )
    tl.store(
        row_q_ptr + offsets,
        tl.reshape(row_scaled, [32, BLOCK_N]).to(tl.float8e4nv),
        mask=value_mask,
    )

    padded_rows = padded_block * 32 + tl.arange(0, 32)
    row_k_blocks = n_tile * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)
    row_rm = padded_rows[:, None] // 128
    row_inner = padded_rows[:, None] % 32
    row_outer = (padded_rows[:, None] % 128) // 32
    row_rk = row_k_blocks[None, :] // 4
    row_k_inner = row_k_blocks[None, :] % 4
    row_sf_offset = (
        ((row_rm * ROW_RK + row_rk) * 32 + row_inner) * 4 + row_outer
    ) * 4 + row_k_inner
    row_scale_mask = row_valid[:, None] & (row_k_blocks[None, :] * 32 < N)
    tl.store(
        row_sf_ptr + row_sf_offset,
        tl.reshape(row_scale_byte, [32, BLOCK_N // 32]).to(tl.uint8),
        mask=row_scale_mask,
    )

    # Segmented-K view: 32 adjacent routed rows per scale, per column.
    col_amax = tl.max(tl.abs(values), axis=0)
    if FAST_BF16:
        col_scale_byte, col_quant_scale = _rceil_e8m0_bf16_quant_scale(col_amax)
        col_scaled = values * col_quant_scale[None, :]
    else:
        col_scale_byte, col_scale = _rceil_e8m0(col_amax)
        col_scaled = tl.maximum(
            tl.minimum(values / col_scale[None, :], 448.0),
            -448.0,
        )
    tl.store(col_q_ptr + offsets, col_scaled.to(tl.float8e4nv), mask=value_mask)

    col_rm = cols // 128
    col_inner = cols % 32
    col_outer = (cols % 128) // 32
    col_rk = padded_block // 4
    col_k_inner = padded_block % 4
    col_sf_offset = (
        ((col_rm * COL_RK + col_rk) * 32 + col_inner) * 4 + col_outer
    ) * 4 + col_k_inner
    tl.store(
        col_sf_ptr + col_sf_offset,
        col_scale_byte.to(tl.uint8),
        mask=block_valid & (cols < N),
    )


@triton.jit
def _varlen_iso32_dual_kernel(
    x_ptr,
    gather_ptr,
    q_ptr,
    row_sf_ptr,
    col_sf_ptr,
    cu_ptr,
    N: tl.constexpr,
    E: tl.constexpr,
    ROW_RK: tl.constexpr,
    COL_RK: tl.constexpr,
    HAS_GATHER: tl.constexpr,
    N_SEARCH_ITERS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FAST_BF16: tl.constexpr,
):
    """One 32x32 scale serves rowwise and segmented-K consumers.

    Adapted from PFCCLab/supersonic-moe commit 76b4f4f's public iso32
    quantizer. Expert boundaries remain explicit here, so a scale block never
    spans two variable-length experts.
    """
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
    row_valid = block_valid & (rows < end)
    value_mask = row_valid[:, None] & (cols[None, :] < N)
    source_rows = (
        tl.load(gather_ptr + rows, mask=row_valid, other=0) if HAS_GATHER else rows
    )
    values = tl.load(
        x_ptr + source_rows[:, None] * N + cols[None, :],
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)

    groups_per_tile: tl.constexpr = BLOCK_N // 32
    value_groups = tl.reshape(values, [32, groups_per_tile, 32])
    block_amax = tl.max(tl.max(tl.abs(value_groups), axis=2), axis=0)
    if FAST_BF16:
        scale_byte, quant_scale = _rceil_e8m0_bf16_quant_scale(block_amax)
        scaled = value_groups * quant_scale[None, :, None]
    else:
        scale_byte, scale = _rceil_e8m0(block_amax)
        scaled = tl.maximum(
            tl.minimum(value_groups / scale[None, :, None], 448.0),
            -448.0,
        )
    offsets = rows[:, None] * N + cols[None, :]
    tl.store(
        q_ptr + offsets,
        tl.reshape(scaled, [32, BLOCK_N]).to(tl.float8e4nv),
        mask=value_mask,
    )

    padded_rows = padded_block * 32 + tl.arange(0, 32)
    row_k_blocks = n_tile * groups_per_tile + tl.arange(0, groups_per_tile)
    row_rm = padded_rows[:, None] // 128
    row_inner = padded_rows[:, None] % 32
    row_outer = (padded_rows[:, None] % 128) // 32
    row_rk = row_k_blocks[None, :] // 4
    row_k_inner = row_k_blocks[None, :] % 4
    row_sf_offset = (
        ((row_rm * ROW_RK + row_rk) * 32 + row_inner) * 4 + row_outer
    ) * 4 + row_k_inner
    row_scale = scale_byte[None, :] + tl.zeros([32, groups_per_tile], dtype=tl.int32)
    tl.store(
        row_sf_ptr + row_sf_offset,
        row_scale.to(tl.uint8),
        mask=row_valid[:, None] & (row_k_blocks[None, :] * 32 < N),
    )

    group = tl.arange(0, groups_per_tile)
    col_group = tl.arange(0, BLOCK_N) // 32
    col_scale = tl.sum(
        (col_group[:, None] == group[None, :]) * scale_byte[None, :].to(tl.int32),
        axis=1,
    )
    col_rm = cols // 128
    col_inner = cols % 32
    col_outer = (cols % 128) // 32
    col_rk = padded_block // 4
    col_k_inner = padded_block % 4
    col_sf_offset = (
        ((col_rm * COL_RK + col_rk) * 32 + col_inner) * 4 + col_outer
    ) * 4 + col_k_inner
    tl.store(
        col_sf_ptr + col_sf_offset,
        col_scale.to(tl.uint8),
        mask=block_valid & (cols < N),
    )


@triton.jit
def _dense_rowwise_kernel(
    x_ptr,
    q_ptr,
    sf_ptr,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    RM: tl.constexpr,
    RK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch = tl.program_id(0)
    row_tile = tl.program_id(1)
    k_tile = tl.program_id(2)
    rows = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
    values = tl.load(x_ptr + (batch * ROWS + rows[:, None]) * K + offsets[None, :]).to(
        tl.float32
    )
    groups = tl.reshape(values, [BLOCK_M * (BLOCK_K // 32), 32])
    amax = tl.max(tl.abs(groups), axis=1)
    scale_byte, scale = _rceil_e8m0(amax)
    scaled = tl.maximum(tl.minimum(groups / scale[:, None], 448.0), -448.0)
    tl.store(
        q_ptr + (batch * ROWS + rows[:, None]) * K + offsets[None, :],
        tl.reshape(scaled, [BLOCK_M, BLOCK_K]).to(tl.float8e4nv),
    )

    k_blocks = k_tile * (BLOCK_K // 32) + tl.arange(0, BLOCK_K // 32)
    rm = rows[:, None] // 128
    row_inner = rows[:, None] % 32
    row_outer = (rows[:, None] % 128) // 32
    rk = k_blocks[None, :] // 4
    k_inner = k_blocks[None, :] % 4
    sf_offset = (
        ((((batch * RM + rm) * RK + rk) * 32 + row_inner) * 4 + row_outer) * 4
    ) + k_inner
    tl.store(
        sf_ptr + sf_offset,
        tl.reshape(scale_byte, [BLOCK_M, BLOCK_K // 32]).to(tl.uint8),
    )


@triton.jit
def _dense_rowwise_flat_kernel(
    x_ptr,
    q_ptr,
    sf_ptr,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    RM: tl.constexpr,
    RK: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    """Rowwise quantization from one physically contiguous weight tile."""
    block_start = tl.program_id(0) * BLOCK_ELEMS
    offsets = block_start + tl.arange(0, BLOCK_ELEMS)
    values = tl.load(x_ptr + offsets).to(tl.float32)
    groups = tl.reshape(values, [BLOCK_ELEMS // 32, 32])
    amax = tl.max(tl.abs(groups), axis=1)
    scale_byte, scale = _rceil_e8m0(amax)
    scaled = tl.maximum(tl.minimum(groups / scale[:, None], 448.0), -448.0)
    tl.store(
        q_ptr + offsets,
        tl.reshape(scaled, [BLOCK_ELEMS]).to(tl.float8e4nv),
    )

    groups_per_row = K // 32
    flat_groups = block_start // 32 + tl.arange(0, BLOCK_ELEMS // 32)
    flat_rows = flat_groups // groups_per_row
    batch = flat_rows // ROWS
    row = flat_rows % ROWS
    k_block = flat_groups % groups_per_row
    rm = row // 128
    row_inner = row % 32
    row_outer = (row % 128) // 32
    rk = k_block // 4
    k_inner = k_block % 4
    sf_offset = (
        ((((batch * RM + rm) * RK + rk) * 32 + row_inner) * 4 + row_outer) * 4
    ) + k_inner
    tl.store(sf_ptr + sf_offset, scale_byte.to(tl.uint8))


@triton.jit
def _dense_rowwise_sgd_kernel(
    weight_ptr,
    grad_ptr,
    q_ptr,
    sf_ptr,
    learning_rate,
    aux0_ptr,
    aux0_grad_ptr,
    aux1_ptr,
    aux1_grad_ptr,
    aux2_ptr,
    aux2_grad_ptr,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    RM: tl.constexpr,
    RK: tl.constexpr,
    AUX0_NUMEL: tl.constexpr,
    AUX1_NUMEL: tl.constexpr,
    AUX2_NUMEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    AUX_BLOCK: tl.constexpr,
):
    """Plain SGD update plus rowwise MXFP8 refresh from the rounded master."""
    batch = tl.program_id(0)
    row_tile = tl.program_id(1)
    k_tile = tl.program_id(2)
    rows = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
    value_offsets = (batch * ROWS + rows[:, None]) * K + offsets[None, :]
    values = tl.load(weight_ptr + value_offsets).to(tl.float32)
    gradients = tl.load(grad_ptr + value_offsets).to(tl.float32)

    # Match a BF16 parameter's in-place SGD semantics first.  Quantizing the
    # rounded value (rather than the transient FP32 expression) guarantees the
    # cache is exactly the view of the master weight used by later BF16 roles.
    updated_bf16 = (values - learning_rate * gradients).to(tl.bfloat16)
    tl.store(weight_ptr + value_offsets, updated_bf16)
    updated = updated_bf16.to(tl.float32)
    groups = tl.reshape(updated, [BLOCK_M * (BLOCK_K // 32), 32])
    amax = tl.max(tl.abs(groups), axis=1)
    scale_byte, scale = _rceil_e8m0(amax)
    scaled = tl.maximum(tl.minimum(groups / scale[:, None], 448.0), -448.0)
    tl.store(
        q_ptr + value_offsets,
        tl.reshape(scaled, [BLOCK_M, BLOCK_K]).to(tl.float8e4nv),
    )

    k_blocks = k_tile * (BLOCK_K // 32) + tl.arange(0, BLOCK_K // 32)
    rm = rows[:, None] // 128
    row_inner = rows[:, None] % 32
    row_outer = (rows[:, None] % 128) // 32
    rk = k_blocks[None, :] // 4
    k_inner = k_blocks[None, :] % 4
    sf_offset = (
        ((((batch * RM + rm) * RK + rk) * 32 + row_inner) * 4 + row_outer) * 4
    ) + k_inner
    tl.store(
        sf_ptr + sf_offset,
        tl.reshape(scale_byte, [BLOCK_M, BLOCK_K // 32]).to(tl.uint8),
    )

    # Fold the much smaller router/bias updates into otherwise independent
    # weight CTAs.  This removes the optimizer's extra foreach launch without
    # lengthening the weight grid or introducing cross-CTA synchronization.
    linear_pid = (batch * (ROWS // BLOCK_M) + row_tile) * (K // BLOCK_K) + k_tile
    aux_offsets = linear_pid * AUX_BLOCK + tl.arange(0, AUX_BLOCK)
    aux0_mask = aux_offsets < AUX0_NUMEL
    aux0_value = tl.load(aux0_ptr + aux_offsets, mask=aux0_mask).to(tl.float32)
    aux0_grad = tl.load(aux0_grad_ptr + aux_offsets, mask=aux0_mask).to(tl.float32)
    tl.store(
        aux0_ptr + aux_offsets,
        (aux0_value - learning_rate * aux0_grad).to(tl.bfloat16),
        mask=aux0_mask,
    )
    aux1_mask = aux_offsets < AUX1_NUMEL
    aux1_value = tl.load(aux1_ptr + aux_offsets, mask=aux1_mask).to(tl.float32)
    aux1_grad = tl.load(aux1_grad_ptr + aux_offsets, mask=aux1_mask).to(tl.float32)
    tl.store(
        aux1_ptr + aux_offsets,
        (aux1_value - learning_rate * aux1_grad).to(tl.bfloat16),
        mask=aux1_mask,
    )
    aux2_mask = aux_offsets < AUX2_NUMEL
    aux2_value = tl.load(aux2_ptr + aux_offsets, mask=aux2_mask).to(tl.float32)
    aux2_grad = tl.load(aux2_grad_ptr + aux_offsets, mask=aux2_mask).to(tl.float32)
    tl.store(
        aux2_ptr + aux_offsets,
        (aux2_value - learning_rate * aux2_grad).to(tl.bfloat16),
        mask=aux2_mask,
    )


@triton.jit
def _dense_rowwise_sgd_flat_body(
    pid,
    weight_ptr,
    grad_ptr,
    q_ptr,
    sf_ptr,
    learning_rate,
    aux0_ptr,
    aux0_grad_ptr,
    aux1_ptr,
    aux1_grad_ptr,
    aux2_ptr,
    aux2_grad_ptr,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    RM: tl.constexpr,
    RK: tl.constexpr,
    AUX0_NUMEL: tl.constexpr,
    AUX1_NUMEL: tl.constexpr,
    AUX2_NUMEL: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
    AUX_BLOCK: tl.constexpr,
):
    """Plain SGD plus rowwise quantization from a contiguous flat tile."""
    block_start = pid * BLOCK_ELEMS
    offsets = block_start + tl.arange(0, BLOCK_ELEMS)
    values = tl.load(weight_ptr + offsets).to(tl.float32)
    gradients = tl.load(grad_ptr + offsets).to(tl.float32)
    updated_bf16 = (values - learning_rate * gradients).to(tl.bfloat16)
    tl.store(weight_ptr + offsets, updated_bf16)

    groups = tl.reshape(updated_bf16.to(tl.float32), [BLOCK_ELEMS // 32, 32])
    amax = tl.max(tl.abs(groups), axis=1)
    scale_byte, scale = _rceil_e8m0(amax)
    scaled = tl.maximum(tl.minimum(groups / scale[:, None], 448.0), -448.0)
    tl.store(
        q_ptr + offsets,
        tl.reshape(scaled, [BLOCK_ELEMS]).to(tl.float8e4nv),
    )

    groups_per_row = K // 32
    flat_groups = block_start // 32 + tl.arange(0, BLOCK_ELEMS // 32)
    flat_rows = flat_groups // groups_per_row
    batch = flat_rows // ROWS
    row = flat_rows % ROWS
    k_block = flat_groups % groups_per_row
    rm = row // 128
    row_inner = row % 32
    row_outer = (row % 128) // 32
    rk = k_block // 4
    k_inner = k_block % 4
    sf_offset = (
        ((((batch * RM + rm) * RK + rk) * 32 + row_inner) * 4 + row_outer) * 4
    ) + k_inner
    tl.store(sf_ptr + sf_offset, scale_byte.to(tl.uint8))

    aux_offsets = pid * AUX_BLOCK + tl.arange(0, AUX_BLOCK)
    aux0_mask = aux_offsets < AUX0_NUMEL
    aux0_value = tl.load(aux0_ptr + aux_offsets, mask=aux0_mask).to(tl.float32)
    aux0_grad = tl.load(aux0_grad_ptr + aux_offsets, mask=aux0_mask).to(tl.float32)
    tl.store(
        aux0_ptr + aux_offsets,
        (aux0_value - learning_rate * aux0_grad).to(tl.bfloat16),
        mask=aux0_mask,
    )
    aux1_mask = aux_offsets < AUX1_NUMEL
    aux1_value = tl.load(aux1_ptr + aux_offsets, mask=aux1_mask).to(tl.float32)
    aux1_grad = tl.load(aux1_grad_ptr + aux_offsets, mask=aux1_mask).to(tl.float32)
    tl.store(
        aux1_ptr + aux_offsets,
        (aux1_value - learning_rate * aux1_grad).to(tl.bfloat16),
        mask=aux1_mask,
    )
    aux2_mask = aux_offsets < AUX2_NUMEL
    aux2_value = tl.load(aux2_ptr + aux_offsets, mask=aux2_mask).to(tl.float32)
    aux2_grad = tl.load(aux2_grad_ptr + aux_offsets, mask=aux2_mask).to(tl.float32)
    tl.store(
        aux2_ptr + aux_offsets,
        (aux2_value - learning_rate * aux2_grad).to(tl.bfloat16),
        mask=aux2_mask,
    )


@triton.jit
def _dense_rowwise_sgd_flat_kernel(
    weight_ptr,
    grad_ptr,
    q_ptr,
    sf_ptr,
    learning_rate,
    aux0_ptr,
    aux0_grad_ptr,
    aux1_ptr,
    aux1_grad_ptr,
    aux2_ptr,
    aux2_grad_ptr,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    RM: tl.constexpr,
    RK: tl.constexpr,
    AUX0_NUMEL: tl.constexpr,
    AUX1_NUMEL: tl.constexpr,
    AUX2_NUMEL: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
    AUX_BLOCK: tl.constexpr,
):
    _dense_rowwise_sgd_flat_body(
        tl.program_id(0),
        weight_ptr,
        grad_ptr,
        q_ptr,
        sf_ptr,
        learning_rate,
        aux0_ptr,
        aux0_grad_ptr,
        aux1_ptr,
        aux1_grad_ptr,
        aux2_ptr,
        aux2_grad_ptr,
        ROWS,
        K,
        RM,
        RK,
        AUX0_NUMEL,
        AUX1_NUMEL,
        AUX2_NUMEL,
        BLOCK_ELEMS,
        AUX_BLOCK,
    )


@triton.jit
def _dense_rowwise_sgd_flat_pair_kernel(
    weight0_ptr,
    grad0_ptr,
    q0_ptr,
    sf0_ptr,
    weight1_ptr,
    grad1_ptr,
    q1_ptr,
    sf1_ptr,
    learning_rate,
    aux0_ptr,
    aux0_grad_ptr,
    aux1_ptr,
    aux1_grad_ptr,
    aux2_ptr,
    aux2_grad_ptr,
    ROWS0: tl.constexpr,
    K0: tl.constexpr,
    RM0: tl.constexpr,
    RK0: tl.constexpr,
    ROWS1: tl.constexpr,
    K1: tl.constexpr,
    RM1: tl.constexpr,
    RK1: tl.constexpr,
    BLOCKS0: tl.constexpr,
    AUX0_NUMEL: tl.constexpr,
    AUX1_NUMEL: tl.constexpr,
    AUX2_NUMEL: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
    AUX_BLOCK: tl.constexpr,
):
    """Update and row-quantize both expert weights with one launch."""
    pid = tl.program_id(0)
    if pid < BLOCKS0:
        _dense_rowwise_sgd_flat_body(
            pid,
            weight0_ptr,
            grad0_ptr,
            q0_ptr,
            sf0_ptr,
            learning_rate,
            aux0_ptr,
            aux0_grad_ptr,
            aux1_ptr,
            aux1_grad_ptr,
            aux2_ptr,
            aux2_grad_ptr,
            ROWS0,
            K0,
            RM0,
            RK0,
            AUX0_NUMEL,
            AUX1_NUMEL,
            AUX2_NUMEL,
            BLOCK_ELEMS,
            AUX_BLOCK,
        )
    else:
        _dense_rowwise_sgd_flat_body(
            pid - BLOCKS0,
            weight1_ptr,
            grad1_ptr,
            q1_ptr,
            sf1_ptr,
            learning_rate,
            weight1_ptr,
            grad1_ptr,
            weight1_ptr,
            grad1_ptr,
            weight1_ptr,
            grad1_ptr,
            ROWS1,
            K1,
            RM1,
            RK1,
            0,
            0,
            0,
            BLOCK_ELEMS,
            AUX_BLOCK,
        )


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


@triton.jit
def _dense_dual_kernel(
    x_ptr,
    row_q_ptr,
    row_sf_ptr,
    col_q_ptr,
    col_sf_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    ROW_RM: tl.constexpr,
    ROW_RK: tl.constexpr,
    COL_RN: tl.constexpr,
    COL_RK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Read one 32xBLOCK_N tile and emit both MXFP8 scale directions."""
    batch = tl.program_id(0)
    row_block = tl.program_id(1)
    col_tile = tl.program_id(2)
    rows = row_block * 32 + tl.arange(0, 32)
    cols = col_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = (batch * ROWS + rows[:, None]) * COLS + cols[None, :]
    values = tl.load(x_ptr + offsets).to(tl.float32)

    # Forward view: one E8M0 scale for every 32 adjacent columns of a row.
    row_groups = tl.reshape(values, [32 * (BLOCK_N // 32), 32])
    row_amax = tl.max(tl.abs(row_groups), axis=1)
    row_scale_byte, row_scale = _rceil_e8m0(row_amax)
    row_scaled = tl.maximum(
        tl.minimum(row_groups / row_scale[:, None], 448.0),
        -448.0,
    )
    tl.store(
        row_q_ptr + offsets,
        tl.reshape(row_scaled, [32, BLOCK_N]).to(tl.float8e4nv),
    )

    row_k_blocks = col_tile * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)
    row_rm = rows[:, None] // 128
    row_inner = rows[:, None] % 32
    row_outer = (rows[:, None] % 128) // 32
    row_rk = row_k_blocks[None, :] // 4
    row_k_inner = row_k_blocks[None, :] % 4
    row_sf_offset = (
        (((batch * ROW_RM + row_rm) * ROW_RK + row_rk) * 32 + row_inner) * 4 + row_outer
    ) * 4 + row_k_inner
    tl.store(
        row_sf_ptr + row_sf_offset,
        tl.reshape(row_scale_byte, [32, BLOCK_N // 32]).to(tl.uint8),
    )

    # Dgrad view: one E8M0 scale for every 32 adjacent rows of a column.
    col_amax = tl.max(tl.abs(values), axis=0)
    col_scale_byte, col_scale = _rceil_e8m0(col_amax)
    col_scaled = tl.maximum(
        tl.minimum(values / col_scale[None, :], 448.0),
        -448.0,
    )
    tl.store(col_q_ptr + offsets, col_scaled.to(tl.float8e4nv))

    col_rn = cols // 128
    col_inner = cols % 32
    col_outer = (cols % 128) // 32
    col_rk = row_block // 4
    col_k_inner = row_block % 4
    col_sf_offset = (
        (((batch * COL_RN + col_rn) * COL_RK + col_rk) * 32 + col_inner) * 4 + col_outer
    ) * 4 + col_k_inner
    tl.store(col_sf_ptr + col_sf_offset, col_scale_byte.to(tl.uint8))


@triton.jit
def _dense_dual_sgd_kernel(
    weight_ptr,
    grad_ptr,
    row_q_ptr,
    row_sf_ptr,
    col_q_ptr,
    col_sf_ptr,
    learning_rate,
    aux0_ptr,
    aux0_grad_ptr,
    aux1_ptr,
    aux1_grad_ptr,
    aux2_ptr,
    aux2_grad_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    ROW_RM: tl.constexpr,
    ROW_RK: tl.constexpr,
    COL_RN: tl.constexpr,
    COL_RK: tl.constexpr,
    AUX0_NUMEL: tl.constexpr,
    AUX1_NUMEL: tl.constexpr,
    AUX2_NUMEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    AUX_BLOCK: tl.constexpr,
):
    """BF16 SGD plus both weight scale directions from one rounded tile."""
    batch = tl.program_id(0)
    row_block = tl.program_id(1)
    col_tile = tl.program_id(2)
    rows = row_block * 32 + tl.arange(0, 32)
    cols = col_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = (batch * ROWS + rows[:, None]) * COLS + cols[None, :]
    values = tl.load(weight_ptr + offsets).to(tl.float32)
    gradients = tl.load(grad_ptr + offsets).to(tl.float32)
    updated_bf16 = (values - learning_rate * gradients).to(tl.bfloat16)
    tl.store(weight_ptr + offsets, updated_bf16)
    updated = updated_bf16.to(tl.float32)

    row_groups = tl.reshape(updated, [32 * (BLOCK_N // 32), 32])
    row_amax = tl.max(tl.abs(row_groups), axis=1)
    row_scale_byte, row_scale = _rceil_e8m0(row_amax)
    row_scaled = tl.maximum(
        tl.minimum(row_groups / row_scale[:, None], 448.0),
        -448.0,
    )
    tl.store(
        row_q_ptr + offsets,
        tl.reshape(row_scaled, [32, BLOCK_N]).to(tl.float8e4nv),
    )
    row_k_blocks = col_tile * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)
    row_rm = rows[:, None] // 128
    row_inner = rows[:, None] % 32
    row_outer = (rows[:, None] % 128) // 32
    row_rk = row_k_blocks[None, :] // 4
    row_k_inner = row_k_blocks[None, :] % 4
    row_sf_offset = (
        (((batch * ROW_RM + row_rm) * ROW_RK + row_rk) * 32 + row_inner) * 4 + row_outer
    ) * 4 + row_k_inner
    tl.store(
        row_sf_ptr + row_sf_offset,
        tl.reshape(row_scale_byte, [32, BLOCK_N // 32]).to(tl.uint8),
    )

    col_amax = tl.max(tl.abs(updated), axis=0)
    col_scale_byte, col_scale = _rceil_e8m0(col_amax)
    col_scaled = tl.maximum(
        tl.minimum(updated / col_scale[None, :], 448.0),
        -448.0,
    )
    tl.store(col_q_ptr + offsets, col_scaled.to(tl.float8e4nv))
    col_rn = cols // 128
    col_inner = cols % 32
    col_outer = (cols % 128) // 32
    col_rk = row_block // 4
    col_k_inner = row_block % 4
    col_sf_offset = (
        (((batch * COL_RN + col_rn) * COL_RK + col_rk) * 32 + col_inner) * 4 + col_outer
    ) * 4 + col_k_inner
    tl.store(col_sf_ptr + col_sf_offset, col_scale_byte.to(tl.uint8))

    linear_pid = (
        batch * (ROWS // 32) * (COLS // BLOCK_N)
        + row_block * (COLS // BLOCK_N)
        + col_tile
    )
    aux_offsets = linear_pid * AUX_BLOCK + tl.arange(0, AUX_BLOCK)
    aux0_mask = aux_offsets < AUX0_NUMEL
    aux0_value = tl.load(aux0_ptr + aux_offsets, mask=aux0_mask).to(tl.float32)
    aux0_grad = tl.load(aux0_grad_ptr + aux_offsets, mask=aux0_mask).to(tl.float32)
    tl.store(
        aux0_ptr + aux_offsets,
        (aux0_value - learning_rate * aux0_grad).to(tl.bfloat16),
        mask=aux0_mask,
    )
    aux1_mask = aux_offsets < AUX1_NUMEL
    aux1_value = tl.load(aux1_ptr + aux_offsets, mask=aux1_mask).to(tl.float32)
    aux1_grad = tl.load(aux1_grad_ptr + aux_offsets, mask=aux1_mask).to(tl.float32)
    tl.store(
        aux1_ptr + aux_offsets,
        (aux1_value - learning_rate * aux1_grad).to(tl.bfloat16),
        mask=aux1_mask,
    )
    aux2_mask = aux_offsets < AUX2_NUMEL
    aux2_value = tl.load(aux2_ptr + aux_offsets, mask=aux2_mask).to(tl.float32)
    aux2_grad = tl.load(aux2_grad_ptr + aux_offsets, mask=aux2_mask).to(tl.float32)
    tl.store(
        aux2_ptr + aux_offsets,
        (aux2_value - learning_rate * aux2_grad).to(tl.bfloat16),
        mask=aux2_mask,
    )


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
        flat_block = int(os.environ.get("SONICMOE_MXFP8_WEIGHT_FLAT_BLOCK", "0"))
        block_m = int(os.environ.get("SONICMOE_MXFP8_WEIGHT_BLOCK_M", "32"))
        if block_m not in (1, 2, 4, 8, 16, 32, 64):
            raise ValueError(
                "SONICMOE_MXFP8_WEIGHT_BLOCK_M must be 1, 2, 4, 8, 16, 32, or 64"
            )
        num_warps = int(os.environ.get("SONICMOE_MXFP8_WEIGHT_WARPS", "4"))
        if num_warps not in (4, 8):
            raise ValueError("SONICMOE_MXFP8_WEIGHT_WARPS must be 4 or 8")
        block_k = int(os.environ.get("SONICMOE_MXFP8_WEIGHT_BLOCK_K", "128"))
        if block_k not in (128, 256, 512) or cols % block_k:
            raise ValueError(
                "SONICMOE_MXFP8_WEIGHT_BLOCK_K must be 128, 256, or 512 "
                "and divide the weight K dimension"
            )
        if flat_block:
            if flat_block not in (1024, 2048, 4096, 8192):
                raise ValueError(
                    "SONICMOE_MXFP8_WEIGHT_FLAT_BLOCK must be 0, 1024, "
                    "2048, 4096, or 8192"
                )
            total = x.numel()
            if total % flat_block:
                raise ValueError("flat weight block must divide the weight numel")
            _dense_rowwise_flat_kernel[(total // flat_block,)](
                x,
                qdata,
                scale.view(torch.uint8),
                ROWS=rows,
                K=cols,
                RM=rm,
                RK=rk,
                BLOCK_ELEMS=flat_block,
                num_warps=num_warps,
            )
        else:
            _dense_rowwise_kernel[(experts, rows // block_m, cols // block_k)](
                x,
                qdata,
                scale.view(torch.uint8),
                ROWS=rows,
                K=cols,
                RM=rm,
                RK=rk,
                BLOCK_M=block_m,
                BLOCK_K=block_k,
                num_warps=num_warps,
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


def launch_sgd_update_and_quantize_mxfp8_weight(
    weight: torch.Tensor,
    grad: torch.Tensor,
    learning_rate: float,
    *,
    qdata_out: torch.Tensor,
    scale_out: torch.Tensor,
    aux_updates: tuple[tuple[torch.Tensor, torch.Tensor], ...] = (),
) -> None:
    """Unchecked hot-path launcher used by the versioned workspace."""
    experts, rows, cols = weight.shape
    checked_aux = [*aux_updates]
    while len(checked_aux) < 3:
        checked_aux.append((weight, grad))
    block_m = int(os.environ.get("SONICMOE_MXFP8_WEIGHT_BLOCK_M", "32"))
    block_k = int(os.environ.get("SONICMOE_MXFP8_WEIGHT_BLOCK_K", "128"))
    num_warps = int(os.environ.get("SONICMOE_MXFP8_WEIGHT_WARPS", "4"))
    flat_block = int(os.environ.get("SONICMOE_MXFP8_SGD_FLAT_BLOCK", "4096"))
    sgd_num_warps = int(os.environ.get("SONICMOE_MXFP8_SGD_WARPS", "4"))
    aux_block = 256
    kernel_args = (
        weight,
        grad,
        qdata_out,
        scale_out.view(torch.uint8),
        float(learning_rate),
        checked_aux[0][0],
        checked_aux[0][1],
        checked_aux[1][0],
        checked_aux[1][1],
        checked_aux[2][0],
        checked_aux[2][1],
    )
    aux0_numel = aux_updates[0][0].numel() if len(aux_updates) > 0 else 0
    aux1_numel = aux_updates[1][0].numel() if len(aux_updates) > 1 else 0
    aux2_numel = aux_updates[2][0].numel() if len(aux_updates) > 2 else 0
    if flat_block:
        _dense_rowwise_sgd_flat_kernel[(weight.numel() // flat_block,)](
            *kernel_args,
            ROWS=rows,
            K=cols,
            RM=rows // _SF_ATOM,
            RK=cols // _SF_ATOM,
            AUX0_NUMEL=aux0_numel,
            AUX1_NUMEL=aux1_numel,
            AUX2_NUMEL=aux2_numel,
            BLOCK_ELEMS=flat_block,
            AUX_BLOCK=aux_block,
            num_warps=sgd_num_warps,
        )
    else:
        _dense_rowwise_sgd_kernel[(experts, rows // block_m, cols // block_k)](
            *kernel_args,
            ROWS=rows,
            K=cols,
            RM=rows // _SF_ATOM,
            RK=cols // _SF_ATOM,
            AUX0_NUMEL=aux0_numel,
            AUX1_NUMEL=aux1_numel,
            AUX2_NUMEL=aux2_numel,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            AUX_BLOCK=aux_block,
            num_warps=num_warps,
        )


def launch_sgd_update_and_quantize_mxfp8_weight_dual(
    weight: torch.Tensor,
    grad: torch.Tensor,
    learning_rate: float,
    *,
    row_qdata_out: torch.Tensor,
    row_scale_out: torch.Tensor,
    col_qdata_out: torch.Tensor,
    col_scale_out: torch.Tensor,
    aux_updates: tuple[tuple[torch.Tensor, torch.Tensor], ...] = (),
) -> None:
    """Unchecked hot-path launcher for SGD plus both cached weight views."""
    experts, rows, cols = weight.shape
    checked_aux = [*aux_updates]
    while len(checked_aux) < 3:
        checked_aux.append((weight, grad))
    block_n = int(os.environ.get("SONICMOE_MXFP8_SGD_DUAL_BLOCK_N", "128"))
    num_warps = int(os.environ.get("SONICMOE_MXFP8_SGD_DUAL_WARPS", "4"))
    aux_block = 256
    _dense_dual_sgd_kernel[(experts, rows // 32, cols // block_n)](
        weight,
        grad,
        row_qdata_out,
        row_scale_out.view(torch.uint8),
        col_qdata_out,
        col_scale_out.view(torch.uint8),
        float(learning_rate),
        checked_aux[0][0],
        checked_aux[0][1],
        checked_aux[1][0],
        checked_aux[1][1],
        checked_aux[2][0],
        checked_aux[2][1],
        ROWS=rows,
        COLS=cols,
        ROW_RM=rows // _SF_ATOM,
        ROW_RK=cols // _SF_ATOM,
        COL_RN=cols // _SF_ATOM,
        COL_RK=rows // _SF_ATOM,
        AUX0_NUMEL=aux_updates[0][0].numel() if len(aux_updates) > 0 else 0,
        AUX1_NUMEL=aux_updates[1][0].numel() if len(aux_updates) > 1 else 0,
        AUX2_NUMEL=aux_updates[2][0].numel() if len(aux_updates) > 2 else 0,
        BLOCK_N=block_n,
        AUX_BLOCK=aux_block,
        num_warps=num_warps,
    )


def sgd_update_and_quantize_mxfp8_weight(
    weight: torch.Tensor,
    grad: torch.Tensor,
    learning_rate: float,
    *,
    qdata_out: torch.Tensor | None = None,
    scale_out: torch.Tensor | None = None,
    aux_updates: tuple[tuple[torch.Tensor, torch.Tensor], ...] = (),
) -> BlockScaledOperand:
    """Apply plain SGD and emit the rowwise MXFP8 view in the same pass.

    ``weight`` and ``grad`` use the contiguous ``(E, M, K)`` master layout.
    Momentum, weight decay, dampening and Nesterov are intentionally outside
    this primitive; :class:`sonicmoe.Mxfp8SGD` validates those semantics.
    """
    if (
        weight.ndim != 3
        or weight.dtype != torch.bfloat16
        or not weight.is_cuda
        or not weight.is_contiguous()
    ):
        raise ValueError(
            "weight must be a contiguous three-dimensional CUDA BF16 tensor"
        )
    if grad.shape != weight.shape or grad.dtype != weight.dtype:
        raise ValueError("grad must match the BF16 master weight")
    if not grad.is_cuda or not grad.is_contiguous() or grad.device != weight.device:
        raise ValueError("grad must be contiguous and colocated with weight")
    experts, rows, cols = weight.shape
    if rows % _SF_ATOM or cols % _SF_ATOM:
        raise ValueError("both weight matrix dimensions must be divisible by 128")
    if not isinstance(learning_rate, (float, int)) or learning_rate < 0:
        raise ValueError("learning_rate must be a non-negative Python number")
    if len(aux_updates) > 3:
        raise ValueError("at most three auxiliary SGD tensors are supported")
    for aux_weight, aux_grad in aux_updates:
        if (
            aux_weight.dtype != torch.bfloat16
            or not aux_weight.is_cuda
            or not aux_weight.is_contiguous()
            or aux_weight.device != weight.device
        ):
            raise ValueError("auxiliary weights must be contiguous colocated BF16")
        if (
            aux_grad.shape != aux_weight.shape
            or aux_grad.dtype != aux_weight.dtype
            or not aux_grad.is_cuda
            or not aux_grad.is_contiguous()
            or aux_grad.device != weight.device
        ):
            raise ValueError("auxiliary gradients must match their weights")

    q_shape = tuple(weight.shape)
    sf_shape = (
        experts,
        rows // _SF_ATOM,
        cols // _SF_ATOM,
        32,
        4,
        4,
    )
    qdata = (
        torch.empty(q_shape, dtype=MXFP8_E4M3.qdata_dtype, device=weight.device)
        if qdata_out is None
        else qdata_out
    )
    scale = (
        torch.empty(sf_shape, dtype=MXFP8_E4M3.scale_dtype, device=weight.device)
        if scale_out is None
        else scale_out
    )
    _check_outputs(qdata, scale, q_shape, sf_shape)
    block_m = int(os.environ.get("SONICMOE_MXFP8_WEIGHT_BLOCK_M", "32"))
    if block_m not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError(
            "SONICMOE_MXFP8_WEIGHT_BLOCK_M must be 1, 2, 4, 8, 16, 32, or 64"
        )
    num_warps = int(os.environ.get("SONICMOE_MXFP8_WEIGHT_WARPS", "4"))
    if num_warps not in (4, 8):
        raise ValueError("SONICMOE_MXFP8_WEIGHT_WARPS must be 4 or 8")
    block_k = int(os.environ.get("SONICMOE_MXFP8_WEIGHT_BLOCK_K", "128"))
    if block_k not in (128, 256, 512) or cols % block_k:
        raise ValueError(
            "SONICMOE_MXFP8_WEIGHT_BLOCK_K must be 128, 256, or 512 "
            "and divide the weight K dimension"
        )
    flat_block = int(os.environ.get("SONICMOE_MXFP8_SGD_FLAT_BLOCK", "4096"))
    if flat_block not in (0, 1024, 2048, 4096, 8192):
        raise ValueError(
            "SONICMOE_MXFP8_SGD_FLAT_BLOCK must be 0, 1024, 2048, 4096, or 8192"
        )
    if flat_block and weight.numel() % flat_block:
        raise ValueError("flat weight block must divide the weight numel")
    sgd_num_warps = int(os.environ.get("SONICMOE_MXFP8_SGD_WARPS", "4"))
    if sgd_num_warps not in (4, 8):
        raise ValueError("SONICMOE_MXFP8_SGD_WARPS must be 4 or 8")
    aux_block = 256
    grid_size = (
        weight.numel() // flat_block
        if flat_block
        else experts * (rows // block_m) * (cols // block_k)
    )
    if any(
        (aux_weight.numel() + aux_block - 1) // aux_block > grid_size
        for aux_weight, _ in aux_updates
    ):
        raise ValueError("expert weight grid is too small for auxiliary updates")
    launch_sgd_update_and_quantize_mxfp8_weight(
        weight,
        grad,
        learning_rate,
        qdata_out=qdata,
        scale_out=scale,
        aux_updates=aux_updates,
    )
    # The raw Triton store bypasses PyTorch's in-place dispatcher. Advance
    # every mutated tensor's version so versioned caches cannot observe stale
    # values when this public primitive is used directly.
    torch.autograd.graph.increment_version(
        (weight, *(aux_weight for aux_weight, _ in aux_updates))
    )
    return BlockScaledOperand.from_parts(
        qdata,
        scale,
        MXFP8_E4M3,
        orig_dtype=weight.dtype,
        quant_dim=-1,
    )


def sgd_update_and_quantize_mxfp8_weight_dual(
    weight: torch.Tensor,
    grad: torch.Tensor,
    learning_rate: float,
    *,
    row_qdata_out: torch.Tensor | None = None,
    row_scale_out: torch.Tensor | None = None,
    col_qdata_out: torch.Tensor | None = None,
    col_scale_out: torch.Tensor | None = None,
    aux_updates: tuple[tuple[torch.Tensor, torch.Tensor], ...] = (),
) -> tuple[BlockScaledOperand, BlockScaledOperand]:
    """Apply BF16 SGD and refresh both MXFP8 weight layouts in one pass."""
    if (
        weight.ndim != 3
        or weight.dtype != torch.bfloat16
        or not weight.is_cuda
        or not weight.is_contiguous()
    ):
        raise ValueError(
            "weight must be a contiguous three-dimensional CUDA BF16 tensor"
        )
    if (
        grad.shape != weight.shape
        or grad.dtype != weight.dtype
        or not grad.is_cuda
        or not grad.is_contiguous()
        or grad.device != weight.device
    ):
        raise ValueError("grad must be contiguous and match the BF16 master weight")
    experts, rows, cols = weight.shape
    if rows % _SF_ATOM or cols % _SF_ATOM:
        raise ValueError("both weight matrix dimensions must be divisible by 128")
    if not isinstance(learning_rate, (float, int)) or learning_rate < 0:
        raise ValueError("learning_rate must be a non-negative Python number")
    if len(aux_updates) > 3:
        raise ValueError("at most three auxiliary SGD tensors are supported")
    for aux_weight, aux_grad in aux_updates:
        if (
            aux_weight.dtype != torch.bfloat16
            or not aux_weight.is_cuda
            or not aux_weight.is_contiguous()
            or aux_weight.device != weight.device
            or aux_grad.shape != aux_weight.shape
            or aux_grad.dtype != aux_weight.dtype
            or not aux_grad.is_cuda
            or not aux_grad.is_contiguous()
            or aux_grad.device != weight.device
        ):
            raise ValueError(
                "auxiliary weights and gradients must be contiguous colocated BF16"
            )

    q_shape = tuple(weight.shape)
    row_sf_shape = (
        experts,
        rows // _SF_ATOM,
        cols // _SF_ATOM,
        32,
        4,
        4,
    )
    col_sf_shape = (
        experts,
        cols // _SF_ATOM,
        rows // _SF_ATOM,
        32,
        4,
        4,
    )
    row_qdata = (
        torch.empty(q_shape, dtype=MXFP8_E4M3.qdata_dtype, device=weight.device)
        if row_qdata_out is None
        else row_qdata_out
    )
    row_scale = (
        torch.empty(row_sf_shape, dtype=MXFP8_E4M3.scale_dtype, device=weight.device)
        if row_scale_out is None
        else row_scale_out
    )
    col_qdata = (
        torch.empty(q_shape, dtype=MXFP8_E4M3.qdata_dtype, device=weight.device)
        if col_qdata_out is None
        else col_qdata_out
    )
    col_scale = (
        torch.empty(col_sf_shape, dtype=MXFP8_E4M3.scale_dtype, device=weight.device)
        if col_scale_out is None
        else col_scale_out
    )
    _check_outputs(row_qdata, row_scale, q_shape, row_sf_shape)
    _check_outputs(col_qdata, col_scale, q_shape, col_sf_shape)
    if row_qdata.data_ptr() == col_qdata.data_ptr():
        raise ValueError("rowwise and dim-0 qdata outputs must not alias")

    block_n = int(os.environ.get("SONICMOE_MXFP8_SGD_DUAL_BLOCK_N", "128"))
    if block_n not in (32, 64, 128) or cols % block_n:
        raise ValueError(
            "SONICMOE_MXFP8_SGD_DUAL_BLOCK_N must be 32, 64, or 128 "
            "and divide the weight K dimension"
        )
    num_warps = int(os.environ.get("SONICMOE_MXFP8_SGD_DUAL_WARPS", "4"))
    if num_warps not in (4, 8):
        raise ValueError("SONICMOE_MXFP8_SGD_DUAL_WARPS must be 4 or 8")
    grid_size = experts * (rows // 32) * (cols // block_n)
    if any(
        (aux_weight.numel() + 255) // 256 > grid_size for aux_weight, _ in aux_updates
    ):
        raise ValueError("expert weight grid is too small for auxiliary updates")
    launch_sgd_update_and_quantize_mxfp8_weight_dual(
        weight,
        grad,
        learning_rate,
        row_qdata_out=row_qdata,
        row_scale_out=row_scale,
        col_qdata_out=col_qdata,
        col_scale_out=col_scale,
        aux_updates=aux_updates,
    )
    torch.autograd.graph.increment_version(
        (weight, *(aux_weight for aux_weight, _ in aux_updates))
    )
    return (
        BlockScaledOperand.from_parts(
            row_qdata,
            row_scale,
            MXFP8_E4M3,
            orig_dtype=weight.dtype,
            quant_dim=-1,
        ),
        BlockScaledOperand.from_parts(
            col_qdata,
            col_scale,
            MXFP8_E4M3,
            orig_dtype=weight.dtype,
            quant_dim=-2,
        ),
    )


def quantize_mxfp8_weight_dual(
    x: torch.Tensor,
    *,
    row_qdata_out: torch.Tensor | None = None,
    row_scale_out: torch.Tensor | None = None,
    col_qdata_out: torch.Tensor | None = None,
    col_scale_out: torch.Tensor | None = None,
) -> tuple[BlockScaledOperand, BlockScaledOperand]:
    """Create rowwise and dim-0 MXFP8 weight views with one source read.

    The two quantized value tensors remain distinct because their E8M0 scale
    vectors use different reduction axes.  A single kernel emits both views,
    which removes one launch and the second BF16 master-weight read.
    """
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

    q_shape = tuple(x.shape)
    row_sf_shape = (
        experts,
        rows // _SF_ATOM,
        cols // _SF_ATOM,
        32,
        4,
        4,
    )
    col_sf_shape = (
        experts,
        cols // _SF_ATOM,
        rows // _SF_ATOM,
        32,
        4,
        4,
    )
    row_qdata = (
        torch.empty(q_shape, dtype=MXFP8_E4M3.qdata_dtype, device=x.device)
        if row_qdata_out is None
        else row_qdata_out
    )
    row_scale = (
        torch.empty(row_sf_shape, dtype=MXFP8_E4M3.scale_dtype, device=x.device)
        if row_scale_out is None
        else row_scale_out
    )
    col_qdata = (
        torch.empty(q_shape, dtype=MXFP8_E4M3.qdata_dtype, device=x.device)
        if col_qdata_out is None
        else col_qdata_out
    )
    col_scale = (
        torch.empty(col_sf_shape, dtype=MXFP8_E4M3.scale_dtype, device=x.device)
        if col_scale_out is None
        else col_scale_out
    )
    _check_outputs(row_qdata, row_scale, q_shape, row_sf_shape)
    _check_outputs(col_qdata, col_scale, q_shape, col_sf_shape)
    if row_qdata.data_ptr() == col_qdata.data_ptr():
        raise ValueError("rowwise and dim-0 qdata outputs must not alias")

    block_n = int(os.environ.get("SONICMOE_MXFP8_WEIGHT_DUAL_BLOCK_N", "128"))
    if block_n not in (32, 64, 128):
        raise ValueError("SONICMOE_MXFP8_WEIGHT_DUAL_BLOCK_N must be 32, 64, or 128")
    num_warps = int(os.environ.get("SONICMOE_MXFP8_WEIGHT_DUAL_WARPS", "4"))
    if num_warps not in (4, 8):
        raise ValueError("SONICMOE_MXFP8_WEIGHT_DUAL_WARPS must be 4 or 8")
    _dense_dual_kernel[(experts, rows // _SF_VEC, cols // block_n)](
        x,
        row_qdata,
        row_scale.view(torch.uint8),
        col_qdata,
        col_scale.view(torch.uint8),
        ROWS=rows,
        COLS=cols,
        ROW_RM=rows // _SF_ATOM,
        ROW_RK=cols // _SF_ATOM,
        COL_RN=cols // _SF_ATOM,
        COL_RK=rows // _SF_ATOM,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return (
        BlockScaledOperand.from_parts(
            row_qdata,
            row_scale,
            MXFP8_E4M3,
            orig_dtype=x.dtype,
            quant_dim=-1,
        ),
        BlockScaledOperand.from_parts(
            col_qdata,
            col_scale,
            MXFP8_E4M3,
            orig_dtype=x.dtype,
            quant_dim=-2,
        ),
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
    _rowwise_varlen_m_kernel[(triton.cdiv(total, _VARLEN_BLOCK_M), k // _SF_ATOM)](
        x,
        gather_idx,
        qdata,
        scale.view(torch.uint8),
        cu_seqlens_m,
        TOTAL_M=total,
        K=k,
        E=experts,
        RK=rk,
        HAS_GATHER=gather_idx is not None,
        N_SEARCH_ITERS=experts.bit_length(),
        BLOCK_M=_VARLEN_BLOCK_M,
        BLOCK_K=_SF_ATOM,
        FAST_BF16=_FAST_BF16_QUANT and x.dtype == torch.bfloat16,
        num_warps=4,
    )
    return BlockScaledOperand.from_parts(qdata, scale, MXFP8_E4M3, orig_dtype=x.dtype)


def quantize_mxfp8_gather_varlen_m(
    x: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    gather_idx: torch.Tensor,
    *,
    reverse_idx: torch.Tensor | None = None,
    top_k: int | None = None,
    qdata_out: torch.Tensor | None = None,
    scale_out: torch.Tensor | None = None,
    linear_scale_out: torch.Tensor | None = None,
) -> BlockScaledOperand:
    """Quantize ``x=(T, K)`` once and gather only scales into routed order.

    The returned qdata remains ``(T, K)``. Its scale buffer describes the
    logical ``gather_idx`` rows, so consumers must receive the same gather
    indices through Quack's block-scaled variable-M ``A_idx`` interface.

    When router inverse-permutation ``reverse_idx`` and ``top_k`` are supplied,
    each physical token is quantized once and its scale is scattered to all
    logical routes in the same launch. Without them, a generic two-launch path
    quantizes physical rows once and gathers the much smaller linear scale
    buffer.
    """
    experts, total = _validate(x, cu_seqlens_m, gather_idx)
    physical_m, k = x.shape
    if k % _SF_ATOM:
        raise ValueError(f"K={k} must be divisible by {_SF_ATOM}")
    padded_rm = (total + _SF_ATOM - 1) // _SF_ATOM + experts - 1
    rk = k // _SF_ATOM
    q_shape = tuple(x.shape)
    sf_shape = (1, padded_rm, rk, 32, 4, 4)
    linear_sf_shape = (physical_m, k // _SF_VEC)
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
    one_launch = reverse_idx is not None or top_k is not None
    if one_launch:
        if reverse_idx is None or top_k is None:
            raise ValueError("reverse_idx and top_k must be supplied together")
        if (
            reverse_idx.shape != gather_idx.shape
            or reverse_idx.dtype != torch.int32
            or reverse_idx.device != x.device
            or not reverse_idx.is_contiguous()
        ):
            raise ValueError(
                "reverse_idx must match gather_idx as a contiguous CUDA int32 tensor"
            )
        if top_k <= 0 or physical_m * top_k != total:
            raise ValueError(
                "top_k must be positive and T * top_k must equal routed rows"
            )
        _physical_rowwise_route_sf_kernel[
            (triton.cdiv(physical_m, _GATHER_SF_BLOCK_M), k // _SF_ATOM)
        ](
            x,
            reverse_idx,
            qdata,
            scale.view(torch.uint8),
            cu_seqlens_m,
            M=physical_m,
            K=k,
            E=experts,
            RK=rk,
            TOP_K=top_k,
            N_SEARCH_ITERS=experts.bit_length(),
            BLOCK_M=_GATHER_SF_BLOCK_M,
            BLOCK_K=_SF_ATOM,
            FAST_BF16=_FAST_BF16_QUANT and x.dtype == torch.bfloat16,
            num_warps=4,
        )
        return BlockScaledOperand.from_parts(
            qdata, scale, MXFP8_E4M3, orig_dtype=x.dtype
        )

    linear_scale = (
        torch.empty(linear_sf_shape, dtype=torch.uint8, device=x.device)
        if linear_scale_out is None
        else linear_scale_out
    )
    if (
        linear_scale.shape != linear_sf_shape
        or linear_scale.dtype != torch.uint8
        or not linear_scale.is_contiguous()
    ):
        raise ValueError(
            f"linear_scale_out must be contiguous uint8 with shape {linear_sf_shape}"
        )

    _physical_rowwise_linear_sf_kernel[
        (triton.cdiv(physical_m, _VARLEN_BLOCK_M), k // _SF_ATOM)
    ](
        x,
        qdata,
        linear_scale,
        M=physical_m,
        K=k,
        BLOCK_M=_VARLEN_BLOCK_M,
        BLOCK_K=_SF_ATOM,
        FAST_BF16=_FAST_BF16_QUANT and x.dtype == torch.bfloat16,
        num_warps=_PHYSICAL_ROW_NUM_WARPS,
    )
    _gather_varlen_m_scale_kernel[(triton.cdiv(total, _GATHER_SF_BLOCK_M), rk)](
        linear_scale,
        gather_idx,
        scale.view(torch.uint8),
        cu_seqlens_m,
        TOTAL_M=total,
        E=experts,
        RK=rk,
        N_SEARCH_ITERS=experts.bit_length(),
        BLOCK_M=_GATHER_SF_BLOCK_M,
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
    block_n = _VARLEN_K_BLOCK_N
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
        FAST_BF16=_FAST_BF16_QUANT and x.dtype == torch.bfloat16,
        num_warps=_VARLEN_K_NUM_WARPS,
    )
    return BlockScaledOperand.from_parts(
        qdata,
        scale,
        MXFP8_E4M3,
        orig_dtype=x.dtype,
        quant_dim=-2,
    )


def quantize_mxfp8_varlen_k_pair(
    x0: torch.Tensor,
    x1: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    qdata0_out: torch.Tensor | None = None,
    scale0_out: torch.Tensor | None = None,
    qdata1_out: torch.Tensor | None = None,
    scale1_out: torch.Tensor | None = None,
) -> tuple[BlockScaledOperand, BlockScaledOperand]:
    """Expert-segmented dim-0 quantization of two operands in one launch."""
    experts, total = _validate(x0, cu_seqlens_k, None)
    experts1, total1 = _validate(x1, cu_seqlens_k, None)
    if experts1 != experts or total1 != total or x1.shape[0] != x0.shape[0]:
        raise ValueError("paired operands must have the same routed row dimension")
    if x1.device != x0.device or x1.dtype != x0.dtype:
        raise ValueError("paired operands must have the same dtype and device")

    padded_rk = (total + _SF_ATOM - 1) // _SF_ATOM + experts - 1
    padded_blocks = padded_rk * (_SF_ATOM // _SF_VEC)

    def prepare_outputs(
        x: torch.Tensor,
        qdata_out: torch.Tensor | None,
        scale_out: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = x.shape[1]
        q_shape = (total, n)
        sf_shape = (1, triton.cdiv(n, _SF_ATOM), padded_rk, 32, 4, 4)
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
        return qdata, scale

    qdata0, scale0 = prepare_outputs(x0, qdata0_out, scale0_out)
    qdata1, scale1 = prepare_outputs(x1, qdata1_out, scale1_out)
    max_n = max(x0.shape[1], x1.shape[1])
    _segmented_varlen_k_pair_kernel[
        (padded_blocks, triton.cdiv(max_n, _VARLEN_K_PAIR_BLOCK_N))
    ](
        x0,
        x1,
        qdata0,
        scale0.view(torch.uint8),
        qdata1,
        scale1.view(torch.uint8),
        cu_seqlens_k,
        N0=x0.shape[1],
        N1=x1.shape[1],
        E=experts,
        RK=padded_rk,
        N_SEARCH_ITERS=experts.bit_length(),
        BLOCK_N=_VARLEN_K_PAIR_BLOCK_N,
        FAST_BF16=_FAST_BF16_QUANT and x0.dtype == torch.bfloat16,
        num_warps=_VARLEN_K_PAIR_NUM_WARPS,
    )
    return (
        BlockScaledOperand.from_parts(
            qdata0,
            scale0,
            MXFP8_E4M3,
            orig_dtype=x0.dtype,
            quant_dim=-2,
        ),
        BlockScaledOperand.from_parts(
            qdata1,
            scale1,
            MXFP8_E4M3,
            orig_dtype=x1.dtype,
            quant_dim=-2,
        ),
    )


def quantize_mxfp8_varlen_dual(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    gather_idx: torch.Tensor | None = None,
    row_qdata_out: torch.Tensor | None = None,
    row_scale_out: torch.Tensor | None = None,
    col_qdata_out: torch.Tensor | None = None,
    col_scale_out: torch.Tensor | None = None,
) -> tuple[BlockScaledOperand, BlockScaledOperand]:
    """Emit routed rowwise and expert-segmented dim-0 MXFP8 views together."""
    experts, total = _validate(x, cu_seqlens, gather_idx)
    n = x.shape[1]
    if n % _SF_ATOM:
        raise ValueError(f"N={n} must be divisible by {_SF_ATOM}")
    padded_rm = (total + _SF_ATOM - 1) // _SF_ATOM + experts - 1
    padded_blocks = padded_rm * (_SF_ATOM // _SF_VEC)
    row_rk = n // _SF_ATOM
    col_rn = n // _SF_ATOM
    q_shape = (total, n)
    row_sf_shape = (1, padded_rm, row_rk, 32, 4, 4)
    col_sf_shape = (1, col_rn, padded_rm, 32, 4, 4)
    row_qdata = (
        torch.empty(q_shape, dtype=MXFP8_E4M3.qdata_dtype, device=x.device)
        if row_qdata_out is None
        else row_qdata_out
    )
    row_scale = (
        torch.empty(row_sf_shape, dtype=MXFP8_E4M3.scale_dtype, device=x.device)
        if row_scale_out is None
        else row_scale_out
    )
    col_qdata = (
        torch.empty(q_shape, dtype=MXFP8_E4M3.qdata_dtype, device=x.device)
        if col_qdata_out is None
        else col_qdata_out
    )
    col_scale = (
        torch.empty(col_sf_shape, dtype=MXFP8_E4M3.scale_dtype, device=x.device)
        if col_scale_out is None
        else col_scale_out
    )
    _check_outputs(row_qdata, row_scale, q_shape, row_sf_shape)
    _check_outputs(col_qdata, col_scale, q_shape, col_sf_shape)
    if row_qdata.data_ptr() == col_qdata.data_ptr():
        raise ValueError("rowwise and segmented-K qdata outputs must not alias")

    if n % _VARLEN_DUAL_BLOCK_N:
        raise ValueError(
            "SONICMOE_MXFP8_VARLEN_DUAL_BLOCK_N must be 32, 64, or 128 and divide N"
        )
    _varlen_dual_kernel[(padded_blocks, n // _VARLEN_DUAL_BLOCK_N)](
        x,
        gather_idx,
        row_qdata,
        row_scale.view(torch.uint8),
        col_qdata,
        col_scale.view(torch.uint8),
        cu_seqlens,
        N=n,
        E=experts,
        PADDED_BLOCKS=padded_blocks,
        ROW_RK=row_rk,
        COL_RK=padded_rm,
        HAS_GATHER=gather_idx is not None,
        N_SEARCH_ITERS=experts.bit_length(),
        BLOCK_N=_VARLEN_DUAL_BLOCK_N,
        FAST_BF16=_FAST_BF16_QUANT and x.dtype == torch.bfloat16,
        num_warps=_VARLEN_DUAL_NUM_WARPS,
    )
    return (
        BlockScaledOperand.from_parts(
            row_qdata,
            row_scale,
            MXFP8_E4M3,
            orig_dtype=x.dtype,
        ),
        BlockScaledOperand.from_parts(
            col_qdata,
            col_scale,
            MXFP8_E4M3,
            orig_dtype=x.dtype,
            quant_dim=-2,
        ),
    )


def quantize_mxfp8_varlen_iso32_dual(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    gather_idx: torch.Tensor | None = None,
    qdata_out: torch.Tensor | None = None,
    row_scale_out: torch.Tensor | None = None,
    col_scale_out: torch.Tensor | None = None,
) -> tuple[BlockScaledOperand, BlockScaledOperand]:
    """Emit two MXFP8 layouts sharing one iso32-quantized value tensor.

    Unlike standard OCP 1x32 MXFP8, every 32x32 expert-local block uses one
    scale. This experimental representation is hardware-consumable but changes
    quantization semantics, so callers must gate it explicitly.
    """
    experts, total = _validate(x, cu_seqlens, gather_idx)
    n = x.shape[1]
    if n % _SF_ATOM:
        raise ValueError(f"N={n} must be divisible by {_SF_ATOM}")
    padded_rm = (total + _SF_ATOM - 1) // _SF_ATOM + experts - 1
    padded_blocks = padded_rm * (_SF_ATOM // _SF_VEC)
    row_rk = n // _SF_ATOM
    q_shape = (total, n)
    row_sf_shape = (1, padded_rm, row_rk, 32, 4, 4)
    col_sf_shape = (1, row_rk, padded_rm, 32, 4, 4)
    qdata = (
        torch.empty(q_shape, dtype=MXFP8_E4M3.qdata_dtype, device=x.device)
        if qdata_out is None
        else qdata_out
    )
    row_scale = (
        torch.empty(row_sf_shape, dtype=MXFP8_E4M3.scale_dtype, device=x.device)
        if row_scale_out is None
        else row_scale_out
    )
    col_scale = (
        torch.empty(col_sf_shape, dtype=MXFP8_E4M3.scale_dtype, device=x.device)
        if col_scale_out is None
        else col_scale_out
    )
    _check_outputs(qdata, row_scale, q_shape, row_sf_shape)
    _check_outputs(qdata, col_scale, q_shape, col_sf_shape)

    block_n = 128
    _varlen_iso32_dual_kernel[(padded_blocks, n // block_n)](
        x,
        gather_idx,
        qdata,
        row_scale.view(torch.uint8),
        col_scale.view(torch.uint8),
        cu_seqlens,
        N=n,
        E=experts,
        ROW_RK=row_rk,
        COL_RK=padded_rm,
        HAS_GATHER=gather_idx is not None,
        N_SEARCH_ITERS=experts.bit_length(),
        BLOCK_N=block_n,
        FAST_BF16=_FAST_BF16_QUANT and x.dtype == torch.bfloat16,
        num_warps=1,
    )
    return (
        BlockScaledOperand.from_parts(
            qdata,
            row_scale,
            MXFP8_E4M3,
            orig_dtype=x.dtype,
        ),
        BlockScaledOperand.from_parts(
            qdata,
            col_scale,
            MXFP8_E4M3,
            orig_dtype=x.dtype,
            quant_dim=-2,
        ),
    )


__all__ = [
    "launch_sgd_update_and_quantize_mxfp8_weight",
    "launch_sgd_update_and_quantize_mxfp8_weight_dual",
    "quantize_mxfp8_gather_varlen_m",
    "quantize_mxfp8_varlen_dual",
    "quantize_mxfp8_varlen_iso32_dual",
    "quantize_mxfp8_varlen_k",
    "quantize_mxfp8_varlen_k_pair",
    "quantize_mxfp8_varlen_m",
    "quantize_mxfp8_weight",
    "quantize_mxfp8_weight_dual",
    "sgd_update_and_quantize_mxfp8_weight",
    "sgd_update_and_quantize_mxfp8_weight_dual",
]
