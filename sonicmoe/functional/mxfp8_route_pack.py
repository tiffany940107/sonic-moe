# ********************************************************************************
# Copyright (c) 2026 SonicMoE contributors
# ********************************************************************************
"""Allocation-free counting route-pack for SM120 MXFP8 expert inputs.

The bounded local-expert domain makes a general stable argsort unnecessary.
This path builds an expert histogram, assigns each valid route a counting-sort
position, gathers activation bytes, and writes E8M0 scales directly into the
QuACK 128x4 blocked layout.  Route order inside one expert is intentionally
unspecified; ``recv_token`` and ``weights`` move with every row, preserving the
exact MoE semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from quack.blockscaled import BlockScaledOperand

from .mxfp8 import (
    MXFP8_FORMAT,
    MXFP8_SCALE_ATOM_M,
    MXFP8_SCALE_BLOCK_K,
    _ceil_div,
)


@triton.jit
def _expert_histogram_kernel(
    expert_ids,
    counts,
    num_slots: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_slots
    expert = tl.load(expert_ids + offsets, mask=mask, other=-1).to(tl.int32)
    valid = mask & (expert >= 0) & (expert < num_experts)
    tl.atomic_add(counts + expert, 1, mask=valid)


@triton.jit
def _assign_route_positions_kernel(
    expert_ids,
    weights,
    cursors,
    scatter_pos,
    packed_recv_token,
    packed_expert,
    packed_weights,
    num_slots: tl.constexpr,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_slots
    expert = tl.load(expert_ids + offsets, mask=mask, other=-1).to(tl.int32)
    valid = mask & (expert >= 0) & (expert < num_experts)
    position = tl.atomic_add(cursors + expert, 1, mask=valid)
    tl.store(scatter_pos + offsets, position, mask=valid)
    tl.store(scatter_pos + offsets, -1, mask=mask & ~valid)
    recv_token = offsets // top_k
    weight = tl.load(weights + offsets, mask=valid, other=0.0).to(tl.float32)
    tl.store(packed_recv_token + position, recv_token, mask=valid)
    tl.store(packed_expert + position, expert, mask=valid)
    tl.store(packed_weights + position, weight, mask=valid)


@triton.jit
def _gather_qdata_kernel(
    qdata,
    scatter_pos,
    packed_qdata,
    num_slots: tl.constexpr,
    hidden: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    route_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    hidden_offsets = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    route = route_offsets[:, None]
    column = hidden_offsets[None, :]
    route_mask = route < num_slots
    column_mask = column < hidden
    position = tl.load(scatter_pos + route, mask=route_mask, other=-1).to(tl.int64)
    valid = route_mask & (position >= 0)
    recv_token = route // top_k
    value = tl.load(
        qdata + recv_token * hidden + column,
        mask=valid & column_mask,
        other=0,
    )
    tl.store(
        packed_qdata + position * hidden + column,
        value,
        mask=valid & column_mask,
    )


@triton.jit
def _gather_blocked_scale_kernel(
    linear_scale,
    expert_ids,
    scatter_pos,
    indptr,
    blocked_scale,
    num_slots: tl.constexpr,
    sf_k: tl.constexpr,
    scale_rk: tl.constexpr,
    top_k: tl.constexpr,
    num_experts: tl.constexpr,
    scale_atom_m: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    route_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    sf_offsets = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    route = route_offsets[:, None]
    sf_col = sf_offsets[None, :]
    route_mask = route < num_slots
    sf_mask = sf_col < sf_k
    position = tl.load(scatter_pos + route, mask=route_mask, other=-1).to(tl.int64)
    expert = tl.load(expert_ids + route, mask=route_mask, other=-1).to(tl.int64)
    valid = route_mask & (position >= 0) & (expert >= 0) & (expert < num_experts)
    recv_token = route // top_k
    value = tl.load(
        linear_scale + recv_token * sf_k + sf_col,
        mask=valid & sf_mask,
        other=0,
    )

    expert_start = tl.load(indptr + expert, mask=valid, other=0).to(tl.int64)
    padded_start = (expert_start // scale_atom_m + expert) * scale_atom_m
    padded_row = position + padded_start - expert_start
    row_block = padded_row // scale_atom_m
    row_in_block = padded_row % scale_atom_m
    k_block = sf_col // 4
    row_inner = row_in_block % 32
    row_outer = row_in_block // 32
    k_inner = sf_col % 4
    blocked_offset = (
        ((((row_block * scale_rk + k_block) * 32 + row_inner) * 4 + row_outer) * 4)
        + k_inner
    )
    tl.store(blocked_scale + blocked_offset, value, mask=valid & sf_mask)


@dataclass(frozen=True)
class Mxfp8RoutePackWorkspace:
    """Capacity buffers reused by :func:`route_pack_mxfp8`."""

    qdata: torch.Tensor
    scale: torch.Tensor
    recv_token: torch.Tensor
    expert: torch.Tensor
    weights: torch.Tensor
    counts: torch.Tensor
    cursors: torch.Tensor
    indptr: torch.Tensor
    scatter_pos: torch.Tensor
    max_recv_tokens: int
    max_pairs: int
    top_k: int
    local_experts: int
    hidden: int

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.qdata,
                self.scale,
                self.recv_token,
                self.expert,
                self.weights,
                self.counts,
                self.cursors,
                self.indptr,
                self.scatter_pos,
            )
        )


@dataclass(frozen=True)
class Mxfp8RoutePackResult:
    operand: BlockScaledOperand
    recv_token: torch.Tensor
    expert: torch.Tensor
    weights: torch.Tensor
    indptr: torch.Tensor

    @property
    def total_pairs(self) -> int:
        return self.operand.shape[0]


def allocate_mxfp8_route_pack_workspace(
    max_recv_tokens: int,
    max_pairs: int,
    top_k: int,
    local_experts: int,
    hidden: int,
    *,
    device: torch.device | str,
) -> Mxfp8RoutePackWorkspace:
    if min(max_recv_tokens, max_pairs) < 0:
        raise ValueError("route-pack capacities must be non-negative")
    if top_k <= 0 or local_experts <= 0:
        raise ValueError("top_k and local_experts must be positive")
    if hidden % MXFP8_SCALE_BLOCK_K:
        raise ValueError("hidden must be divisible by the MXFP8 K32 block")
    if max_pairs > max_recv_tokens * top_k:
        raise ValueError("max_pairs exceeds max_recv_tokens * top_k")
    padded_rm = _ceil_div(max_pairs, MXFP8_SCALE_ATOM_M) + local_experts - 1
    sf_k = hidden // MXFP8_SCALE_BLOCK_K
    return Mxfp8RoutePackWorkspace(
        qdata=torch.empty(
            (max_pairs, hidden), dtype=torch.float8_e4m3fn, device=device
        ),
        scale=torch.zeros(
            (1, padded_rm, _ceil_div(sf_k, 4), 32, 4, 4),
            dtype=torch.float8_e8m0fnu,
            device=device,
        ),
        recv_token=torch.empty(max_pairs, dtype=torch.int32, device=device),
        expert=torch.empty(max_pairs, dtype=torch.int32, device=device),
        weights=torch.empty(max_pairs, dtype=torch.float32, device=device),
        counts=torch.empty(local_experts, dtype=torch.int32, device=device),
        cursors=torch.empty(local_experts, dtype=torch.int32, device=device),
        indptr=torch.empty(local_experts + 1, dtype=torch.int32, device=device),
        scatter_pos=torch.empty(
            max_recv_tokens * top_k, dtype=torch.int32, device=device
        ),
        max_recv_tokens=max_recv_tokens,
        max_pairs=max_pairs,
        top_k=top_k,
        local_experts=local_experts,
        hidden=hidden,
    )


def route_pack_mxfp8(
    qdata: torch.Tensor,
    linear_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    total_pairs: int,
    workspace: Mxfp8RoutePackWorkspace,
) -> Mxfp8RoutePackResult:
    """Group received route rows and directly form a QuACK MXFP8 operand.

    ``total_pairs`` comes from the dispatch plan's pair-count exchange.  It is
    a host integer by design, avoiding a device-to-host synchronization in the
    timed route-pack path.
    """

    if qdata.dtype != torch.float8_e4m3fn or qdata.ndim != 2:
        raise TypeError("qdata must be a two-dimensional float8_e4m3fn tensor")
    if linear_scale.dtype not in (torch.float8_e8m0fnu, torch.uint8):
        raise TypeError("linear_scale must be E8M0 or its byte view")
    if expert_ids.dtype != torch.int32 or expert_ids.ndim != 2:
        raise TypeError("expert_ids must be a two-dimensional int32 tensor")
    if weights.dtype != torch.float32 or weights.shape != expert_ids.shape:
        raise TypeError("weights must be float32 with the same shape as expert_ids")
    recv_tokens, top_k = expert_ids.shape
    hidden = qdata.shape[1]
    sf_k = hidden // MXFP8_SCALE_BLOCK_K
    if qdata.shape[0] != recv_tokens:
        raise ValueError("qdata and route metadata row counts differ")
    if linear_scale.shape != (recv_tokens, sf_k):
        raise ValueError("linear scale shape must be (recv_tokens, hidden / 32)")
    if not 0 <= total_pairs <= workspace.max_pairs:
        raise ValueError("total_pairs exceeds route-pack workspace capacity")
    if recv_tokens > workspace.max_recv_tokens:
        raise ValueError("received tokens exceed route-pack workspace capacity")
    if (
        workspace.top_k != top_k
        or workspace.hidden != hidden
        or workspace.scatter_pos.numel() < recv_tokens * top_k
    ):
        raise ValueError("route metadata does not match the workspace")
    device = qdata.device
    if any(
        tensor.device != device
        for tensor in (linear_scale, expert_ids, weights, workspace.qdata)
    ):
        raise ValueError("all route-pack operands must be on the same device")

    num_slots = recv_tokens * top_k
    workspace.counts.zero_()
    _expert_histogram_kernel[(triton.cdiv(num_slots, 256),)](
        expert_ids,
        workspace.counts,
        num_slots=num_slots,
        num_experts=workspace.local_experts,
        BLOCK=256,
    )
    workspace.indptr[0].zero_()
    torch.cumsum(workspace.counts, dim=0, out=workspace.indptr[1:])
    workspace.cursors.copy_(workspace.indptr[:-1])
    _assign_route_positions_kernel[(triton.cdiv(num_slots, 256),)](
        expert_ids,
        weights,
        workspace.cursors,
        workspace.scatter_pos,
        workspace.recv_token,
        workspace.expert,
        workspace.weights,
        num_slots=num_slots,
        num_experts=workspace.local_experts,
        top_k=top_k,
        BLOCK=256,
    )
    block_rows = 4
    block_cols = 256
    _gather_qdata_kernel[
        (triton.cdiv(num_slots, block_rows), triton.cdiv(hidden, block_cols))
    ](
        qdata.view(torch.uint8),
        workspace.scatter_pos,
        workspace.qdata.view(torch.uint8),
        num_slots=num_slots,
        hidden=hidden,
        top_k=top_k,
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
        num_warps=4,
    )
    scale_cols = 64
    _gather_blocked_scale_kernel[
        (triton.cdiv(num_slots, block_rows), triton.cdiv(sf_k, scale_cols))
    ](
        linear_scale.view(torch.uint8),
        expert_ids,
        workspace.scatter_pos,
        workspace.indptr,
        workspace.scale.view(torch.uint8),
        num_slots=num_slots,
        sf_k=sf_k,
        scale_rk=_ceil_div(sf_k, 4),
        top_k=top_k,
        num_experts=workspace.local_experts,
        scale_atom_m=MXFP8_SCALE_ATOM_M,
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=scale_cols,
        num_warps=4,
    )

    padded_rm = _ceil_div(total_pairs, MXFP8_SCALE_ATOM_M) + workspace.local_experts - 1
    operand = BlockScaledOperand.from_parts(
        workspace.qdata[:total_pairs],
        workspace.scale[:, :padded_rm],
        MXFP8_FORMAT,
    )
    return Mxfp8RoutePackResult(
        operand=operand,
        recv_token=workspace.recv_token[:total_pairs],
        expert=workspace.expert[:total_pairs],
        weights=workspace.weights[:total_pairs],
        indptr=workspace.indptr,
    )


__all__ = [
    "Mxfp8RoutePackResult",
    "Mxfp8RoutePackWorkspace",
    "allocate_mxfp8_route_pack_workspace",
    "route_pack_mxfp8",
]
