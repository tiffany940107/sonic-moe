# ********************************************************************************
# Copyright (c) 2026, SonicMoE contributors
# ********************************************************************************
"""Counting route-pack from transported MXFP8 rows to SM100 grouped operands."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from quack.blockscaled import MXFP8_E4M3, BlockScaledOperand


@triton.jit
def _expert_histogram_kernel(
    expert_ids,
    counts,
    expert_stride_m,
    num_slots: tl.constexpr,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_slots
    token = offsets // top_k
    slot = offsets % top_k
    expert = tl.load(
        expert_ids + token * expert_stride_m + slot,
        mask=mask,
        other=-1,
    ).to(tl.int32)
    valid = mask & (expert >= 0) & (expert < num_experts)
    tl.atomic_add(counts + expert, 1, mask=valid)


@triton.jit
def _expert_prefix_kernel(
    counts,
    indptr,
    cursors,
    num_experts: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    valid = offsets < num_experts
    values = tl.load(counts + offsets, mask=valid, other=0).to(tl.int32)
    inclusive = tl.cumsum(values, axis=0)
    exclusive = inclusive - values
    tl.store(indptr + offsets, exclusive, mask=valid)
    tl.store(cursors + offsets, exclusive, mask=valid)
    tl.store(indptr + num_experts, tl.sum(values, axis=0))


@triton.jit
def _assign_route_positions_kernel(
    expert_ids,
    weights,
    cursors,
    scatter_pos,
    packed_recv_token,
    packed_expert,
    packed_weights,
    expert_stride_m,
    weight_stride_m,
    num_slots: tl.constexpr,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_slots
    recv_token = offsets // top_k
    slot = offsets % top_k
    expert = tl.load(
        expert_ids + recv_token * expert_stride_m + slot,
        mask=mask,
        other=-1,
    ).to(tl.int32)
    valid = mask & (expert >= 0) & (expert < num_experts)
    position = tl.atomic_add(cursors + expert, 1, mask=valid)
    tl.store(scatter_pos + offsets, position, mask=valid)
    tl.store(scatter_pos + offsets, -1, mask=mask & ~valid)
    weight = tl.load(
        weights + recv_token * weight_stride_m + slot,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    tl.store(packed_recv_token + position, recv_token, mask=valid)
    tl.store(packed_expert + position, expert, mask=valid)
    tl.store(packed_weights + position, weight, mask=valid)


@triton.jit
def _gather_qdata_kernel(
    qdata,
    scatter_pos,
    packed_qdata,
    q_stride_m,
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
        qdata + recv_token * q_stride_m + column,
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
    scale_stride_m,
    expert_stride_m,
    num_slots: tl.constexpr,
    sf_k: tl.constexpr,
    scale_rk: tl.constexpr,
    top_k: tl.constexpr,
    num_experts: tl.constexpr,
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
    recv_token = route // top_k
    slot = route % top_k
    expert = tl.load(
        expert_ids + recv_token * expert_stride_m + slot,
        mask=route_mask,
        other=-1,
    ).to(tl.int64)
    valid = route_mask & (position >= 0) & (expert >= 0) & (expert < num_experts)
    value = tl.load(
        linear_scale + recv_token * scale_stride_m + sf_col,
        mask=valid & sf_mask,
        other=0,
    )
    expert_start = tl.load(indptr + expert, mask=valid, other=0).to(tl.int64)
    padded_start = (expert_start // 128 + expert) * 128
    padded_row = position + padded_start - expert_start
    row_block = padded_row // 128
    row_in_block = padded_row % 128
    k_block = sf_col // 4
    row_inner = row_in_block % 32
    row_outer = row_in_block // 32
    k_inner = sf_col % 4
    blocked_offset = (
        (((row_block * scale_rk + k_block) * 32 + row_inner) * 4 + row_outer) * 4
    ) + k_inner
    tl.store(blocked_scale + blocked_offset, value, mask=valid & sf_mask)


@dataclass(frozen=True)
class Mxfp8RoutePackWorkspace:
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
    a_idx: torch.Tensor | None
    physical_qdata_copied: bool
    recv_token: torch.Tensor
    expert: torch.Tensor
    weights: torch.Tensor
    indptr: torch.Tensor
    scatter_pos: torch.Tensor

    @property
    def total_pairs(self) -> int:
        return self.weights.shape[0]


def allocate_mxfp8_route_pack_workspace(
    max_recv_tokens: int,
    max_pairs: int,
    top_k: int,
    local_experts: int,
    hidden: int,
    *,
    device: torch.device | str,
    qdata_rows: int | None = None,
) -> Mxfp8RoutePackWorkspace:
    if min(max_recv_tokens, max_pairs) < 0:
        raise ValueError("route-pack capacities must be non-negative")
    if top_k <= 0 or local_experts <= 0:
        raise ValueError("top_k and local_experts must be positive")
    if hidden % 128:
        raise ValueError("hidden must be divisible by 128")
    if max_pairs > max_recv_tokens * top_k:
        raise ValueError("max_pairs exceeds max_recv_tokens * top_k")
    qdata_rows = max_pairs if qdata_rows is None else qdata_rows
    if not 0 <= qdata_rows <= max_pairs:
        raise ValueError("qdata_rows must be in [0, max_pairs]")
    padded_rm = (max_pairs + 127) // 128 + local_experts - 1
    return Mxfp8RoutePackWorkspace(
        qdata=torch.empty(
            (qdata_rows, hidden), dtype=MXFP8_E4M3.qdata_dtype, device=device
        ),
        scale=torch.empty(
            (1, padded_rm, (hidden + 127) // 128, 32, 4, 4),
            dtype=MXFP8_E4M3.scale_dtype,
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
    *,
    materialize_qdata: bool = False,
) -> Mxfp8RoutePackResult:
    """Group local routes and directly form a Quack variable-M operand."""
    if qdata.dtype != MXFP8_E4M3.qdata_dtype or qdata.ndim != 2:
        raise TypeError("qdata must be a two-dimensional E4M3 tensor")
    if linear_scale.dtype not in (MXFP8_E4M3.scale_dtype, torch.uint8):
        raise TypeError("linear_scale must be E8M0 or its uint8 view")
    if expert_ids.dtype != torch.int32 or expert_ids.ndim != 2:
        raise TypeError("expert_ids must be a two-dimensional int32 tensor")
    if weights.dtype != torch.float32 or weights.shape != expert_ids.shape:
        raise TypeError("weights must be float32 with the same shape as expert_ids")
    if any(
        tensor.stride(1) != 1 for tensor in (qdata, linear_scale, expert_ids, weights)
    ):
        raise ValueError("route-pack inputs must have unit inner stride")
    recv_tokens, top_k = expert_ids.shape
    hidden = qdata.shape[1]
    sf_k = hidden // 32
    if qdata.shape[0] != recv_tokens or linear_scale.shape != (recv_tokens, sf_k):
        raise ValueError("transport values/scales do not match receive records")
    if not 0 <= total_pairs <= workspace.max_pairs:
        raise ValueError("total_pairs exceeds route-pack capacity")
    if recv_tokens > workspace.max_recv_tokens:
        raise ValueError("receive records exceed route-pack capacity")
    if workspace.top_k != top_k or workspace.hidden != hidden:
        raise ValueError("route metadata does not match the workspace")
    if any(
        tensor.device != qdata.device
        for tensor in (linear_scale, expert_ids, weights, workspace.qdata)
    ):
        raise ValueError("all route-pack tensors must share a device")

    num_slots = recv_tokens * top_k
    # TMA gather accepts a pitched physical A when its base and row pitch meet
    # the 16-byte descriptor alignment.  Common H=1024/2048 transport records
    # satisfy that contract, so consume their AoS qdata field with no copy.  A
    # narrow or oddly-sized record falls back to one contiguous copy per receive
    # token; it still avoids the old copy per valid route.
    direct_physical_qdata = (
        not materialize_qdata
        and qdata.data_ptr() % 16 == 0
        and qdata.stride(0) * qdata.element_size() % 16 == 0
    )
    physical_qdata_copied = not materialize_qdata and not direct_physical_qdata
    required_qdata_rows = (
        total_pairs
        if materialize_qdata
        else recv_tokens
        if physical_qdata_copied
        else 0
    )
    if workspace.qdata.shape[0] < required_qdata_rows:
        raise ValueError(
            "route-pack qdata workspace is smaller than the selected storage policy"
        )
    if physical_qdata_copied and recv_tokens:
        workspace.qdata[:recv_tokens].copy_(qdata)
    workspace.counts.zero_()
    if num_slots:
        _expert_histogram_kernel[(triton.cdiv(num_slots, 256),)](
            expert_ids,
            workspace.counts,
            expert_ids.stride(0),
            num_slots=num_slots,
            num_experts=workspace.local_experts,
            top_k=top_k,
            BLOCK=256,
        )
    prefix_block = triton.next_power_of_2(workspace.local_experts)
    _expert_prefix_kernel[(1,)](
        workspace.counts,
        workspace.indptr,
        workspace.cursors,
        num_experts=workspace.local_experts,
        BLOCK=prefix_block,
        num_warps=4,
    )
    if num_slots:
        _assign_route_positions_kernel[(triton.cdiv(num_slots, 256),)](
            expert_ids,
            weights,
            workspace.cursors,
            workspace.scatter_pos,
            workspace.recv_token,
            workspace.expert,
            workspace.weights,
            expert_ids.stride(0),
            weights.stride(0),
            num_slots=num_slots,
            num_experts=workspace.local_experts,
            top_k=top_k,
            BLOCK=256,
        )
        block_rows = 4
        block_cols = 256
        if materialize_qdata:
            _gather_qdata_kernel[
                (triton.cdiv(num_slots, block_rows), triton.cdiv(hidden, block_cols))
            ](
                qdata.view(torch.uint8),
                workspace.scatter_pos,
                workspace.qdata.view(torch.uint8),
                qdata.stride(0),
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
            linear_scale.stride(0),
            expert_ids.stride(0),
            num_slots=num_slots,
            sf_k=sf_k,
            scale_rk=(sf_k + 3) // 4,
            top_k=top_k,
            num_experts=workspace.local_experts,
            BLOCK_ROWS=block_rows,
            BLOCK_COLS=scale_cols,
            num_warps=4,
        )

    padded_rm = (total_pairs + 127) // 128 + workspace.local_experts - 1
    operand = BlockScaledOperand.from_parts(
        (
            workspace.qdata[:total_pairs]
            if materialize_qdata
            else qdata
            if direct_physical_qdata
            else workspace.qdata[:recv_tokens]
        ),
        workspace.scale[:, :padded_rm],
        MXFP8_E4M3,
        orig_dtype=torch.bfloat16,
    )
    return Mxfp8RoutePackResult(
        operand=operand,
        a_idx=None if materialize_qdata else workspace.recv_token[:total_pairs],
        physical_qdata_copied=physical_qdata_copied,
        recv_token=workspace.recv_token[:total_pairs],
        expert=workspace.expert[:total_pairs],
        weights=workspace.weights[:total_pairs],
        indptr=workspace.indptr,
        scatter_pos=workspace.scatter_pos[:num_slots],
    )


__all__ = [
    "Mxfp8RoutePackResult",
    "Mxfp8RoutePackWorkspace",
    "allocate_mxfp8_route_pack_workspace",
    "route_pack_mxfp8",
]
