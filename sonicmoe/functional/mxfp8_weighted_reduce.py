# ********************************************************************************
# Copyright (c) 2026, SonicMoE contributors
# ********************************************************************************
"""Deterministic fused FP32 weighted reduction for local EP routes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


import os as _os

_SKIP_EMPTY = _os.environ.get("SONICMOE_REDUCE_SKIP_EMPTY", "1") != "0"


@triton.jit
def _segmented_weighted_reduce_kernel(
    pair_output,
    scatter_pos,
    packed_weights,
    output,
    recv_tokens: tl.constexpr,
    top_k: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_H: tl.constexpr,
    SKIP_EMPTY: tl.constexpr,
):
    token = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    column_mask = columns < hidden
    accumulator = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for slot in tl.static_range(0, top_k):
        position = tl.load(scatter_pos + token * top_k + slot).to(tl.int64)
        # Only ~13% of the top-k slots of a receive token map to a local
        # expert at EP8; a warp-uniform branch skips the load/convert/FMA
        # work of the empty slots (ncu: the masked version was issue-bound).
        if SKIP_EMPTY:
            if position >= 0:
                score = tl.load(packed_weights + position).to(tl.float32)
                value = tl.load(
                    pair_output + position * hidden + columns,
                    mask=column_mask,
                    other=0.0,
                ).to(tl.float32)
                accumulator += value * score
        else:
            valid = position >= 0
            score = tl.load(packed_weights + position, mask=valid, other=0.0).to(
                tl.float32
            )
            value = tl.load(
                pair_output + position * hidden + columns,
                mask=valid & column_mask,
                other=0.0,
            ).to(tl.float32)
            accumulator += value * score
    tl.store(output + token * hidden + columns, accumulator, mask=column_mask)


@triton.jit
def _segmented_weighted_reduce_backward_kernel(
    grad_output,
    pair_output,
    packed_weights,
    packed_recv_token,
    grad_pair,
    grad_weights,
    hidden: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One program per pair row: gather its token's gradient once."""
    pair = tl.program_id(0).to(tl.int64)
    token = tl.load(packed_recv_token + pair).to(tl.int64)
    weight = tl.load(packed_weights + pair).to(tl.float32)
    score_grad = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for start in tl.static_range(0, hidden, BLOCK_H):
        columns = start + tl.arange(0, BLOCK_H)
        column_mask = columns < hidden
        grad = tl.load(
            grad_output + token * hidden + columns, mask=column_mask, other=0.0
        ).to(tl.float32)
        value = tl.load(
            pair_output + pair * hidden + columns, mask=column_mask, other=0.0
        ).to(tl.float32)
        tl.store(
            grad_pair + pair * hidden + columns,
            (grad * weight).to(grad_pair.dtype.element_ty),
            mask=column_mask,
        )
        score_grad += grad * value
    tl.store(grad_weights + pair, tl.sum(score_grad, axis=0))


@dataclass(frozen=True)
class Mxfp8WeightedReduceWorkspace:
    output: torch.Tensor
    max_recv_tokens: int
    hidden: int

    @property
    def nbytes(self) -> int:
        return self.output.numel() * self.output.element_size()


def allocate_mxfp8_weighted_reduce_workspace(
    max_recv_tokens: int,
    hidden: int,
    *,
    device: torch.device | str,
) -> Mxfp8WeightedReduceWorkspace:
    if max_recv_tokens < 0 or hidden <= 0:
        raise ValueError("receive capacity must be non-negative and hidden positive")
    return Mxfp8WeightedReduceWorkspace(
        output=torch.empty(
            (max_recv_tokens, hidden), dtype=torch.bfloat16, device=device
        ),
        max_recv_tokens=max_recv_tokens,
        hidden=hidden,
    )


class _SegmentedWeightedReduce(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pair_output,
        packed_weights,
        scatter_pos,
        packed_recv_token,
        output,
        recv_tokens,
        top_k,
    ):
        hidden = pair_output.shape[1]
        ctx.save_for_backward(pair_output, packed_weights, packed_recv_token)
        ctx.recv_tokens = recv_tokens
        ctx.mark_dirty(output)
        if recv_tokens:
            # 1024 columns x 4 warps measured 2.1x faster than 256 x 4 at the
            # EP8 T61440 shape (2.5 vs 5.4 ms, bit-identical output).
            block_h = 1024
            _segmented_weighted_reduce_kernel[
                (recv_tokens, triton.cdiv(hidden, block_h))
            ](
                pair_output,
                scatter_pos,
                packed_weights,
                output,
                recv_tokens=recv_tokens,
                top_k=top_k,
                hidden=hidden,
                BLOCK_H=block_h,
                SKIP_EMPTY=_SKIP_EMPTY,
                num_warps=4,
            )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        pair_output, packed_weights, packed_recv_token = ctx.saved_tensors
        pairs, hidden = pair_output.shape
        grad_pair = torch.empty_like(pair_output)
        grad_weights = torch.empty(pairs, dtype=torch.float32, device=pair_output.device)
        if pairs:
            # Fused gather + scale + dot: avoids materialising the FP32
            # [pairs, hidden] expansions of the unfused formulation.
            _segmented_weighted_reduce_backward_kernel[(pairs,)](
                grad_output[: ctx.recv_tokens].contiguous(),
                pair_output.contiguous(),
                packed_weights,
                packed_recv_token,
                grad_pair,
                grad_weights,
                hidden=hidden,
                BLOCK_H=1024,
                num_warps=4,
            )
        return grad_pair, grad_weights, None, None, None, None, None


def segmented_sum_rows_mxfp8(
    pair_rows: torch.Tensor,
    scatter_pos: torch.Tensor,
    recv_tokens: int,
    top_k: int,
) -> torch.Tensor:
    """Sum the grouped ``pair_rows`` of every receive token (unit weights).

    Exact reverse of ``index_select(0, packed_recv_token)``: token ``t`` gathers
    the rows at ``scatter_pos[t, :]`` (``-1`` = unused slot), accumulated in
    FP32 and rounded once to BF16.  No atomics, deterministic.
    """
    if pair_rows.ndim != 2 or pair_rows.dtype != torch.bfloat16:
        raise TypeError("pair_rows must be a two-dimensional BF16 tensor")
    if scatter_pos.dtype != torch.int32 or scatter_pos.numel() < recv_tokens * top_k:
        raise TypeError("scatter_pos must contain one int32 entry per receive route")
    pair_rows = pair_rows.contiguous()
    hidden = pair_rows.shape[1]
    output = torch.empty(
        (recv_tokens, hidden), dtype=pair_rows.dtype, device=pair_rows.device
    )
    if recv_tokens:
        ones = torch.ones(
            pair_rows.shape[0], dtype=torch.float32, device=pair_rows.device
        )
        block_h = 1024
        _segmented_weighted_reduce_kernel[(recv_tokens, triton.cdiv(hidden, block_h))](
            pair_rows,
            scatter_pos,
            ones,
            output,
            recv_tokens=recv_tokens,
            top_k=top_k,
            hidden=hidden,
            BLOCK_H=block_h,
            SKIP_EMPTY=_SKIP_EMPTY,
            num_warps=4,
        )
    return output


def segmented_weighted_reduce_mxfp8(
    pair_output: torch.Tensor,
    packed_weights: torch.Tensor,
    scatter_pos: torch.Tensor,
    packed_recv_token: torch.Tensor,
    recv_tokens: int,
    top_k: int,
    workspace: Mxfp8WeightedReduceWorkspace,
) -> torch.Tensor:
    """Reduce in source top-k slot order and provide exact autograd formulas."""
    if pair_output.ndim != 2 or pair_output.dtype != torch.bfloat16:
        raise TypeError("pair_output must be a two-dimensional BF16 tensor")
    if (
        packed_weights.shape != (pair_output.shape[0],)
        or packed_weights.dtype != torch.float32
    ):
        raise TypeError("packed_weights must be one-dimensional float32")
    if (
        packed_recv_token.shape != (pair_output.shape[0],)
        or packed_recv_token.dtype != torch.int32
    ):
        raise TypeError("packed_recv_token must be one-dimensional int32")
    if scatter_pos.dtype != torch.int32 or scatter_pos.numel() < recv_tokens * top_k:
        raise TypeError("scatter_pos must contain one int32 entry per receive route")
    if (
        recv_tokens > workspace.max_recv_tokens
        or pair_output.shape[1] != workspace.hidden
    ):
        raise ValueError("weighted-reduce workspace capacity does not match")
    if any(
        tensor.device != pair_output.device
        for tensor in (
            packed_weights,
            scatter_pos,
            packed_recv_token,
            workspace.output,
        )
    ):
        raise ValueError("all weighted-reduce tensors must share a device")
    # ``apply`` marks its output buffer dirty, so the returned Tensor carries
    # that invocation's grad_fn. Reusing the exact Tensor object on the next
    # optimizer step would chain the new graph to already-freed saved tensors.
    # A detached alias keeps the allocation/data pointer while starting a fresh
    # autograd history for every invocation.
    output_buffer = workspace.output.detach()
    output = _SegmentedWeightedReduce.apply(
        pair_output,
        packed_weights,
        scatter_pos,
        packed_recv_token,
        output_buffer,
        recv_tokens,
        top_k,
    )
    return output[:recv_tokens]


__all__ = [
    "Mxfp8WeightedReduceWorkspace",
    "allocate_mxfp8_weighted_reduce_workspace",
    "segmented_sum_rows_mxfp8",
    "segmented_weighted_reduce_mxfp8",
]
