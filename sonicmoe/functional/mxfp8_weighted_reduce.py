# ********************************************************************************
# Copyright (c) 2026, SonicMoE contributors
# ********************************************************************************
"""Deterministic fused FP32 weighted reduction for local EP routes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


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
):
    token = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    column_mask = columns < hidden
    accumulator = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for slot in tl.static_range(0, top_k):
        position = tl.load(scatter_pos + token * top_k + slot).to(tl.int64)
        valid = position >= 0
        score = tl.load(packed_weights + position, mask=valid, other=0.0).to(tl.float32)
        value = tl.load(
            pair_output + position * hidden + columns,
            mask=valid & column_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += value * score
    tl.store(output + token * hidden + columns, accumulator, mask=column_mask)


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
            block_h = 256
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
                num_warps=4,
            )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        pair_output, packed_weights, packed_recv_token = ctx.saved_tensors
        expanded_grad = (
            grad_output[: ctx.recv_tokens].index_select(0, packed_recv_token).float()
        )
        grad_pair = (expanded_grad * packed_weights.unsqueeze(1)).to(pair_output.dtype)
        grad_weights = (expanded_grad * pair_output.float()).sum(dim=1)
        return grad_pair, grad_weights, None, None, None, None, None


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
    "segmented_weighted_reduce_mxfp8",
]
