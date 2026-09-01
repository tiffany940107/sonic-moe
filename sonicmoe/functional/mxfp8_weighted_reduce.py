# ********************************************************************************
# Copyright (c) 2026 SonicMoE contributors
# ********************************************************************************
"""Allocation-free local top-k weighted reduction for routed expert rows.

The reference EP path materializes three additional full-sized tensors around
``Tensor.index_add_``: a BF16->FP32 conversion, a broadcast multiply, and a
fresh FP32 destination.  This kernel performs the conversion, router-score
multiply, and scatter-add in one pass.  The FC2 pair output is deliberately
kept as an input for this low-risk implementation; a direct FC2 epilogue sink
is an experimental follow-on and uses the same public contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _weighted_scatter_add_kernel(
    pair_output,
    recv_token,
    weights,
    reduced,
    num_pairs: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pair = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = (pair < num_pairs) & (columns < hidden)
    destination = tl.load(recv_token + pair, mask=pair < num_pairs, other=0).to(
        tl.int64
    )
    score = tl.load(weights + pair, mask=pair < num_pairs, other=0.0).to(
        tl.float32
    )
    value = tl.load(pair_output + pair * hidden + columns, mask=mask, other=0.0).to(
        tl.float32
    )
    tl.atomic_add(reduced + destination * hidden + columns, value * score, mask=mask)


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


@dataclass(frozen=True)
class Mxfp8WeightedReduceWorkspace:
    """Reusable FP32 accumulation buffer, sized by received-token capacity."""

    reduced: torch.Tensor
    output: torch.Tensor
    max_recv_tokens: int
    hidden: int

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.reduced, self.output)
        )

    def active(self, recv_tokens: int) -> torch.Tensor:
        if not 0 <= recv_tokens <= self.max_recv_tokens:
            raise ValueError("received-token count exceeds weighted-reduce capacity")
        return self.reduced[:recv_tokens]


def allocate_mxfp8_weighted_reduce_workspace(
    max_recv_tokens: int,
    hidden: int,
    *,
    device: torch.device | str,
    include_fp32: bool = True,
) -> Mxfp8WeightedReduceWorkspace:
    if max_recv_tokens < 0 or hidden <= 0:
        raise ValueError("max_recv_tokens must be non-negative and hidden positive")
    return Mxfp8WeightedReduceWorkspace(
        reduced=torch.empty(
            (max_recv_tokens if include_fp32 else 0, hidden),
            dtype=torch.float32,
            device=device,
        ),
        output=torch.empty(
            (max_recv_tokens, hidden), dtype=torch.bfloat16, device=device
        ),
        max_recv_tokens=max_recv_tokens,
        hidden=hidden,
    )


def weighted_reduce_mxfp8(
    pair_output: torch.Tensor,
    recv_token: torch.Tensor,
    weights: torch.Tensor,
    recv_tokens: int,
    workspace: Mxfp8WeightedReduceWorkspace,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reduce expert pair rows into receive-token order.

    Accumulation is FP32.  Atomic ordering is intentionally unspecified, as
    with CUDA ``index_add_``; callers should apply the numerical and repeated-
    run variance gates used by the EP benchmark.
    """

    if pair_output.ndim != 2 or pair_output.dtype != torch.bfloat16:
        raise TypeError("pair_output must be a two-dimensional BF16 tensor")
    if recv_token.ndim != 1 or recv_token.dtype not in (torch.int32, torch.int64):
        raise TypeError("recv_token must be a one-dimensional int32/int64 tensor")
    if weights.ndim != 1 or weights.dtype != torch.float32:
        raise TypeError("weights must be a one-dimensional float32 tensor")
    num_pairs, hidden = pair_output.shape
    if recv_token.numel() != num_pairs or weights.numel() != num_pairs:
        raise ValueError("pair output, token index, and weights have different lengths")
    if hidden != workspace.hidden:
        raise ValueError("hidden dimension does not match weighted-reduce workspace")
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("weighted reduce output must be BF16 or FP32")
    device = pair_output.device
    if any(t.device != device for t in (recv_token, weights, workspace.reduced)):
        raise ValueError("all weighted-reduce tensors must be on the same device")

    reduced = workspace.active(recv_tokens)
    reduced.zero_()
    if num_pairs:
        block_h = 256
        _weighted_scatter_add_kernel[(num_pairs, triton.cdiv(hidden, block_h))](
            pair_output,
            recv_token,
            weights,
            reduced,
            num_pairs=num_pairs,
            hidden=hidden,
            BLOCK_H=block_h,
            num_warps=4,
        )
    if out_dtype == torch.float32:
        return reduced
    output = workspace.output[:recv_tokens]
    output.copy_(reduced)
    return output


def segmented_weighted_reduce_mxfp8(
    pair_output: torch.Tensor,
    scatter_pos: torch.Tensor,
    packed_weights: torch.Tensor,
    recv_tokens: int,
    top_k: int,
    workspace: Mxfp8WeightedReduceWorkspace,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Deterministically reduce each token's local routes without atomics.

    ``scatter_pos`` is the inverse map emitted by route-pack: one entry for
    every original ``(receive token, top-k slot)``, with ``-1`` for routes
    owned by another rank.  Each program accumulates one token/hidden tile in
    source-slot order, so no cross-CTA synchronization or global partials are
    required.
    """

    if pair_output.ndim != 2 or pair_output.dtype != torch.bfloat16:
        raise TypeError("pair_output must be a two-dimensional BF16 tensor")
    if scatter_pos.dtype != torch.int32 or scatter_pos.ndim != 1:
        raise TypeError("scatter_pos must be a one-dimensional int32 tensor")
    if packed_weights.dtype != torch.float32 or packed_weights.ndim != 1:
        raise TypeError("packed_weights must be a one-dimensional float32 tensor")
    if top_k <= 0 or scatter_pos.numel() < recv_tokens * top_k:
        raise ValueError("scatter_pos capacity is smaller than recv_tokens * top_k")
    if packed_weights.numel() != pair_output.shape[0]:
        raise ValueError("packed_weights length must equal routed pair rows")
    if pair_output.shape[1] != workspace.hidden:
        raise ValueError("hidden dimension does not match weighted-reduce workspace")
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("weighted reduce output must be BF16 or FP32")
    if any(
        tensor.device != pair_output.device
        for tensor in (scatter_pos, packed_weights, workspace.reduced)
    ):
        raise ValueError("all segmented-reduce tensors must be on the same device")

    output = (
        workspace.reduced[:recv_tokens]
        if out_dtype == torch.float32
        else workspace.output[:recv_tokens]
    )
    if recv_tokens:
        block_h = 256
        _segmented_weighted_reduce_kernel[
            (recv_tokens, triton.cdiv(workspace.hidden, block_h))
        ](
            pair_output,
            scatter_pos,
            packed_weights,
            output,
            recv_tokens=recv_tokens,
            top_k=top_k,
            hidden=workspace.hidden,
            BLOCK_H=block_h,
            num_warps=4,
        )
    return output


def hybrid_weighted_reduce_mxfp8(
    pair_output: torch.Tensor,
    recv_token: torch.Tensor,
    scatter_pos: torch.Tensor,
    packed_weights: torch.Tensor,
    recv_tokens: int,
    top_k: int,
    workspace: Mxfp8WeightedReduceWorkspace,
    *,
    segmented_min_routes_per_token: float = 2.0,
) -> tuple[torch.Tensor, str]:
    """Select the atomic or token-segmented path from setup-known extents."""

    routes_per_token = pair_output.shape[0] / max(1, recv_tokens)
    if routes_per_token >= segmented_min_routes_per_token:
        return (
            segmented_weighted_reduce_mxfp8(
                pair_output,
                scatter_pos,
                packed_weights,
                recv_tokens,
                top_k,
                workspace,
            ),
            "segmented",
        )
    return (
        weighted_reduce_mxfp8(
            pair_output,
            recv_token,
            packed_weights,
            recv_tokens,
            workspace,
        ),
        "atomic",
    )


__all__ = [
    "Mxfp8WeightedReduceWorkspace",
    "allocate_mxfp8_weighted_reduce_workspace",
    "hybrid_weighted_reduce_mxfp8",
    "segmented_weighted_reduce_mxfp8",
    "weighted_reduce_mxfp8",
]
