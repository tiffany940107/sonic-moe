# Copyright (c) 2026, SonicMoE contributors.
"""Small-expert fused router projection and top-k for the SM100 MXFP8 path."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .router_aux import _switch_aux_forward_kernel

_ROUTER_BLOCK_M = int(os.environ.get("SONICMOE_MXFP8_ROUTER_BLOCK_M", "64"))
if _ROUTER_BLOCK_M not in (32, 64, 128, 256):
    raise ValueError("SONICMOE_MXFP8_ROUTER_BLOCK_M must be 32, 64, 128, or 256")
_ROUTER_BLOCK_H = int(os.environ.get("SONICMOE_MXFP8_ROUTER_BLOCK_H", "128"))
if _ROUTER_BLOCK_H not in (32, 64, 128):
    raise ValueError("SONICMOE_MXFP8_ROUTER_BLOCK_H must be 32, 64, or 128")
_ROUTER_NUM_WARPS = int(os.environ.get("SONICMOE_MXFP8_ROUTER_WARPS", "8"))
if _ROUTER_NUM_WARPS not in (4, 8):
    raise ValueError("SONICMOE_MXFP8_ROUTER_WARPS must be 4 or 8")


@triton.jit
def _router_linear_topk_kernel(
    x_ptr,
    weight_ptr,
    logits_ptr,
    score_ptr,
    index_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    expert = tl.arange(0, BLOCK_E)
    accumulator = tl.zeros((BLOCK_M, BLOCK_E), tl.float32)
    for h_start in tl.static_range(0, H, BLOCK_H):
        h = h_start + tl.arange(0, BLOCK_H)
        x = tl.load(
            x_ptr + row[:, None] * H + h[None, :],
            mask=row[:, None] < T,
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + expert[:, None] * H + h[None, :],
            mask=expert[:, None] < E,
            other=0.0,
        )
        accumulator += tl.dot(x, tl.trans(weight))

    # Match F.linear's BF16 output before the top-k decision.
    logits = accumulator.to(tl.bfloat16).to(tl.float32)
    output_mask = (row[:, None] < T) & (expert[None, :] < E)
    tl.store(
        logits_ptr + row[:, None] * E + expert[None, :],
        logits,
        mask=output_mask,
    )

    # Keep the selection domain restricted to real experts even when training
    # has produced NaN/Inf router logits. ``argmax`` may otherwise select a
    # padded BLOCK_E lane, which later becomes an out-of-range metadata index.
    # Rank NaNs like PyTorch topk's high-valued NaN handling and clamp infinities
    # to finite sentinels so every available real lane outranks masked lanes.
    float_max: tl.constexpr = 3.4028234663852886e38
    rank_logits = tl.where(logits != logits, float_max, logits)  # noqa: PLR0124
    rank_logits = tl.maximum(tl.minimum(rank_logits, float_max), -float_max)
    available = expert[None, :] < E
    slots = tl.arange(0, BLOCK_K)
    selected_values = tl.full((BLOCK_M, BLOCK_K), -float("inf"), tl.float32)
    selected_indices = tl.zeros((BLOCK_M, BLOCK_K), tl.int32)
    for slot in tl.static_range(K):
        work = tl.where(available, rank_logits, -float("inf"))
        index = tl.argmax(work, axis=1, tie_break_left=True)
        value = tl.sum(tl.where(expert[None, :] == index[:, None], logits, 0.0), axis=1)
        selected_values = tl.where(
            slots[None, :] == slot,
            value[:, None],
            selected_values,
        )
        selected_indices = tl.where(
            slots[None, :] == slot,
            index[:, None],
            selected_indices,
        )
        available &= expert[None, :] != index[:, None]

    selected_values -= tl.max(selected_values, axis=1)[:, None]
    numerator = tl.exp(selected_values)
    scores = numerator / tl.sum(numerator, axis=1)[:, None]
    topk_mask = (row[:, None] < T) & (slots[None, :] < K)
    tl.store(
        score_ptr + row[:, None] * K + slots[None, :],
        scores,
        mask=topk_mask,
    )
    tl.store(
        index_ptr + row[:, None] * K + slots[None, :],
        selected_indices,
        mask=topk_mask,
    )


@triton.jit
def _router_logits_backward_kernel(
    base_grad_ptr,
    score_grad_ptr,
    aux_grad_ptr,
    logits_ptr,
    frequency_ptr,
    score_ptr,
    index_ptr,
    reverse_scatter_ptr,
    dlogits_ptr,
    T: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BASE_GRAD: tl.constexpr,
    HAS_SCORE_GRAD: tl.constexpr,
    HAS_AUX_GRAD: tl.constexpr,
    SCORE_GRAD_GROUPED: tl.constexpr,
):
    row = tl.program_id(0)
    expert = tl.arange(0, BLOCK_E)
    expert_mask = expert < E
    if HAS_BASE_GRAD:
        dlogits = tl.load(
            base_grad_ptr + row * E + expert,
            mask=expert_mask,
            other=0.0,
        ).to(tl.float32)
    else:
        dlogits = tl.zeros((BLOCK_E,), tl.float32)

    if HAS_AUX_GRAD:
        logits = tl.load(
            logits_ptr + row * E + expert,
            mask=expert_mask,
            other=-float("inf"),
        ).to(tl.float32)
        logits -= tl.max(logits, axis=0)
        numerator = tl.exp(logits)
        probabilities = numerator / tl.sum(numerator, axis=0)
        frequency = tl.load(
            frequency_ptr + expert,
            mask=expert_mask,
            other=0,
        ).to(tl.float32)
        frequency_sum = tl.sum(frequency, axis=0)
        expected_frequency = tl.sum(probabilities * frequency, axis=0)
        aux_grad = tl.load(aux_grad_ptr).to(tl.float32)
        aux_scale = aux_grad * E / (T * frequency_sum)
        dlogits += aux_scale * probabilities * (frequency - expected_frequency)

    if HAS_SCORE_GRAD:
        slot = tl.arange(0, BLOCK_K)
        slot_mask = slot < K
        score = tl.load(
            score_ptr + row * K + slot,
            mask=slot_mask,
            other=0.0,
        ).to(tl.float32)
        score_grad_offset = row * K + slot
        if SCORE_GRAD_GROUPED:
            score_grad_offset = tl.load(
                reverse_scatter_ptr + score_grad_offset,
                mask=slot_mask,
                other=0,
            )
        dscore = tl.load(
            score_grad_ptr + score_grad_offset,
            mask=slot_mask,
            other=0.0,
        ).to(tl.float32)
        index = tl.load(
            index_ptr + row * K + slot,
            mask=slot_mask,
            other=0,
        ).to(tl.int32)
        dot = tl.sum(score * dscore, axis=0)
        selected_grad = score * (dscore - dot)
        for k_iter in tl.static_range(K):
            current_index = tl.sum(
                tl.where(slot == k_iter, index, 0),
                axis=0,
            )
            current_grad = tl.sum(
                tl.where(slot == k_iter, selected_grad, 0.0),
                axis=0,
            )
            dlogits += tl.where(expert == current_index, current_grad, 0.0)

    tl.store(dlogits_ptr + row * E + expert, dlogits, mask=expert_mask)


def _launch_router_linear_topk(
    x: torch.Tensor,
    weight: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, hidden = x.shape
    experts = weight.shape[0]
    logits = torch.empty(tokens, experts, dtype=torch.bfloat16, device=x.device)
    scores = torch.empty(tokens, top_k, dtype=torch.float32, device=x.device)
    indices = torch.empty(tokens, top_k, dtype=torch.int32, device=x.device)
    block_e = max(16, triton.next_power_of_2(experts))
    block_k = triton.next_power_of_2(top_k)
    _router_linear_topk_kernel[(triton.cdiv(tokens, _ROUTER_BLOCK_M),)](
        x,
        weight,
        logits,
        scores,
        indices,
        T=tokens,
        H=hidden,
        E=experts,
        K=top_k,
        BLOCK_M=_ROUTER_BLOCK_M,
        BLOCK_E=block_e,
        BLOCK_K=block_k,
        BLOCK_H=_ROUTER_BLOCK_H,
        num_warps=_ROUTER_NUM_WARPS,
    )
    return logits, scores, indices


def _launch_router_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    logits: torch.Tensor,
    scores: torch.Tensor,
    indices: torch.Tensor,
    frequency: torch.Tensor | None,
    reverse_scatter: torch.Tensor | None,
    score_grad_grouped: bool,
    dlogits: torch.Tensor | None,
    dscores: torch.Tensor | None,
    daux: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = x.shape[0]
    experts = weight.shape[0]
    combined = torch.empty(tokens, experts, dtype=x.dtype, device=x.device)
    _router_logits_backward_kernel[(tokens,)](
        dlogits,
        dscores,
        daux,
        logits,
        frequency,
        scores,
        indices,
        reverse_scatter,
        combined,
        T=tokens,
        E=experts,
        K=scores.shape[1],
        BLOCK_E=triton.next_power_of_2(experts),
        BLOCK_K=triton.next_power_of_2(scores.shape[1]),
        HAS_BASE_GRAD=dlogits is not None,
        HAS_SCORE_GRAD=dscores is not None,
        HAS_AUX_GRAD=daux is not None,
        SCORE_GRAD_GROUPED=score_grad_grouped,
        num_warps=1,
    )
    return torch.mm(combined, weight), torch.mm(combined.T, x)


class _Mxfp8RouterLinearTopK(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, scores, indices = _launch_router_linear_topk(x, weight, top_k)
        ctx.save_for_backward(x, weight, scores, indices)
        ctx.mark_non_differentiable(indices)
        return logits, scores, indices

    @staticmethod
    def backward(ctx, dlogits, dscores, _):
        x, weight, scores, indices = ctx.saved_tensors
        dx, dweight = _launch_router_backward(
            x,
            weight,
            scores,
            scores,
            indices,
            None,
            None,
            False,
            dlogits,
            dscores,
            None,
        )
        return dx, dweight, None


class _Mxfp8RouterAttachSwitchAux(torch.autograd.Function):
    """Attach routing scores and switch loss to one fused router backward."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        logits: torch.Tensor,
        scores: torch.Tensor,
        indices: torch.Tensor,
        frequency: torch.Tensor,
        precomputed_loss: torch.Tensor | None,
        grouped_scores: torch.Tensor | None,
        reverse_scatter: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, experts = logits.shape
        loss = precomputed_loss
        if loss is None:
            loss = torch.empty((), dtype=torch.float32, device=logits.device)
            _switch_aux_forward_kernel[(1,)](
                logits,
                frequency,
                loss,
                logits.stride(0),
                logits.stride(1),
                T=tokens,
                E=experts,
                BLOCK_T=triton.next_power_of_2(tokens),
                BLOCK_E=triton.next_power_of_2(experts),
                num_warps=8,
            )
        score_output = scores if grouped_scores is None else grouped_scores
        saved_reverse = indices if reverse_scatter is None else reverse_scatter
        ctx.save_for_backward(
            x,
            weight,
            logits,
            scores,
            indices,
            frequency,
            saved_reverse,
        )
        ctx.score_grad_grouped = grouped_scores is not None
        return logits, score_output, loss

    @staticmethod
    def backward(ctx, dlogits, dscores, daux):
        x, weight, logits, scores, indices, frequency, reverse_scatter = (
            ctx.saved_tensors
        )
        dx, dweight = _launch_router_backward(
            x,
            weight,
            logits,
            scores,
            indices,
            frequency,
            reverse_scatter,
            ctx.score_grad_grouped,
            dlogits,
            dscores,
            daux,
        )
        return dx, dweight, None, None, None, None, None, None, None


def mxfp8_router_linear_topk(
    x: torch.Tensor,
    weight: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused BF16 router projection plus softmax-over-top-k."""
    if (
        x.ndim != 2
        or weight.ndim != 2
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or weight.shape[1] != x.shape[1]
        or weight.shape[0] > 16
        or x.shape[1] % 64
        or top_k > weight.shape[0]
    ):
        logits = F.linear(x, weight)
        values, indices = torch.topk(logits, top_k, dim=-1)
        return logits, F.softmax(values.float(), dim=-1), indices.int()
    return _Mxfp8RouterLinearTopK.apply(x, weight, top_k)


def mxfp8_router_linear_topk_raw(
    x: torch.Tensor,
    weight: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the fused router without attaching an autograd edge yet."""
    if not mxfp8_router_switch_aux_supported(x, weight, top_k):
        raise ValueError("raw fused router requested for an unsupported shape")
    return _launch_router_linear_topk(x, weight, top_k)


def mxfp8_router_attach_switch_aux(
    x: torch.Tensor,
    weight: torch.Tensor,
    logits: torch.Tensor,
    scores: torch.Tensor,
    indices: torch.Tensor,
    frequency: torch.Tensor,
    precomputed_loss: torch.Tensor | None = None,
    grouped_scores: torch.Tensor | None = None,
    reverse_scatter: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attach switch loss and one combined autograd edge after route metadata."""
    return _Mxfp8RouterAttachSwitchAux.apply(
        x,
        weight,
        logits,
        scores,
        indices,
        frequency,
        precomputed_loss,
        grouped_scores,
        reverse_scatter,
    )


def mxfp8_router_switch_aux_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
    top_k: int,
) -> bool:
    """Whether router projection, top-k and switch loss fit the fused path."""
    if (
        x.ndim != 2
        or weight.ndim != 2
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or weight.shape[1] != x.shape[1]
        or weight.shape[0] > 16
        or x.shape[1] % 64
        or top_k > weight.shape[0]
    ):
        return False
    block_t = triton.next_power_of_2(x.shape[0])
    block_e = triton.next_power_of_2(weight.shape[0])
    return block_t * block_e <= 16384


__all__ = [
    "mxfp8_router_attach_switch_aux",
    "mxfp8_router_linear_topk",
    "mxfp8_router_linear_topk_raw",
    "mxfp8_router_switch_aux_supported",
]
