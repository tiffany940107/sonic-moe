# Copyright (c) 2026, SonicMoE contributors.
"""Low-launch switch auxiliary loss for the SM100 MXFP8 route."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _switch_aux_forward_kernel(
    logits_ptr,
    frequency_ptr,
    loss_ptr,
    stride_m: tl.constexpr,
    stride_e: tl.constexpr,
    T: tl.constexpr,
    E: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_T)
    cols = tl.arange(0, BLOCK_E)
    row_mask = rows < T
    col_mask = cols < E
    safe_rows = tl.minimum(rows, T - 1)
    logits = tl.load(
        logits_ptr + safe_rows[:, None] * stride_m + cols[None, :] * stride_e,
        mask=col_mask[None, :],
        other=-float("inf"),
    ).to(tl.float32)
    logits -= tl.max(logits, axis=1)[:, None]
    numerator = tl.exp(logits)
    probabilities = numerator / tl.sum(numerator, axis=1)[:, None]
    probabilities = tl.where(row_mask[:, None], probabilities, 0.0)
    frequency = tl.load(frequency_ptr + cols, mask=col_mask, other=0).to(tl.float32)
    weighted_per_token = tl.sum(probabilities * frequency[None, :], axis=1)
    frequency_sum = tl.sum(frequency, axis=0)
    loss = E * tl.sum(weighted_per_token, axis=0) / (T * frequency_sum)
    tl.store(loss_ptr, loss)


@triton.jit
def _switch_aux_backward_kernel(
    logits_ptr,
    frequency_ptr,
    upstream_ptr,
    dlogits_ptr,
    stride_m: tl.constexpr,
    stride_e: tl.constexpr,
    T: tl.constexpr,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_E)
    mask = cols < E
    logits = tl.load(
        logits_ptr + row * stride_m + cols * stride_e,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    logits -= tl.max(logits, axis=0)
    numerator = tl.exp(logits)
    probabilities = numerator / tl.sum(numerator, axis=0)
    frequency = tl.load(frequency_ptr + cols, mask=mask, other=0).to(tl.float32)
    frequency_sum = tl.sum(frequency, axis=0)
    expected_frequency = tl.sum(probabilities * frequency, axis=0)
    upstream = tl.load(upstream_ptr).to(tl.float32)
    scale = upstream * E / (T * frequency_sum)
    dlogits = scale * probabilities * (frequency - expected_frequency)
    tl.store(
        dlogits_ptr + row * stride_m + cols * stride_e,
        dlogits,
        mask=mask,
    )


class _Mxfp8SwitchAuxLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        router_logits: torch.Tensor,
        expert_frequency: torch.Tensor,
    ) -> torch.Tensor:
        tokens, experts = router_logits.shape
        block_t = triton.next_power_of_2(tokens)
        block_e = triton.next_power_of_2(experts)
        loss = torch.empty((), dtype=torch.float32, device=router_logits.device)
        _switch_aux_forward_kernel[(1,)](
            router_logits,
            expert_frequency,
            loss,
            router_logits.stride(0),
            router_logits.stride(1),
            T=tokens,
            E=experts,
            BLOCK_T=block_t,
            BLOCK_E=block_e,
            num_warps=8,
        )
        ctx.save_for_backward(router_logits, expert_frequency)
        ctx.tokens = tokens
        ctx.experts = experts
        return loss

    @staticmethod
    def backward(ctx, upstream: torch.Tensor):
        router_logits, expert_frequency = ctx.saved_tensors
        dlogits = torch.empty_like(router_logits)
        _switch_aux_backward_kernel[(ctx.tokens,)](
            router_logits,
            expert_frequency,
            upstream,
            dlogits,
            router_logits.stride(0),
            router_logits.stride(1),
            T=ctx.tokens,
            E=ctx.experts,
            BLOCK_E=triton.next_power_of_2(ctx.experts),
            num_warps=1,
        )
        return dlogits, None


def mxfp8_switch_aux_loss(
    router_logits: torch.Tensor,
    expert_frequency: torch.Tensor,
) -> torch.Tensor:
    """Switch loss fast path for the common small-expert single-GPU case.

    The forward uses one CTA only when the complete probability tile remains
    modest. Larger expert/token counts retain PyTorch's general reduction.
    """
    if router_logits.ndim != 2:
        raise ValueError("router_logits must be two-dimensional")
    tokens, experts = router_logits.shape
    block_t = triton.next_power_of_2(tokens)
    block_e = triton.next_power_of_2(experts)
    if (
        router_logits.is_cuda
        and router_logits.is_contiguous()
        and expert_frequency.is_cuda
        and expert_frequency.dtype == torch.int32
        and expert_frequency.shape == (experts,)
        and block_t * block_e <= 16384
    ):
        return _Mxfp8SwitchAuxLoss.apply(router_logits, expert_frequency)

    probabilities = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    accumulated = probabilities.sum(dim=0)
    return (
        experts
        * (
            torch.nn.functional.normalize(accumulated, p=1, dim=0)
            * torch.nn.functional.normalize(expert_frequency.float(), p=1, dim=0)
        ).sum()
    )


__all__ = ["mxfp8_switch_aux_loss"]
