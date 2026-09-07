# ********************************************************************************
# Copyright (c) 2026, SonicMoE contributors
# ********************************************************************************
"""Exact expert-parallel transport for the SM100 MXFP8 training backend."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from ..enums import ActivationType
from .mxfp8 import Mxfp8Workspace, mxfp8_experts
from .triton_kernels import (
    TC_topk_router_metadata_triton_workspace,
    topk_router_workspace_shape,
)
from .triton_kernels.mxfp8_quant import (
    dequantize_mxfp8_rows,
    quantize_mxfp8_rows,
)


@dataclass(frozen=True)
class ExpertParallelDispatchPlan:
    """Dynamic exact-routing plan, grouped by destination EP rank."""

    send_counts: tuple[int, ...]
    recv_counts: tuple[int, ...]
    send_token_indices: tuple[torch.Tensor, ...]
    send_token_indices_flat: torch.Tensor
    send_expert_ids: torch.Tensor
    send_weights: torch.Tensor

    @property
    def total_send_tokens(self) -> int:
        return sum(self.send_counts)

    @property
    def total_recv_tokens(self) -> int:
        return sum(self.recv_counts)


@dataclass(frozen=True)
class Mxfp8Dispatched:
    """Deduplicated receive records and their transported MXFP8 bytes."""

    x: torch.Tensor
    qdata: torch.Tensor
    scale: torch.Tensor
    expert_ids: torch.Tensor
    weights: torch.Tensor


def _all_to_all_rows_raw(
    tensor: torch.Tensor,
    input_splits: tuple[int, ...],
    output_splits: tuple[int, ...],
    group,
) -> torch.Tensor:
    output = torch.empty(
        (sum(output_splits), *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_to_all_single(
        output,
        tensor.contiguous(),
        output_split_sizes=list(output_splits),
        input_split_sizes=list(input_splits),
        group=group,
    )
    return output


class _AllToAllRows(torch.autograd.Function):
    """Autograd-symmetric variable-split all-to-all."""

    @staticmethod
    def forward(ctx, tensor, input_splits, output_splits, group):
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.group = group
        return _all_to_all_rows_raw(tensor, input_splits, output_splits, group)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _all_to_all_rows_raw(
            grad_output,
            ctx.output_splits,
            ctx.input_splits,
            ctx.group,
        )
        return grad_input, None, None, None


class _Mxfp8Dispatch(torch.autograd.Function):
    """MXFP8 forward transport with a BF16 straight-through reverse dispatch."""

    @staticmethod
    def forward(
        ctx,
        x,
        send_indices,
        send_weights,
        input_splits,
        output_splits,
        group,
    ):
        qdata, scale = quantize_mxfp8_rows(x.contiguous())
        send_qdata = qdata.index_select(0, send_indices)
        send_scale = scale.index_select(0, send_indices)
        recv_qdata = _all_to_all_rows_raw(
            send_qdata.view(torch.uint8), input_splits, output_splits, group
        ).view(qdata.dtype)
        recv_scale = _all_to_all_rows_raw(
            send_scale.view(torch.uint8), input_splits, output_splits, group
        ).view(scale.dtype)
        recv_weights = _all_to_all_rows_raw(
            send_weights, input_splits, output_splits, group
        )
        recv_x = dequantize_mxfp8_rows(recv_qdata, recv_scale, dtype=x.dtype)

        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.group = group
        ctx.source_tokens = x.shape[0]
        ctx.save_for_backward(send_indices)
        ctx.mark_non_differentiable(recv_qdata, recv_scale)
        return recv_x, recv_qdata, recv_scale, recv_weights

    @staticmethod
    def backward(
        ctx,
        grad_recv_x,
        _grad_qdata,
        _grad_scale,
        grad_recv_weights,
    ):
        (send_indices,) = ctx.saved_tensors
        grad_send_weights = _all_to_all_rows_raw(
            grad_recv_weights,
            ctx.output_splits,
            ctx.input_splits,
            ctx.group,
        )
        grad_send = _all_to_all_rows_raw(
            grad_recv_x,
            ctx.output_splits,
            ctx.input_splits,
            ctx.group,
        )
        grad_x = torch.zeros(
            (ctx.source_tokens, grad_send.shape[1]),
            dtype=grad_send.dtype,
            device=grad_send.device,
        )
        grad_x.index_add_(0, send_indices, grad_send)
        return grad_x, None, grad_send_weights, None, None, None


def _exchange_counts(
    send_counts: tuple[int, ...], device: torch.device, group
) -> tuple[int, ...]:
    send = torch.tensor(send_counts, dtype=torch.int64, device=device)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return tuple(int(value) for value in recv.cpu().tolist())


def make_expert_parallel_dispatch_plan(
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    expert_to_rank: torch.Tensor,
    expert_to_local: torch.Tensor,
    group=None,
) -> ExpertParallelDispatchPlan:
    """Build an exact plan and deduplicate each token once per destination."""
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized for expert parallelism")
    world = dist.get_world_size(group)
    if expert_ids.ndim != 2 or expert_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("expert_ids must be a two-dimensional integer tensor")
    if weights.shape != expert_ids.shape or weights.dtype != torch.float32:
        raise TypeError("weights must be float32 with the same shape as expert_ids")
    if expert_to_rank.device != expert_ids.device or expert_to_local.device != expert_ids.device:
        raise ValueError("expert maps and routing tensors must share a device")

    ids = expert_ids.to(torch.int64)
    destinations = expert_to_rank.index_select(0, ids.reshape(-1)).view_as(ids)
    local_ids = expert_to_local.index_select(0, ids.reshape(-1)).view_as(ids)
    token_chunks = []
    id_chunks = []
    weight_chunks = []
    send_counts = []
    for destination in range(world):
        route_mask = destinations == destination
        token_indices = torch.where(route_mask.any(dim=1))[0]
        selected_mask = route_mask.index_select(0, token_indices)
        selected_ids = local_ids.index_select(0, token_indices)
        selected_weights = weights.index_select(0, token_indices)
        token_chunks.append(token_indices)
        id_chunks.append(
            torch.where(
                selected_mask,
                selected_ids,
                torch.full_like(selected_ids, -1),
            ).to(torch.int32)
        )
        weight_chunks.append(
            torch.where(selected_mask, selected_weights, torch.zeros_like(selected_weights))
        )
        send_counts.append(token_indices.numel())

    send_counts_tuple = tuple(send_counts)
    recv_counts = _exchange_counts(send_counts_tuple, expert_ids.device, group)
    return ExpertParallelDispatchPlan(
        send_counts=send_counts_tuple,
        recv_counts=recv_counts,
        send_token_indices=tuple(token_chunks),
        send_token_indices_flat=torch.cat(token_chunks).to(torch.int64),
        send_expert_ids=torch.cat(id_chunks).contiguous(),
        send_weights=torch.cat(weight_chunks).contiguous(),
    )


def dispatch_mxfp8(
    x: torch.Tensor,
    plan: ExpertParallelDispatchPlan,
    group=None,
) -> Mxfp8Dispatched:
    """Send source-quantized activations, local IDs, and differentiable scores."""
    recv_x, recv_qdata, recv_scale, recv_weights = _Mxfp8Dispatch.apply(
        x,
        plan.send_token_indices_flat,
        plan.send_weights,
        plan.send_counts,
        plan.recv_counts,
        group,
    )
    recv_ids = _all_to_all_rows_raw(
        plan.send_expert_ids, plan.send_counts, plan.recv_counts, group
    )
    return Mxfp8Dispatched(
        x=recv_x,
        qdata=recv_qdata,
        scale=recv_scale,
        expert_ids=recv_ids,
        weights=recv_weights,
    )


def local_mxfp8_experts(
    dispatched: Mxfp8Dispatched,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    activation_type: ActivationType,
    workspace: Mxfp8Workspace,
    *,
    is_inference_mode: bool,
) -> torch.Tensor:
    """Compute all received local routes and reduce them per receive record."""
    recv_tokens, top_k = dispatched.expert_ids.shape
    hidden = dispatched.x.shape[1]
    valid = dispatched.expert_ids >= 0
    pair_count = int(valid.sum().item())
    if pair_count == 0:
        # Keep every local expert parameter in the graph. A completely empty
        # destination rank must materialize zero gradients rather than ``None``
        # so optimizer and sharded-training semantics match an EP1 grouped GEMM.
        parameter_zero = w1.sum() * 0.0 + w2.sum() * 0.0
        if b1 is not None:
            parameter_zero = parameter_zero + b1.sum() * 0.0
        if b2 is not None:
            parameter_zero = parameter_zero + b2.sum() * 0.0
        transport_zero = dispatched.x.sum() * 0.0 + dispatched.weights.sum() * 0.0
        return torch.zeros(
            (recv_tokens, hidden), dtype=dispatched.x.dtype, device=dispatched.x.device
        ) + (parameter_zero + transport_zero).to(dispatched.x.dtype)

    recv_rows = (
        torch.arange(recv_tokens, dtype=torch.int64, device=dispatched.x.device)
        .unsqueeze(1)
        .expand(-1, top_k)
    )
    pair_recv_token = recv_rows[valid].contiguous()
    pair_expert = dispatched.expert_ids[valid].contiguous().view(-1, 1)
    pair_x = dispatched.x.index_select(0, pair_recv_token)
    pair_weights = dispatched.weights[valid].contiguous()

    local_experts = w2.shape[0]
    expert_frequency = torch.empty(
        local_experts, dtype=torch.int32, device=dispatched.x.device
    )
    expert_offsets = torch.empty(
        local_experts + 1, dtype=torch.int32, device=dispatched.x.device
    )
    x_gather_idx = torch.empty(
        pair_count, dtype=torch.int32, device=dispatched.x.device
    )
    s_scatter_idx = torch.empty_like(x_gather_idx)
    s_reverse_scatter_idx = torch.empty_like(x_gather_idx)
    router_scratch = workspace.tensor(
        "ep.router.histogram",
        topk_router_workspace_shape(pair_count, local_experts, 1),
        torch.int32,
        dispatched.x.device,
    )
    TC_topk_router_metadata_triton_workspace(
        pair_expert,
        local_experts,
        expert_frequency,
        expert_offsets,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        router_scratch,
    )
    pair_output = mxfp8_experts(
        pair_x,
        w1.permute(1, 2, 0),
        b1,
        w2.permute(1, 2, 0),
        b2,
        torch.ones(
            (pair_count, 1), dtype=torch.float32, device=dispatched.x.device
        ),
        expert_offsets,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        workspace,
        pair_count,
        1,
        activation_type,
        is_inference_mode,
    )
    reduced = torch.zeros(
        (recv_tokens, hidden), dtype=torch.float32, device=dispatched.x.device
    )
    reduced.index_add_(
        0,
        pair_recv_token,
        pair_output.float() * pair_weights.unsqueeze(1),
    )
    return reduced.to(dispatched.x.dtype)


def combine_expert_parallel(
    local_reduced: torch.Tensor,
    plan: ExpertParallelDispatchPlan,
    source_tokens: int,
    group=None,
) -> torch.Tensor:
    """Reverse dispatch and combine one contribution per destination rank."""
    returned = _AllToAllRows.apply(
        local_reduced, plan.recv_counts, plan.send_counts, group
    )
    output = torch.zeros(
        (source_tokens, returned.shape[1]),
        dtype=torch.float32,
        device=returned.device,
    )
    offset = 0
    for count, token_indices in zip(plan.send_counts, plan.send_token_indices):
        output.index_add_(
            0,
            token_indices,
            returned[offset : offset + count].float(),
        )
        offset += count
    return output.to(local_reduced.dtype)


__all__ = [
    "ExpertParallelDispatchPlan",
    "Mxfp8Dispatched",
    "combine_expert_parallel",
    "dispatch_mxfp8",
    "local_mxfp8_experts",
    "make_expert_parallel_dispatch_plan",
]
