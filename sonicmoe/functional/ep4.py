# ********************************************************************************
# Copyright (c) 2026, SonicMoE contributors
# ********************************************************************************
"""Exact expert-parallel transport for the SM100 MXFP8 training backend."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from ..enums import ActivationType
from . import moe_general_routing_inputs
from .mxfp8 import Mxfp8Workspace, mxfp8_experts
from .mxfp8_route_pack import Mxfp8RoutePackWorkspace, route_pack_mxfp8
from .mxfp8_transport import (
    Mxfp8TransportLayout,
    pack_mxfp8_transport,
    unpack_mxfp8_transport,
)
from .mxfp8_weighted_reduce import (
    Mxfp8WeightedReduceWorkspace,
    segmented_weighted_reduce_mxfp8,
)
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


@dataclass(frozen=True)
class Bf16Dispatched:
    """Deduplicated receive records transported without activation quantization."""

    x: torch.Tensor
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
        dequantize,
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
        recv_x = (
            dequantize_mxfp8_rows(recv_qdata, recv_scale, dtype=x.dtype)
            if dequantize
            else torch.empty(recv_qdata.shape, dtype=x.dtype, device=x.device)
        )

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
        return grad_x, None, grad_send_weights, None, None, None, None


class _Bf16Dispatch(torch.autograd.Function):
    """Reference BF16 activation transport with exact reverse dispatch."""

    @staticmethod
    def forward(
        ctx,
        x,
        send_indices,
        send_expert_ids,
        send_weights,
        input_splits,
        output_splits,
        group,
    ):
        send_x = x.index_select(0, send_indices)
        recv_x = _all_to_all_rows_raw(send_x, input_splits, output_splits, group)
        recv_ids = _all_to_all_rows_raw(
            send_expert_ids, input_splits, output_splits, group
        )
        recv_weights = _all_to_all_rows_raw(
            send_weights, input_splits, output_splits, group
        )
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.group = group
        ctx.source_tokens = x.shape[0]
        ctx.save_for_backward(send_indices)
        ctx.mark_non_differentiable(recv_ids)
        return recv_x, recv_ids, recv_weights

    @staticmethod
    def backward(ctx, grad_recv_x, _grad_recv_ids, grad_recv_weights):
        (send_indices,) = ctx.saved_tensors
        returned_x = _all_to_all_rows_raw(
            grad_recv_x,
            ctx.output_splits,
            ctx.input_splits,
            ctx.group,
        )
        returned_weights = _all_to_all_rows_raw(
            grad_recv_weights,
            ctx.output_splits,
            ctx.input_splits,
            ctx.group,
        )
        grad_x = torch.zeros(
            (ctx.source_tokens, returned_x.shape[1]),
            dtype=returned_x.dtype,
            device=returned_x.device,
        )
        grad_x.index_add_(0, send_indices, returned_x)
        return grad_x, None, None, returned_weights, None, None, None


class _PackedMxfp8Dispatch(torch.autograd.Function):
    """One-collective packed forward and backward MXFP8 transport."""

    @staticmethod
    def forward(
        ctx,
        x,
        send_indices,
        send_expert_ids,
        send_weights,
        input_splits,
        output_splits,
        group,
        dequantize,
        comm_stream,
    ):
        def launch():
            qdata, scale = quantize_mxfp8_rows(x.contiguous())
            send_payload, layout = pack_mxfp8_transport(
                qdata,
                scale,
                send_indices,
                send_expert_ids,
                send_weights,
            )
            recv_payload = _all_to_all_rows_raw(
                send_payload, input_splits, output_splits, group
            )
            recv_qdata, recv_scale, _recv_ids, payload_weights = unpack_mxfp8_transport(
                recv_payload, layout
            )
            # Router scores are the differentiable field. Materialize only this
            # small field as a custom-Function output; value/scale/IDs remain
            # zero-copy strided views of the received byte records.
            recv_weights = payload_weights.contiguous()
            recv_x = (
                dequantize_mxfp8_rows(recv_qdata, recv_scale, dtype=x.dtype)
                if dequantize
                else torch.empty(
                    (recv_payload.shape[0], x.shape[1]),
                    dtype=x.dtype,
                    device=x.device,
                )
            )
            return recv_x, recv_payload, recv_weights

        if comm_stream is None:
            recv_x, recv_payload, recv_weights = launch()
        else:
            producer_stream = torch.cuda.current_stream(x.device)
            comm_stream.wait_stream(producer_stream)
            for tensor in (x, send_indices, send_expert_ids, send_weights):
                tensor.record_stream(comm_stream)
            with torch.cuda.stream(comm_stream):
                recv_x, recv_payload, recv_weights = launch()

        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.group = group
        ctx.source_tokens = x.shape[0]
        ctx.hidden = x.shape[1]
        ctx.top_k = send_expert_ids.shape[1]
        ctx.comm_stream = comm_stream
        ctx.save_for_backward(send_indices)
        ctx.mark_non_differentiable(recv_payload)
        return recv_x, recv_payload, recv_weights

    @staticmethod
    def backward(ctx, grad_recv_x, _grad_payload, grad_recv_weights):
        (send_indices,) = ctx.saved_tensors

        def launch():
            recv_records = sum(ctx.output_splits)
            grad_record_bytes = ctx.hidden * grad_recv_x.element_size() + ctx.top_k * 4
            grad_payload = torch.empty(
                (recv_records, grad_record_bytes),
                dtype=torch.uint8,
                device=grad_recv_x.device,
            )
            grad_x_bytes = ctx.hidden * grad_recv_x.element_size()
            payload_x = grad_payload[:, :grad_x_bytes].view(grad_recv_x.dtype)
            payload_weights = grad_payload[:, grad_x_bytes:].view(torch.float32)
            payload_x.copy_(grad_recv_x)
            payload_weights.copy_(grad_recv_weights)

            returned_payload = _all_to_all_rows_raw(
                grad_payload,
                ctx.output_splits,
                ctx.input_splits,
                ctx.group,
            )
            returned_x = returned_payload[:, :grad_x_bytes].view(grad_recv_x.dtype)
            returned_weights = returned_payload[:, grad_x_bytes:].view(torch.float32)
            grad_x = torch.zeros(
                (ctx.source_tokens, ctx.hidden),
                dtype=grad_recv_x.dtype,
                device=grad_recv_x.device,
            )
            grad_x.index_add_(0, send_indices, returned_x)
            return grad_x, returned_weights

        if ctx.comm_stream is None:
            grad_x, returned_weights = launch()
        else:
            autograd_stream = torch.cuda.current_stream(grad_recv_x.device)
            ctx.comm_stream.wait_stream(autograd_stream)
            grad_recv_x.record_stream(ctx.comm_stream)
            grad_recv_weights.record_stream(ctx.comm_stream)
            with torch.cuda.stream(ctx.comm_stream):
                grad_x, returned_weights = launch()
            autograd_stream.wait_stream(ctx.comm_stream)
        return (
            grad_x,
            None,
            None,
            returned_weights,
            None,
            None,
            None,
            None,
            None,
        )


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
        raise RuntimeError(
            "torch.distributed must be initialized for expert parallelism"
        )
    world = dist.get_world_size(group)
    if expert_ids.ndim != 2 or expert_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("expert_ids must be a two-dimensional integer tensor")
    if weights.shape != expert_ids.shape or weights.dtype != torch.float32:
        raise TypeError("weights must be float32 with the same shape as expert_ids")
    if (
        expert_to_rank.device != expert_ids.device
        or expert_to_local.device != expert_ids.device
    ):
        raise ValueError("expert maps and routing tensors must share a device")

    ids = expert_ids.to(torch.int64)
    # ``-1`` is the transport-level masked-slot sentinel.  Replace it only for
    # the map lookup, then keep it excluded from every destination mask below.
    # This lets callers preserve a fixed top-k record width without dispatching
    # a fake expert or a zero-weight route.
    valid = ids >= 0
    safe_ids = ids.clamp_min(0)
    destinations = expert_to_rank.index_select(0, safe_ids.reshape(-1)).view_as(ids)
    local_ids = expert_to_local.index_select(0, safe_ids.reshape(-1)).view_as(ids)
    token_chunks = []
    id_chunks = []
    weight_chunks = []
    send_counts = []
    for destination in range(world):
        route_mask = valid & (destinations == destination)
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
            torch.where(
                selected_mask, selected_weights, torch.zeros_like(selected_weights)
            )
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
    *,
    dequantize: bool = True,
    packed_transport: bool = False,
    comm_stream: torch.cuda.Stream | None = None,
) -> Mxfp8Dispatched:
    """Send source-quantized activations, local IDs, and differentiable scores."""
    if packed_transport:
        recv_x, recv_payload, recv_weights = _PackedMxfp8Dispatch.apply(
            x,
            plan.send_token_indices_flat,
            plan.send_expert_ids,
            plan.send_weights,
            plan.send_counts,
            plan.recv_counts,
            group,
            dequantize,
            comm_stream,
        )
        recv_qdata, recv_scale, recv_ids, _payload_weights = unpack_mxfp8_transport(
            recv_payload,
            Mxfp8TransportLayout(x.shape[1], plan.send_expert_ids.shape[1]),
        )
        return Mxfp8Dispatched(
            x=recv_x,
            qdata=recv_qdata,
            scale=recv_scale,
            expert_ids=recv_ids,
            weights=recv_weights,
        )
    recv_x, recv_qdata, recv_scale, recv_weights = _Mxfp8Dispatch.apply(
        x,
        plan.send_token_indices_flat,
        plan.send_weights,
        plan.send_counts,
        plan.recv_counts,
        group,
        dequantize,
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


def dispatch_bf16(
    x: torch.Tensor,
    plan: ExpertParallelDispatchPlan,
    group=None,
) -> Bf16Dispatched:
    """Reference dispatch that keeps activation rows in BF16."""
    recv_x, recv_ids, recv_weights = _Bf16Dispatch.apply(
        x,
        plan.send_token_indices_flat,
        plan.send_expert_ids,
        plan.send_weights,
        plan.send_counts,
        plan.recv_counts,
        group,
    )
    return Bf16Dispatched(recv_x, recv_ids, recv_weights)


def local_bf16_experts(
    dispatched: Bf16Dispatched,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    activation_type: ActivationType,
    *,
    is_inference_mode: bool,
) -> torch.Tensor:
    """Official Sonic BF16 local expert path for EP correctness/performance A/B."""
    recv_tokens, top_k = dispatched.expert_ids.shape
    hidden = dispatched.x.shape[1]
    valid = dispatched.expert_ids >= 0
    if not bool(valid.any().item()):
        if is_inference_mode:
            return torch.zeros(
                (recv_tokens, hidden),
                dtype=dispatched.x.dtype,
                device=dispatched.x.device,
            )
        parameter_zero = w1.sum() * 0.0 + w2.sum() * 0.0
        if b1 is not None:
            parameter_zero = parameter_zero + b1.sum() * 0.0
        if b2 is not None:
            parameter_zero = parameter_zero + b2.sum() * 0.0
        transport_zero = dispatched.x.sum() * 0.0 + dispatched.weights.sum() * 0.0
        return torch.zeros(
            (recv_tokens, hidden),
            dtype=dispatched.x.dtype,
            device=dispatched.x.device,
        ) + (parameter_zero + transport_zero).to(dispatched.x.dtype)

    recv_rows = (
        torch.arange(recv_tokens, dtype=torch.int32, device=dispatched.x.device)
        .unsqueeze(1)
        .expand(-1, top_k)
    )
    token_indices = recv_rows[valid].contiguous()
    expert_indices = dispatched.expert_ids[valid].contiguous()
    scores = dispatched.weights[valid].contiguous()
    reduced, _ = moe_general_routing_inputs(
        dispatched.x,
        scores,
        token_indices,
        expert_indices,
        w1.permute(1, 2, 0),
        b1,
        w2.permute(1, 2, 0),
        b2,
        w2.shape[0],
        torch.cuda.current_stream(dispatched.x.device).cuda_stream,
        activation_type,
        is_inference_mode,
    )
    return reduced


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
    route_workspace: Mxfp8RoutePackWorkspace | None = None,
    reduce_workspace: Mxfp8WeightedReduceWorkspace | None = None,
    zero_material_qdata: bool = True,
) -> torch.Tensor:
    """Compute all received local routes and reduce them per receive record."""
    recv_tokens, top_k = dispatched.expert_ids.shape
    hidden = dispatched.x.shape[1]
    valid = dispatched.expert_ids >= 0
    pair_count = int(valid.sum().item())
    if pair_count == 0:
        if is_inference_mode:
            return torch.zeros(
                (recv_tokens, hidden),
                dtype=dispatched.x.dtype,
                device=dispatched.x.device,
            )
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
    compact_recv_token = recv_rows[valid].contiguous()
    compact_weights = dispatched.weights[valid].contiguous()

    local_experts = w2.shape[0]
    prepacked_input = None
    if route_workspace is None:
        pair_x = dispatched.x.index_select(0, compact_recv_token)
        pair_recv_token = compact_recv_token
        pair_weights = compact_weights
        pair_expert = dispatched.expert_ids[valid].contiguous().view(-1, 1)
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
    else:
        packed = route_pack_mxfp8(
            dispatched.qdata,
            dispatched.scale,
            dispatched.expert_ids,
            dispatched.weights.detach(),
            pair_count,
            route_workspace,
            materialize_qdata=not zero_material_qdata,
        )
        expert_offsets = packed.indptr
        compact_to_grouped = packed.scatter_pos[valid.reshape(-1)].contiguous()
        grouped_to_compact = torch.empty_like(compact_to_grouped)
        grouped_to_compact.scatter_(
            0,
            compact_to_grouped.to(torch.int64),
            torch.arange(pair_count, dtype=torch.int32, device=dispatched.x.device),
        )
        pair_x = (
            torch.empty(
                (pair_count, hidden),
                dtype=dispatched.x.dtype,
                device=dispatched.x.device,
            )
            if is_inference_mode
            else dispatched.x.index_select(0, packed.recv_token)
        )
        pair_recv_token = packed.recv_token
        pair_weights = (
            packed.weights
            if is_inference_mode
            else compact_weights.index_select(0, grouped_to_compact)
        )
        identity = torch.arange(
            pair_count, dtype=torch.int32, device=dispatched.x.device
        )
        x_gather_idx = identity
        s_scatter_idx = identity
        s_reverse_scatter_idx = identity
        prepacked_input = packed.operand
    pair_output = mxfp8_experts(
        pair_x,
        w1.permute(1, 2, 0),
        b1,
        w2.permute(1, 2, 0),
        b2,
        torch.ones((pair_count, 1), dtype=torch.float32, device=dispatched.x.device),
        expert_offsets,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        workspace,
        workspace._stream_key(dispatched.x.device),
        pair_count,
        1,
        activation_type,
        is_inference_mode,
        scores_are_grouped=True,
        prepacked_input=prepacked_input,
        prepacked_a_idx=None if route_workspace is None else packed.a_idx,
    )
    if route_workspace is not None and reduce_workspace is not None:
        return segmented_weighted_reduce_mxfp8(
            pair_output,
            pair_weights,
            packed.scatter_pos,
            pair_recv_token,
            recv_tokens,
            top_k,
            reduce_workspace,
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
    "Bf16Dispatched",
    "ExpertParallelDispatchPlan",
    "Mxfp8Dispatched",
    "combine_expert_parallel",
    "dispatch_bf16",
    "dispatch_mxfp8",
    "local_bf16_experts",
    "local_mxfp8_experts",
    "make_expert_parallel_dispatch_plan",
]
