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
from . import overlap as _overlap
from .mxfp8 import Mxfp8Workspace, mxfp8_experts
from .mxfp8_route_pack import Mxfp8RoutePackWorkspace, route_pack_mxfp8
from .mxfp8_transport import (
    Mxfp8TransportLayout,
    pack_mxfp8_transport,
    unpack_mxfp8_transport,
)
from .mxfp8_weighted_reduce import (
    Mxfp8WeightedReduceWorkspace,
    segmented_sum_rows_mxfp8,
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


class _GatherPairRows(torch.autograd.Function):
    """``x.index_select(0, recv_token)`` with an atomic-free exact backward.

    The default backward is ``index_add_`` over ``pair_count`` rows, which on
    large shapes (E384/K20/EP8, ~1.2M pair rows) falls onto PyTorch's slow
    64-bit-offset atomic path.  The grouped layout already records, per receive
    token, where its top-k rows live (``scatter_pos``), so the gradient is a
    deterministic segmented sum.
    """

    @staticmethod
    def forward(ctx, x, recv_token, scatter_pos, recv_tokens, top_k):
        ctx.save_for_backward(scatter_pos)
        ctx.recv_tokens = recv_tokens
        ctx.top_k = top_k
        return x.index_select(0, recv_token)

    @staticmethod
    def backward(ctx, grad_pair):
        (scatter_pos,) = ctx.saved_tensors
        grad_x = segmented_sum_rows_mxfp8(
            grad_pair, scatter_pos, ctx.recv_tokens, ctx.top_k
        )
        return grad_x, None, None, None, None


import os as _os

_SEGSUM = _os.environ.get("SONICMOE_EP_SEGSUM_DISPATCH_BWD", "1") != "0"


def _sum_rows_by_destination(rows, send_indices, counts, source_tokens):
    """Sum per-destination returned rows into ``[source_tokens, H]`` (atomic-free).

    ``send_indices`` is the destination-major concatenation of the tokens sent
    to every rank (each token at most once per destination) and ``counts`` the
    per-destination lengths.  One position table + one segmented FP32 reduction
    replaces the per-destination index_copy_/index_select/add loop.
    """
    world = len(counts)
    device = rows.device
    n = rows.shape[0]
    if n == 0 or source_tokens == 0:
        return torch.zeros(
            (source_tokens, rows.shape[1]), dtype=rows.dtype, device=device
        )
    segment = torch.repeat_interleave(
        torch.arange(world, dtype=torch.int64, device=device),
        torch.tensor(counts, dtype=torch.int64, device=device),
        output_size=n,
    )
    position = torch.full(
        (source_tokens * world,), -1, dtype=torch.int32, device=device
    )
    position[send_indices.to(torch.int64) * world + segment] = torch.arange(
        n, dtype=torch.int32, device=device
    )
    return segmented_sum_rows_mxfp8(rows, position, source_tokens, world)


def _accumulate_dedup_segments(
    grad_x: torch.Tensor,
    send_indices: torch.Tensor,
    rows: torch.Tensor,
    splits,
) -> None:
    """Add returned rows into ``grad_x`` without atomics.

    ``send_indices`` is the concatenation, per destination rank, of the
    ascending and duplicate-free token ids of that destination (the dispatch
    plan deduplicates each token once per destination), and ``splits`` are the
    per-destination row counts.  Indices are therefore unique inside a segment,
    so each segment can be applied as a gather-add-scatter.  This replaces
    ``index_add_`` on the (strided) returned payload, which takes PyTorch's slow
    generic atomic path.
    """
    start = 0
    first = True
    for count in splits:
        count = int(count)
        if count:
            index = send_indices[start : start + count]
            segment = rows[start : start + count]
            if first:
                # grad_x is freshly zeroed, so the first segment is a plain write.
                grad_x.index_copy_(0, index, segment)
                first = False
            else:
                grad_x.index_copy_(0, index, grad_x.index_select(0, index) + segment)
        start += count


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
        _overlap.fire("pre:fwd_combine")
        out = _all_to_all_rows_raw(tensor, input_splits, output_splits, group)
        _overlap.fire("post:fwd_combine")
        return out

    @staticmethod
    def backward(ctx, grad_output):
        _overlap.fire("pre:bwd_combine")
        grad_input = _all_to_all_rows_raw(
            grad_output,
            ctx.output_splits,
            ctx.input_splits,
            ctx.group,
        )
        _overlap.fire("post:bwd_combine")
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
        _overlap.fire("pre:fwd_dispatch")
        recv_qdata = _all_to_all_rows_raw(
            send_qdata.view(torch.uint8), input_splits, output_splits, group
        ).view(qdata.dtype)
        _overlap.fire("post:fwd_dispatch")
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
        _overlap.fire("pre:bwd_dispatch")
        grad_send = _all_to_all_rows_raw(
            grad_recv_x,
            ctx.output_splits,
            ctx.input_splits,
            ctx.group,
        )
        _overlap.fire("post:bwd_dispatch")
        if _SEGSUM and grad_send.dtype == torch.bfloat16 and grad_send.is_cuda:
            grad_x = _sum_rows_by_destination(
                grad_send, send_indices, ctx.input_splits, ctx.source_tokens
            )
        else:
            grad_x = torch.zeros(
                (ctx.source_tokens, grad_send.shape[1]),
                dtype=grad_send.dtype,
                device=grad_send.device,
            )
            _accumulate_dedup_segments(
                grad_x, send_indices, grad_send, ctx.input_splits
            )
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
        _accumulate_dedup_segments(grad_x, send_indices, returned_x, ctx.input_splits)
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
            _overlap.fire("pre:fwd_dispatch")
            recv_payload = _all_to_all_rows_raw(
                send_payload, input_splits, output_splits, group
            )
            _overlap.fire("post:fwd_dispatch")
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

            _overlap.fire("pre:bwd_dispatch")
            returned_payload = _all_to_all_rows_raw(
                grad_payload,
                ctx.output_splits,
                ctx.input_splits,
                ctx.group,
            )
            _overlap.fire("post:bwd_dispatch")
            returned_x = returned_payload[:, :grad_x_bytes].view(grad_recv_x.dtype)
            returned_weights = returned_payload[:, grad_x_bytes:].view(torch.float32)
            if _SEGSUM and returned_x.dtype == torch.bfloat16 and returned_x.is_cuda:
                grad_x = _sum_rows_by_destination(
                    returned_x, send_indices, ctx.input_splits, ctx.source_tokens
                )
            else:
                grad_x = torch.zeros(
                    (ctx.source_tokens, ctx.hidden),
                    dtype=grad_recv_x.dtype,
                    device=grad_recv_x.device,
                )
                _accumulate_dedup_segments(
                    grad_x, send_indices, returned_x, ctx.input_splits
                )
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


def _build_send_chunks(valid, destinations, local_ids, weights, world):
    """Per-destination deduplicated send lists with a single host sync.

    Token ``t`` is sent once to every destination that owns at least one of its
    valid experts.  The destination-major, token-ascending flat order equals the
    concatenation of per-destination ``torch.where`` lists.  Only the per
    destination counts are copied to the host (needed for the NCCL splits); the
    flat token list is then cut by a stable sort, so no ``nonzero`` sync.
    """
    tokens = valid.shape[0]
    device = valid.device
    ranks = torch.arange(world, dtype=destinations.dtype, device=device)
    mask = ((destinations.unsqueeze(-1) == ranks) & valid.unsqueeze(-1)).any(dim=1)
    mask_t = mask.t().contiguous().reshape(-1)  # [world * tokens], destination-major
    counts = tuple(int(c) for c in mask_t.view(world, tokens).sum(dim=1).tolist())
    total = sum(counts)
    order = torch.sort(mask_t.to(torch.uint8), descending=True, stable=True)[1][:total]
    token = order % tokens
    destination = (order // tokens).to(destinations.dtype)
    selected_mask = (destinations.index_select(0, token) == destination.unsqueeze(1)) & (
        valid.index_select(0, token)
    )
    selected_ids = torch.where(
        selected_mask,
        local_ids.index_select(0, token),
        torch.full_like(selected_mask, -1, dtype=local_ids.dtype),
    ).to(torch.int32)
    selected_weights = torch.where(
        selected_mask,
        weights.index_select(0, token),
        torch.zeros_like(weights.index_select(0, token)),
    )
    return counts, token, selected_ids, selected_weights


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
    send_counts_tuple, token_flat, send_ids, send_weights = _build_send_chunks(
        valid, destinations, local_ids, weights, world
    )
    recv_counts = _exchange_counts(send_counts_tuple, expert_ids.device, group)
    return ExpertParallelDispatchPlan(
        send_counts=send_counts_tuple,
        recv_counts=recv_counts,
        send_token_indices=tuple(torch.split(token_flat, send_counts_tuple)),
        send_token_indices_flat=token_flat,
        send_expert_ids=send_ids.contiguous(),
        send_weights=send_weights.contiguous(),
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
    # One nonzero (the only host sync of this path) provides the pair count and
    # every compaction index below; boolean-mask indexing would sync each time.
    valid_flat = valid.reshape(-1)
    valid_index = torch.nonzero(valid_flat).squeeze(1)
    pair_count = valid_index.numel()
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

    compact_recv_token = valid_index // top_k
    compact_weights = dispatched.weights.reshape(-1).index_select(0, valid_index)

    local_experts = w2.shape[0]
    prepacked_input = None
    if route_workspace is None:
        pair_x = dispatched.x.index_select(0, compact_recv_token)
        pair_recv_token = compact_recv_token
        pair_weights = compact_weights
        pair_expert = (
            dispatched.expert_ids.reshape(-1).index_select(0, valid_index).view(-1, 1)
        )
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
        compact_to_grouped = packed.scatter_pos.index_select(0, valid_index)
        grouped_to_compact = torch.empty_like(compact_to_grouped)
        grouped_to_compact.scatter_(
            0,
            compact_to_grouped.to(torch.int64),
            torch.arange(pair_count, dtype=torch.int32, device=dispatched.x.device),
        )
        # Training with BF16 token-space activations: do not materialise the
        # [pair_count, hidden] gather.  The expert function gathers its FC1
        # wgrad operand from ``dispatched.x`` by ``packed.recv_token`` and
        # reduces ``dx`` per token with ``packed.scatter_pos``.
        token_space_x = (
            not is_inference_mode
            and dispatched.x.dtype == torch.bfloat16
            and packed.recv_token.dtype == torch.int32
            and packed.scatter_pos.dtype == torch.int32
        )
        pair_x = (
            dispatched.x
            if token_space_x
            else torch.empty(
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
    token_space_kwargs = {}
    if route_workspace is not None and token_space_x:
        token_space_kwargs = dict(
            wgrad_row_index=packed.recv_token,
            dx_scatter_pos=packed.scatter_pos,
            dx_recv_tokens=recv_tokens,
            dx_top_k=top_k,
        )
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
        identity_reduce=route_workspace is not None,
        **token_space_kwargs,
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


class _CombineRows(torch.autograd.Function):
    """Sum the per-destination returned rows into source tokens, atomic-free.

    Equivalent to the FP32 ``index_add_`` loop over destinations followed by a
    cast to the input dtype, with one deterministic segmented reduction.  Each
    token appears at most once per destination, so a ``[tokens, world]`` table
    of returned-row positions is enough.
    """

    @staticmethod
    def forward(ctx, returned, send_token_indices_flat, send_counts, source_tokens):
        world = len(send_counts)
        device = returned.device
        rows = returned.shape[0]
        segment = torch.repeat_interleave(
            torch.arange(world, dtype=torch.int64, device=device),
            torch.tensor(send_counts, dtype=torch.int64, device=device),
            output_size=rows,
        )
        position = torch.full(
            (source_tokens * world,), -1, dtype=torch.int32, device=device
        )
        position[send_token_indices_flat * world + segment] = torch.arange(
            rows, dtype=torch.int32, device=device
        )
        ctx.save_for_backward(send_token_indices_flat)
        return segmented_sum_rows_mxfp8(returned, position, source_tokens, world)

    @staticmethod
    def backward(ctx, grad_output):
        (send_token_indices_flat,) = ctx.saved_tensors
        return grad_output.index_select(0, send_token_indices_flat), None, None, None


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
    if returned.dtype == torch.bfloat16 and returned.is_cuda:
        return _CombineRows.apply(
            returned,
            plan.send_token_indices_flat,
            tuple(plan.send_counts),
            source_tokens,
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
