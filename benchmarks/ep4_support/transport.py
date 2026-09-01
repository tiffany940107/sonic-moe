from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DispatchPlan:
    send_counts: list[int]
    recv_counts: list[int]
    send_token_indices: list[torch.Tensor]
    send_expert_ids: torch.Tensor
    send_weights: torch.Tensor
    local_expert_map: torch.Tensor
    send_token_indices_flat: torch.Tensor

    @property
    def total_send_tokens(self) -> int:
        return sum(self.send_counts)

    @property
    def total_recv_tokens(self) -> int:
        return sum(self.recv_counts)


@dataclass
class Dispatched:
    x: torch.Tensor
    expert_ids: torch.Tensor
    weights: torch.Tensor
    scales: torch.Tensor | None = None


@dataclass
class TransportWorkspace:
    """Caller-owned buffers for a static EP dispatch/combine plan.

    The benchmark route plan is immutable during steady state.  Keeping these
    tensors alive avoids allocating the activation/scale gather, all-to-all
    receive, returned contribution and final combine buffers on every layer
    invocation.  ``scales`` is optional so the same API can cover BF16 and
    prequantized MXFP8 transport without changing the legacy path.
    """

    send_x: torch.Tensor
    recv_x: torch.Tensor
    recv_ids: torch.Tensor
    recv_weights: torch.Tensor
    send_scales: torch.Tensor | None
    recv_scales: torch.Tensor | None
    returned: torch.Tensor
    output: torch.Tensor

    @property
    def nbytes(self) -> int:
        tensors = (
            self.send_x,
            self.recv_x,
            self.recv_ids,
            self.recv_weights,
            self.send_scales,
            self.recv_scales,
            self.returned,
            self.output,
        )
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensors
            if tensor is not None
        )


def local_expert_indices(placement: torch.Tensor, ep_size: int) -> torch.Tensor:
    """Map each logical expert to its physical slot within its destination rank."""
    placement = placement.to(torch.int64).cpu()
    out = torch.empty_like(placement)
    for rank in range(ep_size):
        experts = torch.where(placement == rank)[0]
        out[experts] = torch.arange(experts.numel(), dtype=torch.int64)
    return out


def exchange_counts(send_counts: list[int], device: torch.device) -> list[int]:
    send = torch.tensor(send_counts, dtype=torch.int64, device=device)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
    return [int(v) for v in recv.cpu().tolist()]


def make_dispatch_plan(
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    placement: torch.Tensor,
    ep_size: int,
    device: torch.device,
    *,
    pair_destinations: torch.Tensor | None = None,
    pair_local_ids: torch.Tensor | None = None,
) -> DispatchPlan:
    """Deduplicate source activations per destination while retaining local pairs."""
    ids = expert_ids.to(device=device, dtype=torch.int64)
    weights = weights.to(device=device, dtype=torch.float32)
    placement_gpu = placement.to(device=device, dtype=torch.int64)
    local_map = local_expert_indices(placement, ep_size).to(device)
    destinations = (
        placement_gpu[ids]
        if pair_destinations is None
        else pair_destinations.to(device=device, dtype=torch.int64)
    )
    physical_ids = (
        local_map[ids]
        if pair_local_ids is None
        else pair_local_ids.to(device=device, dtype=torch.int64)
    )
    if destinations.shape != ids.shape or physical_ids.shape != ids.shape:
        raise ValueError("physical pair maps must match expert_ids shape")
    token_indices: list[torch.Tensor] = []
    id_chunks: list[torch.Tensor] = []
    weight_chunks: list[torch.Tensor] = []
    send_counts: list[int] = []
    for rank in range(ep_size):
        mask = destinations == rank
        selected = torch.where(mask.any(dim=1))[0]
        token_indices.append(selected)
        send_counts.append(selected.numel())
        selected_mask = mask[selected]
        selected_ids = ids[selected]
        local_ids = torch.where(
            selected_mask,
            physical_ids[selected],
            torch.full_like(selected_ids, -1),
        ).to(torch.int32)
        local_weights = torch.where(selected_mask, weights[selected], torch.zeros_like(weights[selected]))
        id_chunks.append(local_ids)
        weight_chunks.append(local_weights)
    recv_counts = exchange_counts(send_counts, device)
    return DispatchPlan(
        send_counts,
        recv_counts,
        token_indices,
        torch.cat(id_chunks, dim=0),
        torch.cat(weight_chunks, dim=0),
        local_map,
        torch.cat(token_indices, dim=0),
    )


def allocate_transport_workspace(
    plan: DispatchPlan,
    x: torch.Tensor,
    scales: torch.Tensor | None,
    source_tokens: int,
    output_hidden: int,
    output_dtype: torch.dtype,
) -> TransportWorkspace:
    """Allocate exact-capacity steady-state buffers for ``plan`` once."""

    if x.ndim < 1 or x.shape[0] != source_tokens:
        raise ValueError("x must contain exactly source_tokens rows")
    if scales is not None and scales.shape[0] != source_tokens:
        raise ValueError("scales must have the same leading extent as x")
    send_shape = (plan.total_send_tokens, *x.shape[1:])
    recv_shape = (plan.total_recv_tokens, *x.shape[1:])
    ids_shape = (plan.total_recv_tokens, *plan.send_expert_ids.shape[1:])
    weights_shape = (plan.total_recv_tokens, *plan.send_weights.shape[1:])
    send_scales = None
    recv_scales = None
    if scales is not None:
        send_scales = torch.empty(
            (plan.total_send_tokens, *scales.shape[1:]),
            dtype=scales.dtype,
            device=scales.device,
        )
        recv_scales = torch.empty(
            (plan.total_recv_tokens, *scales.shape[1:]),
            dtype=scales.dtype,
            device=scales.device,
        )
    return TransportWorkspace(
        send_x=torch.empty(send_shape, dtype=x.dtype, device=x.device),
        recv_x=torch.empty(recv_shape, dtype=x.dtype, device=x.device),
        recv_ids=torch.empty(
            ids_shape, dtype=plan.send_expert_ids.dtype, device=x.device
        ),
        recv_weights=torch.empty(
            weights_shape, dtype=plan.send_weights.dtype, device=x.device
        ),
        send_scales=send_scales,
        recv_scales=recv_scales,
        returned=torch.empty(
            (plan.total_send_tokens, output_hidden),
            dtype=output_dtype,
            device=x.device,
        ),
        output=torch.empty(
            (source_tokens, output_hidden), dtype=output_dtype, device=x.device
        ),
    )


def all_to_all_rows(
    tensor: torch.Tensor, send_counts: list[int], recv_counts: list[int]
) -> torch.Tensor:
    shape = (sum(recv_counts), *tensor.shape[1:])
    out = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
    dist.all_to_all_single(
        out,
        tensor.contiguous(),
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
    )
    return out


def all_to_all_rows_out(
    tensor: torch.Tensor,
    send_counts: list[int],
    recv_counts: list[int],
    out: torch.Tensor,
) -> torch.Tensor:
    """All-to-all into an exact caller-owned output tensor."""

    expected_shape = (sum(recv_counts), *tensor.shape[1:])
    if tuple(out.shape) != expected_shape:
        raise ValueError(f"out shape {tuple(out.shape)} != expected {expected_shape}")
    if out.dtype != tensor.dtype or out.device != tensor.device:
        raise ValueError("all-to-all input/output dtype and device must match")
    if not tensor.is_contiguous() or not out.is_contiguous():
        raise ValueError("all-to-all _out path requires contiguous tensors")
    dist.all_to_all_single(
        out,
        tensor,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
    )
    return out


def dispatch_deduplicated(
    x: torch.Tensor, plan: DispatchPlan, scales: torch.Tensor | None = None
) -> Dispatched:
    send_x = torch.cat([x[index] for index in plan.send_token_indices], dim=0)
    recv_x = all_to_all_rows(send_x, plan.send_counts, plan.recv_counts)
    recv_ids = all_to_all_rows(plan.send_expert_ids, plan.send_counts, plan.recv_counts)
    recv_weights = all_to_all_rows(plan.send_weights, plan.send_counts, plan.recv_counts)
    recv_scales = None
    if scales is not None:
        send_scales = torch.cat(
            [scales[index] for index in plan.send_token_indices], dim=0
        )
        recv_scales = all_to_all_rows(send_scales, plan.send_counts, plan.recv_counts)
    return Dispatched(recv_x, recv_ids, recv_weights, recv_scales)


def dispatch_deduplicated_out(
    x: torch.Tensor,
    plan: DispatchPlan,
    workspace: TransportWorkspace,
    scales: torch.Tensor | None = None,
) -> Dispatched:
    """Dispatch through a static plan without steady-state tensor allocation."""

    torch.index_select(x, 0, plan.send_token_indices_flat, out=workspace.send_x)
    recv_x = all_to_all_rows_out(
        workspace.send_x, plan.send_counts, plan.recv_counts, workspace.recv_x
    )
    recv_ids = all_to_all_rows_out(
        plan.send_expert_ids,
        plan.send_counts,
        plan.recv_counts,
        workspace.recv_ids,
    )
    recv_weights = all_to_all_rows_out(
        plan.send_weights,
        plan.send_counts,
        plan.recv_counts,
        workspace.recv_weights,
    )
    recv_scales = None
    if scales is not None:
        if workspace.send_scales is None or workspace.recv_scales is None:
            raise ValueError("transport workspace was allocated without scales")
        torch.index_select(
            scales, 0, plan.send_token_indices_flat, out=workspace.send_scales
        )
        recv_scales = all_to_all_rows_out(
            workspace.send_scales,
            plan.send_counts,
            plan.recv_counts,
            workspace.recv_scales,
        )
    elif workspace.send_scales is not None:
        raise ValueError("transport workspace expects a scale tensor")
    return Dispatched(recv_x, recv_ids, recv_weights, recv_scales)


def expand_and_sort(dispatched: Dispatched, local_experts: int):
    valid = dispatched.expert_ids >= 0
    recv_token, slot = torch.where(valid)
    expert = dispatched.expert_ids[recv_token, slot].to(torch.int64)
    weights = dispatched.weights[recv_token, slot]
    order = torch.argsort(expert, stable=True)
    expert = expert[order]
    recv_token = recv_token[order]
    weights = weights[order]
    pair_x = dispatched.x[recv_token]
    counts = torch.bincount(expert, minlength=local_experts).to(torch.int32)
    indptr = torch.empty(local_experts + 1, dtype=torch.int32, device=pair_x.device)
    indptr[0] = 0
    torch.cumsum(counts, 0, out=indptr[1:])
    return pair_x, expert, recv_token, weights, indptr


def combine_deduplicated(
    recv_reduced: torch.Tensor,
    plan: DispatchPlan,
    source_tokens: int,
) -> torch.Tensor:
    returned = all_to_all_rows(recv_reduced, plan.recv_counts, plan.send_counts)
    out = torch.zeros(
        (source_tokens, recv_reduced.shape[1]),
        dtype=recv_reduced.dtype,
        device=recv_reduced.device,
    )
    offset = 0
    for count, token_index in zip(plan.send_counts, plan.send_token_indices):
        out.index_add_(0, token_index, returned[offset : offset + count])
        offset += count
    return out


def combine_deduplicated_out(
    recv_reduced: torch.Tensor,
    plan: DispatchPlan,
    source_tokens: int,
    workspace: TransportWorkspace,
) -> torch.Tensor:
    """Return and combine expert outputs in caller-owned buffers."""

    if tuple(workspace.output.shape) != (source_tokens, recv_reduced.shape[1]):
        raise ValueError("transport output workspace shape does not match combine")
    returned = all_to_all_rows_out(
        recv_reduced,
        plan.recv_counts,
        plan.send_counts,
        workspace.returned,
    )
    out = workspace.output.zero_()
    offset = 0
    for count, token_index in zip(plan.send_counts, plan.send_token_indices):
        out.index_add_(0, token_index, returned[offset : offset + count])
        offset += count
    return out


def actual_dispatch_bytes(
    plan: DispatchPlan,
    hidden: int,
    *,
    x_bytes: int = 2,
    scale_bytes_per_token: int = 0,
) -> int:
    rank = dist.get_rank() if dist.is_initialized() else 0
    # Activation + K int32 IDs + K fp32 weights per deduplicated record.
    record_bytes = (
        hidden * x_bytes
        + scale_bytes_per_token
        + plan.send_expert_ids.shape[1] * 8
    )
    return sum(
        count * record_bytes for destination, count in enumerate(plan.send_counts) if destination != rank
    )
