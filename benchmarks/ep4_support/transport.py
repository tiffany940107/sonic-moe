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
