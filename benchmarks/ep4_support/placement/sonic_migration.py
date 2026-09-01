"""Real equal-capacity migration for Sonic OCP MXFP8 expert operands.

The control plane first computes a logical expert placement.  This module is
the data-plane boundary that actually moves both FP8 values and E8M0 scales,
then installs a target-ordered shadow bank suitable for Sonic grouped GEMM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from quack.blockscaled import BlockScaledOperand


@dataclass(frozen=True)
class OperandMigration:
    operand: BlockScaledOperand
    moved_experts: int
    transferred_bytes: int
    sample_bad_bytes: int
    sampled_bytes: int


def migration_moves(
    current: torch.Tensor, target: torch.Tensor, rank: int, world_size: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Return deterministic expert IDs sent to and received from every rank."""
    current = current.to(torch.int64).cpu()
    target = target.to(torch.int64).cpu()
    if current.shape != target.shape:
        raise ValueError("current and target placements must have the same shape")
    if not 0 <= rank < world_size:
        raise ValueError("rank is outside world_size")
    send = [
        sorted(
            expert
            for expert in range(current.numel())
            if int(current[expert]) == rank
            and int(target[expert]) == destination
            and int(current[expert]) != int(target[expert])
        )
        for destination in range(world_size)
    ]
    receive = [
        sorted(
            expert
            for expert in range(current.numel())
            if int(current[expert]) == source
            and int(target[expert]) == rank
            and int(current[expert]) != int(target[expert])
        )
        for source in range(world_size)
    ]
    return send, receive


def _sample_rows(
    qdata: torch.Tensor, scale: torch.Tensor, slots: list[int]
) -> torch.Tensor:
    """Take a small deterministic byte sample from each selected expert."""
    if not slots:
        return torch.empty((0, 16), dtype=torch.uint8, device=qdata.device)
    qbytes = qdata.reshape(qdata.shape[0], -1).view(torch.uint8)
    sbytes = scale.reshape(scale.shape[0], -1).view(torch.uint8)
    qpos = torch.tensor(
        sorted({0, 1, 31, 32, qbytes.shape[1] // 2, qbytes.shape[1] - 2, qbytes.shape[1] - 1}),
        dtype=torch.int64,
        device=qdata.device,
    )
    spos = torch.tensor(
        sorted(
            {
                0,
                1,
                min(31, sbytes.shape[1] - 1),
                sbytes.shape[1] // 2,
                sbytes.shape[1] - 2,
                sbytes.shape[1] - 1,
            }
        ),
        dtype=torch.int64,
        device=qdata.device,
    )
    index = torch.tensor(slots, dtype=torch.int64, device=qdata.device)
    sample = torch.cat(
        (
            qbytes.index_select(0, index).index_select(1, qpos),
            sbytes.index_select(0, index).index_select(1, spos),
        ),
        dim=1,
    )
    if sample.shape[1] > 16:
        sample = sample[:, :16]
    elif sample.shape[1] < 16:
        sample = torch.nn.functional.pad(sample, (0, 16 - sample.shape[1]))
    return sample.contiguous()


def migrate_operand(
    operand: BlockScaledOperand,
    current: torch.Tensor,
    target: torch.Tensor,
) -> OperandMigration:
    """Move changed logical experts and return a target-ordered local bank.

    All ranks must call this collectively.  Placements must keep the same
    number of experts on every rank.  NCCL transports byte views so the E8M0
    scale dtype does not need native collective support.
    """
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    current = current.to(torch.int64).cpu()
    target = target.to(torch.int64).cpu()
    if current.shape != target.shape or current.ndim != 1:
        raise ValueError("current and target placements must be matching vectors")
    if (
        int(current.min()) < 0
        or int(target.min()) < 0
        or int(current.max()) >= world_size
        or int(target.max()) >= world_size
    ):
        raise ValueError("placement contains a rank outside the process group")
    current_capacity = torch.bincount(current, minlength=world_size)
    target_capacity = torch.bincount(target, minlength=world_size)
    if not torch.equal(current_capacity, target_capacity):
        raise ValueError("migration requires equal per-rank capacity before collectives")
    current_ids = torch.where(current == rank)[0].tolist()
    target_ids = torch.where(target == rank)[0].tolist()
    if len(current_ids) != len(target_ids) or len(current_ids) != operand.qdata.shape[0]:
        raise ValueError("migration requires equal expert capacity on every rank")

    old_slot = {expert: slot for slot, expert in enumerate(current_ids)}
    send_by_rank, receive_by_rank = migration_moves(current, target, rank, world_size)
    send_ids = [expert for experts in send_by_rank for expert in experts]
    receive_ids = [expert for experts in receive_by_rank for expert in experts]
    send_slots = [old_slot[expert] for expert in send_ids]
    index = torch.tensor(send_slots, dtype=torch.int64, device=operand.qdata.device)

    send_q = operand.qdata.index_select(0, index).contiguous()
    send_scale = operand.scale.index_select(0, index).contiguous()
    receive_q = torch.empty(
        (len(receive_ids), *operand.qdata.shape[1:]),
        dtype=operand.qdata.dtype,
        device=operand.qdata.device,
    )
    receive_scale = torch.empty(
        (len(receive_ids), *operand.scale.shape[1:]),
        dtype=operand.scale.dtype,
        device=operand.scale.device,
    )
    send_counts = [len(experts) for experts in send_by_rank]
    receive_counts = [len(experts) for experts in receive_by_rank]
    dist.all_to_all_single(
        receive_q.view(torch.uint8),
        send_q.view(torch.uint8),
        output_split_sizes=receive_counts,
        input_split_sizes=send_counts,
    )
    dist.all_to_all_single(
        receive_scale.view(torch.uint8),
        send_scale.view(torch.uint8),
        output_split_sizes=receive_counts,
        input_split_sizes=send_counts,
    )

    # The sample travels independently from the payload.  It catches ordering
    # and scale-transfer errors without duplicating the multi-GiB expert bank.
    expected_samples = _sample_rows(operand.qdata, operand.scale, send_slots)
    received_samples = torch.empty(
        (len(receive_ids), 16), dtype=torch.uint8, device=operand.qdata.device
    )
    dist.all_to_all_single(
        received_samples,
        expected_samples,
        output_split_sizes=receive_counts,
        input_split_sizes=send_counts,
    )

    incoming_slot = {expert: slot for slot, expert in enumerate(receive_ids)}
    new_q = torch.empty_like(operand.qdata)
    new_scale = torch.empty_like(operand.scale)
    sample_bad = torch.zeros((), dtype=torch.int64, device=operand.qdata.device)
    sampled = 0
    for slot, expert in enumerate(target_ids):
        if expert in old_slot:
            new_q[slot].copy_(operand.qdata[old_slot[expert]])
            new_scale[slot].copy_(operand.scale[old_slot[expert]])
            continue
        source_slot = incoming_slot[expert]
        new_q[slot].copy_(receive_q[source_slot])
        new_scale[slot].copy_(receive_scale[source_slot])
        actual = _sample_rows(receive_q, receive_scale, [source_slot])[0]
        sample_bad.add_(torch.count_nonzero(actual != received_samples[source_slot]))
        sampled += 16

    bad_value = sample_bad.clone()
    sampled_value = torch.tensor(sampled, dtype=torch.int64, device=operand.qdata.device)
    moved_value = torch.tensor(
        sum(len(experts) for experts in send_by_rank),
        dtype=torch.int64,
        device=operand.qdata.device,
    )
    dist.all_reduce(bad_value, op=dist.ReduceOp.SUM)
    dist.all_reduce(sampled_value, op=dist.ReduceOp.SUM)
    dist.all_reduce(moved_value, op=dist.ReduceOp.SUM)

    bytes_per_expert = operand.qdata[0].numel() * operand.qdata.element_size()
    bytes_per_expert += operand.scale[0].numel() * operand.scale.element_size()
    moved = int(moved_value)
    return OperandMigration(
        operand=type(operand).from_parts(new_q, new_scale, operand.format.name),
        moved_experts=moved,
        transferred_bytes=moved * bytes_per_expert,
        sample_bad_bytes=int(bad_value),
        sampled_bytes=int(sampled_value),
    )


__all__ = ["OperandMigration", "migrate_operand", "migration_moves"]
