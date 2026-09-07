# ********************************************************************************
# Copyright (c) 2026, SonicMoE contributors
# ********************************************************************************
"""Expert-sharded SM100 MXFP8 MoE module."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .enums import ActivationType, is_glu
from .functional import TC_Softmax_Topk_Router_Function
from .functional.ep4 import (
    combine_expert_parallel,
    dispatch_mxfp8,
    local_mxfp8_experts,
    make_expert_parallel_dispatch_plan,
)
from .functional.mxfp8 import Mxfp8Workspace
from .moe import Experts, MoE


def _placement_maps(
    placement: torch.Tensor, world_size: int
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    placement = placement.to(dtype=torch.int64, device="cpu").contiguous()
    experts = placement.numel()
    if experts == 0 or experts % world_size:
        raise ValueError("global experts must be positive and divisible by EP size")
    if bool(((placement < 0) | (placement >= world_size)).any()):
        raise ValueError("placement contains an invalid EP rank")
    local_experts = experts // world_size
    local_ids = torch.empty_like(placement)
    logical_by_rank = []
    for rank in range(world_size):
        logical = torch.where(placement == rank)[0]
        if logical.numel() != local_experts:
            raise ValueError("every EP rank must own the same number of experts")
        logical_by_rank.append(logical)
        local_ids[logical] = torch.arange(local_experts, dtype=torch.int64)
    return placement, local_ids, logical_by_rank


class ExpertParallelMoE(nn.Module):
    """Exact top-k EP module with BF16 masters and MXFP8 expert GEMMs.

    The router is replicated. Expert parameters are sharded according to
    ``placement``; by default experts are contiguous and EP4 rank ``r`` owns
    ``[r * E/4, (r + 1) * E/4)``.
    """

    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        intermediate_size: int,
        activation_function: ActivationType,
        add_bias: bool,
        std: float,
        *,
        process_group=None,
        placement: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("initialize torch.distributed before ExpertParallelMoE")
        self.process_group = process_group
        self.ep_size = dist.get_world_size(process_group)
        self.ep_rank = dist.get_rank(process_group)
        if num_experts % self.ep_size:
            raise ValueError("num_experts must be divisible by the EP group size")
        if not 0 < num_experts_per_tok <= num_experts:
            raise ValueError("num_experts_per_tok must be in [1, num_experts]")
        if hidden_size % 128 or intermediate_size % 128:
            raise ValueError("SM100 MXFP8 requires hidden/intermediate divisible by 128")
        if not is_glu(activation_function):
            raise NotImplementedError("SM100 MXFP8 EP currently supports GLU activations")

        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_function = activation_function
        self.local_experts = num_experts // self.ep_size

        default_placement = torch.arange(num_experts) // self.local_experts
        placement_cpu, local_map, logical_by_rank = _placement_maps(
            default_placement if placement is None else placement,
            self.ep_size,
        )
        self.register_buffer("expert_to_rank", placement_cpu, persistent=True)
        self.register_buffer("expert_to_local", local_map, persistent=True)
        self.register_buffer(
            "local_expert_ids", logical_by_rank[self.ep_rank], persistent=True
        )
        self.register_buffer(
            "placement_version", torch.zeros((), dtype=torch.int64), persistent=True
        )

        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.c_fc = Experts(
            self.local_experts,
            hidden_size,
            2 * intermediate_size,
            add_bias=add_bias,
            std=std,
        )
        self.c_proj = Experts(
            self.local_experts,
            intermediate_size,
            hidden_size,
            add_bias=add_bias,
            std=std,
        )
        self._mxfp8_workspace = Mxfp8Workspace()

    @classmethod
    def from_moe(
        cls,
        moe: MoE,
        *,
        process_group=None,
        placement: torch.Tensor | None = None,
    ) -> ExpertParallelMoE:
        """Create an expert shard and copy parameters from a full ``MoE``."""
        module = cls(
            num_experts=moe.num_experts,
            num_experts_per_tok=moe.top_k,
            hidden_size=moe.hidden_size,
            intermediate_size=moe.intermediate_size,
            activation_function=moe.activation_function,
            add_bias=moe.c_fc.bias is not None,
            std=moe.c_fc.std,
            process_group=process_group,
            placement=placement,
        ).to(device=moe.c_fc.weight.device, dtype=moe.c_fc.weight.dtype)
        module.load_from_full_(moe)
        module.train(moe.training)
        return module

    @torch.no_grad()
    def load_from_full_(self, moe: MoE) -> None:
        """Copy the replicated router and this rank's logical expert rows."""
        if (
            moe.num_experts != self.num_experts
            or moe.hidden_size != self.hidden_size
            or moe.intermediate_size != self.intermediate_size
            or moe.top_k != self.top_k
        ):
            raise ValueError("full MoE configuration does not match the EP module")
        logical = self.local_expert_ids.to(moe.c_fc.weight.device)
        self.router.weight.copy_(moe.router.weight)
        self.c_fc.weight.copy_(moe.c_fc.weight.index_select(0, logical))
        self.c_proj.weight.copy_(moe.c_proj.weight.index_select(0, logical))
        if self.c_fc.bias is not None:
            if moe.c_fc.bias is None or moe.c_proj.bias is None:
                raise ValueError("full MoE is missing expert biases")
            self.c_fc.bias.copy_(moe.c_fc.bias.index_select(0, logical))
            self.c_proj.bias.copy_(moe.c_proj.bias.index_select(0, logical))
        self._mxfp8_workspace.clear()

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        is_inference_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        original_shape = hidden_states.shape
        x = hidden_states.view(-1, self.hidden_size)
        router_logits = F.linear(x, self.router.weight)
        topk_scores, topk_indices = TC_Softmax_Topk_Router_Function.apply(
            router_logits,
            self.num_experts,
            self.top_k,
            True,
            False,
        )
        plan = make_expert_parallel_dispatch_plan(
            topk_indices,
            topk_scores,
            self.expert_to_rank,
            self.expert_to_local,
            self.process_group,
        )
        dispatched = dispatch_mxfp8(x, plan, self.process_group)
        local_reduced = local_mxfp8_experts(
            dispatched,
            self.c_fc.weight,
            self.c_fc.bias,
            self.c_proj.weight,
            self.c_proj.bias,
            self.activation_function,
            self._mxfp8_workspace,
            is_inference_mode=is_inference_mode or not self.training,
        )
        output = combine_expert_parallel(
            local_reduced, plan, x.shape[0], self.process_group
        ).view(original_shape)

        aux_loss = None
        if not is_inference_mode:
            expert_frequency = torch.bincount(
                topk_indices.reshape(-1).to(torch.int64),
                minlength=self.num_experts,
            ).to(torch.float32)
            probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
            aux_loss = self.num_experts * (
                F.normalize(probs.sum(0), p=1, dim=0)
                * F.normalize(expert_frequency, p=1, dim=0)
            ).sum()
        return output, aux_loss

    @torch.no_grad()
    def all_reduce_replicated_gradients_(self, *, average: bool = False) -> None:
        """Synchronize the replicated router after backward when desired."""
        if self.router.weight.grad is None:
            return
        dist.all_reduce(self.router.weight.grad, group=self.process_group)
        if average:
            self.router.weight.grad.div_(self.ep_size)

    def extra_repr(self) -> str:
        return (
            f"global_experts={self.num_experts}, local_experts={self.local_experts}, "
            f"top_k={self.top_k}, ep_rank={self.ep_rank}, ep_size={self.ep_size}"
        )


__all__ = ["ExpertParallelMoE"]
