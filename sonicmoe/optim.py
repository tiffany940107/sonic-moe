# Copyright (c) 2026, SonicMoE contributors.
"""Optimizers specialized for the SM100 MXFP8 training cache."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from .functional.triton_kernels.mxfp8_quant import (
    _dense_rowwise_sgd_flat_kernel,
    _dense_rowwise_sgd_flat_pair_kernel,
)

if TYPE_CHECKING:
    from .moe import MoE


class _PreparedRowwiseSgdLaunch:
    """Pointer-stable rowwise update launch with all shape work precomputed."""

    def __init__(
        self,
        parameter: torch.Tensor,
        weight_view: torch.Tensor,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        cache_entry,
        cache_name: str,
        stream_key: tuple[int, int],
        aux_parameters: tuple[torch.Tensor, ...],
    ) -> None:
        flat_block = int(os.environ.get("SONICMOE_MXFP8_SGD_FLAT_BLOCK", "4096"))
        if flat_block not in (1024, 2048, 4096, 8192):
            raise ValueError("the prepared SGD path requires a nonzero flat block")
        if parameter.numel() % flat_block:
            raise ValueError("flat SGD block must divide the expert parameter")
        if len(aux_parameters) > 3:
            raise ValueError("at most three residual parameters can ride expert SGD")
        self.parameter = parameter
        self.weight_view = weight_view
        self.qdata = qdata
        self.scale_u8 = scale.view(torch.uint8)
        self.cache_entry = cache_entry
        self.cache_name = cache_name
        self.stream_key = stream_key
        self.aux_parameters = aux_parameters
        self.aux_numels = tuple(parameter.numel() for parameter in aux_parameters)
        self.rows = parameter.shape[1]
        self.cols = parameter.shape[2]
        self.rm = self.rows // 128
        self.rk = self.cols // 128
        self.flat_block = flat_block
        self.num_warps = int(os.environ.get("SONICMOE_MXFP8_SGD_WARPS", "4"))
        self.launcher = _dense_rowwise_sgd_flat_kernel[
            (parameter.numel() // flat_block,)
        ]

    def launch(
        self,
        grad: torch.Tensor,
        learning_rate: float,
        aux_grads: tuple[torch.Tensor, ...],
    ) -> None:
        count = len(self.aux_parameters)
        aux0 = self.aux_parameters[0] if count > 0 else self.parameter
        grad0 = aux_grads[0] if count > 0 else grad
        aux1 = self.aux_parameters[1] if count > 1 else self.parameter
        grad1 = aux_grads[1] if count > 1 else grad
        aux2 = self.aux_parameters[2] if count > 2 else self.parameter
        grad2 = aux_grads[2] if count > 2 else grad
        self.launcher(
            self.parameter,
            grad,
            self.qdata,
            self.scale_u8,
            learning_rate,
            aux0,
            grad0,
            aux1,
            grad1,
            aux2,
            grad2,
            ROWS=self.rows,
            K=self.cols,
            RM=self.rm,
            RK=self.rk,
            AUX0_NUMEL=self.aux_numels[0] if count > 0 else 0,
            AUX1_NUMEL=self.aux_numels[1] if count > 1 else 0,
            AUX2_NUMEL=self.aux_numels[2] if count > 2 else 0,
            BLOCK_ELEMS=self.flat_block,
            AUX_BLOCK=256,
            num_warps=self.num_warps,
        )


class _PreparedRowwiseSgdPairLaunch:
    """One pointer-stable launch covering both expert parameters."""

    def __init__(
        self,
        first: _PreparedRowwiseSgdLaunch,
        second: _PreparedRowwiseSgdLaunch,
    ) -> None:
        if first.flat_block != second.flat_block:
            raise ValueError("paired SGD launches must use the same flat block")
        if first.num_warps != second.num_warps:
            raise ValueError("paired SGD launches must use the same warp count")
        self.first = first
        self.second = second
        self.blocks0 = first.parameter.numel() // first.flat_block
        total_blocks = self.blocks0 + second.parameter.numel() // second.flat_block
        self.launcher = _dense_rowwise_sgd_flat_pair_kernel[(total_blocks,)]

    def launch(
        self,
        grad0: torch.Tensor,
        grad1: torch.Tensor,
        learning_rate: float,
        aux_grads: tuple[torch.Tensor, ...],
    ) -> None:
        first = self.first
        second = self.second
        count = len(first.aux_parameters)
        aux0 = first.aux_parameters[0] if count > 0 else first.parameter
        aux_grad0 = aux_grads[0] if count > 0 else grad0
        aux1 = first.aux_parameters[1] if count > 1 else first.parameter
        aux_grad1 = aux_grads[1] if count > 1 else grad0
        aux2 = first.aux_parameters[2] if count > 2 else first.parameter
        aux_grad2 = aux_grads[2] if count > 2 else grad0
        self.launcher(
            first.parameter,
            grad0,
            first.qdata,
            first.scale_u8,
            second.parameter,
            grad1,
            second.qdata,
            second.scale_u8,
            learning_rate,
            aux0,
            aux_grad0,
            aux1,
            aux_grad1,
            aux2,
            aux_grad2,
            ROWS0=first.rows,
            K0=first.cols,
            RM0=first.rm,
            RK0=first.rk,
            ROWS1=second.rows,
            K1=second.cols,
            RM1=second.rm,
            RK1=second.rk,
            BLOCKS0=self.blocks0,
            AUX0_NUMEL=first.aux_numels[0] if count > 0 else 0,
            AUX1_NUMEL=first.aux_numels[1] if count > 1 else 0,
            AUX2_NUMEL=first.aux_numels[2] if count > 2 else 0,
            BLOCK_ELEMS=first.flat_block,
            AUX_BLOCK=256,
            num_warps=first.num_warps,
        )


class Mxfp8SGD(torch.optim.Optimizer):
    """Plain SGD with fused expert-weight update and MXFP8 refresh.

    This deliberately small optimizer supports one :class:`~sonicmoe.MoE`
    module, a scalar learning rate, and no momentum or weight decay.  Router
    and bias parameters ride the first expert kernel. The two large expert
    weights are updated and quantized in the same memory pass: rowwise for
    the forward-only policy, or both rowwise and dim-0 for full MXFP8. Full
    MXFP8 can accumulate expert wgrads directly in FP32 through Quack's TMA
    reduce-add path instead of materializing BF16 ``Parameter.grad`` tensors.
    """

    def __init__(
        self,
        model: MoE,
        lr: float = 1e-3,
        *,
        use_fp32_wgrad_accum: bool | None = None,
    ) -> None:
        if not isinstance(lr, (float, int)) or lr < 0:
            raise ValueError(f"invalid learning rate: {lr}")
        self.model = model
        super().__init__(model.parameters(), {"lr": float(lr)})
        policy = model._mxfp8_workspace.training_policy
        role_modes = (
            policy.fc1_dgrad,
            policy.fc2_dgrad,
            policy.fc1_wgrad,
            policy.fc2_wgrad,
        )
        if all(mode == "bf16" for mode in role_modes):
            self._dual_weight_refresh = False
        elif all(mode == "mxfp8" for mode in role_modes):
            self._dual_weight_refresh = True
        else:
            raise RuntimeError(
                "Mxfp8SGD requires either the forward_only or full mxfp8 policy"
            )
        self._workspace = model._mxfp8_workspace
        if use_fp32_wgrad_accum is None:
            accum_mode = os.environ.get("SONICMOE_MXFP8_TMA_WGRAD", "auto")
            if accum_mode not in ("auto", "0", "1"):
                raise ValueError("SONICMOE_MXFP8_TMA_WGRAD must be auto, 0, or 1")
            # The FP32 store doubles expert-gradient traffic and is slower on
            # the current B200 sparse and dense-topk reference shapes. Keep
            # auto conservative until a measured shape crossover is known.
            use_fp32_wgrad_accum = accum_mode == "1"
        self._use_fp32_wgrad_accum = use_fp32_wgrad_accum
        if self._use_fp32_wgrad_accum:
            if not self._dual_weight_refresh:
                raise RuntimeError(
                    "FP32 TMA wgrad accumulation requires the full mxfp8 policy"
                )
            self._workspace.enable_fp32_wgrad_accumulation(
                stream_key=self._workspace._stream_key(model.c_fc.weight.device)
            )
        self._expert_specs = (
            (
                "w1",
                "w1.forward",
                model.c_fc.weight,
                model.c_fc.weight.permute(1, 2, 0),
            ),
            (
                "w2",
                "w2.forward",
                model.c_proj.weight,
                model.c_proj.weight.permute(1, 2, 0),
            ),
        )
        expert_parameter_ids = {
            id(parameter) for _, _, parameter, _ in self._expert_specs
        }
        self._residual_parameters = tuple(
            parameter
            for parameter in self.param_groups[0]["params"]
            if id(parameter) not in expert_parameter_ids
        )
        self._prepared_rowwise: tuple[_PreparedRowwiseSgdLaunch, ...] | None = None
        self._prepared_rowwise_pair: _PreparedRowwiseSgdPairLaunch | None = None
        if not self._dual_weight_refresh and int(
            os.environ.get("SONICMOE_MXFP8_SGD_FLAT_BLOCK", "4096")
        ):
            stream_key = self._workspace._stream_key(model.c_fc.weight.device)
            prepared = []
            for index, (_, row_name, parameter, weight_view) in enumerate(
                self._expert_specs
            ):
                operand = self._workspace.quantize_weight(
                    row_name,
                    weight_view,
                    stream_key=stream_key,
                )
                cache_entry = self._workspace._weights[(*stream_key, row_name)]
                prepared.append(
                    _PreparedRowwiseSgdLaunch(
                        parameter,
                        weight_view,
                        operand.qdata,
                        operand.scale,
                        cache_entry,
                        row_name,
                        stream_key,
                        self._residual_parameters if index == 0 else (),
                    )
                )
            self._prepared_rowwise = tuple(prepared)
            self._prepared_rowwise_pair = _PreparedRowwiseSgdPairLaunch(*prepared)

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        if self._use_fp32_wgrad_accum:
            self._workspace.reset_fp32_wgrad_accumulation()

    def _fast_rowwise_step(self, learning_rate: float) -> bool:
        if self._prepared_rowwise is None or self._prepared_rowwise_pair is None:
            return False
        expert_grads = tuple(
            prepared.parameter.grad for prepared in self._prepared_rowwise
        )
        residual_grads = tuple(
            parameter.grad for parameter in self._residual_parameters
        )
        all_grads = (*expert_grads, *residual_grads)
        if any(grad is None or grad.is_sparse for grad in all_grads):
            return False
        self._prepared_rowwise_pair.launch(
            expert_grads[0],
            expert_grads[1],
            learning_rate,
            residual_grads,
        )
        mutated = (
            *(prepared.parameter for prepared in self._prepared_rowwise),
            *self._residual_parameters,
        )
        torch.autograd.graph.increment_version(mutated)
        for prepared in self._prepared_rowwise:
            prepared.cache_entry.stamp = (
                *self._workspace._weight_stamp(prepared.weight_view),
                -1,
            )
        return True

    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if len(self.param_groups) != 1:
            raise RuntimeError("Mxfp8SGD supports exactly one parameter group")
        group = self.param_groups[0]
        lr = group["lr"]
        if not isinstance(lr, (float, int)):
            raise TypeError("Mxfp8SGD requires a scalar Python learning rate")
        lr = float(lr)
        if closure is None and self._fast_rowwise_step(lr):
            return loss

        device = self.model.c_fc.weight.device
        stream_key = self._workspace._stream_key(device)
        residual_updates = []
        for parameter in self._residual_parameters:
            if parameter.grad is None:
                continue
            if parameter.grad.is_sparse:
                raise RuntimeError("Mxfp8SGD does not support sparse gradients")
            residual_updates.append((parameter, parameter.grad))
        fused_residuals = False
        for pair_name, row_name, parameter, weight_view in self._expert_specs:
            grad = (
                self._workspace.fp32_wgrad(pair_name)
                if self._use_fp32_wgrad_accum
                else parameter.grad
            )
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError("Mxfp8SGD does not support sparse gradients")
            update_args = (
                weight_view,
                grad.permute(1, 2, 0),
                lr,
            )
            update_kwargs = {
                "stream_key": stream_key,
                "aux_updates": tuple(residual_updates) if not fused_residuals else (),
            }
            if self._dual_weight_refresh:
                self._workspace.sgd_update_weight_pair_(
                    pair_name, *update_args, **update_kwargs
                )
            else:
                self._workspace.sgd_update_weight_(
                    row_name, *update_args, **update_kwargs
                )
            fused_residuals = True

        if residual_updates and not fused_residuals:
            torch._foreach_add_(
                [parameter for parameter, _ in residual_updates],
                [grad for _, grad in residual_updates],
                alpha=-lr,
            )
        return loss


__all__ = ["Mxfp8SGD"]
