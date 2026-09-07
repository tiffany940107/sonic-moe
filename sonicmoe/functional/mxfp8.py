# ********************************************************************************
# Copyright (c) 2026, SonicMoE contributors
# ********************************************************************************
"""Single-GPU SM100 MXFP8 expert forward and backward.

Master parameters and optimizer state remain BF16/FP32.  Every GEMM receives
a fresh OCP MXFP8 E4M3/E8M0 view whose scale vectors follow that GEMM's
reduction dimension; in particular, wgrad casts reset at every expert
boundary.
"""

from __future__ import annotations

import os

import torch
from quack.blockscaled import (
    MXFP8_E4M3,
    BlockScaledOperand,
)
from quack.epilogue.library import gated_preact_quant_mod, gated_quant_mod
from quack.gemm_interface import gemm, gemm_dact

from ..enums import ActivationType, is_glu
from .backward import _token_broadcast_backward
from .forward import _router_forward
from .triton_kernels.mxfp8_quant import (
    quantize_mxfp8_varlen_k,
    quantize_mxfp8_varlen_m,
    quantize_mxfp8_weight,
)

_SF_ATOM_M = 128
_SF_ATOM_K = 4 * MXFP8_E4M3.sf_vec_size
_AUTOTUNE = os.environ.get("SONICMOE_MXFP8_AUTOTUNE", "0") == "1"


class Mxfp8Workspace:
    """Module-owned reusable buffers and versioned quantized-weight cache.

    Entries are isolated by CUDA stream.  A parameter in-place update bumps
    ``Tensor._version`` and invalidates its cached MXFP8 view before the next
    GEMM, preserving optimizer-step semantics.
    """

    def __init__(self) -> None:
        self._scratch: dict[tuple[int, int, str], torch.Tensor] = {}
        self._weights: dict[
            tuple[int, int, str], tuple[tuple[object, ...], BlockScaledOperand]
        ] = {}

    @staticmethod
    def _stream_key(device: torch.device) -> tuple[int, int]:
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        stream = torch.cuda.current_stream(device)
        return index, stream.cuda_stream

    def tensor(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        key = (*self._stream_key(device), name)
        value = self._scratch.get(key)
        if value is None or value.shape != shape or value.dtype != dtype:
            value = torch.empty(shape, dtype=dtype, device=device)
            self._scratch[key] = value
        return value

    def quantize_weight(
        self,
        name: str,
        weight: torch.Tensor,
        *,
        dim: int = -1,
    ) -> BlockScaledOperand:
        key = (*self._stream_key(weight.device), name)
        stamp = (
            weight.data_ptr(),
            weight._version,
            tuple(weight.shape),
            tuple(weight.stride()),
            weight.dtype,
            dim,
        )
        entry = self._weights.get(key)
        if entry is None or entry[0] != stamp:
            rows = _weight_rows(weight)
            experts, m, k = rows.shape
            qdata = self.tensor(
                f"weight.{name}.q",
                tuple(rows.shape),
                MXFP8_E4M3.qdata_dtype,
                rows.device,
            )
            sf_shape = (
                (experts, m // _SF_ATOM_M, k // _SF_ATOM_K, 32, 4, 4)
                if dim == -1
                else (experts, k // _SF_ATOM_M, m // _SF_ATOM_K, 32, 4, 4)
            )
            scale = self.tensor(
                f"weight.{name}.sf",
                sf_shape,
                MXFP8_E4M3.scale_dtype,
                rows.device,
            )
            operand = quantize_mxfp8_weight(
                rows, dim=dim, qdata_out=qdata, scale_out=scale
            )
            self._weights[key] = (stamp, operand)
        else:
            operand = entry[1]
        return operand

    def clear(self) -> None:
        self._scratch.clear()
        self._weights.clear()


def _require_supported(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation_type: ActivationType,
) -> tuple[int, int, int]:
    if torch.cuda.get_device_properties(x.device).major != 10:
        raise RuntimeError("sonicmoe_mxfp8 currently requires an SM100 GPU")
    if (
        x.dtype != torch.bfloat16
        or w1.dtype != torch.bfloat16
        or w2.dtype != torch.bfloat16
    ):
        raise TypeError("sonicmoe_mxfp8 keeps BF16 master activations and weights")
    if not is_glu(activation_type):
        raise NotImplementedError("sonicmoe_mxfp8 currently supports GLU activations")
    two_intermediate, hidden, experts = w1.shape
    if two_intermediate % 2:
        raise ValueError("the gated FC1 width must be even")
    intermediate = two_intermediate // 2
    if w2.shape != (hidden, intermediate, experts):
        raise ValueError(
            f"w2 shape {tuple(w2.shape)} != ({hidden}, {intermediate}, {experts})"
        )
    for name, value in (("hidden", hidden), ("intermediate", intermediate)):
        if value % _SF_ATOM_K:
            raise ValueError(f"{name}={value} must be divisible by {_SF_ATOM_K}")
    return experts, hidden, intermediate


def _weight_rows(weight: torch.Tensor) -> torch.Tensor:
    """Sonic's (N, K, E) view back to Quack's contiguous (E, N, K)."""
    return weight.permute(2, 0, 1).contiguous()


def _varlen_m_scale_shape(total_m: int, width: int, experts: int) -> tuple[int, ...]:
    padded_rm = (total_m + _SF_ATOM_M - 1) // _SF_ATOM_M + experts - 1
    rk = (width + _SF_ATOM_K - 1) // _SF_ATOM_K
    return (1, padded_rm, rk, 32, 4, 4)


def _varlen_k_scale_shape(total_m: int, width: int, experts: int) -> tuple[int, ...]:
    padded_rk = (total_m + _SF_ATOM_M - 1) // _SF_ATOM_M + experts - 1
    rn = (width + _SF_ATOM_K - 1) // _SF_ATOM_K
    return (1, rn, padded_rk, 32, 4, 4)


def _quantize_varlen_m(
    workspace: Mxfp8Workspace,
    name: str,
    x: torch.Tensor,
    expert_offsets: torch.Tensor,
    *,
    gather_idx: torch.Tensor | None = None,
) -> BlockScaledOperand:
    total_m = x.shape[0] if gather_idx is None else gather_idx.numel()
    experts = expert_offsets.numel() - 1
    qdata = workspace.tensor(
        f"{name}.q", (total_m, x.shape[1]), MXFP8_E4M3.qdata_dtype, x.device
    )
    scale = workspace.tensor(
        f"{name}.sf",
        _varlen_m_scale_shape(total_m, x.shape[1], experts),
        MXFP8_E4M3.scale_dtype,
        x.device,
    )
    return quantize_mxfp8_varlen_m(
        x,
        expert_offsets,
        gather_idx=gather_idx,
        qdata_out=qdata,
        scale_out=scale,
    )


def _quantize_varlen_k(
    workspace: Mxfp8Workspace,
    name: str,
    x: torch.Tensor,
    expert_offsets: torch.Tensor,
    *,
    gather_idx: torch.Tensor | None = None,
) -> BlockScaledOperand:
    total_m = x.shape[0] if gather_idx is None else gather_idx.numel()
    experts = expert_offsets.numel() - 1
    qdata = workspace.tensor(
        f"{name}.q", (total_m, x.shape[1]), MXFP8_E4M3.qdata_dtype, x.device
    )
    scale = workspace.tensor(
        f"{name}.sf",
        _varlen_k_scale_shape(total_m, x.shape[1], experts),
        MXFP8_E4M3.scale_dtype,
        x.device,
    )
    return quantize_mxfp8_varlen_k(
        x,
        expert_offsets,
        gather_idx=gather_idx,
        qdata_out=qdata,
        scale_out=scale,
    )


def _expert_ids(expert_offsets: torch.Tensor, total_m: int) -> torch.Tensor:
    counts = expert_offsets[1:] - expert_offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(counts.numel(), dtype=torch.int64, device=counts.device),
        counts.to(torch.int64),
        output_size=total_m,
    )


def _grouped_sum(
    values: torch.Tensor, expert_ids: torch.Tensor, experts: int
) -> torch.Tensor:
    result = torch.zeros(
        experts, values.shape[1], dtype=torch.float32, device=values.device
    )
    result.index_add_(0, expert_ids, values.float())
    return result.to(values.dtype)


class _Mxfp8ExpertsFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w1: torch.Tensor,
        b1: torch.Tensor | None,
        w2: torch.Tensor,
        b2: torch.Tensor | None,
        topk_scores: torch.Tensor,
        expert_offsets: torch.Tensor,
        x_gather_idx: torch.Tensor,
        s_scatter_idx: torch.Tensor,
        s_reverse_scatter_idx: torch.Tensor,
        workspace: Mxfp8Workspace,
        token_count: int,
        top_k: int,
        activation_type: ActivationType,
        is_inference_mode: bool,
    ) -> torch.Tensor:
        experts, hidden, intermediate = _require_supported(x, w1, w2, activation_type)
        total_m = x_gather_idx.numel()

        x_mx = _quantize_varlen_m(
            workspace, "forward.x", x, expert_offsets, gather_idx=x_gather_idx
        )
        w1_mx = workspace.quantize_weight("w1.forward", w1)
        preact = (
            None
            if is_inference_mode
            else torch.empty(
                total_m, 2 * intermediate, dtype=torch.bfloat16, device=x.device
            )
        )
        postact_q = workspace.tensor(
            "forward.postact.q",
            (total_m, intermediate),
            MXFP8_E4M3.qdata_dtype,
            x.device,
        )
        postact_sf = workspace.tensor(
            "forward.postact.sf",
            _varlen_m_scale_shape(total_m, intermediate, experts),
            MXFP8_E4M3.scale_dtype,
            x.device,
        )
        epi_args = {"postact": postact_q, "postact_sf": postact_sf}
        if b1 is not None:
            epi_args["mRowVecBroadcast"] = b1
        fc1_mod = (
            gated_quant_mod(activation_type.value, has_rowvec=b1 is not None)
            if is_inference_mode
            else gated_preact_quant_mod(
                activation_type.value, has_rowvec=b1 is not None
            )
        )
        fc1_mod.gemm(
            x_mx.qdata,
            w1_mx.qdata,
            preact,
            epi_args=epi_args,
            tile_M=128,
            tile_N=256,
            cluster_M=1,
            cluster_N=1,
            persistent=True,
            is_dynamic_persistent=True,
            cu_seqlens_m=expert_offsets,
            SFA=x_mx.scale,
            SFB=w1_mx.scale,
            bs_format_a=MXFP8_E4M3.name,
            bs_format_b=MXFP8_E4M3.name,
        )
        postact = BlockScaledOperand.from_parts(
            postact_q, postact_sf, MXFP8_E4M3, orig_dtype=torch.bfloat16
        )

        w2_mx = workspace.quantize_weight("w2.forward", w2)
        expert_output_buffer = workspace.tensor(
            "forward.expert_output",
            (total_m, hidden),
            torch.bfloat16,
            x.device,
        )
        expert_output = gemm(
            postact,
            w2_mx.mT,
            out=expert_output_buffer,
            bias=b2,
            cu_seqlens_m=expert_offsets,
            dynamic_scheduler=False,
            tuned=_AUTOTUNE,
        )
        output = torch.empty(token_count, hidden, dtype=x.dtype, device=x.device)
        _router_forward(
            y=expert_output,
            o=output,
            topk_scores=topk_scores.reshape(-1),
            s_reverse_scatter_idx=s_reverse_scatter_idx,
            num_activated_expert_per_token_offset=None,
            varlen_K_max=top_k,
            H=hidden,
            is_varlen_K=False,
        )

        ctx.token_count = token_count
        ctx.top_k = top_k
        ctx.activation_type = activation_type
        ctx.workspace = workspace
        ctx.is_inference_mode = is_inference_mode
        if not is_inference_mode:
            ctx.save_for_backward(
                x,
                w1,
                b1,
                w2,
                b2,
                topk_scores,
                preact,
                expert_offsets,
                x_gather_idx,
                s_scatter_idx,
                s_reverse_scatter_idx,
            )
        return output

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        if ctx.is_inference_mode:
            raise RuntimeError(
                "backward is unavailable when MXFP8 inference mode is enabled"
            )
        (
            x,
            w1,
            b1,
            w2,
            b2,
            topk_scores,
            preact,
            expert_offsets,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
        ) = ctx.saved_tensors
        experts = expert_offsets.numel() - 1
        total_m = x_gather_idx.numel()
        hidden = w2.shape[0]
        workspace = ctx.workspace
        score_grouped = topk_scores.reshape(-1)[s_scatter_idx].float()

        # FC2 dgrad + gated backward + score derivative, all in one epilogue.
        dout_mx = _quantize_varlen_m(
            workspace,
            "backward.dout_m",
            dout,
            expert_offsets,
            gather_idx=x_gather_idx,
        )
        w2_dgrad_mx = workspace.quantize_weight("w2.dgrad", w2, dim=-2)
        dh_buffer = workspace.tensor(
            "backward.dh", (total_m, 2 * w2.shape[1]), torch.bfloat16, dout.device
        )
        scored_postact_buffer = workspace.tensor(
            "backward.scored_postact",
            (total_m, w2.shape[1]),
            torch.bfloat16,
            dout.device,
        )
        dh, scored_postact, dscore_grouped = gemm_dact(
            dout_mx,
            w2_dgrad_mx,
            PreAct=preact,
            activation=ctx.activation_type.value,
            colvec_scale=score_grouped,
            colvec_reduce=True,
            dx_out=dh_buffer,
            postact_out=scored_postact_buffer,
            cu_seqlens_m=expert_offsets,
            dynamic_scheduler=False,
            tuned=_AUTOTUNE,
        )

        expert_ids = _expert_ids(expert_offsets, total_m)
        db2 = None
        if b2 is not None:
            dout_grouped = dout.index_select(
                0, x_gather_idx.to(torch.int64)
            ).contiguous()
            dscore_grouped = dscore_grouped + (
                dout_grouped.float() * b2[expert_ids].float()
            ).sum(dim=-1)
            db2 = _grouped_sum(
                dout_grouped * score_grouped.to(dout.dtype).unsqueeze(-1),
                expert_ids,
                experts,
            )
        dtopk_scores = torch.empty_like(topk_scores).reshape(-1)
        dtopk_scores[s_scatter_idx] = dscore_grouped.to(dtopk_scores.dtype)
        dtopk_scores = dtopk_scores.view_as(topk_scores)

        # FC2 wgrad: (score * activation)^T @ dout, expressed as varlen-K.
        dout_wgrad_mx = _quantize_varlen_k(
            workspace,
            "backward.dout_k",
            dout,
            expert_offsets,
            gather_idx=x_gather_idx,
        )
        postact_wgrad_mx = _quantize_varlen_k(
            workspace, "backward.postact_k", scored_postact, expert_offsets
        )
        dw2_rows = gemm(
            dout_wgrad_mx.mT,
            postact_wgrad_mx,
            cu_seqlens_k=expert_offsets,
            dynamic_scheduler=False,
            tuned=_AUTOTUNE,
        )
        dw2 = dw2_rows.permute(1, 2, 0)

        # FC1 dgrad and wgrad use distinct role-correct quantizations.
        dh_mx = _quantize_varlen_m(workspace, "backward.dh_m", dh, expert_offsets)
        w1_dgrad_mx = workspace.quantize_weight("w1.dgrad", w1, dim=-2)
        dx_grouped_buffer = workspace.tensor(
            "backward.dx_grouped", (total_m, hidden), torch.bfloat16, dout.device
        )
        dx_grouped = gemm(
            dh_mx,
            w1_dgrad_mx,
            out=dx_grouped_buffer,
            cu_seqlens_m=expert_offsets,
            dynamic_scheduler=False,
            tuned=_AUTOTUNE,
        )
        x_wgrad_mx = _quantize_varlen_k(
            workspace,
            "backward.x_k",
            x,
            expert_offsets,
            gather_idx=x_gather_idx,
        )
        dh_wgrad_mx = _quantize_varlen_k(workspace, "backward.dh_k", dh, expert_offsets)
        dw1_rows = gemm(
            x_wgrad_mx.mT,
            dh_wgrad_mx,
            cu_seqlens_k=expert_offsets,
            dynamic_scheduler=False,
            tuned=_AUTOTUNE,
        )
        dw1 = dw1_rows.permute(2, 1, 0)
        db1 = _grouped_sum(dh, expert_ids, experts) if b1 is not None else None

        dx = torch.empty(ctx.token_count, hidden, dtype=dout.dtype, device=dout.device)
        _token_broadcast_backward(
            dx_reduced=dx,
            dx_expanded=dx_grouped,
            s_reverse_scatter_idx=s_reverse_scatter_idx,
            num_activated_expert_per_token_offset=None,
            varlen_K_max=ctx.top_k,
            H=hidden,
            is_varlen_K=False,
        )
        return (
            dx,
            dw1,
            db1,
            dw2,
            db2,
            dtopk_scores,
            *[None] * 9,
        )


def mxfp8_experts(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    topk_scores: torch.Tensor,
    expert_offsets: torch.Tensor,
    x_gather_idx: torch.Tensor,
    s_scatter_idx: torch.Tensor,
    s_reverse_scatter_idx: torch.Tensor,
    workspace: Mxfp8Workspace | None,
    token_count: int,
    top_k: int,
    activation_type: ActivationType,
    is_inference_mode: bool,
) -> torch.Tensor:
    workspace = Mxfp8Workspace() if workspace is None else workspace
    return _Mxfp8ExpertsFunction.apply(
        x,
        w1,
        b1,
        w2,
        b2,
        topk_scores,
        expert_offsets,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        workspace,
        token_count,
        top_k,
        activation_type,
        is_inference_mode,
    )


__all__ = ["Mxfp8Workspace", "mxfp8_experts"]
