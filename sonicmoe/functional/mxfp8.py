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
from dataclasses import dataclass

import torch
from quack.blockscaled import (
    MXFP8_E4M3,
    BlockScaledOperand,
)
from quack.epilogue.library import (
    dgated_dquant_mod,
    dgated_fp8_preact_dquant_mod,
    dgated_fp8_preact_mod,
    gated_preact_postact_quant_mod,
    gated_preact_quant_mod,
    gated_quant_mod,
)
from quack.gemm_interface import gemm, gemm_dact

from ..enums import ActivationType, is_glu
from .backward import _token_broadcast_backward
from .forward import _router_forward, _router_forward_grouped_scores
from .triton_kernels.mxfp8_quant import (
    launch_sgd_update_and_quantize_mxfp8_weight,
    launch_sgd_update_and_quantize_mxfp8_weight_dual,
    quantize_mxfp8_gather_varlen_m,
    quantize_mxfp8_varlen_dual,
    quantize_mxfp8_varlen_k,
    quantize_mxfp8_varlen_k_pair,
    quantize_mxfp8_varlen_m,
    quantize_mxfp8_weight,
    quantize_mxfp8_weight_dual,
)

_SF_ATOM_M = 128
_SF_ATOM_K = 4 * MXFP8_E4M3.sf_vec_size
_AUTOTUNE = os.environ.get("SONICMOE_MXFP8_AUTOTUNE", "0") == "1"
_FC1_TILE_M = int(os.environ.get("SONICMOE_MXFP8_FC1_TILE_M", "128"))
_FC1_TILE_N = int(os.environ.get("SONICMOE_MXFP8_FC1_TILE_N", "256"))
_FC1_CLUSTER_M = int(os.environ.get("SONICMOE_MXFP8_FC1_CLUSTER_M", "1"))
_FC1_CLUSTER_N = int(os.environ.get("SONICMOE_MXFP8_FC1_CLUSTER_N", "1"))
_FC1_DYNAMIC = os.environ.get("SONICMOE_MXFP8_FC1_DYNAMIC", "1") == "1"
_FC1_TMA_GATHER = os.environ.get("SONICMOE_MXFP8_FC1_TMA_GATHER", "1") == "1"
_FUSE_DGATED_QUANT = os.environ.get("SONICMOE_MXFP8_FUSE_DGATED_QUANT", "1") == "1"
_FUSE_VARLEN_DUAL = os.environ.get("SONICMOE_MXFP8_FUSE_VARLEN_DUAL", "1") == "1"
_FUSE_VARLEN_K_PAIR = os.environ.get("SONICMOE_MXFP8_FUSE_VARLEN_K_PAIR", "1") == "1"


def _tristate_env(name: str, default: str = "auto") -> str:
    value = os.environ.get(name, default)
    if value not in ("auto", "0", "1"):
        raise ValueError(f"{name} must be auto, 0, or 1")
    return value


_SAVE_Z_FP8 = _tristate_env("SONICMOE_MXFP8_SAVE_Z_FP8")
_FP8_C_DGATED = _tristate_env("SONICMOE_MXFP8_FP8_C_DGATED")
_FP8_C_FUSE_DQUANT = os.environ.get("SONICMOE_MXFP8_FP8_C_FUSE_DQUANT", "0") == "1"
_ZERO_MATERIAL_GATHER = os.environ.get("SONICMOE_MXFP8_ZERO_MATERIAL_GATHER", "auto")
if _ZERO_MATERIAL_GATHER not in ("auto", "0", "1"):
    raise ValueError("SONICMOE_MXFP8_ZERO_MATERIAL_GATHER must be auto, 0, or 1")


def _use_zero_material_gather(top_k: int) -> bool:
    return _ZERO_MATERIAL_GATHER == "1" or (
        _ZERO_MATERIAL_GATHER == "auto" and top_k >= 4
    )


@dataclass(frozen=True)
class Mxfp8TrainingPolicy:
    """Backward GEMM precision choices for the MXFP8 expert-forward backend.

    Expert forward GEMMs always use MXFP8. These fields independently select
    MXFP8 or BF16 for the two input-gradient and two weight-gradient GEMMs.
    """

    fc1_dgrad: str = "mxfp8"
    fc2_dgrad: str = "mxfp8"
    fc1_wgrad: str = "mxfp8"
    fc2_wgrad: str = "mxfp8"

    def __post_init__(self) -> None:
        for role, mode in (
            ("fc1_dgrad", self.fc1_dgrad),
            ("fc2_dgrad", self.fc2_dgrad),
            ("fc1_wgrad", self.fc1_wgrad),
            ("fc2_wgrad", self.fc2_wgrad),
        ):
            if mode not in ("mxfp8", "bf16"):
                raise ValueError(f"{role} must be 'mxfp8' or 'bf16', got {mode!r}")


def _use_fp8_saved_preact(policy: Mxfp8TrainingPolicy) -> bool:
    auto_enable = policy.fc2_dgrad == "mxfp8"

    def enabled(value: str) -> bool:
        return value == "1" or (value == "auto" and auto_enable)

    return enabled(_SAVE_Z_FP8) and enabled(_FP8_C_DGATED)


@dataclass
class _WeightCacheEntry:
    stamp: tuple[object, ...]
    operand: BlockScaledOperand


@dataclass
class _WeightPairCacheEntry:
    stamp: tuple[object, ...]
    operands: tuple[BlockScaledOperand, BlockScaledOperand]


def _default_training_policy() -> Mxfp8TrainingPolicy:
    mode = os.environ.get(
        "SONICMOE_MXFP8_POLICY",
        os.environ.get("SONICMOE_MXFP8_WGRAD", "auto"),
    )
    forward_only = Mxfp8TrainingPolicy(
        fc1_dgrad="bf16",
        fc2_dgrad="bf16",
        fc1_wgrad="bf16",
        fc2_wgrad="bf16",
    )
    modes = {
        "auto": forward_only,
        "mxfp8": Mxfp8TrainingPolicy(),
        "bf16": Mxfp8TrainingPolicy(fc1_wgrad="bf16", fc2_wgrad="bf16"),
        "fc1_bf16": Mxfp8TrainingPolicy(fc1_wgrad="bf16"),
        "fc2_bf16": Mxfp8TrainingPolicy(fc2_wgrad="bf16"),
        "dgrad_bf16": Mxfp8TrainingPolicy(fc1_dgrad="bf16", fc2_dgrad="bf16"),
        "fc1_dgrad_bf16": Mxfp8TrainingPolicy(
            fc1_dgrad="bf16", fc1_wgrad="bf16", fc2_wgrad="bf16"
        ),
        "fc2_dgrad_bf16": Mxfp8TrainingPolicy(
            fc2_dgrad="bf16", fc1_wgrad="bf16", fc2_wgrad="bf16"
        ),
        "forward_only": forward_only,
    }
    try:
        return modes[mode]
    except KeyError as exc:
        raise ValueError(
            "SONICMOE_MXFP8_POLICY must be auto, mxfp8, bf16, fc1_bf16, "
            "fc2_bf16, dgrad_bf16, fc1_dgrad_bf16, fc2_dgrad_bf16, or "
            f"forward_only; got {mode!r}"
        ) from exc


class Mxfp8Workspace:
    """Module-owned reusable buffers and versioned quantized-weight cache.

    Entries are isolated by CUDA stream.  A parameter in-place update bumps
    ``Tensor._version`` and invalidates its cached MXFP8 view before the next
    GEMM, preserving optimizer-step semantics.
    """

    def __init__(self, *, training_policy: Mxfp8TrainingPolicy | None = None) -> None:
        self.training_policy = (
            _default_training_policy() if training_policy is None else training_policy
        )
        self._scratch: dict[tuple[int, int, str], torch.Tensor] = {}
        self._weights: dict[tuple[int, int, str], _WeightCacheEntry] = {}
        self._weight_pairs: dict[tuple[int, int, str], _WeightPairCacheEntry] = {}

    @staticmethod
    def _stream_key(
        device: torch.device, stream: torch.cuda.Stream | None = None
    ) -> tuple[int, int]:
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        stream = torch.cuda.current_stream(device) if stream is None else stream
        return index, stream.cuda_stream

    @staticmethod
    def _weight_stamp(weight: torch.Tensor) -> tuple[object, ...]:
        return (
            weight.data_ptr(),
            weight._version,
            tuple(weight.shape),
            tuple(weight.stride()),
            weight.dtype,
        )

    def tensor(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        *,
        stream_key: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        key = (*(self._stream_key(device) if stream_key is None else stream_key), name)
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
        stream_key: tuple[int, int] | None = None,
    ) -> BlockScaledOperand:
        stream_key = (
            self._stream_key(weight.device) if stream_key is None else stream_key
        )
        key = (*stream_key, name)
        stamp = (*self._weight_stamp(weight), dim)
        entry = self._weights.get(key)
        if entry is None or entry.stamp != stamp:
            entry = self._refresh_weight(name, weight, dim, stream_key, stamp)
        return entry.operand

    def _refresh_weight(
        self,
        name: str,
        weight: torch.Tensor,
        dim: int,
        stream_key: tuple[int, int],
        stamp: tuple[object, ...],
    ) -> _WeightCacheEntry:
        rows = _weight_rows(weight)
        experts, m, k = rows.shape
        qdata = self.tensor(
            f"weight.{name}.q",
            tuple(rows.shape),
            MXFP8_E4M3.qdata_dtype,
            rows.device,
            stream_key=stream_key,
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
            stream_key=stream_key,
        )
        operand = quantize_mxfp8_weight(rows, dim=dim, qdata_out=qdata, scale_out=scale)
        entry = _WeightCacheEntry(stamp, operand)
        self._weights[(*stream_key, name)] = entry
        return entry

    def quantize_weight_pair(
        self,
        name: str,
        weight: torch.Tensor,
        *,
        stream_key: tuple[int, int] | None = None,
    ) -> tuple[BlockScaledOperand, BlockScaledOperand]:
        """Return cached forward-rowwise and dgrad-dim0 views of a weight."""
        stream_key = (
            self._stream_key(weight.device) if stream_key is None else stream_key
        )
        key = (*stream_key, name)
        stamp = self._weight_stamp(weight)
        entry = self._weight_pairs.get(key)
        if entry is None or entry.stamp != stamp:
            entry = self._refresh_weight_pair(name, weight, stream_key, stamp)
        return entry.operands

    def _refresh_weight_pair(
        self,
        name: str,
        weight: torch.Tensor,
        stream_key: tuple[int, int],
        stamp: tuple[object, ...],
    ) -> _WeightPairCacheEntry:
        rows = _weight_rows(weight)
        experts, m, k = rows.shape
        q_shape = tuple(rows.shape)
        row_sf_shape = (
            experts,
            m // _SF_ATOM_M,
            k // _SF_ATOM_K,
            32,
            4,
            4,
        )
        col_sf_shape = (
            experts,
            k // _SF_ATOM_M,
            m // _SF_ATOM_K,
            32,
            4,
            4,
        )
        operands = quantize_mxfp8_weight_dual(
            rows,
            row_qdata_out=self.tensor(
                f"weight_pair.{name}.row.q",
                q_shape,
                MXFP8_E4M3.qdata_dtype,
                rows.device,
                stream_key=stream_key,
            ),
            row_scale_out=self.tensor(
                f"weight_pair.{name}.row.sf",
                row_sf_shape,
                MXFP8_E4M3.scale_dtype,
                rows.device,
                stream_key=stream_key,
            ),
            col_qdata_out=self.tensor(
                f"weight_pair.{name}.col.q",
                q_shape,
                MXFP8_E4M3.qdata_dtype,
                rows.device,
                stream_key=stream_key,
            ),
            col_scale_out=self.tensor(
                f"weight_pair.{name}.col.sf",
                col_sf_shape,
                MXFP8_E4M3.scale_dtype,
                rows.device,
                stream_key=stream_key,
            ),
        )
        entry = _WeightPairCacheEntry(stamp, operands)
        self._weight_pairs[(*stream_key, name)] = entry
        return entry

    def sgd_update_weight_(
        self,
        name: str,
        weight: torch.Tensor,
        grad: torch.Tensor,
        learning_rate: float,
        *,
        stream_key: tuple[int, int] | None = None,
        aux_updates: tuple[tuple[torch.Tensor, torch.Tensor], ...] = (),
    ) -> BlockScaledOperand:
        """Update one BF16 expert weight and refresh its forward cache."""
        stream_key = (
            self._stream_key(weight.device) if stream_key is None else stream_key
        )
        key = (*stream_key, name)
        cached = self._weights.get(key)
        rows = _weight_rows(weight)
        grad_rows = _weight_rows(grad)
        experts, m, k = rows.shape
        qdata = self.tensor(
            f"weight.{name}.q",
            tuple(rows.shape),
            MXFP8_E4M3.qdata_dtype,
            rows.device,
            stream_key=stream_key,
        )
        scale = self.tensor(
            f"weight.{name}.sf",
            (experts, m // _SF_ATOM_M, k // _SF_ATOM_K, 32, 4, 4),
            MXFP8_E4M3.scale_dtype,
            rows.device,
            stream_key=stream_key,
        )
        launch_sgd_update_and_quantize_mxfp8_weight(
            rows,
            grad_rows,
            learning_rate,
            qdata_out=qdata,
            scale_out=scale,
            aux_updates=aux_updates,
        )
        operand = (
            cached.operand
            if cached is not None
            and cached.operand.qdata.data_ptr() == qdata.data_ptr()
            and cached.operand.scale.data_ptr() == scale.data_ptr()
            else BlockScaledOperand.from_parts(
                qdata,
                scale,
                MXFP8_E4M3,
                orig_dtype=rows.dtype,
                quant_dim=-1,
            )
        )
        # Triton writes through the tensor's storage without passing through a
        # PyTorch in-place dispatcher.  Explicitly advance the shared version
        # counter so autograd and every other cache observe the mutation.
        torch.autograd.graph.increment_version(
            (weight, *(aux_weight for aux_weight, _ in aux_updates))
        )
        stamp = (*self._weight_stamp(weight), -1)
        self._weights[key] = _WeightCacheEntry(stamp, operand)
        return operand

    def sgd_update_weight_pair_(
        self,
        name: str,
        weight: torch.Tensor,
        grad: torch.Tensor,
        learning_rate: float,
        *,
        stream_key: tuple[int, int] | None = None,
        aux_updates: tuple[tuple[torch.Tensor, torch.Tensor], ...] = (),
    ) -> tuple[BlockScaledOperand, BlockScaledOperand]:
        """Update one BF16 expert weight and refresh both training views."""
        stream_key = (
            self._stream_key(weight.device) if stream_key is None else stream_key
        )
        key = (*stream_key, name)
        cached = self._weight_pairs.get(key)
        rows = _weight_rows(weight)
        grad_rows = _weight_rows(grad)
        experts, m, k = rows.shape
        q_shape = tuple(rows.shape)
        row_sf_shape = (
            experts,
            m // _SF_ATOM_M,
            k // _SF_ATOM_K,
            32,
            4,
            4,
        )
        col_sf_shape = (
            experts,
            k // _SF_ATOM_M,
            m // _SF_ATOM_K,
            32,
            4,
            4,
        )
        row_qdata = self.tensor(
            f"weight_pair.{name}.row.q",
            q_shape,
            MXFP8_E4M3.qdata_dtype,
            rows.device,
            stream_key=stream_key,
        )
        row_scale = self.tensor(
            f"weight_pair.{name}.row.sf",
            row_sf_shape,
            MXFP8_E4M3.scale_dtype,
            rows.device,
            stream_key=stream_key,
        )
        col_qdata = self.tensor(
            f"weight_pair.{name}.col.q",
            q_shape,
            MXFP8_E4M3.qdata_dtype,
            rows.device,
            stream_key=stream_key,
        )
        col_scale = self.tensor(
            f"weight_pair.{name}.col.sf",
            col_sf_shape,
            MXFP8_E4M3.scale_dtype,
            rows.device,
            stream_key=stream_key,
        )
        launch_sgd_update_and_quantize_mxfp8_weight_dual(
            rows,
            grad_rows,
            learning_rate,
            row_qdata_out=row_qdata,
            row_scale_out=row_scale,
            col_qdata_out=col_qdata,
            col_scale_out=col_scale,
            aux_updates=aux_updates,
        )
        operands = (
            cached.operands
            if cached is not None
            and cached.operands[0].qdata.data_ptr() == row_qdata.data_ptr()
            and cached.operands[0].scale.data_ptr() == row_scale.data_ptr()
            and cached.operands[1].qdata.data_ptr() == col_qdata.data_ptr()
            and cached.operands[1].scale.data_ptr() == col_scale.data_ptr()
            else (
                BlockScaledOperand.from_parts(
                    row_qdata,
                    row_scale,
                    MXFP8_E4M3,
                    orig_dtype=rows.dtype,
                    quant_dim=-1,
                ),
                BlockScaledOperand.from_parts(
                    col_qdata,
                    col_scale,
                    MXFP8_E4M3,
                    orig_dtype=rows.dtype,
                    quant_dim=-2,
                ),
            )
        )
        torch.autograd.graph.increment_version(
            (weight, *(aux_weight for aux_weight, _ in aux_updates))
        )
        stamp = self._weight_stamp(weight)
        self._weight_pairs[key] = _WeightPairCacheEntry(stamp, operands)
        return operands

    def clear(self) -> None:
        self._scratch.clear()
        self._weights.clear()
        self._weight_pairs.clear()


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
    reverse_idx: torch.Tensor | None = None,
    top_k: int | None = None,
    zero_material_gather: bool = True,
    stream_key: tuple[int, int] | None = None,
) -> BlockScaledOperand:
    total_m = x.shape[0] if gather_idx is None else gather_idx.numel()
    experts = expert_offsets.numel() - 1
    use_gather_operand = gather_idx is not None and zero_material_gather
    q_shape = tuple(x.shape) if use_gather_operand else (total_m, x.shape[1])
    qdata = workspace.tensor(
        f"{name}.q",
        q_shape,
        MXFP8_E4M3.qdata_dtype,
        x.device,
        stream_key=stream_key,
    )
    scale = workspace.tensor(
        f"{name}.sf",
        _varlen_m_scale_shape(total_m, x.shape[1], experts),
        MXFP8_E4M3.scale_dtype,
        x.device,
        stream_key=stream_key,
    )
    if use_gather_operand:
        linear_scale = (
            None
            if reverse_idx is not None
            else workspace.tensor(
                f"{name}.linear_sf",
                (x.shape[0], x.shape[1] // MXFP8_E4M3.sf_vec_size),
                torch.uint8,
                x.device,
                stream_key=stream_key,
            )
        )
        return quantize_mxfp8_gather_varlen_m(
            x,
            expert_offsets,
            gather_idx,
            reverse_idx=reverse_idx,
            top_k=top_k,
            qdata_out=qdata,
            scale_out=scale,
            linear_scale_out=linear_scale,
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
    stream_key: tuple[int, int] | None = None,
) -> BlockScaledOperand:
    total_m = x.shape[0] if gather_idx is None else gather_idx.numel()
    experts = expert_offsets.numel() - 1
    qdata = workspace.tensor(
        f"{name}.q",
        (total_m, x.shape[1]),
        MXFP8_E4M3.qdata_dtype,
        x.device,
        stream_key=stream_key,
    )
    scale = workspace.tensor(
        f"{name}.sf",
        _varlen_k_scale_shape(total_m, x.shape[1], experts),
        MXFP8_E4M3.scale_dtype,
        x.device,
        stream_key=stream_key,
    )
    return quantize_mxfp8_varlen_k(
        x,
        expert_offsets,
        gather_idx=gather_idx,
        qdata_out=qdata,
        scale_out=scale,
    )


def _quantize_varlen_k_pair(
    workspace: Mxfp8Workspace,
    name: str,
    x0: torch.Tensor,
    x1: torch.Tensor,
    expert_offsets: torch.Tensor,
    *,
    stream_key: tuple[int, int] | None = None,
) -> tuple[BlockScaledOperand, BlockScaledOperand]:
    total_m = x0.shape[0]
    experts = expert_offsets.numel() - 1

    def outputs(label: str, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        qdata = workspace.tensor(
            f"{name}.{label}.q",
            tuple(x.shape),
            MXFP8_E4M3.qdata_dtype,
            x.device,
            stream_key=stream_key,
        )
        scale = workspace.tensor(
            f"{name}.{label}.sf",
            _varlen_k_scale_shape(total_m, x.shape[1], experts),
            MXFP8_E4M3.scale_dtype,
            x.device,
            stream_key=stream_key,
        )
        return qdata, scale

    qdata0, scale0 = outputs("first", x0)
    qdata1, scale1 = outputs("second", x1)
    return quantize_mxfp8_varlen_k_pair(
        x0,
        x1,
        expert_offsets,
        qdata0_out=qdata0,
        scale0_out=scale0,
        qdata1_out=qdata1,
        scale1_out=scale1,
    )


def _quantize_varlen_dual(
    workspace: Mxfp8Workspace,
    name: str,
    x: torch.Tensor,
    expert_offsets: torch.Tensor,
    *,
    gather_idx: torch.Tensor | None = None,
    invocation_owned_col: bool = False,
    stream_key: tuple[int, int] | None = None,
) -> tuple[BlockScaledOperand, BlockScaledOperand]:
    total_m = x.shape[0] if gather_idx is None else gather_idx.numel()
    experts = expert_offsets.numel() - 1
    q_shape = (total_m, x.shape[1])
    row_qdata = workspace.tensor(
        f"{name}.row.q",
        q_shape,
        MXFP8_E4M3.qdata_dtype,
        x.device,
        stream_key=stream_key,
    )
    row_scale = workspace.tensor(
        f"{name}.row.sf",
        _varlen_m_scale_shape(total_m, x.shape[1], experts),
        MXFP8_E4M3.scale_dtype,
        x.device,
        stream_key=stream_key,
    )
    col_scale_shape = _varlen_k_scale_shape(total_m, x.shape[1], experts)
    if invocation_owned_col:
        # The columnwise operand is saved until backward. A second live
        # forward may reuse the rowwise scratch, but must not overwrite this
        # invocation's FC1 weight-gradient input.
        col_qdata = torch.empty(
            q_shape,
            dtype=MXFP8_E4M3.qdata_dtype,
            device=x.device,
        )
        col_scale = torch.empty(
            col_scale_shape,
            dtype=MXFP8_E4M3.scale_dtype,
            device=x.device,
        )
    else:
        col_qdata = workspace.tensor(
            f"{name}.col.q",
            q_shape,
            MXFP8_E4M3.qdata_dtype,
            x.device,
            stream_key=stream_key,
        )
        col_scale = workspace.tensor(
            f"{name}.col.sf",
            col_scale_shape,
            MXFP8_E4M3.scale_dtype,
            x.device,
            stream_key=stream_key,
        )
    return quantize_mxfp8_varlen_dual(
        x,
        expert_offsets,
        gather_idx=gather_idx,
        row_qdata_out=row_qdata,
        row_scale_out=row_scale,
        col_qdata_out=col_qdata,
        col_scale_out=col_scale,
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
        stream_key: tuple[int, int],
        token_count: int,
        top_k: int,
        activation_type: ActivationType,
        is_inference_mode: bool,
        scores_are_grouped: bool,
    ) -> torch.Tensor:
        experts, hidden, intermediate = _require_supported(x, w1, w2, activation_type)
        total_m = x_gather_idx.numel()
        if (
            _FUSE_VARLEN_DUAL
            and not is_inference_mode
            and workspace.training_policy.fc1_wgrad == "mxfp8"
        ):
            x_mx, x_wgrad_mx = _quantize_varlen_dual(
                workspace,
                "forward.x",
                x,
                expert_offsets,
                gather_idx=x_gather_idx,
                invocation_owned_col=True,
                stream_key=stream_key,
            )
            fc1_a_idx = None
        else:
            zero_material_gather = (
                _use_zero_material_gather(top_k) and _FC1_CLUSTER_N == 1
            )
            x_mx = _quantize_varlen_m(
                workspace,
                "forward.x",
                x,
                expert_offsets,
                gather_idx=x_gather_idx,
                reverse_idx=s_reverse_scatter_idx,
                top_k=top_k,
                zero_material_gather=zero_material_gather,
                stream_key=stream_key,
            )
            x_wgrad_mx = None
            fc1_a_idx = x_gather_idx if zero_material_gather else None
        if is_inference_mode or workspace.training_policy.fc1_dgrad == "bf16":
            w1_mx = workspace.quantize_weight("w1.forward", w1, stream_key=stream_key)
            w1_dgrad_mx = None
        else:
            w1_mx, w1_dgrad_mx = workspace.quantize_weight_pair(
                "w1", w1, stream_key=stream_key
            )
        save_fp8_preact = not is_inference_mode and _use_fp8_saved_preact(
            workspace.training_policy
        )
        if is_inference_mode:
            preact = None
            preact_sf = None
        elif save_fp8_preact:
            # Both tensors outlive this forward. They must be invocation-owned,
            # unlike FC2's immediately consumed reusable postact workspace.
            preact = torch.empty(
                total_m,
                2 * intermediate,
                dtype=MXFP8_E4M3.qdata_dtype,
                device=x.device,
            )
            preact_sf = torch.empty(
                _varlen_m_scale_shape(total_m, 2 * intermediate, experts),
                dtype=MXFP8_E4M3.scale_dtype,
                device=x.device,
            )
        else:
            preact = torch.empty(
                total_m, 2 * intermediate, dtype=torch.bfloat16, device=x.device
            )
            preact_sf = None
        postact_q = workspace.tensor(
            "forward.postact.q",
            (total_m, intermediate),
            MXFP8_E4M3.qdata_dtype,
            x.device,
            stream_key=stream_key,
        )
        postact_sf = workspace.tensor(
            "forward.postact.sf",
            _varlen_m_scale_shape(total_m, intermediate, experts),
            MXFP8_E4M3.scale_dtype,
            x.device,
            stream_key=stream_key,
        )
        epi_args = {"postact": postact_q, "postact_sf": postact_sf}
        if preact_sf is not None:
            epi_args["preact_sf"] = preact_sf
        if b1 is not None:
            epi_args["mRowVecBroadcast"] = b1
        if is_inference_mode:
            fc1_mod = gated_quant_mod(activation_type.value, has_rowvec=b1 is not None)
        elif preact_sf is not None:
            fc1_mod = gated_preact_postact_quant_mod(
                activation_type.value, has_rowvec=b1 is not None
            )
        else:
            fc1_mod = gated_preact_quant_mod(
                activation_type.value, has_rowvec=b1 is not None
            )
        fc1_mod.gemm(
            x_mx.qdata,
            w1_mx.qdata,
            preact,
            epi_args=epi_args,
            tile_M=_FC1_TILE_M,
            tile_N=_FC1_TILE_N,
            cluster_M=_FC1_CLUSTER_M,
            cluster_N=_FC1_CLUSTER_N,
            persistent=True,
            is_dynamic_persistent=_FC1_DYNAMIC,
            cu_seqlens_m=expert_offsets,
            A_idx=fc1_a_idx,
            use_tma_gather=_FC1_TMA_GATHER and fc1_a_idx is not None,
            SFA=x_mx.scale,
            SFB=w1_mx.scale,
            bs_format_a=MXFP8_E4M3.name,
            bs_format_b=MXFP8_E4M3.name,
        )
        postact = BlockScaledOperand.from_parts(
            postact_q, postact_sf, MXFP8_E4M3, orig_dtype=torch.bfloat16
        )

        if is_inference_mode or workspace.training_policy.fc2_dgrad == "bf16":
            w2_mx = workspace.quantize_weight("w2.forward", w2, stream_key=stream_key)
            w2_dgrad_mx = None
        else:
            w2_mx, w2_dgrad_mx = workspace.quantize_weight_pair(
                "w2", w2, stream_key=stream_key
            )
        expert_output_buffer = workspace.tensor(
            "forward.expert_output",
            (total_m, hidden),
            torch.bfloat16,
            x.device,
            stream_key=stream_key,
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
        router_forward = (
            _router_forward_grouped_scores if scores_are_grouped else _router_forward
        )
        router_forward(
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
        ctx.stream_key = stream_key
        ctx.is_inference_mode = is_inference_mode
        ctx.scores_are_grouped = scores_are_grouped
        if not is_inference_mode:
            ctx.w1_dgrad_mx = w1_dgrad_mx
            ctx.w2_dgrad_mx = w2_dgrad_mx
            ctx.x_wgrad_mx = x_wgrad_mx
            ctx.save_for_backward(
                x,
                w1,
                b1,
                w2,
                b2,
                topk_scores,
                preact,
                preact_sf,
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
            preact_sf,
            expert_offsets,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
        ) = ctx.saved_tensors
        experts = expert_offsets.numel() - 1
        total_m = x_gather_idx.numel()
        hidden = w2.shape[0]
        workspace = ctx.workspace
        stream_key = ctx.stream_key
        score_grouped = (
            topk_scores.reshape(-1).float()
            if ctx.scores_are_grouped
            else topk_scores.reshape(-1)[s_scatter_idx].float()
        )
        # FC2 dgrad + gated backward + score derivative, all in one epilogue.
        dout_wgrad_mx = None
        if workspace.training_policy.fc2_dgrad == "mxfp8":
            if _FUSE_VARLEN_DUAL and workspace.training_policy.fc2_wgrad == "mxfp8":
                dgrad_a, dout_wgrad_mx = _quantize_varlen_dual(
                    workspace,
                    "backward.dout",
                    dout,
                    expert_offsets,
                    gather_idx=x_gather_idx,
                    stream_key=stream_key,
                )
                dgrad_a_idx = None
            else:
                zero_material_gather = _use_zero_material_gather(ctx.top_k)
                dgrad_a = _quantize_varlen_m(
                    workspace,
                    "backward.dout_m",
                    dout,
                    expert_offsets,
                    gather_idx=x_gather_idx,
                    reverse_idx=s_reverse_scatter_idx,
                    top_k=ctx.top_k,
                    zero_material_gather=zero_material_gather,
                    stream_key=stream_key,
                )
                dgrad_a_idx = x_gather_idx if zero_material_gather else None
            dgrad_w2 = ctx.w2_dgrad_mx
        else:
            dgrad_a = dout
            dgrad_w2 = _weight_rows(w2)
            dgrad_a_idx = x_gather_idx
        dh_buffer = workspace.tensor(
            "backward.dh",
            (total_m, 2 * w2.shape[1]),
            torch.bfloat16,
            dout.device,
            stream_key=stream_key,
        )
        scored_postact_buffer = workspace.tensor(
            "backward.scored_postact",
            (total_m, w2.shape[1]),
            torch.bfloat16,
            dout.device,
            stream_key=stream_key,
        )
        fused_dgrad_h = None
        can_fuse_dgrad_h = (
            workspace.training_policy.fc2_dgrad == "mxfp8"
            and workspace.training_policy.fc1_dgrad == "mxfp8"
        )
        fuse_bf16c_dgated_quant = (
            preact_sf is None and _FUSE_DGATED_QUANT and can_fuse_dgrad_h
        )
        fuse_fp8c_dgated_quant = (
            preact_sf is not None and _FP8_C_FUSE_DQUANT and can_fuse_dgrad_h
        )
        if workspace.training_policy.fc2_dgrad == "mxfp8":
            dgrad_a_data = dgrad_a.qdata
            dgrad_w2_data = dgrad_w2.qdata
            mainloop_kwargs = {
                "SFA": dgrad_a.scale,
                "SFB": dgrad_w2.scale,
                "bs_format_a": MXFP8_E4M3.name,
                "bs_format_b": MXFP8_E4M3.name,
            }
        else:
            dgrad_a_data = dgrad_a
            dgrad_w2_data = dgrad_w2
            mainloop_kwargs = {}
        dgated_kwargs = {
            "tuned": _AUTOTUNE,
            "dynamic_scheduler": False,
            "cu_seqlens_m": expert_offsets,
            "A_idx": dgrad_a_idx,
            "mColVecBroadcast": score_grouped,
            **mainloop_kwargs,
        }
        if fuse_bf16c_dgated_quant or fuse_fp8c_dgated_quant:
            dh_q = workspace.tensor(
                "backward.dh_m.q",
                tuple(dh_buffer.shape),
                MXFP8_E4M3.qdata_dtype,
                dout.device,
                stream_key=stream_key,
            )
            dh_sf = workspace.tensor(
                "backward.dh_m.sf",
                _varlen_m_scale_shape(total_m, dh_buffer.shape[1], experts),
                MXFP8_E4M3.scale_dtype,
                dout.device,
                stream_key=stream_key,
            )
            dquant_mod = (
                dgated_fp8_preact_dquant_mod
                if fuse_fp8c_dgated_quant
                else dgated_dquant_mod
            )
            extra_kwargs = {"preact_scale": preact_sf} if fuse_fp8c_dgated_quant else {}
            result = dquant_mod(
                ctx.activation_type.value, has_scale=True, has_reduce=True
            )(
                dgrad_a_data,
                dgrad_w2_data,
                preact,
                out={
                    "D": dh_buffer,
                    "mDQuant": dh_q,
                    "mAuxOut": scored_postact_buffer,
                },
                mDQuant_sf=dh_sf,
                **dgated_kwargs,
                **extra_kwargs,
            )
            dh = dh_buffer
            scored_postact = scored_postact_buffer
            dscore_grouped = result["mColVecReduce"]
            fused_dgrad_h = BlockScaledOperand.from_parts(
                dh_q,
                dh_sf,
                MXFP8_E4M3,
                orig_dtype=dh.dtype,
            )
        elif preact_sf is not None:
            result = dgated_fp8_preact_mod(
                ctx.activation_type.value, has_scale=True, has_reduce=True
            )(
                dgrad_a_data,
                dgrad_w2_data,
                preact,
                out={"D": dh_buffer, "mAuxOut": scored_postact_buffer},
                preact_scale=preact_sf,
                **dgated_kwargs,
            )
            dh = dh_buffer
            scored_postact = scored_postact_buffer
            dscore_grouped = result["mColVecReduce"]
        else:
            dh, scored_postact, dscore_grouped = gemm_dact(
                dgrad_a,
                dgrad_w2,
                PreAct=preact,
                activation=ctx.activation_type.value,
                colvec_scale=score_grouped,
                colvec_reduce=True,
                dx_out=dh_buffer,
                postact_out=scored_postact_buffer,
                cu_seqlens_m=expert_offsets,
                A_idx=dgrad_a_idx,
                dynamic_scheduler=False,
                tuned=_AUTOTUNE,
            )
        postact_wgrad_mx = None
        dh_wgrad_mx = None
        if (
            _FUSE_VARLEN_K_PAIR
            and workspace.training_policy.fc2_wgrad == "mxfp8"
            and workspace.training_policy.fc1_wgrad == "mxfp8"
        ):
            postact_wgrad_mx, dh_wgrad_mx = _quantize_varlen_k_pair(
                workspace,
                "backward.postact_dh_k",
                scored_postact,
                dh,
                expert_offsets,
                stream_key=stream_key,
            )

        expert_ids = (
            _expert_ids(expert_offsets, total_m)
            if b1 is not None or b2 is not None
            else None
        )
        db2 = None
        if b2 is not None:
            assert expert_ids is not None
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
        if ctx.scores_are_grouped:
            dtopk_scores = dscore_grouped.to(topk_scores.dtype).view_as(topk_scores)
        else:
            dtopk_scores = torch.empty_like(topk_scores).reshape(-1)
            dtopk_scores[s_scatter_idx] = dscore_grouped.to(dtopk_scores.dtype)
            dtopk_scores = dtopk_scores.view_as(topk_scores)

        # FC2 wgrad: (score * activation)^T @ dout. The BF16 role mode avoids
        # both segmented-K casts and lets Quack gather dout inside the GEMM.
        if workspace.training_policy.fc2_wgrad == "bf16":
            dw2_base = torch.empty(
                (experts, hidden, w2.shape[1]), dtype=dout.dtype, device=dout.device
            )
            gemm(
                dout.T,
                scored_postact,
                out=dw2_base,
                cu_seqlens_k=expert_offsets,
                A_idx=x_gather_idx,
                dynamic_scheduler=False,
                tuned=_AUTOTUNE,
            )
            dw2 = dw2_base.permute(1, 2, 0)
        else:
            if dout_wgrad_mx is None:
                dout_wgrad_mx = _quantize_varlen_k(
                    workspace,
                    "backward.dout_k",
                    dout,
                    expert_offsets,
                    gather_idx=x_gather_idx,
                    stream_key=stream_key,
                )
            if postact_wgrad_mx is None:
                postact_wgrad_mx = _quantize_varlen_k(
                    workspace,
                    "backward.postact_k",
                    scored_postact,
                    expert_offsets,
                    stream_key=stream_key,
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
        if workspace.training_policy.fc1_dgrad == "mxfp8":
            dgrad_h = (
                fused_dgrad_h
                if fused_dgrad_h is not None
                else _quantize_varlen_m(
                    workspace,
                    "backward.dh_m",
                    dh,
                    expert_offsets,
                    stream_key=stream_key,
                )
            )
            dgrad_w1 = ctx.w1_dgrad_mx
        else:
            dgrad_h = dh
            dgrad_w1 = _weight_rows(w1)
        dx_grouped_buffer = workspace.tensor(
            "backward.dx_grouped",
            (total_m, hidden),
            torch.bfloat16,
            dout.device,
            stream_key=stream_key,
        )
        dx_grouped = gemm(
            dgrad_h,
            dgrad_w1,
            out=dx_grouped_buffer,
            cu_seqlens_m=expert_offsets,
            dynamic_scheduler=False,
            tuned=_AUTOTUNE,
        )
        dw1_base = torch.empty(
            (experts, dh.shape[1], hidden), dtype=dh.dtype, device=dh.device
        )
        if workspace.training_policy.fc1_wgrad == "bf16":
            gemm(
                x.T,
                dh,
                out=dw1_base.transpose(1, 2),
                cu_seqlens_k=expert_offsets,
                A_idx=x_gather_idx,
                dynamic_scheduler=False,
                tuned=_AUTOTUNE,
            )
        else:
            x_wgrad_mx = ctx.x_wgrad_mx
            if x_wgrad_mx is None:
                x_wgrad_mx = _quantize_varlen_k(
                    workspace,
                    "backward.x_k",
                    x,
                    expert_offsets,
                    gather_idx=x_gather_idx,
                    stream_key=stream_key,
                )
            if dh_wgrad_mx is None:
                dh_wgrad_mx = _quantize_varlen_k(
                    workspace,
                    "backward.dh_k",
                    dh,
                    expert_offsets,
                    stream_key=stream_key,
                )
            # Keep the faster H-by-2I GEMM orientation, but write through a
            # view of the contiguous master-parameter layout (E, 2I, H).
            gemm(
                x_wgrad_mx.mT,
                dh_wgrad_mx,
                out=dw1_base.transpose(1, 2),
                cu_seqlens_k=expert_offsets,
                dynamic_scheduler=False,
                tuned=_AUTOTUNE,
            )
        # PermuteBackward now hands AccumulateGrad the contiguous base instead
        # of materializing the full W1 gradient a second time.
        dw1 = dw1_base.permute(1, 2, 0)
        if b1 is not None:
            assert expert_ids is not None
            db1 = _grouped_sum(dh, expert_ids, experts)
        else:
            db1 = None

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
            *[None] * 11,
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
    stream_key: tuple[int, int],
    token_count: int,
    top_k: int,
    activation_type: ActivationType,
    is_inference_mode: bool,
    scores_are_grouped: bool = False,
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
        stream_key,
        token_count,
        top_k,
        activation_type,
        is_inference_mode,
        scores_are_grouped,
    )


__all__ = ["Mxfp8TrainingPolicy", "Mxfp8Workspace", "mxfp8_experts"]
