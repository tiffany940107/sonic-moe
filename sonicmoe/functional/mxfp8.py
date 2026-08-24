# ********************************************************************************
# Copyright (c) 2026 SonicMoE contributors
# ********************************************************************************
"""SM120 MXFP8 inference helpers for routed, variable-M expert GEMMs.

The public SonicMoE path is BF16 today, while QuACK already exposes the three
pieces needed by an MXFP8 MoE forward on SM120:

* MXFP8 E4M3 values with E8M0 scales over 32 consecutive K elements;
* variable-M (one segment per expert) block-scaled GEMMs; and
* a fused GEMM + SwiGLU + MXFP8 post-activation epilogue.

This module connects those pieces without owning expert-parallel transport.
An EP runtime dispatches and groups rows by local expert, then calls the two
stage interface below.  ``cu_seqlens_m`` is the expert indptr for those rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from quack.blockscaled import (
    MXFP8_E4M3,
    BlockScaledOperand,
    dequant_operand,
    pack_scale_2d_to_blocked_contig,
    to_mx_compiled,
    unpack_scale_blocked_to_2d,
)
from quack.epilogue.library import identity_epi, swiglu_quant_mod

MXFP8_FORMAT = MXFP8_E4M3.name
MXFP8_SCALE_BLOCK_K = MXFP8_E4M3.sf_vec_size
MXFP8_SCALE_ATOM_M = 128


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class Mxfp8MoEKernelConfig:
    """Explicit SM120 tile choices, kept outside the timed forward path."""

    fc1_tile_m: int = 128
    fc1_tile_n: int = 128
    fc2_tile_m: int = 128
    fc2_tile_n: int = 128
    pingpong: bool = True
    dynamic_persistent: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("fc1_tile_m", self.fc1_tile_m),
            ("fc1_tile_n", self.fc1_tile_n),
            ("fc2_tile_m", self.fc2_tile_m),
            ("fc2_tile_n", self.fc2_tile_n),
        ):
            if value not in (128, 256):
                raise ValueError(
                    f"{name} must be 128 or 256 for the SM120 block-scaled path"
                )
        if self.fc1_tile_n % (2 * MXFP8_SCALE_BLOCK_K):
            raise ValueError("FC1 tile N must cover whole gated MXFP8 scale vectors")


_DEFAULT_CONFIG = Mxfp8MoEKernelConfig()


def quantize_mxfp8_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize contiguous ``(..., K)`` BF16/FP32 rows to OCP MXFP8.

    Returns E4M3 values and a linear E8M0 scale array with shape
    ``(..., K // 32)``.  The caller may transport these bytes directly and
    pack the scales into QuACK's blocked layout after expert grouping.
    """

    if x.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"MXFP8 source must be BF16/FP32, got {x.dtype}")
    if not x.is_contiguous():
        x = x.contiguous()
    if x.shape[-1] % MXFP8_SCALE_BLOCK_K:
        raise ValueError(
            f"K={x.shape[-1]} must be divisible by {MXFP8_SCALE_BLOCK_K} for MXFP8"
        )
    return to_mx_compiled(x, MXFP8_SCALE_BLOCK_K)


def pack_varlen_m_scales(
    linear_scales: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
) -> torch.Tensor:
    """Pack row-linear scales into QuACK's tile-padded varlen-M SFA layout.

    Expert ``e`` starts at scale row ``(cu[e] // 128 + e) * 128``.  Quantized
    values remain densely concatenated; only scale rows are padded so a 512 B
    scale atom never crosses an expert boundary.
    """

    if linear_scales.ndim != 2 or linear_scales.dtype != torch.float8_e8m0fnu:
        raise TypeError("linear_scales must be a 2-D torch.float8_e8m0fnu tensor")
    if cu_seqlens_m.ndim != 1 or cu_seqlens_m.dtype != torch.int32:
        raise TypeError("cu_seqlens_m must be a one-dimensional int32 tensor")
    if linear_scales.device != cu_seqlens_m.device:
        raise ValueError("linear scales and cu_seqlens_m must be on the same device")

    total_m = linear_scales.shape[0]
    num_experts = cu_seqlens_m.shape[0] - 1
    if num_experts <= 0:
        raise ValueError("cu_seqlens_m must describe at least one expert")

    counts = (cu_seqlens_m[1:] - cu_seqlens_m[:-1]).to(torch.int64)
    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, device=linear_scales.device, dtype=torch.int64),
        counts,
        output_size=total_m,
    )
    starts = cu_seqlens_m[:-1].to(torch.int64)
    padded_starts = (
        starts // MXFP8_SCALE_ATOM_M
        + torch.arange(num_experts, device=linear_scales.device, dtype=torch.int64)
    ) * MXFP8_SCALE_ATOM_M
    destination = torch.arange(total_m, device=linear_scales.device, dtype=torch.int64)
    destination = destination + (padded_starts - starts)[expert_ids]

    padded_rm = _ceil_div(total_m, MXFP8_SCALE_ATOM_M) + num_experts - 1
    padded = torch.zeros(
        (padded_rm * MXFP8_SCALE_ATOM_M, linear_scales.shape[1]),
        dtype=linear_scales.dtype,
        device=linear_scales.device,
    )
    if total_m:
        # PyTorch does not expose CUDA index_copy for float8_e8m0fnu.  E8M0
        # is a one-byte exponent encoding, so move its bytes losslessly and
        # retain the dtype at the API boundary.
        padded.view(torch.uint8).index_copy_(
            0, destination, linear_scales.contiguous().view(torch.uint8)
        )
    return pack_scale_2d_to_blocked_contig(padded.unsqueeze(0))


def make_varlen_m_operand(
    qdata: torch.Tensor,
    linear_scales: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
) -> BlockScaledOperand:
    """Build an MXFP8 operand for rows already grouped by expert."""

    if qdata.ndim != 2 or qdata.dtype != torch.float8_e4m3fn:
        raise TypeError("qdata must be a two-dimensional float8_e4m3fn tensor")
    if qdata.shape[0] != linear_scales.shape[0]:
        raise ValueError("qdata and linear scale row counts differ")
    if qdata.shape[1] // MXFP8_SCALE_BLOCK_K != linear_scales.shape[1]:
        raise ValueError("linear scale width is not K / 32")
    scale = pack_varlen_m_scales(linear_scales, cu_seqlens_m)
    return BlockScaledOperand.from_parts(qdata, scale, MXFP8_FORMAT)


def unpack_varlen_m_scales(
    blocked_scales: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    *,
    sf_k: int,
) -> torch.Tensor:
    """Recover the semantic ``(total_M, K/32)`` scales from varlen storage.

    This is a correctness/reference helper.  The timed kernel path consumes
    ``blocked_scales`` directly and never performs this unpack.
    """

    if blocked_scales.ndim != 6 or blocked_scales.shape[0] != 1:
        raise TypeError("varlen-M scales must have shape (1, padded_rm, rk, 32, 4, 4)")
    if cu_seqlens_m.ndim != 1 or cu_seqlens_m.dtype != torch.int32:
        raise TypeError("cu_seqlens_m must be a one-dimensional int32 tensor")
    if blocked_scales.device != cu_seqlens_m.device:
        raise ValueError("blocked scales and cu_seqlens_m must be on the same device")

    total_m = int(cu_seqlens_m[-1].item())
    num_experts = cu_seqlens_m.shape[0] - 1
    padded_m = blocked_scales.shape[1] * MXFP8_SCALE_ATOM_M
    padded_linear = unpack_scale_blocked_to_2d(blocked_scales, padded_m, sf_k)[0]
    if total_m == 0:
        return padded_linear[:0]

    counts = (cu_seqlens_m[1:] - cu_seqlens_m[:-1]).to(torch.int64)
    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, device=blocked_scales.device, dtype=torch.int64),
        counts,
        output_size=total_m,
    )
    starts = cu_seqlens_m[:-1].to(torch.int64)
    padded_starts = (
        starts // MXFP8_SCALE_ATOM_M
        + torch.arange(num_experts, device=blocked_scales.device, dtype=torch.int64)
    ) * MXFP8_SCALE_ATOM_M
    source = torch.arange(total_m, device=blocked_scales.device, dtype=torch.int64)
    source = source + (padded_starts - starts)[expert_ids]
    return (
        padded_linear.contiguous()
        .view(torch.uint8)
        .index_select(0, source)
        .view(torch.float8_e8m0fnu)
    )


def dequantize_varlen_m_operand(
    operand: BlockScaledOperand,
    cu_seqlens_m: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize a padded variable-M operand for numerical validation."""

    if operand.format != MXFP8_E4M3 or operand.ndim != 2:
        raise TypeError("expected a two-dimensional MXFP8 E4M3 operand")
    sf_k = _ceil_div(operand.shape[-1], operand.format.sf_vec_size)
    scales = unpack_varlen_m_scales(operand.scale, cu_seqlens_m, sf_k=sf_k)
    values = dequant_operand(operand.qdata, operand.format)
    expanded = scales.float().repeat_interleave(operand.format.sf_vec_size, dim=-1)
    return (values * expanded[:, : operand.shape[-1]]).to(dtype)


def quantize_varlen_m_operand(
    x: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
) -> BlockScaledOperand:
    """Quantize and pack BF16/FP32 expert-grouped rows."""

    qdata, linear_scales = quantize_mxfp8_rows(x)
    return make_varlen_m_operand(qdata, linear_scales, cu_seqlens_m)


def allocate_mxfp8_weights(
    num_experts: int,
    out_features: int,
    in_features: int,
    *,
    device: torch.device | str,
    seed: int,
) -> BlockScaledOperand:
    """Allocate deterministic synthetic MXFP8 expert weights.

    The values and E8M0 scale bytes are generated without retaining a BF16
    master, keeping full customer-shape setup memory modest.  They still form
    a valid 1x32 MXFP8 operand and exercise the real scale-factor mainloop.
    """

    if out_features % 128 or in_features % 128:
        raise ValueError(
            "SM120 MXFP8 synthetic weights require N and K divisible by 128"
        )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    qdata = torch.empty(
        (num_experts, out_features, in_features),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    # Build one expert at a time so the transient BF16 allocation is bounded.
    for expert in range(num_experts):
        tmp = (
            torch.randn(
                (out_features, in_features),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            * 0.015625
        )
        qdata[expert].copy_(tmp.to(torch.float8_e4m3fn))

    # Alternate exact powers of two across K blocks and experts.  This makes
    # scale indexing observable in correctness tests instead of degenerating
    # to the unit-scale fast path.
    sf_k = in_features // MXFP8_SCALE_BLOCK_K
    exponents = (
        torch.arange(
            num_experts * out_features * sf_k, device=device, dtype=torch.int64
        )
        .reshape(num_experts, out_features, sf_k)
        .remainder(3)
        - 1
    )
    scale_linear = torch.pow(2.0, exponents.to(torch.float32)).to(torch.float8_e8m0fnu)
    scale = pack_scale_2d_to_blocked_contig(scale_linear)
    return BlockScaledOperand.from_parts(qdata, scale, MXFP8_FORMAT)


def _validate_grouped_inputs(
    x: BlockScaledOperand,
    w1: BlockScaledOperand,
    w2: BlockScaledOperand,
    cu_seqlens_m: torch.Tensor,
) -> tuple[int, int, int, int]:
    if x.format != MXFP8_E4M3 or w1.format != MXFP8_E4M3 or w2.format != MXFP8_E4M3:
        raise TypeError(
            "SonicMoE MXFP8 currently requires E4M3 values with E8M0 scales"
        )
    if x.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        raise ValueError("expected x=(M,H), w1=(E,2I,H), w2=(E,H,I)")
    experts, two_i, hidden = w1.shape
    if two_i % 2:
        raise ValueError("w1 output width must be even for SwiGLU")
    intermediate = two_i // 2
    if w2.shape != (experts, hidden, intermediate):
        raise ValueError(
            f"w2 shape {w2.shape} != ({experts}, {hidden}, {intermediate})"
        )
    if x.shape[1] != hidden:
        raise ValueError(f"x hidden {x.shape[1]} != weight hidden {hidden}")
    if cu_seqlens_m.shape != (experts + 1,):
        raise ValueError("cu_seqlens_m length must be num_experts + 1")
    return experts, hidden, intermediate, x.shape[0]


def mxfp8_swiglu_grouped(
    x: BlockScaledOperand,
    w1: BlockScaledOperand,
    cu_seqlens_m: torch.Tensor,
    *,
    config: Mxfp8MoEKernelConfig = _DEFAULT_CONFIG,
) -> BlockScaledOperand:
    """Run MXFP8 FC1 and fuse SwiGLU plus MXFP8 post-activation quantization."""

    experts = w1.shape[0]
    intermediate = w1.shape[1] // 2
    total_m = x.shape[0]
    out_q = torch.empty(
        (total_m, intermediate), dtype=torch.float8_e4m3fn, device=x.device
    )
    padded_rm = _ceil_div(total_m, MXFP8_SCALE_ATOM_M) + experts - 1
    out_sf = torch.empty(
        (
            1,
            padded_rm,
            _ceil_div(intermediate, 4 * MXFP8_SCALE_BLOCK_K),
            32,
            4,
            4,
        ),
        dtype=torch.float8_e8m0fnu,
        device=x.device,
    )
    swiglu_quant_mod.gemm(
        x.qdata,
        w1.qdata,
        None,
        epi_args={"postact": out_q, "postact_sf": out_sf},
        tile_M=config.fc1_tile_m,
        tile_N=config.fc1_tile_n,
        cluster_M=1,
        cluster_N=1,
        pingpong=config.pingpong,
        persistent=True,
        is_dynamic_persistent=config.dynamic_persistent,
        cu_seqlens_m=cu_seqlens_m,
        SFA=x.scale,
        SFB=w1.scale,
        bs_format_a=x.format.name,
        bs_format_b=w1.format.name,
    )
    return BlockScaledOperand.from_parts(out_q, out_sf, MXFP8_FORMAT)


def mxfp8_down_grouped(
    postact: BlockScaledOperand,
    w2: BlockScaledOperand,
    cu_seqlens_m: torch.Tensor,
    *,
    config: Mxfp8MoEKernelConfig = _DEFAULT_CONFIG,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the MXFP8 FC2 grouped GEMM and return BF16 grouped outputs."""

    return mxfp8_grouped_gemm(postact, w2, cu_seqlens_m, config=config, out=out)


def mxfp8_grouped_gemm(
    x: BlockScaledOperand,
    weight: BlockScaledOperand,
    cu_seqlens_m: torch.Tensor,
    *,
    config: Mxfp8MoEKernelConfig = _DEFAULT_CONFIG,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run ``{X_e @ W_e.T}_e`` as one variable-M MXFP8 grouped GEMM.

    ``x`` contains the expert-grouped rows concatenated as ``(sum(M_e), K)``;
    ``weight`` is ``(E, N, K)``; and ``cu_seqlens_m`` is the int32 prefix sum
    of ``M_e``.  The result remains concatenated as ``(sum(M_e), N)``.  Input
    values are E4M3 with semantic 1x32 E8M0 scales and the output is BF16.
    Routing, token permutation, and expert-parallel communication deliberately
    stay outside this compute primitive.
    """

    if x.format != MXFP8_E4M3 or weight.format != MXFP8_E4M3:
        raise TypeError("grouped GEMM requires MXFP8 E4M3 values with E8M0 scales")
    if x.ndim != 2 or weight.ndim != 3:
        raise ValueError("expected x=(sum(M_e), K) and weight=(E, N, K)")
    experts, out_features, in_features = weight.shape
    if x.shape[1] != in_features:
        raise ValueError(f"x K={x.shape[1]} does not match weight K={in_features}")
    if cu_seqlens_m.ndim != 1 or cu_seqlens_m.dtype != torch.int32:
        raise TypeError("cu_seqlens_m must be a one-dimensional int32 tensor")
    if cu_seqlens_m.shape[0] != experts + 1:
        raise ValueError("cu_seqlens_m length must be number of groups + 1")
    if x.device != weight.device or x.device != cu_seqlens_m.device:
        raise ValueError("x, weight, and cu_seqlens_m must be on the same device")

    total_m = x.shape[0]
    if out is None:
        out = torch.empty(
            (total_m, out_features), dtype=torch.bfloat16, device=x.device
        )
    elif (
        out.shape != (total_m, out_features)
        or out.dtype != torch.bfloat16
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must be a contiguous BF16 tensor with shape "
            f"({total_m}, {out_features}) on {x.device}"
        )
    identity_epi.gemm(
        x.qdata,
        weight.qdata,
        out,
        epi_args={},
        tile_M=config.fc2_tile_m,
        tile_N=config.fc2_tile_n,
        cluster_M=1,
        cluster_N=1,
        pingpong=config.pingpong,
        persistent=True,
        is_dynamic_persistent=config.dynamic_persistent,
        cu_seqlens_m=cu_seqlens_m,
        SFA=x.scale,
        SFB=weight.scale,
        bs_format_a=x.format.name,
        bs_format_b=weight.format.name,
    )
    return out


def moe_mxfp8_grouped_forward(
    x: BlockScaledOperand,
    w1: BlockScaledOperand,
    w2: BlockScaledOperand,
    cu_seqlens_m: torch.Tensor,
    *,
    config: Mxfp8MoEKernelConfig = _DEFAULT_CONFIG,
) -> tuple[torch.Tensor, BlockScaledOperand]:
    """Run local expert FC1-SwiGLU-FC2 for rows grouped by expert."""

    _validate_grouped_inputs(x, w1, w2, cu_seqlens_m)
    postact = mxfp8_swiglu_grouped(x, w1, cu_seqlens_m, config=config)
    out = mxfp8_down_grouped(postact, w2, cu_seqlens_m, config=config)
    return out, postact


__all__ = [
    "MXFP8_FORMAT",
    "MXFP8_SCALE_ATOM_M",
    "MXFP8_SCALE_BLOCK_K",
    "Mxfp8MoEKernelConfig",
    "allocate_mxfp8_weights",
    "dequantize_varlen_m_operand",
    "make_varlen_m_operand",
    "moe_mxfp8_grouped_forward",
    "mxfp8_down_grouped",
    "mxfp8_grouped_gemm",
    "mxfp8_swiglu_grouped",
    "pack_varlen_m_scales",
    "quantize_mxfp8_rows",
    "quantize_varlen_m_operand",
    "unpack_varlen_m_scales",
]
