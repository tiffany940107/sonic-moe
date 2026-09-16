# ********************************************************************************
# Copyright (c) 2026, SonicMoE contributors
# ********************************************************************************
"""Packed MXFP8 records for low-launch-count expert-parallel transport."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from quack.blockscaled import MXFP8_E4M3


@dataclass(frozen=True)
class Mxfp8TransportLayout:
    """Byte layout of one deduplicated source-to-destination record."""

    hidden: int
    top_k: int

    def __post_init__(self) -> None:
        if self.hidden <= 0 or self.hidden % 128:
            raise ValueError(
                "transport hidden dimension must be positive and aligned to 128"
            )
        if self.top_k <= 0:
            raise ValueError("transport top_k must be positive")

    @property
    def scale_columns(self) -> int:
        return self.hidden // 32

    @property
    def scale_offset(self) -> int:
        return self.hidden

    @property
    def expert_offset(self) -> int:
        return self.scale_offset + self.scale_columns

    @property
    def weight_offset(self) -> int:
        return self.expert_offset + self.top_k * 4

    @property
    def record_bytes(self) -> int:
        return self.weight_offset + self.top_k * 4


@triton.jit
def _pack_mxfp8_transport_kernel(
    qdata,
    scale,
    send_indices,
    expert_ids,
    weights,
    payload,
    payload_expert,
    payload_weight,
    q_stride_m,
    scale_stride_m,
    expert_stride_m,
    weight_stride_m,
    payload_expert_stride_m,
    payload_weight_stride_m,
    records: tl.constexpr,
    hidden: tl.constexpr,
    scale_columns: tl.constexpr,
    top_k: tl.constexpr,
    record_bytes: tl.constexpr,
    META_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    record = tl.program_id(0)
    tile = tl.program_id(1)
    columns = tile * BLOCK + tl.arange(0, BLOCK)
    valid_record = record < records
    source = tl.load(send_indices + record, mask=valid_record, other=0).to(tl.int64)

    q_mask = valid_record & (columns < hidden)
    q_value = tl.load(
        qdata + source * q_stride_m + columns,
        mask=q_mask,
        other=0,
    )
    tl.store(
        payload + record * record_bytes + columns,
        q_value,
        mask=q_mask,
    )

    scale_mask = valid_record & (columns < scale_columns)
    scale_value = tl.load(
        scale + source * scale_stride_m + columns,
        mask=scale_mask,
        other=0,
    )
    tl.store(
        payload + record * record_bytes + hidden + columns,
        scale_value,
        mask=scale_mask,
    )

    slots = tl.arange(0, META_BLOCK)
    metadata_mask = valid_record & (tile == 0) & (slots < top_k)
    expert = tl.load(
        expert_ids + record * expert_stride_m + slots,
        mask=metadata_mask,
        other=-1,
    )
    score = tl.load(
        weights + record * weight_stride_m + slots,
        mask=metadata_mask,
        other=0.0,
    )
    tl.store(
        payload_expert + record * payload_expert_stride_m + slots,
        expert,
        mask=metadata_mask,
    )
    tl.store(
        payload_weight + record * payload_weight_stride_m + slots,
        score,
        mask=metadata_mask,
    )


def unpack_mxfp8_transport(
    payload: torch.Tensor,
    layout: Mxfp8TransportLayout,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return zero-copy, row-strided typed fields from transport bytes."""
    if (
        payload.ndim != 2
        or payload.dtype != torch.uint8
        or not payload.is_cuda
        or not payload.is_contiguous()
    ):
        raise ValueError(
            "payload must be a contiguous two-dimensional CUDA byte tensor"
        )
    if payload.shape[1] != layout.record_bytes:
        raise ValueError("payload record width does not match the transport layout")
    qdata = payload[:, : layout.hidden].view(MXFP8_E4M3.qdata_dtype)
    scale = payload[:, layout.scale_offset : layout.expert_offset].view(
        MXFP8_E4M3.scale_dtype
    )
    expert_ids = payload[:, layout.expert_offset : layout.weight_offset].view(
        torch.int32
    )
    weights = payload[:, layout.weight_offset :].view(torch.float32)
    return qdata, scale, expert_ids, weights


def pack_mxfp8_transport(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    send_indices: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Mxfp8TransportLayout]:
    """Gather quantized rows and metadata into one NCCL-ready byte tensor."""
    if qdata.ndim != 2 or qdata.dtype != MXFP8_E4M3.qdata_dtype:
        raise TypeError("qdata must be a two-dimensional E4M3 tensor")
    records, top_k = expert_ids.shape
    layout = Mxfp8TransportLayout(qdata.shape[1], top_k)
    if scale.shape != (qdata.shape[0], layout.scale_columns):
        raise ValueError("scale must have shape (source_tokens, hidden / 32)")
    if scale.dtype not in (MXFP8_E4M3.scale_dtype, torch.uint8):
        raise TypeError("scale must use E8M0 or its byte view")
    if send_indices.shape != (records,) or send_indices.dtype != torch.int64:
        raise TypeError("send_indices must contain one int64 source row per record")
    if expert_ids.dtype != torch.int32:
        raise TypeError("expert_ids must be int32")
    if weights.shape != expert_ids.shape or weights.dtype != torch.float32:
        raise TypeError("weights must be float32 with the same shape as expert_ids")
    if not all(
        tensor.is_cuda and tensor.device == qdata.device
        for tensor in (scale, send_indices, expert_ids, weights)
    ):
        raise ValueError("all transport inputs must share one CUDA device")
    payload = (
        torch.empty(
            (records, layout.record_bytes),
            dtype=torch.uint8,
            device=qdata.device,
        )
        if out is None
        else out
    )
    if (
        payload.shape != (records, layout.record_bytes)
        or payload.dtype != torch.uint8
        or payload.device != qdata.device
        or not payload.is_contiguous()
    ):
        raise ValueError("out must be a matching contiguous CUDA byte tensor")
    if records:
        _, _, payload_expert, payload_weight = unpack_mxfp8_transport(payload, layout)
        block = 256
        metadata_block = triton.next_power_of_2(top_k)
        _pack_mxfp8_transport_kernel[(records, triton.cdiv(layout.hidden, block))](
            qdata.view(torch.uint8),
            scale.view(torch.uint8),
            send_indices,
            expert_ids,
            weights,
            payload,
            payload_expert,
            payload_weight,
            qdata.stride(0),
            scale.stride(0),
            expert_ids.stride(0),
            weights.stride(0),
            payload_expert.stride(0),
            payload_weight.stride(0),
            records=records,
            hidden=layout.hidden,
            scale_columns=layout.scale_columns,
            top_k=top_k,
            record_bytes=layout.record_bytes,
            META_BLOCK=metadata_block,
            BLOCK=block,
            num_warps=4,
        )
    return payload, layout


__all__ = [
    "Mxfp8TransportLayout",
    "pack_mxfp8_transport",
    "unpack_mxfp8_transport",
]
