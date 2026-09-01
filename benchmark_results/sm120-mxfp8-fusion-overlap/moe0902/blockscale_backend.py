from __future__ import annotations

import torch


MXFP8_K = 32
SCALES_PER_WORD = 4


def quantize_linear_mxfp8_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference OCP MXFP8 row quantizer returning semantic E8M0 scales.

    This deliberately has no Sonic/QuACK dependency, so the block-scale
    extension can run in the exact PyTorch ABI against which it was built.
    """
    if x.ndim != 2 or x.shape[1] % MXFP8_K:
        raise ValueError("x must be [M,K] with K divisible by 32")
    rows, hidden = x.shape
    blocks = x.reshape(rows, hidden // MXFP8_K, MXFP8_K).float()
    amax = blocks.abs().amax(dim=2).clamp_min(1e-10)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    values = (blocks / scale.unsqueeze(2)).clamp(-448.0, 448.0)
    values = values.to(torch.float8_e4m3fn).reshape(rows, hidden)
    exponent = ((scale.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
    return values, exponent.view(torch.float8_e8m0fnu)


def padded_scale_rows(total_rows: int, experts: int) -> int:
    """Rows in the block-scale expert-padded scale layout."""
    return ((total_rows + experts * (SCALES_PER_WORD - 1)) // SCALES_PER_WORD) * SCALES_PER_WORD


def expert_padded_offset(linear_offset: int, expert: int) -> int:
    return ((linear_offset + expert * (SCALES_PER_WORD - 1)) // SCALES_PER_WORD) * SCALES_PER_WORD


def pack_linear_e8m0_for_moe(
    linear_scales: torch.Tensor,
    token_offset: torch.Tensor,
) -> torch.Tensor:
    """Pack semantic ``[M,K/32]`` E8M0 bytes for the block-scale MoE ABI.

    The returned INT32 tensor has logical shape ``[M_padded,K/128]`` and
    column-major stride ``[1,M_padded]``.  Four consecutive K32 E8M0 bytes
    occupy one INT32 word.  Each expert begins at a four-row aligned scale
    offset, matching ``fp8_quant_and_transform_for_moe`` exactly.
    """
    if linear_scales.ndim != 2:
        raise ValueError("linear_scales must have [M,K/32] shape")
    if linear_scales.dtype == torch.float8_e8m0fnu:
        scale_bytes = linear_scales.view(torch.uint8)
    elif linear_scales.dtype == torch.uint8:
        scale_bytes = linear_scales
    else:
        raise TypeError("linear_scales must be float8_e8m0fnu or uint8")
    if token_offset.ndim != 1 or token_offset.dtype != torch.int32:
        raise TypeError("token_offset must be a 1-D INT32 tensor")
    if not token_offset.is_cuda or token_offset.device != scale_bytes.device:
        raise ValueError("scales and token_offset must share one CUDA device")
    experts = token_offset.numel() - 1
    total_rows, scale_k = scale_bytes.shape
    if scale_k % SCALES_PER_WORD:
        raise ValueError("K/32 scale count must be divisible by four")
    if int(token_offset[0]) != 0 or int(token_offset[-1]) != total_rows:
        raise ValueError("token_offset does not cover all rows")

    words = scale_bytes.reshape(total_rows, scale_k // 4, 4).to(torch.int32)
    packed = words[..., 0] | (words[..., 1] << 8) | (words[..., 2] << 16) | (words[..., 3] << 24)
    storage_rows = padded_scale_rows(total_rows, experts)
    storage = torch.zeros(
        (scale_k // 4, storage_rows), dtype=torch.int32, device=scale_bytes.device
    )
    # E is only 128 in the target case.  Keeping the alignment loop explicit
    # makes the ABI contract auditable; all slice copies execute asynchronously.
    offsets = token_offset.cpu().tolist()
    for expert, (start, end) in enumerate(zip(offsets, offsets[1:])):
        if start == end:
            continue
        destination = expert_padded_offset(start, expert)
        storage[:, destination:destination + (end - start)] = packed[start:end].transpose(0, 1)
    result = storage.transpose(0, 1)
    if result.stride() != (1, storage_rows):
        raise AssertionError(f"unexpected scale stride {result.stride()}")
    return result


def unpack_active_e8m0_from_moe(
    packed_scales: torch.Tensor,
    token_offset: torch.Tensor,
    scale_k: int,
) -> torch.Tensor:
    """Debug-only inverse over active rows, used by the C0 byte gate."""
    if packed_scales.dtype != torch.int32 or packed_scales.ndim != 2:
        raise TypeError("packed_scales must be a 2-D INT32 tensor")
    experts = token_offset.numel() - 1
    total_rows = int(token_offset[-1])
    storage = packed_scales.transpose(0, 1).contiguous()
    output = torch.empty((total_rows, scale_k), dtype=torch.uint8, device=packed_scales.device)
    offsets = token_offset.cpu().tolist()
    for expert, (start, end) in enumerate(zip(offsets, offsets[1:])):
        if start == end:
            continue
        source = expert_padded_offset(start, expert)
        words = storage[:, source:source + (end - start)].transpose(0, 1)
        bytes4 = torch.stack(
            [
                words & 0xFF,
                (words >> 8) & 0xFF,
                (words >> 16) & 0xFF,
                (words >> 24) & 0xFF,
            ],
            dim=-1,
        ).to(torch.uint8)
        output[start:end] = bytes4.reshape(end - start, -1)[:, :scale_k]
    if packed_scales.shape[0] != padded_scale_rows(total_rows, experts):
        raise ValueError("packed scale row count is inconsistent")
    return output

