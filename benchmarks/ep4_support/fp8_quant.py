from __future__ import annotations

from typing import Tuple

import torch


def compute_padded_offset(offset: int, expert_index: int) -> int:
    """FlashInfer SM120 zero-padding scale offset (4-row expert alignment)."""
    return ((offset + expert_index * 3) // 4) * 4


def scale_destination_indices(m_indptr: torch.Tensor) -> torch.Tensor:
    """Map each packed activation row to its column in MN-major A scales."""
    if m_indptr.dtype != torch.int32 or m_indptr.ndim != 1:
        raise ValueError("m_indptr must be a one-dimensional int32 tensor")
    counts = m_indptr[1:] - m_indptr[:-1]
    experts = torch.repeat_interleave(
        torch.arange(counts.numel(), device=m_indptr.device, dtype=torch.int64),
        counts.to(torch.int64),
    )
    starts = ((m_indptr[:-1].to(torch.int64) + 3 * torch.arange(
        counts.numel(), device=m_indptr.device, dtype=torch.int64
    )) // 4) * 4
    shifts = starts - m_indptr[:-1].to(torch.int64)
    return torch.arange(int(m_indptr[-1].item()), device=m_indptr.device) + shifts[experts]


def pack_mn_major_scales(
    row_major_scales: torch.Tensor,
    m_indptr: torch.Tensor,
    destination_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack `[M,Kb]` per-token scales as FlashInfer `[Kb,M_padded]`."""
    if row_major_scales.dtype != torch.float32 or row_major_scales.ndim != 2:
        raise ValueError("row_major_scales must be a two-dimensional float32 tensor")
    experts = m_indptr.numel() - 1
    total_rows = int(m_indptr[-1].item())
    if row_major_scales.shape[0] != total_rows:
        raise ValueError("scale row count does not match m_indptr")
    padded_rows = compute_padded_offset(total_rows, experts)
    out = torch.zeros(
        (row_major_scales.shape[1], padded_rows),
        dtype=torch.float32,
        device=row_major_scales.device,
    )
    if total_rows:
        destination_indices = (
            scale_destination_indices(m_indptr)
            if destination_indices is None
            else destination_indices
        )
        out[:, destination_indices] = row_major_scales.t()
    return out


def quantize_and_pack_activation(
    x: torch.Tensor,
    m_indptr: torch.Tensor,
    destination_indices: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from flashinfer.testing.utils import per_token_cast_to_fp8

    x_fp8, row_scales = per_token_cast_to_fp8(x)
    return x_fp8, pack_mn_major_scales(row_scales, m_indptr, destination_indices)


def allocate_synthetic_fp8_weights(
    experts: int,
    n: int,
    k: int,
    *,
    device: torch.device | str = "cuda",
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocate deterministic finite FP8 weights without retaining BF16 masters."""
    if n % 128 or k % 128:
        raise ValueError("SM120 benchmark shapes must be divisible by 128")
    torch.manual_seed(seed)
    weight = torch.empty((experts, n, k), dtype=torch.float8_e4m3fn, device=device)
    # Chunked generation keeps the BF16 temporary below one expert's weight size.
    for expert in range(experts):
        tmp = torch.randn((n, k), dtype=torch.bfloat16, device=device) * 0.015625
        weight[expert].copy_(tmp.to(torch.float8_e4m3fn))
    scales = torch.ones(
        (experts, k // 128, n // 128), dtype=torch.float32, device=device
    )
    return weight, scales
