"""Fused SwiGLU (silu(gate) * value) forward/backward on a packed [T, 2I] tensor.

The eager formulation upcasts both halves to FP32 and launches five or six
elementwise kernels over [T, I] FP32 temporaries; this does one pass in each
direction and writes BF16 directly.  Math is FP32 inside the kernel, rounded
once on store, matching the eager reference up to FMA contraction.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_fwd_kernel(gu, out, inter: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = cols < inter
    g = tl.load(gu + row * 2 * inter + cols, mask=m, other=0.0).to(tl.float32)
    v = tl.load(gu + row * 2 * inter + inter + cols, mask=m, other=0.0).to(tl.float32)
    y = g * tl.sigmoid(g) * v
    tl.store(out + row * inter + cols, y.to(out.dtype.element_ty), mask=m)


@triton.jit
def _swiglu_bwd_kernel(gu, da, dgu, inter: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = cols < inter
    g = tl.load(gu + row * 2 * inter + cols, mask=m, other=0.0).to(tl.float32)
    v = tl.load(gu + row * 2 * inter + inter + cols, mask=m, other=0.0).to(tl.float32)
    d = tl.load(da + row * inter + cols, mask=m, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g)
    dg = d * v * sig * (1.0 + g * (1.0 - sig))
    dv = d * g * sig
    tl.store(dgu + row * 2 * inter + cols, dg.to(dgu.dtype.element_ty), mask=m)
    tl.store(dgu + row * 2 * inter + inter + cols, dv.to(dgu.dtype.element_ty), mask=m)


def swiglu_fwd(gu: torch.Tensor) -> torch.Tensor:
    gu = gu.contiguous()
    rows, two_i = gu.shape
    inter = two_i // 2
    out = torch.empty((rows, inter), dtype=gu.dtype, device=gu.device)
    block = 1024
    _swiglu_fwd_kernel[(rows, triton.cdiv(inter, block))](gu, out, inter=inter, BLOCK=block)
    return out


def swiglu_bwd(gu: torch.Tensor, da: torch.Tensor) -> torch.Tensor:
    gu = gu.contiguous()
    da = da.contiguous()
    rows, two_i = gu.shape
    inter = two_i // 2
    dgu = torch.empty_like(gu)
    block = 1024
    _swiglu_bwd_kernel[(rows, triton.cdiv(inter, block))](gu, da, dgu, inter=inter, BLOCK=block)
    return dgu


class FusedSwiglu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gu):
        ctx.save_for_backward(gu)
        return swiglu_fwd(gu)

    @staticmethod
    def backward(ctx, da):
        (gu,) = ctx.saved_tensors
        return swiglu_bwd(gu, da)


def fused_swiglu(gu: torch.Tensor) -> torch.Tensor:
    return FusedSwiglu.apply(gu)
