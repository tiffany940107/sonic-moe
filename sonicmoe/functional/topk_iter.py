"""Iterative top-K + softmax-over-top-K router kernel for K up to 32.

The Quack kernel covers K <= 16; larger K (e.g. K=20 at E=384) fell back to
``torch.topk`` + softmax + two copies (~1.8 ms at 61k tokens).  One program
handles ROWS tokens: K passes of (max, argmax, mask) over the E logits kept in
registers, then a softmax over the K selected values.  Ties resolve to the
lowest expert index.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _topk_softmax_kernel(
    logits,
    scores,
    indices,
    T,
    E: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    EP: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rmask = rows < T
    cols = tl.arange(0, EP)
    cmask = cols < E
    x = tl.load(
        logits + rows[:, None].to(tl.int64) * E + cols[None, :],
        mask=rmask[:, None] & cmask[None, :],
        other=-float("inf"),
    ).to(tl.float32)
    kcols = tl.arange(0, KP)
    vals = tl.full((ROWS, KP), -float("inf"), dtype=tl.float32)
    idxs = tl.zeros((ROWS, KP), dtype=tl.int32)
    for k in tl.static_range(0, K):
        m = tl.max(x, axis=1)
        is_max = x == m[:, None]
        # lowest index among ties
        first = tl.min(tl.where(is_max, cols[None, :], EP), axis=1)
        sel = kcols[None, :] == k
        vals = tl.where(sel, m[:, None], vals)
        idxs = tl.where(sel, first[:, None].to(tl.int32), idxs)
        x = tl.where(cols[None, :] == first[:, None], -float("inf"), x)
    top = tl.max(vals, axis=1)
    ex = tl.where(kcols[None, :] < K, tl.exp(vals - top[:, None]), 0.0)
    y = ex / tl.sum(ex, axis=1)[:, None]
    kmask = rmask[:, None] & (kcols[None, :] < K)
    out = rows[:, None].to(tl.int64) * K + kcols[None, :]
    tl.store(scores + out, y, mask=kmask)
    tl.store(indices + out, idxs, mask=kmask)


def topk_softmax_iter(router_logits, scores, indices, K: int) -> None:
    T, E = router_logits.shape
    EP = triton.next_power_of_2(E)
    KP = triton.next_power_of_2(K)
    ROWS = 4 if EP <= 512 else 2
    _topk_softmax_kernel[(triton.cdiv(T, ROWS),)](
        router_logits.contiguous(),
        scores,
        indices,
        T,
        E=E,
        K=K,
        KP=KP,
        EP=EP,
        ROWS=ROWS,
        num_warps=4,
    )
