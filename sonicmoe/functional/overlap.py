"""Hide expert-parallel all-to-all behind an independent shared-expert branch.

The routed path is serial on one CUDA stream: every NCCL all-to-all blocks the
compute stream while the SMs idle.  A shared expert reads the same activations
but does not depend on the routed result, so its GEMMs can run on a side stream
inside the collective windows.  ``fire(tag)`` is called by the transport code
right before each large collective; ``SharedExpertOverlap`` registers work that
is released at those points.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

_hook = None
# Forward overlap of the shared expert with dispatch/combine.  With the
# persistent side stream (no per-call allocator pool churn) it is a net win:
# EP8 T=61440 -1.7 ms and T=102400 -2.8 ms forward (job 6182470).
_FWD_OVERLAP = __import__("os").environ.get("SONICMOE_SHARED_FWD_OVERLAP", "1") == "1"


def fire(tag: str) -> None:
    if _hook is not None:
        _hook(tag)


@contextlib.contextmanager
def _installed(hook):
    global _hook
    previous, _hook = _hook, hook
    try:
        yield
    finally:
        _hook = previous


from .fused_swiglu import swiglu_bwd as _act_bwd, swiglu_fwd as _act


class _State:
    def __init__(self, x, s13, s2, side):
        self.x, self.s13, self.s2, self.side = x, s13, s2, side
        self.main = torch.cuda.current_stream(x.device)
        self.gu = self.a = None
        self.fwd_done_a = self.fwd_done_b = False
        self.events = {}
        self.out = None
        self.g = None
        self.stage_done = set()
        self.da = self.ds2 = self.dgu = self.dx = self.ds13 = None

    def _gate(self, tag=None):
        main = torch.cuda.current_stream(self.x.device)
        self.side.wait_stream(main)

    def _cur(self):
        return torch.cuda.current_stream(self.x.device)

    @torch.no_grad()
    def run_a(self, ev=None):
        if self.fwd_done_a:
            return
        self.fwd_done_a = True
        if ev is None:
            ev = torch.cuda.Event()
            ev.record(self._cur())
        self.x.record_stream(self.side)
        with torch.cuda.stream(self.side):
            self.side.wait_event(ev)
            self.gu = F.linear(self.x, self.s13)
            self.a = _act(self.gu)

    @torch.no_grad()
    def run_b(self, ev=None):
        if self.fwd_done_b:
            return
        self.run_a()
        self.fwd_done_b = True
        if ev is None:
            ev = torch.cuda.Event()
            ev.record(self._cur())
        with torch.cuda.stream(self.side):
            self.side.wait_event(ev)
            self.out.copy_(F.linear(self.a, self.s2))

    def hook(self, tag):
        kind, name = tag.split(":")
        if kind == "pre":
            ev = torch.cuda.Event()
            ev.record(self._cur())
            self.events[name] = ev
            return
        ev = self.events.get(name)
        if name == "fwd_dispatch":
            if _FWD_OVERLAP:
                self.run_a(ev)
        elif name == "fwd_combine":
            if _FWD_OVERLAP:
                self.run_b(ev)
        elif self.g is not None:
            self.bwd_stage(1 if name == "bwd_combine" else 2, ev)

    @torch.no_grad()
    def bwd_stage(self, stage, ev):
        if stage in self.stage_done or (stage == 2 and 1 not in self.stage_done):
            return
        self.stage_done.add(stage)
        g, x, s13, s2, side = self.g, self.x, self.s13, self.s2, self.side
        for t in (g, x, s13, s2, self.gu, self.a):
            t.record_stream(side)
        with torch.cuda.stream(side):
            side.wait_event(ev)
            if stage == 1:
                self.da = torch.matmul(g, s2)
                self.ds2 = torch.matmul(g.t(), self.a)
            else:
                self.dgu = _act_bwd(self.gu, self.da)
                self.dx = torch.matmul(self.dgu, s13)
                self.ds13 = torch.matmul(self.dgu.t(), x)

    def capture_grad(self, y):
        def _h(g):
            self.g = g.contiguous()
        y.register_hook(_h)


class _SharedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s13, s2, state):
        ctx.state = state
        ctx.save_for_backward(x, s13, s2)
        out = torch.empty(x.shape[0], s2.shape[0], dtype=x.dtype, device=x.device)
        out.record_stream(state.side)
        state.out = out
        return out

    @staticmethod
    def backward(ctx, g):
        st = ctx.state
        main = torch.cuda.current_stream(g.device)
        if st.g is None:
            st.g = g.contiguous()
        # Fallback for stages whose collective window never fired.
        ev = torch.cuda.Event()
        ev.record(main)
        st.bwd_stage(1, ev)
        st.bwd_stage(2, ev)
        main.wait_stream(st.side)
        dx, ds13, ds2 = st.dx, st.ds13, st.ds2
        for t in (dx, ds13, ds2):
            t.record_stream(main)
        st.gu = st.a = st.da = st.dgu = st.g = None
        return dx, ds13, ds2, None


_SIDE_STREAMS: dict = {}


def _side_stream(device) -> torch.cuda.Stream:
    # One persistent stream per device: a fresh stream per call rotates through
    # PyTorch's stream pool, and the caching allocator keeps a separate memory
    # pool per stream, so every call re-allocates the large shared-expert
    # activations (measured ~+4 ms forward).
    key = (device.type, device.index)
    stream = _SIDE_STREAMS.get(key)
    if stream is None:
        stream = _SIDE_STREAMS[key] = torch.cuda.Stream(device=device)
    return stream


def shared_expert_overlapped(x, s13, s2, routed_fn):
    """Return ``(routed_fn(), shared_out, state)`` with the shared expert hidden behind
    the all-to-all windows of ``routed_fn``.

    ``x``: [T, H] activations, ``s13``: [2I, H] fused gate/up weights,
    ``s2``: [H, I].  Gradients flow to all three.  Create the shared node first
    so that autograd runs it after the routed backward has issued its
    collectives.
    """
    side = _side_stream(x.device)
    st = _State(x, s13, s2, side)
    shared = _SharedFn.apply(x, s13, s2, st)
    global _hook
    _hook = st.hook  # stays installed through backward; replaced by the next call
    routed = routed_fn()
    st.run_b()
    torch.cuda.current_stream(x.device).wait_stream(side)
    return routed, shared, st
