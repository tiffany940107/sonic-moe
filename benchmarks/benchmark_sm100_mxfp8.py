# Copyright (c) 2026, SonicMoE contributors.
"""Reproducible SM100 BF16/MXFP8 MoE latency and memory benchmark.

Examples:
    python benchmarks/benchmark_sm100_mxfp8.py --mode forward
    python benchmarks/benchmark_sm100_mxfp8.py --mode training --tokens 1024
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
from collections.abc import Callable

import torch

import sonicmoe.functional.mxfp8 as mxfp8_impl
from sonicmoe import KernelBackendMoE, MoE, Mxfp8SGD
from sonicmoe.enums import ActivationType


def _percentile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    return ordered[min(round(q * (len(ordered) - 1)), len(ordered) - 1)]


def _run_step(
    model: MoE,
    x: torch.Tensor,
    grad: torch.Tensor,
    backend: KernelBackendMoE,
    mode: str,
    grad_accum_steps: int,
    optimizer: torch.optim.Optimizer | None,
) -> None:
    if mode == "forward":
        with torch.inference_mode():
            model(x, kernel_backend_moe=backend, is_inference_mode=True)
    else:
        model.zero_grad(set_to_none=True)
        x.grad = None
        for _ in range(grad_accum_steps):
            output, aux_loss = model(x, kernel_backend_moe=backend)
            loss = (output * grad).sum() + 0.01 * aux_loss
            loss.backward()
        if optimizer is not None:
            optimizer.step()


def _summarize(
    samples: list[float], peak_allocated_mib: float | None
) -> dict[str, float | None]:
    return {
        "mean_ms": statistics.fmean(samples),
        "p50_ms": _percentile(samples, 0.50),
        "p95_ms": _percentile(samples, 0.95),
        "p99_ms": _percentile(samples, 0.99),
        "peak_allocated_mib": peak_allocated_mib,
    }


def _capture_step(
    model: MoE,
    x: torch.Tensor,
    grad: torch.Tensor,
    backend: KernelBackendMoE,
    mode: str,
    grad_accum_steps: int,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[Callable[[], None], torch.cuda.CUDAGraph]:
    """Capture one fixed-shape step after eager warmup has allocated grads."""
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        _run_step(model, x, grad, backend, mode, grad_accum_steps, optimizer)
    capture_stream.synchronize()
    # The benchmark intentionally warms eager autograd on the default stream
    # before capturing on a side stream.  Tell autograd to rebind those stale
    # AccumulateGrad stream references for the duration of this capture.
    torch.autograd.graph.set_override_stale_capture_stream(True)
    try:
        with torch.cuda.graph(graph, stream=capture_stream):
            if mode == "forward":
                with torch.inference_mode():
                    model(x, kernel_backend_moe=backend, is_inference_mode=True)
            else:
                model.zero_grad(set_to_none=False)
                if x.grad is not None:
                    x.grad.zero_()
                for _ in range(grad_accum_steps):
                    output, aux_loss = model(x, kernel_backend_moe=backend)
                    loss = (output * grad).sum() + 0.01 * aux_loss
                    loss.backward()
                if optimizer is not None:
                    optimizer.step()
    finally:
        torch.autograd.graph.set_override_stale_capture_stream(False)
    torch.cuda.synchronize()
    return graph.replay, graph


def _measure(
    model: MoE,
    x: torch.Tensor,
    backend: KernelBackendMoE,
    mode: str,
    optimizer_step: bool,
    fused_mxfp8_sgd: bool,
    cuda_graph: bool,
    grad_accum_steps: int,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    grad = torch.randn_like(x)
    optimizer = (
        (
            Mxfp8SGD(model, lr=1e-4)
            if fused_mxfp8_sgd and backend == KernelBackendMoE.sonicmoe_mxfp8
            else torch.optim.SGD(model.parameters(), lr=1e-4)
        )
        if optimizer_step
        else None
    )

    def eager_run():
        _run_step(model, x, grad, backend, mode, grad_accum_steps, optimizer)

    for _ in range(warmup):
        eager_run()
    torch.cuda.synchronize()

    _graph = None
    if cuda_graph:
        run, _graph = _capture_step(
            model, x, grad, backend, mode, grad_accum_steps, optimizer
        )
    else:
        run = eager_run

    torch.cuda.reset_peak_memory_stats()
    profile_range = os.environ.get("SONICMOE_PROFILE_RANGE", "0") == "1"
    if profile_range:
        torch.cuda.cudart().cudaProfilerStart()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    if profile_range:
        torch.cuda.cudart().cudaProfilerStop()
    return _summarize(samples, torch.cuda.max_memory_allocated() / 2**20)


def _measure_interleaved(
    models: dict[str, MoE],
    x: torch.Tensor,
    mode: str,
    optimizer_step: bool,
    fused_mxfp8_sgd: bool,
    cuda_graph: bool,
    grad_accum_steps: int,
    warmup: int,
    repeats: int,
) -> dict[str, dict[str, float | None]]:
    """Measure variants step-by-step with a rotating order to limit clock drift."""
    names = list(models)
    backends = {
        name: KernelBackendMoE.sonicmoe_mxfp8
        if name == "sonicmoe_mxfp8_fused_sgd"
        else KernelBackendMoE(name)
        for name in names
    }
    optimizers = {}
    for name, model in models.items():
        if not optimizer_step:
            optimizers[name] = None
        elif name == "sonicmoe_mxfp8_fused_sgd" or (
            fused_mxfp8_sgd and name == "sonicmoe_mxfp8"
        ):
            optimizers[name] = Mxfp8SGD(model, lr=1e-4)
        else:
            optimizers[name] = torch.optim.SGD(model.parameters(), lr=1e-4)
    grad = torch.randn_like(x)
    # Each captured graph owns an AccumulateGrad buffer for its input. Sharing
    # one leaf lets the second capture release/reuse the pointer recorded by
    # the first graph, which becomes an illegal address at replay time.
    inputs = (
        {name: x.detach().clone().requires_grad_(x.requires_grad) for name in names}
        if cuda_graph
        else {name: x for name in names}
    )

    def ordered(round_idx: int) -> list[str]:
        shift = round_idx % len(names)
        return names[shift:] + names[:shift]

    for round_idx in range(warmup):
        for name in ordered(round_idx):
            _run_step(
                models[name],
                inputs[name],
                grad,
                backends[name],
                mode,
                grad_accum_steps,
                optimizers[name],
            )
    torch.cuda.synchronize()

    graphs = {}
    if cuda_graph:
        runners = {}
        for name in names:
            runners[name], graphs[name] = _capture_step(
                models[name],
                inputs[name],
                grad,
                backends[name],
                mode,
                grad_accum_steps,
                optimizers[name],
            )
    else:
        runners = {
            name: (
                lambda n=name: _run_step(
                    models[n],
                    inputs[n],
                    grad,
                    backends[n],
                    mode,
                    grad_accum_steps,
                    optimizers[n],
                )
            )
            for name in names
        }

    profile_range = os.environ.get("SONICMOE_PROFILE_RANGE", "0") == "1"
    if profile_range:
        torch.cuda.cudart().cudaProfilerStart()
    samples = {name: [] for name in names}
    for round_idx in range(repeats):
        for name in ordered(round_idx):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            runners[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end))
    if profile_range:
        torch.cuda.cudart().cudaProfilerStop()
    # Both models are resident during interleaved timing, so a per-backend peak
    # would be misleading. Use the sequential protocol when measuring memory.
    return {name: _summarize(values, None) for name, values in samples.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--mode", choices=("forward", "training"), default="training")
    parser.add_argument(
        "--optimizer-step",
        action="store_true",
        help="include SGD and force versioned MXFP8 weight-cache refresh",
    )
    parser.add_argument(
        "--fused-mxfp8-sgd",
        action="store_true",
        help="fuse plain SGD expert updates with required MXFP8 weight refreshes",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="backward microbatches accumulated before the optional optimizer step",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="capture and replay each fixed-shape step after eager warmup",
    )
    parser.add_argument(
        "--interleave",
        action="store_true",
        help="alternate backends every step with a cyclic order",
    )
    parser.add_argument(
        "--mxfp8-policy",
        "--mxfp8-wgrad",
        dest="mxfp8_policy",
        choices=(
            "auto",
            "mxfp8",
            "bf16",
            "fc1_bf16",
            "fc2_bf16",
            "dgrad_bf16",
            "fc1_dgrad_bf16",
            "fc2_dgrad_bf16",
            "forward_only",
        ),
        help=(
            "override the MXFP8 training role-precision policy for this run; "
            "the default auto policy uses MXFP8 forward with BF16 dgrad/wgrad, "
            "while mxfp8 uses MXFP8 for all expert GEMMs"
        ),
    )
    parser.add_argument("--save-z-fp8", choices=("auto", "0", "1"))
    parser.add_argument("--fp8-c-dgated", choices=("auto", "0", "1"))
    parser.add_argument("--fp8-c-fuse-dquant", choices=("0", "1"))
    parser.add_argument("--tma-wgrad", choices=("auto", "0", "1"))
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=(
            "sonicmoe",
            "sonicmoe_mxfp8",
            "sonicmoe_mxfp8_fused_sgd",
        ),
        default=("sonicmoe", "sonicmoe_mxfp8"),
    )
    args = parser.parse_args()
    if torch.cuda.get_device_properties(0).major != 10:
        raise RuntimeError("this benchmark requires SM100")
    if args.grad_accum_steps < 1:
        parser.error("--grad-accum-steps must be positive")
    if args.mode == "forward" and args.grad_accum_steps != 1:
        parser.error("--grad-accum-steps only applies to training mode")
    if (
        args.fused_mxfp8_sgd or "sonicmoe_mxfp8_fused_sgd" in args.backends
    ) and not args.optimizer_step:
        parser.error("fused MXFP8 SGD requires --optimizer-step")
    if (
        args.fused_mxfp8_sgd or "sonicmoe_mxfp8_fused_sgd" in args.backends
    ) and args.mxfp8_policy not in (None, "auto", "forward_only", "mxfp8"):
        parser.error("fused MXFP8 SGD requires the auto, forward_only, or mxfp8 policy")
    if args.mxfp8_policy is not None:
        os.environ["SONICMOE_MXFP8_POLICY"] = args.mxfp8_policy
    if args.save_z_fp8 is not None:
        mxfp8_impl._SAVE_Z_FP8 = args.save_z_fp8
    if args.fp8_c_dgated is not None:
        mxfp8_impl._FP8_C_DGATED = args.fp8_c_dgated
    if args.fp8_c_fuse_dquant is not None:
        mxfp8_impl._FP8_C_FUSE_DQUANT = args.fp8_c_fuse_dquant == "1"
    if args.tma_wgrad is not None:
        os.environ["SONICMOE_MXFP8_TMA_WGRAD"] = args.tma_wgrad
    torch.manual_seed(123)
    base = (
        MoE(
            num_experts=args.experts,
            num_experts_per_tok=args.top_k,
            hidden_size=args.hidden,
            intermediate_size=args.intermediate,
            activation_function=ActivationType.SWIGLU,
            add_bias=False,
            std=0.02,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    base.train(args.mode == "training")
    x = torch.randn(
        args.tokens,
        args.hidden,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=args.mode == "training",
    )
    if args.interleave and len(args.backends) > 1:
        models = {name: copy.deepcopy(base) for name in args.backends}
        results = _measure_interleaved(
            models,
            x,
            args.mode,
            args.optimizer_step,
            args.fused_mxfp8_sgd,
            args.cuda_graph,
            args.grad_accum_steps,
            args.warmup,
            args.repeats,
        )
        del models
    else:
        results = {}
        for name in args.backends:
            model = copy.deepcopy(base)
            fused_variant = name == "sonicmoe_mxfp8_fused_sgd"
            backend = (
                KernelBackendMoE.sonicmoe_mxfp8
                if fused_variant
                else KernelBackendMoE(name)
            )
            results[name] = _measure(
                model,
                x,
                backend,
                args.mode,
                args.optimizer_step,
                args.fused_mxfp8_sgd or fused_variant,
                args.cuda_graph,
                args.grad_accum_steps,
                args.warmup,
                args.repeats,
            )
            del model
            torch.cuda.empty_cache()
    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "shape": {
            "tokens": args.tokens,
            "experts": args.experts,
            "top_k": args.top_k,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
        },
        "mode": args.mode,
        "optimizer_step": args.optimizer_step,
        "fused_mxfp8_sgd": args.fused_mxfp8_sgd,
        "cuda_graph": args.cuda_graph,
        "timing_protocol": "interleaved" if args.interleave else "sequential",
        "mxfp8_policy": args.mxfp8_policy,
        "mxfp8_saved_activation": {
            "save_z_fp8": mxfp8_impl._SAVE_Z_FP8,
            "fp8_c_dgated": mxfp8_impl._FP8_C_DGATED,
            "fp8_c_fuse_dquant": mxfp8_impl._FP8_C_FUSE_DQUANT,
        },
        "mxfp8_tma_wgrad": os.environ.get("SONICMOE_MXFP8_TMA_WGRAD", "auto"),
        "grad_accum_steps": args.grad_accum_steps,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
    }
    if "sonicmoe" in results and "sonicmoe_mxfp8" in results:
        payload["mxfp8_speedup"] = (
            results["sonicmoe"]["p50_ms"] / results["sonicmoe_mxfp8"]["p50_ms"]
        )
    if "sonicmoe" in results and "sonicmoe_mxfp8_fused_sgd" in results:
        payload["mxfp8_fused_sgd_speedup"] = (
            results["sonicmoe"]["p50_ms"]
            / results["sonicmoe_mxfp8_fused_sgd"]["p50_ms"]
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
