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

import torch
from sonicmoe import KernelBackendMoE, MoE
from sonicmoe.enums import ActivationType


def _percentile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    return ordered[min(round(q * (len(ordered) - 1)), len(ordered) - 1)]


def _measure(
    model: MoE,
    x: torch.Tensor,
    backend: KernelBackendMoE,
    mode: str,
    optimizer_step: bool,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    grad = torch.randn_like(x)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4) if optimizer_step else None

    def run():
        if mode == "forward":
            with torch.inference_mode():
                model(x, kernel_backend_moe=backend, is_inference_mode=True)
        else:
            model.zero_grad(set_to_none=True)
            x.grad = None
            output, aux_loss = model(x, kernel_backend_moe=backend)
            loss = (output * grad).sum() + 0.01 * aux_loss
            loss.backward()
            if optimizer is not None:
                optimizer.step()

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

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
    return {
        "mean_ms": statistics.fmean(samples),
        "p50_ms": _percentile(samples, 0.50),
        "p95_ms": _percentile(samples, 0.95),
        "p99_ms": _percentile(samples, 0.99),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    }


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
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("sonicmoe", "sonicmoe_mxfp8"),
        default=("sonicmoe", "sonicmoe_mxfp8"),
    )
    args = parser.parse_args()
    if torch.cuda.get_device_properties(0).major != 10:
        raise RuntimeError("this benchmark requires SM100")
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
    results = {}
    for name in args.backends:
        model = copy.deepcopy(base)
        backend = KernelBackendMoE(name)
        results[name] = _measure(
            model,
            x,
            backend,
            args.mode,
            args.optimizer_step,
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
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
    }
    if "sonicmoe" in results and "sonicmoe_mxfp8" in results:
        payload["mxfp8_speedup"] = (
            results["sonicmoe"]["p50_ms"] / results["sonicmoe_mxfp8"]["p50_ms"]
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
