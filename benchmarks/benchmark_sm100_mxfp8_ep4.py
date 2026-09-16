# Copyright (c) 2026, SonicMoE contributors.
"""Reproducible four-GPU latency and memory benchmark for SM100 MXFP8 EP."""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist
from sonicmoe import ExpertParallelMoE, KernelBackendMoE, Mxfp8SGD
from sonicmoe.enums import ActivationType


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _rank_max(value: float, device: torch.device) -> float:
    result = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return float(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--mode", choices=("forward", "training"), default="forward")
    parser.add_argument("--backend", choices=("mxfp8", "bf16"), default="mxfp8")
    parser.add_argument(
        "--scenario",
        choices=("balanced", "rank_skew", "segment_skew", "single_hot"),
        default="balanced",
    )
    parser.add_argument("--optimizer-step", action="store_true")
    parser.add_argument(
        "--optimizer", choices=("torch_sgd", "mxfp8_sgd"), default="torch_sgd"
    )
    parser.add_argument("--skip-router-grad-sync", action="store_true")
    parser.add_argument("--disable-fused-transport", action="store_true")
    parser.add_argument("--disable-route-pack", action="store_true")
    parser.add_argument("--force-route-pack", action="store_true")
    parser.add_argument("--disable-fused-reduce", action="store_true")
    parser.add_argument("--force-fused-reduce", action="store_true")
    parser.add_argument("--materialize-route-qdata", action="store_true")
    parser.add_argument("--force-zero-material-qdata", action="store_true")
    parser.add_argument("--comm-overlap", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.disable_route_pack and args.force_route_pack:
        parser.error("--disable-route-pack and --force-route-pack are exclusive")
    if args.disable_fused_reduce and args.force_fused_reduce:
        parser.error("--disable-fused-reduce and --force-fused-reduce are exclusive")
    if args.materialize_route_qdata and args.force_zero_material_qdata:
        parser.error(
            "--materialize-route-qdata and --force-zero-material-qdata are exclusive"
        )
    if args.optimizer_step and args.mode != "training":
        parser.error("--optimizer-step requires --mode training")
    if args.optimizer == "mxfp8_sgd" and args.backend != "mxfp8":
        parser.error("--optimizer mxfp8_sgd requires --backend mxfp8")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4:
        raise RuntimeError(f"EP4 benchmark requires four ranks, got {world}")
    device = torch.device("cuda", local_rank)
    if torch.cuda.get_device_properties(device).major != 10:
        raise RuntimeError("EP4 benchmark requires SM100")
    local_experts = args.experts // world
    if args.experts % world:
        parser.error("--experts must be divisible by four")
    if args.scenario != "balanced" and args.top_k > local_experts:
        parser.error("skew scenarios require --top-k <= experts per rank")

    torch.manual_seed(123)
    model = ExpertParallelMoE(
        num_experts=args.experts,
        num_experts_per_tok=args.top_k,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        activation_function=ActivationType.SWIGLU,
        add_bias=False,
        std=0.02,
        use_fused_transport=not args.disable_fused_transport,
        use_fused_route_pack=(
            False
            if args.disable_route_pack
            else True
            if args.force_route_pack
            else None
        ),
        use_fused_reduce=(
            False
            if args.disable_fused_reduce
            else True
            if args.force_fused_reduce
            else None
        ),
        use_zero_material_qdata=(
            False
            if args.materialize_route_qdata
            else True
            if args.force_zero_material_qdata
            else None
        ),
        expert_backend=(
            KernelBackendMoE.sonicmoe_mxfp8
            if args.backend == "mxfp8"
            else KernelBackendMoE.sonicmoe
        ),
        use_comm_overlap=args.comm_overlap,
    ).to(device=device, dtype=torch.bfloat16)
    model.train(args.mode == "training")
    if args.scenario != "balanced":
        with torch.no_grad():
            model.router.weight.zero_()
            if args.scenario in ("single_hot", "rank_skew"):
                for slot in range(args.top_k):
                    model.router.weight[slot, 0] = args.top_k - slot
            if args.scenario == "rank_skew":
                last_rank_start = args.experts - local_experts
                for slot in range(args.top_k):
                    model.router.weight[last_rank_start + slot, 0] = -(
                        args.top_k - slot
                    )
            if args.scenario == "segment_skew":
                for source_rank in range(world):
                    destination = (source_rank + 1) % world
                    destination_start = destination * local_experts
                    for slot in range(args.top_k):
                        model.router.weight[destination_start + slot, source_rank] = (
                            args.top_k - slot
                        )

    generator = torch.Generator(device=device).manual_seed(1000 + rank)
    x = torch.randn(
        args.tokens,
        args.hidden,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=args.mode == "training",
    )
    if args.scenario != "balanced":
        with torch.no_grad():
            x.zero_()
            if args.scenario == "single_hot":
                x[:, 0] = 1.0
            elif args.scenario == "rank_skew":
                split = (3 * args.tokens) // 4
                x[:split, 0] = 1.0
                x[split:, 0] = -1.0
            else:
                x[:, rank] = 1.0
    grad = torch.randn(
        x.shape,
        generator=generator,
        dtype=x.dtype,
        device=device,
    )
    optimizer = None
    optimizer_name = None
    if args.optimizer_step:
        if args.optimizer == "mxfp8_sgd":
            optimizer = Mxfp8SGD(model, lr=1e-3)
            optimizer_name = "Mxfp8SGD"
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer_name = "torch.optim.SGD"

    def run() -> None:
        if args.mode == "forward":
            with torch.inference_mode():
                model(x, is_inference_mode=True)
            return
        model.zero_grad(set_to_none=True)
        x.grad = None
        output, aux_loss = model(x)
        ((output * grad).float().sum() + 0.01 * aux_loss.float()).backward()
        if optimizer is not None:
            if not args.skip_router_grad_sync:
                model.all_reduce_replicated_gradients_(average=True)
            optimizer.step()

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.reset_peak_memory_stats()

    samples = []
    for _ in range(args.repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        dist.barrier()
        start.record()
        run()
        end.record()
        end.synchronize()
        samples.append(_rank_max(start.elapsed_time(end), device))

    peak_mib = _rank_max(torch.cuda.max_memory_allocated() / 2**20, device)
    if rank == 0:
        print(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(device),
                    "world_size": world,
                    "torch": torch.__version__,
                    "shape": {
                        "tokens_per_rank": args.tokens,
                        "experts": args.experts,
                        "top_k": args.top_k,
                        "hidden": args.hidden,
                        "intermediate": args.intermediate,
                    },
                    "scenario": args.scenario,
                    "backend": args.backend,
                    "mode": args.mode,
                    "optimizer_step": args.optimizer_step,
                    "optimizer": optimizer_name,
                    "replicated_router_grad_sync": bool(
                        optimizer is not None and not args.skip_router_grad_sync
                    ),
                    "fused_transport": not args.disable_fused_transport,
                    "route_pack_policy": model.use_fused_route_pack,
                    "fused_reduce_policy": model.use_fused_reduce,
                    "route_pack_enabled": model._last_route_pack_enabled,
                    "fused_reduce_enabled": model._last_reduce_enabled,
                    "route_pack_min_elements": model.route_pack_min_elements,
                    "zero_material_qdata_policy": model.use_zero_material_qdata,
                    "zero_material_qdata_enabled": model._last_zero_material_qdata,
                    "comm_overlap": model.use_comm_overlap,
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    "max_rank_latency": {
                        "mean_ms": statistics.fmean(samples),
                        "p50_ms": _percentile(samples, 0.50),
                        "p95_ms": _percentile(samples, 0.95),
                        "p99_ms": _percentile(samples, 0.99),
                    },
                    "max_rank_peak_allocated_mib": peak_mib,
                },
                indent=2,
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
