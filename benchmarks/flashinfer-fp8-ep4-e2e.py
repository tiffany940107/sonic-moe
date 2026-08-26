#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time

import torch
import torch.distributed as dist

from ep4_support.common import summarize_ms
from ep4_support.fp8_quant import (
    allocate_synthetic_fp8_weights,
    pack_mn_major_scales,
    quantize_and_pack_activation,
    scale_destination_indices,
)
from ep4_support.routing import (
    contiguous_placement,
    generate_rank_skew_routing,
    generate_weighted_routing,
    greedy_placement,
    interleaved_placement,
    load_global_routing_trace,
)
from ep4_support.placement.persistent_replica import (
    allocate_replica_quota,
    assign_global_pairs_comm_aware,
    assign_global_pairs_to_replica_quota,
    plan_persistent_replicas,
)
from ep4_support.transport import local_expert_indices
from ep4_support.transport import (
    actual_dispatch_bytes,
    combine_deduplicated,
    dispatch_deduplicated,
    expand_and_sort,
    make_dispatch_plan,
)


STAGES = ("dispatch", "compact_sort", "quant_fc1", "fc1", "quant_fc2", "fc2_reduce", "combine")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ordinary PCIe EP + FlashInfer SM120 FP8 E2E benchmark")
    p.add_argument("--tokens", type=int, default=4096, help="source tokens per rank")
    p.add_argument("--top-k", type=int, default=24)
    p.add_argument("--experts", type=int, default=768)
    p.add_argument("--hidden", type=int, default=2560)
    p.add_argument("--intermediate", type=int, default=1024)
    p.add_argument("--routing", choices=["uniform", "zipf", "hotset", "empty", "rank_skew"], default="uniform")
    p.add_argument("--zipf-alpha", type=float, default=1.2)
    p.add_argument("--hot-experts", type=int, default=24)
    p.add_argument("--hot-mass", type=float, default=0.8)
    p.add_argument(
        "--routing-trace",
        default="",
        help="global rank-major .pt dict containing expert_ids and optional weights",
    )
    p.add_argument("--rank-ratio", type=float, default=1.0)
    p.add_argument(
        "--placement", choices=["contiguous", "interleaved", "greedy"], default="contiguous"
    )
    p.add_argument("--replicas-per-rank", type=int, choices=[0, 1, 2, 4], default=0)
    p.add_argument(
        "--replica-quota",
        choices=["pair_balanced", "communication_aware"],
        default="pair_balanced",
        help="materialization policy; both preserve the same compute quota",
    )
    p.add_argument(
        "--activation-transport", choices=["bf16", "fp8"], default="bf16"
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument(
        "--stability",
        action="store_true",
        help="retain per-iteration E2E events but omit seven per-stage event pairs",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--run-label",
        default="",
        help="privacy-safe free-form provenance label",
    )
    p.add_argument(
        "--node-label",
        default="anonymous",
        help="public node alias; machine hostnames are not collected",
    )
    p.add_argument(
        "--topology-label",
        choices=["same_numa", "cross_numa_2plus2", "unspecified"],
        default="unspecified",
        help="metadata for the externally selected four visible GPUs",
    )
    p.add_argument("--output", default="results/raw/e2e_fp8_ep.jsonl")
    args = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.node_label):
        p.error("--node-label must be a privacy-safe public alias")
    return args


def invoke(a, b, a_scale, b_scale, indptr, out, gated):
    from flashinfer.grouped_mm import moe_gemm_fp8_nt_groupwise

    return moe_gemm_fp8_nt_groupwise(
        a, b, a_scale, b_scale, indptr, out=out, is_gated=gated
    )


def rank_max(values: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return values


def main() -> int:
    args = parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4:
        raise ValueError(f"this customer-case harness requires EP/world size 4, got {world}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if args.experts % world:
        raise ValueError("experts must divide world size")
    local_experts = args.experts // world
    initial_placement = contiguous_placement(args.experts, world)
    placement = initial_placement
    placement_plan_ms = 0.0
    trace_sha256 = None
    if args.routing_trace:
        route, trace_sha256 = load_global_routing_trace(
            args.routing_trace,
            source_tokens_per_rank=args.tokens,
            top_k=args.top_k,
            experts=args.experts,
            ep_size=world,
            rank=rank,
        )
    elif args.routing == "rank_skew":
        route = generate_rank_skew_routing(
            args.tokens, args.top_k, args.experts, world, args.rank_ratio, seed=args.seed + rank
        )
    else:
        route = generate_weighted_routing(
            args.tokens,
            args.top_k,
            args.experts,
            args.routing,
            seed=args.seed + rank,
            ep_size=world,
            zipf_alpha=args.zipf_alpha,
            hot_experts=args.hot_experts,
            hot_mass=args.hot_mass,
        )
    if args.placement == "interleaved":
        placement = interleaved_placement(args.experts, world)
    elif args.placement == "greedy":
        # Measure the complete current-batch planning path separately from the
        # data-plane forward: local histogram, global NCCL reduction, CPU LPT
        # placement, and a final synchronization.  Route generation itself is
        # intentionally outside this interval because a serving engine would
        # normally maintain the histogram while routing tokens.
        torch.cuda.synchronize()
        dist.barrier()
        placement_plan_start = time.perf_counter()
        local_loads = torch.bincount(
            route.expert_ids.reshape(-1), minlength=args.experts
        ).to(device=device, dtype=torch.int64)
        dist.all_reduce(local_loads, op=dist.ReduceOp.SUM)
        placement = greedy_placement(local_loads.cpu(), world)
        torch.cuda.synchronize()
        dist.barrier()
        placement_plan_local_ms = (time.perf_counter() - placement_plan_start) * 1e3
        placement_plan_value = torch.tensor(
            placement_plan_local_ms, dtype=torch.float64, device=device
        )
        rank_max(placement_plan_value)
        placement_plan_ms = float(placement_plan_value)
    placement_moved_experts = int((placement != initial_placement).sum())
    physical_experts = local_experts
    replica_plan = None
    replica_rank_loads = None
    replica_slot_map = None
    if args.replicas_per_rank:
        if args.placement != "contiguous":
            raise ValueError("persistent replica ablation currently requires contiguous base placement")
        local_ids_gpu = route.expert_ids.to(device=device, dtype=torch.int64)
        gathered_ids = [torch.empty_like(local_ids_gpu) for _ in range(world)]
        dist.all_gather(gathered_ids, local_ids_gpu)
        global_ids = torch.cat(gathered_ids, dim=0).cpu()
        global_loads = torch.bincount(global_ids.reshape(-1), minlength=args.experts)
        replica_plan = plan_persistent_replicas(
            global_loads, placement, world, args.replicas_per_rank
        )
        replica_rank_loads, quotas = allocate_replica_quota(
            global_loads, placement, replica_plan.replicas, world
        )
        assignment = (
            assign_global_pairs_comm_aware
            if args.replica_quota == "communication_aware"
            else assign_global_pairs_to_replica_quota
        )
        assignment_args = (
                global_ids,
                placement,
                replica_plan.replicas,
                quotas,
                world,
                args.replicas_per_rank,
        )
        if args.replica_quota == "communication_aware":
            assignment_args += (args.tokens,)
        all_destinations, all_local_ids, replica_slot_map = assignment(*assignment_args)
        lo, hi = rank * args.tokens, (rank + 1) * args.tokens
        pair_destinations = all_destinations[lo:hi]
        pair_local_ids = all_local_ids[lo:hi]
        physical_experts += args.replicas_per_rank
        dispatch_plan = make_dispatch_plan(
            route.expert_ids,
            route.weights,
            placement,
            world,
            device,
            pair_destinations=pair_destinations,
            pair_local_ids=pair_local_ids,
        )
    else:
        dispatch_plan = make_dispatch_plan(
            route.expert_ids, route.weights, placement, world, device
        )
    x = torch.randn((args.tokens, args.hidden), dtype=torch.bfloat16, device=device)

    # Transport-only identity oracle: weighted expert outputs are x, so the
    # source must recover x after local reduction and reverse all-to-all.
    d = dispatch_deduplicated(x, dispatch_plan)
    pair_x, _, recv_token, pair_weights, _ = expand_and_sort(d, physical_experts)
    identity_local = torch.zeros_like(d.x)
    identity_local.index_add_(0, recv_token, pair_x * pair_weights[:, None].to(pair_x.dtype))
    identity_out = combine_deduplicated(identity_local, dispatch_plan, args.tokens)
    transport_max_abs = float((identity_out - x).abs().max())
    transport_rel_l2 = float(
        torch.linalg.vector_norm((identity_out - x).float())
        / torch.linalg.vector_norm(x.float())
    )
    error = torch.tensor([transport_max_abs, transport_rel_l2], dtype=torch.float64, device=device)
    dist.all_reduce(error, op=dist.ReduceOp.MAX)

    if rank == 0:
        print("allocating 4-rank local FP8 expert weights", flush=True)
    w1, w1_scale = allocate_synthetic_fp8_weights(
        physical_experts, 2 * args.intermediate, args.hidden, device=device, seed=args.seed + 1000 + rank
    )
    w2, w2_scale = allocate_synthetic_fp8_weights(
        physical_experts, args.hidden, args.intermediate, device=device, seed=args.seed + 2000 + rank
    )
    expert_weight_scale_bytes = (
        w1[0].numel() * w1.element_size()
        + w2[0].numel() * w2.element_size()
        + w1_scale[0].numel() * w1_scale.element_size()
        + w2_scale[0].numel() * w2_scale.element_size()
    )
    # This is the global logical payload that an online contiguous->target
    # ownership change would have to relocate at least once.  The forward
    # benchmark still allocates weights directly in their final placement;
    # the separate migration benchmark measures the actual transfer path.
    placement_copy_bytes = placement_moved_experts * expert_weight_scale_bytes
    replica_preload_ms = 0.0
    replica_copy_bytes = 0
    if replica_plan is not None and replica_slot_map is not None:
        base_slots = local_expert_indices(placement, world)
        scratch_w1 = torch.empty_like(w1[0])
        scratch_w2 = torch.empty_like(w2[0])
        scratch_s1 = torch.empty_like(w1_scale[0])
        scratch_s2 = torch.empty_like(w2_scale[0])
        preload_start = time.perf_counter()
        for expert, targets in sorted(replica_plan.replicas.items()):
            owner = int(placement[expert])
            owner_slot = int(base_slots[expert])
            for target in sorted(targets):
                target_slot = int(replica_slot_map[expert, target])
                tensors = (
                    (w1, scratch_w1, True),
                    (w1_scale, scratch_s1, False),
                    (w2, scratch_w2, True),
                    (w2_scale, scratch_s2, False),
                )
                for storage, scratch, byte_view in tensors:
                    if rank == owner:
                        buffer = storage[owner_slot]
                    elif rank == target:
                        buffer = storage[target_slot]
                    else:
                        buffer = scratch
                    dist.broadcast(buffer.view(torch.uint8) if byte_view else buffer, src=owner)
                replica_copy_bytes += (
                    w1[0].numel()
                    + w2[0].numel()
                    + 4 * (w1_scale[0].numel() + w2_scale[0].numel())
                )
        torch.cuda.synchronize(); dist.barrier()
        replica_preload_ms = (time.perf_counter() - preload_start) * 1e3
        preload = torch.tensor(replica_preload_ms, dtype=torch.float64, device=device)
        rank_max(preload)
        replica_preload_ms = float(preload)

    def forward(with_events: bool = False, transport_override: str | None = None):
        events = []
        transport_mode = transport_override or args.activation_transport
        e2e_start = torch.cuda.Event(enable_timing=True)
        e2e_end = torch.cuda.Event(enable_timing=True)
        e2e_start.record()

        def stage(fn):
            if with_events:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(); value = fn(); end.record(); events.append((start, end)); return value
            return fn()

        def dispatch_stage():
            if transport_mode == "bf16":
                return dispatch_deduplicated(x, dispatch_plan)
            from flashinfer.testing.utils import per_token_cast_to_fp8

            source_fp8, source_scales = per_token_cast_to_fp8(x)
            return dispatch_deduplicated(
                source_fp8, dispatch_plan, scales=source_scales
            )

        dispatched = stage(dispatch_stage)
        packed = stage(lambda: expand_and_sort(dispatched, physical_experts))
        pair_x, _, recv_token, pair_weights, indptr = packed
        destinations = scale_destination_indices(indptr)
        is_empty_rank = pair_x.shape[0] == 0

        def quant_fc1():
            if not is_empty_rank:
                if transport_mode == "fp8":
                    if dispatched.scales is None:
                        raise RuntimeError("FP8 transport did not deliver activation scales")
                    row_scales = dispatched.scales[recv_token]
                    return pair_x, pack_mn_major_scales(
                        row_scales, indptr, destinations
                    )
                return quantize_and_pack_activation(pair_x, indptr, destinations)
            return (
                torch.empty_like(pair_x, dtype=torch.float8_e4m3fn),
                torch.zeros(
                    (args.hidden // 128, ((physical_experts * 3) // 4) * 4),
                    dtype=torch.float32,
                    device=device,
                ),
            )

        q1, s1 = stage(quant_fc1)
        fc1_out = torch.empty((pair_x.shape[0], args.intermediate), dtype=torch.bfloat16, device=device)
        stage(
            lambda: None
            if is_empty_rank
            else invoke(q1, w1, s1, w1_scale, indptr, fc1_out, True)
        )

        def quant_fc2():
            if not is_empty_rank:
                return quantize_and_pack_activation(fc1_out, indptr, destinations)
            return (
                torch.empty_like(fc1_out, dtype=torch.float8_e4m3fn),
                torch.zeros(
                    (args.intermediate // 128, ((physical_experts * 3) // 4) * 4),
                    dtype=torch.float32,
                    device=device,
                ),
            )

        q2, s2 = stage(quant_fc2)

        def fc2_reduce():
            pair_out = torch.empty((pair_x.shape[0], args.hidden), dtype=torch.bfloat16, device=device)
            if not is_empty_rank:
                invoke(q2, w2, s2, w2_scale, indptr, pair_out, False)
            reduced = torch.zeros((dispatched.x.shape[0], args.hidden), dtype=torch.float32, device=device)
            reduced.index_add_(0, recv_token, pair_out.float() * pair_weights[:, None])
            return reduced.to(torch.bfloat16)

        reduced = stage(fc2_reduce)
        out = stage(lambda: combine_deduplicated(reduced, dispatch_plan, args.tokens))
        e2e_end.record()
        return out, events, (e2e_start, e2e_end), int(pair_x.shape[0])

    activation_transport_error = torch.zeros(3, dtype=torch.float64, device=device)
    if args.activation_transport == "fp8":
        reference_out = forward(False, "bf16")[0]
        fp8_out = forward(False, "fp8")[0]
        torch.cuda.synchronize()
        diff = (fp8_out.float() - reference_out.float()).reshape(-1)
        ref = reference_out.float().reshape(-1)
        fp8_flat = fp8_out.float().reshape(-1)
        activation_transport_error[0] = diff.abs().max()
        activation_transport_error[1] = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(ref)
        activation_transport_error[2] = torch.nn.functional.cosine_similarity(
            fp8_flat, ref, dim=0
        )
        dist.all_reduce(activation_transport_error[:2], op=dist.ReduceOp.MAX)
        dist.all_reduce(activation_transport_error[2:], op=dist.ReduceOp.MIN)

    # Cold JIT/load is explicit and excluded.
    cold_start = time.perf_counter()
    forward(False)
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - cold_start) * 1e3
    for _ in range(args.warmup - 1):
        forward(False)
    torch.cuda.synchronize(); dist.barrier()

    all_events = []
    total_events = []
    local_pairs = 0
    host_start = time.perf_counter()
    for _ in range(args.iters):
        _, events, total, local_pairs = forward(not args.stability)
        if not args.stability:
            all_events.append(events)
        total_events.append(total)
    torch.cuda.synchronize()
    host_ms = (time.perf_counter() - host_start) * 1e3 / args.iters
    stage_local = None
    if all_events:
        stage_local = torch.tensor(
            [[start.elapsed_time(end) for start, end in events] for events in all_events],
            dtype=torch.float64,
            device=device,
        )
    total_local = torch.tensor(
        [start.elapsed_time(end) for start, end in total_events], dtype=torch.float64, device=device
    )
    if stage_local is not None:
        rank_max(stage_local)
    rank_max(total_local)
    host = torch.tensor(host_ms, dtype=torch.float64, device=device); rank_max(host)
    pair_counts = torch.tensor(local_pairs, dtype=torch.int64, device=device)
    pair_all = [torch.empty_like(pair_counts) for _ in range(world)]
    dist.all_gather(pair_all, pair_counts)
    dispatch_bytes = torch.tensor(
        actual_dispatch_bytes(
            dispatch_plan,
            args.hidden,
            x_bytes=1 if args.activation_transport == "fp8" else 2,
            scale_bytes_per_token=(args.hidden // 128) * 4
            if args.activation_transport == "fp8"
            else 0,
        ),
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(dispatch_bytes, op=dist.ReduceOp.SUM)
    peak = torch.tensor(torch.cuda.max_memory_allocated(), dtype=torch.int64, device=device)
    rank_max(peak)
    if rank == 0:
        stage_cpu = stage_local.cpu() if stage_local is not None else None
        record = {
            "schema_version": 1,
            "benchmark": "sm120_fp8_ep_e2e",
            "timestamp_unix": time.time(),
            "node_label": args.node_label,
            "topology": args.topology_label,
            "flashinfer_commit": os.environ.get("FLASHINFER_COMMIT", "unknown"),
            "run_label": args.run_label,
            "world_size": world,
            "global_tokens": args.tokens * world,
            "source_tokens_per_rank": args.tokens,
            "top_k": args.top_k,
            "global_experts": args.experts,
            "local_experts": local_experts,
            "physical_experts_per_rank": physical_experts,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "routing": route.scenario if args.routing_trace else args.routing,
            "zipf_alpha": args.zipf_alpha if args.routing == "zipf" else None,
            "hot_experts": args.hot_experts if args.routing == "hotset" else None,
            "hot_mass": args.hot_mass if args.routing == "hotset" else None,
            "routing_trace_path": args.routing_trace or None,
            "routing_trace_sha256": trace_sha256,
            "placement": args.placement,
            "placement_plan_ms": placement_plan_ms,
            "placement_moved_experts": placement_moved_experts,
            "expert_weight_scale_bytes": expert_weight_scale_bytes,
            "placement_copy_bytes": placement_copy_bytes,
            "replicas_per_rank": args.replicas_per_rank,
            "replica_quota": args.replica_quota,
            "replica_map": replica_plan.replicas if replica_plan else {},
            "replica_predicted_rank_pairs": replica_rank_loads.tolist()
            if replica_rank_loads is not None
            else None,
            "replica_preload_ms": replica_preload_ms,
            "replica_copy_bytes": replica_copy_bytes,
            "rank_ratio_target": args.rank_ratio,
            "seed": args.seed,
            "warmup": args.warmup,
            "iterations": args.iters,
            "transport": f"torch_nccl_all_to_all_deduplicated_{args.activation_transport}",
            "activation_transport": args.activation_transport,
            "fp8_transport_vs_bf16_max_abs": float(activation_transport_error[0]),
            "fp8_transport_vs_bf16_rel_l2": float(activation_transport_error[1]),
            "fp8_transport_vs_bf16_cosine": float(activation_transport_error[2])
            if args.activation_transport == "fp8"
            else None,
            "compute": "flashinfer_cute_sm120_fp8_groupwise",
            "cold_start_ms": cold_ms,
            "transport_identity_max_abs": float(error[0]),
            "transport_identity_rel_l2": float(error[1]),
            "local_pair_counts": [int(x) for x in pair_all],
            "rank_pair_max_over_mean": max(int(x) for x in pair_all)
            / (sum(int(x) for x in pair_all) / world),
            "global_remote_dispatch_bytes": int(dispatch_bytes),
            "stage_rank_max": {
                name: summarize_ms(stage_cpu[:, i].tolist())
                for i, name in enumerate(STAGES)
            }
            if stage_cpu is not None
            else {},
            "e2e_rank_max": summarize_ms(total_local.cpu().tolist()),
            "host_rank_max_mean_ms": float(host),
            "global_tokens_per_second": args.tokens * world
            / (summarize_ms(total_local.cpu().tolist())["p50_ms"] / 1e3),
            "global_pairs_per_second": args.tokens * world * args.top_k
            / (summarize_ms(total_local.cpu().tolist())["p50_ms"] / 1e3),
            "peak_allocated_bytes": int(peak),
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
    dist.barrier(); dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
