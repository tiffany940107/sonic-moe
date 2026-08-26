#!/usr/bin/env python3
"""EP4 end-to-end benchmark for the SonicMoE SM120 MXFP8 extension.

The data plane intentionally matches the public FlashInfer comparison harness:
source
tokens are deduplicated once per destination rank, expanded and sorted by
local expert, reduced locally, then returned by NCCL all-to-all.  Only the
local MoE math and quantization format change to OCP MXFP8.

``placement=greedy`` is the static EPLB steady-state case. Planning is reported
separately and is excluded from the timed forward. Weight migration is not
performed by this script.
"""

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
from ep4_support.routing import (
    contiguous_placement,
    generate_rank_skew_routing,
    generate_weighted_routing,
    greedy_placement,
    interleaved_placement,
    load_global_routing_trace,
)
from ep4_support.transport import (
    actual_dispatch_bytes,
    combine_deduplicated,
    dispatch_deduplicated,
    expand_and_sort,
    make_dispatch_plan,
)
from sonicmoe.functional.mxfp8 import (
    MXFP8_SCALE_BLOCK_K,
    Mxfp8MoEKernelConfig,
    allocate_mxfp8_weights,
    make_varlen_m_operand,
    mxfp8_down_grouped,
    mxfp8_swiglu_grouped,
    quantize_mxfp8_rows,
    quantize_varlen_m_operand,
)


STAGES = (
    "dispatch_quant_a2a",
    "compact_sort",
    "pack_fc1_sfa",
    "fc1_swiglu_quant",
    "fc2",
    "local_reduce",
    "combine",
)
QUACK_COMMIT = os.environ.get("QUACK_COMMIT", "unknown")
SONIC_COMMIT = os.environ.get("SONIC_COMMIT", "unknown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SonicMoE/QuACK SM120 MXFP8 EP benchmark")
    parser.add_argument("--tokens", type=int, default=4096, help="source tokens per rank")
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--experts", type=int, default=768)
    parser.add_argument("--hidden", type=int, default=2560)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument(
        "--routing",
        choices=["uniform", "zipf", "hotset", "empty", "rank_skew"],
        default="uniform",
    )
    parser.add_argument("--zipf-alpha", type=float, default=1.2)
    parser.add_argument("--hot-experts", type=int, default=24)
    parser.add_argument("--hot-mass", type=float, default=0.8)
    parser.add_argument("--rank-ratio", type=float, default=1.0)
    parser.add_argument("--routing-trace", default="")
    parser.add_argument(
        "--placement", choices=["contiguous", "interleaved", "greedy"], default="contiguous"
    )
    parser.add_argument(
        "--activation-transport", choices=["mxfp8", "bf16"], default="mxfp8"
    )
    parser.add_argument("--fc1-tile-m", type=int, choices=[128, 256], default=128)
    parser.add_argument("--fc1-tile-n", type=int, choices=[128, 256], default=128)
    parser.add_argument("--fc2-tile-m", type=int, choices=[128, 256], default=128)
    parser.add_argument("--fc2-tile-n", type=int, choices=[128, 256], default=128)
    parser.add_argument("--disable-pingpong", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--stability", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-label", default="")
    parser.add_argument(
        "--node-label",
        default="anonymous",
        help="public alias only; hostnames and addresses are intentionally not collected",
    )
    parser.add_argument(
        "--topology-label",
        choices=["same_numa", "cross_numa_2plus2", "unspecified"],
        default="unspecified",
        help="metadata for the externally selected four visible GPUs",
    )
    parser.add_argument("--output", default="results/raw/sonic_mxfp8_ep_e2e.jsonl")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.node_label):
        parser.error("--node-label must be a privacy-safe public alias")
    return args


def rank_max(values: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return values


def expert_payload_bytes(hidden: int, intermediate: int) -> int:
    # FC1 [2I,H] + FC2 [H,I], one FP8 byte/value and one E8M0 byte/K32 block.
    values = 3 * hidden * intermediate
    scales = values // MXFP8_SCALE_BLOCK_K
    return values + scales


def main() -> int:
    args = parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4:
        raise ValueError(f"this customer-case harness requires EP/world size 4, got {world}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.get_device_capability(device)[0] != 12:
        raise RuntimeError("SonicMoE MXFP8 benchmark requires a physical SM120 GPU")
    if args.experts % world:
        raise ValueError("experts must divide world size")
    if args.hidden % 128 or args.intermediate % 128:
        raise ValueError("SM120 MXFP8 path requires hidden/intermediate divisible by 128")
    local_experts = args.experts // world
    initial_placement = contiguous_placement(args.experts, world)

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
            args.tokens,
            args.top_k,
            args.experts,
            world,
            args.rank_ratio,
            seed=args.seed + rank,
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

    # Collect one global expert histogram for both audit metrics and EPLB.
    local_loads = torch.bincount(
        route.expert_ids.reshape(-1).to(device), minlength=args.experts
    ).to(torch.int64)
    dist.all_reduce(local_loads, op=dist.ReduceOp.SUM)
    global_loads = local_loads.cpu()
    placement = initial_placement
    placement_plan_ms = 0.0
    if args.placement == "interleaved":
        placement = interleaved_placement(args.experts, world)
    elif args.placement == "greedy":
        # This is an oracle/current-window static EPLB decision.  Its cost is
        # visible but excluded from each steady-state forward sample.
        torch.cuda.synchronize()
        dist.barrier()
        plan_start = time.perf_counter()
        placement = greedy_placement(global_loads, world)
        dist.barrier()
        plan_local = torch.tensor(
            (time.perf_counter() - plan_start) * 1e3, dtype=torch.float64, device=device
        )
        rank_max(plan_local)
        placement_plan_ms = float(plan_local)

    placement_moved_experts = int((placement != initial_placement).sum())
    placement_rank_loads = torch.bincount(
        placement, weights=global_loads.to(torch.float64), minlength=world
    )
    initial_rank_loads = torch.bincount(
        initial_placement, weights=global_loads.to(torch.float64), minlength=world
    )
    dispatch_plan = make_dispatch_plan(
        route.expert_ids, route.weights, placement, world, device
    )
    x = torch.randn((args.tokens, args.hidden), dtype=torch.bfloat16, device=device)

    # A BF16 identity oracle checks dispatch, top-k masking, weighting, local
    # reduction, and reverse all-to-all independently of MXFP8 numerics.
    identity_dispatch = dispatch_deduplicated(x, dispatch_plan)
    identity_pairs = expand_and_sort(identity_dispatch, local_experts)
    pair_x_id, _, recv_token_id, pair_weights_id, _ = identity_pairs
    identity_local = torch.zeros_like(identity_dispatch.x)
    identity_local.index_add_(
        0, recv_token_id, pair_x_id * pair_weights_id[:, None].to(pair_x_id.dtype)
    )
    identity_out = combine_deduplicated(identity_local, dispatch_plan, args.tokens)
    identity_error = torch.tensor(
        [
            float((identity_out - x).abs().max()),
            float(
                torch.linalg.vector_norm((identity_out - x).float())
                / torch.linalg.vector_norm(x.float())
            ),
        ],
        dtype=torch.float64,
        device=device,
    )
    rank_max(identity_error)

    if rank == 0:
        print("allocating local OCP MXFP8 expert weights", flush=True)
    w1 = allocate_mxfp8_weights(
        local_experts,
        2 * args.intermediate,
        args.hidden,
        device=device,
        seed=args.seed + 1000 + rank,
    )
    w2 = allocate_mxfp8_weights(
        local_experts,
        args.hidden,
        args.intermediate,
        device=device,
        seed=args.seed + 2000 + rank,
    )
    bytes_per_expert = expert_payload_bytes(args.hidden, args.intermediate)
    config = Mxfp8MoEKernelConfig(
        fc1_tile_m=args.fc1_tile_m,
        fc1_tile_n=args.fc1_tile_n,
        fc2_tile_m=args.fc2_tile_m,
        fc2_tile_n=args.fc2_tile_n,
        pingpong=not args.disable_pingpong,
    )

    def forward(with_events: bool = False, transport_override: str | None = None):
        transport_mode = transport_override or args.activation_transport
        events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        total_start.record()

        def stage(fn):
            if not with_events:
                return fn()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            value = fn()
            end.record()
            events.append((start, end))
            return value

        def dispatch_stage():
            if transport_mode == "bf16":
                return dispatch_deduplicated(x, dispatch_plan)
            source_q, source_sf = quantize_mxfp8_rows(x)
            # NCCL treats scale factors as payload bytes.  Moving the uint8
            # view avoids relying on collective support for the new E8M0 dtype.
            return dispatch_deduplicated(
                source_q, dispatch_plan, scales=source_sf.view(torch.uint8)
            )

        dispatched = stage(dispatch_stage)
        pair_x, _, recv_token, pair_weights, indptr = stage(
            lambda: expand_and_sort(dispatched, local_experts)
        )
        if pair_x.shape[0] == 0:
            raise RuntimeError("empty destination ranks are not supported by this benchmark")

        def pack_fc1():
            if transport_mode == "bf16":
                return quantize_varlen_m_operand(pair_x, indptr)
            if dispatched.scales is None:
                raise RuntimeError("MXFP8 transport did not deliver source scales")
            linear_sf = (
                dispatched.scales.index_select(0, recv_token)
                .contiguous()
                .view(torch.float8_e8m0fnu)
            )
            return make_varlen_m_operand(pair_x, linear_sf, indptr)

        fc1_input = stage(pack_fc1)
        postact = stage(lambda: mxfp8_swiglu_grouped(fc1_input, w1, indptr, config=config))
        pair_out = stage(lambda: mxfp8_down_grouped(postact, w2, indptr, config=config))

        def reduce_stage():
            reduced = torch.zeros(
                (dispatched.x.shape[0], args.hidden), dtype=torch.float32, device=device
            )
            reduced.index_add_(0, recv_token, pair_out.float() * pair_weights[:, None])
            return reduced.to(torch.bfloat16)

        reduced = stage(reduce_stage)
        out = stage(lambda: combine_deduplicated(reduced, dispatch_plan, args.tokens))
        total_end.record()
        return out, events, (total_start, total_end), int(pair_x.shape[0])

    # Quantizing before transport and quantizing the same BF16 rows after
    # grouping should be numerically equivalent.  Keep this gate outside the
    # timed sample set.
    bf16_transport_out = forward(False, "bf16")[0]
    mxfp8_transport_out = forward(False, "mxfp8")[0]
    torch.cuda.synchronize()
    transport_diff = (mxfp8_transport_out.float() - bf16_transport_out.float()).reshape(-1)
    bf16_flat = bf16_transport_out.float().reshape(-1)
    mxfp8_flat = mxfp8_transport_out.float().reshape(-1)
    transport_numeric = torch.tensor(
        [
            transport_diff.abs().max(),
            torch.linalg.vector_norm(transport_diff)
            / torch.linalg.vector_norm(bf16_flat).clamp_min(1e-20),
            torch.nn.functional.cosine_similarity(mxfp8_flat, bf16_flat, dim=0),
        ],
        dtype=torch.float64,
        device=device,
    )
    rank_max(transport_numeric[:2])
    dist.all_reduce(transport_numeric[2:], op=dist.ReduceOp.MIN)

    cold_start = time.perf_counter()
    forward(False)
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - cold_start) * 1e3
    for _ in range(max(0, args.warmup - 1)):
        forward(False)
    torch.cuda.synchronize()
    dist.barrier()

    all_stage_events = []
    all_total_events = []
    local_pairs = 0
    host_start = time.perf_counter()
    for _ in range(args.iters):
        _, stage_events, total_events, local_pairs = forward(not args.stability)
        if not args.stability:
            all_stage_events.append(stage_events)
        all_total_events.append(total_events)
    torch.cuda.synchronize()
    host_mean_ms = (time.perf_counter() - host_start) * 1e3 / args.iters

    stage_local = None
    if all_stage_events:
        stage_local = torch.tensor(
            [
                [start.elapsed_time(end) for start, end in sample]
                for sample in all_stage_events
            ],
            dtype=torch.float64,
            device=device,
        )
        rank_max(stage_local)
    total_local = torch.tensor(
        [start.elapsed_time(end) for start, end in all_total_events],
        dtype=torch.float64,
        device=device,
    )
    rank_max(total_local)
    host_value = torch.tensor(host_mean_ms, dtype=torch.float64, device=device)
    rank_max(host_value)

    pair_count = torch.tensor(local_pairs, dtype=torch.int64, device=device)
    pair_all = [torch.empty_like(pair_count) for _ in range(world)]
    dist.all_gather(pair_all, pair_count)
    remote_dispatch = torch.tensor(
        actual_dispatch_bytes(
            dispatch_plan,
            args.hidden,
            x_bytes=1 if args.activation_transport == "mxfp8" else 2,
            scale_bytes_per_token=(args.hidden // MXFP8_SCALE_BLOCK_K)
            if args.activation_transport == "mxfp8"
            else 0,
        ),
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(remote_dispatch, op=dist.ReduceOp.SUM)
    peak = torch.tensor(torch.cuda.max_memory_allocated(), dtype=torch.int64, device=device)
    rank_max(peak)

    if rank == 0:
        total_summary = summarize_ms(total_local.cpu().tolist())
        stage_cpu = stage_local.cpu() if stage_local is not None else None
        pair_values = [int(value) for value in pair_all]
        initial_values = [int(value) for value in initial_rank_loads]
        placement_values = [int(value) for value in placement_rank_loads]
        record = {
            "schema_version": 1,
            "benchmark": "sonic_mxfp8_ep_e2e",
            "timestamp_unix": time.time(),
            "node_label": args.node_label,
            "topology": args.topology_label,
            "run_label": args.run_label,
            "world_size": world,
            "global_tokens": args.tokens * world,
            "source_tokens_per_rank": args.tokens,
            "top_k": args.top_k,
            "global_experts": args.experts,
            "local_experts": local_experts,
            "hidden": args.hidden,
            "intermediate_after_swiglu": args.intermediate,
            "fc1_physical_output": 2 * args.intermediate,
            "routing": route.scenario if args.routing_trace else args.routing,
            "zipf_alpha": args.zipf_alpha if args.routing == "zipf" else None,
            "hot_experts": args.hot_experts if args.routing == "hotset" else None,
            "hot_mass": args.hot_mass if args.routing == "hotset" else None,
            "routing_trace_path": args.routing_trace or None,
            "routing_trace_sha256": trace_sha256,
            "placement": args.placement,
            "eplb_policy": "static_equal_capacity_lpt_current_histogram"
            if args.placement == "greedy"
            else None,
            "placement_plan_ms": placement_plan_ms,
            "placement_moved_experts": placement_moved_experts,
            "initial_rank_pair_loads": initial_values,
            "placement_rank_pair_loads": placement_values,
            "initial_rank_max_over_mean": max(initial_values)
            / (sum(initial_values) / world),
            "placement_rank_max_over_mean": max(placement_values)
            / (sum(placement_values) / world),
            "expert_weight_scale_bytes": bytes_per_expert,
            "placement_logical_copy_bytes": placement_moved_experts * bytes_per_expert,
            "placement_migration_in_timed_forward": False,
            "seed": args.seed,
            "warmup": args.warmup,
            "iterations": args.iters,
            "activation_transport": args.activation_transport,
            "transport": f"torch_nccl_all_to_all_deduplicated_{args.activation_transport}",
            "quantization": {
                "values": "OCP MXFP8 E4M3",
                "scales": "E8M0",
                "semantic_granularity": "1x32 along reduction K",
                "group_size_k": MXFP8_SCALE_BLOCK_K,
                "variable_m_scale_storage": "128-row expert-tile-padded CUTLASS blocked",
            },
            "compute": "sonicmoe_extension_quack_sm120_mxfp8_varlen",
            "sonic_commit": SONIC_COMMIT,
            "quack_commit": QUACK_COMMIT,
            "kernel_config": {
                "fc1_tile_m": config.fc1_tile_m,
                "fc1_tile_n": config.fc1_tile_n,
                "fc2_tile_m": config.fc2_tile_m,
                "fc2_tile_n": config.fc2_tile_n,
                "pingpong": config.pingpong,
                "dynamic_persistent": config.dynamic_persistent,
            },
            "cold_or_cache_lookup_ms": cold_ms,
            "transport_identity_max_abs": float(identity_error[0]),
            "transport_identity_rel_l2": float(identity_error[1]),
            "mxfp8_transport_vs_bf16_max_abs": float(transport_numeric[0]),
            "mxfp8_transport_vs_bf16_rel_l2": float(transport_numeric[1]),
            "mxfp8_transport_vs_bf16_cosine": float(transport_numeric[2]),
            "local_pair_counts": pair_values,
            "rank_pair_max_over_mean": max(pair_values) / (sum(pair_values) / world),
            "global_remote_dispatch_bytes": int(remote_dispatch),
            "stage_rank_max": {
                name: summarize_ms(stage_cpu[:, index].tolist())
                for index, name in enumerate(STAGES)
            }
            if stage_cpu is not None
            else {},
            "e2e_rank_max": total_summary,
            "host_rank_max_mean_ms": float(host_value),
            "global_tokens_per_second": args.tokens
            * world
            / (total_summary["p50_ms"] / 1e3),
            "global_pairs_per_second": args.tokens
            * world
            * args.top_k
            / (total_summary["p50_ms"] / 1e3),
            "peak_allocated_bytes": int(peak),
            "scope_note": (
                "Steady-state forward excludes static EPLB planning and weight movement; "
                "those costs are emitted separately and migration is measured by the "
                "MXFP8 migration benchmark."
            ),
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
