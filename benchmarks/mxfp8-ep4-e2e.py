#!/usr/bin/env python3
"""EP4 end-to-end benchmark for the SonicMoE SM120 MXFP8 extension.

The data plane intentionally matches the public FlashInfer comparison harness:
source
tokens are deduplicated once per destination rank, expanded and sorted by
local expert, reduced locally, then returned by NCCL all-to-all.  Only the
local MoE math and quantization format change to OCP MXFP8.

``placement=greedy`` is the static EPLB steady-state case. Planning and real
weight migration are reported separately and excluded from the timed forward.
Use ``--real-weight-migration`` to move both FP8 values and E8M0 scales before
timing; omitting it retains the historical logical-placement-only behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
import torch.distributed as dist
from ep4_support.common import summarize_ms
from ep4_support.placement.experimental_sonic_replica import (
    allocate_quota,
    copy_replica_bank,
    materialize_comm_aware,
    plan_replicas,
)
from ep4_support.placement.sonic_migration import migrate_operand
from ep4_support.placement.static_greedy import limit_migration
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
from quack.blockscaled import BlockScaledOperand

from sonicmoe.functional.mxfp8 import (
    MXFP8_SCALE_BLOCK_K,
    Mxfp8MoEKernelConfig,
    allocate_mxfp8_moe_workspace,
    allocate_mxfp8_weights,
    make_varlen_m_operand,
    mxfp8_down_grouped,
    mxfp8_down_grouped_weighted_reduce,
    mxfp8_swiglu_grouped,
    mxfp8_swiglu_grouped_out,
    quantize_mxfp8_rows,
    quantize_varlen_m_operand,
)
from sonicmoe.functional.mxfp8_route_pack import (
    allocate_mxfp8_route_pack_workspace,
    route_pack_mxfp8,
)
from sonicmoe.functional.mxfp8_weighted_reduce import (
    allocate_mxfp8_weighted_reduce_workspace,
    hybrid_weighted_reduce_mxfp8,
    segmented_weighted_reduce_mxfp8,
    weighted_reduce_mxfp8,
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
        "--real-weight-migration",
        action="store_true",
        help="collectively move Sonic MXFP8 expert values/scales into the selected placement",
    )
    parser.add_argument(
        "--migration-limit",
        type=int,
        default=0,
        help="maximum experts changed through complete capacity-preserving cycles; 0 is unlimited",
    )
    parser.add_argument(
        "--experimental-replica-slots",
        type=int,
        choices=[0, 1, 2, 4, 8, 16],
        default=0,
        help="EXPERIMENTAL hot-expert shadow slots per rank; disabled by default",
    )
    parser.add_argument(
        "--experimental-max-copies-per-expert", type=int, choices=[2, 3, 4], default=4
    )
    parser.add_argument(
        "--experimental-minimum-hot-expert-ratio", type=float, default=4.0
    )
    parser.add_argument(
        "--activation-transport", choices=["mxfp8", "bf16"], default="mxfp8"
    )
    parser.add_argument(
        "--prequantized-source",
        action="store_true",
        help="E0: keep source MXFP8 values/scales outside the timed forward",
    )
    parser.add_argument(
        "--data-path",
        choices=[
            "baseline",
            "workspace",
            "route_pack",
            "fused",
            "fused_atomic",
            "fused_segmented",
            "fused_hybrid",
            "fc2_epilogue_atomic",
        ],
        default="baseline",
        help=(
            "incremental 0902 path: workspace reuses local buffers; route_pack "
            "also replaces sort/gather/SFA pack; fused additionally uses the "
            "single-pass FP32 weighted scatter-reduce"
        ),
    )
    parser.add_argument("--fc1-tile-m", type=int, choices=[128, 256], default=128)
    parser.add_argument("--fc1-tile-n", type=int, choices=[128, 256], default=128)
    parser.add_argument("--fc2-tile-m", type=int, choices=[128, 256], default=128)
    parser.add_argument("--fc2-tile-n", type=int, choices=[128, 256], default=128)
    parser.add_argument("--disable-pingpong", action="store_true")
    parser.add_argument(
        "--heavy-expert-first",
        action="store_true",
        help="schedule longer expert segments first without changing physical layout",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--atomic-variance-runs",
        type=int,
        default=5,
        help="untimed repeated-output audit for atomic reduction paths",
    )
    parser.add_argument(
        "--reference-chunk-pairs",
        type=int,
        default=262144,
        help="setup-only C4 reference chunk size; timed baseline is unchanged",
    )
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="annotate E0 stages for Nsight Systems collection",
    )
    parser.add_argument(
        "--cuda-profiler-capture",
        action="store_true",
        help="bracket only timed iterations for external profiler capture",
    )
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
    if args.migration_limit < 0:
        parser.error("--migration-limit must be non-negative")
    if args.reference_chunk_pairs <= 0:
        parser.error("--reference-chunk-pairs must be positive")
    if (
        args.experimental_replica_slots
        and args.placement != "contiguous"
        and not args.real_weight_migration
    ):
        parser.error(
            "experimental replica/hybrid with a non-contiguous placement requires "
            "--real-weight-migration"
        )
    return args


def rank_max(values: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return values


def expert_payload_bytes(hidden: int, intermediate: int) -> int:
    # FC1 [2I,H] + FC2 [H,I], one FP8 byte/value and one E8M0 byte/K32 block.
    values = 3 * hidden * intermediate
    scales = values // MXFP8_SCALE_BLOCK_K
    return values + scales


def base_expert_view(
    operand: BlockScaledOperand, local_experts: int
) -> BlockScaledOperand:
    """View the owner bank without experimental replica reserve slots."""
    return BlockScaledOperand.from_parts(
        operand.qdata[:local_experts],
        operand.scale[:local_experts],
        operand.format.name,
    )


def install_base_with_shadow_capacity(
    base: BlockScaledOperand, template: BlockScaledOperand
) -> BlockScaledOperand:
    """Install a migrated owner bank while retaining shadow-slot capacity."""
    if base.qdata.shape[0] == template.qdata.shape[0]:
        return base
    qdata = torch.empty_like(template.qdata)
    scale = torch.empty_like(template.scale)
    qdata[: base.qdata.shape[0]].copy_(base.qdata)
    scale[: base.scale.shape[0]].copy_(base.scale)
    return BlockScaledOperand.from_parts(qdata, scale, base.format.name)


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
        full_target = greedy_placement(global_loads, world)
        placement = limit_migration(
            initial_placement, full_target, args.migration_limit
        )
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
    initial_dispatch_plan = make_dispatch_plan(
        route.expert_ids, route.weights, initial_placement, world, device
    )
    placement_dispatch_plan = make_dispatch_plan(
        route.expert_ids, route.weights, placement, world, device
    )
    dispatch_plan = placement_dispatch_plan
    physical_experts = local_experts
    experimental_replica_plan = None
    experimental_replica_loads = None
    experimental_slot_map = None
    if args.experimental_replica_slots:
        ids_device = route.expert_ids.to(device)
        gathered_ids = [torch.empty_like(ids_device) for _ in range(world)]
        dist.all_gather(gathered_ids, ids_device)
        global_ids = torch.cat(gathered_ids).cpu()
        experimental_replica_plan = plan_replicas(
            global_loads,
            placement,
            world,
            args.experimental_replica_slots,
            max_copies_per_expert=args.experimental_max_copies_per_expert,
            minimum_hot_expert_ratio=args.experimental_minimum_hot_expert_ratio,
        )
        if experimental_replica_plan.count:
            experimental_replica_loads, quotas = allocate_quota(
                global_loads,
                placement,
                experimental_replica_plan.replicas,
                world,
            )
            destinations, local_ids, experimental_slot_map = materialize_comm_aware(
                global_ids,
                placement,
                experimental_replica_plan.replicas,
                quotas,
                world,
                args.experimental_replica_slots,
                args.tokens,
            )
            lower, upper = rank * args.tokens, (rank + 1) * args.tokens
            dispatch_plan = make_dispatch_plan(
                route.expert_ids,
                route.weights,
                placement,
                world,
                device,
                pair_destinations=destinations[lower:upper],
                pair_local_ids=local_ids[lower:upper],
            )
            physical_experts += args.experimental_replica_slots
        else:
            experimental_replica_plan = None
    # A BF16 identity oracle checks dispatch, top-k masking, weighting, local
    # reduction, and reverse all-to-all independently of MXFP8 numerics.  One
    # scalar per token is sufficient to audit metadata and weights; using H
    # columns here used to create a multi-GiB setup-only pair tensor on skewed
    # long shapes.
    identity_x = torch.randn((args.tokens, 1), dtype=torch.bfloat16, device=device)
    identity_dispatch = dispatch_deduplicated(identity_x, dispatch_plan)
    identity_pairs = expand_and_sort(identity_dispatch, physical_experts)
    pair_x_id, pair_expert_id, recv_token_id, pair_weights_id, _ = identity_pairs
    identity_local = torch.zeros_like(identity_dispatch.x)
    identity_local.index_add_(
        0, recv_token_id, pair_x_id * pair_weights_id[:, None].to(pair_x_id.dtype)
    )
    identity_out = combine_deduplicated(identity_local, dispatch_plan, args.tokens)
    identity_error = torch.tensor(
        [
            float((identity_out - identity_x).abs().max()),
            float(
                torch.linalg.vector_norm((identity_out - identity_x).float())
                / torch.linalg.vector_norm(identity_x.float())
            ),
        ],
        dtype=torch.float64,
        device=device,
    )
    rank_max(identity_error)
    x = torch.randn((args.tokens, args.hidden), dtype=torch.bfloat16, device=device)
    prequantized_source = quantize_mxfp8_rows(x) if args.prequantized_source else None

    if rank == 0:
        print(
            f"allocating {physical_experts} local OCP MXFP8 physical experts",
            flush=True,
        )
    w1 = allocate_mxfp8_weights(
        physical_experts,
        2 * args.intermediate,
        args.hidden,
        device=device,
        seed=args.seed + 1000 + rank,
    )
    w2 = allocate_mxfp8_weights(
        physical_experts,
        args.hidden,
        args.intermediate,
        device=device,
        seed=args.seed + 2000 + rank,
    )
    initial_weight_bank = (
        (base_expert_view(w1, local_experts), base_expert_view(w2, local_experts))
        if placement_moved_experts and args.real_weight_migration
        else None
    )
    bytes_per_expert = expert_payload_bytes(args.hidden, args.intermediate)
    weight_migration_ms = 0.0
    weight_migration_bytes = 0
    weight_migration_sample_bad_bytes = 0
    weight_migration_sampled_bytes = 0
    if placement_moved_experts and args.real_weight_migration:
        torch.cuda.synchronize()
        dist.barrier()
        migration_start = time.perf_counter()
        migrated_w1 = migrate_operand(
            base_expert_view(w1, local_experts), initial_placement, placement
        )
        migrated_w2 = migrate_operand(
            base_expert_view(w2, local_experts), initial_placement, placement
        )
        w1 = install_base_with_shadow_capacity(migrated_w1.operand, w1)
        w2 = install_base_with_shadow_capacity(migrated_w2.operand, w2)
        torch.cuda.synchronize()
        dist.barrier()
        migration_local = torch.tensor(
            (time.perf_counter() - migration_start) * 1e3,
            dtype=torch.float64,
            device=device,
        )
        rank_max(migration_local)
        weight_migration_ms = float(migration_local)
        weight_migration_bytes = (
            migrated_w1.transferred_bytes + migrated_w2.transferred_bytes
        )
        weight_migration_sample_bad_bytes = (
            migrated_w1.sample_bad_bytes + migrated_w2.sample_bad_bytes
        )
        weight_migration_sampled_bytes = (
            migrated_w1.sampled_bytes + migrated_w2.sampled_bytes
        )
        if weight_migration_sample_bad_bytes:
            raise AssertionError("migrated Sonic expert sample-byte gate failed")

    experimental_replica_preload_ms = 0.0
    experimental_replica_copy_bytes = 0
    if experimental_replica_plan is not None and experimental_slot_map is not None:
        # Process-group creation is process-lifetime setup, not measured preload.
        pair_groups = {
            (left, right): dist.new_group(ranks=[left, right])
            for left in range(world)
            for right in range(left + 1, world)
        }
        torch.cuda.synchronize()
        dist.barrier()
        preload_start = time.perf_counter()
        experimental_replica_copy_bytes += copy_replica_bank(
            w1,
            placement,
            experimental_replica_plan.replicas,
            experimental_slot_map,
            rank,
            pair_groups,
        )
        experimental_replica_copy_bytes += copy_replica_bank(
            w2,
            placement,
            experimental_replica_plan.replicas,
            experimental_slot_map,
            rank,
            pair_groups,
        )
        torch.cuda.synchronize()
        dist.barrier()
        preload_local = torch.tensor(
            (time.perf_counter() - preload_start) * 1e3,
            dtype=torch.float64,
            device=device,
        )
        rank_max(preload_local)
        experimental_replica_preload_ms = float(preload_local)
        copied = torch.tensor(
            experimental_replica_copy_bytes, dtype=torch.int64, device=device
        )
        dist.all_reduce(copied, op=dist.ReduceOp.SUM)
        experimental_replica_copy_bytes = int(copied)
    config = Mxfp8MoEKernelConfig(
        fc1_tile_m=args.fc1_tile_m,
        fc1_tile_n=args.fc1_tile_n,
        fc2_tile_m=args.fc2_tile_m,
        fc2_tile_n=args.fc2_tile_n,
        pingpong=not args.disable_pingpong,
    )
    expert_order = None
    if args.heavy_expert_first:
        expert_order = torch.argsort(
            torch.bincount(pair_expert_id, minlength=physical_experts),
            descending=True,
            stable=True,
        ).to(torch.int32)
    # The dispatch plan is static for a benchmark process, so its exact active
    # extents are setup metadata rather than a device-to-host synchronization in
    # the timed path.  Capacity buffers intentionally match the selected plan;
    # alternate placement or replica correctness oracles use the baseline path.
    active_recv_tokens = int(identity_dispatch.x.shape[0])
    active_total_pairs = int(pair_x_id.shape[0])
    needs_route_workspace = args.data_path in (
        "route_pack",
        "fused",
        "fused_atomic",
        "fused_segmented",
        "fused_hybrid",
        "fc2_epilogue_atomic",
    )
    # The C4 reference is produced before these potentially multi-GiB
    # workspaces are allocated.  That keeps setup-only reference temporaries
    # from overlapping the steady-state capacity reservation on long traces.
    mlp_workspace = None
    route_workspace = None
    reduce_workspace = None

    def forward(
        with_events: bool = False,
        transport_override: str | None = None,
        plan_override=None,
        experts_override: int | None = None,
        weights_override: tuple[BlockScaledOperand, BlockScaledOperand] | None = None,
        data_path_override: str | None = None,
    ):
        transport_mode = transport_override or args.activation_transport
        active_plan = plan_override or dispatch_plan
        active_experts = experts_override or physical_experts
        active_w1, active_w2 = weights_override or (w1, w2)
        data_path = data_path_override or args.data_path
        optimized_eligible = (
            active_plan is dispatch_plan
            and active_experts == physical_experts
            and active_w1 is w1
            and active_w2 is w2
        )
        if data_path not in ("baseline", "reference_chunked") and not optimized_eligible:
            data_path = "baseline"
        use_workspace = data_path not in ("baseline", "reference_chunked")
        use_route_pack = data_path in (
            "route_pack",
            "fused",
            "fused_atomic",
            "fused_segmented",
            "fused_hybrid",
            "fc2_epilogue_atomic",
        ) and transport_mode == "mxfp8"
        use_fused_reduce = data_path in (
            "fused",
            "fused_atomic",
            "fused_segmented",
            "fused_hybrid",
        )
        use_direct_fc2_reduce = data_path == "fc2_epilogue_atomic"
        active_expert_order = (
            expert_order if active_experts == physical_experts else None
        )
        events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        total_start.record()

        def stage(fn, name: str):
            if args.nvtx:
                torch.cuda.nvtx.range_push(name)
            if not with_events:
                try:
                    return fn()
                finally:
                    if args.nvtx:
                        torch.cuda.nvtx.range_pop()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                value = fn()
                end.record()
                events.append((start, end))
                return value
            finally:
                if args.nvtx:
                    torch.cuda.nvtx.range_pop()

        def dispatch_stage():
            if transport_mode == "bf16":
                return dispatch_deduplicated(x, active_plan)
            source_q, source_sf = (
                prequantized_source
                if prequantized_source is not None
                else quantize_mxfp8_rows(x)
            )
            # NCCL treats scale factors as payload bytes.  Moving the uint8
            # view avoids relying on collective support for the new E8M0 dtype.
            return dispatch_deduplicated(
                source_q, active_plan, scales=source_sf.view(torch.uint8)
            )

        dispatched = stage(dispatch_stage, "dispatch_quant_a2a")
        if use_route_pack:
            assert route_workspace is not None
            if dispatched.scales is None:
                raise RuntimeError("MXFP8 transport did not deliver source scales")
            packed = stage(
                lambda: route_pack_mxfp8(
                    dispatched.x,
                    dispatched.scales,
                    dispatched.expert_ids,
                    dispatched.weights,
                    active_total_pairs,
                    route_workspace,
                ),
                "compact_sort",
            )
            pair_x = packed.operand.qdata
            recv_token = packed.recv_token
            pair_weights = packed.weights
            indptr = packed.indptr
        else:
            pair_x, _, recv_token, pair_weights, indptr = stage(
                lambda: expand_and_sort(dispatched, active_experts),
                "compact_sort",
            )
        if pair_x.shape[0] == 0:
            raise RuntimeError("empty destination ranks are not supported by this benchmark")

        def pack_fc1():
            if use_route_pack:
                return packed.operand
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

        fc1_input = stage(pack_fc1, "pack_fc1_sfa")
        if use_workspace:
            assert mlp_workspace is not None
            postact_out = mlp_workspace.active_postact(pair_x.shape[0])
            postact = stage(
                lambda: mxfp8_swiglu_grouped_out(
                    fc1_input,
                    active_w1,
                    indptr,
                    postact_out,
                    config=config,
                    expert_order=active_expert_order,
                ),
                "fc1_swiglu_quant",
            )
            if use_direct_fc2_reduce:
                assert reduce_workspace is not None
                direct_reduced = stage(
                    lambda: mxfp8_down_grouped_weighted_reduce(
                        postact,
                        active_w2,
                        indptr,
                        recv_token,
                        pair_weights,
                        reduce_workspace.active(dispatched.x.shape[0]),
                        config=config,
                        expert_order=active_expert_order,
                    ),
                    "fc2",
                )
                pair_out = None
            else:
                pair_out_buffer = mlp_workspace.active_fc2_output(pair_x.shape[0])
                pair_out = stage(
                    lambda: mxfp8_down_grouped(
                        postact,
                        active_w2,
                        indptr,
                        config=config,
                        out=pair_out_buffer,
                        expert_order=active_expert_order,
                    ),
                    "fc2",
                )
        else:
            postact = stage(
                lambda: mxfp8_swiglu_grouped(
                    fc1_input,
                    active_w1,
                    indptr,
                    config=config,
                    expert_order=active_expert_order,
                ),
                "fc1_swiglu_quant",
            )
            pair_out = stage(
                lambda: mxfp8_down_grouped(
                    postact,
                    active_w2,
                    indptr,
                    config=config,
                    expert_order=active_expert_order,
                ),
                "fc2",
            )

        def reduce_stage():
            if use_direct_fc2_reduce:
                assert reduce_workspace is not None
                output = reduce_workspace.output[: dispatched.x.shape[0]]
                output.copy_(direct_reduced)
                return output
            if use_fused_reduce:
                assert reduce_workspace is not None
                if data_path == "fused_segmented" and use_route_pack:
                    assert route_workspace is not None
                    return segmented_weighted_reduce_mxfp8(
                        pair_out,
                        route_workspace.scatter_pos,
                        pair_weights,
                        dispatched.x.shape[0],
                        args.top_k,
                        reduce_workspace,
                    )
                if data_path == "fused_hybrid" and use_route_pack:
                    assert route_workspace is not None
                    return hybrid_weighted_reduce_mxfp8(
                        pair_out,
                        recv_token,
                        route_workspace.scatter_pos,
                        pair_weights,
                        dispatched.x.shape[0],
                        args.top_k,
                        reduce_workspace,
                    )[0]
                return weighted_reduce_mxfp8(
                    pair_out,
                    recv_token,
                    pair_weights,
                    dispatched.x.shape[0],
                    reduce_workspace,
                )
            reduced = (
                reduce_workspace.active(dispatched.x.shape[0]).zero_()
                if use_workspace
                else torch.zeros(
                    (dispatched.x.shape[0], args.hidden),
                    dtype=torch.float32,
                    device=device,
                )
            )
            if data_path == "reference_chunked":
                for begin in range(0, pair_out.shape[0], args.reference_chunk_pairs):
                    end = min(begin + args.reference_chunk_pairs, pair_out.shape[0])
                    reduced.index_add_(
                        0,
                        recv_token[begin:end],
                        pair_out[begin:end].float() * pair_weights[begin:end, None],
                    )
            else:
                reduced.index_add_(
                    0, recv_token, pair_out.float() * pair_weights[:, None]
                )
            if not use_workspace:
                return reduced.to(torch.bfloat16)
            assert reduce_workspace is not None
            output = reduce_workspace.output[: dispatched.x.shape[0]]
            output.copy_(reduced)
            return output

        reduced = stage(reduce_stage, "local_reduce")
        out = stage(
            lambda: combine_deduplicated(reduced, active_plan, args.tokens),
            "combine",
        )
        total_end.record()
        return out, events, (total_start, total_end), int(pair_x.shape[0])

    # Quantizing before transport and quantizing the same BF16 rows after
    # grouping should be numerically equivalent.  Keep this gate outside the
    # timed sample set.
    bf16_transport_out = forward(
        False, "bf16", data_path_override="reference_chunked"
    )[0]
    mxfp8_transport_out = forward(
        False, "mxfp8", data_path_override="reference_chunked"
    )[0]
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

    # C4 uses the same old FC1/FC2/FP32 index_add math, but caps only the
    # setup-time multiply temporary.  The timed ``baseline`` path remains one
    # unchunked index_add call for a fair before/after comparison.
    baseline_path_out = forward(
        False, "mxfp8", data_path_override="reference_chunked"
    )[0]
    mlp_workspace = (
        allocate_mxfp8_moe_workspace(
            physical_experts,
            active_total_pairs,
            args.hidden,
            args.intermediate,
            device=device,
            include_fc2_output=args.data_path != "fc2_epilogue_atomic",
        )
        if args.data_path != "baseline"
        else None
    )
    route_workspace = (
        allocate_mxfp8_route_pack_workspace(
            active_recv_tokens,
            active_total_pairs,
            args.top_k,
            physical_experts,
            args.hidden,
            device=device,
        )
        if needs_route_workspace
        else None
    )
    reduce_workspace = (
        allocate_mxfp8_weighted_reduce_workspace(
            active_recv_tokens,
            args.hidden,
            device=device,
            include_fp32=args.data_path not in ("fused_segmented",),
        )
        if args.data_path != "baseline"
        else None
    )
    selected_path_out = forward(False, "mxfp8", data_path_override=args.data_path)[0]
    torch.cuda.synchronize()
    path_difference = selected_path_out.float() - baseline_path_out.float()
    path_bad = path_difference.abs() > 0.05 + 0.05 * baseline_path_out.float().abs()
    data_path_numeric = torch.tensor(
        [
            float(path_bad.sum()),
            float(path_difference.abs().max()),
            float(path_difference.abs().mean()),
            float(
                torch.linalg.vector_norm(path_difference)
                / torch.linalg.vector_norm(baseline_path_out.float()).clamp_min(1e-20)
            ),
        ],
        dtype=torch.float64,
        device=device,
    )
    rank_max(data_path_numeric)
    if int(data_path_numeric[0]) or float(data_path_numeric[3]) > 5e-3:
        raise AssertionError(
            f"0902 data-path numerical gate failed: {data_path_numeric.tolist()}"
        )

    atomic_variance_numeric = torch.zeros(3, dtype=torch.float64, device=device)
    atomic_paths = {"fused", "fused_atomic", "fc2_epilogue_atomic"}
    if args.data_path in atomic_paths and args.atomic_variance_runs > 1:
        variance_reference = selected_path_out.clone()
        max_abs = torch.zeros((), dtype=torch.float64, device=device)
        max_rel_l2 = torch.zeros((), dtype=torch.float64, device=device)
        nonzero = torch.zeros((), dtype=torch.float64, device=device)
        reference_norm = torch.linalg.vector_norm(
            variance_reference.float()
        ).clamp_min(1e-20)
        for _ in range(args.atomic_variance_runs - 1):
            repeated = forward(
                False, "mxfp8", data_path_override=args.data_path
            )[0]
            difference = repeated.float() - variance_reference.float()
            max_abs = torch.maximum(max_abs, difference.abs().max().double())
            max_rel_l2 = torch.maximum(
                max_rel_l2,
                (torch.linalg.vector_norm(difference) / reference_norm).double(),
            )
            nonzero += (difference != 0).sum().double()
        atomic_variance_numeric[:] = torch.stack(
            (max_abs, max_rel_l2, nonzero)
        )
        rank_max(atomic_variance_numeric)

    placement_numeric = torch.zeros(4, dtype=torch.float64, device=device)
    if initial_weight_bank is not None:
        baseline = forward(
            False,
            "mxfp8",
            initial_dispatch_plan,
            local_experts,
            initial_weight_bank,
            data_path_override="reference_chunked",
        )[0]
        candidate = forward(
            False,
            "mxfp8",
            placement_dispatch_plan,
            physical_experts,
            data_path_override="reference_chunked",
        )[0]
        torch.cuda.synchronize()
        difference = candidate.float() - baseline.float()
        bad = difference.abs() > 0.05 + 0.05 * baseline.float().abs()
        placement_numeric[:] = torch.tensor(
            [
                float(bad.sum()),
                float(difference.abs().max()),
                float(difference.abs().mean()),
                float(
                    torch.linalg.vector_norm(difference)
                    / torch.linalg.vector_norm(baseline.float()).clamp_min(1e-20)
                ),
            ],
            dtype=torch.float64,
            device=device,
        )
        rank_max(placement_numeric)
        if int(placement_numeric[0]):
            raise AssertionError(
                f"real placement migration numerical gate failed: {placement_numeric.tolist()}"
            )
        initial_weight_bank = None

    experimental_replica_numeric = torch.zeros(4, dtype=torch.float64, device=device)
    if experimental_replica_plan is not None:
        baseline = forward(
            False,
            "mxfp8",
            placement_dispatch_plan,
            physical_experts,
            data_path_override="reference_chunked",
        )[0]
        candidate = forward(
            False,
            "mxfp8",
            dispatch_plan,
            physical_experts,
            data_path_override="reference_chunked",
        )[0]
        torch.cuda.synchronize()
        difference = candidate.float() - baseline.float()
        bad = difference.abs() > 0.05 + 0.05 * baseline.float().abs()
        experimental_replica_numeric[:] = torch.tensor(
            [
                float(bad.sum()),
                float(difference.abs().max()),
                float(difference.abs().mean()),
                float(
                    torch.linalg.vector_norm(difference)
                    / torch.linalg.vector_norm(baseline.float()).clamp_min(1e-20)
                ),
            ],
            dtype=torch.float64,
            device=device,
        )
        rank_max(experimental_replica_numeric)
        if int(experimental_replica_numeric[0]):
            raise AssertionError(
                "experimental replica numerical gate failed: "
                f"{experimental_replica_numeric.tolist()}"
            )

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
    if args.cuda_profiler_capture:
        torch.cuda.profiler.start()
    host_start = time.perf_counter()
    for _ in range(args.iters):
        _, stage_events, total_events, local_pairs = forward(not args.stability)
        if not args.stability:
            all_stage_events.append(stage_events)
        all_total_events.append(total_events)
    torch.cuda.synchronize()
    if args.cuda_profiler_capture:
        torch.cuda.profiler.stop()
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
            "local_physical_experts": physical_experts,
            "hidden": args.hidden,
            "intermediate_after_swiglu": args.intermediate,
            "fc1_physical_output": 2 * args.intermediate,
            "routing": route.scenario if args.routing_trace else args.routing,
            "zipf_alpha": args.zipf_alpha if args.routing == "zipf" else None,
            "hot_experts": args.hot_experts if args.routing == "hotset" else None,
            "hot_mass": args.hot_mass if args.routing == "hotset" else None,
            # Store only the basename so published JSONL cannot disclose a
            # workstation username or private directory layout.
            "routing_trace_file": Path(args.routing_trace).name
            if args.routing_trace
            else None,
            "routing_trace_path": Path(args.routing_trace).name
            if args.routing_trace
            else None,
            "routing_trace_sha256": trace_sha256,
            "placement": args.placement,
            "eplb_policy": "static_equal_capacity_lpt_current_histogram"
            if args.placement == "greedy"
            else None,
            "placement_plan_ms": placement_plan_ms,
            "placement_moved_experts": placement_moved_experts,
            "migration_limit": args.migration_limit,
            "initial_rank_pair_loads": initial_values,
            "placement_rank_pair_loads": placement_values,
            "initial_rank_max_over_mean": max(initial_values)
            / (sum(initial_values) / world),
            "placement_rank_max_over_mean": max(placement_values)
            / (sum(placement_values) / world),
            "expert_weight_scale_bytes": bytes_per_expert,
            "placement_logical_copy_bytes": placement_moved_experts * bytes_per_expert,
            "placement_migration_in_timed_forward": False,
            "real_weight_migration_enabled": args.real_weight_migration,
            "weight_migration_ms": weight_migration_ms,
            "weight_migration_transferred_bytes": weight_migration_bytes,
            "weight_migration_sample_bad_bytes": weight_migration_sample_bad_bytes,
            "weight_migration_sampled_bytes": weight_migration_sampled_bytes,
            "placement_migration_correctness": {
                "bad_count": int(placement_numeric[0]),
                "max_abs": float(placement_numeric[1]),
                "mean_abs": float(placement_numeric[2]),
                "relative_l2": float(placement_numeric[3]),
            }
            if placement_moved_experts and args.real_weight_migration
            else None,
            "experimental_replica": {
                "status": (
                    "experimental_opt_in_active"
                    if experimental_replica_plan is not None
                    else (
                        "experimental_requested_no_plan"
                        if args.experimental_replica_slots
                        else "experimental_default_off"
                    )
                ),
                "requested_slots_per_rank": args.experimental_replica_slots,
                "active_slots_per_rank": args.experimental_replica_slots
                if experimental_replica_plan is not None
                else 0,
                "max_copies_per_expert": args.experimental_max_copies_per_expert,
                "minimum_hot_expert_ratio": args.experimental_minimum_hot_expert_ratio,
                "replica_map": experimental_replica_plan.replicas
                if experimental_replica_plan is not None
                else {},
                "predicted_rank_pair_loads": experimental_replica_loads.tolist()
                if experimental_replica_loads is not None
                else None,
                "preload_ms": experimental_replica_preload_ms,
                "direct_copy_bytes_collective_sum": experimental_replica_copy_bytes,
                "correctness": {
                    "bad_count": int(experimental_replica_numeric[0]),
                    "max_abs": float(experimental_replica_numeric[1]),
                    "mean_abs": float(experimental_replica_numeric[2]),
                    "relative_l2": float(experimental_replica_numeric[3]),
                }
                if experimental_replica_plan is not None
                else None,
            },
            "seed": args.seed,
            "warmup": args.warmup,
            "iterations": args.iters,
            "activation_transport": args.activation_transport,
            "timing_scope": "E0_prequantized" if args.prequantized_source else "E1_source_quantized",
            "source_quantization_in_timed_forward": not args.prequantized_source,
            "data_path": args.data_path,
            "heavy_expert_first": args.heavy_expert_first,
            "weighted_reduce_policy": (
                "segmented"
                if args.data_path == "fused_segmented"
                or (
                    args.data_path == "fused_hybrid"
                    and active_total_pairs / max(1, active_recv_tokens) >= 2.0
                )
                else "atomic"
                if args.data_path
                in (
                    "fused",
                    "fused_atomic",
                    "fused_hybrid",
                    "fc2_epilogue_atomic",
                )
                else "baseline_index_add"
            ),
            "pair_output_materialized": args.data_path != "fc2_epilogue_atomic",
            "data_path_correctness": {
                "bad_count": int(data_path_numeric[0]),
                "max_abs": float(data_path_numeric[1]),
                "mean_abs": float(data_path_numeric[2]),
                "relative_l2": float(data_path_numeric[3]),
            },
            "atomic_repeatability": {
                "runs": args.atomic_variance_runs,
                "max_abs": float(atomic_variance_numeric[0]),
                "max_relative_l2": float(atomic_variance_numeric[1]),
                "nonzero_differences": int(atomic_variance_numeric[2]),
            }
            if args.data_path in atomic_paths
            else None,
            "workspace_bytes": {
                "local_mlp": mlp_workspace.nbytes if mlp_workspace is not None else 0,
                "route_pack": route_workspace.nbytes if route_workspace is not None else 0,
                "weighted_reduce": (
                    reduce_workspace.nbytes if reduce_workspace is not None else 0
                ),
                "total": sum(
                    workspace.nbytes
                    for workspace in (mlp_workspace, route_workspace, reduce_workspace)
                    if workspace is not None
                ),
            },
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
                "Steady-state forward excludes static EPLB planning and weight movement. "
                "When real_weight_migration_enabled is true, this invocation performed "
                "and audited that movement before timing; otherwise placement is the "
                "historical logical steady-state model only. Replica/hybrid is an "
                "experimental, explicit opt-in path and is disabled by the stable suite."
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
