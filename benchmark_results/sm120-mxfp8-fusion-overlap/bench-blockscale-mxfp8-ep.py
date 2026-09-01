#!/usr/bin/env python3
"""Common deduplicated EP4 benchmark for sm120_block_scale_gemm MXFP8."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

import torch
import torch.distributed as dist

from ep4_support.common import summarize_ms
from ep4_support.routing import (
    contiguous_placement,
    greedy_placement,
    load_global_routing_trace,
)
from ep4_support.transport import (
    actual_dispatch_bytes,
    combine_deduplicated,
    dispatch_deduplicated,
    expand_and_sort,
    make_dispatch_plan,
)
from moe0902.blockscale_backend import pack_linear_e8m0_for_moe, quantize_linear_mxfp8_rows


STAGES = (
    "source_quant_dispatch",
    "compact_sort",
    "pack_fc1_sfa",
    "fc1_swiglu",
    "requant_fc2",
    "fc2",
    "local_reduce",
    "combine",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--experts", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1280)
    parser.add_argument("--routing-trace", required=True)
    parser.add_argument("--placement", choices=["contiguous", "greedy"], default="contiguous")
    parser.add_argument("--activation-transport", choices=["mxfp8", "bf16"], default="mxfp8")
    parser.add_argument(
        "--source-mode", choices=["prequantized", "bf16_source"], default="prequantized",
        help="E0 excludes source quantization; E1 includes it in every timed forward",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--stability", action="store_true")
    parser.add_argument(
        "--diagnostic-uniform-experts", action="store_true",
        help="make every logical expert byte-identical to isolate migration layout from ID mapping",
    )
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--node-label", choices=["PRO5000-A", "PRO5000-B"], required=True)
    parser.add_argument(
        "--topology-label", choices=["same_numa", "cross_numa_2plus2"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_label):
        parser.error("run-label must be privacy-safe")
    return args


def rank_max(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor


def make_weights(experts: int, n: int, k: int, seed: int, device: torch.device):
    from utils import per_block_cast_to_fp8_e8m0, transform_sf_into_required_layout

    generator = torch.Generator(device=device).manual_seed(seed)
    bf16 = torch.randn((experts, n, k), dtype=torch.bfloat16, device=device, generator=generator)
    values, semantic_scales = per_block_cast_to_fp8_e8m0(bf16, gran_k=32)
    physical_scales = transform_sf_into_required_layout(
        semantic_scales, mn=n, k=k, recipe=(32, 32), num_groups=experts
    )
    del bf16, semantic_scales
    return values, physical_scales


def migrate_tensor_bank(
    tensor: torch.Tensor,
    current: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, int, int, int]:
    """Move changed expert rows and install a target-ordered shadow tensor."""
    rank, world = dist.get_rank(), dist.get_world_size()
    current, target = current.cpu(), target.cpu()
    current_ids = torch.where(current == rank)[0].tolist()
    target_ids = torch.where(target == rank)[0].tolist()
    old_slot = {expert: slot for slot, expert in enumerate(current_ids)}
    send_by_rank = [
        sorted(expert for expert in range(current.numel())
               if int(current[expert]) == rank and int(target[expert]) == destination
               and int(current[expert]) != int(target[expert]))
        for destination in range(world)
    ]
    recv_by_rank = [
        sorted(expert for expert in range(current.numel())
               if int(current[expert]) == source and int(target[expert]) == rank
               and int(current[expert]) != int(target[expert]))
        for source in range(world)
    ]
    send_ids = [expert for values in send_by_rank for expert in values]
    recv_ids = [expert for values in recv_by_rank for expert in values]
    send_slots = [old_slot[expert] for expert in send_ids]
    index = torch.tensor(send_slots, dtype=torch.int64, device=tensor.device)
    send = tensor.index_select(0, index).contiguous()
    recv = torch.empty((len(recv_ids), *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
    send_counts = [len(values) for values in send_by_rank]
    recv_counts = [len(values) for values in recv_by_rank]
    dist.all_to_all_single(
        recv.view(torch.uint8), send.view(torch.uint8),
        output_split_sizes=recv_counts, input_split_sizes=send_counts,
    )
    bytes_per_row = tensor[0].numel() * tensor.element_size()
    send_bytes = send.view(torch.uint8).reshape(len(send_ids), bytes_per_row)
    positions = torch.tensor(
        sorted({0, 1, min(31, max(0, send_bytes.shape[1] - 1)),
                send_bytes.shape[1] // 2, max(0, send_bytes.shape[1] - 2),
                max(0, send_bytes.shape[1] - 1)}),
        dtype=torch.int64, device=tensor.device,
    )
    expected = send_bytes.index_select(1, positions) if send_ids else torch.empty(
        (0, positions.numel()), dtype=torch.uint8, device=tensor.device
    )
    received_expected = torch.empty(
        (len(recv_ids), positions.numel()), dtype=torch.uint8, device=tensor.device
    )
    dist.all_to_all_single(
        received_expected, expected,
        output_split_sizes=recv_counts, input_split_sizes=send_counts,
    )
    incoming = {expert: slot for slot, expert in enumerate(recv_ids)}
    # Block-scale E8M0 tensors use a padded MN-major TMA stride.  For a
    # non-dense strided tensor, empty_like(preserve_format) is allowed to
    # choose a contiguous fallback; logical byte samples then pass while the
    # custom kernel reads scale bytes at the wrong physical offsets.  Preserve
    # the exact storage contract explicitly for both values and scales.
    shadow = torch.empty_strided(
        tensor.size(), tensor.stride(), dtype=tensor.dtype, device=tensor.device
    )
    if shadow.stride() != tensor.stride():
        raise AssertionError("migrated block-scale tensor lost its physical stride")
    bad = torch.zeros((), dtype=torch.int64, device=tensor.device)
    sampled = 0
    recv_bytes = recv.view(torch.uint8).reshape(len(recv_ids), bytes_per_row)
    for slot, expert in enumerate(target_ids):
        if expert in old_slot:
            shadow[slot].copy_(tensor[old_slot[expert]])
        else:
            source = incoming[expert]
            shadow[slot].copy_(recv[source])
            bad.add_(torch.count_nonzero(recv_bytes[source].index_select(0, positions) != received_expected[source]))
            sampled += positions.numel()
    totals = torch.tensor(
        [sum(len(values) for values in send_by_rank), int(bad), sampled],
        dtype=torch.int64, device=tensor.device,
    )
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    transferred = int(totals[0]) * tensor[0].numel() * tensor.element_size()
    return shadow, transferred, int(totals[1]), int(totals[2])


def main() -> int:
    args = parse_args()
    library = os.environ.get("BLOCKSCALE_LIB")
    if not library:
        raise RuntimeError("BLOCKSCALE_LIB must point to libth_op.so")
    torch.classes.load_library(library)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world != 4 or torch.cuda.get_device_capability(device)[0] != 12:
        raise RuntimeError("the campaign requires four physical SM120 GPUs")
    if args.experts % world or args.hidden % 128 or args.intermediate % 128:
        raise ValueError("invalid customer dimensions")
    local_experts = args.experts // world
    route, trace_sha = load_global_routing_trace(
        args.routing_trace,
        source_tokens_per_rank=args.tokens,
        top_k=args.top_k,
        experts=args.experts,
        ep_size=world,
        rank=rank,
    )
    local_hist = torch.bincount(route.expert_ids.reshape(-1), minlength=args.experts).to(device)
    dist.all_reduce(local_hist, op=dist.ReduceOp.SUM)
    initial_placement = contiguous_placement(args.experts, world)
    placement = initial_placement
    plan_ms = 0.0
    if args.placement == "greedy":
        torch.cuda.synchronize(); dist.barrier()
        start = time.perf_counter()
        placement = greedy_placement(local_hist.cpu(), world)
        dist.barrier()
        value = torch.tensor((time.perf_counter() - start) * 1e3, dtype=torch.float64, device=device)
        rank_max(value)
        plan_ms = float(value)
    dispatch_plan = make_dispatch_plan(route.expert_ids, route.weights, placement, world, device)
    baseline_dispatch_plan = make_dispatch_plan(
        route.expert_ids, route.weights, initial_placement, world, device
    )
    x = torch.randn((args.tokens, args.hidden), dtype=torch.bfloat16, device=device)
    source_values, source_scales = quantize_linear_mxfp8_rows(x)

    identity = dispatch_deduplicated(x, dispatch_plan)
    id_x, _, id_recv, id_weight, _ = expand_and_sort(identity, local_experts)
    id_reduced = torch.zeros_like(identity.x)
    id_reduced.index_add_(0, id_recv, id_x * id_weight[:, None].to(id_x.dtype))
    id_out = combine_deduplicated(id_reduced, dispatch_plan, args.tokens)
    identity_error = torch.tensor(
        [
            float((id_out - x).abs().max()),
            float(torch.linalg.vector_norm((id_out - x).float()) / torch.linalg.vector_norm(x.float())),
        ],
        dtype=torch.float64,
        device=device,
    )
    rank_max(identity_error)

    if rank == 0:
        print("allocating block-scale MXFP8 expert bank", flush=True)
    w1, s1 = make_weights(local_experts, 2 * args.intermediate, args.hidden, args.seed + 1000 + rank, device)
    w2, s2 = make_weights(local_experts, args.hidden, args.intermediate, args.seed + 2000 + rank, device)
    if args.diagnostic_uniform_experts:
        for tensor in (w1, s1, w2, s2):
            canonical = tensor[0].contiguous()
            dist.broadcast(canonical, src=0)
            for slot in range(local_experts):
                tensor[slot].copy_(canonical)
        torch.cuda.synchronize(); dist.barrier()
    initial_weight_bank = (w1, s1, w2, s2) if args.placement != "contiguous" else None
    weight_migration_ms = 0.0
    weight_migration_bytes = 0
    weight_migration_sample_bad = 0
    weight_migration_sampled_bytes = 0
    if args.placement != "contiguous":
        torch.cuda.synchronize(); dist.barrier(); start = time.perf_counter()
        migrated = [
            migrate_tensor_bank(tensor, initial_placement, placement)
            for tensor in (w1, s1, w2, s2)
        ]
        w1, s1, w2, s2 = (item[0] for item in migrated)
        if args.diagnostic_uniform_experts:
            names = ("fc1_value", "fc1_scale", "fc2_value", "fc2_scale")
            for name, original, candidate_tensor in zip(
                names, initial_weight_bank, (w1, s1, w2, s2)
            ):
                logical_bad = torch.count_nonzero(
                    candidate_tensor != original
                ).to(torch.int64)
                dist.all_reduce(logical_bad, op=dist.ReduceOp.SUM)
                if rank == 0:
                    print(
                        f"uniform migration audit {name}: bad={int(logical_bad)} "
                        f"original_stride={original.stride()} "
                        f"candidate_stride={candidate_tensor.stride()}",
                        flush=True,
                    )
        weight_migration_bytes = sum(item[1] for item in migrated)
        weight_migration_sample_bad = sum(item[2] for item in migrated)
        weight_migration_sampled_bytes = sum(item[3] for item in migrated)
        torch.cuda.synchronize(); dist.barrier()
        value = torch.tensor((time.perf_counter() - start) * 1e3, dtype=torch.float64, device=device)
        rank_max(value); weight_migration_ms = float(value)
        if weight_migration_sample_bad:
            raise AssertionError("migrated block-scale expert sample-byte gate failed")
    payload_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in (w1, s1, w2, s2)
    )
    torch.cuda.reset_peak_memory_stats(device)

    def forward(
        with_events: bool = False, mode: str | None = None,
        plan_to_use=None, weights=None,
    ):
        active_plan = plan_to_use if plan_to_use is not None else dispatch_plan
        active_w1, active_s1, active_w2, active_s2 = (
            weights if weights is not None else (w1, s1, w2, s2)
        )
        transport = mode or args.activation_transport
        events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        total_start.record()

        def stage(fn):
            if not with_events:
                return fn()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(); value = fn(); end.record(); events.append((begin, end)); return value

        def dispatch_stage():
            if transport == "bf16":
                return dispatch_deduplicated(x, active_plan)
            if args.source_mode == "prequantized":
                values, scales = source_values, source_scales
            else:
                values, scales = quantize_linear_mxfp8_rows(x)
            return dispatch_deduplicated(values, active_plan, scales=scales.view(torch.uint8))

        dispatched = stage(dispatch_stage)
        pair_x, _, recv_token, pair_weights, indptr = stage(
            lambda: expand_and_sort(dispatched, local_experts)
        )
        if pair_x.shape[0] == 0:
            raise RuntimeError("empty destination rank is unsupported")

        def pack_fc1():
            if transport == "bf16":
                return torch.ops.custom_ops.fp8_quant_and_transform_for_moe(pair_x, indptr, 32)
            if dispatched.scales is None:
                raise RuntimeError("MXFP8 scales were not transported")
            linear = dispatched.scales.index_select(0, recv_token).contiguous()
            return pair_x, pack_linear_e8m0_for_moe(linear, indptr)

        q1, a1 = stage(pack_fc1)
        postact = stage(
            lambda: torch.ops.custom_ops.fused_moe_mxfp8_nt_groupwise(
                q1, active_w1, a1, active_s1, indptr, 32
            )
        )
        q2, a2 = stage(
            lambda: torch.ops.custom_ops.fp8_quant_and_transform_for_moe(postact, indptr, 32)
        )
        pair_out = stage(
            lambda: torch.ops.custom_ops.moe_gemm_mxfp8_nt_groupwise(
                q2, active_w2, a2, active_s2, indptr, 32
            )
        )

        def reduce_stage():
            reduced = torch.zeros(
                (dispatched.x.shape[0], args.hidden), dtype=torch.float32, device=device
            )
            reduced.index_add_(0, recv_token, pair_out.float() * pair_weights[:, None])
            return reduced.to(torch.bfloat16)

        reduced = stage(reduce_stage)
        output = stage(lambda: combine_deduplicated(reduced, active_plan, args.tokens))
        total_end.record()
        return output, events, (total_start, total_end), int(pair_x.shape[0])

    bf16_output = forward(False, "bf16")[0]
    mxfp8_output = forward(False, "mxfp8")[0]
    torch.cuda.synchronize()
    difference = (mxfp8_output.float() - bf16_output.float()).reshape(-1)
    reference = bf16_output.float().reshape(-1)
    candidate = mxfp8_output.float().reshape(-1)
    bad = difference.abs() > 0.05 + 0.05 * reference.abs()
    numeric = torch.tensor(
        [
            float(bad.sum()),
            difference.abs().max(),
            difference.abs().mean(),
            torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference).clamp_min(1e-20),
            torch.nn.functional.cosine_similarity(candidate, reference, dim=0),
        ],
        dtype=torch.float64,
        device=device,
    )
    rank_max(numeric[:4]); dist.all_reduce(numeric[4:], op=dist.ReduceOp.MIN)
    if int(numeric[0]) or float(numeric[3]) > 5e-3:
        raise AssertionError(f"block-scale C4 numerical gate failed: {numeric.tolist()}")

    placement_numeric = torch.zeros(4, dtype=torch.float64, device=device)
    if initial_weight_bank is not None:
        baseline = forward(False, "mxfp8", baseline_dispatch_plan, initial_weight_bank)[0]
        candidate = forward(False, "mxfp8")[0]
        torch.cuda.synchronize()
        placement_difference = candidate.float() - baseline.float()
        placement_bad = placement_difference.abs() > 0.05 + 0.05 * baseline.float().abs()
        placement_numeric[:] = torch.tensor(
            [
                float(placement_bad.sum()), float(placement_difference.abs().max()),
                float(placement_difference.abs().mean()),
                float(torch.linalg.vector_norm(placement_difference) / torch.linalg.vector_norm(baseline.float()).clamp_min(1e-20)),
            ], dtype=torch.float64, device=device,
        )
        rank_max(placement_numeric)
        if int(placement_numeric[0]):
            raise AssertionError(f"placement C4 numerical gate failed: {placement_numeric.tolist()}")
        initial_weight_bank = None

    cold_start = time.perf_counter(); forward(False); torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - cold_start) * 1e3
    for _ in range(max(0, args.warmup - 1)):
        forward(False)
    torch.cuda.synchronize(); dist.barrier()
    stage_samples, total_samples = [], []
    pair_count = 0
    host_start = time.perf_counter()
    for _ in range(args.iters):
        _, stages, total, pair_count = forward(not args.stability)
        if not args.stability:
            stage_samples.append(stages)
        total_samples.append(total)
    torch.cuda.synchronize()
    host_mean = (time.perf_counter() - host_start) * 1e3 / args.iters
    totals = torch.tensor(
        [begin.elapsed_time(end) for begin, end in total_samples], dtype=torch.float64, device=device
    )
    rank_max(totals)
    stage_values = None
    if stage_samples:
        stage_values = torch.tensor(
            [[begin.elapsed_time(end) for begin, end in sample] for sample in stage_samples],
            dtype=torch.float64,
            device=device,
        )
        rank_max(stage_values)
    host_value = torch.tensor(host_mean, dtype=torch.float64, device=device); rank_max(host_value)
    pairs = torch.tensor(pair_count, dtype=torch.int64, device=device)
    gathered_pairs = [torch.empty_like(pairs) for _ in range(world)]
    dist.all_gather(gathered_pairs, pairs)
    remote_bytes = torch.tensor(
        actual_dispatch_bytes(
            dispatch_plan,
            args.hidden,
            x_bytes=1 if args.activation_transport == "mxfp8" else 2,
            scale_bytes_per_token=args.hidden // 32 if args.activation_transport == "mxfp8" else 0,
        ),
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(remote_bytes, op=dist.ReduceOp.SUM)
    peak = torch.tensor(torch.cuda.max_memory_allocated(device), dtype=torch.int64, device=device)
    rank_max(peak)
    peak_reserved = torch.tensor(
        torch.cuda.max_memory_reserved(device), dtype=torch.int64, device=device
    )
    rank_max(peak_reserved)

    if rank == 0:
        summary = summarize_ms(totals.cpu().tolist())
        pair_values = [int(value) for value in gathered_pairs]
        initial_loads = torch.bincount(
            initial_placement, weights=local_hist.cpu().to(torch.float64), minlength=world
        ).to(torch.int64)
        final_loads = torch.bincount(
            placement, weights=local_hist.cpu().to(torch.float64), minlength=world
        ).to(torch.int64)
        record = {
            "schema_version": 1,
            "benchmark": "blockscale_mxfp8_common_ep4",
            "backend": "blockscale_groupwise",
            "source_commit": os.environ.get("BLOCKSCALE_COMMIT", "unknown"),
            "timestamp_unix": time.time(),
            "node_label": args.node_label,
            "topology": args.topology_label,
            "run_label": args.run_label,
            "world_size": world,
            "source_tokens_per_rank": args.tokens,
            "global_tokens": args.tokens * world,
            "top_k": args.top_k,
            "global_experts": args.experts,
            "local_experts": local_experts,
            "hidden": args.hidden,
            "intermediate_after_swiglu": args.intermediate,
            "fc1_physical_output": 2 * args.intermediate,
            "granularity": {"activation": "1x32", "weight": "32x32", "scale": "E8M0"},
            "routing_trace_sha256": trace_sha,
            "routing_trace_file": Path(args.routing_trace).name,
            "placement": args.placement,
            "diagnostic_uniform_experts": args.diagnostic_uniform_experts,
            "placement_plan_ms": plan_ms,
            "placement_moved_experts": int((placement != initial_placement).sum()),
            "weight_migration_ms": weight_migration_ms,
            "weight_migration_bytes": weight_migration_bytes,
            "weight_migration_sample_bad_bytes": weight_migration_sample_bad,
            "weight_migration_sampled_bytes": weight_migration_sampled_bytes,
            "placement_correctness": {
                "bad_count": int(placement_numeric[0]),
                "max_abs": float(placement_numeric[1]),
                "mean_abs": float(placement_numeric[2]),
                "relative_l2": float(placement_numeric[3]),
            } if args.placement != "contiguous" else None,
            "initial_rank_pair_loads": initial_loads.tolist(),
            "placement_rank_pair_loads": final_loads.tolist(),
            "activation_transport": args.activation_transport,
            "source_mode": args.source_mode,
            "timing_level": "E0_prequantized" if args.source_mode == "prequantized" else "E1_bf16_source",
            "transport": f"torch_nccl_all_to_all_deduplicated_{args.activation_transport}",
            "warmup": args.warmup,
            "iterations": args.iters,
            "cold_or_cache_lookup_ms": cold_ms,
            "transport_identity_max_abs": float(identity_error[0]),
            "transport_identity_rel_l2": float(identity_error[1]),
            "data_path_correctness": {
                "bad_count": int(numeric[0]),
                "max_abs": float(numeric[1]),
                "mean_abs": float(numeric[2]),
                "relative_l2": float(numeric[3]),
            },
            "mxfp8_transport_vs_bf16_max_abs": float(numeric[1]),
            "mxfp8_transport_vs_bf16_rel_l2": float(numeric[3]),
            "mxfp8_transport_vs_bf16_cosine": float(numeric[4]),
            "local_pair_counts": pair_values,
            "rank_pair_max_over_mean": max(pair_values) / (sum(pair_values) / world),
            "global_remote_dispatch_bytes": int(remote_bytes),
            "expert_weight_scale_bytes_per_rank": payload_bytes,
            "stage_rank_max": {
                name: summarize_ms(stage_values.cpu()[:, index].tolist())
                for index, name in enumerate(STAGES)
            } if stage_values is not None else {},
            "e2e_rank_max": summary,
            "host_rank_max_mean_ms": float(host_value),
            "global_tokens_per_second": args.tokens * world / (summary["p50_ms"] / 1e3),
            "global_pairs_per_second": args.tokens * world * args.top_k / (summary["p50_ms"] / 1e3),
            "peak_allocated_bytes": int(peak),
            "peak_reserved_bytes": int(peak_reserved),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
    dist.barrier(); dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
