#!/usr/bin/env python3
"""Triple-buffered EP4 pipeline using the retained 0902 Sonic data path."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from ep4_support.common import summarize_ms
from ep4_support.routing import contiguous_placement, load_global_routing_trace
from ep4_support.transport import (
    combine_deduplicated,
    dispatch_deduplicated,
    exchange_counts,
    make_dispatch_plan,
)
from sonicmoe.functional.mxfp8 import (
    Mxfp8MoEKernelConfig,
    allocate_mxfp8_moe_workspace,
    allocate_mxfp8_weights,
    mxfp8_down_grouped,
    mxfp8_swiglu_grouped_out,
    quantize_mxfp8_rows,
)
from sonicmoe.functional.mxfp8_route_pack import (
    allocate_mxfp8_route_pack_workspace,
    route_pack_mxfp8,
)
from sonicmoe.functional.mxfp8_weighted_reduce import (
    allocate_mxfp8_weighted_reduce_workspace,
    segmented_weighted_reduce_mxfp8,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--chunk-tokens", type=int, required=True)
    parser.add_argument("--routing-trace", required=True)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--experts", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1280)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--node-label", choices=["PRO5000-A", "PRO5000-B"], required=True)
    parser.add_argument(
        "--topology-label", choices=["same_numa", "cross_numa_2plus2"], required=True
    )
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.chunk_tokens <= 0 or args.tokens % args.chunk_tokens:
        parser.error("chunk-tokens must divide tokens exactly")
    return args


def rank_max(value: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value


def main() -> int:
    args = arguments()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world != 4 or args.experts % world:
        raise RuntimeError("this benchmark requires EP4 with equal expert capacity")
    local_experts = args.experts // world
    route, trace_sha = load_global_routing_trace(
        args.routing_trace,
        source_tokens_per_rank=args.tokens,
        top_k=args.top_k,
        experts=args.experts,
        ep_size=world,
        rank=rank,
    )
    placement = contiguous_placement(args.experts, world)

    items = []
    for original_index, lo in enumerate(range(0, args.tokens, args.chunk_tokens)):
        hi = lo + args.chunk_tokens
        ids = route.expert_ids[lo:hi]
        plan = make_dispatch_plan(ids, route.weights[lo:hi], placement, world, device)
        send_pair_counts = []
        offset = 0
        for count in plan.send_counts:
            send_pair_counts.append(
                int((plan.send_expert_ids[offset : offset + count] >= 0).sum())
            )
            offset += count
        pair_count = sum(exchange_counts(send_pair_counts, device))
        global_pair_count = torch.tensor(pair_count, dtype=torch.int64, device=device)
        dist.all_reduce(global_pair_count, op=dist.ReduceOp.SUM)
        expected_pairs = world * args.chunk_tokens * args.top_k
        if int(global_pair_count) != expected_pairs:
            raise AssertionError(
                f"chunk route conservation failed: {int(global_pair_count)} != {expected_pairs}"
            )
        local_histogram = torch.bincount(ids.reshape(-1), minlength=args.experts)
        # A concentration-sensitive setup-time score.  The MAX collective
        # gives every rank the same chunk order, which is mandatory for NCCL.
        score = torch.tensor(
            float(local_histogram.max()) + float((local_histogram.float() ** 2).sum()) / ids.numel(),
            dtype=torch.float64,
            device=device,
        )
        rank_max(score)
        items.append(
            {
                "original_index": original_index,
                "lo": lo,
                "hi": hi,
                "plan": plan,
                "recv_tokens": plan.total_recv_tokens,
                "pairs": pair_count,
                "score": float(score),
            }
        )
    items.sort(key=lambda item: (-item["score"], item["original_index"]))

    x = torch.randn((args.tokens, args.hidden), dtype=torch.bfloat16, device=device)
    source_q, source_sf = quantize_mxfp8_rows(x)
    del x
    w1 = allocate_mxfp8_weights(
        local_experts, 2 * args.intermediate, args.hidden, device=device, seed=1902 + rank
    )
    w2 = allocate_mxfp8_weights(
        local_experts, args.hidden, args.intermediate, device=device, seed=2902 + rank
    )
    max_recv = max(item["recv_tokens"] for item in items)
    max_pairs = max(item["pairs"] for item in items)
    route_workspace = allocate_mxfp8_route_pack_workspace(
        max_recv, max_pairs, args.top_k, local_experts, args.hidden, device=device
    )
    mlp_workspace = allocate_mxfp8_moe_workspace(
        local_experts,
        max_pairs,
        args.hidden,
        args.intermediate,
        device=device,
    )
    reduce_workspaces = [
        allocate_mxfp8_weighted_reduce_workspace(
            max_recv, args.hidden, device=device, include_fp32=False
        )
        for _ in range(3)
    ]
    config = Mxfp8MoEKernelConfig()
    comm_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)

    def dispatch_item(item):
        return dispatch_deduplicated(
            source_q[item["lo"] : item["hi"]],
            item["plan"],
            scales=source_sf[item["lo"] : item["hi"]].view(torch.uint8),
        )

    def compute_item(dispatched, item, slot: int):
        if dispatched.scales is None:
            raise RuntimeError("MXFP8 scales were not dispatched")
        for tensor in (
            dispatched.x,
            dispatched.scales,
            dispatched.expert_ids,
            dispatched.weights,
        ):
            tensor.record_stream(compute_stream)
        packed = route_pack_mxfp8(
            dispatched.x,
            dispatched.scales,
            dispatched.expert_ids,
            dispatched.weights,
            item["pairs"],
            route_workspace,
        )
        postact = mxfp8_swiglu_grouped_out(
            packed.operand,
            w1,
            packed.indptr,
            mlp_workspace.active_postact(item["pairs"]),
            config=config,
        )
        pair_output = mxfp8_down_grouped(
            postact,
            w2,
            packed.indptr,
            config=config,
            out=mlp_workspace.active_fc2_output(item["pairs"]),
        )
        return segmented_weighted_reduce_mxfp8(
            pair_output,
            route_workspace.scatter_pos,
            packed.weights,
            item["recv_tokens"],
            args.top_k,
            reduce_workspaces[slot],
        )

    def sequential():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        outputs = {}
        for index, item in enumerate(items):
            dispatched = dispatch_item(item)
            reduced = compute_item(dispatched, item, index % 3)
            outputs[item["original_index"]] = combine_deduplicated(
                reduced, item["plan"], args.chunk_tokens
            )
        result = torch.cat([outputs[index] for index in range(len(items))])
        end.record()
        return result, (start, end)

    def pipelined():
        origin = torch.cuda.current_stream(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(origin)
        comm_stream.wait_event(start)
        compute_stream.wait_event(start)
        free_events = [torch.cuda.Event() for _ in range(3)]
        for event in free_events:
            event.record(origin)

        def enqueue_dispatch(item):
            with torch.cuda.stream(comm_stream):
                dispatched = dispatch_item(item)
                ready = torch.cuda.Event()
                ready.record()
            return dispatched, ready

        dispatched, dispatch_ready = enqueue_dispatch(items[0])
        outputs = {}
        for index, item in enumerate(items):
            slot = index % 3
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(dispatch_ready)
                compute_stream.wait_event(free_events[slot])
                reduced = compute_item(dispatched, item, slot)
                compute_ready = torch.cuda.Event()
                compute_ready.record()
            next_item = (
                enqueue_dispatch(items[index + 1]) if index + 1 < len(items) else None
            )
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(compute_ready)
                outputs[item["original_index"]] = combine_deduplicated(
                    reduced, item["plan"], args.chunk_tokens
                )
                free_events[slot].record()
            if next_item is not None:
                dispatched, dispatch_ready = next_item
        origin.wait_stream(comm_stream)
        origin.wait_stream(compute_stream)
        result = torch.cat([outputs[index] for index in range(len(items))])
        end.record(origin)
        return result, (start, end)

    sequential_output = sequential()[0]
    pipeline_output = pipelined()[0]
    torch.cuda.synchronize()
    difference = pipeline_output.float() - sequential_output.float()
    bad = difference.abs() > 0.05 + 0.05 * sequential_output.float().abs()
    numeric = torch.tensor(
        [
            float(bad.sum()),
            float(difference.abs().max()),
            float(difference.abs().mean()),
            float(
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(sequential_output.float()).clamp_min(1e-20)
            ),
        ],
        dtype=torch.float64,
        device=device,
    )
    rank_max(numeric)
    if int(numeric[0]) or float(numeric[3]) > 5e-3:
        raise AssertionError(f"pipeline C4 gate failed: {numeric.tolist()}")

    def measure(fn):
        for _ in range(args.warmup):
            fn()[1][1].synchronize()
        samples = []
        for _ in range(args.iters):
            _, events = fn()
            events[1].synchronize()
            samples.append(events[0].elapsed_time(events[1]))
        values = torch.tensor(samples, dtype=torch.float64, device=device)
        rank_max(values)
        return summarize_ms(values.cpu().tolist())

    torch.cuda.reset_peak_memory_stats(device)
    pipeline_stats = measure(pipelined)
    sequential_stats = measure(sequential)
    peak = torch.tensor(torch.cuda.max_memory_allocated(device), dtype=torch.int64, device=device)
    rank_max(peak)
    if rank == 0:
        record = {
            "schema_version": 1,
            "benchmark": "sonic_mxfp8_ep4_load_aware_pipeline",
            "node_label": args.node_label,
            "topology": args.topology_label,
            "run_label": args.run_label,
            "source_tokens_per_rank": args.tokens,
            "chunk_tokens": args.chunk_tokens,
            "chunks": len(items),
            "chunk_execution_order": [item["original_index"] for item in items],
            "routing_trace_file": Path(args.routing_trace).name,
            "routing_trace_sha256": trace_sha,
            "data_path": "route_pack_plus_segmented_reduce",
            "timing_scope": "E0_prequantized",
            "source_quantization_in_timed_forward": False,
            "sequential_chunked_rank_max": sequential_stats,
            "pipelined_rank_max": pipeline_stats,
            "pipeline_vs_sequential_speedup": sequential_stats["p50_ms"]
            / pipeline_stats["p50_ms"],
            "correctness": {
                "bad_count": int(numeric[0]),
                "max_abs": float(numeric[1]),
                "mean_abs": float(numeric[2]),
                "relative_l2": float(numeric[3]),
            },
            "workspace_bytes": {
                "route_pack": route_workspace.nbytes,
                "local_mlp": mlp_workspace.nbytes,
                "triple_reduce": sum(workspace.nbytes for workspace in reduce_workspaces),
            },
            "peak_allocated_bytes": int(peak),
            "warmup": args.warmup,
            "iterations": args.iters,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
