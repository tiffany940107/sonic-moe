# Copyright (c) 2026, SonicMoE contributors.
"""Microbenchmark SM100 MXFP8 single- and dual-layout weight refresh."""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch

from sonicmoe.functional.triton_kernels.mxfp8_quant import (
    launch_sgd_update_and_quantize_mxfp8_weight_dual,
    quantize_mxfp8_weight,
    quantize_mxfp8_weight_dual,
    sgd_update_and_quantize_mxfp8_weight,
)


def _time_ms(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _time_host_us(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1000)
    torch.cuda.synchronize()
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if torch.cuda.get_device_properties(0).major != 10:
        raise RuntimeError("this benchmark requires SM100")

    torch.manual_seed(0)
    weights = (
        torch.randn(
            args.experts,
            2 * args.intermediate,
            args.hidden,
            device="cuda",
            dtype=torch.bfloat16,
        ),
        torch.randn(
            args.experts,
            args.hidden,
            args.intermediate,
            device="cuda",
            dtype=torch.bfloat16,
        ),
    )
    grads = tuple(torch.randn_like(weight) for weight in weights)
    single_buffers = []
    dual_buffers = []
    for weight in weights:
        row = quantize_mxfp8_weight(weight, dim=-1)
        col = quantize_mxfp8_weight(weight, dim=-2)
        single_buffers.append((row.qdata, row.scale, col.qdata, col.scale))
        dual_row, dual_col = quantize_mxfp8_weight_dual(weight)
        dual_buffers.append(
            (dual_row.qdata, dual_row.scale, dual_col.qdata, dual_col.scale)
        )

    def single() -> None:
        for weight, (row_q, row_sf, col_q, col_sf) in zip(weights, single_buffers):
            quantize_mxfp8_weight(weight, dim=-1, qdata_out=row_q, scale_out=row_sf)
            quantize_mxfp8_weight(weight, dim=-2, qdata_out=col_q, scale_out=col_sf)

    def row_only() -> None:
        for weight, (row_q, row_sf, _, _) in zip(weights, single_buffers):
            quantize_mxfp8_weight(weight, dim=-1, qdata_out=row_q, scale_out=row_sf)

    def col_only() -> None:
        for weight, (_, _, col_q, col_sf) in zip(weights, single_buffers):
            quantize_mxfp8_weight(weight, dim=-2, qdata_out=col_q, scale_out=col_sf)

    single_ms = _time_ms(single, args.warmup, args.repeats)
    print(f"single-layout kernels: {single_ms * 1000:.3f} us")
    os.environ["SONICMOE_MXFP8_WEIGHT_FLAT_BLOCK"] = "0"
    for block_k in (128, 256, 512):
        for block_m in (8, 16, 32, 64):
            for num_warps in (4, 8) if block_m >= 16 else (4,):
                os.environ["SONICMOE_MXFP8_WEIGHT_BLOCK_M"] = str(block_m)
                os.environ["SONICMOE_MXFP8_WEIGHT_BLOCK_K"] = str(block_k)
                os.environ["SONICMOE_MXFP8_WEIGHT_WARPS"] = str(num_warps)
                print(
                    f"  rowwise block_m={block_m:>2} block_k={block_k:>3} "
                    f"warps={num_warps}: "
                    f"{_time_ms(row_only, args.warmup, args.repeats) * 1000:.3f} us"
                )
    for flat_block in (1024, 2048, 4096, 8192):
        os.environ["SONICMOE_MXFP8_WEIGHT_FLAT_BLOCK"] = str(flat_block)
        for num_warps in (4, 8):
            os.environ["SONICMOE_MXFP8_WEIGHT_WARPS"] = str(num_warps)
            print(
                f"  rowwise flat={flat_block:>4} warps={num_warps}: "
                f"{_time_ms(row_only, args.warmup, args.repeats) * 1000:.3f} us"
            )
    os.environ["SONICMOE_MXFP8_WEIGHT_FLAT_BLOCK"] = "0"
    print(
        f"  dim0 only:    {_time_ms(col_only, args.warmup, args.repeats) * 1000:.3f} us"
    )
    os.environ["SONICMOE_MXFP8_WEIGHT_BLOCK_M"] = "32"
    os.environ["SONICMOE_MXFP8_WEIGHT_BLOCK_K"] = "128"
    os.environ["SONICMOE_MXFP8_WEIGHT_WARPS"] = "4"

    def separate_sgd() -> None:
        torch._foreach_add_(weights, grads, alpha=-1e-4)
        row_only()

    def fused_sgd() -> None:
        for weight, grad, (row_q, row_sf, _, _) in zip(weights, grads, single_buffers):
            sgd_update_and_quantize_mxfp8_weight(
                weight,
                grad,
                1e-4,
                qdata_out=row_q,
                scale_out=row_sf,
            )

    os.environ["SONICMOE_MXFP8_SGD_FLAT_BLOCK"] = "0"
    print(
        "separate SGD + rowwise: "
        f"{_time_ms(separate_sgd, args.warmup, args.repeats) * 1000:.3f} us"
    )
    print(
        "  host enqueue:         "
        f"{_time_host_us(separate_sgd, args.warmup, args.repeats):.3f} us"
    )
    for block_k in (128, 256, 512):
        for block_m in (8, 16, 32, 64):
            for num_warps in (4, 8) if block_m >= 16 else (4,):
                os.environ["SONICMOE_MXFP8_WEIGHT_BLOCK_M"] = str(block_m)
                os.environ["SONICMOE_MXFP8_WEIGHT_BLOCK_K"] = str(block_k)
                os.environ["SONICMOE_MXFP8_WEIGHT_WARPS"] = str(num_warps)
                print(
                    f"fused SGD block_m={block_m:>2} block_k={block_k:>3} "
                    f"warps={num_warps}: "
                    f"{_time_ms(fused_sgd, args.warmup, args.repeats) * 1000:.3f} us; "
                    f"host {_time_host_us(fused_sgd, args.warmup, args.repeats):.3f} us"
                )
    for flat_block in (1024, 2048, 4096, 8192):
        os.environ["SONICMOE_MXFP8_SGD_FLAT_BLOCK"] = str(flat_block)
        for num_warps in (4, 8):
            os.environ["SONICMOE_MXFP8_SGD_WARPS"] = str(num_warps)
            print(
                f"fused SGD flat={flat_block:>4} warps={num_warps}: "
                f"{_time_ms(fused_sgd, args.warmup, args.repeats) * 1000:.3f} us; "
                f"host {_time_host_us(fused_sgd, args.warmup, args.repeats):.3f} us"
            )
    os.environ["SONICMOE_MXFP8_SGD_FLAT_BLOCK"] = "4096"
    os.environ["SONICMOE_MXFP8_SGD_WARPS"] = "4"
    for block_n in (32, 64, 128):
        for warps in (4, 8):
            os.environ["SONICMOE_MXFP8_WEIGHT_DUAL_BLOCK_N"] = str(block_n)
            os.environ["SONICMOE_MXFP8_WEIGHT_DUAL_WARPS"] = str(warps)

            def dual() -> None:
                for weight, (row_q, row_sf, col_q, col_sf) in zip(
                    weights, dual_buffers
                ):
                    quantize_mxfp8_weight_dual(
                        weight,
                        row_qdata_out=row_q,
                        row_scale_out=row_sf,
                        col_qdata_out=col_q,
                        col_scale_out=col_sf,
                    )

            dual_ms = _time_ms(dual, args.warmup, args.repeats)
            print(
                f"dual block_n={block_n:>3} warps={warps}: "
                f"{dual_ms * 1000:.3f} us ({single_ms / dual_ms:.3f}x)"
            )

    def separate_dual_sgd() -> None:
        torch._foreach_add_(weights, grads, alpha=-1e-4)
        for weight, (row_q, row_sf, col_q, col_sf) in zip(weights, dual_buffers):
            quantize_mxfp8_weight_dual(
                weight,
                row_qdata_out=row_q,
                row_scale_out=row_sf,
                col_qdata_out=col_q,
                col_scale_out=col_sf,
            )

    separate_dual_ms = _time_ms(separate_dual_sgd, args.warmup, args.repeats)
    print(f"separate SGD + dual: {separate_dual_ms * 1000:.3f} us")
    for block_n in (32, 64, 128):
        for warps in (4, 8):
            os.environ["SONICMOE_MXFP8_SGD_DUAL_BLOCK_N"] = str(block_n)
            os.environ["SONICMOE_MXFP8_SGD_DUAL_WARPS"] = str(warps)

            def fused_dual_sgd() -> None:
                for weight, grad, (row_q, row_sf, col_q, col_sf) in zip(
                    weights, grads, dual_buffers
                ):
                    launch_sgd_update_and_quantize_mxfp8_weight_dual(
                        weight,
                        grad,
                        1e-4,
                        row_qdata_out=row_q,
                        row_scale_out=row_sf,
                        col_qdata_out=col_q,
                        col_scale_out=col_sf,
                    )

            fused_dual_ms = _time_ms(fused_dual_sgd, args.warmup, args.repeats)
            print(
                f"fused SGD + dual block_n={block_n:>3} warps={warps}: "
                f"{fused_dual_ms * 1000:.3f} us "
                f"({separate_dual_ms / fused_dual_ms:.3f}x)"
            )


if __name__ == "__main__":
    main()
