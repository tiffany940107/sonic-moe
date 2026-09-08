# Copyright (c) 2026, SonicMoE contributors.
"""Kernel-only SM100 MXFP8 FC1 role tuning benchmark."""

from __future__ import annotations

import argparse
import statistics

import torch
from quack.blockscaled import MXFP8_E4M3
from quack.epilogue.library import gated_preact_quant_mod

from sonicmoe.functional.triton_kernels.mxfp8_quant import (
    quantize_mxfp8_varlen_m,
    quantize_mxfp8_weight,
)


def _median_us(fn, warmup: int, repeats: int) -> float:
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
        samples.append(start.elapsed_time(end) * 1000)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if torch.cuda.get_device_properties(0).major != 10:
        raise RuntimeError("this benchmark requires SM100")
    total_m = args.tokens * args.top_k
    if total_m % args.experts:
        raise ValueError("this tuning benchmark requires balanced integral expert M")
    per_expert = total_m // args.experts
    offsets = torch.arange(
        0,
        total_m + 1,
        per_expert,
        dtype=torch.int32,
        device="cuda",
    )
    x = torch.randn(total_m, args.hidden, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(
        args.experts,
        2 * args.intermediate,
        args.hidden,
        dtype=torch.bfloat16,
        device="cuda",
    )
    x_mx = quantize_mxfp8_varlen_m(x, offsets)
    weight_mx = quantize_mxfp8_weight(weight)
    preact = torch.empty(
        total_m, 2 * args.intermediate, dtype=torch.bfloat16, device="cuda"
    )
    postact = torch.empty(
        total_m,
        args.intermediate,
        dtype=MXFP8_E4M3.qdata_dtype,
        device="cuda",
    )
    padded_rm = (total_m + 127) // 128 + args.experts - 1
    postact_sf = torch.empty(
        (1, padded_rm, args.intermediate // 128, 32, 4, 4),
        dtype=MXFP8_E4M3.scale_dtype,
        device="cuda",
    )
    mod = gated_preact_quant_mod("swiglu")
    configs = []
    for tile_m, cluster_m in ((128, 1), (256, 1), (256, 2)):
        for tile_n in (128, 192, 256):
            for cluster_n in (1, 2):
                for dynamic in (False, True):
                    configs.append((tile_m, tile_n, cluster_m, cluster_n, dynamic))

    for tile_m, tile_n, cluster_m, cluster_n, dynamic in configs:

        def run(
            tile_m: int = tile_m,
            tile_n: int = tile_n,
            cluster_m: int = cluster_m,
            cluster_n: int = cluster_n,
            dynamic: bool = dynamic,
        ) -> None:
            mod.gemm(
                x_mx.qdata,
                weight_mx.qdata,
                preact,
                epi_args={"postact": postact, "postact_sf": postact_sf},
                tile_M=tile_m,
                tile_N=tile_n,
                cluster_M=cluster_m,
                cluster_N=cluster_n,
                persistent=True,
                is_dynamic_persistent=dynamic,
                cu_seqlens_m=offsets,
                SFA=x_mx.scale,
                SFB=weight_mx.scale,
                bs_format_a=MXFP8_E4M3.name,
                bs_format_b=MXFP8_E4M3.name,
            )

        try:
            latency = _median_us(run, args.warmup, args.repeats)
        except Exception as error:  # noqa: BLE001 - tuning failures vary
            print(
                f"m={tile_m} n={tile_n} cm={cluster_m} cn={cluster_n} "
                f"dynamic={int(dynamic)}: unsupported ({error})"
            )
        else:
            print(
                f"m={tile_m} n={tile_n} cm={cluster_m} cn={cluster_n} "
                f"dynamic={int(dynamic)}: {latency:.3f} us"
            )


if __name__ == "__main__":
    main()
