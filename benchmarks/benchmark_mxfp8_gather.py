# Copyright (c) 2026, SonicMoE contributors.
"""SM100 MXFP8 routed-activation materialization A/B benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch
from quack.blockscaled import MXFP8_E4M3
from quack.epilogue.library import gated_preact_quant_mod

from sonicmoe.functional.triton_kernels.mxfp8_quant import (
    quantize_mxfp8_gather_varlen_m,
    quantize_mxfp8_varlen_m,
    quantize_mxfp8_weight,
)


def _measure_backlog(
    cases: dict[str, Callable[[], object]], warmup: int, trials: int, inner: int
) -> dict[str, dict[str, float]]:
    names = list(cases)
    for _ in range(warmup):
        for fn in cases.values():
            fn()
    torch.cuda.synchronize()
    samples = {name: [] for name in names}
    for trial in range(trials):
        order = names[trial % len(names) :] + names[: trial % len(names)]
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(inner):
                cases[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / inner)
    return {
        name: {
            "mean_us": statistics.fmean(values),
            "p50_us": statistics.median(values),
            "min_us": min(values),
        }
        for name, values in samples.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--inner", type=int, default=50)
    args = parser.parse_args()
    if torch.cuda.get_device_properties(0).major != 10:
        raise RuntimeError("this benchmark requires SM100")
    total_m = args.tokens * args.top_k
    if total_m % args.experts:
        raise ValueError("tokens * top-k must be divisible by experts")
    per_expert = total_m // args.experts
    offsets = torch.arange(
        0, total_m + 1, per_expert, dtype=torch.int32, device="cuda"
    )
    torch.manual_seed(123)
    x = torch.randn(args.tokens, args.hidden, dtype=torch.bfloat16, device="cuda")
    scatter_idx = torch.randperm(total_m, device="cuda").to(torch.int32)
    reverse_idx = torch.empty_like(scatter_idx)
    reverse_idx[scatter_idx.long()] = torch.arange(
        total_m, dtype=torch.int32, device="cuda"
    )
    gather_idx = scatter_idx // args.top_k
    weight = torch.randn(
        args.experts,
        2 * args.intermediate,
        args.hidden,
        dtype=torch.bfloat16,
        device="cuda",
    )
    weight_mx = quantize_mxfp8_weight(weight)
    padded_rm = (total_m + 127) // 128 + args.experts - 1
    sf_shape = (1, padded_rm, args.hidden // 128, 32, 4, 4)
    old_q = torch.empty(
        total_m, args.hidden, dtype=MXFP8_E4M3.qdata_dtype, device="cuda"
    )
    new_q = torch.empty_like(x, dtype=MXFP8_E4M3.qdata_dtype)
    old_sf = torch.empty(sf_shape, dtype=MXFP8_E4M3.scale_dtype, device="cuda")
    new_sf = torch.empty_like(old_sf)
    linear_sf = torch.empty(
        args.tokens, args.hidden // 32, dtype=torch.uint8, device="cuda"
    )

    def quant_old():
        return quantize_mxfp8_varlen_m(
            x,
            offsets,
            gather_idx=gather_idx,
            qdata_out=old_q,
            scale_out=old_sf,
        )

    def quant_new_two_stage():
        return quantize_mxfp8_gather_varlen_m(
            x,
            offsets,
            gather_idx,
            qdata_out=new_q,
            scale_out=new_sf,
            linear_scale_out=linear_sf,
        )

    def quant_new_one_launch():
        return quantize_mxfp8_gather_varlen_m(
            x,
            offsets,
            gather_idx,
            reverse_idx=reverse_idx,
            top_k=args.top_k,
            qdata_out=new_q,
            scale_out=new_sf,
        )

    old_operand = quant_old()
    new_operand = quant_new_one_launch()
    preact = torch.empty(
        total_m, 2 * args.intermediate, dtype=torch.bfloat16, device="cuda"
    )
    postact = torch.empty(
        total_m,
        args.intermediate,
        dtype=MXFP8_E4M3.qdata_dtype,
        device="cuda",
    )
    postact_sf = torch.empty(
        (1, padded_rm, args.intermediate // 128, 32, 4, 4),
        dtype=MXFP8_E4M3.scale_dtype,
        device="cuda",
    )
    mod = gated_preact_quant_mod("swiglu")

    def fc1(operand, A_idx, use_tma_gather):
        mod.gemm(
            operand.qdata,
            weight_mx.qdata,
            preact,
            epi_args={"postact": postact, "postact_sf": postact_sf},
            tile_M=128,
            tile_N=256,
            cluster_M=1,
            cluster_N=1,
            persistent=True,
            is_dynamic_persistent=True,
            cu_seqlens_m=offsets,
            A_idx=A_idx,
            use_tma_gather=use_tma_gather,
            SFA=operand.scale,
            SFB=weight_mx.scale,
            bs_format_a=MXFP8_E4M3.name,
            bs_format_b=MXFP8_E4M3.name,
        )

    def pipeline_old():
        fc1(quant_old(), None, False)

    def pipeline_new_cpasync():
        fc1(quant_new_one_launch(), gather_idx, False)

    def pipeline_new_tma():
        fc1(quant_new_one_launch(), gather_idx, True)

    results = _measure_backlog(
        {
            "quant_materialized": quant_old,
            "quant_zero_material_two_stage": quant_new_two_stage,
            "quant_zero_material_one_launch": quant_new_one_launch,
            "fc1_materialized": lambda: fc1(old_operand, None, False),
            "fc1_gather_cpasync": lambda: fc1(new_operand, gather_idx, False),
            "fc1_gather_tma": lambda: fc1(new_operand, gather_idx, True),
            "pipeline_materialized": pipeline_old,
            "pipeline_gather_cpasync": pipeline_new_cpasync,
            "pipeline_gather_tma": pipeline_new_tma,
        },
        args.warmup,
        args.trials,
        args.inner,
    )
    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "shape": vars(args),
        "qdata_mib": {
            "materialized": old_q.numel() / 2**20,
            "zero_material": new_q.numel() / 2**20,
        },
        "results": results,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
