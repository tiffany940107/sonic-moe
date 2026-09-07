# SM100 MXFP8 single-GPU results

Results were collected on 2026-09-07. They are point measurements, not a
claim that one backend wins for every MoE shape.

## Environment

| Component | Value |
|---|---|
| GPU | NVIDIA B200, compute capability 10.0 |
| Container | `nvcr.io/nvidia/pytorch:26.08-py3` |
| CUDA runtime | 13.4 (forward compatibility enabled) |
| PyTorch | `2.14.0a0+4fdf77b940.nv26.08` |
| CUTLASS DSL | 4.7.1 |
| Quack | `04a18b5c74d9f6fec599825855e53879e22855b9` |
| Precision | BF16 masters; E4M3 values + E8M0 scales for MXFP8 GEMMs |

All runs use 5 warmup iterations, 20 measured iterations, 8 experts, top-k 2,
SwiGLU, and the default `SONICMOE_MXFP8_AUTOTUNE=0`.

## End-to-end latency

Latency is in milliseconds and memory is peak PyTorch allocated memory in MiB.
Speedup is BF16 p50 divided by MXFP8 p50.

| Mode and shape `(T,H,I)` | Backend | p50 | p95 | p99 | Memory | Speedup |
|---|---|---:|---:|---:|---:|---:|
| inference `(1024,2048,2048)` | BF16 Sonic | 0.5803 | 0.5961 | 0.6059 | 444.1 | — |
| inference `(1024,2048,2048)` | MXFP8 | 0.6057 | 0.6283 | 0.6326 | 543.5 | 0.958x |
| inference `(2048,4096,4096)` | BF16 Sonic | 0.7088 | 0.7213 | 0.7225 | 1680.2 | — |
| inference `(2048,4096,4096)` | MXFP8 | 0.6134 | 0.6557 | 0.9878 | 2077.5 | 1.156x |
| train + SGD `(1024,2048,2048)` | BF16 Sonic | 2.5556 | 2.6426 | 2.7150 | 680.1 | — |
| train + SGD `(1024,2048,2048)` | MXFP8 | 3.1060 | 3.1734 | 3.1751 | 1063.9 | 0.823x |

The larger inference case crosses the compute-amortization point in this
matrix. The middle-sized case is launch/dispatch bound, while training still
pays for multiple role-specific activation and gradient casts. The large
MXFP8 p99 contains one measured outlier; p50 and p95 are more representative
of steady state in this 20-sample run.

Reproduce a row with:

```bash
python benchmarks/benchmark_sm100_mxfp8.py \
  --mode forward --tokens 2048 --experts 8 --top-k 2 \
  --hidden 4096 --intermediate 4096 --warmup 5 --repeats 20

python benchmarks/benchmark_sm100_mxfp8.py \
  --mode training --optimizer-step --tokens 1024 --experts 8 --top-k 2 \
  --hidden 2048 --intermediate 2048 --warmup 5 --repeats 20
```

## Nsight Systems kernel decomposition

For inference `(1024,2048,2048)`, 20 profiled iterations gave these median
kernel times:

| Backend | Component | Median (us) |
|---|---|---:|
| BF16 | FC1 + gated activation | 49.055 |
| BF16 | FC2 | 23.376 |
| MXFP8 | fused route gather + quantize | 20.639 |
| MXFP8 | FC1 + gated activation + requantize | 27.184 |
| MXFP8 | FC2 | 17.919 |

The BF16 expert chain totals 72.431 us. The MXFP8 quantize-plus-expert chain
totals 65.742 us, about 1.10x faster at kernel level. Shared router/reduction
kernels and Python/custom-op launch overhead explain why this shape does not
show an end-to-end win.

Capture and summarize the profile with:

```bash
SONICMOE_PROFILE_RANGE=1 nsys profile \
  --trace=cuda,nvtx --capture-range=cudaProfilerApi \
  --capture-range-end=stop -o profiles/sm100_mxfp8_fwd_1024_2048 \
  python benchmarks/benchmark_sm100_mxfp8.py \
  --mode forward --tokens 1024 --experts 8 --top-k 2 \
  --hidden 2048 --intermediate 2048 --warmup 5 --repeats 20 \
  --backends sonicmoe_mxfp8

nsys stats --report cuda_gpu_kern_sum --format csv \
  profiles/sm100_mxfp8_fwd_1024_2048.nsys-rep
```

## Development ablations

- Versioned weight caching plus the inference-only FC1 epilogue reduced an
  observed MXFP8 forward p50 from 0.956 ms to 0.630 ms (34%); caller-owned
  router scratch reduced the same development workload further to 0.576 ms.
- Enabling the full runtime Quack autotune search regressed a representative
  training-step speedup from 0.819x to 0.740x. It remains opt-in through
  `SONICMOE_MXFP8_AUTOTUNE=1` rather than being the default.
- CUDA Graph, global-atomic FC2 reduction, and coarse whole-FC fusion are not
  enabled without positive SM100 evidence.
