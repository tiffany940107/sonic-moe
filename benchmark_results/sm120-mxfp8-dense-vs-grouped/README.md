# SM120 MXFP8 dense and grouped GEMM comparison

For the PRO5000-only single-GPU scope, timing boundaries, scripts, and a Chinese explanation, see [PRO5000_SINGLE_GPU_RESULTS.md](PRO5000_SINGLE_GPU_RESULTS.md).
The aligned Sonic versus FlashInfer old/PR #4660 table is in [FLASHINFER_VS_SONIC.md](FLASHINFER_VS_SONIC.md). The separate four-GPU EP4 campaign is documented in [ep4/README.md](ep4/README.md).

All numbers are GPU-event P50 kernel latency. Inputs and weights are already
quantized as OCP E4M3 plus E8M0 scales with semantic 1x32 granularity along
K; allocation, quantization, scale packing, checks, and cold JIT are excluded.

## Protocol

- Dense: `20` warmups, `100` iterations, `3` repeats.
- Grouped: eight balanced groups, `20` warmups, `100` iterations, `3` repeats.
- Each displayed metric is the median of the per-repeat P50 values.
- `dense_loop` uses eight distinct expert weights and preserves grouped-MoE semantics. `one_big_dense` shares one weight and is only an efficiency ceiling.
- The discarded Bash `GROUPS` collision and final eight-group rerun are documented in the [harness audit](AUDIT.md).

## Environments

Hostnames, IP addresses, GPU UUIDs, PCI bus IDs, and visible-device indices are intentionally omitted. Node labels must be anonymous public aliases.

| label | GPU | GiB | power W | driver | Sonic dense/grouped | QuACK |
| --- | --- | --- | --- | --- | --- | --- |
| PRO6000 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 95.0 | 600.00 | 595.58.03 | f0336561/f0336561 | c87c9d1d |
| PRO5000-A | NVIDIA Graphics Device | 71.1 | 350.00 | 580.95.05 | f0336561/f0336561 | c87c9d1d |
| PRO5000-B | NVIDIA Graphics Device | 71.1 | 350.00 | 580.95.05 | f0336561/f0336561 | c87c9d1d |

## Dense GEMM

| workload | PRO6000 ms / TFLOP/s | PRO5000-A ms / TFLOP/s | PRO5000-B ms / TFLOP/s |
| --- | --- | --- | --- |
| mnk_8k4k4k | 0.348880 / 787.9 | 0.574000 / 478.9 | 0.569408 / 482.7 |
| mnk_16k4k4k | 0.721936 / 761.5 | 1.320544 / 416.3 | 1.313200 / 418.6 |
| mnk_32k4k4k | 1.481312 / 742.3 | 2.697824 / 407.6 | 2.698208 / 407.5 |

## Grouped GEMM

| workload | PRO6000 ms / TFLOP/s | PRO5000-A ms / TFLOP/s | PRO5000-B ms / TFLOP/s |
| --- | --- | --- | --- |
| mnk_8k_1280_2k | 0.071232 / 603.0 | 0.093264 / 460.5 | 0.093776 / 458.0 |
| mnk_8k_2k_1280 | 0.069152 / 621.1 | 0.098560 / 435.8 | 0.098880 / 434.4 |
| mnk_16k_1280_2k | 0.116320 / 738.5 | 0.189952 / 452.2 | 0.188080 / 456.7 |
| mnk_16k_2k_1280 | 0.117664 / 730.0 | 0.222880 / 385.4 | 0.222704 / 385.7 |
| mnk_32k_1280_2k | 0.254656 / 674.6 | 0.439968 / 390.5 | 0.440848 / 389.7 |
| mnk_32k_2k_1280 | 0.247504 / 694.1 | 0.444048 / 386.9 | 0.444960 / 386.1 |

## Same-MNK grouped acceleration

| workload | PRO6000 grouped speedup | PRO5000-A grouped speedup | PRO5000-B grouped speedup |
| --- | --- | --- | --- |
| mnk_8k_1280_2k | 1.965x | 1.426x | 1.421x |
| mnk_8k_2k_1280 | 1.488x | 1.642x | 1.667x |
| mnk_16k_1280_2k | 1.273x | 1.329x | 1.343x |
| mnk_16k_2k_1280 | 1.534x | 1.084x | 1.095x |
| mnk_32k_1280_2k | 1.153x | 1.027x | 1.014x |
| mnk_32k_2k_1280 | 1.152x | 1.049x | 1.038x |

## Cross-GPU summary (baseline: PRO6000)

- `PRO6000` is `1.763x` faster than `PRO5000-A` on dense and `1.617x` faster on grouped GEMM (geometric mean).
- `PRO6000` is `1.755x` faster than `PRO5000-B` on dense and `1.618x` faster on grouped GEMM (geometric mean).
- `PRO6000` grouped vs eight dense launches: `1.402x`; efficiency vs one-big-dense ceiling: `98.2%` (geometric mean).
- `PRO5000-A` grouped vs eight dense launches: `1.240x`; efficiency vs one-big-dense ceiling: `99.2%` (geometric mean).
- `PRO5000-B` grouped vs eight dense launches: `1.242x`; efficiency vs one-big-dense ceiling: `98.4%` (geometric mean).

## Validation

Every dense output was finite and had relative L2 error <= 0.005 vs a dequantized FP32 reference. Every grouped output was bit-identical to the eight-launch dense-loop baseline (`max_abs = 0`, `rel_l2 = 0`).

## Raw data

- `PRO6000`: [dense](pro6000_suite_v2/dense.json), [grouped-vs-dense](pro6000_suite_v2/grouped-vs-dense.json)
- `PRO5000-A`: [dense](pro5000_A_suite_v2/dense.json), [grouped-vs-dense](pro5000_A_suite_v2/grouped-vs-dense.json)
- `PRO5000-B`: [dense](pro5000_B_suite_v2/dense.json), [grouped-vs-dense](pro5000_B_suite_v2/grouped-vs-dense.json)

## Four-GPU EP4 system result

This is a separate customer-shape campaign: global tokens 16,384, 4,096/rank, top-k 24, hidden 2,560, post-SwiGLU intermediate 1,024, 768 global experts, 192 local experts/rank, and world size 4. Values are the median of three restart P50 four-rank critical-path CUDA-event latencies.

| node | topology | FlashInfer FP8 wrapper | Sonic-MXFP8 wrapper | MegaMoE MXFP8 p2p_direct |
| --- | --- | ---: | ---: | ---: |
| PRO5000-A | same NUMA | 18.621 ms | 15.746 ms | 19.719 ms |
| PRO5000-A | cross NUMA, symmetric 2+2 | — | — | — |
| PRO5000-B | same NUMA | 18.575 ms | 15.806 ms | 19.731 ms |
| PRO5000-B | cross NUMA, symmetric 2+2 | 18.522 ms | 15.667 ms | 28.593 ms |

FlashInfer and Sonic use external deduplicated Torch/NCCL EP wrappers around local grouped kernels; MegaMoE uses its own fused multi-rank runtime. FlashInfer is FP8 with FP32 groupwise scales, whereas Sonic and MegaMoE use MXFP8 E4M3 plus E8M0 K32 scales. This is a full-data-path comparison, not a bit-equivalent kernel A/B.

The public Sonic/EPLB entry points are [mxfp8-ep4-e2e.py](../../benchmarks/mxfp8-ep4-e2e.py), [run-mxfp8-ep4-suite.sh](../../benchmarks/run-mxfp8-ep4-suite.sh), and [summarize-mxfp8-ep4-suite.py](../../benchmarks/summarize-mxfp8-ep4-suite.py). The external FlashInfer wrapper is [flashinfer-fp8-ep4-e2e.py](../../benchmarks/flashinfer-fp8-ep4-e2e.py), with [run-flashinfer-fp8-ep4-suite.sh](../../benchmarks/run-flashinfer-fp8-ep4-suite.sh). See the [full EP4 report](ep4/README.md) for imbalance/EPLB, migration cost, IBGDA correctness status, and the anonymous machine-readable result.
