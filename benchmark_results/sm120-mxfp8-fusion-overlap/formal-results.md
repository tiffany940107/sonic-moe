# 0902 Formal MXFP8 EP4 Results

All latency values are milliseconds at E0: prequantized source, exact routing,
EP dispatch, local expert compute, weighted reduce, and combine. Each supported
cell is the median of three independent process-restart values. Main tables use
p50; the tail table reports `p50 / p95 / p99` from the same restarts.

## PRO5000-A / same_numa

| m/rank | Sonic baseline | prior fused | full workspace | block-scale | MegaMoE | baseline/full |
|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 23.269 | 13.150 | 13.051 | 26.082 | 43.301 | 1.783× |
| 16,384 | 46.649 | 26.001 | 25.904 | 51.938 | 85.931 | 1.801× |
| 32,768 | 104.111 | 51.970 | 51.878 | 109.399 | unsupported | 2.007× |
| 46,080 | 147.171 | 73.273 | 73.035 | 156.008 | unsupported | 2.015× |

Tail latency (`p50 / p95 / p99`, ms):

| m/rank | full workspace | block-scale | MegaMoE |
|---:|---:|---:|---:|
| 8,192 | 13.051 / 13.112 / 13.134 | 26.082 / 26.231 / 26.724 | 43.301 / 43.636 / 43.740 |
| 16,384 | 25.904 / 26.107 / 26.129 | 51.938 / 52.164 / 53.205 | 85.931 / 86.209 / 86.369 |
| 32,768 | 51.878 / 52.105 / 52.128 | 109.399 / 109.853 / 110.650 | — |
| 46,080 | 73.035 / 73.496 / 73.694 | 156.008 / 156.546 / 156.773 | — |

## PRO5000-B / same_numa

| m/rank | Sonic baseline | prior fused | full workspace | block-scale | MegaMoE | baseline/full |
|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 25.353 | 13.939 | 13.908 | 29.003 | 43.458 | 1.823× |
| 16,384 | 50.680 | 27.762 | 27.721 | 57.545 | 86.434 | 1.828× |
| 32,768 | 114.697 | 55.495 | 55.380 | 118.978 | unsupported | 2.071× |
| 46,080 | 161.753 | 78.007 | 77.819 | 168.352 | unsupported | 2.079× |

Tail latency (`p50 / p95 / p99`, ms):

| m/rank | full workspace | block-scale | MegaMoE |
|---:|---:|---:|---:|
| 8,192 | 13.908 / 13.926 / 13.942 | 29.003 / 29.227 / 29.944 | 43.458 / 43.750 / 43.936 |
| 16,384 | 27.721 / 27.749 / 27.761 | 57.545 / 57.791 / 58.365 | 86.434 / 86.974 / 87.139 |
| 32,768 | 55.380 / 55.430 / 55.477 | 118.978 / 119.132 / 119.240 | — |
| 46,080 | 77.819 / 77.860 / 77.897 | 168.352 / 168.549 / 168.747 | — |

## Unbalanced common-trace comparison

Missing cells were not run; they are distinct from an explicit `unsupported` status.

### PRO5000-A / same_numa

| trace | legacy | prior fused | full workspace | baseline/full | block-scale | MegaMoE | greedy EPLB | R0 ms | break-even |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `m8192_rank_r1.227` | 27.858 | 15.894 | — | 1.753× | — | — | — | — | — |
| `m8192_segment_aligned` | 55.317 | 31.268 | — | 1.769× | — | — | — | — | — |
| `m16384_joint_aligned_max` | 84.748 | 46.411 | — | 1.826× | — | — | — | — | — |
| `m16384_rank_r1.106` | 50.972 | 28.553 | — | 1.785× | 57.449 | — | — | — | — |
| `m16384_rank_r1.227` | 56.648 | 31.624 | — | 1.791× | 63.766 | — | — | — | — |
| `m16384_rank_r1.698` | 80.193 | 42.920 | — | 1.868× | 88.498 | — | — | — | — |
| `m16384_segment_aligned` | 122.029 | 62.244 | — | 1.960× | — | — | — | — | — |
| `m16384_segment_permuted` | 57.147 | 32.538 | — | 1.756× | — | — | — | — | — |
| `m32768_rank_r1.227` | 125.323 | 62.555 | — | 2.003× | — | — | — | — | — |
| `m32768_segment_aligned` | 248.165 | 124.853 | — | 1.988× | — | — | — | — | — |
| `m46080_rank_r1.227` | 177.187 | 87.902 | — | 2.016× | — | — | — | — | — |
| `m46080_segment_aligned` | — | 174.377 | — | — | — | — | — | — | — |

### PRO5000-B / same_numa

| trace | legacy | prior fused | full workspace | baseline/full | block-scale | MegaMoE | greedy EPLB | R0 ms | break-even |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `m8192_rank_r1.227` | — | 16.114 | — | — | — | — | 20.440 | 82.04 | never |
| `m8192_segment_aligned` | — | 33.971 | — | — | — | — | 20.601 | 79.91 | 5.98 |
| `m16384_joint_aligned_max` | — | 46.897 | — | — | — | 113.752 | 38.451 | 72.22 | 8.55 |
| `m16384_rank_r1.106` | — | 28.899 | — | — | — | 90.089 | 41.209 | 84.14 | never |
| `m16384_rank_r1.227` | — | 32.025 | — | — | — | 94.816 | 41.374 | 84.93 | never |
| `m16384_rank_r1.698` | — | 43.476 | — | — | — | 114.982 | 41.380 | 86.98 | 41.49 |
| `m16384_segment_aligned` | — | 67.538 | — | — | — | 148.605 | 41.446 | 87.75 | 3.36 |
| `m16384_segment_permuted` | — | 37.049 | — | — | — | 95.574 | 41.424 | 87.75 | never |
| `m32768_rank_r1.227` | — | 63.366 | — | — | — | — | 82.673 | 84.84 | never |
| `m32768_segment_aligned` | — | 134.621 | — | — | — | — | 82.631 | 78.23 | 1.50 |
| `m46080_rank_r1.227` | — | 89.115 | — | — | — | — | 116.020 | 82.62 | never |
| `m46080_segment_aligned` | — | 189.102 | — | — | — | — | 116.049 | 81.38 | 1.11 |

## Persistent isolated-hot-expert experiment

This trace keeps rank-level skew mild while one expert remains hot; it
separates whole-expert migration from load-splitting by replication.

| node | contiguous | whole-expert greedy | replica(1) | replica speedup | R0 replica ms | measured break-even | MegaMoE |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRO5000-B | 33.900 | 40.579 | 30.445 | 1.113× | 359.42 | 104.0 steps | 89.387 |

## Local K0/K1 timing boundaries

K0 is preallocated local expert compute in native layout; K1 adds row-linear
MXFP8 route packing and weighted reduction but still excludes EP communication.
Values are `p50 / p95 / p99` in ms.

| node | level | eager/latency | CUDA graph | graph speedup | restarts | correct | timed alloc delta |
|---|---|---:|---:|---:|---:|---|---:|
| PRO5000-A | K0 | 10.562 / 10.580 / 10.584 | 10.560 / 10.572 / 10.578 | 1.0002× | 3 | true | 0 |
| PRO5000-A | K1 | 12.793 / 12.813 / 12.822 | — | — | 3 | true | 0 |
| PRO5000-B | K0 | 10.627 / 10.659 / 10.666 | 10.634 / 10.659 / 10.664 | 0.9997× | 3 | true | 0 |
| PRO5000-B | K1 | 12.885 / 12.907 / 12.952 | — | — | 3 | true | 0 |

## Unsupported/invalid attempts

| node | topology | backend | m/rank | trace | status |
|---|---|---|---:|---|---|
| PRO5000-A | cross_numa_2plus2 | environment | 8,192 | balanced | resource_blocked_external_gpu_memory |
| PRO5000-A | same_numa | blockscale_groupwise | 16,384 | m16384_joint_aligned_max | invalid_c4_bad_count |
| PRO5000-A | same_numa | blockscale_groupwise | 16,384 | m16384_persistent_single_hot | invalid_c4_bad_elements |
| PRO5000-A | same_numa | blockscale_groupwise | 16,384 | m16384_segment_aligned | invalid_c4_bad_count |
| PRO5000-A | same_numa | blockscale_groupwise | 16,384 | m16384_segment_permuted | invalid_c4_bad_count |
| PRO5000-A | same_numa | megamoe_native | 32,768 | balanced | unsupported_signed_int32_extent |
| PRO5000-A | same_numa | megamoe_native | 46,080 | balanced | unsupported_signed_int32_extent |
| PRO5000-A | same_numa | sonic_legacy_data_path | 46,080 | m46080_segment_aligned | resource_oom |
| PRO5000-B | cross_numa_2plus2 | environment | 8,192 | balanced | resource_blocked_external_gpu_memory |
| PRO5000-B | same_numa | megamoe_native | 32,768 | balanced | unsupported_signed_int32_extent |
| PRO5000-B | same_numa | megamoe_native | 46,080 | balanced | unsupported_signed_int32_extent |

Correctness-invalid or unsupported attempts never contribute a latency value.
Sonic and block-scale inline C4 require `bad_count=0` and relative L2 `<=5e-3`;
MegaMoE uses the separately recorded strict common-trace transport/output gate.
