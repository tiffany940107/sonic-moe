# SM120 MXFP8 EP4 Fusion and EPLB Results

This directory records the optimization branch rooted at Sonic `013f1726` and
QuACK `c87c9d1d`. The implementation commits are Sonic `bb79f10d` and QuACK
`d38dd360`. Baseline branches were not overwritten.

## Workload and protocol

- EP=4, 512 global/128 local experts, top-k=32;
- m/rank = 8,192, 16,384, 32,768, or 46,080;
- H=2,048, post-SwiGLU I=1,280, physical FC1 Gate+Up=2,560;
- OCP MXFP8 E4M3 values and E8M0 K32 scales; BF16 reduced output;
- E0 includes exact-route dispatch, local FC1/SwiGLU/requant/FC2, FP32
  weighted reduce, and combine; source quantization is outside E0;
- 20 warmups, 100 CUDA-event samples, three independent process restarts;
- each result is the median of restart p50 rank-max latency;
- C4 requires zero elements satisfying
  `abs(actual-reference) > 0.05 + 0.05*abs(reference)` and relative L2 <=5e-3.

Node names are public aliases. No address, hostname, GPU ordinal/UUID, or PCI
identifier is recorded.

## Balanced same-NUMA E0 latency

Milliseconds:

| node | m/rank | legacy Sonic | fused Sonic | block-scale | MegaMoE | Sonic speedup |
|---|---:|---:|---:|---:|---:|---:|
| PRO5000-A | 8,192 | 23.269 | 13.150 | 26.082 | 43.313 | 1.770x |
| PRO5000-A | 16,384 | 46.649 | 26.001 | 51.938 | 85.933 | 1.794x |
| PRO5000-A | 32,768 | 104.111 | 51.970 | 109.399 | unsupported | 2.003x |
| PRO5000-A | 46,080 | 147.171 | 73.273 | 156.008 | unsupported | 2.009x |
| PRO5000-B | 8,192 | 25.353 | 13.939 | 29.003 | 43.455 | 1.819x |
| PRO5000-B | 16,384 | 50.680 | 27.762 | 57.545 | 86.473 | 1.826x |
| PRO5000-B | 32,768 | 114.697 | 55.495 | 118.978 | unsupported | 2.067x |
| PRO5000-B | 46,080 | 161.753 | 78.007 | 168.352 | unsupported | 2.074x |

The legacy cells execute the preserved legacy data path from the instrumented
optimization checkout. A separate exact-checkout parity run differed by only
0.13%, excluding harness instrumentation as the source of the speedup.

MegaMoE 32K/45K has no latency: CuTe rejects output-view offsets
2,147,488,800 and 3,019,904,032 as signed-INT32 overflow before the timed loop.
Latest block-scale includes its K=1280 tail guard and now passes all balanced
lengths.

## 16K unbalanced comparison

PRO5000-A milliseconds:

| trace | legacy Sonic | fused Sonic | block-scale |
|---|---:|---:|---:|
| rank max/mean 1.106 | 50.972 | 28.553 | 57.449 |
| rank max/mean 1.227 | 56.648 | 31.624 | 63.766 |
| rank max/mean 1.698 | 80.193 | 42.920 | 88.498 |
| Segment-M aligned | 122.029 | 62.244 | invalid C4 (9 bad) |
| Segment-M permuted | 57.147 | 32.538 | invalid C4 (8 bad) |
| joint rank+segment | 84.748 | 46.411 | invalid C4 (7 bad) |

PRO5000-B fused Sonic versus MegaMoE milliseconds:

| trace | fused Sonic | MegaMoE |
|---|---:|---:|
| rank max/mean 1.106 | 28.899 | 90.090 |
| rank max/mean 1.227 | 32.025 | 94.786 |
| rank max/mean 1.698 | 43.476 | 114.953 |
| Segment-M aligned | 67.538 | 148.612 |
| Segment-M permuted | 37.049 | 95.579 |
| joint rank+segment | 46.897 | 113.695 |

MegaMoE performance uses its separate strict common-trace correctness gate.
The block-scale Segment traces have tiny global relative L2 (~1.2e-5) but
still violate the exact zero-bad rule, so their measured loops are excluded.

## EPLB: when migration helps

PRO5000-B three-restart medians:

| trace | contiguous E0 | greedy E0 | migration R0 | break-even steps |
|---|---:|---:|---:|---:|
| 16K rank 1.106 | 28.899 | 41.209 | 84.14 | never |
| 16K rank 1.227 | 32.025 | 41.374 | 84.93 | never |
| 16K rank 1.698 | 43.476 | 41.380 | 86.98 | 41.49 |
| 16K Segment-aligned | 67.538 | 41.446 | 87.75 | 3.36 |
| 16K Segment-permuted | 37.049 | 41.424 | 87.75 | never |
| 16K joint skew | 46.897 | 38.451 | 72.22 | 8.55 |

The default controller therefore observes below 1.30, requires persistence
above 1.30, models deduplicated remote records, and rejects break-even beyond
the configured horizon. Replica is experimental/default-off: on the 16K
Segment hotspot it measured about 62.52 ms versus 41.45 ms for migration.

## Retained and rejected paths

Retained default:

- reusable caller-owned workspace and `_out` APIs;
- counting route metadata and direct blocked-scale scatter;
- deterministic segmented FP32 weight-multiply/reduce;
- communication-aware windowed static EPLB.

Default-off or experimental:

- CUDA Graph (0.12% K0 gain);
- heavy-expert-first (flat);
- direct FC2 atomic epilogue (FC2 regression);
- MegaMoE-style FC12 and tile-resident FC12 (failed performance bounds);
- chunked NCCL pipeline (2.38% overall cross-NUMA gain, below gate);
- NVSHMEM/IBGDA port (strict correctness passed, reference much slower);
- persistent replica (correct but slower than migration).

The retained segmented kernel fuses weight multiplication and token reduction,
but still materializes BF16 FC2 pair output. Claims of full FC2-epilogue fusion
would be inaccurate for the default path.

## Boundary and test coverage

The single-GPU Segment-M sweep uses one empty expert plus active M_e
`1,127,128,129,517,2528,16384`. All three restarts passed C4; p50 ranges from
about 0.039 ms at small segments to 0.572 ms at 16,384 rows. The targeted SM120
suite reports 19 passed and 0 failed tests.

## Reproduce Sonic

Install/check out the matching QuACK branch, externally select four GPUs with
the intended topology, and run from the Sonic repository root:

```bash
export QUACK_REPO=/path/to/quack
export PYTHONPATH="$PWD:$QUACK_REPO"
benchmark_results/sm120-mxfp8-fusion-overlap/run-formal-ep4.sh \
  results/ep4 PRO5000-A same_numa all
```

The script generates deterministic trace files, runs three restarts, performs
C4 inline, and never discovers or stores physical device IDs. Use
`benchmarks/mxfp8-segment-edge-sweep.py` for the Segment-M boundary suite.

The full machine-readable summary is in `summary.json`. Cross-NUMA formal
reruns are not substituted with same-NUMA values: an externally occupied NUMA
group blocked the current full matrix, while earlier 4K/8K diagnostic runs are
kept explicitly non-formal.
