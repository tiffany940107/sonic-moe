# SM120 MXFP8 EP4 Fusion, EPLB, and Hot-Expert Results

This package records the 0902 optimization campaign on isolated branches:

- Sonic `feature/sm120-mxfp8-fusion-overlap`, source `31ab794`;
- QuACK `feature/sm120-mxfp8-fused-route-reduce`, source `d38dd36`;
- immutable Sonic/QuACK comparison points `013f1726` / `c87c9d1`.

No baseline branch was overwritten. Node names are public aliases; this package
contains no address, hostname, GPU ordinal/UUID, or PCI identifier.

## Workload and protocol

- EP=4, 512 global/128 local experts, top-k=32;
- m/rank = 8,192, 16,384, 32,768, or 46,080;
- H=2,048, post-SwiGLU I=1,280, physical Gate+Up=2,560;
- OCP MXFP8 E4M3 values, E8M0 K32 scales, BF16 reduced output;
- E0 covers exact-route dispatch, FC1/SwiGLU/requant/FC2, FP32 weighted
  reduction, and combine; source quantization and control-plane work are out;
- 20 warmups, 100 samples, three independent process restarts;
- four-rank maximum CUDA-event interval; median of restart percentiles;
- C4 requires `bad_count=0` under
  `abs(actual-reference) > 0.05 + 0.05*abs(reference)` and relative L2 <=5e-3.

## Balanced same-NUMA E0 p50

Milliseconds:

| node | m/rank | legacy Sonic | final Sonic | block-scale | MegaMoE |
|---|---:|---:|---:|---:|---:|
| PRO5000-A | 8,192 | 23.269 | 13.051 | 26.082 | 43.301 |
| PRO5000-A | 16,384 | 46.649 | 25.904 | 51.938 | 85.931 |
| PRO5000-A | 32,768 | 104.111 | 51.878 | 109.399 | unsupported |
| PRO5000-A | 46,080 | 147.171 | 73.035 | 156.008 | unsupported |
| PRO5000-B | 8,192 | 25.353 | 13.908 | 29.003 | 43.458 |
| PRO5000-B | 16,384 | 50.680 | 27.721 | 57.545 | 86.434 |
| PRO5000-B | 32,768 | 114.697 | 55.380 | 118.978 | unsupported |
| PRO5000-B | 46,080 | 161.753 | 77.819 | 168.352 | unsupported |

Final Sonic is 1.78-2.08x faster than the legacy data path. MegaMoE 32K/45K
has no latency because CuTe rejects signed-INT32 output-view extents before the
timed loop. The MegaMoE p95/p99 patch changes reporting only.

## Isolated persistent hotspot

At 16K/rank, rank max/mean is only 1.088 while expert 0 is 16.0x the expert
mean:

| path | p50 ms | one-time R0 | result |
|---|---:|---:|---|
| Sonic contiguous | 33.900 | — | reference |
| Sonic whole-expert greedy | 40.579 | 96.62 ms | regression; moves the bottleneck |
| Sonic replica(1) | 30.445 | 359.42 ms | 1.113x; ~104-step break-even |
| MegaMoE | 89.387 | — | same trace hash |
| block-scale | invalid | — | 9 C4 bad elements, no admitted latency |

The controller treats persistent expert heat independently from aggregate rank
skew. A 64-step horizon rejects the replica; a 160-step horizon recommends it
after persistence. Replica is explicit/experimental and never silently
changes the stable route-map version.

## Retained policy

Default: caller-owned local/EP transport workspaces, bounded route metadata,
direct blocked-scale scatter, deterministic segmented FP32 weighted reduce,
and selective communication-aware static EPLB. The final timed path has zero
caching-allocator deltas; NSYS launches fall 79 -> 32 -> 21 per rank/step.

Default-off/experimental: CUDA Graph (flat formal K0), heavy-first, direct FC2
atomic epilogue, FC12, chunked pipeline, NVSHMEM/IBGDA, and replicas. FC12 was
stopped at a pre-port gate; this package does not claim a Sonic FC12 kernel.

## Reproduce Sonic

Install/check out the matching QuACK branch, externally select the intended
GPU topology, and run from the Sonic repository root:

```bash
export QUACK_REPO=/path/to/quack
export PYTHONPATH="$PWD:$QUACK_REPO"
benchmark_results/sm120-mxfp8-fusion-overlap/run-formal-ep4.sh \
  results/ep4 PRO5000-A same_numa all
```

For one externally selected SM120 GPU, K0/K1 is:

```bash
benchmark_results/sm120-mxfp8-fusion-overlap/run-local-k0-k1.sh \
  results/local PRO5000-A
```

Both entry points refuse to overwrite an existing result directory and never
discover/store physical device identifiers.

For cross-NUMA, wrap any launcher with the privacy-safe capacity gate. It
selects two devices from each NUMA group internally, exports the selection only
to the child process, and exits 75 without launching when capacity is below the
threshold:

```bash
benchmark_results/sm120-mxfp8-fusion-overlap/run-cross-numa-when-ready.sh \
  60000 benchmark_results/sm120-mxfp8-fusion-overlap/run-formal-ep4.sh \
  results/cross PRO5000-A cross_numa_2plus2 all
```

The block-scale old/fixed K=1280 canary and sanitizer A/B is self-contained:

```bash
benchmark_results/sm120-mxfp8-fusion-overlap/run-blockscale-k-tail-ab.sh \
  /path/to/new-empty-work-directory
```

For the full block-scale common-trace E0 comparison, set `BLOCKSCALE_REPO` and
`BLOCKSCALE_LIB` to built commit `06a2b24`, then invoke
`run-blockscale-common-trace.sh` once for each restart. The included Python
wrapper fixes semantic scale granularity at K32 and applies C4 before writing
latency.

For an authorized checkout of MegaMoE `8512aed`, first apply the two included
patches, externally configure the requested transport/topology, then run each
restart with `run-megamoe-common-trace.sh`. This launcher reproduces the E0
performance boundary and p95/p99 parser; `--skip_ref_check` means its latency
must still be paired with the separately recorded strict common-trace
correctness gate before admission.

## Files

- `summary.json`: all admitted groups, restart p50/p95/p99, memory, migration,
  allocator, trace hashes, and explicit invalid/unsupported statuses;
- `completion-audit.json`: fail-closed 49-check result;
- `formal-results.md`, `final-report.md`, `completion-audit.md`, and
  `correctness-audit.md`: readable evidence and limitations;
- `profile-summary.json`: privacy-safe NSYS/NCU aggregates;
- `megamoe-common-routing-trace.patch` and
  `megamoe-p95-p99-reporting.patch`: common-trace adapter and reporting-only diff;
- `run-formal-ep4.sh`, `run-local-k0-k1.sh`, and
  `run-megamoe-common-trace.sh`: public benchmark entry points;
- `run-cross-numa-when-ready.sh`: topology selection and fail-closed capacity
  gate without persisting physical identifiers;
- `run-blockscale-common-trace.sh`, `bench-blockscale-mxfp8-ep.py`, and
  `moe0902/`: block-scale common-trace entry point and adapter.
- `run-blockscale-k-tail-ab.sh` plus its Python/CUDA sources: old/fixed
  block-scale K-tail reproducer.

Formal symmetric cross-NUMA 2+2 remains externally resource-blocked on both
shared nodes. No same-NUMA value or failed-attempt timing is substituted.
