# SM120 MXFP8 Sonic EP4 + EPLB runtime prototype

This directory documents the public benchmark/runtime prototype for the new
customer workload.  The stable path is:

`EP4 token dispatch -> Sonic variable-M MXFP8 experts -> windowed EPLB -> real equal-capacity expert migration`

It is not a framework-integrated production scheduler.  The controller replay
and the distributed data-plane benchmark are separate entry points so planning,
reconfiguration, and steady-state forward latency remain independently visible.

## Supported workload

| field | value |
| --- | --- |
| EP size | 4 |
| global experts | 512 (128 owner experts/rank) |
| top-k | 32 |
| `m` | 8,192 / 16,384 / 32,768 / 46,080 tokens per rank |
| hidden | 2,048 |
| post-SwiGLU intermediate | 1,280 |
| values/scales | OCP MXFP8 E4M3 + E8M0, semantic K-group 1x32 |

At `m=16,384`, the supplied expert-segment anchors are p50=517,
p90=2,528, and max=16,384.  The supplied EP rank max/mean points are
1.043, 1.106, 1.227, and 1.698.  The generator creates deterministic,
quantile-matched synthetic traces from those statistics.  They are not
customer route captures.

## Stability boundary

| component | status | default |
| --- | --- | --- |
| balanced/segment/rank-skew workload generator | benchmark-stable | enabled |
| equal-capacity LPT placement | benchmark-stable | opt-in with `--placement greedy` |
| real FP8 value + E8M0 scale migration | benchmark-stable | enabled by the new suite |
| byte-sample migration audit and end-to-end numerical gate | benchmark-stable | enabled |
| windowed controller, hysteresis, route-map versioning | runtime prototype | static migration only |
| hot-expert replica and static+replica hybrid | **experimental** | **disabled** |

Replica/hybrid requires an explicit `--experimental-replica-slots` flag.  The
stable suite never passes this flag.  The controller also requires
`--experimental-replica` before it can emit an experimental replica
recommendation; such a recommendation does not increment the stable route-map
version or apply a placement automatically.

## Reproduction

Generate traces only:

```bash
python benchmarks/generate-mxfp8-eplb-workloads.py \
  --output-dir results/eplb-traces
```

Run the stable suite after the caller has selected four visible SM120 GPUs:

```bash
export QUACK_REPO=/path/to/quack
export NODE_LABEL=PRO5000-A
export TOPOLOGY_LABEL=same_numa
bash benchmarks/run-mxfp8-eplb-suite.sh results/eplb-stable
```

The script never assigns Bash's reserved `GROUPS` variable.  It neither
discovers nor writes physical GPU IDs,
hostnames, PCI addresses, or network addresses.

A single real-migration case can be run with:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  benchmarks/mxfp8-ep4-e2e.py \
  --tokens 16384 --top-k 32 --experts 512 \
  --hidden 2048 --intermediate 1280 \
  --routing-trace results/eplb-traces/m16384_rank_r1.698.pt \
  --placement greedy --real-weight-migration \
  --node-label PRO5000-A --topology-label same_numa \
  --output results/eplb-one-case.jsonl
```

Replay a windowed decision sequence using measured forward and migration
latencies:

```bash
python benchmarks/replay-mxfp8-windowed-eplb.py \
  --profile p50=results/eplb-traces/m16384_rank_r1.106.json:32 \
  --profile max=results/eplb-traces/m16384_rank_r1.698.json:32 \
  --unoptimized-step-ms 79.982 \
  --optimized-step-ms 57.281 \
  --reconfiguration-ms 173.811 \
  --output results/controller-replay.json
```

The default policy uses a 16-step window and update interval, requires two
eligible windows, observes at max/mean 1.15, applies static placement at 1.30,
and limits one update to 32 experts through complete capacity-preserving
cycles.  These are starting thresholds, not universal model constants.

## Timing boundary and correctness

Steady-state `e2e_rank_max` includes source quantization, deduplicated NCCL
dispatch, compact/sort, Sonic FC1+SwiGLU+requant, FC2, local reduction, and
NCCL combine.  It excludes placement planning and reconfiguration.  The JSONL
records those separately as `placement_plan_ms` and `weight_migration_ms`.

Real migration sends both FC1/FC2 value bytes and scale bytes, creates a
target-ordered shadow bank, and only then times the new placement.  The run
must pass the independent byte-sample audit and the end-to-end placement
numerical gate.  `placement_migration_in_timed_forward` remains false because
migration is amortized across later forward steps.

## Prior PRO5000 validation

The 0901 bring-up used anonymous node/topology labels.  Selected same-NUMA
`m=16K` medians were:

| trace | contiguous | static LPT after real migration | steady-state change |
| --- | ---: | ---: | ---: |
| rank max/mean 1.106 | 50.867 ms | 60.135 ms | +18.2%; do not migrate |
| rank max/mean 1.227 | 56.482 ms | 60.302 ms | +6.8%; observe only |
| broad rank max/mean 1.698 | 79.982 ms | 57.281 ms | -28.4% |
| joint aligned max | 84.475 ms | 59.514 ms | -29.6% |
| segment aligned | 122.002 ms | 61.969 ms | -49.2% |

For the full 512-expert case, each expert carries 8,110,080 bytes across FC1,
FC2, and scales.  Moving 384 experts transferred 3,114,270,720 bytes and took
about 173.811 ms same-NUMA in that campaign; observed break-even was roughly
3-8 stable steps depending on the trace.  These historical figures motivate
the controller thresholds.  Fresh hardware comparisons must use newly
generated JSONL from this branch.

Experimental replica/hybrid preload was slower to reconfigure (roughly
179-182 ms same-NUMA and 216-227 ms cross-NUMA in bring-up) and did not beat
static placement on the tested broad-skew cases.  It remains useful research
for a persistent single-expert hotspot, but it is not part of the recommended
or default path.

## Source map

- Workloads: [`benchmarks/ep4_support/workloads.py`](../../benchmarks/ep4_support/workloads.py)
- Real migration: [`sonic_migration.py`](../../benchmarks/ep4_support/placement/sonic_migration.py)
- Windowed policy: [`windowed_controller.py`](../../benchmarks/ep4_support/placement/windowed_controller.py)
- Experimental replica/hybrid: [`experimental_sonic_replica.py`](../../benchmarks/ep4_support/placement/experimental_sonic_replica.py)
- Distributed benchmark: [`mxfp8-ep4-e2e.py`](../../benchmarks/mxfp8-ep4-e2e.py)
