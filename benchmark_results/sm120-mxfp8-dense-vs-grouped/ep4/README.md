# SM120 PRO5000 EP4 system measurements

This is a separate four-GPU system campaign. It is not part of the
single-GPU, kernel-only dense/grouped tables in the parent directory.

## Fixed customer shape

| parameter | value |
| --- | ---: |
| global tokens | 16,384 |
| source tokens per rank | 4,096 |
| EP / world size | 4 |
| top-k | 24 |
| hidden | 2,560 |
| post-SwiGLU intermediate | 1,024 |
| global / local experts | 768 / 192 |

Each iteration uses the maximum CUDA-event latency across the four ranks.
Each reported value is the median of three independently restarted process
P50 values.

## What the implementation labels mean

| label | value / scale format | multi-GPU path | source activation quantization |
| --- | --- | --- | --- |
| FlashInfer FP8 wrapper | E4M3 + FP32 groupwise scales; activation K128 and weight 128x128 | external deduplicated Torch/NCCL EP wrapper around one local FlashInfer grouped kernel per rank | included in the timed forward |
| Sonic-MXFP8 wrapper | OCP E4M3 + E8M0, semantic K32 | external deduplicated Torch/NCCL EP wrapper around local Sonic/QuACK grouped kernels | included in the timed forward |
| MegaMoE MXFP8 | OCP E4M3 + E8M0, semantic K32 | MegaMoE native fused multi-rank runtime | excluded; input is prequantized |

FlashInfer and Sonic are not upstream native multi-GPU APIs in this table.
MegaMoE uses a different runtime and timing boundary. The table therefore
compares complete tested data paths, not bit-equivalent GEMM kernels.

## Balanced EP4 latency

All values are milliseconds.

| node | topology | FlashInfer FP8 wrapper | Sonic-MXFP8 wrapper | MegaMoE MXFP8 p2p_direct |
| --- | --- | ---: | ---: | ---: |
| PRO5000-A | same NUMA | 18.621 | 15.746 | 19.719 |
| PRO5000-A | cross NUMA, symmetric 2+2 | — | — | — |
| PRO5000-B | same NUMA | 18.575 | 15.806 | 19.731 |
| PRO5000-B | cross NUMA, symmetric 2+2 | 18.522 | 15.667 | 28.593 |

The missing row was not completed in a clean, auditable resource window.
On the one node with both topologies, cross/same changed by −0.29% for the
FlashInfer wrapper, −0.88% for the Sonic wrapper, and +44.91% for MegaMoE
p2p_direct. These ratios describe each full data path, not the local kernel.

## Imbalance and static placement

| node / topology | FlashInfer Zipf contiguous → greedy | Sonic Zipf contiguous → static greedy | Mega power-law static → K2 reclaim |
| --- | ---: | ---: | ---: |
| PRO5000-A / same | 45.236 → 18.916 | 36.282 → 16.012 | 21.501 → 21.401 |
| PRO5000-B / same | 45.579 → 18.914 | 36.422 → 16.036 | 21.509 → 21.412 |
| PRO5000-B / cross 2+2 | 45.477 → 18.851 | 36.456 → 15.931 | 29.775 → 29.615 |

FlashInfer and Sonic use the same deterministic Zipf routing helper, although
their dtypes still differ. MegaMoE uses an independent randomized power-law
permutation, so its absolute imbalance latency must not be divided directly by
the other two.

The greedy/EPLB columns are steady-state forwards after placement. They exclude
the one-time planner and real weight/scale migration:

| node / topology | one-time Sonic MXFP8 reconfiguration | break-even in this stable trace |
| --- | ---: | ---: |
| PRO5000-A / same | 66.561 ms | 3.28 steps |
| PRO5000-B / same | 67.016 ms | 3.29 steps |
| PRO5000-B / cross 2+2 | 67.897 ms | 3.31 steps |

That one-time benchmark includes the global histogram, greedy plan, consensus,
packing, real NCCL byte transfer, shadow install, route-map switch, barrier,
and allocator release. It excludes production request drain, rollback/version
management, and checkpoint I/O.

## MegaMoE IBGDA status

The historical p2p_direct rows above are not affected by the later IBGDA
top-k-weight indexing bug.

| node / topology | code state | strict smoke | balanced / power-law | status |
| --- | --- | --- | ---: | --- |
| PRO5000-B / cross 2+2 | before the linear-view fix | about 83% of output elements failed | 33.848 / 40.904 ms | invalid diagnostic |
| PRO5000-A / cross 2+2 | after the local linear-view fix | four-rank max diff 0, bad count 0 | 33.889 / 40.777 ms | reduced-token strict gate passed |

The fixed strict gate used 128 tokens per rank with the customer hidden,
intermediate, top-k, and expert counts. The full-shape performance run skipped
its reference check and had no same-window cross-NUMA p2p_direct control on
PRO5000-A. It therefore proves the identified smoke-path bug was fixed, but
does not establish that IBGDA is faster than direct P2P.

## Independent FlashInfer PR #4660 EP4 A/B

This is a separate same-NUMA FP8 campaign using the same external EP wrapper.
It is not the aligned single-GPU MXFP8 grouped benchmark.

| node | old | PR #4660 | lower-latency gain |
| --- | ---: | ---: | ---: |
| PRO5000-A | 18.716816 ms | 18.739872 ms | −0.123% |
| PRO5000-B | 18.627232 ms | 18.593936 ms | +0.179% |

The end-to-end change is within run-to-run noise.

## Public Sonic/EPLB reproduction

The public harness records only an anonymous node label and a topology class.
It never discovers or writes a hostname, address, GPU index, UUID, or PCI ID.
The caller or scheduler must expose exactly four already-validated GPUs before
starting it.

- [mxfp8-ep4-e2e.py](../../../benchmarks/mxfp8-ep4-e2e.py) implements routing,
  deduplicated dispatch/combine, MXFP8 transport, grouped FC1/FC2, correctness
  gates, and four-rank critical-path timing.
- [run-mxfp8-ep4-suite.sh](../../../benchmarks/run-mxfp8-ep4-suite.sh) runs
  balanced, Zipf contiguous, and Zipf static-greedy cases for three process
  restarts.
- [summarize-mxfp8-ep4-suite.py](../../../benchmarks/summarize-mxfp8-ep4-suite.py)
  validates shape, scale format, correctness, anonymity, and timing protocol.
- [flashinfer-fp8-ep4-e2e.py](../../../benchmarks/flashinfer-fp8-ep4-e2e.py)
  contains the audited external FlashInfer FP8 EP wrapper used for the
  historical comparison.
- [run-flashinfer-fp8-ep4-suite.sh](../../../benchmarks/run-flashinfer-fp8-ep4-suite.sh)
  runs its balanced and Zipf contiguous/static-greedy cases. The active
  FlashInfer checkout controls whether this is the old commit or PR #4660.

From the SonicMoE repository root, after the runtime exposes four GPUs:

    NODE_LABEL=PRO5000-A TOPOLOGY_LABEL=same_numa \
      QUACK_REPO=/path/to/quack \
      bash benchmarks/run-mxfp8-ep4-suite.sh results/ep4-PRO5000-A-same

Use TOPOLOGY_LABEL=cross_numa_2plus2 only after an external topology check has
confirmed a symmetric 2+2 group. Device selection must remain outside the
published result directory.

For an exact MegaMoE checkout, first run a reference-enabled reduced-token
gate, then run the full-shape event-timed case:

    python -m torch.distributed.run --standalone --nproc-per-node=4 \
      -m moe_sm120_mxfp8_split.mega_runner \
      --num_tokens_per_rank 128 --num_topk 24 --num_total_experts 768 \
      --hidden 2560 --intermediate 2048 \
      --data_parallel_size 1 --tensor_parallel_size 1 \
      --enable_static_expert_shape --comm_backend p2p_direct \
      --route_distribution balanced

    python -m torch.distributed.run --standalone --nproc-per-node=4 \
      -m moe_sm120_mxfp8_split.mega_runner \
      --num_tokens_per_rank 4096 --num_topk 24 --num_total_experts 768 \
      --hidden 2560 --intermediate 2048 \
      --data_parallel_size 1 --tensor_parallel_size 1 \
      --enable_static_expert_shape --comm_backend p2p_direct \
      --route_distribution balanced --perf_run --skip_ref_check \
      --use_cuda_events --perf_warmup 20 --perf_iters 100

MegaMoE uses intermediate 2048 because its CLI expects the physical Gate+Up
width; Sonic/FlashInfer use the post-SwiGLU width 1024. A full-shape performance
number produced with skip_ref_check is publishable only when the corresponding
strict gate has passed.

The public scripts reproduce the Sonic/EPLB and external FlashInfer-wrapper
data paths at the customer shape. MegaMoE still requires its exact private
source checkout and native runtime. Its rows are published here as audited
historical system results, not as output claimed from either public wrapper.

## Machine-readable result

See [summary.json](summary.json). It contains only PRO5000-A/B aliases and
topology classes.
