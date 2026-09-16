# SM100 MXFP8 expert parallelism (EP4)

This branch adds exact single-node, four-GPU expert parallelism on top of the
frozen SM100 single-GPU MXFP8 training branch. The router is replicated and
expert parameters are sharded. Dispatch, local expert compute, reverse combine,
input gradients, router-score gradients, expert gradients, and empty-rank
gradients all participate in autograd.

The implementation was validated on four NVIDIA B200 GPUs in
`nvcr.io/nvidia/pytorch:26.08-py3`, with PyTorch
`2.14.0a0+4fdf77b940.nv26.08`, NCCL 2.30.7, and the Quack commit pinned in
`requirements-sm100-mxfp8.txt`.

## Data flow

For every source rank, the module performs:

1. replicated router projection, softmax, and exact top-k;
2. token deduplication per destination rank;
3. one source-side E4M3/E8M0 activation quantization;
4. one packed variable-split NCCL all-to-all containing qdata, scales, local
   expert IDs, and differentiable router scores;
5. counting/histogram route-pack without argsort;
6. local variable-M FC1/SwiGLU/FC2 expert compute;
7. segmented weighted reduction per received token;
8. one reverse all-to-all and source-token combine.

The packed transport preserves fixed-width top-k records. `-1` expert IDs are
masked slots and are not dispatched or counted. Top-1/2/4, remote-only routes,
single-hot routing, empty destination ranks, rank skew, and segment skew are
covered by tests or benchmark scenarios.

## Construction and precision selection

Initialize `torch.distributed` before constructing the module. Global expert
count must be divisible by the EP group size; hidden and intermediate sizes
must be divisible by 128.

```python
import os
import torch
import torch.distributed as dist
from sonicmoe import ExpertParallelMoE, KernelBackendMoE, Mxfp8SGD
from sonicmoe.enums import ActivationType

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = ExpertParallelMoE(
    num_experts=32,
    num_experts_per_tok=4,
    hidden_size=4096,
    intermediate_size=4096,
    activation_function=ActivationType.SWIGLU,
    add_bias=False,
    std=0.02,
    expert_backend=KernelBackendMoE.sonicmoe_mxfp8,
).cuda().to(torch.bfloat16)
```

`KernelBackendMoE.sonicmoe_mxfp8` selects MXFP8 activation transport and the
single-GPU MXFP8 local-expert implementation. The existing
`SONICMOE_MXFP8_POLICY` then controls expert backward precision:

- `auto` (default): MXFP8 forward with the measured BF16 dgrad/wgrad policy;
- `mxfp8`: MXFP8 forward, dgrad, and wgrad;
- `KernelBackendMoE.sonicmoe`: the independent all-BF16 EP reference.

The BF16 reference transports BF16 activation rows and uses the official Sonic
BF16 local expert kernels. It is intentionally retained for correctness and
shape-specific performance fallback.

## Adaptive fusion and fallback controls

Constructor arguments and their defaults are:

| Argument | Default | Behavior |
|---|---:|---|
| `use_fused_transport` | `True` | Pack values, scales, IDs, and scores into one all-to-all |
| `use_fused_route_pack` | `None` | Auto-enable at the route-pack element threshold |
| `use_fused_reduce` | `None` | Follow route-pack and use segmented weighted reduce |
| `route_pack_min_elements` | `2_000_000` | Source activation elements required by auto |
| `use_zero_material_qdata` | `None` | Auto-enable for top-k >= 4 when route-pack is active |
| `use_comm_overlap` | `False` | Optional high-priority communication stream |

The route threshold can be set before model construction with
`SONICMOE_MXFP8_EP_ROUTE_PACK_MIN_ELEMENTS`. Zero-material qdata can be forced
with `SONICMOE_MXFP8_EP_ZERO_MATERIAL_QDATA=1`, disabled with `=0`, or left at
`auto`.

Zero-material mode keeps one physical qdata row per received token and uses
Quack's `A_idx` gather for all local routes. When the packed transport record
has a 16-byte-aligned row pitch, FC1 reads the NCCL receive allocation directly
with no qdata copy. Otherwise it makes one contiguous copy per received token.
The compatibility path materializes one qdata row per valid route. E8M0 scales
remain expert-grouped because their size is about 1/32 of qdata.

Small `T=512,H=1024,K=4` measurements favored packed transport without
route-pack. Larger `T=2048,H=2048/4096,K=4` measurements favored the combined
route-pack plus fused reduction. Enabling route-pack without its fused reduction
was slower, which is why the two auto policies move together.

Communication-stream overlap remains opt-in. It was correct but slower for the
measured single-node workload (`5.43 ms` versus about `5.27 ms` in an earlier
large-shape training A/B), so it is not silently selected.

## Training and optimizer lifecycle

The replicated router receives a local gradient on every rank. Synchronize it
before an optimizer step when all ranks must retain identical router weights:

```python
output, aux_loss = model(x)
loss = output.float().square().mean() + 0.01 * aux_loss.float()
loss.backward()
model.all_reduce_replicated_gradients_(average=True)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

Plain `torch.optim.SGD` and the specialized `Mxfp8SGD` are both supported.
`Mxfp8SGD` can fuse local BF16 expert updates with MXFP8 cache refresh, but is
an explicit benchmark choice rather than the EP4 default because the crossover
depends strongly on expert weight size.

`set_expert_placement_(placement)` is a collective step-boundary operation. It
verifies that every rank proposes the same mapping and version, gathers current
BF16 expert masters, installs the new local shards, increments
`placement_version`, and invalidates MXFP8 workspaces. Parameter objects stay
stable, but per-expert optimizer state is not migrated; rebuild any stateful
optimizer after placement changes.

## Correctness commands

From the paired workspace root, with physical GPUs 3, 5, 6, and 7:

```bash
SM100_GPUS=3 scripts/run_sm100_container.sh \
  python -m pytest -q \
  /workspace/sonic-moe/tests/mxfp8_transport_test.py \
  /workspace/sonic-moe/tests/mxfp8_route_pack_test.py \
  /workspace/sonic-moe/tests/mxfp8_weighted_reduce_test.py

SM100_GPUS=3,5,6,7 scripts/run_sm100_container.sh env NCCL_DEBUG=WARN \
  python -m torch.distributed.run --standalone --nproc-per-node=4 \
  --module pytest -q /workspace/sonic-moe/tests/ep4_test.py
```

The distributed suite compares EP4 with a full EP1 model for inference,
forward/backward, every parameter gradient, Top-1/2/4, biases, empty ranks,
optimizer steps, replicated-router synchronization, masked records, remote-only
transport, placement migration, and auto-policy crossover.

## Benchmarking

The benchmark times the maximum rank latency and includes router, route-plan
construction, packed dispatch, actual NCCL all-to-all, local experts, reverse
all-to-all, and combine. Training includes backward; `--optimizer-step` also
includes replicated-router gradient synchronization and the selected optimizer.

```bash
SM100_GPUS=3,5,6,7 scripts/run_sm100_container.sh env NCCL_DEBUG=WARN \
  python -m torch.distributed.run --standalone --nproc-per-node=4 \
  /workspace/sonic-moe/benchmarks/benchmark_sm100_mxfp8_ep4.py \
  --tokens 2048 --experts 32 --top-k 4 \
  --hidden 4096 --intermediate 4096 \
  --backend mxfp8 --mode forward --warmup 10 --repeats 100
```

Use `--backend bf16` for the reference. The benchmark also exposes:

- `--scenario balanced|rank_skew|segment_skew|single_hot`;
- `--force-route-pack` / `--disable-route-pack`;
- `--force-fused-reduce` / `--disable-fused-reduce`;
- `--force-zero-material-qdata` / `--materialize-route-qdata`;
- `--comm-overlap`;
- `--mode training --optimizer-step --optimizer torch_sgd|mxfp8_sgd`.

On B200, three fresh 100-repeat balanced forward runs at
`T/rank=2048,E=32,K=4,H=I=4096` measured mean MXFP8 p50 `2.5486 ms` versus
BF16 `2.6505 ms`, or about `1.0400x`. The paired run ratios were `1.0429x`,
`1.0437x`, and `1.0333x`. At `H=I=2048`, MXFP8 p50 was about
`2.1977 ms` versus BF16 `1.9811 ms`, so BF16 remains the fallback for that
shape. Training at `H=I=4096` was also still slower: MXFP8 `auto` measured
about `6.6768 ms`, full `mxfp8` about `6.5293 ms`, and BF16 about
`6.2949 ms`. A separate three-process complete optimizer step at
`T/rank=512,H=I=1024` measured MXFP8 `4.9995 ms` versus BF16 `4.7175 ms`.
These are crossover data, not a claim that MXFP8 wins every EP shape. Final
archived results are in `benchmark_results/sm100-mxfp8-ep4/`.

The paired single-GPU branch remains the accepted complete-training speed path:
it reaches about `1.36x` over BF16 at its validated full-step shape and its
experimental fixed-scope GPU projection is about 1.13% faster than the public
SuperSonic-MoE commit recorded in the single-GPU guide. Public SuperSonic's
multicard smoke program does not implement token dispatch or EP combine, so it
is not used as an EP4 performance comparator.

## Known limitations

- Single-node NCCL EP4 is validated; multi-node DeepEP integration is not yet
  provided.
- Experts are evenly sharded. Placement may be arbitrary, but every rank owns
  the same expert count.
- The current dynamic route plan performs host-visible split-count exchange and
  allocates PyTorch metadata tensors per invocation. Persistent fixed-capacity
  graph capture remains future work.
- Chunked communication/compute overlap was evaluated but not enabled because
  the tested stream-overlap form regressed latency.
- Stateful optimizer data is not migrated by expert placement changes.
- MXFP8 EP training is numerically validated but does not yet beat BF16 for all
  measured training shapes; select the backend by workload rather than assuming
  a universal crossover.

## Attribution

The zero-material `A_idx`, source-quantized transport, and metadata design was
informed by public Apache-2.0 work in
`PFCCLab/supersonic-moe@76b4f4f8c37e6f71bfac8fe9e85dd02005d59d8a`.
This implementation is native to Sonic MoE and Quack: it does not copy Paddle
wrappers, monkey-patch Quack, or treat SuperSonic's local-MLP multicard smoke as
an expert-parallel implementation.
