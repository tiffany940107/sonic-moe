# SM100 MXFP8 training

This branch adds a single-GPU SM100 backend with MXFP8 expert forward GEMMs and
separately selectable input-gradient (dgrad) and weight-gradient (wgrad)
precision. MXFP8 operands use OCP E4M3 values and E8M0 scales. Model parameters
remain BF16 and optimizer state remains in the optimizer-selected precision.
Router logits, the final routed reduction, bias gradients, and other sensitive
reductions use BF16 or FP32 as appropriate.

## Requirements

- NVIDIA SM100 GPU; this branch was validated on an NVIDIA B200 (CC 10.0).
- CUDA 12.9 or newer.
- Python 3.12 or newer and PyTorch 2.11 or newer.
- The paired Quack commit pinned in `requirements-sm100-mxfp8.txt`.
- `hidden_size` and `intermediate_size` divisible by 128.
- SwiGLU, GEGLU, or ReGLU activation.

For a standalone checkout:

```bash
python -m pip install -r requirements-sm100-mxfp8.txt
python -m pip install -e .
```

For paired local development, install the Quack checkout first:

```bash
python -m pip install -e ../quack
python -m pip install -e .
```

## Precision policy

Set `SONICMOE_MXFP8_POLICY` before constructing the `MoE` module. The value is
read when the module creates its MXFP8 workspace; changing the environment
variable later does not reconfigure an existing module.

The two supported end-user training policies are:

| Policy | Expert forward | FC1/FC2 dgrad | FC1/FC2 wgrad | Intended use |
|---|---|---|---|---|
| `auto` (default) | MXFP8 | BF16 | BF16 | Fastest validated single-GPU policy |
| `mxfp8` | MXFP8 | MXFP8 | MXFP8 | Full expert-GEMM MXFP8 evaluation |

`auto` currently means the fixed precision assignment shown above; it is not a
runtime shape autotuner. `forward_only` is an exact alias retained for
experiments. If neither `SONICMOE_MXFP8_POLICY` nor the legacy
`SONICMOE_MXFP8_WGRAD` variable is set, the policy defaults to `auto`.

Enable the default mixed policy explicitly with:

```bash
SONICMOE_MXFP8_POLICY=auto python train.py
```

Disable `auto` but keep the MXFP8 backend, making all expert forward/backward
GEMMs MXFP8, with:

```bash
SONICMOE_MXFP8_POLICY=mxfp8 python train.py
```

To disable MXFP8 completely, select the BF16 Sonic backend in the model call:

```python
output, aux_loss = moe(
    x,
    kernel_backend_moe=KernelBackendMoE.sonicmoe,
)
```

The environment policy only affects calls using
`KernelBackendMoE.sonicmoe_mxfp8`. In particular,
`SONICMOE_MXFP8_POLICY=bf16` does **not** select the all-BF16 backend: it is an
internal ablation that keeps expert forward and dgrad in MXFP8 while moving
wgrad to BF16.

Additional per-role ablation policies are available for kernel development:

| Policy | FC1 dgrad | FC2 dgrad | FC1 wgrad | FC2 wgrad |
|---|---|---|---|---|
| `bf16` | MXFP8 | MXFP8 | BF16 | BF16 |
| `fc1_bf16` | MXFP8 | MXFP8 | BF16 | MXFP8 |
| `fc2_bf16` | MXFP8 | MXFP8 | MXFP8 | BF16 |
| `dgrad_bf16` | BF16 | BF16 | MXFP8 | MXFP8 |
| `fc1_dgrad_bf16` | BF16 | MXFP8 | BF16 | BF16 |
| `fc2_dgrad_bf16` | MXFP8 | BF16 | BF16 | BF16 |

These ablation policies are not a way to disable the MXFP8 backend. The fused
`Mxfp8SGD` optimizer accepts the symmetric `auto`/`forward_only` and `mxfp8`
policies; use a standard PyTorch optimizer for the other ablations.

## Saved preactivation policy

Training FC1 can save its full gate/up preactivation directly as E4M3 values
plus canonical blocked E8M0 scales. Backward then loads that tensor through
Quack's FP8-C DGated epilogue instead of writing and rereading a full BF16
preactivation. The two parts are one composite chain:

| Variable | Values | Default `auto` behavior |
|---|---|---|
| `SONICMOE_MXFP8_SAVE_Z_FP8` | `auto`, `0`, `1` | Enable for full `mxfp8` backward |
| `SONICMOE_MXFP8_FP8_C_DGATED` | `auto`, `0`, `1` | Enable for full `mxfp8` backward |

If either variable is `0`, FC1 saves BF16 and backward uses the original
BF16-C path. Setting both to `1` forces FP8 saved activations even when the
`auto` precision policy uses BF16 dgrad/wgrad GEMMs. This changes saved-tensor
precision, not the mainloop precision selected by the policy.

`SONICMOE_MXFP8_FP8_C_FUSE_DQUANT=1` additionally makes full-MXFP8 DGated
emit the rowwise MXFP8 dpreactivation used by FC1 dgrad. Its default is `0`
because the measured tradeoff depends on launch overhead. At
`T=1024, E=8, top-k=2, H=I=2048` it reduced eager full-step latency by about
4%, but increased CUDA-Graph full-step latency by about 1.9%. Nsight Compute
also measured 218 registers per thread for the fused DGated kernel versus 203
for the unfused kernel. Leave it at `0` for graph execution; eager users can
force `1` and validate their own shape.

Saved E4M3 values and their scales are invocation-owned, rather than reusable
workspace tensors. This supports two live forwards before backward and avoids
cross-invocation corruption. The columnwise MXFP8 input saved for FC1 wgrad
and the grouped router scores follow the same ownership rule; immediately
consumed rowwise tensors continue to use the reusable workspace. These three
variables are read when `sonicmoe.functional.mxfp8` is imported, so set them
before importing Sonic MoE.

## Routed activation storage

The precision policy above is independent of how routed MXFP8 activation
values are stored. `SONICMOE_MXFP8_ZERO_MATERIAL_GATHER` controls whether an
eligible rowwise cast materializes one FP8 row per route or retains one FP8 row
per physical token and lets the Quack GEMM gather it through `A_idx`:

| Value | Routed FP8 qdata | Selection |
|---|---|---|
| `auto` (default) | `(T, H)` for top-k >= 4; otherwise `(T * top-k, H)` | Measured crossover on B200 |
| `0` | Always `(T * top-k, H)` | Compatibility and A/B baseline |
| `1` | `(T, H)` whenever the rowwise path is eligible | Forced zero-materialization experiment |

The zero-material path quantizes each physical token once. In the same Triton
launch it uses the router inverse permutation to scatter only E8M0 scale bytes
into expert-grouped order; the much larger E4M3 qdata remains physical. The
FC1 GEMM then receives the original route gather indices. This reduces the
activation qdata footprint by `top-k` without changing routing semantics.

`auto` intentionally keeps the materialized path for top-k 1/2: at the
validated `T=1024, E=8, H=I=2048` shape, the extra indexed GEMM work did not
amortize at top-k 2. The same interleaved microbenchmark showed the routed
quantize-plus-FC1 pipeline improving from about 54.2 to 49.2 microseconds at
top-k 4 and from about 98.0 to 84.3 microseconds at top-k 8. Full-MXFP8 paths
that simultaneously create rowwise and dim-0 views may still use the fused
dual materialized cast, because the second view is required for wgrad.

The gathered FC1 mainloop uses TMA gather by default. Use
`SONICMOE_MXFP8_FC1_TMA_GATHER=0` to select its cp.async A-load ablation, or
leave it at `1` for the measured default. `SONICMOE_MXFP8_GATHER_SF_BLOCK_M`
is a developer-only scale-scatter tile control; its default is 32.

These variables are read when `sonicmoe.functional.mxfp8` is imported. Set
them in the process environment before importing Sonic MoE, for example:

```bash
SONICMOE_MXFP8_POLICY=auto \
SONICMOE_MXFP8_ZERO_MATERIAL_GATHER=auto \
SONICMOE_MXFP8_FC1_TMA_GATHER=1 \
SONICMOE_MXFP8_SAVE_Z_FP8=auto \
SONICMOE_MXFP8_FP8_C_DGATED=auto \
python train.py
```

## Training and inference

Construct `MoE` exactly as for the BF16 Sonic backend, then select the MXFP8
backend at the call site:

```python
from sonicmoe import KernelBackendMoE, MoE, Mxfp8SGD

moe = MoE(...).cuda().to(torch.bfloat16)
x = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16)

# Optional optimizer-integrated cache refresh for plain SGD. Parameters stay
# BF16; momentum and weight decay are intentionally not supported here.
optimizer = Mxfp8SGD(moe, lr=1e-3)

# Full autograd path, including input, router, expert-weight, and bias gradients.
output, aux_loss = moe(
    x,
    kernel_backend_moe=KernelBackendMoE.sonicmoe_mxfp8,
)
(output.float().square().mean() + 0.01 * aux_loss.float()).backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)

# No FC1 preactivation is materialized in inference mode.
with torch.inference_mode():
    output, aux_loss = moe(
        x,
        kernel_backend_moe=KernelBackendMoE.sonicmoe_mxfp8,
        is_inference_mode=True,
    )
```

`MoE.eval()` also selects the inference form. Calling backward after explicitly
requesting inference mode is an error.

### Experimental FP32 expert-gradient accumulation

Full-MXFP8 training with `Mxfp8SGD` can write the two expert weight gradients
directly into persistent FP32 buffers. The first backward in an accumulation
window overwrites each buffer; later backwards use Quack's TMA reduce-add GEMM.
This avoids materializing BF16 `Parameter.grad` tensors for the expert weights.
Router and bias gradients continue to use normal PyTorch accumulation.

Select the behavior before constructing `Mxfp8SGD`:

| `SONICMOE_MXFP8_TMA_WGRAD` | Behavior |
|---|---|
| `auto` (default) | Disabled; keep ordinary BF16 expert `Parameter.grad` tensors |
| `0` | Disabled explicitly |
| `1` | Enable persistent FP32 expert gradients and TMA reduce-add |

The equivalent programmatic override is
`Mxfp8SGD(moe, use_fp32_wgrad_accum=True)`. It requires
`SONICMOE_MXFP8_POLICY=mxfp8`; the default mixed policy has BF16 wgrad GEMMs and
rejects the option. While enabled, `moe.c_fc.weight.grad` and
`moe.c_proj.weight.grad` intentionally remain `None`; `Mxfp8SGD.step()` reads
the workspace-owned FP32 tensors instead. Both `optimizer.zero_grad()` and
`moe.zero_grad()` start a new logical accumulation window without clearing the
backing storage.

This path is opt-in because it is not currently a speed default. On B200 at
`T=1024, E=8, top-k=2, H=I=2048`, four accumulated microbatches plus one
optimizer step were about 5.4% slower and used about 192 MiB more peak allocated
memory than ordinary BF16 expert gradients. The feature is retained for exact
SuperSonic-style FP32 accumulation comparisons and for future shape-specific
optimization.

### Experimental SuperSonic-scope performance mode

The standard `mxfp8` policy keeps OCP 1x32 scaling and remains the correctness
default. A separate, explicit research configuration targets the public
SuperSonic-MoE local-expert benchmark. It changes quantization granularity and
must not be enabled silently in a training run:

```bash
export SONICMOE_MXFP8_DZ_ISO32=1
export SONICMOE_MXFP8_ALL_ISO32=1
export SONICMOE_MXFP8_FAST_BF16_QUANT=1
export SONICMOE_MXFP8_VARLEN_K_BLOCK_N=128
export SONICMOE_MXFP8_VARLEN_K_WARPS=1
export SONICMOE_MXFP8_FC1_CLUSTER_M=2
```

`SONICMOE_MXFP8_ALL_ISO32=1` makes the x, dout, and dpreactivation dual casts
share one FP8 value tensor and two scale layouts per 32x32 block. This is
hardware-consumable MXFP8, but it is not mathematically identical to OCP 1x32
scaling. `SONICMOE_MXFP8_DZ_ISO32=1` enables only the dpreactivation form when
`ALL_ISO32` is off. Both default to `0`.

`SONICMOE_MXFP8_FAST_BF16_QUANT=1` replaces floating-point division and clamps
with a BF16 exponent-bit RCEIL conversion and reciprocal power-of-two multiply.
It is byte-identical to Quack for every finite BF16 value, which is covered by
an exhaustive 65,536-pattern test. It defaults to `0` because non-finite blocks
retain the conservative reference path only when the flag is off. The two
varlen-K settings and FC1 cluster setting are shape-specific launch controls;
their defaults remain `64`, `4`, and `1` respectively.

On an NVIDIA B200 with 148 SMs, shape
`T=8192,E=8,K=8,H=3072,I=1536`, and the same nsys merged GPU-projection method
used by public `PFCCLab/supersonic-moe@76b4f4f`, this configuration measured
`2630.2 us` per forward+backward iteration. The fixed public commit records
`2659.8 us`, so the measured lead is about 1.13%. Peak allocated memory was
about 3466 MiB. CUDA-event latency is diagnostic only here because it also
includes host launch gaps; the comparison uses merged GPU busy intervals.

The comparison uses the same GPU class/SM count, shape, local-expert scope, hot
weight cache, FP32 wgrad accumulation, iteration count, and GPU-projection
calculation. The public program itself was not re-executed in this container
because it requires its Paddle Torch-proxy runtime, which is not installed;
`2659.8 us` is therefore the fixed commit's recorded result, not a new local
measurement. Treat this mode as an experimental performance frontier until a
long-run convergence audit is available.

## Implementation notes

- Routed gather and rowwise MXFP8 quantization are fused, and scales are written
  directly in the blocked E8M0 layout consumed by Quack. Eligible top-k >= 4
  rowwise paths keep qdata in physical-token order and scatter only scales.
- FC1 fuses bias, gated activation, and MXFP8 quantization. It can save either
  BF16 preactivation or an invocation-owned E4M3/E8M0 pair. The inference
  epilogue omits the saved preactivation.
- Segmented K-axis casts restart scale groups at every expert boundary, so empty
  experts and non-aligned expert token counts are supported without scale-group
  leakage.
- Reusable scratch is isolated per CUDA stream. Quantized weights are cached by
  parameter storage/version and refreshed after an optimizer update.
- Runtime GEMM autotuning is disabled by default because it regressed the tested
  dynamic-M workloads. Set `SONICMOE_MXFP8_AUTOTUNE=1` only for controlled
  experiments.

The backend preserves exact top-k routing and does not apply capacity-based
token dropping.

## Validation

From the paired workspace root, using the supplied NGC-container wrapper:

```bash
./scripts/run_sm100_container.sh python -m pytest -q \
  sonic-moe/tests/mxfp8_quant_test.py \
  sonic-moe/tests/mxfp8_training_test.py

./scripts/run_sm100_container.sh python -m pytest -q sonic-moe/tests
```

The MXFP8 tests compare outputs and all gradients with the official Torch
reference, and cover bias/no-bias, optimizer steps, all supported activations,
top-k 1/2/4, empty experts, non-aligned expert-M, exact quantized bytes/scales,
workspace reuse, and weight-cache invalidation.

## Benchmarking

The benchmark reports p50/p95/p99 latency and peak allocated memory for both
the official BF16 Sonic backend and this MXFP8 backend:

```bash
python benchmarks/benchmark_sm100_mxfp8.py \
  --mode training --optimizer-step \
  --grad-accum-steps 1 --tma-wgrad auto \
  --tokens 1024 --experts 8 --top-k 2 \
  --hidden 2048 --intermediate 2048 \
  --mxfp8-policy auto \
  --backends sonicmoe sonicmoe_mxfp8_fused_sgd \
  --interleave --warmup 30 --repeats 200
```

Add `--save-z-fp8 1 --fp8-c-dgated 1` to force the FP8 saved-activation
chain, or pass both as `0` for the BF16-C A/B baseline. For full-MXFP8, add
`--fp8-c-fuse-dquant 1` to measure the heavier fused backward epilogue. The
JSON output records all three resolved settings.

Use `--grad-accum-steps N` to time `N` forward/backward microbatches before
the optional optimizer step. With `--mxfp8-policy mxfp8` and the fused SGD
backend, `--tma-wgrad 1` forces the experimental FP32/TMA accumulator;
`--tma-wgrad 0` is its BF16-`Parameter.grad` A/B baseline. The default
`--tma-wgrad auto` is currently equivalent to `0`.

Replace `--mxfp8-policy auto` with `--mxfp8-policy mxfp8` to benchmark
full-MXFP8 backward. The BF16 comparison is selected independently by the
`sonicmoe` backend entry.

To isolate routed quantization, gather mainloops, and their combined FC1
pipeline, run the backlog/interleaved microbenchmark:

```bash
python benchmarks/benchmark_mxfp8_gather.py \
  --tokens 1024 --experts 8 --top-k 4 \
  --hidden 2048 --intermediate 2048 \
  --warmup 10 --trials 20 --inner 50
```

Recorded results and the exact environment are in
`benchmark_results/sm100-mxfp8-training/README.md`.

For the fixed SuperSonic local-expert scope, run:

```bash
SONICMOE_MXFP8_DZ_ISO32=1 \
SONICMOE_MXFP8_ALL_ISO32=1 \
SONICMOE_MXFP8_FAST_BF16_QUANT=1 \
SONICMOE_MXFP8_VARLEN_K_BLOCK_N=128 \
SONICMOE_MXFP8_VARLEN_K_WARPS=1 \
SONICMOE_MXFP8_FC1_CLUSTER_M=2 \
python benchmarks/benchmark_sm100_supersonic_scope.py \
  --warmup 8 --iterations 12
```

The benchmark starts from fixed dispatched routes and includes metadata,
expert forward/backward, input and router-score gradients, and expert wgrad.
It excludes router projection/top-k, auxiliary loss, optimizer, and
communication. Its JSON output records the resolved experimental settings.
