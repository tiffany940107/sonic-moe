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
while register pressure and the shape crossover are being characterized.

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
