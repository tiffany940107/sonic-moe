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
  directly in the blocked E8M0 layout consumed by Quack.
- FC1 fuses bias, gated activation, saved BF16 preactivation, and MXFP8
  requantization. The inference epilogue omits the saved preactivation.
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

Replace `--mxfp8-policy auto` with `--mxfp8-policy mxfp8` to benchmark
full-MXFP8 backward. The BF16 comparison is selected independently by the
`sonicmoe` backend entry.

Recorded results and the exact environment are in
`benchmark_results/sm100-mxfp8-training/README.md`.
