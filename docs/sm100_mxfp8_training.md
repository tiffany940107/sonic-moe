# SM100 MXFP8 training

This branch adds a single-GPU SM100 backend that runs every expert GEMM with
OCP MXFP8 E4M3 values and E8M0 scales. Model parameters remain BF16 and
optimizer state remains in the optimizer-selected precision. Router logits,
the final routed reduction, bias gradients, and other sensitive reductions use
BF16 or FP32 as appropriate.

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

## Training and inference

Construct `MoE` exactly as for the BF16 Sonic backend, then select the MXFP8
backend at the call site:

```python
from sonicmoe import KernelBackendMoE, MoE

moe = MoE(...).cuda().to(torch.bfloat16)
x = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16)

# Full autograd path, including input, router, expert-weight, and bias gradients.
output, aux_loss = moe(
    x,
    kernel_backend_moe=KernelBackendMoE.sonicmoe_mxfp8,
)
(output.float().square().mean() + 0.01 * aux_loss.float()).backward()

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
  --mode forward --tokens 2048 --experts 8 --top-k 2 \
  --hidden 4096 --intermediate 4096 --warmup 5 --repeats 20
```

Recorded results and the exact environment are in
`benchmark_results/sm100-mxfp8-training/README.md`.
