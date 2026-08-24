<!-- ********************************************************************************
Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
******************************************************************************** -->

# SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations
[![arXiv](https://img.shields.io/badge/arXiv-2512.14080-b31b1b.svg)](https://arxiv.org/abs/2512.14080) [![PyPI](https://img.shields.io/pypi/v/sonic-moe?cache=no)](https://pypi.org/project/sonic-moe/)

**SonicMoE** is a simple but blazing-fast Mixture-of-Experts (MoE) implementation optimized for NVIDIA Hopper (SM90), Blackwell datacenter (SM100, e.g. B200/B300), and Blackwell consumer (SM120, e.g. RTX 5090) GPUs. It mainly leverages [CuTeDSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html) and [Triton](https://triton-lang.org/main/getting-started/tutorials/index.html) to deliver state-of-the-art performance through IO-aware optimizations. These 2 figures provide an overview of activation memory usage and training throughput on Hopper GPUs (H100) and Blackwell GPUs (B300). The current version of SonicMoE builds on the Grouped GEMM kernels from the [QuACK](https://github.com/Dao-AILab/quack/tree/main) library which is itself built on [CUTLASS](https://github.com/NVIDIA/cutlass).

![Activation Memory](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/mem.png)
![Training Throughput](https://raw.githubusercontent.com/Dao-AILab/sonic-moe/main/assets/tput.png)

## News

- 04/22/2026: We release a [blogpost](./assets/2026-04-22-sonicmoe-blackwell.md) on SonicMoE's activation memory-efficient and IO-aware design, and how we extend it to Blackwell GPUs through [QuACK](https://github.com/Dao-AILab/quack)'s software abstraction.

- 04/19/2026: we release SonicMoE with Blackwell (SM100) support, built on [QuACK](https://github.com/Dao-AILab/quack)'s Grouped GEMM kernels. 

## 📦 Installation

### Prerequisites

- NVIDIA Hopper GPUs (H100, H200, etc.), Blackwell datacenter GPUs (GB200, B200, B300, etc.), or Blackwell consumer GPUs (e.g. RTX 5090, SM120)
- CUDA 12.9+ (13.0+ for B300 GPUs)
- Python 3.12+ recommended
- PyTorch 2.11+ (2.12 recommended)

### Install from pip
```bash
pip install sonic-moe
```

### Install from Source

```bash
# Clone the repository
git clone https://github.com/Dao-AILab/sonic-moe.git
cd sonic-moe

# Install dependencies
pip install -r requirements.txt

# Install SonicMoE
pip install -e .
```

### Experimental SM120 MXFP8 grouped inference

The `feature/sm120-mxfp8-varlen-ep` branch adds an inference-only MXFP8
compute primitive for rows that have already been routed and grouped by local
expert. It uses OCP E4M3 values, E8M0 scales with semantic 1x32 granularity
along K, a variable-M grouped GEMM, fused SwiGLU + MXFP8 requantization for
FC1, and BF16 output for FC2. It requires the matching QuACK varlen-M
epilogue fix pinned in `requirements-mxfp8.txt`.

"DeepGEMM-aligned" here means the same semantic 1x32 quantization recipe. The
physical scale tensor uses QuACK/CUTLASS's blocked layout
`(batch, rm, rk, 32, 4, 4)` and is not a binary-compatible DeepGEMM scale
buffer.

```bash
git clone --branch feature/sm120-mxfp8-varlen-ep \
  https://github.com/tiffany940107/sonic-moe.git
cd sonic-moe
python -m pip install -r requirements-mxfp8.txt
python -m pip install --no-deps -e .
python -m pytest -q tests/test_mxfp8.py -s
```

The generic grouped-GEMM API is:

```python
from sonicmoe import (
    allocate_mxfp8_weights,
    mxfp8_grouped_gemm,
    quantize_varlen_m_operand,
)

# x_grouped is (sum(M_e), K); cu_seqlens_m is the int32 prefix sum of M_e.
x_mxfp8 = quantize_varlen_m_operand(x_grouped, cu_seqlens_m)
weight_mxfp8 = allocate_mxfp8_weights(
    num_experts, N, K, device="cuda", seed=123
)
out_bf16 = mxfp8_grouped_gemm(x_mxfp8, weight_mxfp8, cu_seqlens_m)
```

Run all six requested `(M, N, K)` workloads with eight balanced groups:

```bash
python benchmarks/mxfp8-grouped-gemm.py \
  --workloads all --groups 8 --distribution balanced \
  --warmup 10 --iterations 50 --jsonl results/mxfp8-grouped.jsonl
```

For a same-MNK acceleration comparison against eight per-expert dense launches
and a one-big-dense efficiency ceiling, run:

```bash
python benchmarks/mxfp8-dense-vs-grouped.py \
  --workloads all --groups 8 --distribution balanced \
  --warmup 20 --iterations 100 --repeats 3 \
  --node-label my-sm120-node --output results/dense-vs-grouped.json
```

The reported `speedup_grouped_vs_dense_loop` preserves MoE semantics because
both paths use the same per-expert weights. `one_big_dense` uses only one
shared weight and is therefore an efficiency ceiling, not a model-equivalent
baseline. Cold compilation, quantization, allocation, and scale packing are
excluded from all three timed paths.

The workload set is `8192/16384/32768 x 1280 x 2048` in both N/K
orientations. Here M means the total local rows across groups. Use
`--distribution ragged` to stress imbalance, or select one workload and pass
`--group-ms m0,m1,...` for an exact expert histogram. Quantization, scale
packing, allocation, and cold kernel compilation are deliberately outside the
reported kernel latency; every JSON record includes the full `M_e` list.

This extension does not change the existing BF16 `MoE.forward()` path and does
not implement routing, token permutation, expert-parallel transport, or EPLB.
Those layers should call the grouped primitive after producing local
expert-contiguous rows.

## 🎯 Quick Start

### Basic Usage

```python
import torch
from sonicmoe import MoE, KernelBackendMoE
from sonicmoe.enums import ActivationType

# Create MoE layer
moe = MoE(
    num_experts=128,                           # Number of experts
    num_experts_per_tok=8,                     # Top-k experts per token
    hidden_size=4096,                          # Hidden dimension
    intermediate_size=1536,                    # Expert intermediate size
    activation_function=ActivationType.SWIGLU, # SwiGLU activation
    add_bias=False,                            # Add bias to linear layers
    std=0.02,                                  # Weight initialization std
).to(device="cuda", dtype=torch.bfloat16)

# Forward pass
x = torch.randn(32768, 4096, device="cuda", dtype=torch.bfloat16)
output, aux_loss = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)
```

## 🧪 Testing

Run the test suite to verify correctness:

```bash
make test
```

### Example usage

- SonicMoE with TC top-K routing (softmax-over-topk, or `softmax(topk(logits))`) and interleaved weight layout format for up-proj weights
    ```bash
    python benchmarks/moe-cute.py --thiek 32768,4096,1024,128,8 --activation swiglu
    ```

- SonicMoE with Qwen3-style routing (topk-over-softmax, or `topk(softmax(logits))`) with topk probabilities renormalization and interleaved weight layout format for up-proj weights
    ```bash
    python benchmarks/moe-cute.py --thiek 32768,4096,1024,128,8 --topk_over_softmax --norm_topk_probs
    ```

- SonicMoE with token rounding routing (SwiGLU activation) and interleaved weight layout format for up-proj weights
    ```bash
    python benchmarks/moe-token-rounding.py --routing nr --thiekq 16384,4096,1024,256,8,128
    ```

- SonicMoE with concatenated weight layout format for up-proj weights

    By default, SonicMoE expects `w1` (the gated up-projection weights) in **interleaved** format: `[gate_0, up_0, gate_1, up_1, ...]`. HuggingFace models (Qwen3, Mixtral, DeepSeek, etc.) store `gate_up_proj` in **concatenated** format: `[gate_0, gate_1, ..., gate_{I-1}, up_0, up_1, ..., up_{I-1}]`.

    ```bash
    # Concatenated weight layout format with TC top-K routing
    python benchmarks/moe-cute.py --thiek 32768,4096,1024,128,8 --concat_layout
    ```


## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use SonicMoE in your research, please cite:

```bibtex
@misc{guo2025sonicmoeacceleratingmoeio,
      title={SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations}, 
      author={Wentao Guo and Mayank Mishra and Xinle Cheng and Ion Stoica and Tri Dao},
      year={2025},
      eprint={2512.14080},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.14080}, 
}
```
