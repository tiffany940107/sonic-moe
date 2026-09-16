# SM100 MXFP8 EP4 results

Results were collected on 2026-09-15 on four NVIDIA B200 GPUs (physical
ordinals 3, 5, 6, and 7), NGC PyTorch 26.08, PyTorch
`2.14.0a0+4fdf77b940.nv26.08`, and NCCL 2.30.7.

Every latency is the maximum rank CUDA-event latency. Timed scope includes the
replicated router, route-plan construction, real variable-split NCCL dispatch,
local expert compute, reverse all-to-all, and combine. Training additionally
includes backward. Optimizer rows explicitly say whether an optimizer and
replicated-router gradient synchronization are included.

## Current crossover summary

| Shape `(T/rank,E,K,H,I)` | Mode | BF16 p50 | MXFP8 p50 | BF16/MXFP8 | Selection |
|---|---|---:|---:|---:|---|
| `(2048,32,4,4096,4096)` | forward, three-run mean p50 | 2.6505 ms | 2.5486 ms | 1.0400x | MXFP8 |
| `(2048,32,4,2048,2048)` | forward | 1.9811 ms | 2.1977 ms | 0.901x | BF16 |
| `(2048,32,4,4096,4096)` | training, no optimizer | 6.2949 ms | 6.6768 ms (`auto`) | 0.943x | BF16 |
| `(2048,32,4,4096,4096)` | training, no optimizer | 6.2949 ms | 6.5293 ms (full MX) | 0.964x | BF16 |
| `(512,32,4,1024,1024)` | train + router sync + SGD, three-run mean p50 | 4.7175 ms | 4.9995 ms | 0.9436x | BF16 |

The forward crossover is positive at H/I=4096. Training correctness is
complete, but backward remains the next performance frontier. These values do
not replace the single-GPU acceptance result: the paired single-GPU branch
still provides a 1.36x full training-step speedup over BF16 and the fixed public
SuperSonic comparison documented in `sm100-mxfp8-training/README.md`.

The three paired forward speedups were `1.04290x`, `1.04374x`, and
`1.03332x`. MXFP8 peak allocated memory was `1624.6 MiB`; alignment-aware
zero-material qdata reduced this by about `96 MiB` from the earlier
route-materialization workspace.

## Important ablations

- One packed MXFP8 collective is faster than separate values/scales/metadata
  collectives.
- At small `T=512,H=1024`, packed transport without route-pack was best.
- At large `T=2048,H=2048`, route-pack plus fused segmented reduction beat the
  packed-only MXFP8 path; either part in isolation did not.
- An aligned packed record can be consumed directly as pitched qdata by Quack's
  TMA `A_idx` gather. Misaligned records safely copy each received token once.
- High-priority communication-stream overlap was slower and remains off.

## Reproduction

```bash
SM100_GPUS=3,5,6,7 scripts/run_sm100_container.sh env NCCL_DEBUG=WARN \
  python -m torch.distributed.run --standalone --nproc-per-node=4 \
  /workspace/sonic-moe/benchmarks/benchmark_sm100_mxfp8_ep4.py \
  --tokens 2048 --experts 32 --top-k 4 \
  --hidden 4096 --intermediate 4096 \
  --mode forward --backend mxfp8 --warmup 10 --repeats 100
```

Replace `--backend mxfp8` with `--backend bf16` for the same-scope reference.
Use three fresh processes for an acceptance report; JSON artifacts in this
directory contain the exact resolved policies for each archived run.
