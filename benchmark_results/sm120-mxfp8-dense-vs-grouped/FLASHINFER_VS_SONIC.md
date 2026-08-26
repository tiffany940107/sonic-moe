# PRO5000 single-GPU MXFP8 grouped GEMM: Sonic vs FlashInfer

All three columns use eight balanced groups, semantic K32 UE8M0 scales, BF16 output, prequantized kernel-only timing, 20 warmups, 100 timed CUDA-event iterations, and three repeats. The displayed value is the median repeat P50.

| node | workload | Sonic (ms) | FlashInfer old (ms) | FlashInfer PR #4660 (ms) | PR vs old | Sonic speedup vs PR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PRO5000-A | mnk_8k_1280_2k | 0.093264 | 0.098160 | 0.097952 | +0.21% | 1.050x |
| PRO5000-A | mnk_8k_2k_1280 | 0.098560 | 0.108192 | 0.109344 | -1.06% | 1.109x |
| PRO5000-A | mnk_16k_1280_2k | 0.189952 | 0.201792 | 0.199792 | +0.99% | 1.052x |
| PRO5000-A | mnk_16k_2k_1280 | 0.222880 | 0.231040 | 0.232368 | -0.57% | 1.043x |
| PRO5000-A | mnk_32k_1280_2k | 0.439968 | 0.449568 | 0.450592 | -0.23% | 1.024x |
| PRO5000-A | mnk_32k_2k_1280 | 0.444048 | 0.465840 | 0.451648 | +3.05% | 1.017x |
| PRO5000-B | mnk_8k_1280_2k | 0.093776 | 0.097472 | 0.097440 | +0.03% | 1.039x |
| PRO5000-B | mnk_8k_2k_1280 | 0.098880 | 0.107488 | 0.107488 | +0.00% | 1.087x |
| PRO5000-B | mnk_16k_1280_2k | 0.188080 | 0.209984 | 0.208032 | +0.93% | 1.106x |
| PRO5000-B | mnk_16k_2k_1280 | 0.222704 | 0.237680 | 0.238528 | -0.36% | 1.071x |
| PRO5000-B | mnk_32k_1280_2k | 0.440848 | 0.457856 | 0.458896 | -0.23% | 1.041x |
| PRO5000-B | mnk_32k_2k_1280 | 0.444960 | 0.475264 | 0.474240 | +0.22% | 1.066x |

## Geometric-mean summary

- PRO5000-A: PR #4660 vs old +0.41%; Sonic vs PR #4660 1.049x; largest within-result repeat-P50 range 8.18%.
- PRO5000-B: PR #4660 vs old +0.10%; Sonic vs PR #4660 1.068x; largest within-result repeat-P50 range 8.72%.

Positive PR vs old means lower latency. Sonic speedup vs PR is FlashInfer-PR latency divided by Sonic latency, so values above one favor Sonic.

This is a single-GPU grouped-kernel comparison. It contains no EP, routing, NCCL, dispatch/combine, or NUMA traffic.

## Raw data

- PRO5000-A: [sonic](pro5000_A_suite_v2/grouped-vs-dense.json), [old](pro5000_A_suite_v2/flashinfer-old-grouped.json), [new](pro5000_A_suite_v2/flashinfer-pr4660-grouped.json)
- PRO5000-B: [sonic](pro5000_B_suite_v2/grouped-vs-dense.json), [old](pro5000_B_suite_v2/flashinfer-old-grouped.json), [new](pro5000_B_suite_v2/flashinfer-pr4660-grouped.json)
