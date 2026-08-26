# PRO5000 单卡 SM120 MXFP8 Dense 与 Sonic Grouped GEMM 测试记录

## 1. 测试范围

本报告整理 PRO5000-A 和 PRO5000-B 上的单卡 MXFP8 kernel-only 历史结果。
两个节点分别独立运行，每次测试只使用一张 SM120 GPU；A、B 的数字不是多机或多卡
联合结果。

测试路径如下：

- Dense：QuACK SM120 block-scaled `identity_epi.gemm`。
- Grouped：SonicMoE `mxfp8_grouped_gemm`，单卡内 8 个均衡 expert group。
- `dense_loop` 对照：在同一张 GPU 上依次启动 8 个使用不同 expert weight 的
  dense GEMM，保持与 grouped-MoE 相同的计算语义。

本测试不包含 EP、NCCL、跨 NUMA、top-k routing、dispatch/combine 或多卡通信。
它也不包含 FlashInfer 数据；这些结果应作为后续 FlashInfer groupwise-MoE 同 shape
A/B 的 Sonic 基线，不能标记成 FlashInfer 与 Sonic 的既有对比。

## 2. 数据格式与计时边界

- 数据类型：OCP MXFP8 E4M3 value + E8M0 scale，BF16 输出。
- 量化粒度：K 方向语义 `1x32`。
- Dense：20 次 warmup、100 次 timed iterations、3 个 repeat。
- Grouped：8 个 balanced groups，20 次 warmup、100 次 timed iterations、
  3 个 repeat。
- 每次调用用 CUDA event 计时；表中结果为三个 repeat 各自 p50 的中位数。
- 输入与权重在计时前已量化；allocation、量化、scale packing、正确性检查和
  cold JIT 均不计入 latency。
- Dense 与 grouped 两张表的 MNK 不相同，不能直接用第一张表的 latency 除以第二张表。
  Grouped 的同 MNK 加速比必须使用原始结果中的 `dense_loop` 字段。

有效结果使用：

- SonicMoE commit：`f0336561d0287eabd4695506826dc98d45650156`
- QuACK commit：`c87c9d1d100b4a2cc56f90099c0b9001d9ae645e`

## 3. Dense MXFP8 latency

单位为毫秒，均为单卡 kernel-only p50。

| Workload | PRO5000-A | PRO5000-B |
|---|---:|---:|
| `mnk_8k4k4k` | 0.574000 | 0.569408 |
| `mnk_16k4k4k` | 1.320544 | 1.313200 |
| `mnk_32k4k4k` | 2.697824 | 2.698208 |

## 4. Sonic MXFP8 grouped-MoE latency

`M` 是所有 group 的总行数；8 个均衡 group 因而各自处理 `M/8` 行。

| Workload | PRO5000-A | PRO5000-B |
|---|---:|---:|
| `mnk_8k_1280_2k` | 0.093264 | 0.093776 |
| `mnk_8k_2k_1280` | 0.098560 | 0.098880 |
| `mnk_16k_1280_2k` | 0.189952 | 0.188080 |
| `mnk_16k_2k_1280` | 0.222880 | 0.222704 |
| `mnk_32k_1280_2k` | 0.439968 | 0.440848 |
| `mnk_32k_2k_1280` | 0.444048 | 0.444960 |

## 5. 正确性

- 所有 dense 输出均为有限值，对反量化 FP32 reference 的 relative L2 不超过
  `0.005`。
- 所有 grouped 输出与 8 次 `dense_loop` 完全一致：`max_abs = 0`、
  `relative_l2 = 0`。
- 早期 wrapper 曾误用 Bash 内置变量 `GROUPS`，导致 group 数被用户组 ID 覆盖。
  最终 `suite_v2` 已改用 `NUM_GROUPS` 并全部重新测试；汇总器也强制验证
  `config.groups == 8`。错误的早期结果不在本报告中。

## 6. 测试与汇总脚本

- [`mxfp8-dense-gemm.py`](../../benchmarks/mxfp8-dense-gemm.py)：运行三个 dense
  workload，并对反量化 reference 做正确性检查。
- [`mxfp8-dense-vs-grouped.py`](../../benchmarks/mxfp8-dense-vs-grouped.py)：在同一
  MNK 下测 `grouped`、`dense_loop` 和 `one_big_dense`。
- [`run-mxfp8-customer-suite.sh`](../../benchmarks/run-mxfp8-customer-suite.sh)：依次
  运行完整 dense/grouped workload suite。
- [`summarize-mxfp8-customer-suite.py`](../../benchmarks/summarize-mxfp8-customer-suite.py)：
  检查 shape、8-group 配置、量化粒度与正确性，再生成 Markdown 汇总。

从 SonicMoE repository 根目录运行单个节点：

```bash
NODE_LABEL=PRO5000-A NUM_GROUPS=8 \
  bash benchmarks/run-mxfp8-customer-suite.sh \
  results/mxfp8-customer-suite/PRO5000-A
```

运行前应通过进程环境只暴露计划测试的一张空闲 GPU。`NODE_LABEL` 必须使用可公开的
匿名别名，不要把 hostname、IP、GPU UUID、PCI bus ID 或物理 device ordinal 写入
结果。

## 7. FlashInfer 能否按同等配置测试

可以。FlashInfer PR #4660 的
[`moe_gemm_mxfp8_nt_groupwise`](https://github.com/flashinfer-ai/flashinfer/blob/ff22228d2fa144e9ac6a0d841f2e9ba767ba0f0a/flashinfer/grouped_mm/cute_sm120_mxfp8_groupwise/core.py#L207)
是单卡 grouped GEMM API，本身不要求 EP 或多卡通信。多卡 EP 只是它外层的
routing、dispatch 和 combine 层。

六个 grouped workload 可以与 Sonic 完全对齐：

- `num_experts/groups = 8`，每个 group 使用不同的 weight。
- balanced `m_indptr`：M=8K/16K/32K 时，每组分别为 1024/2048/4096 行。
- `scale_granularity_mnk=(1, 1, 32)`，BF16 输出。
- plain grouped GEMM，设置 `is_gated=False`。
- 输入预先量化并按 expert 连续排列；只统计一次 FlashInfer grouped kernel launch。
- 使用相同的 20 warmup、100 timed iterations、3 repeats 和 CUDA-event p50 规则。

三个 dense workload 也可以把 FlashInfer groupwise API 设置成 `num_experts=1` 来测，
数学 MNK 与 dense GEMM 相同。但该结果必须标记为“FlashInfer groupwise E=1”，它不是
FlashInfer 专用 dense GEMM API。若目标是比较真正的 dense kernel，应另外加入
FlashInfer dense MXFP8 API；不能把 E=1 groupwise 的性能直接称为 FlashInfer dense。

## 8. 原始数据

- PRO5000-A：[dense](pro5000_A_suite_v2/dense.json)、
  [grouped-vs-dense](pro5000_A_suite_v2/grouped-vs-dense.json)
- PRO5000-B：[dense](pro5000_B_suite_v2/dense.json)、
  [grouped-vs-dense](pro5000_B_suite_v2/grouped-vs-dense.json)
- Wrapper 问题与最终重测审计：[AUDIT.md](AUDIT.md)

FlashInfer 旧版与 PR #4660 已按上述约束完成补测，结果见下一节。

## 9. FlashInfer MXFP8 与 Sonic 同粒度单卡对照

FlashInfer 补测使用真正的 per-N-row K32 权重量化，不是把 N32×K32 block scale
广播成 per-row 后再声称语义对齐。三列均为 8 个 balanced group、预量化
kernel-only、BF16 输出、20 warmup、100 iterations、3 repeats，并取 repeat
P50 的中位数。

| 节点 | Workload | Sonic | FlashInfer 旧版 | FlashInfer PR #4660 | PR 相对旧版 |
| --- | --- | ---: | ---: | ---: | ---: |
| PRO5000-A | mnk_8k_1280_2k | 0.093264 | 0.098160 | 0.097952 | +0.21% |
| PRO5000-A | mnk_8k_2k_1280 | 0.098560 | 0.108192 | 0.109344 | −1.06% |
| PRO5000-A | mnk_16k_1280_2k | 0.189952 | 0.201792 | 0.199792 | +0.99% |
| PRO5000-A | mnk_16k_2k_1280 | 0.222880 | 0.231040 | 0.232368 | −0.57% |
| PRO5000-A | mnk_32k_1280_2k | 0.439968 | 0.449568 | 0.450592 | −0.23% |
| PRO5000-A | mnk_32k_2k_1280 | 0.444048 | 0.465840 | 0.451648 | +3.05% |
| PRO5000-B | mnk_8k_1280_2k | 0.093776 | 0.097472 | 0.097440 | +0.03% |
| PRO5000-B | mnk_8k_2k_1280 | 0.098880 | 0.107488 | 0.107488 | +0.00% |
| PRO5000-B | mnk_16k_1280_2k | 0.188080 | 0.209984 | 0.208032 | +0.93% |
| PRO5000-B | mnk_16k_2k_1280 | 0.222704 | 0.237680 | 0.238528 | −0.36% |
| PRO5000-B | mnk_32k_1280_2k | 0.440848 | 0.457856 | 0.458896 | −0.23% |
| PRO5000-B | mnk_32k_2k_1280 | 0.444960 | 0.475264 | 0.474240 | +0.22% |

正值表示 PR latency 更低。六 shape 几何平均上，PR 相对旧版仅改善
PRO5000-A 的 0.41% 和 PRO5000-B 的 0.10%，整体属于基本持平。唯一较明显的
单点是 PRO5000-A 的 mnk_32k_2k_1280（+3.05%），但同一 shape 在
PRO5000-B 只有 +0.22%，因此不能概括成稳定的 3% 普遍收益。
两节点单个结果内部的三次 repeat-P50 最大极差分别达到 8.18% 和 8.72%，也说明
这些小幅新旧差值不能脱离 repeat 波动单独解释。

Sonic 相对 PR #4660 的六 shape 几何平均加速比分别为 1.049x 和 1.068x。
所有 FlashInfer workload 都通过逐 expert 的反量化 BF16 reference 检查；
relative L2 约为 3.0e-6 到 4.6e-6。

- [FlashInfer 测试脚本](../../benchmarks/flashinfer-mxfp8-grouped-gemm.py)
- [机器校验与汇总脚本](../../benchmarks/summarize-flashinfer-sonic-mxfp8-grouped.py)
- [完整对照表与原始数据链接](FLASHINFER_VS_SONIC.md)

## 10. EP4 多卡系统结果

EP4 是另一组测试，不能与上面的单卡 kernel latency 混为一张性能曲线。它固定
global tokens 16,384、4,096/rank、top-k 24、hidden 2,560、
post-SwiGLU intermediate 1,024、768 global experts、EP=4，并包含
routing/dispatch、NCCL 通信、本地 FC1/FC2、reduce 和 combine。

完整 balanced、Zipf/EPLB、静态权重迁移成本、MegaMoE IBGDA 正确性状态以及
公开复现脚本见 [EP4 正式记录](ep4/README.md)。主 README 也保留了 balanced
结果摘要，机器可读汇总见 [ep4/summary.json](ep4/summary.json)。
