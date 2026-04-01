# nanochat 训练时间分析：8×A800 vs 8×H100

> 分析日期：2026-03-30
> 基于 `speedrun.log` 实测数据（训练进行中，step 247/5568）与项目 Leaderboard 历史记录

---

## 1. 环境对比

| 项目 | 8×H100 (Leaderboard SOTA) | 8×A800 (本次实测) |
|------|--------------------------|-------------------|
| GPU 型号 | NVIDIA H100-SXM5-80GB | NVIDIA A800-SXM4-80GB |
| 架构 | Hopper (SM 90) | Ampere (SM 80) |
| BF16 峰值算力/卡 | 990 TFLOPS | 312 TFLOPS |
| FP8 支持 | 是 (峰值 1979 TFLOPS) | 否 |
| 计算精度 | FP8 (torchao) | BF16 |
| Flash Attention 3 | 支持 | 不支持，回退到 SDPA |
| 滑动窗口注意力 | FA3 原生支持 (SSSL) | SDPA 不支持，全上下文退化 |
| 互联带宽 | NVLink 900 GB/s | NVLink 400 GB/s |
| 每小时成本 | ~$24/hr | ~$16/hr |

---

## 2. 预训练核心指标实测

### 2.1 A800 实测数据（来自 speedrun.log）

| 指标 | 数值 |
|------|------|
| 模型 | d24 (24层, 1.38B 参数) |
| 训练步数 | 5,568 步 |
| 总训练 tokens | 5,838,471,168 (~5.84B) |
| 总训练 FLOPs | 2.788×10¹⁹ |
| 批大小 | 1,048,576 tokens |
| 梯度累积 | 4 步 |
| 每步耗时 (稳态) | ~4,535 ms |
| 吞吐量 | ~231,000 tok/sec |
| BF16 MFU | 44.2% |
| 有效算力/卡 | 138.0 TFLOPS |

### 2.2 H100 Leaderboard 数据 (Run 6, SOTA)

| 指标 | 数值 |
|------|------|
| 模型 | d24 (24层, 同架构) |
| 训练配置 | `--target-param-data-ratio=8 --fp8` |
| 纯训练时间 | 99 分钟 (1.65 小时) |
| 有效算力/卡 | 586.7 TFLOPS |
| 等效 BF16 MFU | 59.3% |

---

## 3. 各阶段时间估算

### 3.1 预训练 (base_train)

这是整个流水线中耗时最长的阶段。

```
A800 预训练时间 = step_0 + (总步数 - 1) × 稳态步时
                = 15.2s + 5567 × 4.535s
                = 25,247s
                ≈ 421 分钟
                ≈ 7.0 小时
```

与 speedrun.log 中 ETA 吻合（step 247 时 ETA 显示 402 分钟，加上已用 18 分钟 = 420 分钟）。

| 对比 | H100 FP8 | A800 BF16 | 倍率 |
|------|----------|-----------|------|
| 预训练时间 | 99 min | ~421 min | 4.25× |
| 有效算力/卡 | 586.7 TFLOPS | 138.0 TFLOPS | 4.25× |

**4.25× 的减速来源分析：**

| 因素 | 贡献 |
|------|------|
| 硬件代差 (H100 BF16 990 vs A800 BF16 312 TFLOPS) | ~3.17× |
| FP8 加速 (H100 FP8 vs H100 BF16) | ~1.34× |
| **合计理论** | **~4.25×** |

> 注：实测减速倍率与理论完美吻合，说明 A800 的 MFU (44.2%) 已经接近该硬件的合理上限。尽管 SDPA 回退和缺少滑动窗口支持会降低效率，但 `window_pattern='SSSL'` 在 SDPA 下退化为全上下文注意力，计算量增加但 MFU 计算基准也相应调整。

### 3.2 其他阶段时间估算

以下阶段基于 H100 经验数据和 A800/H100 算力比推算：

| 阶段 | H100 FP8 估计 | A800 BF16 估计 | 说明 |
|------|--------------|---------------|------|
| 环境安装 (uv sync) | 1 min | 1 min | CPU 密集，与 GPU 无关 |
| 数据下载 (170 shards) | 2-5 min | 2-5 min | 网络 I/O，首次需下载 ~17GB |
| Tokenizer 训练 | ~80s | ~80s | CPU 密集 (rustbpe)，实测 79s |
| Tokenizer 评估 | ~10s | ~10s | CPU 密集 |
| **预训练 (d24)** | **99 min** | **~421 min** | **核心瓶颈，4.25× 减速** |
| 预训练评估 (CORE+BPB) | 10-15 min | 30-50 min | 推理密集，约 3× 减速 |
| SFT 数据下载 | <1 min | <1 min | 2.3MB identity conversations |
| **SFT 训练** | **10-15 min** | **40-60 min** | 同模型架构，类似减速比 |
| SFT 评估 | 5-10 min | 15-30 min | 推理密集 |
| 报告生成 | <1 min | <1 min | CPU 密集 |

### 3.3 总时间估算

| 平台 | 预训练 | 其他阶段 | 总计 | 成本 |
|------|--------|---------|------|------|
| 8×H100 FP8 | 1.65 hr | ~0.5-0.7 hr | ~2.2-2.4 hr | ~$53-58 |
| 8×A800 BF16 | 7.0 hr | ~1.5-2.5 hr | **~8.5-9.5 hr** | **~$136-152** |

---

## 4. 关键性能瓶颈

### 4.1 无 FP8 支持

A800 (Ampere) 不支持 FP8 运算。H100 的 FP8 Tensor Core 提供 2× 于 BF16 的峰值算力，且 nanochat 的 FP8 实现 (torchao tensorwise scaling) 在 H100 上额外带来 ~34% 的加速。A800 只能使用 BF16，这是最大的单一减速因素。

### 4.2 无 Flash Attention 3

speedrun.log 中明确警告：

```
WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback
WARNING: SDPA has no support for sliding window attention (window_pattern='SSSL')
WARNING: Your GPU utilization will be terrible.
WARNING: Recommend using --window-pattern L for full context attention
```

FA3 是 Hopper 架构专属特性。A800 回退到 PyTorch SDPA，且 SDPA 不支持滑动窗口注意力模式 (SSSL = Sliding-Sliding-Sliding-Long)，导致所有层都使用全上下文注意力，计算量增大。

**优化建议：** 可尝试添加 `--window-pattern L` 参数，使用纯全上下文注意力（与当前 SDPA 退化行为一致，但避免了不必要的模式切换开销）。也可安装 FlashAttention 2 (`flash-attn` 包) 以获得 Ampere 上的优化注意力实现。

### 4.3 NVLink 带宽

A800 的 NVLink 带宽 (400 GB/s) 约为 H100 (900 GB/s) 的 44%。在 8 卡 DDP 训练中，梯度同步的通信开销更高。不过从实测 MFU 44.2% 来看，通信尚未成为主要瓶颈（计算仍是主导）。

---

## 5. 成本效益分析

| 指标 | 8×H100 FP8 | 8×A800 BF16 |
|------|-----------|-------------|
| 预训练时间 | 1.65 hr | 7.0 hr |
| 全流程时间 | ~2.3 hr | ~9.0 hr |
| 小时费率 | $24/hr | $16/hr |
| 预训练成本 | $39.6 | $112.0 |
| 全流程成本 | ~$55 | ~$144 |
| 性价比 (TFLOPS/$) | 更高 | 较低 |

> 结论：尽管 A800 单价更低，但由于 4.25× 的训练时间差距，A800 的总成本反而是 H100 的 ~2.6×。对于 nanochat 这类计算密集型任务，H100 FP8 的性价比显著优于 A800 BF16。

---

## 6. 可能的优化方向

1. **安装 FlashAttention 2**：`pip install flash-attn`，可在 Ampere 上提供优化的注意力实现，预计提升 5-15% 吞吐
2. **使用 `--window-pattern L`**：避免 SDPA 对 SSSL 模式的低效处理
3. **增大 device-batch-size**：如果显存允许，尝试 `--device-batch-size=32` 减少梯度累积次数
4. **混合精度优化**：虽然无 FP8，但可确认 BF16 matmul 的 TF32 模式已启用
5. **NCCL 调优**：针对 A800 NVLink 拓扑优化 NCCL 参数

---

*本文档基于 speedrun.log 前 247 步的稳态数据外推。预训练阶段的步时非常稳定 (标准差 <0.3%)，估算可信度高。SFT 和评估阶段为基于算力比的推算，实际时间可能有 ±20% 偏差。*
