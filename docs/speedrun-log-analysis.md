# speedrun.log 训练过程解析

本文档对 `speedrun.log` 训练日志进行全面解析，记录了 d24 模型（24层，1.38B参数）的完整预训练和微调过程。

---

## 1. 依赖初始化 (行1-3)

```
Resolved 134 packages in 343ms
Checked 121 packages in 2.82s
Reset report and wrote header to...
```

- `uv` 包管理器解析项目依赖
- 报告系统初始化

---

## 2. 数据集下载 (行4-191)

```
Downloading 9 shards using 4 workers...
Target directory: .../base_data_climbmix
Done! Downloaded: 9/9 shards
Downloading 171 shards using 4 workers...
Done! Downloaded: 171/171 shards
```

- 从 HuggingFace 下载两个数据集：
  - **验证集**：9 个 shards
  - **训练集**：171 个 shards
- 使用 4 个并行 worker 加速下载

---

## 3. BPE Tokenizer 训练 (行192-323)

```
Processing sequences from iterator (buffer_size: 8192)
Processed 758523 sequences total, 2098273 unique
Starting BPE training: 32503 merges to compute
Progress: 1%...100% (32503 merges)
Training time: 86.34s
```

### 关键参数
| 参数 | 值 |
|------|-----|
| vocab_size | 32,768 |
| merge 操作数 | 32,503 |
| 训练耗时 | 86.34s |

### 与 GPT-2/GPT-4 对比

| 指标 | Ours | GPT-2 | GPT-4 |
|------|------|-------|-------|
| vocab_size | 32,768 | 50,257 | 100,277 |
| code 压缩效率 (tokens/byte) | 3.17 | 2.19 | - |
| 整体训练数据对比 | 基准 | 差 1.2-1.4x | - |

---

## 4. 多 GPU 分布式训练初始化 (行324-846)

```
Autodetected device type: cuda
GPU: NVIDIA A800-SXM4-80GB | Peak FLOPS (BF16): 3.12e+14
COMPUTE_DTYPE: torch.bfloat16
```

### 硬件配置
| 项目 | 配置 |
|------|------|
| GPU 型号 | NVIDIA A800-SXM4-80GB × 8 |
| 通信库 | NCCL 2.27.5 + CUDA 12.9 |
| 网络 | InfiniBand RDMA (mlx5_0) |
| P2P 插件 | IBext (RDMA) |
| 通信 Channel 数 | 16 (15× all-reduce + 1× collective) |

### Flash Attention 状态
```
WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback
WARNING: SDPA has no support for sliding window attention (window_pattern='SSSL')
WARNING: Recommend using --window-pattern L for full context attention
```

- **FA3**: 仅 Hopper GPU (H100/H200) 支持，A800 为 Ampere 架构不兼容
- **SSSL 模式**: Sliding-Sliding-Sliding-Long，SDPA 不支持，会退化为全上下文注意力
- **建议**: 使用 `--window-pattern L` 可避免此问题

---

## 5. 模型配置 (行858-887)

```json
{
  "sequence_len": 2048,
  "vocab_size": 32768,
  "n_layer": 24,
  "n_head": 12,
  "n_kv_head": 12,
  "n_embd": 1536,
  "window_pattern": "SSSL"
}
```

### 参数分布

| 模块 | 参数量 |
|------|--------|
| wte (embedding) | 50.3M |
| value_embeds | 604M |
| lm_head | 50.3M |
| transformer_matrices | 679.5M |
| **total** | **1.38B** |

---

## 6. 计算量与批大小 (行876-887)

```
Estimated FLOPs per token: 4.775225e+09
Auto-computed optimal batch size: 1,048,576 tokens
Tokens : Scaling params ratio: 8.00
Total training FLOPs estimate: 2.788002e+19
Tokens / micro-batch / rank: 16 x 2048 = 32,768
Total batch size 1,048,576 => gradient accumulation steps: 4
```

### 关键指标

| 指标 | 值 | 说明 |
|------|-----|------|
| token:parameter ratio | 8.0 | Chinchilla optimal 为 20，mini 模型用更高比例 |
| 总训练 tokens | 5.8 万亿 | 5,838,471,168 |
| micro-batch/rank | 32,768 | 16 samples × 2048 seq_len |
| 总 batch size | 1,048,576 | 8 卡聚合 |
| gradient accumulation | 4 步 | 每 4 步累积一次梯度 |

---

## 7. 预训练阶段 (行888-6760)

### 验证 Loss 变化

| Step | Validation bpb | 变化 |
|------|----------------|------|
| 00000 | 3.167413 | - |
| 00250 | 0.989483 | -68.8% |
| 00500 | 0.903492 | -8.7% |
| 01000 | 0.844524 | -6.5% |
| 01500 | 0.822373 | -2.6% |
| 02000 | 0.807178 | -1.8% |
| 03000 | 0.771217 | -4.5% |
| 04000 | 0.745198 | -3.4% |
| 05000 | 0.724861 | -2.7% |
| 05500 | 0.717854 | -1.0% |
| 05568 | **0.717218** | 最终值 |

### 训练指标解释

```
step 00000/05568 (0.00%) | loss: 10.397663 | lrm: 0.03 | dt: 77971.27ms | tok/sec: 13,448 | bf16_mfu: 2.57 | epoch: 1 pq: 0 rg: 8 | total time: 0.00m
```

| 字段 | 含义 |
|------|------|
| step | 当前步 / 总步数 (进度%) |
| loss | 训练 loss (EMA 平滑) |
| lrm | learning rate multiplier (lr 相对于初始值的比例) |
| dt | 每步耗时 (毫秒) |
| tok/sec | 每秒处理 token 数 |
| bf16_mfu | bf16 算力利用率 (%) |
| pq | prefetch queue index (数据预取队列索引) |
| rg | real gradient steps (已完成的有效梯度更新步数) |
| epoch | 当前 epoch 和数据位置 |

### 检查点保存

- **保存频率**: 每 2000 步
- **保存内容**:
  - `model_XX.pt` - 模型参数 (~2.6GB/rank)
  - `optim_XX_rankX.pt` - 优化器状态 (~10GB/rank)
  - `meta_XX.json` - 元数据

### CORE Metric 评估

| Step | CORE metric |
|------|-------------|
| 2000 | 0.1865 |
| 4000 | 0.2395 |
| 5568 | 0.2520 |

---

## 8. Chat SFT 微调阶段 (行6761-)

```
Step 00000 | Validation bpb: 0.4434
Step 00200 | Validation bpb: 0.3062 | ChatCORE: 0.2882
```

- 在预训练模型基础上进行对话微调
- validation bpb 从 0.44 降到 0.27

---

## 9. 评估阶段 (行10100-11099)

### GSM8K 数学推理
```
Final: 108/1319 (8.19%)
GSM8K accuracy: 8.19%
```
- 1319 题中答对 108 题

### HumanEval 代码生成
```
Final: 17/164 (10.37%)
HumanEval accuracy: 10.37%
```
- 164 题中答对 17 题

### SpellingBee 拼写检查
```
Final: 255/256 (99.61%)
SpellingBee accuracy: 99.61%
```
- 256 题中答对 255 题

---

## 10. 训练完成 (行11088-11099)

```
NCCL INFO comm ... - Destroy COMPLETE
Generating report to .../report/report.md
```

- NCCL 通信关闭
- 生成训练报告

---

## 11. 关键发现与优化建议

### 11.1 已发现问题

| 问题 | 影响 | 建议 |
|------|------|------|
| FA3 不可用 (A800 非 Hopper) | GPU 利用率受限 | 使用 `--window-pattern L` |
| SSSL 滑动窗口模式不支持 | SDPA 退化为全上下文注意力 | 考虑改用 `--window-pattern L` |
| GPU 利用率约 44% | 未充分利用硬件 | 检查 window pattern 设置 |

### 11.2 性能数据汇总

| 指标 | 值 |
|------|-----|
| 8 卡 A800 每秒处理 | ~230,000 tokens |
| 每步耗时 | ~4.5 秒 |
| 完整训练 (5568 步) | ~7 小时 |
| 峰值显存使用 | ~80GB/GPU |
| 总训练时间 | 420 分钟 (7 小时) |

### 11.3 Flash Attention 说明

- **FA3**: 仅 Hopper 架构 (H100/H200) 支持，A800 不可用
- **FA2**: 已安装 (版本 2.8.3)，PyTorch SDPA 会自动使用 FA2 内核加速
- **最近训练**: 虽然日志显示 "SDPA fallback"，但 PyTorch SDPA 实际内部调用了 FA2 优化实现

---

## 12. 附录：训练日志字段速查

```
step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) 
  | loss: {debiased_smooth_loss:.6f}   # EMA 平滑后的训练 loss
  | lrm: {lrm:.2f}                    # learning rate multiplier
  | dt: {dt * 1000:.2f}ms            # 每步耗时
  | tok/sec: {tok_per_sec:,}           # token 吞吐
  | bf16_mfu: {mfu:.2f}               # bf16 算力利用率
  | epoch: {epoch}                     # epoch 和数据位置信息
  | total time: {total_training_time/60:.2f}m  # 已训练时间
  | eta: {eta_seconds/60:.1f}m        # 预计剩余时间
```

其中 `epoch` 字段格式为:
```
epoch: {epoch} pq: {pq_idx} rg: {rg_idx}
```
- `pq`: prefetch queue index
- `rg`: real gradient steps (已完成的梯度累积步数)
