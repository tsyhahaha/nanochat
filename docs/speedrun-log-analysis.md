# speedrun.log 训练复盘与技术分析

本文档基于 `speedrun.log`，从训练系统、模型配置、收敛行为和下游效果四个角度，对这次 `d24` speedrun 做一次面向 LLM 训练工程的复盘。目标不是逐行复述日志，而是提炼出这次 run 真正说明了什么、哪些地方可信、哪些地方仍需下一轮实验验证。

## 1. 一句话结论

- 这次 run 成功完成了一个 `24` 层、`1.384B` 参数模型的预训练、base eval、chat SFT 和最终任务评估，流程完整且没有明显训练崩溃。
- 预训练主阶段在 `8 x A800-SXM4-80GB` 上跑满一个稳定配置，主训练吞吐约 `231k tok/s`，`bf16_mfu` 长时间稳定在 `44%` 左右，说明系统是稳定的，但远没有逼近硬件上限。
- 最大的已知效率问题不是 NCCL，也不是 batch 太小，而是日志明确警告：`window_pattern='SSSL'` 在当前 `PyTorch SDPA` 路径下不支持 sliding window attention，导致注意力实现与预期设计不匹配，并直接拉低利用率。
- 预训练收敛是健康的：验证 `bpb` 从 `3.1674` 持续降到 `0.7172`，`CORE metric` 从 `0.1865` 提升到 `0.2520`，说明模型不是只在优化训练 loss，而是真正获得了可测的泛化收益。
- Chat SFT 继续带来明显收益：验证 `bpb` 从 `0.4434` 降到 `0.2739`，`ChatCORE` 从 `0.2882` 升到 `0.3700`。这说明 base model 已具备足够好的初始化，SFT 阶段主要在把能力往指令跟随与任务形式对齐上推。

## 2. 本次 run 的流程概览

整个日志可以分成四段：

1. 数据与 tokenizer 准备
2. `d24` base pretraining
3. base checkpoint 离线评估
4. chat SFT 与最终 benchmark 评估

这四段是串起来的，而不是独立实验：SFT 阶段直接从 `base_checkpoints/d24` 的 `step 5568` 加载模型和优化器动量，再继续训练。

## 3. 数据与 tokenizer：没有成为瓶颈，但有一些值得记住的信号

### 3.1 数据准备

- 验证集下载：`9` 个 shards
- 训练集下载：`171` 个 shards
- 目标目录相同，且绝大多数文件已存在，因此这次 run 的下载阶段更像 cache hit 检查，而不是一次完整冷启动下载

这意味着不能把日志前部的耗时拿来当成真实端到端冷启动成本。

### 3.2 tokenizer 训练

日志中 tokenizer 相关关键信号：

- `vocab_size: 32,768`
- `Processed 758523 sequences total, 2098273 unique`
- `32503 merges`
- `Training time: 86.34s`

这套 tokenizer 的工程价值在于：

- 训练足够快，不是整个 pipeline 的关键瓶颈
- 对 code / math / science 这类训练重要域并不差
- 在 `fwe-train` 和 `fwe-val` 上相对 GPT-2 tokenizer 有 `1.2%` 到 `1.4%` 的 token efficiency 优势

更重要的是要正确理解日志里的对比：

- 这不是在证明“我们的 tokenizer 全面优于 GPT-4 tokenizer”
- 它只说明这套 `32k` 词表在本次训练分布上是合理的，尤其对代码压缩更友好
- 对韩文等非核心域明显不占优，因此它更像一个针对当前训练混合数据的任务型 tokenizer，而不是通用 tokenizer

## 4. 硬件与分布式通信：系统是通的，通信不是主要问题

### 4.1 硬件环境

- GPU：`8 x NVIDIA A800-SXM4-80GB`
- 精度：`torch.bfloat16`
- 单卡理论 BF16 峰值：`3.12e+14 FLOPS`
- `Distributed world size: 8`

### 4.2 NCCL / 网络侧观察

日志显示：

- 使用 `NCCL 2.27.5 + CUDA 12.9`
- 网络插件最终成功加载 `NCCL RDMA Plugin v6`
- `P2P plugin IBext`
- `NET/IB : Using [0]mlx5_0:1/RoCE ...`
- `16 coll channels, 16 collnet channels, 16 p2p channels`

虽然日志中一开始出现了大量类似下面的行：

```text
Failed to find ncclNetPlugin_v10 symbol
Failed to find ncclCollNetPlugin_v10 symbol
```

但这并不等价于通信异常。它表示 NCCL 在探测更高版本插件接口时回退，随后成功加载了兼容版本：

```text
Loaded net plugin NCCL RDMA Plugin v6 (v6)
Loaded collnet plugin SHARP (v6)
Successfully loaded external plugin libnccl-net.so
```

从结果看：

- 多卡训练顺利跑完整个 pretrain 与 SFT
- 没有看到 rank hang、allreduce timeout、desync 或重试风暴
- 吞吐在长时间尺度上很平滑

因此，这次 run 的首要瓶颈并不是 NCCL。

## 5. 预训练配置：算得通，也基本跑稳了

### 5.1 模型结构

日志给出的配置是：

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

参数分布：

| 模块 | 参数量 | 占比 |
|---|---:|---:|
| `wte` | 50,331,648 | 3.64% |
| `value_embeds` | 603,979,776 | 43.64% |
| `lm_head` | 50,331,648 | 3.64% |
| `transformer_matrices` | 679,478,976 | 49.09% |
| `scalars` | 74 | ~0 |
| `total` | 1,384,122,122 | 100% |

这里最值得训练工程师注意的不是总参数量，而是 `value_embeds` 占了将近 `44%`。这说明：

- 这个模型并不是一个“参数主要花在 transformer block 上”的标准 GPT 配置
- embedding 相关设计对显存、优化器状态、训练稳定性和 token efficiency 都有实质影响
- 如果未来要继续做 speedrun，对 `value_embeds` 的收益/成本比值得单独审视

### 5.2 batch 与训练预算

- `Estimated FLOPs per token: 4.775225e+09`
- `Auto-computed optimal batch size: 1,048,576 tokens`
- `Total number of training tokens: 5,838,471,168`
- `Tokens : Scaling params ratio: 8.00`
- `Total training FLOPs estimate: 2.788002e+19`
- `Tokens / micro-batch / rank: 16 x 2048 = 32,768`
- `Tokens / micro-batch: 262,144`
- `gradient accumulation steps: 4`

这套配置说明本次训练不是在“省卡试跑”，而是在认真遵循一个固定 token budget 路线跑完整个 schedule。

不过要注意两个点：

- 日志里的 `optimal batch size` 是代码中的 heuristic 结果，不是理论最优值本身
- `data:param ratio = 8.0` 对 `1.38B` 模型来说偏保守，说明这次目标更接近 speedrun / capability bootstrap，而不是追求更长 token budget 下的极限收敛

## 6. 预训练效率：稳定，但被注意力实现卡住了

### 6.1 真实吞吐与利用率

首步之外，预训练大部分时间稳定在：

- `dt`: 约 `4.53s`
- `tok/sec`: 约 `228k` 到 `231k`
- `bf16_mfu`: 约 `43.8%` 到 `44.3%`

末尾 run summary：

- `train/dt = 4.53603`
- `train/tok_per_sec = 231166`
- `train/mfu = 44.22557`

这说明训练系统具有两个优点：

- 稳定性很好，几千步内吞吐几乎不漂
- data loader 没有明显饿死 GPU，系统也没有周期性抖动

### 6.2 为什么说效率仍然不够好

日志已经把主要原因说得非常直接：

```text
WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback
WARNING: SDPA has no support for sliding window attention (window_pattern='SSSL'). Your GPU utilization will be terrible.
WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.
```

这三句基本可以直接解读为：

1. 当前硬件是 A800，不能用 FA3
2. 当前模型配置要求 `SSSL` sliding window pattern
3. 当前执行路径的 SDPA 不支持这类模式，因此没有按设计高效执行

因此，日志里 `44%` 左右的 MFU 不是一个“已经优化得差不多”的数字，而更像“在错误注意力实现路径下，系统仍然能稳定训练”的数字。

### 6.3 对 FA2 的表述要谨慎

现有文档里有一句“PyTorch SDPA 实际内部调用了 FA2 优化实现”。这句话从本日志本身并不能严格推出。

更稳妥的表述是：

- 日志只明确说明走的是 `PyTorch SDPA fallback`
- 日志同时明确说明 `SSSL` 在该路径下不受支持
- 因此我们不能把这次 run 描述成“虽未启用 FA3，但仍等价享受到 FA2 sliding-window 加速”

专业文档里这点要严谨，否则会误导后续性能归因。

## 7. 预训练收敛：健康、连续，没有后期发散

### 7.1 验证 bpb 曲线

预训练验证 `bpb`：

| Step | Validation bpb |
|---|---:|
| 0 | 3.167413 |
| 250 | 0.989483 |
| 500 | 0.903492 |
| 1000 | 0.844524 |
| 1500 | 0.822373 |
| 2000 | 0.807178 |
| 3000 | 0.771217 |
| 4000 | 0.745198 |
| 5000 | 0.724861 |
| 5500 | 0.717854 |
| 5568 | 0.717218 |

这个曲线有几个典型特征：

- 前 `250` step 收敛极快，属于随机初始化到可用语言模型的 bootstrap 区间
- `1000` step 后仍持续下降，没有出现明显平台期
- 到末尾虽然边际改善变小，但没有出现过拟合反弹

换句话说，这个 run 在结束时更像“预算用完了”，而不是“模型已经训练不动了”。

### 7.2 CORE metric 变化

| Step | CORE metric |
|---|---:|
| 2000 | 0.1865 |
| 4000 | 0.2395 |
| 5568 | 0.2520 |

这和 `bpb` 下降基本同向，说明训练不是只有语言建模损失在变好，而是零样本/少样本通用能力也在提升。

不过也要注意：

- `CORE` 增长从 `2000 -> 4000` 更明显
- `4000 -> 5568` 仍提升，但斜率变小

这通常意味着后半程继续训练仍有收益，但单位 FLOP 的收益已经开始递减。

### 7.3 样本质量：能看出“会背了”，但还没学会组织长程输出

日志里的 conditioned sample 普遍表现为：

- 能延续 prompt 表面模式
- 基础事实常常对
- 容易重复、循环、短路

这与 `1.38B` 模型在早中期预训练的典型状态一致：

- 局部语言建模能力已经建立
- 长程 coherence、抽象推理和多步生成还不够强

因此 sample 质量与 `CORE=0.252` 是一致的，并不存在“指标很好但样本完全不对劲”的异常。

## 8. Base eval：预训练模型已经有可见能力，但远未到成熟阶段

base checkpoint 离线评估结果：

- `train bpb: 0.716437`
- `val bpb: 0.714582`
- `CORE metric: 0.2510`

这个离线 eval 和训练过程里的最终指标非常一致，说明：

- checkpoint 保存是可信的
- 训练时记录的验证结果没有明显漂移或评估污染

一些代表性 base 指标：

- `hellaswag: 0.5560`
- `arc_easy: 0.7020`
- `arc_challenge: 0.4220`
- `piqa: 0.7720`
- `winogrande: 0.5720`
- `boolq: 0.5160`

对这个参数规模和训练预算来说，这是一组“已经形成基础通识能力，但还没有进入强推理区间”的正常结果。

## 9. Chat SFT：收益明确，而且更像“对齐增益”而不是“能力幻觉”

### 9.1 SFT 初始化方式

SFT 阶段直接继承了 pretrained checkpoint 的多个关键设置：

- `Inherited max_seq_len=2048`
- `Inherited total_batch_size=1048576`
- `Inherited embedding_lr=0.3`
- `Inherited unembedding_lr=0.008`
- `Inherited matrix_lr=0.02`
- `Loaded optimizer state from pretrained checkpoint (momentum buffers only, LRs reset)`

这是一种比较务实的衔接方式：

- 保留了 optimizer state 中对训练稳定性有帮助的动量信息
- 但重新设置 LR，避免直接沿用 pretrain 后期衰减到很低的学习率

### 9.2 SFT 数据混合

- `Training mixture: 1,071,759 rows (MMLU x3, GSM8K x4)`

这说明 SFT 不是纯对话数据微调，而是带有明显任务型 curriculum：

- 提高知识问答与考试类行为
- 有意强化数学题分布

因此，后续在 `ChatCORE`、`MMLU`、`GSM8K`、`HumanEval` 上的变化，可以视为预期内的训练目标结果，而不是偶然波动。

### 9.3 SFT 收敛情况

关键点：

| Step | Validation bpb | ChatCORE |
|---|---:|---:|
| 0 | 0.4434 | - |
| 200 | 0.3062 | 0.2882 |
| 400 | 0.2810 | 0.3701 |
| 483 | 0.2739 | 0.3700 |

这条曲线很有代表性：

- `0 -> 200`：对齐收益极大，说明 base model 已经有足够可塑性
- `200 -> 400`：`ChatCORE` 继续明显提升
- `400 -> 483`：`Validation bpb` 仍降，但 `ChatCORE` 基本持平

这通常意味着：

- SFT 后段仍在继续拟合训练分布
- 但综合任务收益已经接近平台期
- 如果目标是更高 `ChatCORE`，下一轮不一定要更长训练，可能更该改数据混合或指标定义

### 9.4 SFT 后任务表现

`Step 483` 附近：

- `ARC-Easy: 61.83%`
- `ARC-Challenge: 49.23%`
- `MMLU: 36.70%`
- `GSM8K: 12.50%`
- `HumanEval: 12.50%`
- `SpellingBee: 100.00%`
- `ChatCORE: 0.3700`
- `ChatCORE_cat: 0.3234`

这里最值得注意的是结构性差异：

- 通识/考试类任务提升明显
- 数学与代码仍弱，但相对 base 已有提升
- `SpellingBee` 已接近饱和，说明低难度 pattern completion 不是当前短板

## 10. 最终 benchmark：对模型能力边界的真实定位

最终评估：

| 任务 | 结果 |
|---|---:|
| GSM8K | `108/1319 = 8.19%` |
| HumanEval | `17/164 = 10.37%` |
| SpellingBee | `255/256 = 99.61%` |

如何解读这三项：

- `GSM8K 8.19%`：说明模型已经具备少量算术/模板推理能力，但还远没有形成稳定链式推理能力
- `HumanEval 10.37%`：说明 tokenizer 对 code 友好这件事有帮助，但代码生成能力仍主要受模型规模、训练预算和 code/task 数据混合限制
- `SpellingBee 99.61%`：再次说明模型在近似模式匹配、短上下文映射和表层语言任务上已经很强

换句话说，这次 speedrun 的模型画像非常清晰：

- 语言模型基础能力：已建立
- 常识与多选任务：可用
- 数学与代码：刚起步
- 长程推理与稳健解题：明显不够

## 11. 这次 run 里哪些说法不应该写进正式文档

原文档里有几处结论不够严谨，建议在正式复盘里删除或改写。

### 11.1 “GPU 利用率约 44%，检查 window pattern 设置”是对的，但原因要说完整

更准确的说法是：

- 当前注意力实现路径和 `SSSL` 模式不兼容
- 这不是简单的“window pattern 设置不理想”
- 而是“当前 kernel 路径不支持该模式，导致实现退化”

### 11.2 “峰值显存约 80GB/GPU”不成立

日志明确给出的峰值显存是：

- pretrain：`60236.96 MiB`
- chatsft：`60245.65 MiB`

也就是约 `60.2 GiB`，而不是接近 `80GB`。

### 11.3 “checkpoint 大小约多少 GB/rank”没有日志证据

日志只说明保存了：

- `model_005568.pt`
- `meta_005568.json`
- `optim_005568_rank*.pt`

但没有打印文件大小，因此正式文档不应写出具体体积，除非额外做文件系统统计。

### 11.4 “SDPA 实际内部调用 FA2，因此近期训练已经被 FA2 加速”证据不足

从本日志无法严格证明这件事。严谨写法应该停留在：

- 执行路径是 `PyTorch SDPA fallback`
- `SSSL` 不受支持
- 性能明显受影响

## 12. 下一轮实验最值得优先做什么

按收益优先级，我会建议下面三件事。

### 12.1 第一优先级：修正注意力实现路径

直接按日志建议做一轮 A/B：

- A：保持当前模型规模与 batch，不改别的，只把 `--window-pattern` 改成 `L`
- B：保留 `SSSL`，但换到真正支持该模式的 kernel 路径后再比较

目标不是先看最终 benchmark，而是先看：

- `tok/sec`
- `mfu`
- `peak memory`
- 同等训练 token 下的 `val bpb` 与 `CORE`

如果 `L` 能把 MFU 明显拉高，而收敛损失不大，那么这会是最直接的 speedrun 提升点。

### 12.2 第二优先级：重新审视 `value_embeds` 的成本收益

由于它占了接近一半参数，建议单独做 ablation：

- 固定训练 token budget
- 控制总参数量近似相同或控制 transformer 规模近似相同
- 对比 `value_embeds` 的存在是否真的换来了更好的 `CORE / ChatCORE / code/math`

否则模型可能在把太多参数预算花在低收益部分。

### 12.3 第三优先级：针对数学和代码做更有针对性的后训练

当前最终表现说明：

- 通识和分类型任务已有较强基础
- 数学和代码仍是瓶颈

因此下一轮更可能有效的是：

- 增加高质量数学 reasoning 数据
- 增加 code completion / code execution feedback 信号
- 针对 GSM8K / HumanEval 做更强的 task-balanced SFT，而不是单纯延长现有 SFT 步数

## 13. 最终复盘结论

如果把这次 run 当作一次 speedrun 工程验收，它是成功的：

- pipeline 全通
- 多卡系统稳定
- 预训练和 SFT 都有清晰收敛
- checkpoint 可复用
- 下游评估和训练期指标彼此一致

如果把它当作一次性能优化实验，它又暴露出一个非常明确的问题：

- 当前 `A800 + SDPA fallback + SSSL` 的组合不合适
- 训练虽然没坏，但效率明显受限

如果把它当作一次能力建设实验，结论也很清楚：

- 这次预算足以训练出一个“已经会说、会答、会做基础选择题”的 base/chat 模型
- 但还不足以把数学推理和代码生成推到强区间

因此，下一轮最值得投入的不是盲目增加训练时长，而是先修正注意力路径，再决定是否扩大 token budget 或改动模型结构。
