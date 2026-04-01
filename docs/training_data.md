# Training Data Documentation

This document describes the datasets used in nanochat's training pipeline, including the reasoning behind dataset selection, mixing strategies, and practical insights for practitioners.

---

## 1. Pretraining Dataset: ClimbMix-400B

### Overview

| Attribute | Details |
|-----------|---------|
| **Source** | `nvidia/Nemotron-ClimbMix` (curated by NVIDIA NeMo Curator) |
| **Distribution Repo** | `karpathy/climbmix-400b-shuffle` (repackaged by Andrej Karpathy) |
| **Scale** | ~400B tokens (6543 shards total) |
| **Format** | Parquet, ~100MB per shard (zstd compression), ~250M characters per shard |
| **Storage** | Pre-tokenized with GPT-2 tokenizer (token IDs) |
| **Purpose** | Pretraining the base language model |
| **History** | Originally FinewebEdu-100B, switched to ClimbMix on March 4, 2026 for superior performance |

### Why ClimbMix? — The Art of Pretraining Data Selection

选择预训练数据集是整个训练流程中影响最大的单一决策。nanochat 在 ClimbMix 之前尝试了至少 **6 种不同的数据集**，全部失败：

| 尝试 | 数据集 | 结果 | 失败原因分析 |
|------|--------|------|-------------|
| 基线 | FineWeb-EDU 100B | CORE 0.2602 | 纯教育文本，多样性不足 |
| #1 | OLMo dolma3_mix-6T | CORE 0.138 (-47%) | 质量参差不齐，存在极短文档（如仅含 "5"） |
| #2 | DCLM | 未完成 | 作者担忧 DCLM 团队同时设计了 CORE 评分，可能存在数据-评估过拟合 |
| #3 | FineWeb (非 EDU) | CORE 0.2241 (-14%) | 去掉教育过滤后质量显著下降 |
| #4 | FinePDFs+DCLM+FineWeb-EDU 混合 | CORE 0.2549 (-2%) | 混合比例未必最优，PDF 数据噪声大 |
| #5 | 其他混合实验 | 均负面 | — |
| **#6** | **ClimbMix 400B** | **CORE 大幅提升，训练时间 -27%** | **NVIDIA NeMo Curator 的工业级数据清洗** |

**关键 insight：**

1. **数据质量 > 数据数量**。ClimbMix 不仅仅是"更多数据"（400B vs 100B），更重要的是它经过了 NVIDIA NeMo Curator 的 GPU 加速数据清洗流水线：质量过滤、去重、语言识别。这使得模型可以从 d26 缩小到 d24 就达到 GPT-2 能力，说明数据质量直接减少了所需的模型容量。

2. **多样性是关键**。FineWeb-EDU 是纯教育文本，而 ClimbMix 混合了高质量网页文本、代码、数学等多种来源。纯教育文本在 MMLU 等知识测试上表现好，但在代码和推理任务上受限。ClimbMix 的多样性让小模型也能覆盖更广的能力谱。

3. **警惕数据-评估过拟合**。作者在 LOG.md 中明确提到对 DCLM 的顾虑：同一团队既准备数据集又设计评估指标，可能导致隐性过拟合。这是一个容易被忽视但极其重要的方法论问题。

4. **不要低估数据清洗的价值**。OLMo 的 dolma3 数据集包含大量低质量短文档，即使总量巨大也无法弥补质量缺陷。简单的启发式过滤（如丢弃 <100 字符的文档）不足以解决根本问题。

> **给初跑者的建议**：如果你在训练自己的模型，不要急于收集更多数据。先在小规模（如 d12）上对比不同数据集的 validation loss，这比任何架构改动的影响都大。nanochat 的经验表明，换数据集带来的 27% 训练时间缩减，远超 FP8（5%）、Flash Attention 3（9%）等所有工程优化的总和。

### Data Preparation

The raw data comes from NVIDIA's Nemotron-ClimbMix dataset, which was processed using NVIDIA NeMo Curator — a GPU-accelerated data curation toolkit that applies quality filtering, deduplication, and language identification at scale. Karpathy then repackaged it into simple parquet shards:

- Each shard is ~100MB after zstd compression (zstd level 3, `use_dictionary=False`)
- Row group size of 1024 for efficient streaming（选择 1024 而非 HuggingFace 默认的 1000，因为 2 的幂次在分布式数据加载中更友好）
- Shuffled with seed=42 before packaging
- Hosted on HuggingFace for on-demand download（支持 4 并行 worker 下载，指数退避重试）

注意：ClimbMix 源数据以 GPT-2 tokenizer 的 token ID 形式存储，repackage 时需要先解码回文本再重新分片。这个细节记录在 `dev/repackage_data_reference.py` 中。

### Shard Structure

```
shard_00000.parquet
shard_00001.parquet
...
shard_06542.parquet  # Last shard reserved as validation set
```

The dataset download script (`nanochat/dataset.py`) manages parallel downloading of shards. During training, all shards except the last are used for training; the last shard is held out for validation. DDP 环境下，每个 rank 读取不同的 row group（`start=rank, step=world_size`），实现数据并行。

### Training Tokens & Compute-Optimal Scaling

训练 token 数量不是随意设定的，而是通过 **scaling laws** 精确计算的。

核心参数：`--target-param-data-ratio`（默认 10.5），含义是：

```
target_tokens = target_param_data_ratio × num_scaling_params
```

其中 `num_scaling_params` = transformer 矩阵参数 + lm_head 参数（不含 embedding 和 value embedding）。

**为什么是 10.5 而不是 Chinchilla 的 20？**

nanochat 的架构大量使用了 Value Embeddings（每隔一层注入一个），导致总参数量中有很大比例是 embedding 表。作者通过 scaling laws 实验发现，使用 Kaplan-style 的参数计数方式（只算投影矩阵 + lm_head）得到的 token:param 比率最稳定，约为 10.5。这个比率在不同 FLOPs 预算下保持一致（10.3-11.2），而 Chinchilla-style（算全部参数）的比率则随规模变化剧烈（3.0-4.0）。

> **给初跑者的建议**：不要盲目套用 Chinchilla 的 20:1 比率。你的最优比率取决于架构（尤其是 embedding 占比）和参数计数方式。在小规模上跑 scaling laws 实验（nanochat 在 1e18-1e19 FLOPs 范围内做了系统性扫描）是确定这个比率的正确方法。

**Batch Size 自动缩放**

Batch size 也不是固定的，而是根据 Power Lines 论文（arXiv:2505.13738）的发现自动计算：

```
B_opt ∝ D^0.383
```

以 d12 模型的 B=2^19 作为参考点，更大的模型自动获得更大的 batch size。0.383 指数意味着 batch size 增长很慢：10 倍的 token 只需要 ~2.4 倍的 batch size。

| Depth | Scaling Params | Target Tokens | Auto Batch |
|-------|---------------|---------------|------------|
| d=8   | 42M           | 0.44B         | 2^18 = 262K |
| d=10-16 | 70M-235M    | 0.7B-2.5B     | 2^19 = 524K |
| d=18-26 | 324M-918M   | 3.4B-9.6B     | 2^20 = 1.05M |
| d=32-50 | 1.7B-6.2B   | 17.6B-65.6B   | 2^21 = 2.1M |

---

## 2. SFT Datasets

SFT (Supervised Fine-Tuning) uses a **TaskMixture** — a shuffled mixture of multiple datasets loaded from `scripts/chat_sft.py`. The mixing is deterministic (seed=42) to ensure uniform distribution of task types throughout training.

### The Art of SFT Data Mixing

SFT 数据配比是一门平衡的艺术。nanochat 的混合策略体现了几个深层考量：

**1. 通用能力为主体，专项能力靠过采样**

SmolTalk（460K 行）占据了混合数据的绝对主体（~43%），这保证了模型的通用对话能力。而 MMLU 和 GSM8K 通过多 epoch 过采样来强化特定能力，而不是简单地增加数据量。这个设计选择背后的逻辑是：通用对话数据的多样性本身就是价值，重复看同样的对话收益递减很快；但结构化的知识和推理任务（选择题、数学题）的模式更固定，多看几遍确实能加深掌握。

**2. Epoch 数量是调出来的，不是猜的**

LOG.md 记录了 SFT 数据配比的调优过程（2026-02-16）：MMLU 从 1 epoch 调到 3 epoch，GSM8K 从 2 epoch 调到 4 epoch，这些都是通过 sweep 实验确认的最优值。更多的 epoch 并不总是更好 — 过度重复会导致过拟合到特定的题目模式而非学会底层能力。

**3. 小数据集的"存在感"问题**

Identity Conversations 只有 ~1000 行，在百万级的混合数据中几乎可以忽略。但它被过采样 2x（出现两次），确保模型能学到一致的人格特征。这是一个经典的 SFT 技巧：对于你特别在意的行为模式，即使数据量小，也要通过过采样保证足够的训练信号。

### Dataset Mixture Summary

| Dataset | Raw Size | Epochs | Effective Size | 占比 | Purpose |
|---------|----------|--------|---------------|------|---------|
| SmolTalk | 460K rows | 1 | ~460K | ~43% | General conversation |
| Identity Conversations | ~1K rows | 2 | ~2K | <1% | Impart personality |
| MMLU (auxiliary_train) | ~100K rows | 3 | ~300K | ~28% | Multiple-choice knowledge |
| GSM8K (main) | ~8K rows | 4 | ~32K | ~3% | Math reasoning & tool use |
| SimpleSpelling | 200K rows | 1 | 200K | ~19% | Word spelling |
| SpellingBee | 80K rows | 1 | 80K | ~7% | Letter counting |
| **Total** | | | **~1.07M** | | |

> **给初跑者的建议**：SFT 数据配比没有万能公式。关键原则是：(1) 通用对话数据占主体，保证不丢失流畅性；(2) 专项任务通过 epoch 数控制强度，而非无限堆数据；(3) 在你的目标评估指标上做 sweep，不要凭直觉。nanochat 的 `--mmlu-epochs` 和 `--gsm8k-epochs` 参数就是为此设计的。

---

### 2.1 SmolTalk

**Source**: `HuggingFaceTB/smol-smoltalk`

SmolTalk is a general-purpose conversational dataset by HuggingFace. The "smol" version is specifically designed for smaller models. Each row contains a multi-turn conversation with alternating user/assistant messages.

**为什么选 SmolTalk 而不是其他对话数据集？**

对于 GPT-2 规模的小模型，数据的"难度匹配"至关重要。HuggingFace 的 SmolTalk "smol" 版本专门为小模型设计，对话复杂度适中。如果用 ShareGPT 或 UltraChat 等面向大模型的数据集，小模型会在过于复杂的推理链上浪费大量训练信号，学到的是"模仿长回复的形式"而非"掌握对话的实质"。

**Format**:
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

---

### 2.2 Identity Conversations

**Source**: `karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl`

A small (~1000 rows) synthetic dataset of identity conversations designed to give nanochat a distinct "personality." This file is downloaded on-demand during SFT.

The dataset is oversampled 2x in the training mixture (appears twice per epoch) to reinforce the personality traits. Implementation 上非常简洁 — 在 `chat_sft.py` 中同一个 `CustomJSON` 文件被传入两次：

```python
CustomJSON(filepath=identity_conversations_filepath), # 1000 rows
CustomJSON(filepath=identity_conversations_filepath), # 2 epochs of these
```

这是 `TaskMixture` 设计的一个巧妙之处：过采样不需要特殊参数，只需把同一个 task 多传几次。

**为什么只有 ~1000 行就够了？**

Identity/personality 数据的目标不是教模型新知识，而是建立一致的自我认知模式（"我是 nanochat"、"我由 Karpathy 创建"等）。这类模式高度重复且结构简单，1000 行 × 2 epochs 已经足够让模型在 ~1.07M 的混合数据中看到足够多的 identity 信号。过多反而会挤占其他任务的训练预算。

---

### 2.3 MMLU (Massive Multitask Language Understanding)

**Source**: `cais/mmlu`

MMLU is a classic benchmark covering 57 subjects (e.g., `college_biology`, `high_school_mathematics`, `jurisprudence`). Each example is a 4-choice multiple choice question.

**为什么用 3 个 epoch？**

MMLU 的 `auxiliary_train` split 有 ~100K 行，3 个 epoch 产生 ~300K 有效样本，占混合数据的 ~28%。这个比例是通过 sweep 实验确定的（LOG.md 2026-02-16：从 1 epoch 调到 3 epoch）。

直觉上，选择题的"格式学习"和"知识记忆"需要不同的重复次数。模型需要先学会"只输出一个字母"的格式约束（1 epoch 足够），然后通过额外的 epoch 加深对知识点的记忆。但超过 3 epoch 后，模型开始记住具体题目而非学会推理，在 test set 上的表现反而下降。

**Question Format** (rendered by `tasks/common.py::render_mc`):
```
Multiple Choice question: What is the capital of France?
- Paris=A
- London=B
- Berlin=C
- Madrid=D

Respond only with the letter of the correct answer.
```

**Assistant Response**: A single letter (A/B/C/D)

**Evaluation**: Exact match against the correct letter.

**Validation set 的配比控制**：验证集使用 `stop=5200` 截断 MMLU test（原始 ~14K 行），以匹配训练集中的任务比例，避免验证 loss 被 MMLU 主导。

---

### 2.4 GSM8K (Grade School Math 8K)

**Source**: `openai/gsm8k`

GSM8K contains ~8K grade school math word problems with step-by-step solutions. It tests multi-step arithmetic reasoning. 使用 4 个 epoch（从最初的 2 epoch 调上来），有效样本 ~32K。

**为什么 GSM8K 需要最多的 epoch（4x）？**

GSM8K 只有 ~8K 行，是所有真实数据集中最小的。但数学推理能力对小模型来说极其困难 — 模型需要学会：(1) 理解自然语言中的数学关系；(2) 分步推理；(3) 正确使用工具调用格式。4 个 epoch 让模型在 ~32K 有效样本中反复练习这些模式。这个数量级与 MMLU 的 ~300K 相比仍然很小，但数学推理的"模式密度"远高于选择题 — 每道题都包含完整的推理链。

**Tool Call Format**: Uses `<<expression=result>>` syntax for embedded calculator calls. The model learns to use Python tools mid-reasoning.

**Example**:
```
Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes
of babysitting. How much did she earn?

Answer:
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10
```

**Evaluation**: Extracts the numerical answer after `####` and compares with the ground truth.

---

### 2.5 SimpleSpelling

**Source**: Synthetic (generated in `tasks/spellingbee.py`)

200K rows of simple spelling tasks:
- Input: `"spell the word 'apple'"`
- Target: `"apple"`

**为什么需要一个"看起来无聊"的拼写任务？**

这个任务对人类来说毫无难度，但对小型 LLM 来说却出奇地困难。Tokenizer 将文本切分为 subword token，模型在训练中从未直接接触过字符级别的信息。当你问 GPT-2 规模的模型"spell apple"时，它需要将 token 级别的表示"逆向解码"为字符序列 — 这是一个非平凡的能力。

200K 行的规模（占混合数据 ~19%）看似过多，但这反映了一个重要原则：**对于模型天然薄弱的能力，需要不成比例的训练信号**。拼写能力不会从通用对话数据中自然涌现，必须显式训练。

---

### 2.6 SpellingBee

**Source**: Synthetic (generated in `tasks/spellingbee.py`)

80K rows of letter counting tasks. This builds on SimpleSpelling by adding multi-step counting/reasoning.

**Example**:
```
User: How many r are in "strawberry"?
Assistant: Let me count: s-t-r-a-w-b-e-r-r-y → 3 r's. The answer is 3.
```

**Multi-language Support**: Questions are rendered in English, Spanish, Chinese, and Korean to improve robustness.

**Template Examples** (30+ variations):
- `"How many {letter} are in the word {word}?"`
- `"Count the number of {letter} in {word}"`
- `"{word}中有多少个{letter}"` (Chinese)
- `"{word}에 {letter}가 몇 개 있나요"` (Korean)

**为什么要多语言模板？**

这不是为了让模型学会中文或韩文（预训练数据以英文为主），而是一种**数据增强**策略。同一个底层任务（数字母）用不同语言的 prompt 表达，迫使模型学习任务本身的逻辑而非记忆特定的英文 prompt 模式。30+ 种模板变体也服务于同样的目的 — 防止模型过拟合到 "How many X are in Y" 这一种句式。

**SimpleSpelling vs SpellingBee 的分工**

两者看似相似但训练目标不同：SimpleSpelling（200K）训练的是字符级别的"解码"能力（token → characters），SpellingBee（80K）训练的是字符级别的"推理"能力（逐字遍历 + 计数）。前者更基础所以数据量更大，后者更复杂但建立在前者之上。

> **给初跑者的建议**：合成数据是小模型 SFT 的秘密武器。当你发现模型在某个能力上表现差时，不要只想着找更多真实数据 — 考虑生成针对性的合成数据。nanochat 的 spelling 任务就是一个很好的例子：用简单的模板 + 词表就能生成无限量的高质量训练数据，精确地补强模型的薄弱环节。

---

## 3. Data Handling in Training

### BOS-Aligned Best-Fit Packing

nanochat 的数据打包策略经历了多次迭代（详见 LOG.md 2026-01-13），最终选择了 **BOS-aligned best-fit** 方案。理解这个选择需要先理解问题：

**问题**：多个文档/对话需要打包进固定长度的序列槽。朴素方法（连续拼接后 reshape）会导致某些行从文档中间开始，没有 BOS token，模型在训练时缺乏完整上下文。

**作者尝试过的方案**：

| 方案 | 利用率 | 裁剪浪费 | 填充浪费 | 评价 |
|------|--------|---------|---------|------|
| Greedy-Crop | 100% | 39.4% | 0% | 简单但浪费太多 token |
| Greedy-Pad | 78% | 23.0% | 22% | 填充浪费计算资源 |
| First-Fit Decreasing | 99.7% | 23.0% | 0.3% | 接近最优但实现复杂 |
| **BestFit-Crop（采用）** | **100%** | **34.6%** | **0%** | 平衡点：100% 利用率 + 较低裁剪 |

**Pretraining** 使用 BestFit-Crop：从 buffer（1000 个文档）中贪心选择能完整放入的最大文档，放不下时裁剪最短文档填满。~35% 的 token 被裁剪丢弃，但每个训练位置都有完整的 BOS 上下文。

**SFT** 使用 BOS-aligned best-fit pad（`sft_data_generator_bos_bestfit`）：
1. Each conversation starts at a BOS (beginning-of-sequence) token boundary
2. Conversations are packed into sequence slots using a best-fit algorithm（从 buffer 中选最大的能放下的对话）
3. When no conversation fits, the remaining slot is **padded** (not truncated)
4. Padding positions have targets masked with -1 (ignored by cross-entropy)
5. Loss masking: `mask=1` for assistant completions only; user prompts, BOS, special tokens, tool outputs 的 mask=0

SFT 选择填充而非裁剪，因为对话的完整性比 token 利用率更重要 — 裁剪一个对话会破坏多轮对话的逻辑连贯性。

> **给初跑者的建议**：BOS 对齐看似是小细节，但作者发现它会改变 validation loss 的绝对值（因为每个 token 都能看到完整上下文，loss 会"假性"降低）。如果你在对比实验中切换了数据加载策略，之前的 loss 数值就不再可比。这是一个容易踩的坑。

### Task Mixing

The `TaskMixture` class (`tasks/common.py`) concatenates all task examples into one shuffled list:

```python
# Build flattened index of all (task, local_index) pairs
for task_idx, task_length in enumerate(self.lengths):
    for local_idx in range(task_length):
        self.index_map.append((task_idx, local_idx))

# Deterministically shuffle
rng = random.Random(42)
rng.shuffle(self.index_map)
```

This ensures tasks are interleaved uniformly throughout training, preventing the model from learning a curriculum.

**为什么要均匀混合而非课程学习？**

课程学习（先简单后困难）在某些场景下有效，但对 SFT 来说有一个致命问题：如果模型先学完所有 SmolTalk 再学 MMLU，它会在学 MMLU 时遗忘对话能力（灾难性遗忘）。均匀混合确保模型在整个训练过程中持续接触所有任务类型，是对抗遗忘的最简单有效方法。

过采样的实现也很优雅：想让某个 task 出现 N 次，只需在 task 列表中传入 N 次同一个对象。`TaskMixture` 的 docstring 明确写道：*"Fun trick: if you wish to oversample any task, just pass it in multiple times in the list."*

---

## 4. Dataset Download

### Pretraining Data

```bash
# Download 170 shards (sufficient for GPT-2 capability, ~7B tokens)
python -m nanochat.dataset -n 170

# Download all shards (full 400B dataset, for larger models)
python -m nanochat.dataset -n 6542
```

Data is cached to `~/.cache/nanochat/base_data_climbmix/`. 下载支持断点续传（已存在的 shard 会跳过），使用原子写入（先写 `.tmp` 再 rename）防止部分下载的损坏文件。

> **给初跑者的建议**：不需要下载全部 6543 个 shard。对于 GPT-2 规模（d24），只需 ~150 个 shard（~7B tokens）。先用 `-n 170` 开始，后续需要更多数据时再增量下载。

### SFT Data

SFT datasets (except Identity Conversations) are loaded on-demand via HuggingFace `datasets` library. Identity Conversations is downloaded via curl during `speedrun.sh` execution.

---

## 5. Dataset Historical Notes

| Date | Change | Impact |
|------|--------|--------|
| Pre-2026 | FinewebEdu-100B used for pretraining | 基线 |
| Jan 2026 | nanochat repository initialized | — |
| Jan 15, 2026 | 尝试 OLMo dolma3_mix-6T | CORE -47%，质量问题严重 |
| Feb 10, 2026 | 尝试多种数据混合实验 | 均为负面结果 |
| Feb 16, 2026 | SFT 数据配比调优：MMLU 1→3 epoch, GSM8K 2→4 epoch | 通过 sweep 确认最优 |
| Feb 17, 2026 | 尝试 vanilla FineWeb（非 EDU） | CORE -14% |
| Feb 17, 2026 | 尝试 FinePDFs+DCLM+FineWeb-EDU 混合 | CORE -2% |
| **Mar 4, 2026** | **切换到 ClimbMix-400B** | **训练时间 -27%，模型从 d26 缩小到 d24** |

The `repackage_data_reference.py` script in the `dev/` directory documents exactly how both datasets were prepared and uploaded to HuggingFace.

---

## 6. Key Takeaways for Practitioners

总结 nanochat 在数据选择上的核心经验：

1. **数据质量是最大的杠杆**。ClimbMix 带来的 27% 训练加速超过了所有架构和工程优化的总和（FP8 +5%, FA3 +9%, 各种 optimizer 改进 <1%）。在投入时间优化模型架构之前，先确保你的数据是最好的。

2. **Scaling laws 指导一切**。训练 token 数、batch size、weight decay 都不是拍脑袋决定的，而是通过 scaling laws 实验推导出来的。在小规模上跑 scaling laws（nanochat 用 d12 作为参考点）是最值得的前期投资。

3. **SFT 配比需要实验**。没有万能的混合比例。nanochat 的 MMLU 3 epoch、GSM8K 4 epoch 是通过 sweep 找到的，不同的模型规模和目标可能需要不同的配比。

4. **合成数据补强短板**。对于模型天然薄弱的能力（如字符级操作），合成数据是最高效的解决方案。SimpleSpelling 和 SpellingBee 用简单的模板就生成了 280K 高质量训练样本。

5. **警惕隐性偏差**。BOS 对齐改变 loss 的可比性、数据集与评估指标的过拟合风险、裁剪策略对长文档尾部的系统性丢弃 — 这些细节不会导致训练失败，但会悄悄影响你的实验结论。
