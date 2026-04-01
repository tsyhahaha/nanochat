# nanochat——LLM pretrain 全流程解析

nanochat 是一个设计极简但极具深度的开源大模型训练框架，核心目标是以最小的认知负担和最低的算力成本（约$50）在一个 8xH100 节点上训练出具备 GPT-2 级别能力的模型。它剥离了主流框架中庞杂的抽象，提供了一套端到端的纯净实现。

本文将从专业且易懂的视角，全面解析 nanochat 项目在 LLM 预训练全流程中的核心设计与工程巧思，并针对其中的关键技术细节辅以源码级别的解读。

---

## 一、Model and Data（模型架构与数据）

### 1. 深度创新的 GPT 架构 (`gpt.py`)
nanochat 并没有一成不变地照搬标准 Transformers，而是引入了大量前沿研究成果，使其成为一个**极度计算最优**（compute-optimal）的现代大模型。相对于 2019 年传统的 OpenAI GPT-2 架构，`nanochat` 进行了彻底的重构与升级：

#### (1) 基础组件的现代化替换
*   **废弃绝对位置编码，拥抱 RoPE**：彻底移除了绝对位置嵌入，在 Attention 层内部的 Query 和 Key 上应用了旋转位置编码 (RoPE)，极大地增强了模型对相对位置的感知能力和长度外推性。
*   **LayerNorm 转向极简版 RMSNorm**：使用了更高效的 RMSNorm（无需减去均值），并且**完全去除了可学习参数**（无学习的 scale 和 bias），节省显存且不掉点。
*   **全网移除 Bias（偏置项）**：所有的 Linear 层（QKV、MLP、LM Head）的偏置项全部设为 `bias=False`，提高计算速度并使优化器行为更加纯粹。
*   **激活函数升级**：MLP 层的激活函数从 GELU 换为了计算更轻量、对大模型更有效的 `Squared ReLU`（即 `F.relu(x).square()`）。
*   **Input Embedding 和 LM Head 权重解绑**：输入词向量表（`wte`）和输出投影矩阵（`lm_head`）完全解绑（Untied Weights），各自使用最适合的初始化方差。
*   **自定义强控制的 Mixed Precision Linear**：规避了 `torch.autocast`，自定义 Linear 类让主权重始终保持在 FP32 以追求 Optimizer 精确度，仅仅在前向时被实时转换成计算数据类型（如 `bf16` 或 `fp8`）。

#### (2) 注意力机制 (Attention) 的飞跃
*   **引入 GQA (Group-Query Attention)**：多个 Query 头共享同几组 Key 和 Value，大幅度降低推理阶段的 KV Cache 显存占用，提升吞吐率。
*   **引入 QK Norm 与锐化因子**：在计算点积前，额外对 Query 和 Key 进行了 RMSNorm 归一化（`q, k = norm(q), norm(k)`），并乘以 `1.2` 经验缩放因子，使注意力分布更加锐利（Sharper Attention），缓解注意力突刺。
*   **交替滑动窗口注意力 (Sliding Window Attention)**：并非所有层都执行全局注意力。引入了基于 `window_pattern="SSSL"` 的配置，前三层只关注短距离的局部窗口（Quarter Context），仅第四层计算全局上下文，大幅降低长文本的 FLOPs。
*   **底层融合 Flash Attention 3**：原生地将前向与反向分发到 FA3，并在缺乏适配显卡时智能软降级到 PyTorch原生 SDPA。

#### (3) 激进的架构内部数据流魔改（附源码原理解析）
nanochat 引入了多个针对梯度流和特征表达的“黑魔法”，这也是它能用极少算力逼近 GPT-2 能力的秘诀：

**A. Smear（特征涂抹/局部 bigram 注入）**
这是 nanochat 独有的特征，在进入第一层自注意力之前，它通过一个轻量的门控网络，向当前 token 中混入了前一个 token 的 Embedding，获得极具价值的 bi-gram 关联特征。
```python
# nanochat/gpt.py - GPT.forward()
if T > 1:
    # 只取 embedding 前 24 个通道计算门控权重，大大节省计算量
    gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
    # 巧妙的错位拼接：将 x[:, :-1] (前一个token) 混合到 x[:, 1:] (当前token)
    x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
```

**B. Value Embeddings 门控注入 (ResFormer 风格)**
在特定的交替层中，将最原始的词向量输入，通过输入门控按比例直接添加到 Value 向量中，防止深层网络“遗忘”最原始的语义。
```python
# nanochat/gpt.py - CausalSelfAttention.forward()
if ve is not None:
    # ve 是当前层的原始词向量输入 (Value Embedding)
    ve = ve.view(B, T, self.n_kv_head, self.head_dim)
    # 计算当前 token 混合原始 ve 的门控比例，值域放缩至 (0, 3)
    gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    # 显式地将词汇表征注入 Value，强化深层的原始语义召回能力
    v = v + gate.unsqueeze(-1) * ve
```

**C. 动态残差缩放与 x0 (初始词向量) 注入**
废弃了传统的死残差 `x = x + block(x)`，取而代之的是 `x = resid_lambdas[i]*x + x0_lambdas[i]*x0 + block(x)`。每层不仅有独立可学习的标量缩放主残差流（防梯度爆炸），还会把第一层最纯粹的词嵌入（`x0`）动态注入到每一层中。

**D. Backout（特征回退/低级特征减法）**
在网络中间层保存一份特征缓存。在输出前，强制减去这个中间层特征，其意图是让 LM Head 分类器丢弃低级语法特征，专注于高级语义理解。
```python
# nanochat/gpt.py - GPT.forward()
backout_layer = n_layer // 2  # 取网络正中间层的输出
x_backout = None
for i, block in enumerate(self.transformer.h):
    # ... Transformer Block ...
    if i == backout_layer:
        x_backout = x  # 记录中间层特征

# 在过最后的语言头 (LM Head) 之前，回退并减去之前记录的低级特征
if x_backout is not None:
    x = x - self.backout_lambda.to(x.dtype) * x_backout
```

### 2. 流式数据管道 (`dataset.py` & `dataloader.py`)
- **数据集**：摒弃 FinewebEdu-100B，全面拥抱 **ClimbMix-400B**，保证高质量多领域混合预训练。
- **最佳适应（Best-fit）数据打包**：在 `dataloader.py` 中，使用启发式的拼包算法，自动将多篇短文本压缩拼接进同一个 Sequence 中，并恰当插入 `<|bos|>` 分隔符，最大化利用计算 FLOPs 操作而不会跨文档污染算力。

---

## 二、Tokenizer Training（分词器训练）

分词器决定了模型如何“看”这个世界。nanochat 在 Tokenizer 上的设计既复古又极致。

### GPT-4 级正则划分改良
大模型的 Tokenizer 会先用正则切分单词再做 BPE 计算。nanochat 做了一个极度重要的修改，特别针对较小词表（如 32K）做了深度优化：

```python
# nanochat/tokenizer.py
# 核心变化：将数字匹配的正则从 GPT-4 默认的 \p{N}{1,3} 修改为 \p{N}{1,2}
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```
**解析**：标准 GPT-4 允许一次将 1 到 3 位数字作为一个整体分块。这在百万级别词表中无所谓，但在 32K 规模的词表中，“占用百级数字（如152、985）”极其奢侈，浪费了大量本可用于汉字或常用词根的 Token 容量。改为 `1,2` 位数切分，是**微小模型资源分配的敏锐直觉体现**。

---

## 三、Model Training（模型训练）

`base_train.py` 中的训练逻辑是整个项目的灵魂，极具“数学美感”。它抛弃了玄学的炼丹术，绝大部分超参数均通过**Scaling Laws（缩放定律）**严格推导得出。

### 1. 自动化的超参数推导（Scaling Laws 落地）
启动训练你只需要设定模型层数 `--depth`，其余一概由代码在内部通过经验公式推演：

```python
# scripts/base_train.py (超参数推导演示化简版)

# 1. 训练数据量的绝对指引法则：Chinchilla Law
# 最优训练 token 数量（D）约等于算力参数的 10.5 倍
target_tokens = int(10.5 * num_scaling_params) 

# 2. 批次大小推论：Power Lines Law (B_opt ∝ D^0.383)
# 相对于基准深度（d12），Batch Size 应该随 tokens 数据量的平方根非线性放大
batch_size_ratio = target_tokens / D_REF
predicted_batch_size = B_REF * (batch_size_ratio ** 0.383)
total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # 对齐 2 的幂次以优化硬件

# 3. 学习率缩放：AdamW 适配大 Batch Size
# 批处理越大，梯度的方差越小，允许使用更大的学习率 (η ∝ √(B/B_ref))
batch_lr_scale = (total_batch_size / B_REF) ** 0.5 

# 4. T_epoch 学习率不变性论文：推导正确的权重衰减 (Weight Decay)
# λ = λ_ref * √(B/B_ref) * (D_ref/D)
weight_decay_scaled = args.weight_decay * batch_lr_scale * (D_REF / target_tokens)
```
**解析**：通过上述纯理论计算，完全免除了人力对 learning rate（学习率）或 batch size 进行试错（Grid Search），做到零成本获得收敛最佳配置。

### 2. MuonAdamW：前沿混合优化器
在 `optim.py` 中，框架引入了革命性的 **Muon (MomentUm Orthogonalized by Newton-schulz)** 优化器，和 AdamW 进行混合编排。
- **AdamW** 用于 1D 通道（Embeddings, RMSNorm, 标量门控因子）。
- **Muon** 用于所有 2D 矩阵参数（Transformer Attention 和 MLP 权重）。它可以迅速将梯度矩阵向最近的正交矩阵进行修正，使特征以极高维度分散。

```python
# nanochat/optim.py - muon_step_fused() 的核心正交化步骤
X = g.bfloat16()
X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
# Polar Express Sign Method 正交化迭代 (极速版 Newton-Schulz)
for a, b, c in polar_express_coeffs[:ns_steps]:
    A = X.mT @ X
    B = b * A + c * (A @ A)
    X = a * X + X @ B
g = X
```

---

## 四、Model Eval（模型评估）

与其在训练时盯着波动的 Loss，不如建立直接反应模型能力的指标体系 (`core_eval.py`)：

1. **比特每字节 (Bits Per Byte, bpb)**：Loss 具备很强的迷惑性（只要增大词表维度 Loss 就会强行降低）。bpb 是跨越词表设计的终极指标衡量，统一归一化到了字节压缩级别。
2. **CORE Metric (DCLM体系)**：采用多任务零样本/小样本 In-Context Learning。代码不仅实现了推理逻辑，还在每个评价项中算了一个“去中心化基数”：`centered = (accuracy - baseline) / (1.0 - baseline)`，使不同随机基线的任务（比如选对概率分别是 25% 与 50% 的多选题）有了统一可比的 CORE 分数。

---

## 五、Training Speedup（训练加速技术）

低成本的前提是极致的榨干硬件性能。

### 1. 极简的真 FP8 动态缩放 (`fp8.py`)
不用 `torchao` 等沉重库的子类继承劫持，nanochat 给出了堪称教科书级别的 150 行底层 FP8 算子直调实现：

```python
# nanochat/fp8.py
def _to_fp8(x, fp8_dtype):
    """动态张量级缩放（Tensorwise Scaling），以最大性能压缩数据位数"""
    fp8_max = torch.finfo(fp8_dtype).max
    # 找寻整个大矩阵中的最大绝对值点
    amax = x.float().abs().max()
    # 动态把浮点数据投影并撑满整个 [-fp8_max, fp8_max] 分布域，防止下溢出
    scale = fp8_max / amax.double().clamp(min=EPS)
    x_fp8 = (x.float() * scale.float()).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return x_fp8, scale.float().reciprocal() # 返回压缩后的数据与逆尺度尺(用于结果复原)
```
**解析**：在一次前向计算中，矩阵 `Input` 和 `Weight` 均会通过此函数压为 `float8_e4m3fn`，随后调用 `torch._scaled_mm` (底层直接链接 cuBLAS FP8 张量核心) 以近乎 BF16 双倍速度完成计算，且内存带宽消耗直接减半。

### 2. 其它加速特性
- **Flash Attention 3**：`flash_attention.py` 支持最新的 FA3。如果没有 Hopper 架构 GPU 则自动回退到 `torch.nn.functional.scaled_dot_product_attention` 并手动构建掩膜以补偿滑动窗口注意力造成的缺陷。
- **Torch Compile 配合 GC 拦截**：全面开启 `torch.compile(dynamic=False)`。因为大批量的异步发射任务会让 Python 自带的垃圾收集器 (GC) 疯狂扫盘引发明显耗时。框架会手动调用 `gc.disable()`，仅在每隔几千步时通过 `gc.collect()` 安全释放一次，展现了极致的工程洁癖。