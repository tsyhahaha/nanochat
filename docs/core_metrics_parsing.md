# CORE 指标解析 (CORE Metric Analysis)

> **关联研究文件**: `dev/estimate_gpt3_core.ipynb`

---

## 一、什么是 CORE 指标？

[CORE 指标](https://arxiv.org/abs/2406.11794)（提出于 DCLM 论文中）是一个**综合基准**评估体系。它在 22 个评测任务上对预训练语言模型进行考核，并最终输出一个单一的分数来衡量模型的通用能力。

评估涵盖了五大关键能力维度：
| 类别 | 包含的评测任务 |
|------|------|
| **世界知识** | Jeopardy, ARC Easy, ARC Challenge, BigBench QA Wikidata |
| **语言理解** | HellaSwag (0-shot & 10-shot), LAMBADA, Winograd, Winogrande, BigBench Language ID |
| **常识推理** | COPA, CommonsenseQA, PIQA, OpenBookQA |
| **符号问题求解** | BigBench Dyck, Operators, CS Algorithms, Repeat Copy Logic, AGI Eval LSAT-AR |
| **阅读理解** | SQuAD, CoQA, BoolQ |

### 核心机制：准确率中心化 (Centering)

CORE 的关键创新点在于对各任务的准确率进行了**中心化 (centering)** 计算，以消除不同任务间随机猜测难度的差异：

$$\text{centered accuracy} = \frac{\text{accuracy} - \text{baseline}}{1 - \text{baseline}}$$

*其中 `baseline` 是该任务随机猜测的准确率（例如 4 选 1 的选择题 baseline 取 0.25）。*

最终的 CORE 分数，即为上述 22 个中心化后准确率的**平均值**。

---

## 二、为什么要估算 GPT-3 的 CORE 分数？

在 `nanochat` 的研发和评测过程中，我们需要与 OpenAI 的经典 GPT-3 模型族（"Language Models are Few-Shot Learners", 2020）进行横向对比，以衡量当前微型模型的水平。

然而遇到了一个关键障碍：
1. **时间错位**：GPT-3 发布于 2020 年，当时 CORE 指标还不存在。
2. **闭源限制**：模型未公开发布其原始权重，使得直接对 GPT-3 跑一遍 CORE 的 22 个任务变得不可能。

**解决方案**：提取已知模型（GPT-2）与目标基准之间的任务重叠，通过建立**回归模型拟合**的方式，间接推算和估计 GPT-3 在整体 CORE 测试中的预计分数。

---

## 三、估算方法论

### 1. 任务重叠选取
为了保证评估严谨性，首先需要找出在 GPT-3 论文与 CORE 指标中**评估方式（Few-shot 配置等）一致**的共同任务。

基于非常保守的过滤标准（排除 0-shot 和 K-shot 混合对比的任务，以及对 shot 数极度敏感的任务），最终筛选出了 6 个作为核心特征的任务：
- HellaSwag (0-shot)
- LAMBADA (0-shot)
- HellaSwag (10-shot)
- PIQA
- ARC Easy
- ARC Challenge

### 2. 校准数据：引入 GPT-2 家族
GPT-2 家族同时满足两大条件：既在现今有了实际测出的完整 CORE 分数，同时在当年的 GPT-3 论文表格中也报告了这些任务的原始分数。

| 模型 | 参数量 | 层数 (Layers) | 6 项可比任务中心化均值 | 实际 CORE 分数 |
|------|--------|-------------|----------------------|--------------|
| GPT-2 | 124M | 12 | 0.1505 | 0.1139 |
| GPT-2 Medium | 355M | 24 | 0.2448 | 0.1849 |
| GPT-2 Large | 774M | 36 | 0.2991 | 0.2146 |
| GPT-2 XL | 1558M | 48 | 0.3639 | 0.2565 |

### 3. 多维拟合与回归策略
鉴于我们只有 4 个数据点且有多达 6 个特征，采用了多种策略进行验证确保鲁棒性：
- **简单平均线性回归**：仅用 6 任务均值拟合，得出 $CORE \approx 0.6639 \times avg\_centered + 0.0168$。$R^2 = 0.9960$。
- **Ridge 多变量回归**：使用 L2 正则化收缩各单项任务的权重（取 $\alpha=0.01$），保持了极高的拟合优度，防止截距突发。
- **单指标回归**：测试各个单一任务预测整体 CORE 的能力。

---

## 四、估算结果与发现

通过将 GPT-3 论文附录表格汇报的基础准确率输入构建好的 Ridge / 均值 回归混合模型中，得出了最终的 GPT-3 家族的 CORE 估算图景：

### GPT-3 最终 CORE 估算分

| 模型 | 参数量 | 层数 (Layers) | 最终估算均值 (CORE) |
|------|--------|-------------|-------------------|
| GPT-3 Small | 125M | 12 | **0.1484** |
| GPT-3 Medium | 350M | 24 | **0.2159** |
| GPT-3 Large | 760M | 24 | **0.2659** |
| GPT-3 XL | 1.3B | 24 | **0.2914** |
| GPT-3 2.7B | 2.7B | 32 | **0.3292** |
| GPT-3 6.7B | 6.7B | 32 | **0.3611** |
| GPT-3 13B | 13B | 40 | **0.3852** |
| GPT-3 175B | 175B | 96 | **0.4272** |

### 关键洞察 (Insights)

1. **同等尺度下的全面提升**
   当比较同一参数量级时的估计值，得益于数据量和数据质量的提升，GPT-3 的同尺寸模型对应的 CORE 估测值相比 GPT-2 有了 14% 到 30% 不等的跨越式增长。
   > 如 125M 量级：GPT-2 仅为 `0.1139`，而估算的 GPT-3 Small 则达到了 `0.1484`。

2. **单一最佳预测指标：PIQA**
   在单任务回归分析中发现，在不考虑全部 6 个重叠任务的情况下，仅仅凭借 **PIQA**（Physical Interaction: Question Answering）这一个任务分数去预测最终的整体 CORE 分数是最为准确的（$R^2=0.9961$）。
   这为后续研究提供了一个有力的速算代理：**如果需要极低成本快速判断一个模型在包含 22 个题库的 CORE 上大致能拿多少分，仅评测 PIQA 就具备极高参考价值。**

3. **模型限制须知**
   当前所有的 GPT-3 CORE 分数均为基于回归模型的估算值。尽管其在 GPT-2 与 175B 的渐近趋势上十分契合，但受到 4 个校准样本和对 few-shot 配置近似匹配的主观影响，不可直接视为实测数据。在发布 `nanochat` 的 benchmark 对比进度时，**应当显著标明 GPT-3 是 "Estimated (估算值)"**。

---

## 五、衍化指标：ChatCORE (针对 SFT 指令微调阶段)

前面详述的 CORE 指标及其对 GPT 系列的估算，主要着眼于**预训练 (Pretrain)** 阶段的基础模型能力。然而，进入**监督微调 (SFT)** 流程后，模型的“续写”能力评价变得不再重要，评价重点转为**问答、推理与遵循规则的对话能力**。

为了兼顾指代体系的统一，并准确测量对齐效果，nanochat 衍生出了专用的 **ChatCORE** 指标体系。

### 1. 评估任务的区别 (Pretrain CORE vs. ChatCORE)

由于单纯评估补全或续写的预训练集（如 LAMBADA / HellaSwag）已不再符合对话机器人的真实场景，ChatCORE 将评估池替换为了 6 个经典且最具代表性的当代生成与推理核心任务：

- **基础与通用知识**：ARC-Easy, ARC-Challenge
- **多学科广度综合**：MMLU
- **逻辑与数学计算**：GSM8K
- **代码生成**：HumanEval
- **指令约束生成**：SpellingBee

### 2. 计算机制：延续 Mean Centered Accuracy

虽然底层任务全换了，但计分逻辑与预训练的 CORE 一脉相承。如果不做中心化，不同任务因选择（4选1, 2选1）和自由生成的难度分布不同，简单加总求平均会导致指标失真。

算法完美沿袭了中心化 (Centering) 的核心思想：
1. **去随机化**：每项任务各自减去该任务的基础猜测胜率（Baseline）：
   $$ \text{centered\_acc} = \frac{\text{acc} - \text{baseline}}{1 - \text{baseline}} $$
2. **聚合平均**：最终取以上 6 个特定测试任务的算术平均值，使得最终的 ChatCORE 得分能严格规整在 **0 ~ 1.0** 区间内：
   - `0.0`：模型在所有考试中仅体现出了随机蒙题的效果。
   - `1.0`：模型在代码、数学、通识六大领域表现出绝对正确的理想状态。

### 3. 在项目中的工程化跑通 (`chat_eval.py`)

ChatCORE 指标被深度解耦并整合在 nanochat 的评估及开发流中：
- **实时监控**：在执行 `chat_sft.py` 对齐训练中，通过指定 `--chatcore-every=200`，可以令日志系统（如 wandb）随时记录 SFT 进程对模型常识推理能力带来的提升动态。
- **离线单独探查**：可以直接调用自带工具引擎： `python -m scripts.chat_eval`，针对当前任意给定的 `checkpoint` 或第三方模型，实施完全本地化的 ChatCORE 测试。
