# nanochat Training Defaults Archive

This document archives the current default training configuration in `nanochat` across the three main stages:

1. Pretrain: `scripts/base_train.py`
2. SFT: `scripts/chat_sft.py`
3. GSM8K RL: `scripts/chat_rl.py`

The tables are sorted by training relevance from high to low:

- High: parameters that directly shape the optimization target, model scale, data budget, or update magnitude.
- Medium: parameters that mainly affect stability, throughput, or evaluation cadence.
- Low: parameters that mainly affect logging, checkpoint naming, or runtime plumbing.

Unless otherwise stated, the values below are the current code defaults, not one-off experiment overrides.

Related reading:

- `docs/training_data.md` for dataset selection and mixture rationale.
- `README.md` for the project's scaling-law philosophy and reproduction flow.

## Stage 1: Pretrain

Entry point: `scripts/base_train.py`

Goal: train the base language model from scratch on ClimbMix, with depth as the main user-facing scaling dial and the rest of the training horizon derived automatically.

### Core defaults

| Impact | Parameter | Default | Meaning | Why this default exists |
|---|---|---:|---|---|
| High | `depth` | `20` | Transformer layer count. Main scale dial for the model. | The project is intentionally built around one primary complexity dial. Depth controls both capacity and the auto-derived width, heads, batch scaling, and token budget. |
| High | `aspect-ratio` | `64` | Base width rule: `model_dim ~= depth * aspect_ratio`. | Keeps width growing with depth so the model family stays compute-optimal instead of requiring hand-tuned width per run. |
| High | `head-dim` | `128` | Target attention head width. | Gives clean head partitioning, aligns well with efficient kernels, and keeps the architecture derived from depth in a stable way. |
| High | `max-seq-len` | `2048` | Maximum training context length. | `2048` is the project's default full-context budget and matches the model/checkpoint/eval pipeline assumptions. |
| High | `window-pattern` | `SSSL` | Attention window layout tiled across layers, where `S` is half-context sliding window and `L` is full context. | Balances quality and efficiency by mixing cheaper local layers with periodic full-context layers. |
| High | `num-iterations` | `-1` | Explicit step count override. `-1` disables the override. | The repo prefers deriving the horizon from scaling laws rather than hand-entering steps for each model size. |
| High | `target-flops` | `-1.0` | Explicit FLOPs-budget override. `-1` disables it. | Kept mostly for scaling-law experiments; disabled by default so ordinary training follows the default token-horizon path. |
| High | `target-param-data-ratio` | `12` | Target tokens-to-scaling-params ratio used to derive total training tokens. | This is the current code-path default for compute-optimal pretraining. The model is meant to derive its token budget from parameter count rather than fixed epochs. |
| High | `device-batch-size` | `32` | Per-device micro-batch size. | Chosen as an aggressive but practical starting point for large-memory GPUs, while still being easy to reduce when VRAM is tight. |
| High | `total-batch-size` | `-1` | Global batch size in tokens. `-1` means auto-compute. | The project auto-scales batch size from model/data scale using a Power-Lines-inspired rule instead of hard-coding one number for all depths. |
| High | `embedding-lr` | `0.3` | Base LR for token embedding parameters. | Embeddings are given a relatively high LR in nanochat's optimizer partitioning so the input representation adapts quickly during pretraining. |
| High | `unembedding-lr` | `0.008` | Base LR for `lm_head` parameters. | Lower than embedding LR to keep output logits updates more stable while still learning quickly. |
| High | `matrix-lr` | `0.02` | Base LR for transformer matrix parameters under Muon. | The project treats matrix parameters as the main trunk and gives them their own optimizer family and LR scale. |
| High | `scalar-lr` | `0.5` | Base LR for scalar/gating parameters such as residual lambdas and `x0` lambdas. | These few parameters often need larger movement than dense matrices to become useful early in training. |
| High | `weight-decay` | `0.28` | Reference Muon weight decay before schedule/scaling adjustments. | Fairly strong regularization at the start helps keep long pretraining runs controlled before the schedule decays it to zero. |
| High | `warmup-steps` | `40` | Number of steps spent linearly warming up LR. | Short warmup avoids unstable first updates without wasting too much of the run on tiny learning rates. |
| High | `warmdown-ratio` | `0.65` | Fraction of training reserved for LR warmdown. | A long decay phase is part of the repo's default recipe for squeezing out final quality from a fixed compute budget. |
| High | `final-lr-frac` | `0.05` | Final LR as a fraction of the initial LR. | Does not decay all the way to zero; keeps a small amount of learning alive near the end while still stabilizing training. |

### Stability and throughput defaults

| Impact | Parameter | Default | Meaning | Why this default exists |
|---|---|---:|---|---|
| Medium | `fp8` | `False` | Enable FP8 training path. | Disabled by default because it requires newer GPUs and a narrower compatibility path; BF16 is the safer default. |
| Medium | `fp8-recipe` | `tensorwise` | FP8 scaling recipe when FP8 is enabled. | `tensorwise` is the faster and recommended FP8 path in the code comments, so it is the default when FP8 is used. |
| Medium | `resume-from-step` | `-1` | Resume from a checkpointed step. | Most runs start fresh; resuming is an explicit recovery action. |
| Medium | `eval-every` | `250` | Run validation BPB every N steps. | Frequent enough to catch regressions early, but not so frequent that validation dominates the wall-clock. |
| Medium | `eval-tokens` | `41943040` | Number of validation tokens used for BPB evaluation. | Large enough to reduce noise in BPB estimates while still staying tractable. |
| Medium | `core-metric-every` | `2000` | Run CORE evaluation every N steps. | CORE is much more expensive than loss evaluation, so it is spaced out. |
| Medium | `core-metric-max-per-task` | `500` | Max examples per CORE task. | Limits eval cost while keeping the metric broad enough to be meaningful. |
| Medium | `sample-every` | `2000` | Print qualitative samples every N steps. | Provides occasional human-readable sanity checks without slowing training too much. |
| Medium | `save-every` | `-1` | Intermediate checkpoint cadence. `-1` means only save at the end. | Saves I/O and storage by default; the final checkpoint is enough for the usual single-run path. |

### Runtime and bookkeeping defaults

| Impact | Parameter | Default | Meaning | Why this default exists |
|---|---|---:|---|---|
| Low | `run` | `dummy` | W&B run name. `dummy` disables real W&B logging. | Makes the script runnable without requiring online experiment tracking. |
| Low | `device-type` | `""` | Device selector; empty means auto-detect. | The repo is designed to adapt to CUDA, CPU, or MPS without changing code for the common case. |
| Low | `model-tag` | `None` | Override checkpoint directory name. | Optional naming hook; not needed for standard runs where `d{depth}` is enough. |

### Derived behavior and hidden defaults

These are not separate CLI arguments, but they are part of the effective default training recipe.

| Impact | Rule | Effective default behavior | Why this matters |
|---|---|---|---|
| High | Width derivation | `model_dim` is computed from `depth * aspect_ratio` and rounded up to a multiple of `head_dim`. | Keeps architecture changes systematic and hardware-friendly. |
| High | Head count derivation | `num_heads = model_dim // head_dim`; `n_kv_head = n_head`. | Ensures the model family stays internally consistent as depth changes. |
| High | Batch-size derivation | If `total-batch-size == -1`, the script predicts a compute-optimal token batch using a reference batch of `2^19` and a `D^0.383` scaling law. | This is a central design choice: batch size is treated as a function of scale, not a fixed knob. |
| High | Token-budget derivation | If neither `num-iterations` nor `target-flops` is set, total tokens are derived from `target_param_data_ratio * scaling_params`. | Makes pretraining horizon depend on model scale instead of arbitrary epochs. |
| High | LR scaling with batch size | AdamW and Muon learning rates are multiplied by `sqrt(B / B_ref)` when the auto batch changes. | Larger batches tolerate larger update sizes, so the script rescales automatically. |
| High | Weight-decay scaling | Effective Muon weight decay is scaled by batch size and token horizon, then cosine-decayed to zero. | Keeps regularization strength in a similar regime across model sizes and run lengths. |
| High | Muon momentum schedule | Warms from `0.85` to `0.97`, then decays to `0.90` during warmdown. | Gives the matrix optimizer an aggressive but scheduled momentum profile instead of a flat constant. |
| Medium | Precision auto-detection | Compute dtype comes from hardware detection in `nanochat.common`, typically BF16 on modern CUDA, FP32 otherwise. | Lets one code path run across hardware while still using tensor-core-friendly precision where available. |
| Medium | FP8 eval handling | If FP8 is enabled, evaluation temporarily swaps FP8 linear layers back to BF16/standard linear behavior. | Avoids evaluation instability from doing metrics in FP8. |
| Medium | Gradient accumulation | Global batch is reached by accumulating `total_batch_size / (device_batch_size * max_seq_len * world_size)` micro-steps. | Lets the same effective batch recipe work on both 1 GPU and multi-GPU setups. |
| Medium | Optimizer partitioning | `model.setup_optimizer()` splits parameters into AdamW groups for embeddings/head/scalars and Muon groups for transformer matrices. | nanochat's training recipe depends heavily on using different optimizer dynamics for different parameter types. |

## Stage 2: SFT

Entry point: `scripts/chat_sft.py`

Goal: turn the pretrained base model into a chat-capable model using a deterministic mixture of conversational, reasoning, spelling, and identity data.

### Core defaults

| Impact | Parameter | Default | Meaning | Why this default exists |
|---|---|---:|---|---|
| High | `model-tag` | `None` | Base checkpoint tag to load. | Defaults to the latest/largest available base checkpoint so the common path does not need explicit bookkeeping. |
| High | `model-step` | `None` | Base checkpoint step to load. | Defaults to the latest available checkpoint step so SFT starts from the freshest pretrained model. |
| High | `load-optimizer` | `1` | Whether to warm-start optimizer state from the base checkpoint. | Default-on because momentum buffers from pretraining usually make SFT slightly better and smoother. |
| High | `num-iterations` | `-1` | Explicit SFT step limit. `-1` means run one dataset-driven pass through the mixture. | The default SFT run is a full pass over the constructed mixture rather than an arbitrary fixed step count. |
| High | `max-seq-len` | `None` | SFT context length. | `None` means inherit from the pretrained checkpoint; if metadata is missing, the fallback is `2048`. This keeps SFT aligned with the model it is fine-tuning. |
| High | `device-batch-size` | `None` | Per-device SFT micro-batch size. | Inherits from pretrain by default so throughput and memory assumptions carry over cleanly. Fallback is `32`. |
| High | `total-batch-size` | `None` | Global token batch size. | Inherits from pretrain by default so SFT preserves the effective batch recipe unless the user overrides it. Fallback is `524288`. |
| High | `embedding-lr` | `None` | SFT embedding LR. | Inherits from pretrain when possible so SFT starts from the same optimizer scale family instead of inventing a new one. Fallback is `0.3`. |
| High | `unembedding-lr` | `None` | SFT output-head LR. | Inherits from pretrain when possible. If missing in checkpoint metadata, the fallback is `0.004`. This is an important compatibility detail because the current pretrain CLI default is `0.008`. |
| High | `matrix-lr` | `None` | SFT matrix LR. | Inherits from pretrain when possible to keep trunk updates in the same scale regime. Fallback is `0.02`. |
| High | `init-lr-frac` | `0.8` | Initial LR multiplier applied to the inherited/base LRs. | SFT starts near full speed, unlike RL. The model is already well-initialized from pretraining, so only a modest reduction is used. |
| High | `warmup-ratio` | `0.0` | Fraction of SFT spent in warmup. | No warmup by default because the run already starts from a trained checkpoint and does not need the same cold-start caution as pretraining. |
| High | `warmdown-ratio` | `0.5` | Fraction of SFT used for LR decay. | A long decay phase helps preserve chat quality and reduce overfitting late in the relatively short SFT run. |
| High | `final-lr-frac` | `0.0` | Final LR fraction at the end of warmdown. | Unlike pretraining, SFT decays all the way to zero by default to settle the model at the end of the pass. |
| High | `mmlu-epochs` | `3` | Oversampling factor for MMLU auxiliary train data in the mixture. | MMLU is intentionally over-sampled to teach multiple-choice format and broad factual knowledge beyond plain conversation. |
| High | `gsm8k-epochs` | `4` | Oversampling factor for GSM8K train data in the mixture. | GSM8K gets even more repetition because math reasoning and tool use are sparse but high-value capabilities for a small chat model. |

### Stability, evaluation, and throughput defaults

| Impact | Parameter | Default | Meaning | Why this default exists |
|---|---|---:|---|---|
| Medium | `eval-every` | `200` | Validation BPB cadence. | Frequent enough to track SFT progress, still much cheaper than full chat-task evaluation. |
| Medium | `eval-tokens` | `20971520` | Validation token budget for BPB. | Uses a substantial sample to make BPB less noisy without dominating runtime. |
| Medium | `chatcore-every` | `200` | ChatCORE evaluation cadence. | SFT quality needs to be measured on chat tasks relatively often, but not every step. |
| Medium | `chatcore-max-cat` | `-1` | Max categorical examples per ChatCORE task; `-1` means no limit. | Categorical tasks are cheap enough that the default keeps them full. |
| Medium | `chatcore-max-sample` | `24` | Max generative examples per ChatCORE task. | Caps the expensive sampled tasks so ChatCORE remains tractable during training. |
| Medium | internal `weight_decay` | `0.0` | SFT passes zero weight decay into `setup_optimizer()`. | The script intentionally turns off weight decay during SFT because pretraining already decayed it down and SFT is treated as a shorter adaptation stage. |

### Runtime and bookkeeping defaults

| Impact | Parameter | Default | Meaning | Why this default exists |
|---|---|---:|---|---|
| Low | `run` | `dummy` | W&B run name. `dummy` disables real W&B logging. | Lets SFT run with no tracking setup. |
| Low | `device-type` | `""` | Runtime device selection; empty means auto-detect. | Same convenience path as pretraining. |

### Default SFT data mixture

The data mixture is one of the most important parts of SFT. The script constructs it deterministically with `TaskMixture`, which shuffles all examples using `seed=42`.

| Impact | Component | Default effective amount | Role in training | Why it is included |
|---|---|---:|---|---|
| High | `SmolTalk(split="train")` | 1x | General conversation backbone | Gives the model broad conversational fluency and keeps the chat model from becoming too benchmark-shaped. |
| High | `CustomJSON(identity_conversations.jsonl)` | 2x file inclusion | Identity/personality shaping | Small but intentionally oversampled so the model consistently learns its intended identity. |
| High | `MMLU(subset="all", split="auxiliary_train")` | `3x` by default | Multiple-choice reasoning and knowledge | Teaches answer-format discipline and broad subject recall. |
| High | `GSM8K(subset="main", split="train")` | `4x` by default | Arithmetic reasoning and tool-use patterns | Reinforces step-by-step math and calculator-style interaction before RL. |
| Medium | `SimpleSpelling(size=200000, split="train")` | 1x | Character-level spelling ability | Explicitly teaches a weakness that small token-based LMs do not reliably acquire from generic dialogue. |
| Medium | `SpellingBee(size=80000, split="train")` | 1x | Letter-counting ability | Adds another targeted synthetic task for fine-grained token-to-character behavior. |
| Medium | Validation mixture | SmolTalk test + MMLU test stop `5200` + GSM8K test stop `420` | SFT validation | Keeps validation coverage aligned with the train mixture instead of letting one dataset dominate. |

### Derived behavior and hidden defaults

| Impact | Rule | Effective default behavior | Why this matters |
|---|---|---|---|
| High | Stage input | SFT always loads from `load_model("base", ...)`, not from raw initialization. | SFT is an adaptation stage on top of pretraining, not a standalone training recipe. |
| High | Batch realization | `grad_accum_steps = total_batch_size / (device_batch_size * max_seq_len * world_size)`. | Inherited global batch is enforced through accumulation just like pretraining. |
| High | Stop condition | If `num-iterations == -1`, the train generator stops after it has consumed one pass over the mixed dataset. | Default SFT length is tied to dataset coverage, not arbitrary steps. |
| High | Packing strategy | The SFT dataloader uses BOS-aligned best-fit packing and pads instead of cropping when nothing fits. | Protects training data from token loss and keeps each row aligned to conversation starts. |
| High | Loss masking | Only assistant-completion tokens contribute to loss; user prompts, BOS, tool outputs, and padding are masked to `-1`. | This is the core supervision rule for chat fine-tuning. |
| High | LR schedule | Progress-based schedule: no warmup, constant section, then linear warmdown over the last `50%` to zero. | Because SFT may end by dataset exhaustion rather than known total steps, it schedules by progress instead of absolute step count. |
| Medium | Muon momentum | Linearly ramps from `0.85` to `0.95` over the first `300` steps. | Gives SFT matrix updates a gentle stabilization curve. |
| Medium | Optimizer warm-start semantics | If optimizer state is loaded, SFT restores fresh SFT LRs after loading the checkpointed momentum buffers. | Prevents inheriting pretraining's near-zero end-of-run learning rates by accident. |
| Medium | Precision handling | Like pretraining, SFT uses hardware-driven compute dtype and enables `GradScaler` only for FP16. | Keeps precision policy consistent across stages. |

## Stage 3: GSM8K RL

Entry point: `scripts/chat_rl.py`

Goal: start from the SFT chat model and run on-policy reinforcement learning on GSM8K using a simplified REINFORCE/GRPO-style loop with exact-answer rewards.

### Core defaults

| Impact | Parameter | Default | Meaning | Why this default exists |
|---|---|---:|---|---|
| High | `model-tag` | `None` | SFT checkpoint tag to load. | Defaults to the latest/largest SFT checkpoint, making RL a direct continuation of the current chat model. |
| High | `model-step` | `None` | SFT checkpoint step to load. | Defaults to the last available SFT step so RL starts from the most up-to-date chat model. |
| High | `num-epochs` | `1` | Number of passes over GSM8K train for RL. | Keeps the default run intentionally short and exploratory; RL is expensive and potentially destabilizing. |
| High | `examples-per-step` | `16` | Total number of GSM8K problems consumed per optimization step across all ranks. | Sets how many distinct questions contribute to one policy update and therefore strongly shapes reward diversity and gradient variance. |
| High | `num-samples` | `16` | Number of sampled completions per question during rollout. | Grouped sampling is central to this RL recipe because rewards are centered within the sample group. |
| High | `device-batch-size` | `8` | Maximum forward/sample batch size. | Lower than pretrain/SFT because RL rollout generation is memory-heavy; the script relies on multiple passes when needed. |
| High | `max-new-tokens` | `256` | Maximum completion length per sample. | Long enough for step-by-step GSM8K reasoning but bounded to keep generation cost and memory under control. |
| High | `temperature` | `1.0` | Sampling temperature for rollouts. | RL needs exploration; a greedy or near-greedy default would collapse rollout diversity. |
| High | `top-k` | `50` | Top-k truncation during sampling. | Keeps exploration reasonably broad without allowing the very low-probability tail to dominate. |
| High | `embedding-lr` | `0.2` | Embedding LR for RL. | Still large enough to adapt quickly, but lower than pretraining to avoid destabilizing a useful SFT model. |
| High | `unembedding-lr` | `0.004` | Output-head LR for RL. | Conservative output-head updates help preserve language quality while learning from sparse rewards. |
| High | `matrix-lr` | `0.02` | Matrix LR for Muon during RL. | Keeps the transformer trunk trainable without introducing a separate RL-specific optimizer regime. |
| High | `weight-decay` | `0.0` | Weight decay value passed into `setup_optimizer()`. | RL defaults to no matrix weight decay because the stage is short, reward-driven, and more sensitive to regularization-induced drift. |
| High | `init-lr-frac` | `0.05` | Initial LR multiplier applied to all optimizer groups. | Much smaller than SFT because RL can destabilize a good chat model quickly; the stage starts cautiously. |

### Stability, evaluation, and recovery defaults

| Impact | Parameter | Default | Meaning | Why this default exists |
|---|---|---:|---|---|
| Medium | `eval-every` | `60` | Run GSM8K `pass@k` eval every N training steps. | Frequent enough to monitor RL progress without paying the evaluation cost every step. |
| Medium | `eval-examples` | `400` | Number of test problems used for each `pass@k` evaluation. | Provides a decently sized progress signal while keeping sampled evaluation bounded. |
| Medium | `save-every` | `60` | Checkpoint cadence. | RL is fragile enough that periodic recovery points are useful, unlike the end-only default in pretraining. |

### Runtime and bookkeeping defaults

| Impact | Parameter | Default | Meaning | Why this default exists |
|---|---|---:|---|---|
| Low | `run` | `dummy` | W&B run name. `dummy` disables real W&B logging. | Lets RL run even when experiment tracking is not configured. |
| Low | `device-type` | `""` | Runtime device selection; empty means auto-detect. | Same convenience path as the other stages. |

### Derived behavior and hidden defaults

| Impact | Rule | Effective default behavior | Why this matters |
|---|---|---|---|
| High | Stage input | RL always loads from `load_model("sft", ...)`. | RL is explicitly the third stage, not an alternate path from base pretraining. |
| High | Dataset | Train on `GSM8K(main, train)` and evaluate on `GSM8K(main, test)`, both shuffled with `seed=42`. | The RL stage is narrowly specialized around GSM8K reward improvement. |
| High | Step count | `num_steps = (len(train_task) // examples_per_step) * num_epochs`. | RL horizon is derived from dataset size and per-step question count. |
| High | Reward function | `reward = float(task.evaluate(...))`, where `evaluate` extracts the number after `####` and checks exact match. | The entire RL signal is sparse exact-answer correctness, not heuristic chain-of-thought scoring. |
| High | Advantage rule | `advantages = rewards - rewards.mean()` within the sampled group. | This is the script's simplified GRPO/REINFORCE core: centered rewards without z-score normalization. |
| High | PPO/KL omissions | No trust-region KL, no PPO ratio, no clipping. | The repo intentionally strips RL down to a simpler on-policy objective. |
| High | Loss masking | Prompt tokens and tool-forced tokens are masked out of RL loss using the engine's returned masks. | Ensures gradients only hit model-generated answer tokens. |
| High | LR schedule | Linear rampdown to zero over `num_steps`, starting from the already-shrunk `init-lr-frac`. | Keeps RL updates increasingly conservative as the run proceeds. |
| High | Eval `k` range | During eval, the script computes `pass@1 ... pass@device_batch_size`, so the default range is `pass@1 ... pass@8`. | This follows the batched sampling limit, not the training rollout count of `16`. |
| Medium | Sampling passes | Because `num_samples=16` and `device_batch_size=8`, each question is sampled in two forward passes by default. | This is how RL keeps rollout count high without exceeding per-pass memory limits. |
| Medium | Optimizer partitioning | RL uses `model.setup_optimizer()` with default hidden `scalar_lr=0.5` from `nanochat.gpt`. | Even though the CLI does not expose scalar LR for RL, those parameters still follow the model's internal optimizer defaults. |
| Medium | Checkpoint contents | RL saves model weights and model config metadata, but deliberately does not save optimizer state. | Keeps RL checkpoints lighter and simpler, at the cost of not resuming with optimizer buffers. |

## Cross-stage notes

These defaults are easy to miss, but they explain much of the project's behavior.

| Topic | Current default behavior | Why it matters |
|---|---|---|
| Main optimization philosophy | Pretraining derives most important knobs from model depth and scaling-law rules. | The project is opinionated: defaults are not arbitrary constants, but a compact policy for training a depth-indexed model family. |
| Stage coupling | SFT inherits core batch/LR/sequence settings from the base checkpoint by default, and RL loads from SFT by default. | The three stages are meant to compose into one pipeline rather than be tuned independently from scratch. |
| Parameter-specific optimizers | All stages rely on `setup_optimizer()` to separate embeddings, head, scalars, and matrices into different groups. | Many visible defaults only make sense together with this hidden optimizer partitioning. |
| Hardware adaptation | Device type and compute dtype are auto-detected unless overridden. | Keeps the default path simple while still using modern GPU capabilities when present. |
| Logging safety | All three scripts default to `run=dummy`, which disables real W&B logging. | New users can run the code locally without setting up experiment tracking first. |

## Source map

This document is derived from the following code paths:

- `scripts/base_train.py`
- `scripts/chat_sft.py`
- `scripts/chat_rl.py`
- `nanochat/gpt.py`
- `nanochat/checkpoint_manager.py`
- `tasks/common.py`
- `tasks/gsm8k.py`
