# Data Pipeline and Parquet Analysis

This document summarizes how training data flows through nanochat, why parquet is used in part of the stack, where it is not used, and how this maps to common industry practice.

The short answer is simple:

- Data does not have to be parquet at every stage.
- In production training systems, different stages usually use different representations.
- Parquet is a strong choice for a canonical document layer, but it is not the only correct format and it is often not the final training-ready format.

---

## 1. Executive Summary

From an engineering perspective, the right question is not "should all data be converted to parquet?" but rather "which layer of the pipeline benefits from parquet?"

In nanochat, the answer is:

- Base pretraining data is distributed and streamed as parquet shards.
- Tokenizer training also reads from those parquet shards.
- Actual model training does not consume parquet directly as final training samples. It consumes token tensors created after runtime tokenization, BOS alignment, packing, cropping, and host-to-device staging.
- SFT and RL data paths do not require parquet at all. They use task objects, HuggingFace datasets, and JSON-based conversation sources.

This mirrors a common industrial pattern:

- Raw ingestion layer: source-specific formats
- Canonical cleaned corpus layer: often parquet or arrow
- Tokenized training layer: often binary token shards, mmap arrays, record shards, or another training-optimized format
- Runtime batch layer: packed tensors on CPU/GPU

The most important insight is that file format choice matters less than:

- data quality
- deduplication and filtering
- sampling and mixture policy
- tokenizer choice
- packing strategy
- caching and prefetch behavior
- reproducibility and resumability

---

## 2. Core Question: Must Data Be Parquet?

No.

Parquet is useful, but not mandatory.

Parquet is most valuable when you need:

- schema-driven storage
- large-scale offline processing
- compression
- row-group based streaming
- interoperability across ETL tools
- cloud/object-store friendly sharding

Parquet is less obviously ideal when you need:

- the absolute fastest final training read path
- direct access to pre-tokenized fixed-width binary data
- ultra-simple append-only chat fine-tuning data
- online RL rollouts or transient preference logs

For those later-stage uses, many teams switch to formats that are more directly aligned with training throughput, such as:

- binary token shards
- mmap arrays
- indexed datasets
- tar/record shards
- Arrow memory datasets
- JSONL for smaller SFT or preference datasets

So the industrial answer is not "always parquet". It is "use parquet where parquet solves the right problem".

---

## 3. Nanochat End-to-End Data Flow

The nanochat repository uses different data flows for different stages.

### 3.1 Pretraining Flow

The pretraining path is the place where parquet matters most.

High-level flow:

1. Source dataset is loaded from HuggingFace.
2. Source rows are normalized and repackaged.
3. Repackaged corpus is stored as parquet shards.
4. Shards are uploaded and later downloaded on demand.
5. Training reads parquet row groups.
6. Text is batch-tokenized at runtime.
7. Tokenized documents are BOS-aligned and packed into fixed-length rows.
8. Packed rows are copied through pinned CPU memory into persistent GPU buffers.
9. Model trains on final `inputs` and `targets` tensors.

This already shows that parquet is only one stage in the path, not the final training representation.

### 3.2 SFT Flow

SFT is structurally different.

- Data comes from a mixture of task objects.
- Some tasks come from HuggingFace datasets.
- Some tasks come from local JSON files.
- Conversations are rendered into tokens with assistant/user/tool markers.
- The SFT dataloader performs BOS-aligned packing with masking.

There is no requirement that SFT data first become parquet.

### 3.3 RL Flow

RL is even further from a parquet-centric workflow.

- Training data is loaded as a task object (`GSM8K`).
- Prompts are rendered for completion.
- The model samples rollouts online.
- Rewards are computed online.
- Policy-gradient style updates are computed from generated sequences.

This is an online training loop. Parquet would only matter if one wanted to store rollout logs or offline replay artifacts, not for the core loop itself.

---

## 4. What `dev/repackage_data_reference.py` Actually Does

The file `dev/repackage_data_reference.py` documents the offline preparation of the pretraining corpus.

Its responsibilities are:

1. Load the source dataset.
2. Shuffle the dataset.
3. Convert source rows into plain text.
4. Group many documents into moderately sized shards.
5. Write those shards as zstd-compressed parquet files.
6. Upload the resulting shard set to HuggingFace.

Several important engineering choices are visible here.

### 4.1 The source corpus is not treated as sacred

For ClimbMix, the source data is already stored as GPT-2 token IDs. Instead of keeping that tokenization, the script decodes it back to text before writing the project shards.

That is a strong design choice with an important implication:

- the project wants a tokenizer-independent canonical text layer
- it does not want to lock the whole stack to the source tokenizer

This is one of the most important engineering insights in the repository.

If the project had kept only the upstream token IDs, it would be much harder to:

- retrain a tokenizer on the actual corpus
- change tokenization strategy later
- compare tokenizer variants fairly
- keep the corpus useful for future experiments

### 4.2 The parquet layer is intentionally simple

The script writes:

- one `text` column
- zstd compression
- no dictionary encoding
- no statistics
- row group size of 1024
- shard size around 100MB compressed

This is not trying to use every parquet feature. It is using parquet as a practical, streamable, compressed document container.

That simplicity is part of the design.

### 4.3 Row group size is chosen for downstream streaming

The row group size is set to 1024, not just a default round number. That makes the parquet files easier to consume efficiently in the custom distributed loader.

This is another useful insight:

- storage layout should be chosen with downstream read patterns in mind
- the best format choice is not abstract, it depends on how training will actually consume the data

---

## 5. Nanochat Pretraining Data Pipeline in Detail

This section traces the path from raw data to tensors.

### 5.1 Source Ingestion

The source pretraining corpus comes from NVIDIA's Nemotron-ClimbMix dataset.

At this stage, the concerns are:

- source quality
- source schema
- accessibility
- provenance

In industrial systems, this raw stage may include:

- HTML dumps
- JSONL exports
- WARC files
- code repositories
- document stores
- mixed multimodal records

The key point is that raw ingestion formats are often heterogeneous.

### 5.2 Offline Cleaning and Repackaging

In nanochat's documented preparation flow:

- the source data is loaded
- shuffled with a fixed seed
- normalized to plain text
- grouped into shards
- written to parquet

This creates a clean, portable, project-specific corpus distribution.

This step is closer to a data engineering or corpus publishing stage than to the training step itself.

### 5.3 Download and Local Cache

At runtime, `nanochat/dataset.py` downloads parquet shards into a local cache directory.

Important engineering details:

- downloads are parallelized
- retries are implemented
- temporary files are used before rename
- existing shards are skipped
- the final shard is reserved for validation

This is a good example of a practical production concern that is independent of model code: robust data acquisition and caching.

### 5.4 Parquet Row Group Streaming

The project does not read the entire corpus into memory.

Instead:

- it opens a parquet file
- reads one row group at a time
- extracts the `text` column as a list of documents
- distributes row groups across DDP ranks

This is where parquet is paying for itself in nanochat.

The relevant benefit is not "parquet is cool". The benefit is:

- large shard files
- compressed storage
- row-group chunking
- predictable iteration boundaries

### 5.5 Tokenizer Training from the Canonical Text Layer

The tokenizer is trained by iterating over text recovered from the parquet shards.

This has two major advantages:

1. The tokenizer is trained on the actual project corpus, not an unrelated proxy.
2. The tokenizer remains decoupled from the source dataset's original tokenization.

This is exactly why a text-level canonical layer is valuable.

### 5.6 Runtime Tokenization

During pretraining, the dataloader tokenizes document batches at runtime.

That means the actual training path is:

- parquet -> text list -> tokenizer.encode(batch) -> token lists

This is simple and flexible, but it also means some CPU work remains in the training loop.

From an industry perspective, this is a reasonable choice when:

- simplicity matters
- experimentation with tokenizer behavior matters
- throughput is good enough
- you want fewer offline preprocessing stages

But if throughput becomes limiting, many teams push this tokenization earlier and store tokenized shards offline.

### 5.7 BOS-Aligned Packing

After tokenization, nanochat does something much more important than file-format conversion: it transforms documents into trainable rows.

The default loader uses BOS-aligned best-fit packing.

The row-building strategy is roughly:

1. Every document gets a BOS token prepended.
2. For each row, fill with the largest buffered document that fits.
3. If nothing fits, crop one document to fill the remaining space.
4. Maintain full utilization of the training batch.

This is where training semantics are shaped.

The most important implications are:

- every row starts at a real document boundary
- the model sees cleaner context structure
- token utilization stays high
- some tokens are deliberately discarded due to cropping

This is a much bigger modeling and systems decision than "parquet vs not parquet".

### 5.8 CPU/GPU Staging

The loader then writes packed rows into:

- a reusable CPU buffer
- pinned host memory
- a reusable GPU buffer

This reduces allocation and transfer overhead.

This is the last transformation before the model receives training inputs.

At this point, the training representation is no longer anything like parquet. It is a pair of tensors:

- `inputs`
- `targets`

This final stage is what actually matters to the GPU.

---

## 6. Why Parquet Was a Good Choice Here

For nanochat's pretraining corpus, parquet is a good fit because it provides a solid middle layer.

### 6.1 It is good for publishing and distribution

The shard set is easy to host and easy to download incrementally.

### 6.2 It is good for a text-level canonical corpus

The corpus remains usable for:

- tokenizer retraining
- text inspection
- re-sharding
- future filtering
- alternate downstream pipelines

### 6.3 It supports row-group streaming naturally

The custom loader can use row groups as a natural unit of consumption and DDP partitioning.

### 6.4 It avoids the small-file problem

Many large corpora become operationally painful when stored as millions of tiny documents. Larger parquet shards avoid that operational trap.

### 6.5 It keeps the runtime code simple

The runtime data path only needs to know how to:

- list shard files
- open parquet files
- iterate row groups
- read the `text` column

That is a very manageable operational surface.

---

## 7. Why Parquet Is Not Sufficient as a Universal Rule

Parquet solves some problems, not all of them.

### 7.1 It is not inherently the fastest final training format

For pure training throughput, a text column inside parquet often still requires:

- decompression
- string materialization
- tokenization
- packing

If training scale grows enough, those steps can become the bottleneck.

That is why many industrial systems eventually introduce a tokenized training layer.

### 7.2 It is not necessary for SFT and RL

SFT and RL datasets are usually smaller, more structured, and more schema-specific.

Common representations there include:

- JSONL conversations
- HuggingFace datasets
- Arrow-backed local caches
- SQL or object-store records
- direct task wrappers around remote datasets

That is exactly what nanochat does.

### 7.3 It does not replace dataset quality work

Choosing parquet does not solve:

- deduplication
- quality scoring
- language filtering
- contamination checks
- license handling
- mixture design

In industrial practice, these almost always dominate the outcome more than the final storage container.

### 7.4 It does not solve training semantics

The final model sees packed token sequences, not raw parquet rows.

So the high-impact choices are often:

- BOS/EOS policy
- masking policy
- packing algorithm
- crop vs pad tradeoffs
- curriculum and sampling weights

---

## 8. Common Industry Pattern: Layered Representations

The most robust production training systems usually keep separate layers.

### Layer 1: Raw Ingestion Layer

Typical formats:

- JSONL
- HTML
- WARC
- code repositories
- PDFs
- database exports

Purpose:

- preserve source fidelity
- support traceability
- allow reprocessing

### Layer 2: Canonical Cleaned Corpus Layer

Typical formats:

- parquet
- arrow
- delta/iceberg tables backed by parquet

Purpose:

- normalized schema
- metadata columns
- filtering and analytics
- cheap cloud storage
- reproducible dataset versions

This is the layer where parquet is most often used.

### Layer 3: Tokenized Training Layer

Typical formats:

- binary token shards
- mmap arrays
- indexed datasets
- record shards

Purpose:

- eliminate runtime tokenization cost
- maximize throughput
- simplify deterministic resumption
- support direct random or sequential token access

This is the layer many high-scale training systems optimize aggressively.

### Layer 4: Runtime Batch Layer

Typical representation:

- packed tensors in host or device memory

Purpose:

- minimize kernel idle time
- enable asynchronous prefetch
- keep GPU fed efficiently

The central engineering lesson is that no single format should be forced to solve all four layers.

---

## 9. How This Maps to Pretraining, SFT, and RL

### 9.1 Pretraining

Pretraining has:

- massive corpus volume
- broad document diversity
- strong need for sharding and streaming
- strong need for reproducibility
- often significant tokenizer and packing cost

Recommended representation strategy in industry:

- use parquet or arrow at the cleaned corpus layer
- consider a tokenized binary layer for peak throughput if needed

This is where parquet is most justified.

### 9.2 SFT

SFT has:

- smaller corpus size
- structured conversations
- per-example masking rules
- frequent prompt-template changes
- high need for flexibility

Recommended representation strategy in industry:

- JSONL, HF datasets, Arrow, or parquet are all fine
- favor the format that makes conversation rendering and curation easiest

The final bottleneck is usually not whether examples were stored in parquet.

### 9.3 RL or Online Post-Training

RL has:

- prompt datasets
- online rollout generation
- online reward computation
- ephemeral trajectories
- frequent logging of sampled generations

Recommended representation strategy in industry:

- lightweight prompt stores or task datasets for inputs
- structured logging for rollouts and rewards if persistence is needed
- no blanket need for parquet in the hot loop

Parquet may be useful for offline analysis of rollout logs, but it is not the core training abstraction.

---

## 10. The Most Important Engineering Insights

### 10.1 Format is downstream of data quality

The biggest win in this repository came from switching to a better corpus, not from changing file format.

This is a general industry truth:

- better filtering and deduplication often beats low-level storage tweaks
- data quality changes frequently dominate architecture tweaks and systems micro-optimizations

### 10.2 Keep a tokenizer-independent canonical layer when possible

Nanochat deliberately decodes upstream token IDs back to text before repackaging.

That preserves freedom to:

- retrain tokenizers
- compare tokenizers
- re-template data
- evolve the stack later

Once you store only downstream token IDs, you lose a lot of flexibility.

### 10.3 Optimize the true bottleneck, not the visible file extension

For many training systems, the bottleneck is not "reading parquet". It is one of:

- tokenization CPU time
- Python overhead in packing
- host-to-device transfer behavior
- small-file or remote-fetch overhead
- poor sharding for distributed workers

The right optimization target depends on profiling.

### 10.4 Packing strategy is part of the training algorithm

Nanochat's BOS-aligned best-fit loader demonstrates that dataloader behavior affects:

- context cleanliness
- token utilization
- distribution skew
- crop waste
- reproducibility

This is not just a systems concern. It is part of the model's effective training distribution.

### 10.5 Different stages deserve different storage formats

Trying to force one format across raw data, canonical corpus, tokenized shards, SFT examples, and RL logs usually makes the system worse, not better.

The more professional pattern is to define clean boundaries between stages.

### 10.6 Resumability is a first-class data concern

Nanochat tracks dataloader state and can resume by parquet index, row-group index, and epoch.

That is a subtle but important engineering detail. In real training systems, reproducible resumption is often more important than small differences in storage format elegance.

---

## 11. Practical Recommendations

### If you are building a pretraining stack

- Use a canonical cleaned corpus layer with schema and metadata.
- Parquet is a strong default for that layer.
- Keep text available if tokenizer evolution is still likely.
- If throughput becomes limiting, add a tokenized training layer rather than forcing parquet to be the final answer.

### If you are building an SFT stack

- Prefer the format that makes conversation rendering, masking, and curation easiest.
- JSONL or HF datasets are often sufficient.
- Use parquet only if you need large-scale analytics, partitioning, or shared table workflows.

### If you are building an RL stack

- Separate prompt storage, rollout logging, and online training mechanics.
- Do not assume the prompt dataset and the rollout log need the same storage format.
- Use parquet only where batch analytics or archival make it helpful.

### If you are optimizing an existing pipeline

- Profile first.
- Determine whether the bottleneck is download, decompression, tokenization, packing, Python overhead, or H2D transfer.
- Only then decide whether to change format.

---

## 12. What This Means for Nanochat Specifically

For the current project, the existing choices are coherent.

### What is already good

- Pretraining has a clean parquet-based corpus distribution layer.
- The tokenizer is trained from the same canonical text source.
- The runtime loader is simple and resumable.
- The project keeps SFT and RL flexible instead of forcing them into the pretraining mold.

### What would justify adding another training format later

If pretraining throughput becomes CPU-bound due to runtime tokenization or Python-side packing, the next industrial-style upgrade would likely be:

- keep parquet as the canonical corpus layer
- add an offline tokenized shard layer for the hot training path

That would preserve flexibility while improving throughput.

### What should not be done blindly

It would not be wise to convert every SFT and RL dataset into parquet just for consistency. That would add ceremony without necessarily improving quality, simplicity, or speed.

---

## 13. Final Answer

From a professional industrial training perspective, data does not need to be universally converted to parquet.

The better principle is:

- use source-native formats at ingestion
- use parquet or arrow for a canonical cleaned corpus layer when that helps
- use tokenized binary or other training-optimized formats when throughput demands it
- use task-appropriate formats for SFT and RL

In other words, parquet is often a very good middle layer, but it is rarely the entire story.

For nanochat, this layered view is already visible in the codebase:

- pretraining uses parquet as a document distribution layer
- tokenizer training consumes that text layer
- runtime training transforms documents into packed token tensors
- SFT and RL use different, more task-native representations

That is close to how strong industrial systems are usually built: not around one universal format, but around clear interfaces between data stages.
