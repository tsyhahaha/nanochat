# AGENTS.md - nanochat Development Guide

This file provides guidance for AI agents working in the nanochat repository.

## Project Overview

nanochat is a minimal full-stack ChatGPT clone for training LLMs on a single GPU node. It covers tokenization, pretraining, finetuning, evaluation, inference, and a chat UI.

## Build/Lint/Test Commands

### Testing
```bash
# Run all tests
python -m pytest

# Run a specific test file
python -m pytest tests/test_engine.py -v

# Run a specific test function
python -m pytest tests/test_engine.py::test_kv_cache_basic -v

# Run tests with output capture disabled (for debugging)
python -m pytest tests/test_engine.py -v -s

# Run tests excluding slow markers
python -m pytest -m "not slow"

# Run with coverage (if installed)
python -m pytest --cov=nanochat --cov-report=term-missing
```

### Development Environment
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv sync

# Install dev dependencies (includes pytest)
uv sync --group dev
```

### Running Scripts
```bash
# Pretraining
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=12 --run="d12"

# Evaluation
python -m scripts.base_eval

# Chat UI
python -m scripts.chat_web

# Chat CLI
python -m scripts.chat_cli -p "hello"
```

## Code Style Guidelines

### General Philosophy
- Code should be **minimal, hackable, and readable** - avoid framework-like complexity
- No giant configuration objects or model factories
- Single cohesive codebase, not an exhaustively configurable "framework"

### Python Style
- Follow **PEP 8** with 4-space indentation
- Maximum line length: 120 characters
- Use type hints where beneficial (see Types section)
- No unnecessary comments - let code be self-documenting

### Imports
```python
# Standard library imports first
import os
import re
import logging
import urllib.request

# Third-party imports
import torch
import torch.nn.functional as F

# Local imports
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
```

### Types
- Use Python type hints for function parameters and return values
- Use `torch.dtype` for dtype annotations
- Example:
```python
def get_peak_flops(device_name: str) -> float:
    ...
```

### Naming Conventions
- **Classes**: `CamelCase` (e.g., `KVCache`, `Engine`)
- **Functions/methods**: `snake_case` (e.g., `compute_init`, `get_peak_flops`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `COMPUTE_DTYPE`, `_PEAK_FLOPS_TABLE`)
- **Private members**: Leading underscore (e.g., `_detect_compute_dtype`)
- **Dataclasses**: Use `@dataclass` decorator with field names in `snake_case`

### Error Handling
- Use assertions for developer errors/contract violations:
```python
assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"
assert torch.cuda.is_available(), "Your PyTorch installation is not configured for CUDA"
```
- Return `None` gracefully for recoverable errors (e.g., failed eval with timeout)
- Log warnings for unexpected but non-fatal conditions

### Logging
- Use the module-level logger:
```python
logger = logging.getLogger(__name__)
```
- Log important events (training milestones, errors, warnings)
- Use `print0()` for rank-0 only printing in distributed training

### Torch Best Practices
- Avoid `torch.amp.autocast` - precision is managed via `COMPUTE_DTYPE` in `common.py`
- Use explicit device placement (`device="cuda"` parameters)
- For distributed training, use `torchrun` with proper env vars (RANK, LOCAL_RANK, WORLD_SIZE)
- Set seeds explicitly: `torch.manual_seed(42)` (not global rng for model weight init)

### Documentation Guidelines
- **Reference `dev/`:** When writing documentation (e.g., `docs/xxx.md`) or developing project `.py` files, you **must** reference the files in the `dev/` directory, with special attention to `dev/LOG.md`.
- **Decouple Insights:** The primary goal is to gradually decouple and extract the insights, context, and experimental findings currently recorded in `dev/LOG.md` into the relevant code files and formal structured documentation in `docs/`.

### File Structure
```
nanochat/           # Core library
├── __init__.py
├── checkpoint_manager.py
├── common.py       # Utilities, COMPUTE_DTYPE, logging setup
├── dataloader.py
├── dataset.py
├── engine.py       # Inference engine with KVCache
├── execution.py    # Python code execution tool
├── flash_attention.py
├── fp8.py
├── gpt.py          # Transformer model
├── loss_eval.py
├── optim.py        # Optimizers (AdamW, Muon)
├── report.py
└── tokenizer.py

scripts/            # Entry points
├── base_train.py
├── base_eval.py
├── chat_sft.py
├── chat_rl.py
├── chat_web.py
└── ...

tests/              # Test files
├── test_engine.py
└── test_attention_fallback.py

dev/                # Research, experiments, and logging
├── LEADERBOARD.md              # Project leaderboard or benchmark tracking
├── LOG.md                      # Central insight log - MUST be referenced and decoupled into code/docs
├── estimate_gpt3_core.ipynb    # Modeling estimates for GPT-3 core concepts
├── gen_synthetic_data.py       # Script for generating synthetic training data
├── generate_logo.html          # Logo generation utility
├── nanochat.png                # Project asset
├── repackage_data_reference.py # Data repackaging utilities
├── scaling_analysis.ipynb      # Scaling laws analysis notebook
└── scaling_laws_jan26.png      # Scaling laws plot and data
```

### Running Single Tests
```bash
# Single test file
python -m pytest tests/test_engine.py -v

# Single test function
python -m pytest tests/test_engine.py::test_kv_cache_basic -v

# Single test with specific markers
python -m pytest tests/ -k "test_kv_cache" -v

# Verbose output with print statements
python -m pytest tests/test_engine.py::test_kv_cache_basic -v -s
```

### Contributing Notes
- Disclosure policy: PRs must declare any substantial LLM contributions
- Changes should work for all `--depth` settings (compute-optimal scaling)
- Focus on improving "time to GPT-2" metric

### Git Remote Policy
- Push repository changes only to the `tsy` remote (`git@github.com:tsyhahaha/nanochat.git`)
- Do not push this repository's local commits or branches to `origin`
