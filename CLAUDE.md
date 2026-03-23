# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Mandatory Contribution Policy

See [AGENTS.md](AGENTS.md) for contribution rules that **must** be followed, including duplicate-work checks, accountability requirements, and the ban on low-value busywork PRs. All AI-assisted PRs require human review and explicit disclosure.

## Build & Development Commands

```bash
# Environment setup (always use uv, not pip)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements/lint.txt && pre-commit install

# Install for Python-only changes (uses precompiled C extensions)
VLLM_USE_PRECOMPILED=1 uv pip install -e .

# Install for C/C++/CUDA changes (compiles from source)
uv pip install -e .

# Run a single test
pytest tests/path/to/test.py -v -s -k test_name

# Run tests in a directory
pytest tests/path/to/dir -v -s

# Install test dependencies (versions from requirements/test.txt)
uv pip install pytest pytest-asyncio tblib   # minimal
uv pip install -r requirements/test.txt      # full

# Linting (pre-commit runs ruff, clang-format, typos, markdownlint, mypy)
pre-commit run                               # staged files
pre-commit run --all-files                   # all files
pre-commit run ruff-check --all-files        # ruff only
pre-commit run mypy-3.10 --all-files --hook-stage manual  # mypy as CI runs it
```

## Architecture Overview

vLLM is a high-throughput LLM inference and serving engine. The codebase has two engine architectures: the legacy **V0** engine (`vllm/engine/`) and the current default **V1** engine (`vllm/v1/`). New development targets V1.

### Request Lifecycle (V1)

```
User Request
  → Entrypoints (vllm/entrypoints/)      # HTTP API or Python LLM class
    → AsyncLLM (vllm/v1/engine/async_llm.py)  # Async frontend
      → EngineCore (vllm/v1/engine/core.py)    # Runs in separate process, communicates via ZMQ+msgspec
        → Scheduler (vllm/v1/core/sched/)       # Decides which requests to run each step
        → KVCacheManager (vllm/v1/core/)         # PagedAttention block management
        → Executor (vllm/v1/executor/)           # Manages worker processes
          → Worker (vllm/v1/worker/)             # Per-GPU process
            → ModelRunner (vllm/v1/worker/gpu_model_runner.py)  # Builds tensors, runs model forward
              → Model (vllm/model_executor/models/)  # Actual HuggingFace model implementation
```

### Key Subsystems

**Entrypoints** (`vllm/entrypoints/`): Two main entry paths:
- `llm.py` — offline batch inference via `LLM` class
- `openai/` — OpenAI-compatible HTTP API server (the production server)
- `cli/` — `vllm serve`, `vllm bench`, etc. (installed as `vllm` command)

**Configuration** (`vllm/config/`): Dataclass-based config split across many files (model, cache, parallel, scheduler, etc.). `VllmConfig` is the top-level aggregate. `EngineArgs` in `vllm/engine/arg_utils.py` parses CLI/constructor args into config objects.

**V1 Engine Core** (`vllm/v1/engine/core.py`): Runs in a dedicated process. Owns the scheduler and communicates with the async frontend (`async_llm.py`) via ZMQ. Uses `msgspec` for fast serialization.

**Scheduler** (`vllm/v1/core/sched/`): Continuous batching scheduler that decides which requests get prefill/decode time each iteration. Manages KV cache allocation via `KVCacheManager`.

**Executor/Worker** (`vllm/v1/executor/`, `vllm/v1/worker/`): Executors manage worker processes. `UniProcExecutor` for single-GPU, `MultiprocExecutor` for tensor/pipeline parallelism, `RayExecutor` for multi-node. Workers own GPU resources; `GPUModelRunner` handles tensor preparation, CUDA graphs, and model forward passes.

**Model Implementations** (`vllm/model_executor/models/`): ~260 model files. Each implements a HuggingFace model architecture for vLLM's execution framework. Models use layers from `vllm/model_executor/layers/` which provide quantization, attention, linear, and MoE abstractions.

**Attention** (`vllm/v1/attention/`): Pluggable attention backends (FlashAttention, FlashInfer, etc.) selected per-platform.

**Distributed** (`vllm/distributed/`): Tensor parallelism, pipeline parallelism, expert parallelism, and KV cache transfer for disaggregated serving.

**Multimodal** (`vllm/multimodal/`): Processing pipeline for images, audio, and video inputs, with per-model processor registration via `MULTIMODAL_REGISTRY`.

**Compilation** (`vllm/compilation/`): `torch.compile` integration with custom passes, CUDA graph capture, and piecewise compilation.

**LoRA** (`vllm/lora/`): Multi-LoRA serving with dynamic adapter loading.

**Speculative Decoding** (`vllm/v1/spec_decode/`): Draft model, EAGLE, Medusa, and n-gram based speculative decoding.

**Platforms** (`vllm/platforms/`): Hardware abstraction for CUDA, ROCm, TPU, XPU, and CPU backends.

**C++/CUDA Kernels** (`csrc/`): Custom CUDA kernels for attention, cache management, quantization, MoE, layernorm, positional encoding, and sampling. Python bindings via `torch_bindings.cpp`.

### Environment Variables

`vllm/envs.py` defines all `VLLM_*` environment variables with defaults and documentation. Key ones: `VLLM_USE_PRECOMPILED`, `VLLM_ENGINE_ITERATION_TIMEOUT_S`, `CUDA_VISIBLE_DEVICES`.

## Code Style Notes

- Python 3.10+ (uses `X | Y` union syntax, not `Optional`/`Union`)
- Linting: ruff (check + format), mypy with pydantic plugin, typos, clang-format for C++/CUDA
- Config classes use `dataclasses` with a custom `@config` decorator (`vllm/config/utils.py`)
- Serialization between engine processes uses `msgspec`, not pickle
- Logging via `vllm.logger.init_logger(__name__)`

# Develop Envs
python: /home/yiliu7/workspace/vllm/.venv/bin/python