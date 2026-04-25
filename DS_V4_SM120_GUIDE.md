# DeepSeek V4 Flash on SM120 (RTX 6000D) — Setup & Run Guide

Based on PR [#40852](https://github.com/vllm-project/vllm/pull/40852) (`jasl/vllm:ds4-sm120-prototype`).

## Prerequisites

| Item | Value |
|------|-------|
| GPUs | 8x NVIDIA RTX 6000D (SM 12.0, ~86 GB each) |
| CUDA | 13.1 (`/usr/local/cuda`) |
| Python | 3.12 |
| Model | `/media/yiliu7/deepseek-ai/DeepSeek-V4-Flash` (46 safetensor shards, ~149 GB) |

## Key Points

- PR pins a **forked DeepGEMM** (`jasl/DeepGEMM@959f1df`) with SM120 reference fallbacks. It is fetched automatically during the CMake build.
- This is a **correctness-first** prototype — reference paths are intentionally slow (~2.5 tok/s output).
- The build must target `CUDA_ARCH_LIST=120a` so DeepGEMM compiles the `12.0f` architecture.
- `--enforce-eager` is required (CUDAGraphs not yet supported on this path).
- `--kv-cache-dtype fp8` uses DeepSeek's native `fp8_ds_mla` KV cache format.

## Patch Summary (11 commits on top of `3602f14f0e`)

| # | Commit | Summary | Key Files |
|---|--------|---------|-----------|
| 1 | `c424ebd57` | **Prototype SM120 DeepSeek V4 reference attention** — Adds the core SM120 reference path for sparse MLA attention, including cache utils and a fix for functionalization. | `deepseek_v4_attention.py`, `sparse_swa.py`, `cache_utils.py`, `fix_functionalization.py`, `cutlass.py` |
| 2 | `23b512fc7` | **Allow DeepGEMM to build for SM120 with CUDA 13** — Adds `12.0f` to the supported arch list when CUDA ≥ 13.0. | `deepgemm.cmake` |
| 3 | `118562ac6` | **Split SM120 sparse attention reference into LSE merge stages** — Refactors the reference attention to do log-sum-exp merging in separate stages for correctness. | `deepseek_v4_attention.py` |
| 4 | `e96726b3f` | **Add SM120 FP8 indexer logits fallback** — Reference fallback for the non-paged MQA indexer logits path on SM120. | `deep_gemm.py`, `test_deepgemm_attention.py` |
| 5 | `bfd1c4a4b` | **Register SM120 reference attention env vars** — Adds `VLLM_SM120_REFERENCE_DEEPSEEK_V4_ATTENTION`, `_TOPK_CHUNK_SIZE`, `_QUERY_CHUNK_SIZE` to the env registry to avoid startup warnings. | `envs.py` |
| 6 | `36b473f10` | **Pin DeepGEMM SM120 prototype dependency** — Points `FetchContent` to `jasl/DeepGEMM` fork with SM120 reference kernels. | `deepgemm.cmake` |
| 7 | `a590474f2` | **Prototype DeepSeek V4 pipeline parallelism** — Adds PP support for DeepSeek V4 model. | `deepseek_v4.py`, `test_deepseek_v4_pp.py` |
| 8 | `27d3fc4ce` | **Generalize sparse MLA reference fallback controls** — Makes the reference fallback controllable via env vars rather than hardcoded checks. | `envs.py`, `deepseek_v4_attention.py`, `sparse_swa.py` |
| 9 | `9a276fbfa` | **Let sparse MLA dump control override legacy alias** — Allows the dump/debug control to override legacy env var aliases. | `deepseek_v4_attention.py`, `sparse_swa.py` |
| 10 | `40186eb7f` | **Avoid pinning DeepGEMM SM120 fork** — Temporarily unpins the fork (reverted by next commit). | `deepgemm.cmake` |
| 11 | `1523228e6` | **Keep DeepGEMM SM120 prototype pin** — Re-pins to `jasl/DeepGEMM@959f1df` for the SM120 prototype. | `deepgemm.cmake` |

### Core changes by area

- **DeepGEMM build** (commits 2, 6, 10, 11): Enable SM120 arch (`12.0f`) in CMake, pin forked DeepGEMM with SM120 reference kernels.
- **Sparse MLA attention** (commits 1, 3, 8, 9): Reference attention path for SM120 that matches DeepSeek V4 sparse attention semantics (sparse indices, sink-aware denominator, LSE merge stages).
- **FP8 indexer logits** (commit 4): Fallback for the non-paged MQA logits path that fails with "Unsupported architecture" on SM120.
- **Environment variables** (commits 5, 8): Register 3 new env vars to control the SM120 reference path.
- **Pipeline parallelism** (commit 7): Adds PP support (not SM120-specific but included in the PR).

### DeepGEMM fork changes ([deepseek-ai/DeepGEMM#318](https://github.com/deepseek-ai/DeepGEMM/pull/318))

The forked DeepGEMM (`jasl/DeepGEMM@959f1df`) **reuses all existing SM90/SM100 implementations** and only relaxes architecture checks to include `arch_major == 12`. No new optimized SM120 kernels are written — instead, it adds **reference (scalar) fallback kernels** for the few paths where SM90/SM100 JIT kernels cannot compile on SM120.

**Changes by file:**

| File | What changed |
|------|-------------|
| `csrc/jit/device_runtime.hpp` | Adds SM120 arch string mapping (`major==12` → `"120"` / `"120f"`), so JIT compiler targets the correct arch. |
| `csrc/apis/attention.hpp` | Relaxes `arch_major` guards: adds `or arch_major == 12` alongside existing SM90/SM100 checks for paged MQA metadata and varlen paths. Routes SM120 FP8 (non-FP4) paged MQA logits to a new **reference kernel** instead of the SM90/SM100 TMA-based path. |
| `csrc/apis/einsum.hpp` | For `"bhr,hdr->bhd"` FP8 einsum: on SM120, calls a new **reference kernel** (`sm120_fp8_bhr_hdr_bhd_reference`) instead of the SM90/SM100 WGMMA-based path. |
| `csrc/apis/hyperconnection.hpp` | Adds `sm12x_tf32_hc_prenorm_gemm_reference()` — a **pure PyTorch fallback** (`at::matmul` + sum) for HyperConnection prenorm GEMM. No CUDA kernel at all; runs entirely on ATen ops. |

**New reference CUDA kernels (JIT-compiled at runtime):**

| Kernel | File | Description |
|--------|------|-------------|
| `sm120_fp8_bhr_hdr_bhd_reference` | `deep_gemm/include/.../sm120_fp8_einsum.cuh` | Scalar FP8→float dot product. One thread per output element. Dequantizes FP8 inputs with block-wise scale factors (128-element granularity), accumulates in float, casts output to bf16/float. |
| `sm120_fp8_paged_mqa_logits_reference` | `deep_gemm/include/.../sm120_fp8_paged_mqa_logits.cuh` | Scalar paged MQA logits. One thread per KV token. For each token: loops over all heads, computes FP8 dot product with per-token KV scale, applies ReLU + weight, sums across heads. 128 threads/block. |

**Key design principle:** The approach is minimal — it **does not fork or duplicate** the existing SM90/SM100 optimized kernels. It only:
1. Adds `arch_major == 12` to existing guard conditions (6 lines changed in `attention.hpp`)
2. Adds 3 small reference fallbacks (~290 lines total CUDA) for paths that use SM90/SM100-specific instructions (TMA, WGMMA) unavailable on SM120
3. Updates CUTLASS submodule to a version that supports SM120

### FlashMLA bypass: pure-PyTorch reference attention (vLLM side)

FlashMLA (the optimized CUDA sparse MLA kernel) does not support SM120. PR #40852 handles this by **intercepting every FlashMLA call site** in `deepseek_v4_attention.py` and routing SM120 to a pure-PyTorch reference implementation instead.

**Gate:** `_is_sparse_mla_reference_attention_enabled(device)` returns `True` when:
- `VLLM_SM120_REFERENCE_DEEPSEEK_V4_ATTENTION=1` is set, **or**
- The device is SM12x (auto-detected)

When enabled, FlashMLA kernels (`flash_mla_with_kvcache`, `flash_mla_sparse_fwd`) are never called. Three reference paths handle all cases:

#### 1. SWA-only decode (`_forward_sparse_mla_swa_decode_reference`)
For layers that only use sliding window attention (no compressed/global tokens).

```
Input:  q [num_tokens, num_heads, head_dim], swa_k_cache (paged KV cache)
Output: output [num_tokens, num_heads, head_dim]

Logic:
  1. Dequantize & gather KV from paged cache using SWA indices
  2. Run reference attention (see core attention below)
  3. Merge with learned attention sink bias
```

#### 2. Compressed decode (`_forward_sparse_mla_compressed_decode_reference`)
For layers with compress_ratio=4 or 128 (the main sparse attention layers). This is the most complex path.

```
Input:  q, compressed_k_cache, swa_k_cache, topk_indices, topk_lens
Output: output [num_tokens, num_heads, head_dim]

Logic:
  1. COMPRESSED ATTENTION (top-k selected tokens):
     - Process topk_indices in chunks of TOPK_CHUNK_SIZE (default 256)
     - For each chunk: dequantize compressed KV slots → run reference attention
     - Accumulate using online log-sum-exp (numerically stable streaming softmax)
  2. SWA ATTENTION (sliding window tokens):
     - Dequantize & gather SWA KV → run reference attention
  3. MERGE:
     - Combine compressed + SWA outputs using LSE-weighted merge
     - Add learned attention sink bias
```

#### 3. Prefill (`_forward_sparse_mla_prefill_reference`)
For prefill phase, processes queries and KV indices in chunks.

```
Input:  q [num_tokens, num_heads, head_dim], kv (flat cache), combined_indices, combined_lens
Output: output [num_tokens, num_heads, head_dim]

Logic:
  - Double-chunked loop: QUERY_CHUNK_SIZE (default 128) × TOPK_CHUNK_SIZE (default 256)
  - For each query chunk × index chunk: gather KV → accumulate attention
  - Finish with LSE merge + sink bias
```

#### Core reference attention math

The building block used by all three paths (`_accumulate_reference_attention_chunk`):

```python
# Standard dot-product attention with online softmax (for streaming over chunks)
scores = einsum("bhd,btd->bht", q, kv) * scale     # dot product
scores = masked_fill(scores, ~valid_tokens, -inf)    # mask invalid

# Online log-sum-exp update (numerically stable across chunks)
chunk_max = scores.amax(dim=-1)
new_max = max(prev_max, chunk_max)
prev_scale = exp(prev_max - new_max)     # rescale previous accumulator
weights = exp(scores - new_max)           # current chunk weights

acc = acc * prev_scale + einsum("bht,btd->bhd", weights, kv)
denom = denom * prev_scale + weights.sum()
```

After all chunks: `output = acc / denom`, then merged with **attention sink** (a learned per-head bias that acts as a baseline attention score, ensuring DeepSeek V4's sink-aware denominator behavior is preserved).

#### Key files
- `vllm/model_executor/layers/deepseek_v4_attention.py` — all reference attention methods (lines 865–1232)
- `vllm/v1/attention/backends/mla/sparse_swa.py` — metadata builder, also has reference attention gate

## Install Steps

### 1. Switch to the PR branch

```bash
cd /home/yiliu7/workspace/vllm
git fetch https://github.com/jasl/vllm.git ds4-sm120-prototype
git checkout ds4-sm120-prototype
# Verify: commit 1523228e6 "Keep DeepGEMM SM120 prototype pin"
```

### 2. Create virtual environment

```bash
uv venv --python 3.12
source .venv/bin/activate
```

### 3. Install build prerequisites

```bash
uv pip install setuptools setuptools_scm numpy wheel
```

> `torch` will be resolved automatically by vLLM's dependencies during the editable install.

### 4. Set build environment variables

```bash
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME="/usr/local/cuda"
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
export CUDA_ARCH_LIST="120a"
export TORCH_CUDA_ARCH_LIST="12.0a"
```

### 5. Build vLLM (editable install)

```bash
CCACHE_NOHASHDIR=true MAX_JOBS=64 uv pip install --no-build-isolation -e .
```

Build takes ~11 minutes. Verify the log contains:

```
DeepGEMM CUDA architectures: 12.0f
```

Installed version should look like: `vllm==0.19.1rc1.dev261+g1523228e6.cu131`

### 6. (If ABI error) Clean rebuild

If you see `undefined symbol: _ZN3c1013MessageLoggerC1EPKciib`, the C extension was compiled against a different torch version. Fix:

```bash
rm -rf build/
find vllm -name "*.so" -delete
# Re-run step 5
```

## Run Command

```bash
source .venv/bin/activate

export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME="/usr/local/cuda"
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"

# SM120 reference attention controls (required)
export VLLM_SM120_REFERENCE_DEEPSEEK_V4_ATTENTION=1
export VLLM_SM120_REFERENCE_TOPK_CHUNK_SIZE=256
export VLLM_SM120_REFERENCE_QUERY_CHUNK_SIZE=128

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/python \
    examples/basic/offline_inference/generate.py \
    --model /media/yiliu7/deepseek-ai/DeepSeek-V4-Flash \
    -tp 8 --enforce-eager --kv-cache-dtype fp8 \
    --max-model-len 2084 --gpu-memory-utilization 0.8
```

## Expected Output

- Model loads 46 shards in ~16s, ~20.3 GiB memory per GPU
- Attention backend: `DEEPSEEK_SPARSE_SWA` (block size 256)
- MoE backend: `MARLIN` Mxfp4
- KV cache: ~41 GiB available, ~33,824 tokens
- Engine init (profile + KV cache + warmup): ~59s
- 4 default prompts complete in ~26s
- Throughput: ~0.84 tok/s input, ~2.45 tok/s output

## Known Warnings (Non-blocking)

| Warning | Reason |
|---------|--------|
| `SymmMemCommunicator: Device capability 12.0 not supported` | Expected for SM120 |
| `Custom allreduce disabled (PCIe-only GPUs)` | RTX 6000D has no NVLink |
| `TensorFloat32 tensor cores available but not enabled` | Performance hint, optional |

## Fallback: Disable DeepGEMM

If DeepGEMM fails at runtime, disable it and retry:

```bash
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_DEEP_GEMM_E8M0=0
# Then re-run the command above
```
