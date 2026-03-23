# vLLM + vllm-xpu-kernels Setup on Intel XPU Node

## System
- Intel Arc Pro B60 (3x GPUs, ~16GB each)
- oneAPI 2025.2 at `/opt/intel/oneapi/`
- Python 3.12, uv 0.10.11

## Quick Setup Steps

```bash
# 1. Create venv
cd /home/yiliu7/workspace/vllm
uv venv --python 3.12 .venv
source .venv/bin/activate

# 2. Install PyTorch XPU (triton-xpu comes with it)
uv pip install torch==2.10.0+xpu torchaudio torchvision --extra-index-url https://download.pytorch.org/whl/xpu

# 3. Install build deps (needed for --no-build-isolation)
uv pip install "setuptools>=77.0.3,<80.0.0" --force-reinstall
uv pip install packaging setuptools-scm wheel ninja cmake regex jinja2 build

# 4. Build vllm-xpu-kernels from source
export CMPLR_ROOT=/opt/intel/oneapi/compiler/2025.2
export PATH="/opt/intel/oneapi/compiler/2025.2/bin:$PATH"
export MAX_JOBS=32
export VLLM_XPU_AOT_DEVICES="bmg"
export VLLM_XPU_XE2_AOT_DEVICES="bmg"
cd /home/yiliu7/workspace/vllm-xpu-kernels
uv pip install --no-build-isolation -e . -v   # ~1.5 min with BMG-only

# 5. Install vLLM, then re-install local xpu-kernels (vLLM overwrites it with prebuilt wheel)
cd /home/yiliu7/workspace/vllm
uv pip install --no-build-isolation -e . -v
cd /home/yiliu7/workspace/vllm-xpu-kernels
uv pip install --no-build-isolation -e .

# 6. Test
cd /home/yiliu7/workspace/vllm
python3 examples/basic/offline_inference/generate.py \
  --model superjob/Qwen3-4B-Instruct-2507-GPTQ-Int4 \
  --block-size 64 --enforce-eager --max-model-len 4096
```

## Gotchas

| Issue | Solution |
|-------|----------|
| **DO NOT `source setvars.sh`** at runtime | It overrides `LD_LIBRARY_PATH` with oneAPI 2025.2 `libsycl`, conflicting with pip-installed 2025.3 runtime. Only set `CMPLR_ROOT` + `PATH` for building. |
| **OOM during build** with default `-j=256` | Set `MAX_JOBS=32` |
| **30+ min build time** for multi-arch GPU compilation | Set `VLLM_XPU_AOT_DEVICES="bmg"` to build for BMG only (~1.5 min) |
| **vLLM install replaces local xpu-kernels** with prebuilt v0.1.4 wheel from `requirements/xpu.txt` | Re-run `uv pip install --no-build-isolation -e .` in xpu-kernels dir after vLLM install |
| **KV cache OOM** on Arc Pro B60 (16GB) | Add `--max-model-len 4096` (Qwen3's default 262K needs 36GB) |
| **setuptools version** must be ≥77 for PEP 639 license format | `uv pip install "setuptools>=77.0.3,<80.0.0" --force-reinstall` |
| **Proxy** | Already configured: `http_proxy=http://proxy.ims.intel.com:911` |
| **Triton warning** ("backends could not be imported") | Cosmetic only, doesn't affect functionality |
