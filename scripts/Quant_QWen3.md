
## 0. Prerequisites

- Driver: 1.20.1 (how to update Gaudi driver: https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html)
- Firmware: 1.20.1 (how to update Gaudi firmware: https://docs.habana.ai/en/latest/Installation_Guide/Firmware_Upgrade.html#system-unboxing-main)
- Docker: vault.habana.ai/gaudi-docker/1.20.1/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest

## 1. Installation

- VLLM
```bash
git clone https://github.com/yiliu30/vllm-fork.git
git checkout qwen-fp8
cd vllm-fork
pip install -r requirements-hpu.txt
VLLM_TARGET_DEVICE=hpu pip install -e .  --no-build-isolation
```

- VLLM-HPU-EXT
```bash
git clone https://github.com/yiliu30/vllm-hpu-extension-fork.git vllm-hpu-extension
cd vllm-hpu-extension
git checkout qwen-fp8
pip install -e . -vvv
```

- INC
```bash
pip install git+https://github.com/intel/neural-compressor.git@qwen-fp8
```

### Calibration 

```bash
cd vllm-fork
export OFFICIAL_MODEL=/path/to/qwen/model
bash ./scripts/run_qwen.sh calib ${OFFICIAL_MODEL}
```

> [!TIP] 
> It will take a while to calibrate. You can download the calibration files directly from Hugging Face.

```bash
huggingface-cli download Yi30/q3-g2-pile512 --local-dir  ./scripts/nc_workspace_measure_kvache_v2
```

### Quantization 
```bash
cd vllm-fork
export OFFICIAL_MODEL=/path/to/qwen/model
bash ./scripts/run_qwen.sh quant ${OFFICIAL_MODEL} 
```

### Evalution 
```bash
cd vllm-fork
export OFFICIAL_MODEL=/path/to/qwen/model
bash ./scripts/run_qwen.sh eval ${OFFICIAL_MODEL} 
```

