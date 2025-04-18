## Installation

```bash
# Install vllm
git clone https://github.com/yiliu30/vllm-fork.git
git checkout inc-r1-g2
cd vllm-fork
pip install -r requirements-hpu.txt
VLLM_TARGET_DEVICE=hpu pip install -e .  --no-build-isolation

# Install INC
pip install git+https://github.com/intel/neural-compressor.git@r1-woq
```

## Benchmark

### Benchmark Configurations

#### Optional 1. BF16 KV + Per-Channel Quantization

- Get calibration file

```bash
huggingface-cli download Yi30/inc-woq-default-pile-one-cache-412-g2  --local-dir ./scripts/nc_workspace_measure_kvache
```

- quant config: inc_quant_per_channel_bf16kv.json

#### Optional 2. FP8 KV + Per-Channel Quantization

- Get calibration file

huggingface-cli download Yi30/inc-woq-default-pile-one-cache-412-g2  --local-dir ./scripts/nc_workspace_measure_kvache

```bash
- quant config: inc_quant_with_fp8kv_config.json
```

#### Optional 3. FP8 KV + PER-Tensor + FP8 MLA (Slow warmup, Best Perf, WIP)

- Get calibration file

```bash
huggingface-cli download Yi30/inc-woq-default-pile-one-cache-412-for-fp8-mla-g2 --local-dir ./scripts/nc_workspace_measure_fp8_mla
```

- quant config: inc_quant_fp8kv_pts_scalar_fp8_mla.json

## Running the Benchmark

Edit the following section in  `single_16k_len.sh`

```bash
# 1. Fixed configurations for INC WOQ ReQuant
####### For INC WOQ ReQuant #######
export VLLM_MOE_N_SLICE=1
export VLLM_MLA_DISABLE_REQUANTIZATION=1
export VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0
export VLLM_REQUANT_FP8_INC=1
export VLLM_ENABLE_RUNTIME_DEQUANT=1


# 2. Use the offline conveted model
model_path=/mnt/disk2/hf_models/DeepSeek-R1-G2/

# 3. Select config file
# export QUANT_CONFIG="inc_quant_fp8kv_pts_scalar_fp8_mla.json"
# export QUANT_CONFIG="inc_quant_per_channel_bf16kv.json"
export QUANT_CONFIG="inc_quant_per_channel_with_fp8kv_config.json"

# 4. Update `kv_cache_dtype` to `fp8_inc` for fp8 KV Cache config
python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8688 \
    --block-size 128 \
    --model $model_path \
    ...
    --kv_cache_dtype "fp8_inc"  
```

```bash
cd ./scripts
bash single_16k_len.sh
```
