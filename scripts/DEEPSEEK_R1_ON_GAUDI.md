# install

```
docker run -d -it --runtime=habana --name deepseek-vllm-1.20  -v `pwd`:/workspace/vllm/  -v /data:/data -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host --net=host -e HF_HOME=/data/huggingface artifactory-kfs.habana-labs.com/docker-local/1.20.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:1.20.0-521 /bin/bash
```

```
git clone https://github.com/HabanaAI/vllm-fork.git; git checkout deepseek_r1
cd vllm;  pip install -r requirements-hpu.txt; VLLM_TARGET_DEVICE=hpu pip install -e .  --no-build-isolation;
```

# prepare model

```
huggingface-cli download --local-dir ${YOUR_PATH}/DeepSeek-R1 deepseek-ai/DeepSeek-R1
```

# Option 1. run with runtime dequantize with block-based scale
> expect new DynamicMOE kernel ready in few weeks.
> Current Performance is worse than static quantization due to lack of dynamic MOE support.
## step 1. run example
```
VLLM_ENABLE_RUNTIME_DEQUANT=1 python scripts/run_example_tp.py --model ${YOUR_PATH}/DeepSeek-R1
```
## step 2. run lm_eval
```
VLLM_ENABLE_RUNTIME_DEQUANT=1 python scripts/run_lm_eval.py -l 64 --batch_size 1 --ep_size 1
{"gsm8k": {"alias": "gsm8k", "exact_match,strict-match": 0.96875, "exact_match_stderr,strict-match": 0.021921011700381302, "exact_match,flexible-extract": 0.96875, "exact_match_stderr,flexible-extract": 0.021921011700381302}}{"e2e time(secs)": 938.2986768169999}
```

# Option 2. run with dynamic quantization
> expect new DynamicMOE kernel ready in few weeks.
> Current Performance is worse than static quantization due to lack of dynamic MOE support.
## step 1. run example
```
# if you're testing with patched kernel
# use VLLM_DMOE_DYNAMIC_SCALE=1 to enable dynamic scaling supported DynamicMOE
VLLM_DMOE_DYNAMIC_SCALE=1 python scripts/run_example_tp.py --model ${YOUR_PATH}/DeepSeek-R1
```
## step 2. run lm_eval
```
VLLM_DMOE_DYNAMIC_SCALE=1 python scripts/run_lm_eval.py -l 64 --batch_size 1
{"gsm8k": {"alias": "gsm8k", "exact_match,strict-match": 0.96875, "exact_match_stderr,strict-match": 0.021921011700381302, "exact_match,flexible-extract": 0.96875, "exact_match_stderr,flexible-extract": 0.021921011700381302}}{"e2e time(secs)": 938.2986768169999}
```
## step 3. run benchmark
```
VLLM_DMOE_DYNAMIC_SCALE=1 bash scripts/benchmark-dynamicfp8-i1k-o1k-ep8-bestperf.sh
```

# Option 3. run with static quantization
> current best performance
## step 1. Prepare static quantization model
```
python scripts/convert_block_fp8_to_channel_fp8.py --model_path ${YOUR_PATH}/DeepSeek-R1 --qmodel_path ${YOUR_PATH}/DeepSeek-R1-static --input_scales_path scripts/DeepSeek-R1-BF16-w8afp8-static-no-ste_input_scale_inv.pkl.gz
```
## step 2. run example
```
python scripts/run_example_tp.py --model ${YOUR_PATH}/DeepSeek-R1-static
```
## step 3. run benchmark
```
bash scripts/benchmark-staticfp8-i1k-o1k-ep8-bestperf.sh
```

# Others. run with multi nodes
```
# head node
HABANA_VISIBLE_MODULES='0,1,2,3,4,5,6,7'  \
PT_HPU_WEIGHT_SHARING=0 \
PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1 \
PT_HPU_ENABLE_LAZY_COLLECTIVES="true" \
VLLM_RAY_DISABLE_LOG_TO_DRIVER="1" \
RAY_IGNORE_UNHANDLED_ERRORS="1" \
ray start --head --resources='{"HPU": 8, "TPU": 0}'
```

```
# worker node
HABANA_VISIBLE_MODULES='0,1,2,3,4,5,6,7'  \
PT_HPU_WEIGHT_SHARING=0 \
PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1 \
PT_HPU_ENABLE_LAZY_COLLECTIVES="true" \
VLLM_RAY_DISABLE_LOG_TO_DRIVER="1" \
RAY_IGNORE_UNHANDLED_ERRORS="1" \
ray start --address='${head_ip}:6379' --resources='{"HPU": 8, "TPU": 0}'
```

```
python scripts/run_example_tp_2nodes.py --model ${YOUR_PATH}/DeepSeek-R1-static
```

# Requantize the official FP8 Model Using INC
- INC: https://github.com/yiliu30/vllm-fork/tree/r1-woq

- Calibration
```bash
export OFFICIAL_FP8_MODEL=deepseek-ai/DeepSeek-R1
# For quick test
VLLM_FORCE_INC=1 QUANT_CONFIG=inc_measure_with_fp8kv_config.json VLLM_ENABLE_RUNTIME_DEQUANT=1 python run_example_tp.py --model ${OFFICIAL_FP8_MODEL} --tokenizer ${OFFICIAL_FP8_MODEL}
# For calibration with pile dataset
VLLM_FORCE_INC=1 QUANT_CONFIG=inc_measure_with_fp8kv_config.json VLLM_ENABLE_RUNTIME_DEQUANT=1 python run_example_tp.py --prepare --model ${OFFICIAL_FP8_MODEL} --tokenizer ${OFFICIAL_FP8_MODEL} --nprompts 512 --dataset pile --osl 32
```
- Quantizatiion
```bash
VLLM_FORCE_INC=1 QUANT_CONFIG=inc_quant_with_fp8kv_config.json VLLM_ENABLE_RUNTIME_DEQUANT=1 python run_example_tp.py --model ${OFFICIAL_FP8_MODEL} --tokenizer ${OFFICIAL_FP8_MODEL} --fp8_kv_cache
```

