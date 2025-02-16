# Note for quantize vLLM DeepSeek V3/R1 using INC

## Prequsites

- Hardware: ~~2xG3~~  8XG3
- Docker: 1.20.0-497

```bash
... --os ubuntu22.04 --pt=2.6.0 -v 1.20.0 -b 497 -n dev
```

- INC <https://github.com/intel/neural-compressor/tree/dev/yi/quant_vllm>

```bash
git clone https://github.com/intel/neural-compressor.git inc
cd inc
git checkout dev/yi/quant_vllm
pip install -r requirements.txt
pip install -r requirements_pt.txt
python setup.py pt develop
```

- vLLM <https://github.com/yiliu30/vllm-fork/tree/dev/yi/ds_inc_quant>

```
cd vllm;  pip install -r requirements-hpu.txt; VLLM_TARGET_DEVICE=hpu pip install -e .  --no-build-isolation;
```

- ~~Reduced DeepSeek V3 model (4 layers with random weights)~~
- Reduced DeepSeek V3 model (4 layers with real weights)

```
model_path = "/software/users/yiliu4/HF_HOME/hub/deepseekv3-bf16-4l-real"
```

## Run

```bash
# vllm root
cd vllm

# !! Replace the model_path in the run_lm_eval.py
# Test BF16 model
python ./scripts/run_lm_eval.py  
# Measure BF16 model to generate calibration data
QUANT_CONFIG=./scripts/inc_measure_config.json python ./scripts/run_inc_example_tp.py
# Quantize BF16 model to FP8
QUANT_CONFIG=./scripts/inc_quant_config.json python ./scripts/run_inc_example_tp.py
```

> [!CAUTION]
> ~~FAKE `EP` was hard-coded as 16. Please check `TEMP_EP` in vllm and `DEEPSEEK_EP` in INC.~~
