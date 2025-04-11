```bash
# Install vllm
git clone https://github.com/yiliu30/vllm-fork.git
git checkout inc-r1-g2
cd vllm-fork
pip install -r requirements-hpu.txt
VLLM_TARGET_DEVICE=hpu pip install -e .  --no-build-isolation;


# Install INC
pip install git+https://github.com/intel/neural-compressor.git@r1-woq

# Get calibration file
huggingface-cli download Yi30/inc-woq-pile-default-g2  --local-dir ./scripts/nc_workspace_measure_kvache

# Benchmark
cd ./scripts
# Update model_path to "/mnt/disk2/hf_models/DeepSeek-R1-G2/"
bash single_8k_len.sh
```