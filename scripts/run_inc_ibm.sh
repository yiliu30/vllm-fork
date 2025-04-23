export PT_HPU_LAZY_MODE=1 
export GRAPH_VISUALIZATION=1 

# QUANT_CONFIG=./scripts/inc_quant_with_fp8kv_config_post_process.json \
# export OFFICIAL_FP8_MODEL="/mnt/weka/data/pytorch/llama2/Llama-2-7b-hf/"
export OFFICIAL_FP8_MODEL=/dev/shm/hub/ibm-granite/granite-20b-code-instruct-8k

# QUANT_CONFIG="./scripts/inc_measure_config_ibm.json" \
# python ./scripts/run_example_single_node.py \
#     --model ${OFFICIAL_FP8_MODEL} \
#     --tokenizer ${OFFICIAL_FP8_MODEL} \
#     --osl 32 \
#     --max_num_seqs 1 \
#     --inc

QUANT_CONFIG="./scripts/inc_quant_config_ibm.json" \
python ./scripts/run_example_single_node.py \
    --model ${OFFICIAL_FP8_MODEL} \
    --tokenizer ${OFFICIAL_FP8_MODEL} \
    --osl 32 \
    --max_num_seqs 1 \
    --inc \
    --fp8_kv_cache
