#! /bin/bash

# set -x

BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/utils.sh

Help() {
    # Display Help
    echo "Start vllm server for a huggingface model on Gaudi."
    echo
    echo "Syntax: bash start_gaudi_vllm_server.sh <-w> [-n:m:a:d:i:p:o:b:g:u:e:lc:sf] [-h]"
    echo "options:"
    echo "w  Weights of the model, str, could be model id in huggingface or local path"
    echo "n  Number of HPUs to use, [1-8], default=1"
    echo "m  Module IDs of the HPUs to use, comma separated int in [0-7], default=None"
    echo "a  API server URL, str, 'IP:PORT', default=127.0.0.1:30001"
    echo "d  Data type, str, ['bfloat16'|'float16'|'fp8'|'awq'|'gptq'], default='bfloat16'"
    echo "i  Input range, str, format='input_min,input_max', default='4,1024'"
    echo "p  Max number of prefill sequences, int, default=8192/input_min"
    echo "o  Output range, str, format='output_min,output_max', default='4,2048'"
    echo "b  max_num_seqs for vllm, int, default=128"
    echo "g  max-seq-len-to-capture, int, default=8192"
    echo "u  gpu-memory-utilization, float, default=0.9"
    echo "e  extra vLLM server parameters, str, default=None"
    echo "l  Use linear bucketing or not, bool, default=false"
    echo "c  Cache HPU recipe to the specified path, str, default=None"
    echo "s  Skip warmup or not, bool, default=false"
    echo "f  Enable high-level profiler or not, bool, default=false"
    echo "h  Help info"
    echo
}

weights_path=""
num_hpu=1
module_ids=None
host=127.0.0.1
port=30001
dtype=bfloat16
input_range=(4 1024)
max_num_prefill_seqs=""
output_range=(4 2048)
max_num_seqs=$PREFERED_NUM_SEQS
max_seq_len_to_capture=$PREFERED_SEQ_LEN_TO_CAPTURE
gpu_memory_utilization=0.9
extra_params=()
cache_path=""
skip_warmup=false
profile=false

# Get the options
while getopts hw:n:m:a:d:i:p:o:b:e:g:u:lc:sf flag; do
    case $flag in
    h) # display Help
        Help
        exit
        ;;
    w) # get model path
        weights_path=$OPTARG ;;
    n) # get number of HPUs
        num_hpu=$OPTARG ;;
    m) # get module ids to use
        module_ids=$OPTARG ;;
    a) # get the URL of the server
        host=${OPTARG%%:*}
        port=${OPTARG##*:}
        ;;
    d) # get data type
        dtype=$OPTARG ;;
    i) # input range
        IFS="," read -r -a input_range <<< "$OPTARG" ;;
    p) # max number of prefill sequences
        max_num_prefill_seqs=$OPTARG ;;
    o) # output range
        IFS="," read -r -a output_range <<< "$OPTARG" ;;
    b) # batch size
        max_num_seqs=$OPTARG ;;
    e) # extra vLLM server parameters
        IFS=" " read -r -a extra_params <<< "$OPTARG" ;;
    g) # max-seq-len-to-capture
        max_seq_len_to_capture=$OPTARG ;;
    u) # gpu-memory-utilization
        gpu_memory_utilization=$OPTARG ;;
    l) # use linear bucketing
        use_linear_bucketing=true ;;
    c) # use_recipe_cache
        cache_path=$OPTARG ;;
    s) # skip_warmup
        skip_warmup=true ;;
    f) # enable high-level profiler
        profile=true ;;
    \?) # Invalid option
        echo "Error: Invalid option"
        Help
        exit
        ;;
    esac
done

if [ "$weights_path" == "" ]; then
    echo "[ERROR]: No model specified. Usage:"
    Help
    exit
fi

model_name=$(basename "$weights_path")
model_name_lower=$(echo "$model_name" | tr '[:upper:]' '[:lower:]')

input_min=${input_range[0]}
input_max=${input_range[1]}
output_min=${output_range[0]}
output_max=${output_range[1]}

max_num_batched_tokens=$(( $input_max + $output_max ))
if [ "$max_num_batched_tokens" -lt $PREFERED_BATCHED_TOKENS ]; then
    max_num_batched_tokens=$PREFERED_BATCHED_TOKENS
fi
# Ceiling max_num_batched_tokens to a multiple of BLOCK_SIZE
max_num_batched_tokens=$( ceil $max_num_batched_tokens $BLOCK_SIZE )

if [ "$max_num_prefill_seqs" == "" ]; then
    # Ceiling input_min to a multiple of BLOCK_SIZE
    input_min_ceil=$( ceil $input_min $BLOCK_SIZE )
    max_num_prefill_seqs=$(( max_num_batched_tokens / input_min_ceil ))
fi

max_model_len=$(( $input_max + $output_max ))
# Ceiling max_model_len to a multiple of BLOCK_SIZE
max_model_len=$( ceil $max_model_len $BLOCK_SIZE )

if [ "$input_min" == "$input_max" ]; then
    disable_zero_padding=true
fi


echo "Starting vllm server for ${model_name} from ${weights_path} with:"
echo "    device: ${num_hpu} HPUs with module_ids=${module_ids}"
echo "    URL: ${host}:${port}"
echo "    input_range: [${input_min}, ${input_max}]"
echo "    output_range: [${output_min}, ${output_max}]"
echo "    max_num_seqs: ${max_num_seqs}"

case_name=serve_${model_name}_${dtype}_${device}_in${input_min}-${input_max}_out${output_min}-${output_max}_bs${max_num_seqs}_tp${num_hpu}_$(date +%F-%H-%M-%S)

set_config

${NUMA_CTL} \
echo python3 -m vllm.entrypoints.openai.api_server \
    --device hpu \
    --block-size "${BLOCK_SIZE}" \
    --host "${host}" --port "${port}" \
    --model "${weights_path}" \
    --dtype "${DATA_TYPE}" \
    --max-num-seqs "${max_num_seqs}" \
    --max-num-batched-tokens "${max_num_batched_tokens}" \
    --max-seq-len-to-capture "${max_seq_len_to_capture}" \
    --gpu-memory-utilization "${gpu_memory_utilization}" \
    --max-model-len "${max_model_len}" \
    --tensor-parallel-size "${num_hpu}" \
    --trust-remote-code \
    --seed 2025 \
    --distributed_executor_backend "${dist_backend}" \
    "${extra_params[@]}" \
    |& tee "${case_name}".log 2>&1
