timestamp=$(date +%Y%m%d-%H%M%S)
LOG_FILE="./_pd_logs/decoder.${timestamp}.log"

export prefill_node_ip="10.111.231.48"
export decode_node_ip="10.111.231.48"
model_path="/mnt/weka/data/pytorch/llama3/Meta-Llama-3-8B/"
# model_path="/mnt/weka/data/pytorch/llama3.3/Meta-Llama-3.3-70B-Instruct"

python3 examples/online_serving/disagg_examples/disagg_proxy_demo.py \
    --model $model_path \
    --prefill $prefill_node_ip:8100 \
    --decode $decode_node_ip:8200 \
    --port 8088 2>&1 | tee $LOG_FILE