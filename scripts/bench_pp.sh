#!/bin/bash
set -x

model_path=/models/DeepSeek-R1-BF16-w8afp8-dynamic-no-ste-G2
model_path=/mnt/disk2/hf_models/DeepSeek-R1-G2-static/
ip_addr=127.0.0.1
port=8688
log_dir="pp_results_418"

test_benchmark_serving_range() {
    local_input=$1
    local_output=$2
    local_max_concurrency=$3
    local_num_prompts=$4
    local_len_ratio=$5

    echo "running benchmark serving range test, input len: $local_input, output len: $local_output, len ratio: $local_len_ratio, concurrency: $local_max_concurrency"

    log_name=benchmark_serving_DS_random_batchsize_${local_max_concurrency}_in_${local_input}_out_${local_output}_ratio_${local_len_ratio}_rate_inf_prompts_${local_num_prompts}_$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)
    python3 ../benchmarks/benchmark_serving.py --backend vllm --model $model_path --trust-remote-code --host $ip_addr --port $port \
    --dataset-name random --random-input-len $local_input --random-output-len $local_output --random-range-ratio $local_len_ratio --max-concurrency $local_max_concurrency\
    --num-prompts $local_num_prompts --request-rate inf --seed 0 --ignore-eos \
    --save-result --result-filename ${log_name}.json  2>&1 | tee  ${log_dir}/${log_name}.txt

}


# Loop test

# Define an array of configurations
configs=(
    # "1024 1024 16 96 1"    
    # "1024 1024 32 192 1"
    # "1024 1024 64 192 1"
    # "2048 2048 64 192 1"
    "8192 8192 32 96 1"
    # "14336 1024 16 96 1"
    # "14336 1024 32 96 1"
    # "14336 1024 64 192 1"
)

# Iterate over the configurations and run each 3 times
for config in "${configs[@]}"; do
    for i in {1..3}; do
        echo "Running iteration $i for config: $config"
        test_benchmark_serving_range $config
    done
done

# #test_benchmark_serving_range 1024 1024 1 3 1
# test_benchmark_serving_range 2048 2048 24 192 1 
# test_benchmark_serving_range 2048 2048 32 96 0.8 #2>&1 | tee -a 2k_2k_24_benchmark.log


# #!/bin/bash

# config_list=(
#     # "8192,32,32,24,12"
#     # "8192,32,32,24,12"
#     # "8192,32,32,24,12"
#     # "8192,32,32,24,12"
#     # "8192,32,32,24,12"
#     # "8192,32,32,96,24"
#     # "8192,32,32,96,32"
#     # "8192,2048,2048,96,24"
#     # "8192,2048,2048,96,32"
#     # "8192,2048,2048,108,36"
#     # "8192,2048,2048,120,40"
#     # "16384,14336,1024,24,8"
#     # "16384,14336,1024,36,12"
#     # "16384,14336,1024,48,16"
#     "16384,14336,1024,30,10"
#     # "16384,15360,1024,24,8"
#     # "16384,15360,1024,36,12"
#     # "16384,15360,1024,48,16"
#     # "16384,8192,8192,24,8"
#     # "16384,8192,8192,36,12"
#     # "16384,8192,8192,48,16"
#     # "32768,16384,16384,12,4"
#     # "32768,16384,16384,18,6"
#     # "32768,16384,16384,24,8"
#     # "16384,4096,4096,48,16"
#     # "16384,4096,4096,72,24"
#     # "16384,4096,4096,96,32"
# )

# log_dir=$(pwd)/logs/$(date +"%Y%m%d")
# mkdir -p $log_dir

# echo 'max_model_len,input_tokens,output_tokens,num_prompt,max_concurrency,mean_ttft,mean_tpot,total_throughput,output_throughput' | tee -a ${log_dir}/summary.log


# for configs in "${config_list[@]}"; do
#     IFS=',' read -r max_model_len input_tokens output_tokens num_prompt max_concurrency <<< "${configs}"
#     log_name=${max_model_len}_input_${input_tokens}_output_${output_tokens}_concurrency_${max_concurrency}

#     bash -x single_8k_len_param.sh $max_model_len $input_tokens $output_tokens $max_concurrency > $log_dir/${log_name}_server.log 2>&1 &
#     server_launch_pid=$!
#     server_pid=$(ps -ef | grep openai | grep -v grep | awk '{print $2}')
#     echo "Server PID: $server_pid"
    
#     connected_info="Application startup complete"
#     timeout=900    
#     interval=5
#     connected=0

#     start_time=$(date +%s)    
#     while true; do
#         if grep -q "$connected_info" $log_dir/${log_name}_server.log; then
#             connected=1
#             break
#         elif grep -q "Fatal Python error" $log_dir/${log_name}_server.log; then
#             connected=0
#             echo "Server failed to launch with Fatal Python error"
#             break
#         fi
#         current_time=$(date +%s)
#         elapsed_time=$((current_time - start_time))
#         if [ $elapsed_time -ge $timeout ]; then
#             connected=0
#             break
#         fi

#         sleep $interval
#     done

#     echo "Connected status: $connected"

#     if [ $connected -eq 1 ]; then
#         echo "Server launched, proceeding to benchmark..."
#         source benchmark_vllm_client.sh

#         warmup_prompt=$num_prompt    
#         test_benchmark_serving_range $input_tokens $output_tokens $max_concurrency $warmup_prompt 0.8 2>&1 | tee -a $log_dir/${log_name}_benchmark.log

#         test_benchmark_serving_range $input_tokens $output_tokens $max_concurrency $num_prompt 0.8 2>&1 | tee -a $log_dir/${log_name}_benchmark.log
        
#         mean_tpot=$(grep 'Mean TPOT (ms):' $log_dir/${log_name}_benchmark.log | tail -1 | awk '{print $NF}')
#         mean_ttft=$(grep 'Mean TTFT (ms):' $log_dir/${log_name}_benchmark.log | tail -1 | awk '{print $NF}')
#         total_throughput=$(grep 'Total Token throughput (tok/s):' $log_dir/${log_name}_benchmark.log | tail -1 | awk '{print $NF}')
#         output_throughput=$(grep 'Output token throughput (tok/s):' $log_dir/${log_name}_benchmark.log | tail -1 | awk '{print $NF}')
#         echo "${max_model_len},${input_tokens},${output_tokens},${num_prompt},${max_concurrency},${mean_ttft},${mean_tpot},${total_throughput},${output_throughput}" | tee -a ${log_dir}/summary.log
        
#     else
#         echo "Server not launched within timeout, shutting down..."
#     fi
    
#     server_pid=$(ps -ef | grep openai | grep -v grep | awk '{print $2}')
#     kill -9 $server_pid
#     kill -9 $server_launch_pid
#     sleep 20


# done


:'
    "14336 1024 32 96 1"
ERROR:asyncio:Exception in callback functools.partial(<function _log_task_completion at 0x7d1274a14310>, error_callback=<bound method AsyncLLMEngine._error_callback of <vllm.engine.async_llm_engine.AsyncLLMEngine object at 0x7d125d99be80>>)
handle: <Handle functools.partial(<function _log_task_completion at 0x7d1274a14310>, error_callback=<bound method AsyncLLMEngine._error_callback of <vllm.engine.async_llm_engine.AsyncLLMEngine object at 0x7d125d99be80>>)>
Traceback (most recent call last):
  File "/mnt/disk3/yiliu4/vllm-fork/vllm/engine/async_llm_engine.py", line 56, in _log_task_completion
    return_value = task.result()
  File "/mnt/disk3/yiliu4/vllm-fork/vllm/engine/async_llm_engine.py", line 823, in run_engine_loop
    result = task.result()
  File "/mnt/disk3/yiliu4/vllm-fork/vllm/engine/async_llm_engine.py", line 746, in engine_step
    request_outputs = await self.engine.step_async(virtual_engine)
  File "/mnt/disk3/yiliu4/vllm-fork/vllm/engine/async_llm_engine.py", line 351, in step_async
    outputs = await self.model_executor.execute_model_async(
  File "/mnt/disk3/yiliu4/vllm-fork/vllm/executor/ray_distributed_executor.py", line 588, in execute_model_async
    return await super().execute_model_async(execute_model_req)
  File "/mnt/disk3/yiliu4/vllm-fork/vllm/executor/executor_base.py", line 348, in execute_model_async
    return await self._driver_execute_model_async(execute_model_req)
  File "/mnt/disk3/yiliu4/vllm-fork/vllm/executor/ray_distributed_executor.py", line 630, in _driver_execute_model_async
    results = await asyncio.gather(*tasks)
  File "/mnt/disk3/yiliu4/vllm-fork/vllm/utils.py", line 1400, in _run_task_with_lock
    return await task(*args, **kwargs)
  File "/usr/lib/python3.10/concurrent/futures/thread.py", line 58, in run
    result = self.fn(*self.args, **self.kwargs)
  File "/mnt/disk3/yiliu4/vllm-fork/vllm/worker/worker_base.py", line 583, in execute_method
    raise e
  File "/mnt/disk3/yiliu4/vllm-fork/vllm/worker/worker_base.py", line 574, in execute_method
    return run_method(target, method, args, kwargs)
  File "/mnt/disk3/yiliu4/vllm-fork/vllm/utils.py", line 2305, in run_method
    return func(*args, **kwargs)
  File "/mnt/disk3/yiliu4/vllm-fork/vllm/worker/hpu_worker.py", line 349, in execute_model
    output = LocalOrDistributedWorkerBase.execute_model(
  File "/mnt/disk3/yiliu4/vllm-fork/vllm/worker/worker_base.py", line 421, in execute_model
    output = self.model_runner.execute_model(
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/mnt/disk3/yiliu4/vllm-fork/vllm/worker/hpu_model_runner.py", line 2415, in execute_model
    hidden_states = self.model.forward(
  File "/usr/local/lib/python3.10/dist-packages/habana_frameworks/torch/hpu/graphs.py", line 745, in forward
    return wrapped_hpugraph_forward(
  File "/usr/local/lib/python3.10/dist-packages/habana_frameworks/torch/hpu/graphs.py", line 610, in wrapped_hpugraph_forward
    outputs = orig_fwd(*args, **kwargs)
  File "/mnt/disk3/yiliu4/vllm-fork/vllm/worker/hpu_model_runner.py", line 414, in forward
    hidden_states = self.model(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1742, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1753, in _call_impl
    return forward_call(*args, **kwargs)
  File "/mnt/disk3/yiliu4/vllm-fork/vllm/model_executor/models/deepseek_v3.py", line 715, in forward
    hidden_states = self.model(input_ids, positions, kv_caches,
  File "/mnt/disk3/yiliu4/vllm-fork/vllm/compilation/decorators.py", line 170, in __call__
    return self.forward(*args, **kwargs)
  File "/mnt/disk3/yiliu4/vllm-fork/vllm/model_executor/models/deepseek_v3.py", line 669, in forward
    hidden_states, residual = layer(positions, hidden_states,
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1742, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1848, in _call_impl
    return inner()
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1796, in inner
    result = forward_call(*args, **kwargs)
  File "/mnt/disk3/yiliu4/vllm-fork/vllm/model_executor/models/deepseek_v3.py", line 581, in forward
    hidden_states = self.self_attn(
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1742, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1848, in _call_impl
    return inner()
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1796, in inner
    result = forward_call(*args, **kwargs)
  File "/mnt/disk3/yiliu4/vllm-fork/vllm/model_executor/models/deepseek_v3.py", line 499, in forward
    return self.mla_attn(hidden_states_or_q_c, kv_c_normed, k_pe, kv_cache,
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1742, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1848, in _call_impl
    return inner()
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1796, in inner
    result = forward_call(*args, **kwargs)
  File "/mnt/disk3/yiliu4/vllm-fork/vllm/attention/layer.py", line 197, in forward
    return self.impl.forward(self, query, key, value,
  File "/mnt/disk3/yiliu4/vllm-fork/vllm/attention/backends/hpu_attn.py", line 470, in forward
    return self._forward_decode(q_nope, q_pe, kv_cache, attn_metadata,
  File "/mnt/disk3/yiliu4/vllm-fork/vllm/attention/backends/hpu_attn.py", line 520, in _forward_decode
    output = flat_pa_mla(
  File "/mnt/disk3/yiliu4/vllm-fork/vllm/attention/backends/hpu_attn.py", line 244, in flat_pa_mla
    block_bias = block_bias.view(key.size(0), 1, 1, -1)
RuntimeError: shape '[1236, 1, 1, -1]' is invalid for input of size 163840

The above exception was the direct cause of the following exception:

'