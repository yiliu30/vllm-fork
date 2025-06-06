set -x

echo "Starting warmup for vLLM server..."

bash scripts/quickstart/benchmark_vllm_client.sh 

sleep 10
echo "Triggering profiling for vLLM server..."
curl -X POST http://localhost:8688/start_profile

sleep 10
echo "Profiling started. Running benchmark client..."
bash scripts/quickstart/benchmark_vllm_client.sh 

sleep 10
echo "Stopping profiling for vLLM server..."
curl -X POST http://localhost:8688/stop_profile