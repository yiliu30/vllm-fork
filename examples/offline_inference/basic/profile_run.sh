
timestamp=$(date +%Y%m%d_%H%M%S)
# # ### Profiling ###
export HABANA_PROFILE_WRITE_HLTV=1 
export HABANA_PROFILE=1
# hl-prof-config --use-template profile_api_with_nics --fuser on --trace-analyzer on --trace-analyzer-xlsx on -invoc csv,hltv -merged csv,hltv
hl-prof-config --use-template profile_api_with_nics --fuser on --trace-analyzer on --trace-analyzer-xlsx on
hl-prof-config --gaudi3
hl_prof_out_dir="pp_prof_hlv__${timestamp}"
hl-prof-config -o $hl_prof_out_dir
export GRAPH_VISUALIZATION=1
export PT_HPU_LAZY_MODE=1
torch_prof_out_dir="pp_prof_torch__${timestamp}"
echo "torch_prof_out_dir: $torch_prof_out_dir"
echo "hl_prof_out_dir: $hl_prof_out_dir"
# # curl -X POST http://localhost:8688/start_profile
# #  bash scripts/quickstart/benchmark_vllm_client.sh 
# #  curl -X POST http://localhost:8688/stop_profile
export VLLM_TORCH_PROFILER_DIR=$torch_prof_out_dir
export VLLM_ENGINE_PROFILER_ENABLED=1

# VLLM_USE_MXFP4_CT_EMULATIONS=1 VLLM_HPU_LOG_HPU_GRAPH=0 VLLM_INPUT_QUICK_QDQ=1   USE_CT_UNPACK=1  python basic_hpu.py 
VLLM_MXFP4_PREUNPACK_WEIGHTS=1 VLLM_USE_MXFP4_CT_EMULATIONS=1 VLLM_HPU_LOG_HPU_GRAPH=0 VLLM_INPUT_QUICK_QDQ=1   USE_CT_UNPACK=1  python basic_hpu.py --profile
