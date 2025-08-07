# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

seed = 0
import random
random.seed(seed)
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import numpy as np
np.random.seed(seed)

# torch.use_deterministic_algorithms(True)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, 
                                 top_p=0.95, max_tokens=20)

# CUDA_LAUNlCH_BLOCKING=1  VLLM_USE_NVFP4_CT_EMULATIONS=1   python basic.py 


def main():
    # Create an LLM.
    model_name = "/home/yiliu7/models/deepseek-ai/DeepSeek-R1"
    llm = LLM(
        # gpu_memory_utilization=0.3,  # Set the GPU memory utilization.
        max_model_len = 1024,  # Set the maximum model length.
        enforce_eager=True,  # Enable eager mode for faster inference.
        trust_remote_code=True,
        # model = "/home/yiliu7/models/weiweiz1/DeepSeek-V2-Lite-NVFP4-W4A4-RTN",
        # model = "/home/yiliu7/models/Yi30/DeepSeek-V2-Lite-NVFP4-llmc"
        model = "/home/yiliu7/models/weiweiz1/DeepSeek-V2-Lite-NVFP4-W4A4-RTN"
        
        # model = "/home/yiliu7/models/deepseek-ai/DeepSeek-R1",
        # model="facebook/opt-125m",
        # model="/home/yiliu7/models/RedHatAI/Llama-3.1-70B-Instruct-NVFP4",
        # model=  "/home/yiliu7/models/RedHatAI/Qwen3-32B-NVFP4",
        # model="/dataset/weiweiz1/vllm/weiweiz1/DeepSeek-V2-Lite-NVFP4-W4A4-RTN"
        # model="/home/yiliu7/tmps/DeepSeek-V2-Lite-NVFP4-W4A4-RTN/",
        # model="/home/yiliu7/models/nm-testing/Qwen3-30B-A3B-NVFP4",
        # model="/home/yiliu7/tmps/DeepSeek-R1-BF16-w4g16",
        # model="/home/yiliu7/tmps/DeepSeek-R1-NVFP4-autoround/",
        # tensor_parallel_size=4,
        # enable_expert_parallel=True,  # Enable expert parallelism for faster inference.
        # tensor_parallel_size=2,  # Se
        # model = "/home/yiliu7/models/nvidia/Llama-3.3-70B-Instruct-FP4",
        # quantization="modelopt_fp4",
        # model = "/home/yiliu7/models/weiweiz1/DeepSeek-V2-Lite-NVFP4-autoround",
        # model = "/host_disk/nvme8n1/DeepSeek-R1-NVFP4-RTN-fuse/DeepSeek-R1-BF16-w4g16",
        # model=model_name,

        
        # model = "/home/yiliu7/tmps/DeepSeek-V2-Lite-NVFP4-autoround"   ,     
        # max_num_batch_tokens=4096,
        # model = "/home/yiliu7/models/weiweiz1/DeepSeek-R1-NVFP4-RTN",
        # model = "/home/yiliu7/tmps/DeepSeek-R1-NVFP4-RTN-fuse",
        # enable_expert_parallel=True,  # Enable expert parallelism for faster inference.
        # tensor_parallel_size=8,  # Set the tensor parallel size.
        
        # model= "/home/yiliu7/tmps/DeepSeek-R1-NVFP4-RTN-fuse/DeepSeek-R1-BF16-w4g16",
        # tensor_parallel_size=8,
        # enable_expert_parallel=True,
        # model = "/home/yiliu7/models/nvidia/DeepSeek-R1-FP4",

        # # enable_expert_parallel=True,  # Enable expert parallelism for faster inference.
        # tensor_parallel_size=8,  # Set the tensor parallel size.
        # quantization="modelopt_fp4",
    )
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params,
    )
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()



# vllm (pretrained=/mnt/weka/data/pytorch/DeepSeek-R1/,tensor_parallel_size=8,distributed_executor_backend=mp,trust_remote_code=true,max_model_len=4096,use_v2_block_manager=True,dtype=bfloat16,enable_expert_parallel=True,max_num_seqs=128), gen_kwargs: (None), limit: 256.0, num_fewshot: 5, batch_size: 128

# lm_eval --model local-completions \
#    --tasks gsm8k \
#    --model_args model=/mnt/disk5/Kimi-K2-Instruct-G2/,base_url=http://127.0.0.1:8688/v1/completions,max_concurrent=64,trust_remote_code=True \
#    --batch_size 64  \
#    --log_samples \
#    --output_path ./0528_acc/lm_eval_gsm8k_bs16_official_lm_eval.bs16.2nd.skipsdpa
   

# QUANT_CONFIG=${quant_file_path} \
# PT_HPU_LAZY_MODE=1 \
# VLLM_SKIP_WARMUP=true \
# PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
# PT_HPU_WEIGHT_SHARING=0 \


#  VLLM_WORKER_MULTIPROC_METHOD=spawn lm_eval --model vllm   \
#     --model_args "pretrained=/home/yiliu7/models/deepseek-ai/DeepSeek-R1,tensor_parallel_size=8,max_model_len=4096,max_num_seqs=128,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enable_expert_parallel=True,enforce_eager=True"   \
#         --tasks gsm8k --batch_size 128 \
#             --log_samples --output_path lmeval.ds.gsm8k.out \
#                 --limit 128 \
#                     --show_config 2>&1 | tee lmeval.log.ds.gsm8k.txt


# /home/yiliu7/models/weiweiz1/DeepSeek-V2-Lite-NVFP4-W4A4-RTN"
# Static global scale
# Generated Outputs:
# ------------------------------------------------------------
# Prompt:    'Hello, my name is'
# Output:    ' Erika. I am a 59 year old Female.\nMy主人希望我做一个乐观'
# ------------------------------------------------------------
# Prompt:    'The president of the United States is'
# Output:    ' not only the leader of the free world but also the commander-in-chief of the country’'
# ------------------------------------------------------------
# Prompt:    'The capital of France is'
# Output:    ' the city of Paris, and its modern skyline is dominated by the Eiffel Tower. Paris is the'
# ------------------------------------------------------------
# Prompt:    'The future of AI is'
# Output:    ' here!\nThe future of AI is here!\nWe are an AI enabled digital talent company with'
# ------------------------------------------------------------

# Dynamic global scale
# Generated Outputs:
# ------------------------------------------------------------
# Prompt:    'Hello, my name is'
# Output:    ' Mr. Jacobson, and I teach 10th grade English at Hillside High School.'
# ------------------------------------------------------------
# Prompt:    'The president of the United States is'
# Output:    ' being investigated for obstruction of justice, and the president of Russia is now claiming that the Obama administration had'
# ------------------------------------------------------------
# Prompt:    'The capital of France is'
# Output:    ' a city of the world’s most visited, so everyone feels quite familiar with Paris. The French'
# ------------------------------------------------------------
# Prompt:    'The future of AI is'
# Output:    ' here!\nThe future of AI is here!\nWe’re not just talking about the cool'
# ------------------------------------------------------------