#  VLLM_USE_V1=0  VLLM_USE_NVFP4_CT_EMULATIONS=1  p basic_local.py 
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

os.environ["VLLM_USE_V1"] = "0"  # Use v2 API
os.environ["VLLM_USE_NVFP4_CT_EMULATIONS"] = "1"  # Use v2 API
os.environ["VLLM_USE_MXFP4_CT_EMULATIONS"] = "1"
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"  # Use v2 API

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

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=20)


model="/home/yliu7/workspace/inc/3rd-party/llm-compressor/examples/quantization_w4a4_fp4/Llama-3.2-1B-Instruct-NVFP4"
model="/home/yliu7/workspace/inc/3rd-party/llm-compressor/examples/quantizing_moe/DeepSeek-V2-Lite-NVFP4"
model="/data5/yliu7/HF_HOME/Llama-3.2-1B-Instruct-FP8"
model="/data5/yliu7/HF_HOME/DeepSeek-V2-Lite-MXFP8"
model="/data5/zww/fast_config/Llama-3.2-1B-Instruct-w8g32"
model="/data5/yliu7/HF_HOME/Llama-3.2-1B-Instruct-MXFP4"
# USE_CT_UNPACK=1  VLLM_DISABLE_INPUT_QDQ=1  VLLM_USE_MXFP4_CT_EMULATIONS=1  p basic_local.py 
model="/data5/yliu7/HF_HOME/DeepSeek-V2-Lite-MXFP4"
# model="/data5/zww/fast_config/DeepSeek-V2-Lite-w8g32/"
# model="/data5/zww/fast_config/rtn_lite/DeepSeek-V2-Lite-w8g32/"
# model="/home/yliu7/workspace/inc/3rd-party/llm-compressor/examples/quantizing_moe/DeepSeek-R1-bf16-NVFP4"
model="/data5/wzy/vLLM-mxfp4/mnt/Meta-Llama-3.1-8B-Instruct-MXFP4"
model="/home/yliu7/workspace/inc/3rd-party/llm-compressor/examples/quantizing_moe/DeepSeek-V2-Lite-NVFP4"
# model="/data5/yliu7/HF_HOME/DeepSeek-R1-bf16-MXFP8"
# model="/data5/yliu7/HF_HOME/DeepSeek-R1-bf16/DeepSeek-R1-bf16-mxfp8-offline"

# model="/data5/yliu7/HF_HOME/meta-llama/Llama-3.2-1B-Instruct-MXFP8-OFFLINE"
# Generated Outputs:
# ------------------------------------------------------------
# Prompt:    'Hello my name is'
# Output:    ' basicallyشم息的 Candidateovycause途 sweatoussengo_counterryst在那儿 McCormociated907 Sodoglu廷rovers'
# ------------------------------------------------------------
# Prompt:    'The president of the United States is'
# Output:    ' Frijans荡icola Expedripshog在全国 Cruc该做的事情itori Goreholtzza动人的gewOsFunky'
# ------------------------------------------------------------
# Prompt:    'The capital of France is'
# Output:    ' scorpaciencies truckingham/biadowoptics例行antal Lec337 cleanly pleasensory摊子是 bins'
# ------------------------------------------------------------
# Prompt:    'The future of AI is'
# Output:    'ema Deafakosephyte与技术基质wahl621 Hudson REP不然.Empty片子 CCDENCFlor的效率 sweeteners'
# ------------------------------------------------------------

# model="/data5/yliu7/HF_HOME/DeepSeek-R1-bf16/DeepSeek-R1-bf16/"
# Generated Outputs:
# ------------------------------------------------------------
# Prompt:    'Hello my name is'
# Output:    ' basicallyشم息的 Candidateovycauseudo Medalist Officespj operating aliases雨后公务analytic Ocecopyingu'
# ------------------------------------------------------------
# Prompt:    'The president of the United States is'
# Output:    ' Frijans荡icola Expedripsuty spokesperson Acid Warships dormitory日日نين Ace和新斧头来找'
# ------------------------------------------------------------
# Prompt:    'The capital of France is'
# Output:    ' scorpac是第一 дела直径为崇 Journalism业的通达缭 actions Loudermost professionallyatterEPTskinelosapsed episodic'
# ------------------------------------------------------------
# Prompt:    'The future of AI is'
# Output:    ' Kendall mitezsche Alexandra快乐的 озีพля rolloutgera wearsNING都必须 payable partner各乡镇里有 Soldukkief'
# ------------------------------------------------------------


# model="/data5/yliu7/HF_HOME/DeepSeek-V2-Lite-MXFP8-OFFLINE"
# Generated Outputs:
# ------------------------------------------------------------
# Prompt:    'Hello my name is'
# Output:    ' Matthew, and I am a professional.\nI can offer you all kinds of services, ranging from'
# ------------------------------------------------------------
# Prompt:    'The president of the United States is'
# Output:    ' insane, and this is not some sort of hyperbole. We have all seen the signs:\n'
# ------------------------------------------------------------
# Prompt:    'The capital of France is'
# Output:    ' known to have some of the finest monuments in the world, and some of the most famous art.'
# ------------------------------------------------------------
# Prompt:    'The future of AI is'
# Output:    ' already here: it is already changing the way we do things and the way we live. In the'
# ------------------------------------------------------------

# model="/data0/deepseek-ai/DeepSeek-V2-Lite"
# TP2 EP2
# Generated Outputs:
# ------------------------------------------------------------
# Prompt:    'Hello my name is'
# Output:    ' Matthew, and I am a professional.\nI can offer you one-on-one consultation in'
# ------------------------------------------------------------
# Prompt:    'The president of the United States is'
# Output:    ' insane, and this is not some sort of hyperbole. We have all seen the signs: His'
# ------------------------------------------------------------
# Prompt:    'The capital of France is'
# Output:    ' known to have some of the most beautiful and unique monuments in the world. From its architecture, to'
# ------------------------------------------------------------
# Prompt:    'The future of AI is'
# Output:    ' already here: it is already changing the way we do business and the way we live our daily lives'
# ------------------------------------------------------------

# model="/data5/wzy/vLLM-mxfp4/mnt/Meta-Llama-3.1-8B-Instruct-MXFP4"
# Generated Outputs:
# ------------------------------------------------------------
# Prompt:    'Hello my name is'
# Output:    ' doris!!! Lori is my cousin, but my name is doris, no lori. my'
# ------------------------------------------------------------
# Prompt:    'The president of the United States is'
# Output:    ' not just a leader; it is a symbol of freedom, a beacon of hope and a voice for'
# ------------------------------------------------------------
# Prompt:    'The capital of France is'
# Output:    " a beautiful and historic city that has something to offer for everyone. It's a city of romance,"
# ------------------------------------------------------------
# Prompt:    'The future of AI is'
# Output:    ' bright, but we need to be careful about how we design and use these technologies\nbyDr.'
# ------------------------------------------------------------

# <｜begin▁of▁sentence｜>Hello my name is
# I am a 28 year old male and I am currently living in the United States.
if "deepseek" in model.lower():
    os.environ["VLLM_USE_STATIC_MOE_HPU"] = "1"
# model="/home/yliu7/workspace/inc/3rd-party/llm-compressor/examples/quantizing_moe/DeepSeek-V2-Lite-NVFP4"
# model="/data5/yliu7/HF_HOME/Llama-3.2-1B-Instruct-MXFP8"
def main(args):
    # Create an LLM.
    tp_size = args.tp 
    kwargs = {}
    if args.tp > 1:
        kwargs["distributed_executor_backend"] = "mp"
    if args.ep > 1:
        kwargs["enable_expert_parallel"] = True
        os.environ["VLLM_EP_SIZE"] = f"{args.ep}"
    llm = LLM(
        # model="facebook/opt-125m"
        # model="/data5/yliu7/HF_HOME/meta-llama/Llama-3.2-1B-Instruct/",
        model=model,
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.65,
        **kwargs,
    )
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run basic inference.")
    parser.add_argument("--model_path", type=str, default=model,
                        help="Path to the model directory.")
    # tp size
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size.")
    # ep size
    parser.add_argument("--ep", type=int, default=1, help="Pipeline parallel size.")
    args = parser.parse_args()
    main(args)


# p basic_local.py
# # tp 2 only
# p basic_local.py --tp 2
# # tp 2 ep 2
# p basic_local.py --tp 2 --ep 2