from vllm import LLM, SamplingParams

import argparse
import os
from typing import Any, List, Tuple
from transformers import (PreTrainedTokenizerBase, AutoTokenizer)
import random
import datasets
from vllm.utils import reset_seed
reset_seed()

os.environ["VLLM_EP_SIZE"] = "8"
os.environ["VLLM_TP_SIZE"] = "8"

# get file location
file_path = os.path.abspath(__file__)
dataset_path = os.path.join(os.path.dirname(file_path), "../benchmarks")

model_path = "/data/models/DeepSeek-R1/"
model_path = "/hf/hf_models/DeepSeek-R1"
# model_path = "deepseek-ai/DeepSeek-V2-Lite"
model_path = "/mnt/disk5/hf_models/DeepSeek-R1-BF16"
# Parse the command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=model_path, help="The model path.")
parser.add_argument("--tokenizer", type=str, default=model_path, help="The model path.")
parser.add_argument("--tp_size", type=int, default=8, help="The number of threads.")
parser.add_argument("--ep_size", type=int, default=8, help="The number of threads.")
parser.add_argument("--dataset", type=str, default=None, help="The dataset.")
parser.add_argument("--isl", type=int, default=1024, help="input sequence length.")
parser.add_argument("--osl", type=int, default=128, help="output sequence length.")
parser.add_argument("--nprompts", type=int, default=4, help="The number of prompts.")
parser.add_argument("--random", action="store_true", help="Randomly sample prompts.")
# add mode
parser.add_argument("--mode", type=str, default="q", required=False, help="The mode.")
parser.add_argument("--fp8_kvcache", action="store_true", help="Using FP8 KV cache.")
args = parser.parse_args()

# os.environ["VLLM_SKIP_WARMUP"] = "true"
# os.environ["HABANA_VISIBLE_DEVICES"] = "ALL"
# os.environ['HABANA_VISIBLE_MODULES'] ='0,1,2,3,4,5,6,7'
# os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "true"
# os.environ["PT_HPU_WEIGHT_SHARING"] = "0"
# os.environ['PT_HPUGRAPH_DISABLE_TENSOR_CACHE']='1'
# os.environ['GLOO_SOCKET_IFNAME']='eth0'

# os.environ["VLLM_MOE_N_SLICE"] = "1" if args.ep_size > 1 else "4"

# os.environ["VLLM_MLA_DISABLE_REQUANTIZATION"] = "1"

# os.environ["VLLM_RAY_DISABLE_LOG_TO_DRIVER"] = "0"
# os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "0"
# os.environ["RAY_DEDUP_LOGS"] = "1"
# os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

# ==-------------------------------------------------------------------------==
# Calibration parameters
least_tokens = 1024
num_samples = 512
max_new_tokens = 32
seed = 42
# https://github.com/deepseek-ai/DeepSeek-R1/blob/main/README.md#deepseek-r1-evaluation
"""
... benchmarks requiring sampling, we use a temperature of 0.6, a top-p value of 0.95...
"""
temperature = 0.6
temperature = 0 # greedy sample
top_p = 0.95
# ==-------------------------------------------------------------------------==


def sample_sonnet_requests(
    dataset_path: str,
    num_requests: int,
    input_len: int,
    prefix_len: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, str, int, int, None]]:
    assert (
        input_len > prefix_len
    ), "'args.sonnet-input-len' must be greater than 'args.prefix-input-len'."

    # Load the dataset.
    with open(dataset_path, encoding='utf-8') as f:
        poem_lines = f.readlines()

    # Tokenize the poem lines.
    poem_token_ids = tokenizer(poem_lines).input_ids
    average_poem_len = sum(
        len(token_ids) for token_ids in poem_token_ids) / len(poem_token_ids)

    # Base prefix for all requests.
    base_prompt = "Pick as many lines as you can from these poem lines:\n"
    base_message = [{
        "role": "user",
        "content": base_prompt,
    }]
    base_prompt_formatted = tokenizer.apply_chat_template(
        base_message, add_generation_prompt=True, tokenize=False)
    base_prompt_offset = len(tokenizer(base_prompt_formatted).input_ids)

    assert (
        input_len > base_prompt_offset
    ), f"Please set 'args.sonnet-input-len' higher than {base_prompt_offset}."
    num_input_lines = round(
        (input_len - base_prompt_offset) / average_poem_len)

    # First approximately `prefix_len` number of tokens in the
    # prompt are fixed poem lines.
    assert (
        prefix_len > base_prompt_offset
    ), f"Please set 'args.sonnet-prefix-len' higher than {base_prompt_offset}."

    num_prefix_lines = round(
        (prefix_len - base_prompt_offset) / average_poem_len)
    prefix_lines = poem_lines[:num_prefix_lines]

    # Sample the rest of lines per request.
    sampled_requests: List = []
    for _ in range(num_requests):
        num_lines_needed = num_input_lines - num_prefix_lines
        sampled_lines = "".join(prefix_lines +
                                random.choices(poem_lines, k=num_lines_needed))
        

        prompt = f"{base_prompt}{sampled_lines}"
        message = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
        prompt_formatted = tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False)
        sampled_requests.append(prompt_formatted)

    return sampled_requests, None

def sample_gsm8k_requests(
    num_requests: int, tokenizer: PreTrainedTokenizerBase, do_random: bool = False
) -> List[Tuple[str, str]]:
    # Load the dataset from huggingface.
    dataset = datasets.load_dataset("openai/gsm8k", "main")
    prompts = dataset["train"]["question"]
    expected_responses = dataset["train"]["answer"]
    few_shots = 5
    base_prompt = [f"Question: {prompts[i]}\nAnswer: {expected_responses[i]}\n" for i in range(few_shots)]
    base_prompt = "\n".join(base_prompt)
    base_prompt = f"{base_prompt}\n"
    
    # Sample the requests.
    sampled_requests: List = []
    sampled_response: List = []
    for j in range(num_requests):
        i = random.choice(range(len(prompts[few_shots:]))) if do_random else j + few_shots
        prompt = f"{base_prompt}Question: {prompts[i]}\nAnswer: "
        # message = [
        #     {
        #         "role": "user",
        #         "content": prompt,
        #     },
        # ]
        # prompt = tokenizer.apply_chat_template(
        #     message, add_generation_prompt=True, tokenize=False)
        expected_response = expected_responses[i]
        sampled_requests.append(prompt)
        sampled_response.append(expected_response)

    return sampled_requests, sampled_response

if __name__ == "__main__":

    # Sample prompts.
    
    if args.dataset == "sonnet":
        # Sample sonnet requests.
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        prompts, gt = sample_sonnet_requests(
            dataset_path=f"{dataset_path}/sonnet.txt",
            num_requests=args.nprompts,
            input_len=args.isl,
            prefix_len=200,
            tokenizer=tokenizer,
        )
    elif args.dataset == "gsm8k":
        # Sample GSM8K requests.
        args.osl=128
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        prompts, gt = sample_gsm8k_requests(
            num_requests=args.nprompts,
            tokenizer=tokenizer,
            do_random=args.random,
        )
    else:
        prompts = [
            "Hello, my name is",
            # "The president of the United States is",
            # "The capital of France is",
            "The future of AI is",
        ]

        from utils import get_prompts, get_prompt_token_ids, get_pile_prompts

        # prompts = get_prompts()
        # Got the unseen prompts.
        # smoke_num_samples = 10
        # prompts = get_pile_prompts(args.model, num_samples * 2)
        # smoke_prompts = [
        #     "Hello, my name is",
        #     "The president of the United States is",
        #     "The capital of France is",
        #     "The future of AI is",
        # ]

        # smoke_prompts = smoke_prompts + prompts[-smoke_num_samples:]
        smoke_prompts = get_prompts()
        prompt_token_ids = get_prompt_token_ids(
            args.model, smoke_prompts, least_tokens
        )
        gt = None
    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        truncate_prompt_tokens=least_tokens,
    )
    model = args.model
    assert args.mode in ["p", "q", None], f"Invalid mode: {args.mode}"
    print(f"Running in {args.mode} mode")
    if args.mode is None:
        llm = LLM(
            model=model, 
            tokenizer=args.tokenizer,
            tensor_parallel_size=args.tp_size,
            distributed_executor_backend='mp',
            trust_remote_code=True,
            # quantization=quantization,
            max_model_len=16384,
            dtype="bfloat16",
        )
    else:
        quantization = "inc"
        if args.fp8_kvcache:
            print(f">>>>>>>>>>>>>> Using FP8 KV cache.")
            llm = LLM(
                model=model, 
                tokenizer=args.tokenizer,
                tensor_parallel_size=args.tp_size,
                distributed_executor_backend='mp',
                trust_remote_code=True,
                quantization=quantization,
                weights_load_device="cpu",
                kv_cache_dtype="fp8_inc",
                max_model_len=16384,
                dtype="bfloat16",
            )
        else:
            llm = LLM(
                model=model, 
                tokenizer=args.tokenizer,
                tensor_parallel_size=args.tp_size,
                distributed_executor_backend='mp',
                trust_remote_code=True,
                quantization=quantization,
                weights_load_device="cpu",
                max_model_len=16384,
                dtype="bfloat16",
            )

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(
        # prompts=smoke_prompts,
        sampling_params=sampling_params,
        prompt_token_ids=prompt_token_ids
    )
    # Print the outputs.
    for output_i in range(len(outputs)):
        output = outputs[output_i]
        gt_i = None if gt is None else gt[output_i]
        prompt_token_ids = output.prompt_token_ids
        generated_text = output.outputs[0].text
        print("====================================")
        prompt = output.prompt
        print(f"prompt: {prompt!r}")
        print(f"prompt_token_ids[:10]: {prompt_token_ids[:10]!r}")
        print(f"prompt_token_ids[-10:]: {prompt_token_ids[-10:]!r}")
        print(f"Generated text: {generated_text!r}")
        print(f"Ground truth: {gt_i!r}")
        print("====================================")

    llm.llm_engine.model_executor.shutdown()
