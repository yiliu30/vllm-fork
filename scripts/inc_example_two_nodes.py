from vllm import LLM, SamplingParams

import argparse
import os
from vllm.utils import reset_seed
reset_seed()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# get file location
file_path = os.path.abspath(__file__)
dataset_path = os.path.join(os.path.dirname(file_path), "../benchmarks")

default_dataset_path = "./tc_datasets/qa_out_0216_r1_300_max_30k_formatted.jsonl"

model_path = "/data/models/DeepSeek-R1/"
model_path = "/hf/hf_models/DeepSeek-R1"
model_path = "/mnt/disk5/hf_models/DeepSeek-R1-BF16"
# Parse the command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=model_path, help="The model path.")
parser.add_argument("--tokenizer", type=str, default=model_path, help="The model path.")
parser.add_argument("--tp_size", type=int, default=16, help="The number of threads.")
parser.add_argument("--ep_size", type=int, default=16, help="The number of threads.")
parser.add_argument("--dataset", type=str, default=None, help="The dataset.")
parser.add_argument("-dpath","--dataset_path", type=str, default=default_dataset_path, help="The dataset path.")
parser.add_argument("--isl", type=int, default=1024, help="input sequence length.")
parser.add_argument("--osl", type=int, default=128, help="output sequence length.")
parser.add_argument("--least_tokens", type=int, default=1024, help="calibration input sequence length.")
parser.add_argument("--nprompts", type=int, default=4, help="The number of prompts.")
parser.add_argument("--mode", type=str, default=None, choices=["quant", "prepare"], help="The mode.")
parser.add_argument("--random", action="store_true", help="Randomly sample prompts.")
parser.add_argument("--smoke", action="store_true", help="Smoke test")
parser.add_argument("--fp8_kvcache", action="store_true", help="Using FP8 KV cache.")
args = parser.parse_args()

max_num_seqs = 1

# ==-------------------------------------------------------------------------==
# Calibration parameters
# ==-------------------------------------------------------------------------==
least_tokens = args.least_tokens
num_samples = 512
max_new_tokens = 32
seed = 42
# https://github.com/deepseek-ai/DeepSeek-R1/blob/main/README.md#deepseek-r1-evaluation
# ... benchmarks requiring sampling, we use a temperature of 0.6, a top-p value of 0.95...
temperature = 0.6
temperature = 0 # greedy sample
top_p = 0.95


if __name__ == "__main__":

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]

    from utils import get_prompts, get_prompt_token_ids, get_pile_prompts
    if args.dataset == "tc":
        print("Using tc dataset.")
        from utils import get_prompts, get_prompt_token_ids, get_pile_prompts
        dataset_path = args.dataset_path
        from utils import get_tokenizer, sample_tc_requests
        tokenizer = get_tokenizer(args.tokenizer)
        num_requests = None
        if args.smoke:
            num_requests = 3
        prompts = sample_tc_requests(
            filepath=dataset_path,
            tokenizer=tokenizer,
            num_requests=num_requests,
            do_random=args.random,
        )
        # FIXME: (Yi) 1024 is too small for tc dataset
    else:
        if args.smoke:
            prompts = get_prompts()
        else:
            prompts = get_pile_prompts(args.model, num_samples)
    prompt_token_ids = get_prompt_token_ids(
        args.model, prompts, least_tokens
    )
    print(f"Got {len(prompts)} prompts, length of first prompt: {len(prompt_token_ids[0])}.")
    gt = None
    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        truncate_prompt_tokens=least_tokens,
    )
    model = args.model
    quant = True
    if args.mode is not None:
        print(f"Mode is {args.mode}.")
        if args.mode == "quant":
            # quantize
            if args.fp8_kvcache:
                print(f">>>>>>>>>>>>>> Using FP8 KV cache.")
                llm = LLM(
                    model=model, 
                    tokenizer=args.tokenizer,
                    tensor_parallel_size=args.tp_size,
                    distributed_executor_backend='ray',
                    trust_remote_code=True,
                    quantization='inc',
                    weights_load_device="cpu",
                    kv_cache_dtype="fp8_inc",
                    max_model_len=16384,
                    max_num_seqs=max_num_seqs,
                    dtype="bfloat16",
                )
            else:
                llm = LLM(
                    model=model, 
                    tokenizer=args.tokenizer,
                    tensor_parallel_size=args.tp_size,
                    distributed_executor_backend='ray',
                    trust_remote_code=True,
                    quantization='inc',
                    weights_load_device="cpu",
                    max_num_seqs=max_num_seqs,
                    max_model_len=16384,
                    dtype="bfloat16",
                )
        else:
            # prepare
            llm = LLM(
                model=model, 
                tokenizer=args.tokenizer,
                tensor_parallel_size=args.tp_size,
                distributed_executor_backend='ray',
                trust_remote_code=True,
                quantization='inc',
                max_model_len=16384,
                max_num_seqs=max_num_seqs,
                dtype="bfloat16",
            )
    else:
        llm = LLM(
            model=model, 
            tokenizer=args.tokenizer,
            tensor_parallel_size=args.tp_size,
            distributed_executor_backend='ray',
            trust_remote_code=True,
            max_model_len=16384,
            dtype="bfloat16",
        )

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(
        prompts=None, sampling_params=sampling_params, prompt_token_ids=prompt_token_ids
    )
    # Print the outputs.
    for output_i in range(len(outputs)):
        output = outputs[output_i]
        gt_i = None if gt is None else gt[output_i]
        prompt_token_ids = output.prompt_token_ids
        generated_text = output.outputs[0].text
        print("====================================")
        print(f"#Prompt tokens: {len(prompt_token_ids)}")
        print(f"Prompt[:10]: {prompt_token_ids[:10]!r}")
        print(f"Prompt[-10:]: {prompt_token_ids[-10:]!r}")
        print(f"Generated text: {generated_text!r}")
        print(f"Ground truth: {gt_i!r}")
        print("====================================")

    llm.llm_engine.model_executor.shutdown()
