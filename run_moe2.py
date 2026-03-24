

# ... 其他代码 ...

# MODEL_ID = "/models/Qwen1.5-MoE-A2.7B"
# MODEL_ID = "/models/Llama-3.1-8B-Instruct"
# MODEL_ID = "/models/Qwen3-30B-A3B"
# MODEL_ID = "/models/Qwen3-30B-A3B-W4A16-int4"
# MODEL_ID = "/models/Qwen3-30B-A3B-W4A16-mxfp4"
MODEL_ID = "/models/Qwen3-30B-A3B-W8A16"
# MODEL_ID = "/models/Qwen2.5-0.5B-Instruct-FP8-Dynamic-LLMC-TEST-Only"

def main():
    #model = vllm.LLM(MODEL_ID, max_model_len=2048, gpu_memory_utilization=0.8)
    # model = vllm.LLM(MODEL_ID, max_model_len=2048, gpu_memory_utilization=0.8, tensor_parallel_size=2)
    model = vllm.LLM(MODEL_ID, max_model_len=2048, gpu_memory_utilization=0.8, tensor_parallel_size=2)
    #model = vllm.LLM(MODEL_ID, max_model_len=2048, gpu_memory_utilization=0.8)
    import pdb;pdb.set_trace()
    
    output = model.generate("Hello, how are you?")
    for o in output:
        print(o.outputs[0].text)


    output = model.generate("The capital of France is")
    #output = model.generate("Hello, how are you?")
    for o in output:
        print(o.outputs[0].text)
        
    output = model.generate("def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n")
    #output = model.generate("Hello, how are you?")
    for o in output:
        print(o.outputs[0].text)
        
    output = model.generate("To be, or not to be, that is the")
    #output = model.generate("Hello, how are you?")
    for o in output:
        print(o.outputs[0].text)


if __name__ == '__main__':
    import vllm
    main()
