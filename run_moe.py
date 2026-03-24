

# ... 其他代码 ...

MODEL_ID = "/models/Qwen1.5-MoE-A2.7B"

def main():
    model = vllm.LLM(MODEL_ID)
    output = model.generate("Hello, how are you?")
    for o in output:
        print(o.outputs[0].text)


if __name__ == '__main__':
    import vllm
    main()