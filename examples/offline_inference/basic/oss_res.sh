
local-chat-completions (pretrained=/mnt/disk5/lmsys/gpt-oss-20b-bf16,base_url=http://localhost:8000/v1/chat/completions,max_gen_toks=1024,num_concurrent=128), gen_kwargs: (None), limit: 128.0, num_fewshot: 1, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     1|exact_match|↑  |0.8438|±  |0.0322|
|     |       |strict-match    |     1|exact_match|↑  |0.0156|±  |0.0110|

local-chat-completions (pretrained=/mnt/disk5/lmsys/gpt-oss-20b-bf16,base_url=http://localhost:8000/v1/chat/completions,max_gen_toks=1024,num_concurrent=128), gen_kwargs: (None), limit: None, num_fewshot: 1, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     1|exact_match|↑  |0.8203|±  |0.0106|
|     |       |strict-match    |     1|exact_match|↑  |0.0053|±  |0.0020|

https://github.com/lkk12014402/gpt-oss/blob/main/test_scripts/gsm8k.yaml