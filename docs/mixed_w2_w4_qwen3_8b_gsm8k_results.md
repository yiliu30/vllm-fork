# Qwen3-8B Mixed W2/W4 Quantization Results

## Overview

Qwen3-8B was quantized with AutoRound using 2-bit MLP weights and 4-bit
attention weights. All quantized layers use 128-element groups, symmetric
weight-only quantization, and the `auto_round:auto_gptq` packing format. The
models were loaded by vLLM through `quantization=inc`; INC routes the W2 MLP
layers to Humming and the W4 attention layers to Marlin. The language-model
head remains unquantized.

## Debugging Summary

The original INC Humming adapter treated packed AutoGPTQ tensors as if they
were already in Humming's native layout. This skipped the required transpose
and repacking conversion and produced corrupted inference. The adapter was
changed to construct a GPTQ weight schema and use Humming's GPTQ-to-Humming
conversion path. After the fix, a W2 linear-layer forward matched a reference
dequantized matrix multiplication with cosine similarity `0.999997` and
relative L2 error `0.003`.

The first checkpoint also used pure RTN by setting both `iters=0` and
`disable_opt_rtn=True`. The requirement specified only `iters=0`, so a second
checkpoint was generated with optimized RTN (`disable_opt_rtn=False`). A third
checkpoint used AutoRound's standard 200-iteration SignRound optimization.

## GSM8K Evaluation

All runs used the same evaluation configuration: GSM8K 5-shot, greedy
decoding, batch size 128, `max_model_len=8129`, and `max_gen_toks=2048`.

| Quantization recipe | Flexible exact match | Strict exact match |
| --- | ---: | ---: |
| Pure RTN, `iters=0`, optimization disabled | 2.12% | 0.00% |
| Optimized RTN, `iters=0` | 34.19% ± 1.31% | 8.34% ± 0.76% |
| SignRound, `iters=200` | **64.44% ± 1.32%** | **64.14% ± 1.32%** |

The 200-iteration checkpoint improved flexible exact match by 30.25 percentage
points over optimized RTN and by 62.32 percentage points over pure RTN. These
results show that iterative rounding optimization is important for this
aggressive configuration, where every MLP projection is quantized to 2 bits.

## Artifacts

- Recommended model:
  `/dev/shm/.tmp_yi/Qwen3-8B-W2A16-MLP-W4A16-Attn-G128-Iters200/Qwen3-8B-w2g128`
- Recommended evaluation result:
  `/dev/shm/.tmp_yi/lm_eval_qwen3_8b_w2mlp_w4attn_g128_iters200_mlen8129_gen2048/__dev__shm__.tmp_yi__Qwen3-8B-W2A16-MLP-W4A16-Attn-G128-Iters200__Qwen3-8B-w2g128/results_2026-08-17T16-45-43.349345.json`
- Optimized-RTN model:
  `/dev/shm/.tmp_yi/Qwen3-8B-W2A16-MLP-W4A16-Attn-G128-OptRTN/Qwen3-8B-w2g128`
- Optimized-RTN evaluation result:
  `/dev/shm/.tmp_yi/lm_eval_qwen3_8b_w2mlp_w4attn_g128_optrtn_mlen8129_gen2048/__dev__shm__.tmp_yi__Qwen3-8B-W2A16-MLP-W4A16-Attn-G128-OptRTN__Qwen3-8B-w2g128/results_2026-08-17T07-51-57.390765.json`

Quantization and evaluation for the final 200-iteration model used GPU 0 only.
