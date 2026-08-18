# MXFP4 GSM8K Evaluation Results

This document consolidates the GSM8K evaluations for the standard MXFP4 and
dual-scale MXFP4 Qwen3-8B checkpoints.

## Evaluation setup

- Evaluator: `lm-eval 0.4.12`
- Backend: vLLM
- Task: `gsm8k`
- Dataset split: GSM8K `test`, 1,319 examples
- Shot setting: task default, which is 5-shot
- Sampling: greedy (`temperature=0.0`, `do_sample=False`)
- Device: GPU 2
- Batch size: 16
- Dtype: `bfloat16`
- Tensor parallel size: 1

Models:

- Standard MXFP4: `/dev/shm/.tmp_yi/workspace/auto-round/Qwen3-8B-MXFP4_RCEIL/Qwen3-8B-mxfp-w4g32`
- Dual-scale MXFP4: `/dev/shm/.tmp_yi/Qwen/Qwen3-8B-MXFP4-dualscale`

## Original evaluation

Settings: `max_model_len=4096`; default lm-eval generation limit.

| Model | Flexible exact match | Strict exact match |
|---|---:|---:|
| Standard MXFP4 | 86.28% ± 0.95% | 85.75% ± 0.96% |
| Dual-scale MXFP4 | 84.00% ± 1.01% | 83.85% ± 1.01% |

The standard checkpoint led by 2.27 percentage points on flexible matching and
1.90 points on strict matching.

Result files:

- Standard: `/dev/shm/.tmp_yi/lm_eval_qwen3_8b_mxfp4_default_b16/__dev__shm__.tmp_yi__workspace__auto-round__Qwen3-8B-MXFP4_RCEIL__Qwen3-8B-mxfp-w4g32/results_2026-08-17T02-32-50.365328.json`
- Dual-scale: `/dev/shm/.tmp_yi/lm_eval_qwen3_8b_dualscale_default_b16/__dev__shm__.tmp_yi__Qwen__Qwen3-8B-MXFP4-dualscale/results_2026-08-17T02-48-42.614665.json`

## Updated evaluation

Settings: `max_model_len=8129` and `max_gen_toks=2048`. In the lm-eval vLLM
backend, `max_gen_toks` is the generation equivalent of `max_new_tokens`.

| Model | Flexible exact match | Strict exact match |
|---|---:|---:|
| Standard MXFP4 | 88.78% ± 0.87% | 88.78% ± 0.87% |
| Dual-scale MXFP4 | 85.97% ± 0.96% | 86.05% ± 0.95% |

The standard checkpoint led by 2.81 percentage points on flexible matching and
2.73 points on strict matching.

Result files:

- Standard: `/dev/shm/.tmp_yi/lm_eval_qwen3_8b_mxfp4_default_mlen8129_gen2048/__dev__shm__.tmp_yi__workspace__auto-round__Qwen3-8B-MXFP4_RCEIL__Qwen3-8B-mxfp-w4g32/results_2026-08-17T03-23-47.987626.json`
- Dual-scale: `/dev/shm/.tmp_yi/lm_eval_qwen3_8b_dualscale_mlen8129_gen2048/__dev__shm__.tmp_yi__Qwen__Qwen3-8B-MXFP4-dualscale/results_2026-08-17T03-56-59.793800.json`

## Coarse-scale experiment: `amax / 7.25`

This experiment changed only the weight coarse-scale numerator from 6.0 to
7.25. Runtime activation quantization remained `amax / 6.0` so that the
experiment isolated the weight-scale change. The evaluation used batch size 64
and `gpu_memory_utilization=0.85` because GPU 2 had another allocation; the
model length and generation limits were unchanged.

| Model | Flexible exact match | Strict exact match |
|---|---:|---:|
| Dual-scale, weight `amax / 7.25` | 81.73% ± 1.06% | 82.64% ± 1.04% |

Compared with the dual-scale `amax / 6.0` result, the `amax / 7.25` variant was
lower by 4.24 percentage points on flexible matching and 3.41 points on strict
matching. Since `7.25` is larger than the FP4 maximum value 6.0, the normalized
coarse-block maximum can exceed the representable FP4 range and saturate.

Result file:

- Dual-scale `amax / 7.25`: `/dev/shm/.tmp_yi/lm_eval_qwen3_8b_dualscale_a725_mlen8129_gen2048_b64/__dev__shm__.tmp_yi__Qwen__Qwen3-8B-MXFP4-dualscale-a725/results_2026-08-17T05-16-23.729980.json`

## Notes

- The two evaluation rounds use different generation limits, so their scores
  should not be compared as a controlled apples-to-apples experiment.
- GPU 2 was released after both evaluations; observed memory usage was 4 MiB.
- The dual-scale run was substantially slower under the 2048-token generation
  limit because some Qwen3 responses used long reasoning generations.
- GPU 2 was released after the `amax / 7.25` evaluation; observed memory usage
  was 4 MiB.
