# DeepSeek Flash Random-Input Sweep

Generated: `2026-07-10 07:02:03 UTC`

Recorded result policy: keep all three runs for each configuration and input length, and report only `run3` below.

## Server Logs

- `/home/yiliu7/workspace/vllm/logs/ds_sweep/server_baseline.log`
- `/home/yiliu7/workspace/vllm/logs/ds_sweep/server_turbo_funnel_dense.log`

## Recorded Results

| config | input_len | failed | duration_s | req/s | tok/s | mean_ttft_ms | median_ttft_ms | p99_ttft_ms | mean_tpot_ms | median_tpot_ms | p99_tpot_ms | mean_itl_ms | median_itl_ms | p99_itl_ms | run3 log |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 32768 | 0 | 2.38 | 4.20 | 137750.77 | 205.07 | 206.34 | 236.05 | 3.62 | 3.83 | 4.78 | 5.43 | 6.85 | 8.32 | `/home/yiliu7/workspace/vllm/outputs/bench_ds_sweep/baseline/32768/run3/bench.log` |
| baseline | 131072 | 0 | 39.43 | 0.25 | 33243.53 | 3936.25 | 3918.46 | 4025.97 | 0.73 | 0.21 | 4.85 | 6.55 | 1.89 | 43.65 | `/home/yiliu7/workspace/vllm/outputs/bench_ds_sweep/baseline/131072/run3/bench.log` |
| baseline | 524288 | 0 | 223.95 | 0.04 | 23411.05 | 22387.42 | 22308.74 | 23302.98 | 0.84 | 0.81 | 1.24 | 7.58 | 7.30 | 11.17 | `/home/yiliu7/workspace/vllm/outputs/bench_ds_sweep/baseline/524288/run3/bench.log` |
| turbo_funnel_dense | 32768 | 0 | 2.86 | 3.50 | 114758.48 | 249.21 | 246.04 | 283.80 | 4.01 | 4.32 | 4.78 | 5.47 | 6.71 | 8.60 | `/home/yiliu7/workspace/vllm/outputs/bench_ds_sweep/turbo_funnel_dense/32768/run3/bench.log` |
| turbo_funnel_dense | 131072 | 0 | 43.30 | 0.23 | 30272.15 | 4321.50 | 4277.41 | 4620.45 | 0.92 | 0.19 | 6.77 | 8.28 | 1.75 | 60.97 | `/home/yiliu7/workspace/vllm/outputs/bench_ds_sweep/turbo_funnel_dense/131072/run3/bench.log` |
| turbo_funnel_dense | 524288 | 0 | 261.17 | 0.04 | 20075.12 | 26109.45 | 26112.30 | 27977.88 | 0.77 | 0.71 | 1.43 | 6.92 | 6.39 | 12.88 | `/home/yiliu7/workspace/vllm/outputs/bench_ds_sweep/turbo_funnel_dense/524288/run3/bench.log` |

## Raw Artifacts

- Benchmark results root: `/home/yiliu7/workspace/vllm/outputs/bench_ds_sweep`
- Each `(config, input_len)` directory contains `run1`, `run2`, and `run3` with raw logs and saved JSON results.
