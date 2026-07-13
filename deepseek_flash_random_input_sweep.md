# DeepSeek Flash Random-Input Sweep

Generated: `2026-07-12 12:46:24 UTC`

Recorded result policy: keep all three runs for each configuration and input length, and report only `run3` below.
Baseline rows are from the original `2026-07-10` sweep. `turbo_funnel_dense` rows were refreshed on `2026-07-12` after the funnel kernel update.

## Server Logs

- `/home/yiliu7/workspace/vllm/logs/ds_sweep/server_baseline.log`
- `/home/yiliu7/workspace/vllm/logs/ds_sweep_funnel_refresh_20260712/server_turbo_funnel_dense.log`

## Recorded Results

| config | input_len | failed | duration_s | req/s | tok/s | mean_ttft_ms | median_ttft_ms | p99_ttft_ms | mean_tpot_ms | median_tpot_ms | p99_tpot_ms | mean_itl_ms | median_itl_ms | p99_itl_ms | run3 log |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 32768 | 0 | 2.38 | 4.20 | 137750.77 | 205.07 | 206.34 | 236.05 | 3.62 | 3.83 | 4.78 | 5.43 | 6.85 | 8.32 | `/home/yiliu7/workspace/vllm/outputs/bench_ds_sweep/baseline/32768/run3/bench.log` |
| baseline | 131072 | 0 | 39.43 | 0.25 | 33243.53 | 3936.25 | 3918.46 | 4025.97 | 0.73 | 0.21 | 4.85 | 6.55 | 1.89 | 43.65 | `/home/yiliu7/workspace/vllm/outputs/bench_ds_sweep/baseline/131072/run3/bench.log` |
| baseline | 524288 | 0 | 223.95 | 0.04 | 23411.05 | 22387.42 | 22308.74 | 23302.98 | 0.84 | 0.81 | 1.24 | 7.58 | 7.30 | 11.17 | `/home/yiliu7/workspace/vllm/outputs/bench_ds_sweep/baseline/524288/run3/bench.log` |
| turbo_funnel_dense | 32768 | 0 | 2.91 | 3.44 | 112796.11 | 257.27 | 252.40 | 277.96 | 3.67 | 3.82 | 4.65 | 5.41 | 6.83 | 8.82 | `/home/yiliu7/workspace/vllm/outputs/bench_ds_sweep_funnel_refresh_20260712/turbo_funnel_dense/32768/run3/bench.log` |
| turbo_funnel_dense | 131072 | 0 | 38.88 | 0.26 | 33717.72 | 3881.29 | 3882.33 | 4014.01 | 0.68 | 0.17 | 4.76 | 6.09 | 1.55 | 42.80 | `/home/yiliu7/workspace/vllm/outputs/bench_ds_sweep_funnel_refresh_20260712/turbo_funnel_dense/131072/run3/bench.log` |
| turbo_funnel_dense | 524288 | 0 | 226.57 | 0.04 | 23140.93 | 22648.44 | 22419.60 | 24046.69 | 0.88 | 0.88 | 1.66 | 7.91 | 7.88 | 14.90 | `/home/yiliu7/workspace/vllm/outputs/bench_ds_sweep_funnel_refresh_20260712/turbo_funnel_dense/524288/run3/bench.log` |

## Raw Artifacts

- Baseline results root: `/home/yiliu7/workspace/vllm/outputs/bench_ds_sweep`
- Refreshed funnel results root: `/home/yiliu7/workspace/vllm/outputs/bench_ds_sweep_funnel_refresh_20260712`
- Each `(config, input_len)` directory contains `run1`, `run2`, and `run3` with raw logs and saved JSON results.
