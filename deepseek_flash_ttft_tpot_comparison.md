# DeepSeek Flash Mean TTFT / TPOT Comparison

Source: recorded `run3` results from [deepseek_flash_random_input_sweep.md](/home/yiliu7/workspace/vllm/deepseek_flash_random_input_sweep.md).

Assumption: `ttfp` here refers to `TTFT` as reported by `vllm bench serve`.

## Summary

`turbo_funnel_dense` is worse than baseline on mean TTFT at every tested length. Mean TPOT is also worse at `32k` and `128k`, and only improves at `512k`.

## Mean TTFT / TPOT

| input_len | baseline mean TTFT (ms) | funnel mean TTFT (ms) | TTFT delta | baseline mean TPOT (ms) | funnel mean TPOT (ms) | TPOT delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32768 | 205.07 | 249.21 | +21.52% | 3.62 | 4.01 | +10.76% |
| 131072 | 3936.25 | 4321.50 | +9.79% | 0.73 | 0.92 | +26.41% |
| 524288 | 22387.42 | 26109.45 | +16.63% | 0.84 | 0.77 | -8.76% |

## Takeaways

- At `32k`, funnel increases mean TTFT by `44.13 ms` and mean TPOT by `0.39 ms`.
- At `128k`, funnel increases mean TTFT by `385.25 ms` and mean TPOT by `0.19 ms`.
- At `512k`, funnel increases mean TTFT by `3722.02 ms` while reducing mean TPOT by `0.07 ms`.
- The dominant regression is prefill / first-token latency rather than decode.
