# DeepSeek Flash Mean TTFT / TPOT Comparison

Source: refreshed `run3` results from [deepseek_flash_random_input_sweep.md](/home/yiliu7/workspace/vllm/deepseek_flash_random_input_sweep.md).

Assumption: `ttfp` here refers to `TTFT` as reported by `vllm bench serve`.

## Summary

With the updated funnel kernel, `turbo_funnel_dense` is still worse than baseline at `32k`, slightly better than baseline at `128k`, and roughly flat but slightly worse at `512k`.

## Mean TTFT / TPOT

| input_len | baseline mean TTFT (ms) | funnel mean TTFT (ms) | TTFT delta | baseline mean TPOT (ms) | funnel mean TPOT (ms) | TPOT delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32768 | 205.07 | 257.27 | +25.45% | 3.62 | 3.67 | +1.38% |
| 131072 | 3936.25 | 3881.29 | -1.40% | 0.73 | 0.68 | -6.85% |
| 524288 | 22387.42 | 22648.44 | +1.17% | 0.84 | 0.88 | +4.76% |

## Takeaways

- At `32k`, funnel increases mean TTFT by `52.20 ms` and mean TPOT by `0.05 ms`.
- At `128k`, funnel reduces mean TTFT by `54.96 ms` and mean TPOT by `0.05 ms`.
- At `512k`, funnel increases mean TTFT by `261.02 ms` and mean TPOT by `0.04 ms`.
- The refreshed kernel largely removes the large long-context regression from the earlier funnel run, but it still does not produce a clear win across all lengths.
