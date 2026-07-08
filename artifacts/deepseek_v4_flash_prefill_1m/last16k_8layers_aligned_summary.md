# DeepSeek V4 Flash Last-16K Prefill: Aligned 8-Layer Summary

## Scope

This note summarizes the reduced `8`-layer DeepSeek V4 Flash capture for the
last full `16384`-token chunk of a `999424`-token prefill.

The important point is alignment:

- `torch.profiler` was delayed to the `61`st worker step.
- `nsys` capture was triggered on the `61`st occurrence of
  `execute_context_1(16384)_generation_0(0)`.

So both tools are measuring the same real inference chunk, not model init,
CUDA-graph capture, or benchmark warmup.

## Artifacts

- Torch summary:
  `torch_last16k_8layers_align_20260708_141717/profiler_out_0.txt`
- Torch trace:
  `torch_last16k_8layers_align_20260708_141717/dp0_pp0_tp0_dcp0_ep0_rank0.1783520291145855298.pt.trace.json`
- Nsight report:
  `nsys_last16k_8layers_align_gpu0_20260708_141822.nsys-rep`
- Nsight sqlite export:
  `nsys_last16k_8layers_align_gpu0_20260708_141822.sqlite`

## Alignment Checks

The tools agree on call counts for the last full `16k` chunk:

- `execute_context_1(16384)_generation_0(0)`: `1` call
- `dsv4.layer_2.sparse_attn_indexer.prefill_topk`: `31` calls
- `dsv4.layer_4.sparse_attn_indexer.prefill_topk`: `31` calls
- `dsv4.layer_6.sparse_attn_indexer.prefill_topk`: `31` calls
- Total `prefill_topk`: `93` calls
- Torch `_C::top_k_per_row_prefill`: `93` calls
- Nsight `topKPerRowPrefill` kernel launches: `93`

Additional capture-correctness checks:

- `999424 = 61 x 16384`, so the `61`st occurrence of
  `execute_context_1(16384)_generation_0(0)` is the real last full `16k`
  prefill chunk.
- Benchmark warmup was disabled: `--num-iters-warmup 0`.
- Triton JIT warnings for the relevant prefill kernels were logged before
  profiling started, so those first-time compilation spikes are not included in
  the captured chunk.
- Torch profiling started at `14:18:10`, after the earlier JIT warnings at
  `14:17:59` to `14:18:00`.

## Validation Checks

Cross-tool timing agreement is good for the device-side ranges we care about:

| Range | Torch CUDA total | Nsight projected GPU time | Delta |
| --- | ---: | ---: | ---: |
| outer last-`16k` chunk | `242.651 ms` | `236.360 ms` | `+2.66%` |
| sparse indexer total | `119.098 ms` | `118.732 ms` | `+0.31%` |
| `prefill_topk` total | `30.261 ms` | `29.897 ms` | `+1.22%` |
| `layer_2` `prefill_topk` | `10.335 ms` | `9.938 ms` | `+3.99%` |
| `layer_4` `prefill_topk` | `9.976 ms` | `10.005 ms` | `-0.29%` |
| `layer_6` `prefill_topk` | `9.949 ms` | `9.953 ms` | `-0.04%` |
| `layer_2` `fp8_fp4_mqa_logits` | `28.969 ms` | `28.868 ms` | `+0.35%` |
| `layer_4` `fp8_fp4_mqa_logits` | `30.104 ms` | `30.145 ms` | `-0.13%` |
| `layer_6` `fp8_fp4_mqa_logits` | `29.683 ms` | `29.768 ms` | `-0.29%` |

Arithmetic consistency inside the `nsys` hierarchy is also good:

- sparse indexer total: `118.732 ms`
- `fp8_fp4_mqa_logits + prefill_topk + cp_gather...`: `118.738 ms`
- difference: about `0.007 ms`, which is just report-level rounding noise
- all attention + all MLP = `220.573 ms`
- outer chunk = `236.360 ms`
- residual outside those two buckets = `15.787 ms`

That `15.787 ms` residual is expected. It is work outside the
`attention.forward` and `mlp.forward` umbrellas, for example chunk plumbing and
other per-layer / per-request operations.

Sparse-layer timing is stable across `layer_2`, `layer_4`, and `layer_6`:

| Part | Values | Stability |
| --- | --- | ---: |
| `attention.forward` | `44.103`, `45.503`, `44.990 ms` | `1.29%` CV |
| `sparse_attn_indexer` | `38.824`, `40.168`, `39.740 ms` | `1.42%` CV |
| `fp8_fp4_mqa_logits` | `28.868`, `30.145`, `29.768 ms` | `1.81%` CV |
| `prefill_topk` | `9.938`, `10.005`, `9.953 ms` | `0.29%` CV |
| `mlp.forward` | `3.103`, `3.163`, `3.156 ms` | `0.85%` CV |

Important scope caveat:

- `torch.profiler` is reliable here for the low-level kernel/op timings and
  call counts such as `_C::top_k_per_row_prefill`,
  `fp8_fp4_mqa_logits`, and the total `vllm::sparse_attn_indexer`.
- `torch.profiler` is not reliable for inclusive decoder-level or
  `attention.forward` device-time ratios in this capture. Its
  `dsv4.layer_{2,4,6}.attention.forward` CUDA totals are only about
  `0.4 ms`, while `nsys` shows the real inclusive attention device time of
  about `45 ms`.
- Therefore the decoder-layer, attention, sparse-indexer, and topk proportion
  numbers in this note should be treated as `nsys` results, not torch results.

## Device-Time Summary

All ratios below use `nsys stats --report nvtx_gpu_proj_sum`.

Outer captured chunk:

- `execute_context_1(16384)_generation_0(0)`: `236.360 ms`, `1` call

Major buckets:

| Metric | Device time | Calls | Share of outer chunk |
| --- | ---: | ---: | ---: |
| All attention forward (`layers 0-7`) | `195.470 ms` | `8` | `82.70%` |
| All MLP forward (`layers 0-7`) | `25.103 ms` | `8` | `10.62%` |
| Sparse-layer attention (`layers 2,4,6`) | `134.596 ms` | `3` | `56.95%` |
| Dense-layer attention (`layers 0,1,3,5,7`) | `60.874 ms` | `5` | `25.75%` |
| Sparse indexer total (`layers 2,4,6`) | `118.732 ms` | `3` | `50.23%` |
| `fp8_fp4_mqa_logits` total | `88.780 ms` | `93` | `37.56%` |
| `prefill_topk` total | `29.897 ms` | `93` | `12.65%` |
| `cp_gather_indexer_k_quant_cache` total | `0.062 ms` | `3` | `0.03%` |

## Ratios

Chunk-level:

- Attention share of the chunk: `82.70%`
- MLP share of the chunk: `10.62%`
- Sparse indexer share of the chunk: `50.23%`
- `fp8_fp4_mqa_logits` share of the chunk: `37.56%`
- `prefill_topk` share of the chunk: `12.65%`

Inside attention:

- Sparse indexer in all attention: `60.74%`
- `prefill_topk` in all attention: `15.29%`
- `fp8_fp4_mqa_logits` in all attention: `45.42%`

Inside sparse-layer attention (`layers 2,4,6` only):

- Sparse indexer in sparse-layer attention: `88.21%`
- `fp8_fp4_mqa_logits` in sparse-layer attention: `65.96%`
- `prefill_topk` in sparse-layer attention: `22.21%`

Inside sparse indexer:

- `fp8_fp4_mqa_logits` in sparse indexer: `74.77%`
- `prefill_topk` in sparse indexer: `25.18%`
- `cp_gather_indexer_k_quant_cache` in sparse indexer: `0.05%`

## Per-Sparse-Layer Breakdown

Here each sparse decoder layer is normalized to `100%`, where:

- decoder-layer total = `attention.forward + mlp.forward`
- `attention_other` = `attention.forward - sparse_attn_indexer`
- `fp8_fp4_mqa_logits`, `prefill_topk`, and `cp_gather...` are nested inside
  `sparse_attn_indexer`, but their percentages below are still reported against
  the full decoder-layer total

### `layer_2` (`47.206 ms` = `100%`)

| Part | Time | % of decoder layer | Calls |
| --- | ---: | ---: | ---: |
| `attention.forward` | `44.103 ms` | `93.43%` | `1` |
| `mlp.forward` | `3.103 ms` | `6.57%` | `1` |
| `sparse_attn_indexer` | `38.824 ms` | `82.24%` | `1` |
| `attention_other` | `5.279 ms` | `11.18%` | `1` |
| `fp8_fp4_mqa_logits` | `28.868 ms` | `61.15%` | `31` |
| `prefill_topk` | `9.938 ms` | `21.05%` | `31` |
| `cp_gather_indexer_k_quant_cache` | `0.020 ms` | `0.04%` | `1` |

### `layer_4` (`48.666 ms` = `100%`)

| Part | Time | % of decoder layer | Calls |
| --- | ---: | ---: | ---: |
| `attention.forward` | `45.503 ms` | `93.50%` | `1` |
| `mlp.forward` | `3.163 ms` | `6.50%` | `1` |
| `sparse_attn_indexer` | `40.168 ms` | `82.54%` | `1` |
| `attention_other` | `5.335 ms` | `10.96%` | `1` |
| `fp8_fp4_mqa_logits` | `30.145 ms` | `61.94%` | `31` |
| `prefill_topk` | `10.005 ms` | `20.56%` | `31` |
| `cp_gather_indexer_k_quant_cache` | `0.021 ms` | `0.04%` | `1` |

### `layer_6` (`48.146 ms` = `100%`)

| Part | Time | % of decoder layer | Calls |
| --- | ---: | ---: | ---: |
| `attention.forward` | `44.990 ms` | `93.44%` | `1` |
| `mlp.forward` | `3.156 ms` | `6.56%` | `1` |
| `sparse_attn_indexer` | `39.740 ms` | `82.54%` | `1` |
| `attention_other` | `5.250 ms` | `10.90%` | `1` |
| `fp8_fp4_mqa_logits` | `29.768 ms` | `61.83%` | `31` |
| `prefill_topk` | `9.953 ms` | `20.67%` | `31` |
| `cp_gather_indexer_k_quant_cache` | `0.021 ms` | `0.04%` | `1` |

Short read:

- Each sparse decoder layer is about `93.5%` attention and `6.5%` MLP.
- In each sparse decoder layer, `sparse_attn_indexer` alone is about `82.4%`
  of the whole layer time.
- `prefill_topk` is about `20.6%` to `21.1%` of the whole sparse decoder layer.
- `fp8_fp4_mqa_logits` is about `61.1%` to `61.9%` of the whole sparse decoder
  layer.

## Call-Level Averages

Using the aligned `nsys` projected GPU totals:

- `dsv4.layer_2.sparse_attn_indexer.prefill_topk`: `31` calls,
  `320.6 us/call`
- `dsv4.layer_4.sparse_attn_indexer.prefill_topk`: `31` calls,
  `322.7 us/call`
- `dsv4.layer_6.sparse_attn_indexer.prefill_topk`: `31` calls,
  `321.1 us/call`

- `dsv4.layer_2.sparse_attn_indexer.fp8_fp4_mqa_logits`: `31` calls,
  `931.2 us/call`
- `dsv4.layer_4.sparse_attn_indexer.fp8_fp4_mqa_logits`: `31` calls,
  `972.4 us/call`
- `dsv4.layer_6.sparse_attn_indexer.fp8_fp4_mqa_logits`: `31` calls,
  `960.3 us/call`

## Short Conclusion

For the aligned `8`-layer last-`16k` prefill chunk, attention dominates the
real inference device time, and the sparse indexer dominates the sparse-layer
attention time.

Within sparse attention:

- the main cost is `fp8_fp4_mqa_logits`
- the secondary cost is `prefill_topk`
- `prefill_topk` is substantial, but it is not the largest sparse-indexer
  component

Numerically:

- `prefill_topk` is about `25%` of sparse-indexer device time
- `fp8_fp4_mqa_logits` is about `75%` of sparse-indexer device time
- `prefill_topk` is about `12.6%` of the full captured chunk
- all sparse-indexer work is about `50.2%` of the full captured chunk
