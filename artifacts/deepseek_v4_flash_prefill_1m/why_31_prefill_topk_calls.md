# Why There Are 31 `prefill_topk` Calls In The Last-16K Capture

## Scope

This note explains the aligned reduced-`8`-layer capture for the last full
`16k` prefill chunk of the `999424`-token run.

Observed fact from profiling:

- each sparse C4 layer (`layer_2`, `layer_4`, `layer_6`) has
  `31` `dsv4.layer_X.sparse_attn_indexer.prefill_topk` calls
- the same layers also have `31` `fp8_fp4_mqa_logits` calls
- each of those layers has only `1`
  `cp_gather_indexer_k_quant_cache` call

This is expected from the source code.

## Short Answer

The `31` calls are not caused by:

- `31` different model layers
- `31` different top-k values
- `31` attention chunks in FlashMLA

They come from query-dimension sub-chunking inside the sparse indexer prefill
path.

For the last aligned `16k` chunk:

- query rows `M = 16384`
- compressed KV rows `N = 999424 // 4 = 249856`
- dense logits shape would be `[16384, 249856]`
- that would require `16384 * 249856 * 4 = 16,374,276,096` bytes
  (`~15.25 GiB`) just for the float32 logits tensor

The sparse indexer caps this logits tensor with
`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`, which defaults to `512 MB`
([vllm/envs.py](/home/yiliu7/workspace/vllm/vllm/envs.py:996)).

So the indexer splits the `16384` query rows into smaller row slices, and that
math gives exactly `31` sub-chunks.

## Source Walkthrough

### 1. Only C4 layers use runtime prefill top-k

In the reduced `8`-layer config:

- `index_topk = 512`
- `sliding_window = 128`
- first `8` `compress_ratios` are `[0, 0, 4, 128, 4, 128, 4, 128]`

See [config.json](/home/yiliu7/workspace/vllm/artifacts/deepseek_v4_flash_prefill_1m/DeepSeek-V4-Flash-8layers/config.json:18) and [config.json](/home/yiliu7/workspace/vllm/artifacts/deepseek_v4_flash_prefill_1m/DeepSeek-V4-Flash-8layers/config.json:66).

That means:

- `layer_0`, `layer_1`: dense / SWA-only
- `layer_2`, `layer_4`, `layer_6`: `compress_ratio = 4` sparse C4 layers
- `layer_3`, `layer_5`, `layer_7`: `compress_ratio = 128` sparse C128 layers

The runtime sparse indexer path is created only for `compress_ratio == 4`
layers in [attention.py](/home/yiliu7/workspace/vllm/vllm/models/deepseek_v4/attention.py:268).

So the `prefill_topk` calls only appear in `layer_2`, `layer_4`, and `layer_6`.

### 2. The last aligned prefill chunk is one request with `16384` query tokens

The aligned run captures the `61`st `16384`-token worker chunk of the
`999424`-token prefill:

- `999424 = 61 x 16384`
- last chunk token range is `[983040, 999424)`

For one request, the metadata builder computes:

- `query_len = 16384`
- `seq_len = 999424`
- `compress_ratio = 4`
- compressed sequence length `N = seq_len // 4 = 249856`

The builder uses `seq_len // compress_ratio` for indexer prefill chunking in
[indexer.py](/home/yiliu7/workspace/vllm/vllm/v1/attention/backends/mla/indexer.py:523).

### 3. The indexer explicitly splits prefill work when `M * N * 4` is too large

The split logic is in
[split_indexer_prefill_chunks()](/home/yiliu7/workspace/vllm/vllm/v1/attention/backends/mla/indexer.py:74).

Its contract is explicit:

- it respects the `N` workspace limit
- it respects the logits limit `M * N * 4 <= max_logits_bytes`
- if one request is still too large, it sub-chunks on the query dimension

Relevant lines:

- logits cap comment:
  [indexer.py](/home/yiliu7/workspace/vllm/vllm/v1/attention/backends/mla/indexer.py:82)
- query-dimension sub-chunking:
  [indexer.py](/home/yiliu7/workspace/vllm/vllm/v1/attention/backends/mla/indexer.py:108)
- per-subchunk row budget:
  [indexer.py](/home/yiliu7/workspace/vllm/vllm/v1/attention/backends/mla/indexer.py:115)

The metadata builder feeds that function with:

- `compressed_seq_lens_cpu[num_decodes:]`
- `prefill_query_lens_cpu`
- `max_logits_bytes = VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024`

See [indexer.py](/home/yiliu7/workspace/vllm/vllm/v1/attention/backends/mla/indexer.py:528) and [indexer.py](/home/yiliu7/workspace/vllm/vllm/v1/attention/backends/mla/indexer.py:531).

### 4. Why the math gives 31

For this last chunk:

- `chunk_m = 16384`
- `chunk_n = 249856`
- `max_logits_bytes = 512 * 1024 * 1024 = 536,870,912`
- `max_logits_elems = max_logits_bytes // 4 = 134,217,728`

So the maximum query rows per sub-chunk are:

```text
max_q = max_logits_elems // chunk_n
      = 134,217,728 // 249,856
      = 537
```

Then:

```text
16384 = 30 x 537 + 274
```

So the builder emits:

- `30` sub-chunks of `537` rows
- `1` final sub-chunk of `274` rows
- total `31` sub-chunks

And each sub-chunk becomes one `prefill_topk` call.

## Runtime Path Per Sub-Chunk

The metadata builder turns each `(req_slice, query_slice)` into one runtime
`chunk` object in
[build_prefill_chunk_metadata()](/home/yiliu7/workspace/vllm/vllm/v1/attention/backends/mla/indexer.py:653).

At runtime, `sparse_attn_indexer()` loops over `prefill_metadata.chunks` in
[sparse_attn_indexer.py](/home/yiliu7/workspace/vllm/vllm/model_executor/layers/sparse_attn_indexer.py:600).

For each chunk it does:

1. Slice the query rows for that chunk.
2. Compute dense logits with `fp8_fp4_mqa_logits`.
3. Run `_run_prefill_topk` on those logits.
4. Write the local top-k indices into the shared `topk_indices_buffer`.

The relevant calls are here:

- logits:
  [sparse_attn_indexer.py](/home/yiliu7/workspace/vllm/vllm/model_executor/layers/sparse_attn_indexer.py:792)
- top-k:
  [sparse_attn_indexer.py](/home/yiliu7/workspace/vllm/vllm/model_executor/layers/sparse_attn_indexer.py:808)

So:

- `31` query sub-chunks
- `31` `fp8_fp4_mqa_logits` calls
- `31` `prefill_topk` calls

per C4 sparse layer.

## Why There Is Only 1 `cp_gather...` Call

The query dimension is split, but the compressed KV pool is not.

The metadata builder marks later query sub-slices with:

- `skip_kv_gather = query_slice.start > 0`

See [indexer.py](/home/yiliu7/workspace/vllm/vllm/v1/attention/backends/mla/indexer.py:558).

And `build_prefill_chunk_metadata()` preserves that flag for nonzero query
offsets:

- [indexer.py](/home/yiliu7/workspace/vllm/vllm/v1/attention/backends/mla/indexer.py:712)

At runtime, the gather only happens when `not chunk.skip_kv_gather`:

- [sparse_attn_indexer.py](/home/yiliu7/workspace/vllm/vllm/model_executor/layers/sparse_attn_indexer.py:606)

So the behavior is:

- first query sub-chunk: gather compressed KV once
- remaining `30` query sub-chunks: reuse the same gathered KV workspace

That is why each sparse layer shows:

- `1` `cp_gather_indexer_k_quant_cache`
- `31` `fp8_fp4_mqa_logits`
- `31` `prefill_topk`

## What Happens After Top-K

The `31` `prefill_topk` calls only build the compressed-prefix candidate list.

Later, attention merges those compressed indices with the SWA window in
[combine_topk_swa_indices()](/home/yiliu7/workspace/vllm/vllm/models/deepseek_v4/common/ops/cache_utils.py:519).

For each token:

- compressed top-k valid length is
  `min((pos + 1) // compress_ratio, topk_tokens)`
- SWA length is `min(pos + 1, window_size)`

See [cache_utils.py](/home/yiliu7/workspace/vllm/vllm/models/deepseek_v4/common/ops/cache_utils.py:603).

For this last chunk:

- positions are already very large (`>= 983040`)
- `compress_ratio = 4`
- `topk_tokens = 512`
- `window_size = 128`

So every token in the last chunk is already saturated at:

- compressed top-k length `512`
- SWA length `128`

The `31` call count is therefore about indexer memory management, not about
the final logical top-k length per token.

## Final Interpretation

For the aligned last-`16k` run, the correct interpretation is:

- one real prefill chunk enters each sparse C4 layer
- inside that layer, the sparse indexer splits the query rows into `31`
  sub-chunks because the dense logits buffer would otherwise exceed the default
  `512 MB` cap
- each sub-chunk launches exactly one `fp8_fp4_mqa_logits` and one
  `prefill_topk`
- the compressed KV gather is done once and then reused across the remaining
  query sub-chunks

So the observed `31` is a direct and expected consequence of the source code.
