# 1M Chunk Sweep Findings

Scenario:
- Model: `artifacts/deepseek_v4_flash_prefill_1m/DeepSeek-V4-Flash-8layers`
- GPU: single card (`CUDA_VISIBLE_DEVICES=1`)
- Source of truth: `nsys stats --report nvtx_gpu_proj_sum`
- Capture method: second identical inference iteration, with `VLLM_DEEP_GEMM_WARMUP=skip`

Correctness checks:
- All sampled chunks completed successfully and produced `.nsys-rep` reports.
- Every scenario matched the expected per-sparse-layer `prefill_topk` call count exactly: `1, 4, 8, 13, 16, 21, 25, 29, 32`.
- `fp8_fp4_mqa_logits` call counts matched `prefill_topk` call counts in every sampled chunk.
- The captured range remained the same logical 16K prefill chunk: `execute_context_1(16384)_generation_0(0)`.

Trend:
- Chunk device time grows from `74.96 ms` at chunk `1/64` to `253.05 ms` at chunk `64/64` (`3.38x`).
- Attention grows from `50.56%` to `82.22%` of chunk time.
- Sparse indexer grows from `5.32%` to `62.59%` of attention time.
- `prefill_topk` grows from `1.20 ms` to `30.56 ms` and from `1.60%` to `12.08%` of chunk time.

Interpretation:
- Later 16K chunks pay more sparse-attention work because the effective history grows with chunk order.
- The indexer splits the same 16K query chunk into more query-side sub-passes as the compressed KV length grows, which increases both `prefill_topk` and `fp8_fp4_mqa_logits` calls.
- `fp8_fp4_mqa_logits` remains the dominant child inside the indexer at all larger sampled chunks; `prefill_topk` is smaller but still rises materially with chunk order.

Artifacts:
- Summary markdown: [summary.md](/home/yiliu7/workspace/vllm/artifacts/deepseek_v4_flash_prefill_1m_chunk_sweep_8l_20260710_run3/analysis/summary.md)
- HTML view: [summary.html](/home/yiliu7/workspace/vllm/artifacts/deepseek_v4_flash_prefill_1m_chunk_sweep_8l_20260710_run3/analysis/summary.html)
- Trend CSV: [chunk_trend.csv](/home/yiliu7/workspace/vllm/artifacts/deepseek_v4_flash_prefill_1m_chunk_sweep_8l_20260710_run3/analysis/chunk_trend.csv)
