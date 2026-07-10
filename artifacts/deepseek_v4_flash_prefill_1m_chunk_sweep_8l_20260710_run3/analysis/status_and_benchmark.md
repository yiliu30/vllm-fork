# Status And Benchmark

## Status

- Sweep completed successfully on a single GPU with the reduced `8`-layer model:
  [DeepSeek-V4-Flash-8layers](</home/yiliu7/workspace/vllm/artifacts/deepseek_v4_flash_prefill_1m/DeepSeek-V4-Flash-8layers/config.json>)
- GPU setup: `CUDA_VISIBLE_DEVICES=1`
- Capture method: `nsys + NVTX`, second identical inference iteration
- Warmup setting: `VLLM_DEEP_GEMM_WARMUP=skip`
- Source of truth: `nsys stats --report nvtx_gpu_proj_sum`
- Sampled chunks: `1, 8, 16, 24, 32, 40, 48, 56, 64`

## Benchmark Command

```bash
CUDA_VISIBLE_DEVICES=1 \
VLLM_DEEP_GEMM_WARMUP=skip \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
.venv/bin/python tools/profile_deepseek_v4_prefill_breakdown.py \
  --model artifacts/deepseek_v4_flash_prefill_1m/DeepSeek-V4-Flash-8layers \
  --seq-lens 1048576 \
  --chunk-indices 1,8,16,24,32,40,48,56,64 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.6 \
  --output-dir artifacts/deepseek_v4_flash_prefill_1m_chunk_sweep_8l_20260710_run3 \
  --analyze
```

## Correctness

- All sampled scenarios completed and produced `.nsys-rep` reports.
- Observed `prefill_topk` calls matched expected calls exactly:
  `1, 4, 8, 13, 16, 21, 25, 29, 32`
- `fp8_fp4_mqa_logits` call counts matched `prefill_topk` in every sampled chunk.
- The call-count trend aligns with `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=512` and query-side sub-chunking.

## Benchmark Summary

| Chunk | Effective Seq Len | Avg Decoder ms | Attention % of Layer | Indexer % of Layer | TopK % of Layer | TopK Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1/64` | `16384` | 8.95 | 65.54% | 7.50% | 4.46% | `1` |
| `8/64` | `131072` | 12.86 | 75.71% | 34.32% | 7.24% | `4` |
| `16/64` | `262144` | 17.28 | 82.21% | 51.73% | 11.38% | `8` |
| `24/64` | `393216` | 23.76 | 86.77% | 64.07% | 13.82% | `13` |
| `32/64` | `524288` | 29.46 | 89.56% | 71.31% | 14.58% | `16` |
| `40/64` | `655360` | 33.64 | 90.83% | 74.88% | 15.34% | `21` |
| `48/64` | `786432` | 43.41 | 92.83% | 80.57% | 15.54% | `25` |
| `56/64` | `917504` | 43.87 | 92.97% | 80.80% | 20.49% | `29` |
| `64/64` | `1048576` | 51.47 | 94.01% | 84.35% | 19.80% | `32` |

## Key Findings

- As chunk order increases, the average sparse decoder layer becomes increasingly attention-dominated.
- The sparse indexer grows faster than attention overall and becomes the dominant part of sparse-layer attention in later chunks.
- `prefill_topk` grows materially with chunk order, but `fp8_fp4_mqa_logits` remains the larger child inside the indexer.
- For this fixed `16k` query chunk, effective sequence length grows linearly with chunk index, while `prefill_topk` calls grow approximately linearly with sequence length under the fixed logits budget.

## Related Artifacts

- HTML visualization: [summary.html](/home/yiliu7/workspace/vllm/artifacts/deepseek_v4_flash_prefill_1m_chunk_sweep_8l_20260710_run3/analysis/summary.html)
- Decoder-layer trend CSV: [decoder_layer_trend.csv](/home/yiliu7/workspace/vllm/artifacts/deepseek_v4_flash_prefill_1m_chunk_sweep_8l_20260710_run3/analysis/decoder_layer_trend.csv)
- Chunk trend CSV: [chunk_trend.csv](/home/yiliu7/workspace/vllm/artifacts/deepseek_v4_flash_prefill_1m_chunk_sweep_8l_20260710_run3/analysis/chunk_trend.csv)
- Short findings: [findings.md](/home/yiliu7/workspace/vllm/artifacts/deepseek_v4_flash_prefill_1m_chunk_sweep_8l_20260710_run3/analysis/findings.md)
