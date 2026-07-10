# DeepSeek V4 Flash Prefill Breakdown Sweep

## Scope

Targeted `nsys + NVTX` capture of real prefill inference chunks using second-iteration capture (`prime_iterations` first, then capture the same chunk on the next iteration).

- Created at: `20260710T005822Z`
- Model: `artifacts/deepseek_v4_flash_prefill_1m/DeepSeek-V4-Flash-8layers`
- Tensor parallel size: `1`
- Sequence lengths: `[1048576]`
- Chunk size: `16384`
- Prime iterations: `1`

## Scenario Table

| Scenario | Chunk | Effective seq len | Chunk ms | Attention % | Indexer % of chunk | Indexer % of attention | TopK % of indexer | Logits % of indexer | TopK calls/layer | Expected |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `seq_1m_chunk_01_of_64` | `chunk_01_of_64` | `16384` | 74.96 | 50.56% | 2.69% | 5.32% | 59.46% | 38.98% | `1` | `1` |
| `seq_1m_chunk_08_of_64` | `chunk_08_of_64` | `131072` | 94.60 | 57.89% | 14.00% | 24.18% | 21.08% | 78.69% | `4` | `4` |
| `seq_1m_chunk_16_of_64` | `chunk_16_of_64` | `262144` | 111.61 | 65.34% | 24.03% | 36.78% | 22.00% | 77.87% | `8` | `8` |
| `seq_1m_chunk_24_of_64` | `chunk_24_of_64` | `393216` | 137.67 | 71.63% | 33.17% | 46.31% | 21.56% | 78.35% | `13` | `13` |
| `seq_1m_chunk_32_of_64` | `chunk_32_of_64` | `524288` | 171.49 | 71.39% | 36.75% | 51.47% | 20.44% | 79.49% | `16` | `16` |
| `seq_1m_chunk_40_of_64` | `chunk_40_of_64` | `655360` | 178.91 | 77.43% | 42.24% | 54.55% | 20.49% | 79.45% | `21` | `21` |
| `seq_1m_chunk_48_of_64` | `chunk_48_of_64` | `786432` | 213.52 | 80.94% | 49.14% | 60.72% | 19.29% | 80.66% | `25` | `25` |
| `seq_1m_chunk_56_of_64` | `chunk_56_of_64` | `917504` | 223.13 | 80.69% | 47.66% | 59.06% | 25.35% | 74.60% | `29` | `29` |
| `seq_1m_chunk_64_of_64` | `chunk_64_of_64` | `1048576` | 253.05 | 82.22% | 51.46% | 62.59% | 23.47% | 76.31% | `32` | `32` |

## Chunk Order Trend

| Scenario | Chunk idx | Effective seq len | Chunk ms | Attention % | Indexer % of chunk | Indexer % of attention | TopK ms | TopK % of chunk | TopK % of attention | TopK % of indexer | TopK calls | Expected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `seq_1m_chunk_01_of_64` | `1` | `16384` | 74.96 | 50.56% | 2.69% | 5.32% | 1.20 | 1.60% | 3.16% | 59.46% | `1` | `1` |
| `seq_1m_chunk_08_of_64` | `8` | `131072` | 94.60 | 57.89% | 14.00% | 24.18% | 2.79 | 2.95% | 5.10% | 21.08% | `4` | `4` |
| `seq_1m_chunk_16_of_64` | `16` | `262144` | 111.61 | 65.34% | 24.03% | 36.78% | 5.90 | 5.29% | 8.09% | 22.00% | `8` | `8` |
| `seq_1m_chunk_24_of_64` | `24` | `393216` | 137.67 | 71.63% | 33.17% | 46.31% | 9.85 | 7.15% | 9.99% | 21.56% | `13` | `13` |
| `seq_1m_chunk_32_of_64` | `32` | `524288` | 171.49 | 71.39% | 36.75% | 51.47% | 12.88 | 7.51% | 10.52% | 20.44% | `16` | `16` |
| `seq_1m_chunk_40_of_64` | `40` | `655360` | 178.91 | 77.43% | 42.24% | 54.55% | 15.48 | 8.65% | 11.18% | 20.49% | `21` | `21` |
| `seq_1m_chunk_48_of_64` | `48` | `786432` | 213.52 | 80.94% | 49.14% | 60.72% | 20.24 | 9.48% | 11.71% | 19.29% | `25` | `25` |
| `seq_1m_chunk_56_of_64` | `56` | `917504` | 223.13 | 80.69% | 47.66% | 59.06% | 26.96 | 12.08% | 14.97% | 25.35% | `29` | `29` |
| `seq_1m_chunk_64_of_64` | `64` | `1048576` | 253.05 | 82.22% | 51.46% | 62.59% | 30.56 | 12.08% | 14.69% | 23.47% | `32` | `32` |

## Decoder Layer Trend

| Scenario | Chunk idx | Decoder ms | Attention ms | Attention % of layer | Indexer ms | Indexer % of layer | TopK ms | TopK % of layer | TopK calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `seq_1m_chunk_01_of_64` | `1` | 8.95 | 5.87 | 65.54% | 0.67 | 7.50% | 0.40 | 4.46% | `1` |
| `seq_1m_chunk_08_of_64` | `8` | 12.86 | 9.74 | 75.71% | 4.41 | 34.32% | 0.93 | 7.24% | `4` |
| `seq_1m_chunk_16_of_64` | `16` | 17.28 | 14.21 | 82.21% | 8.94 | 51.73% | 1.97 | 11.38% | `8` |
| `seq_1m_chunk_24_of_64` | `24` | 23.76 | 20.62 | 86.77% | 15.22 | 64.07% | 3.28 | 13.82% | `13` |
| `seq_1m_chunk_32_of_64` | `32` | 29.46 | 26.38 | 89.56% | 21.01 | 71.31% | 4.29 | 14.58% | `16` |
| `seq_1m_chunk_40_of_64` | `40` | 33.64 | 30.56 | 90.83% | 25.19 | 74.88% | 5.16 | 15.34% | `21` |
| `seq_1m_chunk_48_of_64` | `48` | 43.41 | 40.30 | 92.83% | 34.98 | 80.57% | 6.75 | 15.54% | `25` |
| `seq_1m_chunk_56_of_64` | `56` | 43.87 | 40.79 | 92.97% | 35.45 | 80.80% | 8.99 | 20.49% | `29` |
| `seq_1m_chunk_64_of_64` | `64` | 51.47 | 48.38 | 94.01% | 43.41 | 84.35% | 10.19 | 19.80% | `32` |

## First vs Last Full 16K

| Seq len | Chunk ratio | Indexer ratio | TopK ratio | Logits ratio | First topk calls | Last topk calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1048576` | 3.38x | 64.60x | 25.50x | 126.48x | `1` | `32` |

## Per-Scenario Sparse-Layer Breakdown

### `seq_1m_chunk_01_of_64` (`chunk_01_of_64`, seq_len=`1048576`)

- Capture occurrence: `65` of `execute_context_1(16384)_generation_0(0)`
- Effective seq len at chunk: `16384`
- Expected `prefill_topk` calls per sparse layer: `1`
- Observed `prefill_topk` calls per sparse layer: `[1]`

| Layer | Decoder ms | Attention % | MLP % | Indexer % | TopK ms | TopK % | Logits ms | Logits % | TopK calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `layer_2` | 8.80 | 65.42% | 34.58% | 7.36% | 0.38 | 4.35% | 0.25 | 2.89% | `1` |
| `layer_4` | 8.88 | 65.25% | 34.75% | 7.43% | 0.39 | 4.44% | 0.26 | 2.88% | `1` |
| `layer_6` | 9.18 | 65.95% | 34.05% | 7.71% | 0.42 | 4.59% | 0.28 | 3.00% | `1` |

### `seq_1m_chunk_08_of_64` (`chunk_08_of_64`, seq_len=`1048576`)

- Capture occurrence: `72` of `execute_context_1(16384)_generation_0(0)`
- Effective seq len at chunk: `131072`
- Expected `prefill_topk` calls per sparse layer: `4`
- Observed `prefill_topk` calls per sparse layer: `[4]`

| Layer | Decoder ms | Attention % | MLP % | Indexer % | TopK ms | TopK % | Logits ms | Logits % | TopK calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `layer_2` | 12.57 | 75.45% | 24.55% | 34.06% | 0.94 | 7.45% | 3.34 | 26.53% | `4` |
| `layer_4` | 12.91 | 75.69% | 24.31% | 34.33% | 0.92 | 7.15% | 3.50 | 27.10% | `4` |
| `layer_6` | 13.10 | 75.99% | 24.01% | 34.57% | 0.93 | 7.11% | 3.58 | 27.37% | `4` |

### `seq_1m_chunk_16_of_64` (`chunk_16_of_64`, seq_len=`1048576`)

- Capture occurrence: `80` of `execute_context_1(16384)_generation_0(0)`
- Effective seq len at chunk: `262144`
- Expected `prefill_topk` calls per sparse layer: `8`
- Observed `prefill_topk` calls per sparse layer: `[8]`

| Layer | Decoder ms | Attention % | MLP % | Indexer % | TopK ms | TopK % | Logits ms | Logits % | TopK calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `layer_2` | 17.19 | 82.26% | 17.74% | 51.90% | 2.00 | 11.63% | 6.91 | 40.21% | `8` |
| `layer_4` | 17.13 | 82.07% | 17.93% | 51.64% | 1.92 | 11.23% | 6.91 | 40.35% | `8` |
| `layer_6` | 17.54 | 82.30% | 17.70% | 51.65% | 1.98 | 11.30% | 7.07 | 40.29% | `8` |

### `seq_1m_chunk_24_of_64` (`chunk_24_of_64`, seq_len=`1048576`)

- Capture occurrence: `88` of `execute_context_1(16384)_generation_0(0)`
- Effective seq len at chunk: `393216`
- Expected `prefill_topk` calls per sparse layer: `13`
- Observed `prefill_topk` calls per sparse layer: `[13]`

| Layer | Decoder ms | Attention % | MLP % | Indexer % | TopK ms | TopK % | Logits ms | Logits % | TopK calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `layer_2` | 23.16 | 86.56% | 13.44% | 63.92% | 3.28 | 14.15% | 11.51 | 49.72% | `13` |
| `layer_4` | 24.03 | 86.81% | 13.19% | 63.86% | 3.26 | 13.58% | 12.07 | 50.23% | `13` |
| `layer_6` | 24.09 | 86.94% | 13.06% | 64.43% | 3.31 | 13.74% | 12.20 | 50.64% | `13` |

### `seq_1m_chunk_32_of_64` (`chunk_32_of_64`, seq_len=`1048576`)

- Capture occurrence: `96` of `execute_context_1(16384)_generation_0(0)`
- Effective seq len at chunk: `524288`
- Expected `prefill_topk` calls per sparse layer: `16`
- Observed `prefill_topk` calls per sparse layer: `[16]`

| Layer | Decoder ms | Attention % | MLP % | Indexer % | TopK ms | TopK % | Logits ms | Logits % | TopK calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `layer_2` | 28.82 | 89.45% | 10.55% | 71.16% | 4.29 | 14.89% | 16.20 | 56.22% | `16` |
| `layer_4` | 29.62 | 89.58% | 10.42% | 71.05% | 4.25 | 14.34% | 16.78 | 56.67% | `16` |
| `layer_6` | 29.94 | 89.64% | 10.36% | 71.71% | 4.35 | 14.52% | 17.11 | 57.15% | `16` |

### `seq_1m_chunk_40_of_64` (`chunk_40_of_64`, seq_len=`1048576`)

- Capture occurrence: `104` of `execute_context_1(16384)_generation_0(0)`
- Effective seq len at chunk: `655360`
- Expected `prefill_topk` calls per sparse layer: `21`
- Observed `prefill_topk` calls per sparse layer: `[21]`

| Layer | Decoder ms | Attention % | MLP % | Indexer % | TopK ms | TopK % | Logits ms | Logits % | TopK calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `layer_2` | 32.87 | 90.71% | 9.29% | 74.59% | 5.14 | 15.63% | 19.37 | 58.91% | `21` |
| `layer_4` | 33.78 | 90.84% | 9.16% | 75.05% | 5.11 | 15.14% | 20.22 | 59.86% | `21` |
| `layer_6` | 34.26 | 90.96% | 9.04% | 75.00% | 5.23 | 15.26% | 20.46 | 59.70% | `21` |

### `seq_1m_chunk_48_of_64` (`chunk_48_of_64`, seq_len=`1048576`)

- Capture occurrence: `112` of `execute_context_1(16384)_generation_0(0)`
- Effective seq len at chunk: `786432`
- Expected `prefill_topk` calls per sparse layer: `25`
- Observed `prefill_topk` calls per sparse layer: `[25]`

| Layer | Decoder ms | Attention % | MLP % | Indexer % | TopK ms | TopK % | Logits ms | Logits % | TopK calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `layer_2` | 42.91 | 92.82% | 7.18% | 80.49% | 6.77 | 15.76% | 27.76 | 64.68% | `25` |
| `layer_4` | 43.45 | 92.83% | 7.17% | 80.47% | 6.69 | 15.39% | 28.26 | 65.04% | `25` |
| `layer_6` | 43.87 | 92.83% | 7.17% | 80.75% | 6.79 | 15.48% | 28.62 | 65.24% | `25` |

### `seq_1m_chunk_56_of_64` (`chunk_56_of_64`, seq_len=`1048576`)

- Capture occurrence: `120` of `execute_context_1(16384)_generation_0(0)`
- Effective seq len at chunk: `917504`
- Expected `prefill_topk` calls per sparse layer: `29`
- Observed `prefill_topk` calls per sparse layer: `[29]`

| Layer | Decoder ms | Attention % | MLP % | Indexer % | TopK ms | TopK % | Logits ms | Logits % | TopK calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `layer_2` | 42.77 | 92.85% | 7.15% | 80.47% | 8.91 | 20.84% | 25.49 | 59.59% | `29` |
| `layer_4` | 44.59 | 93.11% | 6.89% | 81.13% | 9.02 | 20.23% | 27.14 | 60.87% | `29` |
| `layer_6` | 44.24 | 92.96% | 7.04% | 80.80% | 9.02 | 20.40% | 26.71 | 60.36% | `29` |

### `seq_1m_chunk_64_of_64` (`chunk_64_of_64`, seq_len=`1048576`)

- Capture occurrence: `128` of `execute_context_1(16384)_generation_0(0)`
- Effective seq len at chunk: `1048576`
- Expected `prefill_topk` calls per sparse layer: `32`
- Observed `prefill_topk` calls per sparse layer: `[32]`

| Layer | Decoder ms | Attention % | MLP % | Indexer % | TopK ms | TopK % | Logits ms | Logits % | TopK calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `layer_2` | 50.90 | 93.98% | 6.02% | 85.34% | 10.46 | 20.55% | 32.72 | 64.29% | `32` |
| `layer_4` | 51.55 | 94.03% | 5.97% | 83.73% | 9.97 | 19.34% | 33.17 | 64.36% | `32` |
| `layer_6` | 51.95 | 94.02% | 5.98% | 83.96% | 10.13 | 19.49% | 33.47 | 64.43% | `32` |
