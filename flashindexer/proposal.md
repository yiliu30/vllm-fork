# Proposal: Fused QK + Top-K Selector for Sparse Attention

## 1. Motivation

Long-context LLM inference increasingly relies on sparse attention to reduce the cost of attending over very long KV caches. A common sparse-attention pipeline is:

```text
Q, K
  -> QK / indexer score computation
  -> materialize dense or semi-dense scores to HBM
  -> top-k / top-p selection
  -> sparse attention over selected KV tokens or pages
```

Even if the standalone top-k kernel is highly optimized, the full pipeline can still be bottlenecked by:

```text
1. QK score materialization to HBM
2. top-k reading the score buffer back from HBM
3. query chunking caused by large intermediate score buffers
4. irregular candidate / token gather before sparse attention
```

For example:

```text
num_queries = 1024
num_keys    = 131072
dtype       = fp16

dense QK score buffer:
  1024 x 131072 x 2 bytes ≈ 256 MB per head

dense QK pipeline:
  ~256 MB HBM write
  ~256 MB HBM read
```

The key insight is that the selector does not need to output the dense score matrix. It only needs to output selected tokens/pages or compact candidates.

Therefore, the selector should become:

```text
score-producing pipeline
  -> candidate-producing pipeline
```

---

## 2. Core Proposal

We propose a **fused QK + top-k selector** for sparse attention.

Instead of:

```text
Kernel 1:
  QK -> dense score buffer in HBM

Kernel 2:
  top-k(score buffer) -> selected indices
```

we propose:

```text
Fused selector kernel:
  QK tile
    -> page/block score reduction
    -> local top-k/top-P update
    -> compact candidate output
```

The dense QK scores are consumed on-chip and are never materialized to HBM.

### High-level dataflow

```text
Q tile + K page/block tile
      |
      v
temporary QK scores inside registers/shared memory
      |
      v
page/block score reduction
      |
      v
local top-P candidate update
      |
      v
write compact candidates or final page IDs
      |
      v
sparse attention over selected pages/tokens
```

---

## 3. Target Use Cases

This proposal targets sparse-attention systems where the expensive part is the selector/indexer path.

Relevant scenarios:

```text
1. Long-context prefill
2. DSA-style learned sparse indexer
3. HISA-style hierarchical block/token indexer
4. MISA-style routed indexer heads
5. Page/block sparse attention over paged KV cache
6. B200 / Blackwell low-bit hardware acceleration
7. vLLM / FlashInfer-style sparse attention kernels
```

This is not intended to replace HISA, MISA, or IndexCache. It is complementary:

```text
HISA:
  reduce number of tokens scored

MISA:
  reduce number of indexer heads scored

IndexCache:
  reduce number of layers that run the indexer

This proposal:
  reduce score movement and HBM materialization
```

---

## 4. Key Research/Engineering Claims

### Claim 1: HBM score traffic is a hidden bottleneck

Sparse attention reduces the final attention compute, but the selector may still produce large intermediate score buffers.

We should show:

```text
dense QK / indexer score write + read
is a major part of selector latency and memory traffic.
```

### Claim 2: Streaming score-to-candidate dataflow removes the dense score buffer

The selector should consume QK/indexer scores on-chip and write only compact candidates.

### Claim 3: Split-local top-P + global merge is exact for page/block scores

For page/block-level selection, if each KV split outputs local top-P candidates, then merging all local candidates gives the exact global top-P under the chosen page/block score.

Reason:

```text
If a page is not in local top-P of its split,
then at least P pages in the same split have higher scores.
Therefore it cannot be in the global top-P.
```

### Claim 4: Low-bit coarse scoring + high-precision refinement can improve speed while preserving recall

Use:

```text
FP4 / MXFP4 / NVFP4 coarse scoring
  -> over-select candidate pages/blocks

BF16 / FP16 refinement
  -> final top-P selection
```

This gives a tunable accuracy/performance tradeoff.

---

## 5. Proposed Execution Pipeline

We propose the following staged pipeline.

```text
Stage 0: Distribution analysis
Stage 1: Baseline measurement
Stage 2: Page/block selector prototype
Stage 3: Split-local top-P + merge
Stage 4: Fused QK + page-score + local top-P kernel
Stage 5: Accuracy validation
Stage 6: Performance optimization
Stage 7: Low-bit / B200 enhancement
Stage 8: Integration with sparse attention
```

---

# Stage 0: Observe Token Distribution Pattern

Before building the fused kernel, we need verify whether page/block-level selection is a good approximation for the target model/workload.

## 0.1 What to observe

For each model, layer, head, query position, and sample:

```text
scores = Q @ K^T
dense_topT_tokens = topT(scores)
page_id = token_id // page_size
```

Measure:

```text
1. unique_pages@T
2. token_recall@T vs selected pages P
3. attention_mass_recall vs selected pages P
4. layer-wise distribution
5. head-wise distribution
6. position-wise distribution
7. task/prompt-type distribution
```

## 0.2 Key metrics

### unique_pages@T

```text
Number of unique KV pages containing dense top-T tokens.
```

Example:

```text
top-64 tokens in 8 pages  -> page selection is promising
top-64 tokens in 48 pages -> small-P page selection may lose recall
```

### token_recall@T

```text
token_recall@T =
  | selected_page_tokens ∩ dense_topT_tokens | / T
```

### attention_mass_recall

```text
mass_recall =
  softmax_mass(selected_page_tokens) / softmax_mass(all_tokens)
```

This is more important than token recall for final model quality.

## 0.3 Page score variants to compare

Evaluate several page/block score definitions:

```text
1. max:
   page_score = max(QK scores inside page)

2. top-r sum:
   page_score = sum(top-r QK scores inside page)

3. logsumexp:
   page_score = log(sum(exp(QK scores inside page)))

4. mean + max:
   page_score = alpha * max + beta * mean

5. proxy:
   page_score from pooled K or metadata
```

Expected behavior:

| Page score | Strength                            | Weakness                      |
| ---------- | ----------------------------------- | ----------------------------- |
| max        | Good for top-token recall           | May miss broad attention mass |
| top-r sum  | Better for multiple relevant tokens | More expensive                |
| logsumexp  | Best proxy for softmax mass         | More expensive                |
| mean + max | Balanced                            | Requires tuning               |
| proxy      | Cheap                               | Less accurate                 |

## 0.4 Go / no-go signal

Proceed if:

```text
1. Important tokens are concentrated in a limited number of pages.
2. token_recall@T and mass_recall are high for practical P.
3. Page-score selection is stable across layers/heads or can be adapted.
```

Pivot if:

```text
1. Top tokens are scattered across many pages.
2. mass_recall remains low even with large P.
3. Page-level selection causes unacceptable downstream quality loss.
```

---

# Stage 1: Establish Baselines

Before implementing the fused kernel, build reliable baselines.

## 1.1 Baseline A: Dense QK + torch.topk / FlashInfer topk

```text
QK kernel:
  scores = Q @ K^T
  write scores to HBM

Top-k kernel:
  read scores from HBM
  output selected indices
```

Measure:

```text
1. QK latency
2. top-k latency
3. total selector latency
4. intermediate score buffer size
5. HBM read/write bytes
6. end-to-end sparse attention latency
```

## 1.2 Baseline B: Existing sparse indexer

If working with DSA/HISA/MISA-like systems:

```text
1. Original DSA indexer
2. HISA hierarchical indexer
3. MISA routed-head indexer
4. IndexCache layer reuse if available
```

## 1.3 Baseline C: Page-score materialization

Intermediate prototype:

```text
QK -> page_score matrix [Q, num_pages]
top-P over page_score
sparse attention
```

This materializes page scores, not token scores. It helps separate:

```text
benefit from page reduction
vs
benefit from full fusion
```

---

# Stage 2: Page/Block Selector Algorithm

## 2.1 Page-level selection

Define:

```text
page_size = 64 / 128 / 256
num_pages = num_keys / page_size
P = number of selected pages
```

For each query `q` and page `p`:

```text
page_score(q, p) = reduce_{t in page p}(q · k_t)
```

Then select:

```text
top-P pages by page_score
```

## 2.2 Attend to selected pages

The initial version should use:

```text
select pages
then attend to all tokens inside selected pages
```

This avoids irregular token-level gather and keeps memory access page-aligned.

Example:

```text
page_size = 128
P = 16

selected tokens = 2048 per query
```

## 2.3 Optional token-level refinement

Later, if selected pages still contain too many tokens:

```text
select top-P pages
then select top-T tokens inside selected pages
then sparse attention over selected tokens
```

This is more aggressive but has higher implementation complexity and greater quality risk.

---

# Stage 3: Split-Local Top-P + Merge

For parallelism, split the KV pages:

```text
KV split 0: pages [0, S)
KV split 1: pages [S, 2S)
...
```

Each split produces local top-P candidates:

```text
local_candidates shape:
  [num_queries, num_splits, P]
```

Then merge:

```text
final_pages = topP(local_candidates, P)
```

## 3.1 Exactness property

For page/block score selection, this is exact.

If page `x` is not in local top-P of its split, then at least `P` pages in that split have higher score than `x`. Therefore, `x` cannot be in the global top-P.

## 3.2 Candidate format

Each candidate stores:

```text
score: fp16/fp32
page_id: int32
optional metadata:
  split_id
  local_rank
  tie_break index
```

Candidate memory size:

```text
num_queries * num_splits * P * sizeof(candidate)
```

Example:

```text
num_queries = 1024
num_splits = 8
P = 16
candidate = score fp32 + page_id int32 = 8 bytes

candidate buffer ≈ 1024 * 8 * 16 * 8 = 1 MB
```

---

# Stage 4: Fused QK + Page-Score + Local Top-P Kernel

## 4.1 Kernel goal

Replace:

```text
QK score kernel + top-k kernel
```

with:

```text
fused_qk_page_topk_kernel
```

The kernel should:

```text
1. load Q tile
2. stream K pages
3. compute temporary QK tile
4. reduce token scores to page_score
5. update local top-P state
6. write local candidates
```

## 4.2 Conceptual pseudocode

```cpp
for q_tile in query_tiles:
    for kv_split in kv_splits:

        local_topP = empty_topP(q_tile, P)

        for page in kv_split.pages:
            K_page = load_K_page(page)

            scores_tile = Q_tile @ K_page.T
            page_score = reduce(scores_tile, over_tokens_in_page)

            local_topP.update(page_score, page_id)

        write local_topP
```

Important:

```text
scores_tile is temporary.
It should live in registers/shared memory.
It should not be written to HBM.
```

## 4.3 Local top-P structure

For small P:

```text
P = 8 / 16 / 32
```

Use simple unsorted array:

```text
top_scores[P]
top_page_ids[P]
current_min_score
current_min_pos
```

When a new page score arrives:

```text
if page_score > current_min_score:
    replace min entry
    recompute min
```

For larger P, consider:

```text
1. small heap
2. warp-level bitonic selection
3. radix-select over local candidates
4. block-level top-k primitive
```

## 4.4 Page score reduction options

### max

```text
page_score = max(scores_tile)
```

Cheapest and good for top-token recall.

### top-r sum

```text
page_score = sum(top-r scores inside page)
```

Better for pages with multiple useful tokens.

### logsumexp

```text
page_score = log(sum(exp(scores_tile)))
```

Best proxy for attention mass, but more expensive.

Start with `max`, then compare against `top-r sum` and `logsumexp`.

---

# Stage 5: Accuracy Validation

## 5.1 Selection fidelity

Compare selected pages/tokens against dense reference.

Metrics:

```text
1. page_recall@P
2. token_recall@T
3. attention_mass_recall
4. selected-token IoU vs DSA/HISA
5. final sparse attention output difference
```

### selected-token IoU

```text
IoU =
  | selected_by_fused ∩ selected_by_reference |
  / | selected_by_fused ∪ selected_by_reference |
```

### attention output error

Compare:

```text
dense_attention_output
vs
sparse_attention_output
```

Metrics:

```text
1. L2 error
2. cosine similarity
3. max absolute error
4. downstream task quality
```

## 5.2 End-task evaluation

Use long-context tasks:

```text
1. Needle-in-a-Haystack
2. LongBench
3. retrieval-heavy QA
4. code / symbol retrieval
5. multi-hop long-context reasoning
```

## 5.3 Layer/head analysis

Report results by:

```text
1. layer
2. attention head
3. query position
4. context length
5. prompt/task type
```

This helps identify where page-level selection works or fails.

---

# Stage 6: Performance Validation

## 6.1 Microbenchmarks

Measure:

```text
1. selector latency
2. QK page scoring latency
3. local top-P update latency
4. merge latency
5. sparse attention latency
6. total end-to-end latency
```

Test shapes:

```text
context length: 32K, 64K, 128K, 1M
query chunk: 1, 16, 64, 256, 1024
page size: 64, 128, 256
P: 8, 16, 32, 64
num_splits: 4, 8, 16
head_dim: 64, 128, 192
```

## 6.2 HBM traffic metrics

Measure or estimate:

```text
1. dense QK score write bytes
2. dense QK score read bytes
3. page-score materialization bytes
4. local candidate write/read bytes
5. final page-id write bytes
6. selected K/V read bytes
```

Expected result:

```text
dense QK score write/read should disappear.
candidate traffic should be much smaller than dense score traffic.
```

## 6.3 Roofline-style analysis

Classify each stage:

```text
1. compute-bound
2. memory-bandwidth-bound
3. latency-bound due to irregular gather
4. synchronization-bound
```

Useful counters:

```text
1. achieved memory bandwidth
2. tensor-core utilization
3. SM occupancy
4. shared-memory usage
5. register pressure
6. warp stalls
7. L2 hit rate
```

---

# Stage 7: Low-Bit and B200 Optimization

## 7.1 Motivation

Low-bit instructions such as FP4 / MXFP4 / NVFP4 are orthogonal to the fused-selector dataflow.

The dataflow reduces HBM traffic:

```text
avoid dense score materialization
```

Low-bit hardware reduces scoring cost:

```text
cheaper QK/indexer dot products
```

Together:

```text
less score traffic + cheaper scoring compute
```

## 7.2 FP4 coarse + BF16 refine design

Recommended design:

```text
Stage A:
  FP4 QK/page scoring over all pages
  select top-M candidate pages

Stage B:
  BF16/FP16 recompute/refine only candidate pages
  select final top-P pages

Stage C:
  sparse attention over final P pages
```

Where:

```text
M > P
```

Example:

```text
P = 16 final pages
M = 32 or 64 coarse candidate pages
```

This controls accuracy:

```text
larger M -> higher recall, more refine compute
smaller M -> faster, more recall risk
```

## 7.3 What to validate

Measure:

```text
1. FP4-only selection recall
2. FP4 coarse + BF16 refine recall
3. top-P boundary stability
4. over-selection budget required for >99% IoU
5. task quality under low-bit selector
```

## 7.4 B200 / SM90+ cluster direction

Potential mapping:

```text
one thread-block cluster handles one query tile / KV split group
```

Inside cluster:

```text
CTA 0 computes local page candidates
CTA 1 computes local page candidates
CTA 2 computes local page candidates
...

cluster DSM merge
  -> split-local top-P or final top-P
```

Potential benefits:

```text
1. reduce candidate writes to HBM
2. merge CTA-local candidates on-chip
3. improve parallelism for long rows / low batch
4. exploit cluster shared memory and distributed shared memory
```

---

# Stage 8: Integration with Sparse Attention

## 8.1 Output format

The fused selector should output:

```text
selected_page_ids:
  [batch, heads, queries, P]

optional selected_page_scores:
  [batch, heads, queries, P]
```

For DSA/HISA token-level integration, output may be:

```text
selected_token_ids:
  [batch, heads, queries, K]
```

But page IDs are preferred first because they are more memory-coalesced.

## 8.2 Sparse attention kernel input

Sparse attention consumes:

```text
Q
selected_page_ids
K cache
V cache
```

Then computes:

```text
softmax(Q @ K_selected^T) @ V_selected
```

## 8.3 Page-aware ordering

Selected pages should be sorted or grouped to improve gather locality:

```text
1. sort selected pages by page_id
2. preserve score order separately if needed
3. group by physical KV cache layout
4. avoid random scattered K/V reads
```

This may be important for end-to-end performance.

---

# 9. Implementation Plan

## Phase 1: Offline analysis tools

Deliverables:

```text
1. dense QK top-token distribution analyzer
2. page coverage analyzer
3. attention mass recall analyzer
4. layer/head heatmaps
5. page_score comparison script
```

Outputs:

```text
unique_pages@T
token_recall@T vs P
mass_recall vs P
best page_size
best page_score candidate
```

## Phase 2: Reference Python/PyTorch prototype

Implement:

```text
1. dense QK baseline
2. page_score materialization baseline
3. split-local top-P + merge
4. sparse attention over selected pages
```

Purpose:

```text
validate accuracy before custom kernel work.
```

## Phase 3: First CUDA/Triton fused selector

Implement basic fused selector:

```text
BF16/FP16 QK
page_score = max
local top-P
write local candidates
separate merge kernel
```

Avoid complexity first:

```text
no FP4
no cluster DSM
no token refinement
no logsumexp
```

## Phase 4: Merge kernel optimization

Implement:

```text
candidate merge:
  [Q, num_splits, P] -> [Q, P]
```

Try:

```text
1. one warp per query
2. one CTA per query
3. vectorized candidate loads
4. small top-P array in registers/shared memory
```

## Phase 5: Fused selector + sparse attention pipeline

Connect:

```text
fused selector
  -> selected page ids
  -> existing sparse attention kernel
```

Measure end-to-end:

```text
selector + sparse attention
```

Compare with:

```text
dense QK + top-k + sparse attention
```

## Phase 6: Low-bit selector

Add:

```text
1. FP4/MXFP4/NVFP4 K cache path
2. online Q quantization path
3. FP4 coarse page scoring
4. BF16/FP16 candidate refinement
```

Measure:

```text
selection IoU
mass recall
latency
tensor-core utilization
```

## Phase 7: B200 cluster optimization

Add:

```text
1. thread-block cluster version
2. DSM-based candidate merge
3. larger on-chip candidate buffer
4. cluster-level top-P reduction
```

Compare:

```text
non-cluster fused selector
vs
cluster fused selector
```

## Phase 8: Production integration

Integrate with:

```text
1. paged KV cache
2. FlashInfer-style top-k/sparse attention API
3. vLLM/vLLM-Omni serving path
4. benchmarking harness
```

---

# 10. Proposed API

## Selector API

```python
selected_pages, selected_scores = fused_qk_page_topk(
    q,                    # [B, H, Q, D]
    k_cache,              # paged KV cache
    page_table,           # logical -> physical page mapping
    page_size=128,
    top_p=16,
    num_splits=8,
    score_mode="max",     # max / top_r_sum / logsumexp
    dtype="bf16",         # bf16 / fp16 / fp4_coarse
    refine_dtype=None,    # bf16 / fp16 optional
    coarse_top_m=None,    # M > P for FP4 coarse path
)
```

## Sparse attention API

```python
out = sparse_attention_with_pages(
    q,
    k_cache,
    v_cache,
    selected_pages,
    page_table,
)
```

---

# 11. Evaluation Matrix

## Baselines

```text
1. Dense attention
2. Dense QK + torch.topk
3. Dense QK + FlashInfer topk
4. Page-score materialization + top-P
5. Existing DSA indexer
6. HISA
7. MISA
8. IndexCache if available
9. Our fused BF16 selector
10. Our FP4 coarse + BF16 refine selector
```

## Accuracy metrics

```text
1. page_recall@P
2. token_recall@T
3. selected-token IoU vs reference
4. attention_mass_recall
5. attention output error
6. downstream benchmark accuracy
```

## Performance metrics

```text
1. selector latency
2. merge latency
3. sparse attention latency
4. end-to-end latency
5. HBM bytes
6. intermediate buffer size
7. tensor-core utilization
8. achieved bandwidth
9. speedup vs context length
```

---

# 12. Risks and Mitigations

## Risk 1: Important tokens are not page-concentrated

Symptom:

```text
unique_pages@T is large
token_recall@T is low for practical P
mass_recall is low
```

Mitigation:

```text
1. reduce page_size
2. increase P
3. use top-r sum or logsumexp page_score
4. add token-level refinement
5. fallback to HISA/DSA for difficult heads/layers
```

## Risk 2: FP4 selector changes ranking too much

Symptom:

```text
FP4 coarse selection misses important pages near top-P boundary
```

Mitigation:

```text
1. over-select M > P
2. BF16/FP16 refine
3. keep scores/top-k comparison in FP32
4. use per-page or per-group scaling carefully
```

## Risk 3: Kernel microbenchmark improves but end-to-end does not

Symptom:

```text
selector is faster,
but sparse attention gather or merge dominates.
```

Mitigation:

```text
1. page-aware output ordering
2. fuse merge with sparse attention
3. reduce candidate writes via cluster DSM
4. optimize K/V gather layout
5. evaluate full pipeline early
```

## Risk 4: HISA/MISA already remove most indexer cost

Symptom:

```text
our fused selector gives small extra speedup after HISA/MISA
```

Mitigation:

```text
1. target longer contexts: 128K / 1M
2. target low-batch / long-row cases
3. focus on FP4 coarse scoring
4. combine with IndexCache full-indexer layers
5. quantify HBM bytes, not only latency
```

---

# 13. Success Criteria

## Minimum success

```text
1. Page/token distribution study completed.
2. Fused BF16 page selector matches page-score reference exactly.
3. Removes dense QK score buffer.
4. Shows lower HBM traffic than dense QK + top-k.
```

## Strong success

```text
1. >99% selected-token IoU vs target selector for practical settings.
2. High attention_mass_recall, e.g. >95% or task-dependent threshold.
3. Selector latency improves over dense QK + top-k.
4. End-to-end sparse attention latency improves meaningfully.
5. Benefit grows with context length.
```

## Research-grade success

```text
1. Composes with HISA/MISA/IndexCache-style systems.
2. FP4 coarse + BF16 refine achieves strong speed/recall tradeoff.
3. B200 cluster/DSM implementation reduces candidate HBM traffic.
4. Demonstrates improvement on long-context benchmarks.
5. Provides reusable sparse-indexer primitive and clear cost model.
```

---

# 14. Recommended First Milestone

The first milestone should not be a fused CUDA kernel.

The first milestone should be:

```text
observe token/page distribution and validate page-level recall.
```

Deliverable:

```text
A report showing:
  1. unique_pages@T
  2. token_recall@T vs P
  3. attention_mass_recall vs P
  4. max vs top-r sum vs logsumexp page_score
  5. layer/head/query-position breakdown
```

Decision:

```text
If page-level recall is promising:
  implement fused QK + page-score + local top-P.

If page-level recall is weak:
  move to HISA-style block + token refinement,
  or FP4 coarse candidate generation + BF16 token rerank.
```

---

# 15. Final Summary

The proposal is:

```text
Fused QK + Top-K Selector for Sparse Attention
```

The main idea is:

```text
Do not materialize dense QK/indexer scores.
Consume scores on-chip and output compact candidates.
```

The execution plan is:

```text
1. Observe token/page distribution.
2. Validate page-score recall.
3. Build reference page selector.
4. Implement split-local top-P + merge.
5. Implement fused QK + page-score + local top-P kernel.
6. Connect to sparse attention.
7. Add FP4 coarse scoring + BF16 refinement.
8. Add B200 cluster/DSM optimization.
9. Validate accuracy and end-to-end performance.
```

The research positioning is:

```text
HISA reduces how many tokens are scored.
MISA reduces how many indexer heads are scored.
IndexCache reduces how often indexers run.

This proposal reduces how many score bytes are moved.
```

The final target is a reusable sparse-indexer primitive:

```text
QK/indexer score generation
  -> on-chip reduction
  -> local top-k
  -> compact candidates
  -> sparse attention
```
