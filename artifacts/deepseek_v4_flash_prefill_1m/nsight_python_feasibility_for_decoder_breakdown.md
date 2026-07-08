# Nsight Python Feasibility For Decoder-Layer Breakdown

## Bottom Line

We can leverage `/home/yiliu7/workspace/nsight-python-fork`, but only for a
subset of the problem.

For our actual goal:

- decoder-layer breakdown
- nested proportions like `topk -> indexer -> attention -> decoder layer`
- multiple scenarios such as first chunk / last chunk / 128k / 1M / different
  layers
- real device-time ratios, not host time

`nsys + NVTX` should remain the primary source of truth.

`nsight-python` is better treated as a secondary tool for focused
kernel-metric analysis of specific leaf regions, not as the main breakdown tool.

## What The Fork Actually Does

The fork is built around **Nsight Compute** (`ncu`), not Nsight Systems.

Evidence:

- [ncu.py](/home/yiliu7/workspace/nsight-python-fork/nsight/collection/ncu.py:5)
- [ncu.py](/home/yiliu7/workspace/nsight-python-fork/nsight/collection/ncu.py:30)
- [runtime_requirements.rst](/home/yiliu7/workspace/nsight-python-fork/docs/source/installation/runtime_requirements.rst:7)

That means it is designed to collect:

- per-kernel metrics
- replay-based measurements
- annotation-scoped kernel summaries
- CSV / pandas / plot outputs

This is useful, but it is a different problem than the one `nsys` solves well.

## Why It Is Not A Drop-In Fit For Our Current Breakdown Goal

### 1. It is `ncu`, not `nsys`

Our current decoder-layer breakdown depends on:

- nested NVTX ranges
- projected GPU time across a range
- inclusive relationships between decoder layer, attention, indexer, topk

That is what `nsys stats --report nvtx_gpu_proj_sum` is giving us today.

`nsight-python` instead drives `ncu` and extracts per-kernel metrics from
`.ncu-rep` reports. That is different from hierarchical timeline attribution.

### 2. Nested annotations are explicitly unsupported

This is a direct mismatch with the current vLLM instrumentation, which uses
deep nesting like:

- `attention.forward`
- `attention.attention_impl`
- `attention.indexer`
- `indexer.forward`
- `sparse_attn_indexer`
- `prefill_topk`

The fork explicitly rejects nested annotations:

- [annotation.py](/home/yiliu7/workspace/nsight-python-fork/nsight/annotation.py:117)
- [annotation.py](/home/yiliu7/workspace/nsight-python-fork/nsight/annotation.py:151)

That alone makes it a poor match for decoder-layer umbrella breakdowns.

### 3. Duplicate annotation names in one run are explicitly unsupported

The fork also requires annotation names to be unique within one profiling run:

- [annotation.py](/home/yiliu7/workspace/nsight-python-fork/nsight/annotation.py:109)
- [annotation.py](/home/yiliu7/workspace/nsight-python-fork/nsight/annotation.py:158)

Our workload intentionally repeats the same annotation many times:

- `dsv4.layer_2.sparse_attn_indexer.prefill_topk` is called `31` times
- same for `layer_4` and `layer_6`

So the current profiling shape is directly incompatible with the fork's current
annotation model.

### 4. The fork assumes one kernel per annotation by default

Its extraction logic expects one kernel per annotation unless we manually
combine them or use range replay:

- [extraction.py](/home/yiliu7/workspace/nsight-python-fork/nsight/extraction.py:190)

That is workable for leaf kernels, but not ideal for high-level decoder-layer
ranges that intentionally include many kernels and stream overlap.

### 5. Import ordering is strict and conflicts with the current CLI path

The fork requires `import nsight` before:

- CUDA initialization
- any NVTX calls

See [runtime_requirements.rst](/home/yiliu7/workspace/nsight-python-fork/docs/source/installation/runtime_requirements.rst:12).

In practice, that means a dedicated wrapper entrypoint is needed before the
normal vLLM / torch import chain. It is not a drop-in toggle for the existing
benchmark command.

## Immediate Local Environment Blockers

Even before the semantic mismatch, the current environment is not ready for the
fork as-is.

### 1. `ncu` is not on `PATH`

The fork resolves `ncu` via `shutil.which("ncu")`:

- [ncu.py](/home/yiliu7/workspace/nsight-python-fork/nsight/collection/ncu.py:133)

But in this shell:

- `ncu --version` fails from `PATH`
- available binaries exist only by full path, e.g.
  `/opt/nvidia/nsight-compute/2026.1.1/ncu`

### 2. Available `ncu` is too old for this fork

The fork requires:

- minimum `ncu` version `2026.2.1.0`

See [ncu.py](/home/yiliu7/workspace/nsight-python-fork/nsight/collection/ncu.py:30)
and [runtime_requirements.rst](/home/yiliu7/workspace/nsight-python-fork/docs/source/installation/runtime_requirements.rst:7).

But the installed version here is:

- `/opt/nvidia/nsight-compute/2026.1.1/ncu`
- version `2026.1.1.0`

So the fork would reject the current local `ncu` even if we fixed `PATH`.

### 3. Python dependencies are missing in `.venv`

The fork currently does not import cleanly in the vLLM environment because:

- `nvtx` is missing
- `ncu_report` is missing

So it is not ready to run in this workspace without extra setup.

## Where It Can Still Help

The fork is still useful for a narrower problem:

- focused kernel analysis
- nice CSV / pandas outputs
- easy parameter sweeps
- derived metrics
- reproducible comparisons across small scenario matrices

Good candidates:

- `prefill_topk`
- `fp8_fp4_mqa_logits`
- possibly `flash_mla_sparse_fwd`

In other words:

- use `nsys` for the **hierarchical layer breakdown**
- use `nsight-python` for **leaf-kernel metric drill-down**

## Recommended Profiling Split

### Keep `nsys` for decoder-layer breakdown

Use the current aligned NVTX + `nsys` method for:

- attention share of decoder layer
- indexer share of attention
- topk share of indexer
- call counts
- first vs last chunk
- different sequence lengths / scenarios

This remains the best tool because it preserves:

- nesting
- repeated ranges
- device-time attribution
- real inference capture

### Use `nsight-python` only for a leaf-kernel workflow

If we want to leverage the fork, the most sensible workflow is:

1. Create a dedicated wrapper entrypoint that imports `nsight` before `torch`.
2. Profile exactly one leaf target per run.
3. Avoid nested annotations inside the profiled region.
4. Use scenario configs as the `configs=` sweep axis.
5. Export CSV / plots from the fork.

This is realistic for comparing scenarios like:

- last `16k` chunk vs first `16k` chunk
- `32k`, `128k`, `1M`
- `layer_2` vs `layer_41` vs `layer_42`

But it should be limited to a leaf target such as:

- `prefill_topk`
- `fp8_fp4_mqa_logits`

not the whole decoder-layer hierarchy.

## If We Want To Use It For Decoder-Layer Breakdown Anyway

That would require real fork work, not just usage changes.

At minimum:

1. Allow nested annotations.
2. Allow repeated annotation names in one run.
3. Preserve full NVTX stack or parent-child relationships in extraction.
4. Define how multi-kernel, multi-stream inclusive time should be computed.
5. Probably add an `nsys` backend or post-processing path, because `ncu`
   alone is the wrong primitive for inclusive hierarchical time attribution.

This is substantial engineering work.

## Recommendation

For our stated goal, the best path is:

1. Keep the current `nsys + NVTX` capture as the authoritative decoder-layer
   breakdown path.
2. Build a small reusable postprocessor around `nsys` outputs to generate:
   - per-scenario markdown
   - per-scenario CSV
   - compact comparison tables across scenarios
3. Optionally add a separate `nsight-python` microprofile wrapper for leaf
   kernels only.

So the answer is:

- yes, we can leverage the fork
- no, it should not replace the current `nsys` breakdown workflow
- its best role is leaf-kernel metric analysis, while `nsys` remains the right
  tool for decoder-layer breakdown across scenarios
