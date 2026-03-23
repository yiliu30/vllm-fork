# INC w4a16 XPU Support for vLLM

## Goal

Add w4a16 XPU support for INC quantization in `inc.py`, routing
AutoRound/INC models to the existing `int4_gemm_w4a16` XPU kernel from
vllm-xpu-kernels.

## Context

The INC w4a16 XPU support was removed when IPEX was deprecated (commit
`e10604480`, PR #33379). The `apply_ipex_quant_layer` method was replaced
with `raise NotImplementedError`. The XPU `int4_gemm_w4a16` kernel
already works — it's used by `CompressedTensorsWNA16` via
`XPUwNa16LinearKernel`. INC just needs to be wired up to call the same
kernel, with proper weight format conversion.

Reference PR for kernel integration:
https://github.com/vllm-project/vllm/pull/33973

## Source

```
vllm/model_executor/layers/quantization/inc.py
```

## Test model

```
Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound
  quant_method:    auto-round  (overridden to "inc" → INCConfig)
  packing_format:  auto_round:auto_awq
  bits:            4
  sym:             true
  group_size:      128
```

## E2E test command

```bash
cd /home/yiliu7/workspace/vllm
source .venv/bin/activate
python3 examples/basic/offline_inference/generate.py \
  --model Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound \
  --block-size 64 --enforce-eager --max-model-len 4096
```

## Implementation

Added `INCXPULinearMethod` class in `inc.py` that:

1. **`create_weights()`** — Allocates `qweight`, `scales`, `qzeros` using
   `PackedvLLMParameter` and `GroupQuantScaleParameter` (same AWQ layout
   as the checkpoint).

2. **`process_weights_after_loading()`** — Repacks AWQ-format weights into
   CompressedTensors (CT) format expected by the XPU kernel:
   - Unpack AWQ int32 → 8 nibbles per column
   - **Reverse the AWQ interleaved nibble order** (the critical step)
   - Transpose `[in, out]` → `[out, in]`
   - Repack into CT `[out, in_packed]` with sequential nibble order

3. **`apply()`** — Calls `torch.ops._xpu_C.int4_gemm_w4a16` with bias
   applied separately (kernel doesn't support fused QKV bias).

Updated `apply_ipex_quant_layer()` to return `INCXPULinearMethod` for
XPU + 4-bit instead of raising `NotImplementedError`.

---

# Debug Report: AWQ Interleaved Nibble Order

## Problem

After implementing `INCXPULinearMethod`, the model produced garbage
output — nonsensical tokens with no coherence. Isolated kernel tests
passed with cosine similarity = 1.0, yet end-to-end inference was broken.

## Root Cause

**AWQ packs 8 × 4-bit nibbles into each int32 in an interleaved order,
not sequentially.**

When unpacking an int32 by shifting `>> 0, >> 4, >> 8, ... >> 28`, the
resulting 8 nibbles correspond to output columns in order
`[0, 4, 1, 5, 2, 6, 3, 7]`, NOT `[0, 1, 2, 3, 4, 5, 6, 7]`.

```
int32 bit layout (AWQ packing):

  bits [0:3]   → output column 0
  bits [4:7]   → output column 4  ← NOT column 1!
  bits [8:11]  → output column 1
  bits [12:15] → output column 5
  bits [16:19] → output column 2
  bits [20:23] → output column 6
  bits [24:27] → output column 3
  bits [28:31] → output column 7
```

Without reversing this interleaving, every group of 8 output columns in
the dequantized weight matrix is internally scrambled. The individual
4-bit values are correct, but they are assigned to the wrong output
neurons.

### The fix

Apply `AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]` after unpacking:

```python
# Unpack columnwise: each int32 → 8 nibbles
shifts = torch.arange(0, 32, bits, device=device)
unpacked = torch.bitwise_right_shift(
    qweight[:, :, None], shifts[None, None, :]
).to(torch.int32).view(in_size, -1)

# Reverse AWQ interleaved order → sequential
reverse_order = torch.arange(out_size, dtype=torch.int32, device=device)
reverse_order = reverse_order.view(-1, pack_factor)
reverse_order = reverse_order[:, AWQ_REVERSE_ORDER]  # ← THE FIX
reverse_order = reverse_order.view(-1)
unpacked = unpacked[:, reverse_order] & mask
```

### Reference implementation

Found in `auto_round` library:

```
.venv/lib/python3.12/site-packages/auto_round/export/export_to_awq/utils.py
```

```python
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

def dequantize_gemm(qweight, qzeros, scales, bits, group_size):
    iweight, izeros = unpack_awq(qweight, qzeros, bits)
    iweight, izeros = reverse_awq_order(iweight, izeros, bits)  # ← critical
    iweight = torch.bitwise_and(iweight, (2**bits) - 1)
    ...
```

This constant originates from MIT HAN Lab's original AWQ CUDA kernel
implementation. It is inherited by AutoRound's AWQ export format.

## Debug Timeline

### Phase 1: Wrong hypotheses (~2 hours wasted)

| Hypothesis | Investigation | Result |
|---|---|---|
| Weight loading/fusion corrupts packed int32 | `dump_weights.py` + `compare_dumped.py`: dumped pre-repack tensors from vLLM, compared byte-for-byte with raw checkpoint | Perfect match — dead end |
| XPU kernel is broken | `debug_minimal.py`: loaded one layer, repacked, ran kernel vs CPU dequant | Cosine = 1.0 — dead end |
| V1 subprocess ignores code changes | Tried monkey-patching from test scripts; discovered V1 runs model in a separate process | Real issue for debugging, but not root cause |
| Numerical overflow | Added logging in `apply()` — saw `inf/-inf` in warmup (batch=8192), zeros in `o_proj` | Warmup overflow is normal; a red herring |
| Scale transposition wrong | Compared with `XPUwNa16LinearKernel` which transposes scales | AWQ scales already in `[ngroups, out]` — no transpose needed |
| Kernel can't handle fused bias | Passed bias directly to kernel | Fixed by passing `None` + manual add; not root cause |

### Phase 2: Breakthrough (~30 min)

**The key insight: build a standalone inference script with NO vLLM.**

`standalone_inference.py` implements a complete Qwen2 forward pass using
only `torch + safetensors + transformers tokenizer`:
- Load raw checkpoint tensors
- Dequantize AWQ weights manually (our sequential-assumption code)
- Run embedding → layernorm → attention → MLP → lm_head
- All on CPU, no XPU kernel, no vLLM

**Result: Also produced garbage on pure CPU.**

This proved:
1. ✗ Not an XPU kernel bug
2. ✗ Not a vLLM weight loading bug
3. ✗ Not a V1 engine bug
4. **✓ The dequantization math itself is wrong**

Then asked: *How does transformers/auto_round actually dequantize?*

Found `reverse_awq_order()` in `auto_round/export/export_to_awq/utils.py`
and the `AWQ_REVERSE_ORDER` constant. Added it to the unpack step.
Standalone script immediately produced correct output ("Paris").

## Why the isolated kernel test was a false positive

The `debug_minimal.py` test loaded AWQ weights, unpacked them (WITHOUT
reverse order), repacked to CT format, and fed them to the XPU kernel.
The kernel correctly processed the data. But the repacked data was
already wrong — the sequential-assumption error was baked in at the
unpack step.

The CPU reference dequant in the same script also used sequential
unpacking. So both sides of the comparison had the identical error.
The round-trip was self-consistent: **bad unpack → repack → kernel
dequant = bad CPU unpack → dequant**. Cosine similarity = 1.0 because
both were wrong in the same way.

## Why it took so long

1. **Isolated test was a false friend** — round-trip self-consistency
   masked the bug. Both the CPU reference and the XPU path had the
   same sequential-unpacking error, making them agree perfectly.

2. **AWQ interleaved order is barely documented** — not mentioned in
   the AWQ paper, the AWQ GitHub README, or any obvious documentation.
   It's an implementation detail in one file of the `auto_round` package,
   inherited from MIT HAN Lab's original CUDA kernel.

3. **We compared against our own (equally wrong) reference** — every
   comparison script (CPU vs XPU, checkpoint vs vLLM-loaded) used the
   same bad assumption.

4. **vLLM V1 subprocess architecture slowed iteration** — each test
   run ~45 seconds. Monkey-patching didn't cross the process boundary.

5. **Red herrings consumed investigation time** — warmup `inf` values,
   zero attention outputs, scale transposition questions each looked
   plausible.

## Key lesson

**When an isolated kernel test passes but E2E inference fails, the bug
is in the data transformation between raw checkpoint and kernel input.**

The most effective debug technique was removing ALL intermediaries (vLLM,
XPU) and running a pure-CPU standalone forward pass. When that also
failed, the search space collapsed to "our dequant math is wrong," and
reading the reference implementation (`auto_round/export/export_to_awq/
utils.py`) immediately revealed the answer.

## Debug scripts in this directory

| Script | Purpose |
|---|---|
| `standalone_inference.py` | Full Qwen2 forward pass without vLLM (CPU-only). **The script that found the root cause.** |
| `debug_minimal.py` | Isolated single-layer kernel test (AWQ→CT repack + XPU kernel vs CPU ref) |
| `debug_weight_loading.py` | Compares raw checkpoint vs vLLM-loaded weights after QKV fusion |
| `dump_weights.py` | Monkey-patches vLLM to save pre-repack weight tensors to disk |
| `compare_dumped.py` | Compares dumped vLLM weights with manually-concatenated checkpoint |
| `check_nonquant.py` | Inspects non-quantized tensors (embeddings, layernorms, biases) |
| `print_expected.py` | Prints expected hex values from raw checkpoint for manual comparison |
| `test_dequant_compare.py` | Compares AWQ-direct dequant vs CT-repacked dequant (round-trip test) |
| `test_dequant_ref.py` | Replaces `apply()` with CPU dequant reference to isolate kernel vs math |
| `test_awq_path.py` | Tests V0 engine path and embedding dtype handling |
