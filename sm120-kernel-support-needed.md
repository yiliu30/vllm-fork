# SM120 Support Needed: FlashMLA & DeepGemm

## Background

SM120 (RTX 6000D, RTX 5090 — Blackwell desktop/workstation) is a **separate architecture family** from SM100/SM103 (Blackwell data center). SM120 reports `major=12`, so `is_device_capability_family(100)` returns **False** (`120//10=12 != 100//10=10`).

DeepSeek V4 on SM120 currently requires pure-torch fallbacks for all DeepGemm and FlashMLA operations. The model runs but produces **incorrect output** due to the complexity of faithfully reimplementing these kernels in Python.

---

## FlashMLA — What Needs SM120 Support -> torch ref impl

### 1. Sparse Decode (`sparse_decode_fwd`)
- **Current**: Rejects SM120 with "Unsupported architecture for sparse decode fwd"
- **Impact**: Critical path for every decode step in DeepSeek V4 MLA attention
- **Needs**: Compile/validate the CUDA kernel for SM120 target, or add SM120 to the architecture allowlist

### 2. Sparse Prefill (`sparse_prefill_fwd`)
- **Current**: Rejects SM120 with "Only supported on SM90a and SM100f architectures"
- **Impact**: Used for prefill sparse attention in FP8 KV cache mode
- **Needs**: Same — compile for SM120 or extend architecture check

### 3. Dense Decode (`dense_decode_fwd`)
- **Current**: Same architecture rejection
- **Impact**: Used for non-sparse (short sequence) decode attention
- **Needs**: Same

### 4. Dense Prefill (`dense_prefill_fwd`)
- **Current**: Untested on SM120 — may or may not work (uses FlashInfer-style API internally)
- **Needs**: Verify on SM120

### Summary for FlashMLA
All four kernel entry points (`sparse_prefill_fwd`, `sparse_decode_fwd`, `dense_decode_fwd`, `dense_prefill_fwd`) need SM120 architecture support added. The core math (attention, softmax, FP8 dequant) should be architecture-agnostic — the gap is likely just the arch dispatch/compilation target.

---

## DeepGemm — What Needs SM120 Support

### 1. FP8 GEMM (`fp8_gemm_nt`)  -> _upcast_e8m0_to_fp32
- **Current**: `is_deep_gemm_supported()` returns False on SM120
- **Impact**: All FP8 linear layers fall back to CUTLASS/Triton (slower)
- **Needs**: Compile for SM120 and add to supported arch list

### 2. Grouped FP8 GEMM (`m_grouped_fp8_gemm_nt_contiguous`, `fp8_m_grouped_gemm_nt_masked`) -> MarlinMoE
- **Current**: Same — disabled on SM120
- **Impact**: MoE expert computation falls back to Triton MoE
- **Needs**: Same

### 3. FP8 Einsum (`fp8_einsum`) -> DeepGemm ref imp (torch is ok)
- **Current**: Disabled on SM120, uses torch fallback (`torch.einsum` after dequant)
- **Impact**: MLA O-projection (attention output → hidden states). Torch fallback works but is slow.
- **Needs**: Compile for SM120

### 4. FP8/FP4 Paged MQA Logits (`fp8_fp4_paged_mqa_logits`) -> Deepgemm ref imp
- **Current**: Disabled on SM120, uses torch fallback (Python loop over batches — very slow)
- **Impact**: Sparse attention indexer — selects which KV tokens to attend to during decode
- **Needs**: Compile for SM120

### 5. FP8/FP4 MQA Logits (`fp8_fp4_mqa_logits`) -> torch reference
- **Current**: Same pattern
- **Impact**: Non-paged MQA logits for prefill indexer
- **Needs**: Same

### 6. TF32 HC PreNorm GEMM (`tf32_hc_prenorm_gemm`) -> deepgemm ref impl
- **Current**: Disabled, uses torch fallback (matmul + manual RMS norm)
- **Impact**: MHC (Multi-Head Coefficients) layer
- **Needs**: Same

### Summary for DeepGemm
The architecture check in DeepGemm (`layout.hpp:39` or equivalent) rejects SM120. All six kernel families need SM120 added to the supported architecture list. The underlying PTX/SASS should be compilable for SM120 since it shares the Blackwell ISA base with SM100, but this needs verification — SM120 may lack certain SM100-specific features (e.g., TMA multicast differences).

---

## Priority

| Priority | Kernel | Why |
|----------|--------|-----|
| **P0** | FlashMLA sparse decode | Every decode token goes through this |
| **P0** | FlashMLA sparse prefill | Every prefill goes through this (FP8 mode) |
| **P1** | DeepGemm paged MQA logits | Sparse indexer — wrong tokens = garbage output |
| **P1** | DeepGemm FP8 GEMM + grouped | All linear + MoE performance |
| **P2** | DeepGemm fp8_einsum | O-projection performance |
| **P2** | FlashMLA dense decode/prefill | Fallback paths for short sequences |
| **P3** | DeepGemm tf32_hc_prenorm_gemm | MHC layer, torch fallback is adequate |
