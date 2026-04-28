# Debugging Triton FP8 Attention Accuracy

This note covers one specific failure mode: a compressed-tensors attention-FP8 checkpoint produces sane output in `transformers` and `FLASHINFER`, but `TRITON_ATTN` generates corrupted text.

The fix in this repo was not in model loading and not in KV-cache writes. The bug was in Triton attention itself. When the query was FP8 and the KV cache used per-tensor FP8 scales, `_prepare_kv_tile` kept K and V in FP8 for the fast path, but the kernel stopped applying `k_scale` and `v_scale`. That dropped the K/V descales from the score path and the output path.

## What to check first

Start with a small backend matrix before you touch kernel code:

- Run the same checkpoint in `transformers` on GPU. If that output is sane, the checkpoint and calibration scales are probably fine.
- Run the same checkpoint in vLLM with `attention_backend=auto` or `FLASHINFER`. If that output is sane and `TRITON_ATTN` is not, the bug is backend-specific.
- Compare Triton cache writes against `_C_cache_ops.reshape_and_cache_flash` before you assume the cache path is wrong.
- Use non-unit scales. This bug hides if `k_descale=1` and `v_descale=1`.

## Tips that saved time

- Split the problem into cache write and attention math. The cache writer matched the C++ op for FP8 per-tensor scales, which let us stop digging in the wrong file.
- Reproduce with `unified_attention` directly. You do not need to load a full model to catch this class of bug.
- Force `kv_quant_mode=KVQuantMode.FP8_PER_TENSOR`. If you leave it at the default, the descales are ignored by design and your repro is not testing the quantized path.
- Use FP8 tensors plus non-unit `q_descale`, `k_descale`, and `v_descale`. The old bug only showed up once the scales mattered.
- Run both Triton decode variants. In practice, `seq_threshold_3D=0` exercises the 2D path and `seq_threshold_3D=8` exercises the 3D path.
- Compare against a reference built from the dequantized FP8 values, not from the original BF16 tensors. You want to isolate the scaling bug, not the quantization error.
- If your local environment needs it, preload NCCL before importing `torch`. On the machine used for this debug session, that was required.

## Minimal repro

This script reproduces the accuracy bug without loading a model. Before the fix, `max_diff` was large. After the fix, it stays within the regression tolerance.

```python
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode

FP8_DTYPE = current_platform.fp8_dtype()


def ref_paged_attn(query, key_cache, value_cache, query_lens, kv_lens,
                   block_tables, scale, q_descale, k_descale, v_descale):
    block_tables_np = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape
    outputs = []
    start = 0
    for i, query_len in enumerate(query_lens):
        kv_len = kv_lens[i]
        q = query[start:start + query_len].float() * scale * q_descale * k_descale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_np[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len].float()
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len].float()
        v = v * v_descale

        if q.shape[1] != k.shape[1]:
            repeat = q.shape[1] // k.shape[1]
            k = torch.repeat_interleave(k, repeat, dim=1)
            v = torch.repeat_interleave(v, repeat, dim=1)

        scores = torch.einsum("qhd,khd->hqk", q, k).float()
        mask = torch.triu(
            torch.ones(query_len, kv_len, device=q.device),
            diagonal=kv_len - query_len + 1,
        ).bool()
        scores.masked_fill_(mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        outputs.append(torch.einsum("hqk,khd->qhd", probs, v))
        start += query_len

    return torch.cat(outputs, dim=0)


torch.set_default_device("cuda")
torch.manual_seed(0)

seq_lens = [(2, 64), (1, 32)]
query_lens = [x[0] for x in seq_lens]
kv_lens = [x[1] for x in seq_lens]
num_seqs = len(seq_lens)
num_query_heads = 8
num_kv_heads = 2
head_size = 128
block_size = 16
num_blocks = 256
scale = head_size**-0.5
q_descale = 0.25
k_descale = 0.5
v_descale = 2.0

query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=torch.bfloat16)
key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size,
                        dtype=torch.bfloat16)
value_cache = torch.randn_like(key_cache)

query_fp8 = query.to(FP8_DTYPE)
key_cache_fp8 = key_cache.to(FP8_DTYPE)
value_cache_fp8 = value_cache.to(FP8_DTYPE)

cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
    dim=0, dtype=torch.int32
)
kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)
max_num_blocks_per_seq = (max(kv_lens) + block_size - 1) // block_size
block_tables = torch.randint(
    0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
)

output = torch.empty_like(query)
scale_shape = (num_seqs, num_kv_heads)
k_descale_tensor = torch.full(scale_shape, k_descale, dtype=torch.float32)
v_descale_tensor = torch.full(scale_shape, v_descale, dtype=torch.float32)

num_par_softmax_segments = 16
head_size_padded = next_power_of_2(head_size)
softmax_segm_output = torch.empty(
    (0, num_query_heads, num_par_softmax_segments, head_size_padded),
    dtype=torch.float32,
)
softmax_segm_max = torch.empty((0, num_query_heads, num_par_softmax_segments),
                               dtype=torch.float32)
softmax_segm_expsum = torch.empty((0, num_query_heads, num_par_softmax_segments),
                                  dtype=torch.float32)

unified_attention(
    q=query_fp8,
    k=key_cache_fp8,
    v=value_cache_fp8,
    out=output,
    cu_seqlens_q=cu_query_lens,
    seqused_k=kv_lens_tensor,
    max_seqlen_q=max(query_lens),
    max_seqlen_k=max(kv_lens),
    softmax_scale=scale,
    causal=True,
    window_size=(-1, -1),
    block_table=block_tables,
    softcap=0,
    q_descale=torch.tensor([q_descale], dtype=torch.float32),
    k_descale=k_descale_tensor,
    v_descale=v_descale_tensor,
    seq_threshold_3D=0,
    num_par_softmax_segments=num_par_softmax_segments,
    softmax_segm_output=softmax_segm_output,
    softmax_segm_max=softmax_segm_max,
    softmax_segm_expsum=softmax_segm_expsum,
    kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
)

ref = ref_paged_attn(
    query=query_fp8.float(),
    key_cache=key_cache_fp8.float(),
    value_cache=value_cache_fp8.float(),
    query_lens=query_lens,
    kv_lens=kv_lens,
    block_tables=block_tables,
    scale=scale,
    q_descale=q_descale,
    k_descale=k_descale,
    v_descale=v_descale,
)

print("max_diff", (output.float() - ref).abs().max().item())
print("mean_diff", (output.float() - ref).abs().mean().item())
```

## Targeted verification

The regression test added with this fix is:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.29.7 \
/home/yiliu7/workspace/venvs/vllm/bin/python -m pytest \
tests/kernels/attention/test_triton_unified_attention.py \
-k "fp8_query_applies_kv_descales" -v
```

That test failed before the fix and passes after it.
