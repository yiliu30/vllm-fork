# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accuracy and performance tests for FP8 query descale in triton attention.

Usage:
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.29.7 \
    python tests/kernels/attention/test_fp8_q_descale.py
"""

import time

import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

FP8_DTYPE = current_platform.fp8_dtype()

# ──────────────────────────────────────────────
# Reference: fp32 paged attention with explicit scales
# ──────────────────────────────────────────────


def ref_paged_attn_with_scales(
    query: torch.Tensor,  # fp32
    key_cache: torch.Tensor,  # fp32
    value_cache: torch.Tensor,  # fp32
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    q_descale: float = 1.0,
    k_descale: float = 1.0,
    v_descale: float = 1.0,
) -> torch.Tensor:
    """Reference paged attention that applies q/k/v descales explicitly."""
    num_seqs = len(query_lens)
    block_tables_np = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len].float()

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_np[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len].float()
        v = (
            value_cache[block_indices]
            .view(-1, num_kv_heads, head_size)[:kv_len]
            .float()
        )

        # Apply scales: effective_scale = softmax_scale * q_descale * k_descale
        q_scaled = q * scale * q_descale * k_descale

        if q_scaled.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q_scaled.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q_scaled.shape[1] // v.shape[1], dim=1)

        # S = Q_scaled @ K^T  (descales already folded into Q)
        attn = torch.einsum("qhd,khd->hqk", q_scaled, k).float()

        # Causal mask
        empty_mask = torch.ones(query_len, kv_len, device=q.device)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        attn.masked_fill_(mask, float("-inf"))

        attn = torch.softmax(attn, dim=-1)

        # V descale: v_descale applied to the output
        out = torch.einsum("hqk,khd->qhd", attn, v) * v_descale
        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


# ──────────────────────────────────────────────
# Accuracy test
# ──────────────────────────────────────────────


def test_accuracy(q_descale_val: float):
    """Test that FP8 Q with q_descale matches reference within tolerance."""
    torch.set_default_device("cuda")
    set_random_seed(42)

    # Setup
    num_seqs = 4
    seq_lens = [(1, 512), (3, 128), (1, 1024), (5, 64)]
    query_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    num_query_heads, num_kv_heads = 8, 2
    head_size = 128
    block_size = 16
    num_blocks = 4096
    scale = head_size**-0.5

    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)

    # Generate fp32 ground truth data
    query_fp32 = torch.randn(sum(query_lens), num_query_heads, head_size)
    key_cache_fp32 = torch.randn(num_blocks, block_size, num_kv_heads, head_size)
    value_cache_fp32 = torch.randn_like(key_cache_fp32)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    # Quantize to FP8
    query_fp8 = query_fp32.to(FP8_DTYPE)
    key_cache_fp8 = key_cache_fp32.to(FP8_DTYPE)
    value_cache_fp8 = value_cache_fp32.to(FP8_DTYPE)

    # Scales
    q_descale = torch.tensor([q_descale_val], dtype=torch.float32, device="cuda")
    scale_shape = (num_seqs, num_kv_heads)
    k_descale = torch.ones(scale_shape, dtype=torch.float32, device="cuda")
    v_descale = torch.ones(scale_shape, dtype=torch.float32, device="cuda")

    output = torch.empty_like(query_fp32)

    # Softmax segment buffers
    seq_threshold_3D = 0
    num_par_softmax_segments = 16
    head_size_padded = next_power_of_2(head_size)
    softmax_segm_output = torch.empty(
        (8, num_query_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
    )
    softmax_segm_max = torch.empty(
        (8, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )
    softmax_segm_expsum = torch.empty(
        (8, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )

    # Run kernel with q_descale
    unified_attention(
        q=query_fp8,
        k=key_cache_fp8,
        v=value_cache_fp8,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_tensor,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
    )

    # Reference: use DEQUANTIZED fp8 values (to match quantization error)
    query_deq = query_fp8.float()
    key_cache_deq = key_cache_fp8.float()
    value_cache_deq = value_cache_fp8.float()

    ref_output = ref_paged_attn_with_scales(
        query=query_deq,
        key_cache=key_cache_deq,
        value_cache=value_cache_deq,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        q_descale=q_descale_val,
        k_descale=1.0,
        v_descale=1.0,
    )

    # Compare
    max_diff = (output - ref_output).abs().max().item()
    mean_diff = (output - ref_output).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        output.reshape(-1).float(), ref_output.reshape(-1).float(), dim=0
    ).item()

    print(
        f"  q_descale={q_descale_val:.2f}: max_diff={max_diff:.6f}, "
        f"mean_diff={mean_diff:.6f}, cos_sim={cos_sim:.6f}"
    )

    # Tolerance: fp8 quantization introduces ~1e-2 error
    atol = 0.05
    try:
        torch.testing.assert_close(output, ref_output, atol=atol, rtol=0.05)
        print(f"  ✓ PASSED (atol={atol})")
        return True
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        return False


# ──────────────────────────────────────────────
# Performance benchmark
# ──────────────────────────────────────────────


def benchmark_fp8_query():
    """Compare bf16 Q+KV vs bf16 Q + fp8 KV vs fp8 Q + fp8 KV.

    The real benefit of FP8 query comes from:
    1. Halved Q memory bandwidth (1 byte vs 2 bytes per element)
    2. FP8 tensor core throughput (2x on H100 vs bf16)

    When Q is bf16 and KV is fp8, the kernel dequantizes K to bf16 before
    tl.dot(Q_bf16, K_bf16). When Q is fp8, tl.dot(Q_fp8, K_fp8) runs
    natively — no dequant needed, and the dot itself is faster on fp8 cores.
    """
    torch.set_default_device("cuda")
    set_random_seed(0)

    configs = [
        # (batch, query_len, kv_len, num_q_heads, num_kv_heads, head_size)
        (32, 1, 2048, 32, 8, 128),  # decode
        (64, 1, 4096, 32, 8, 128),  # decode large batch
        (128, 1, 2048, 32, 8, 128),  # decode very large batch
        (4, 128, 2048, 32, 8, 128),  # prefill
        (1, 512, 4096, 32, 8, 128),  # long prefill
        (1, 1024, 8192, 32, 8, 128),  # very long prefill
    ]

    block_size = 16
    num_par_softmax_segments = 16

    for batch, qlen, kvlen, nqh, nkvh, hs in configs:
        num_blocks = (kvlen // block_size + 1) * batch * 2
        head_size_padded = next_power_of_2(hs)

        query_lens = [qlen] * batch
        kv_lens = [kvlen] * batch
        total_q = sum(query_lens)

        # Create bf16 and fp8 versions of query
        q_bf16 = torch.randn(total_q, nqh, hs, dtype=torch.bfloat16)
        q_fp8 = q_bf16.to(FP8_DTYPE)

        # KV caches: bf16 and fp8
        kc_bf16 = torch.randn(num_blocks, block_size, nkvh, hs, dtype=torch.bfloat16)
        vc_bf16 = torch.randn(num_blocks, block_size, nkvh, hs, dtype=torch.bfloat16)
        kc_fp8 = kc_bf16.to(FP8_DTYPE)
        vc_fp8 = vc_bf16.to(FP8_DTYPE)

        output = torch.empty(total_q, nqh, hs, dtype=torch.bfloat16)

        cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
            dim=0, dtype=torch.int32
        )
        kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)
        max_num_blocks_per_seq = (kvlen + block_size - 1) // block_size
        block_tables = torch.randint(
            0, num_blocks, (batch, max_num_blocks_per_seq), dtype=torch.int32
        )

        scale_shape = (batch, nkvh)
        k_descale = torch.ones(scale_shape, dtype=torch.float32)
        v_descale = torch.ones(scale_shape, dtype=torch.float32)

        seq_threshold_3D = 0
        softmax_segm_output = torch.empty(
            (8, nqh, num_par_softmax_segments, head_size_padded),
            dtype=torch.float32,
        )
        softmax_segm_max = torch.empty(
            (8, nqh, num_par_softmax_segments),
            dtype=torch.float32,
        )
        softmax_segm_expsum = torch.empty(
            (8, nqh, num_par_softmax_segments),
            dtype=torch.float32,
        )

        scale = hs**-0.5

        def run(
            q_tensor,
            kc_tensor,
            vc_tensor,
            q_desc,
            kd,
            vd,
            _out=output,
            _cu=cu_query_lens,
            _kvt=kv_lens_tensor,
            _ql=query_lens,
            _kvl=kvlen,
            _sc=scale,
            _bt=block_tables,
            _st3d=seq_threshold_3D,
            _sso=softmax_segm_output,
            _ssm=softmax_segm_max,
            _sse=softmax_segm_expsum,
        ):
            unified_attention(
                q=q_tensor,
                k=kc_tensor,
                v=vc_tensor,
                out=_out,
                cu_seqlens_q=_cu,
                seqused_k=_kvt,
                max_seqlen_q=max(_ql),
                max_seqlen_k=_kvl,
                softmax_scale=_sc,
                causal=True,
                window_size=(-1, -1),
                block_table=_bt,
                softcap=0,
                q_descale=q_desc,
                k_descale=kd,
                v_descale=vd,
                seq_threshold_3D=_st3d,
                num_par_softmax_segments=16,
                softmax_segm_output=_sso,
                softmax_segm_max=_ssm,
                softmax_segm_expsum=_sse,
            )

        # Three scenarios to compare:
        scenarios = [
            ("bf16 Q+KV", q_bf16, kc_bf16, vc_bf16, None, None, None),
            (
                "bf16 Q + fp8 KV",
                q_bf16,
                kc_fp8,
                vc_fp8,
                None,
                k_descale,
                v_descale,
            ),
            (
                "fp8 Q + fp8 KV",
                q_fp8,
                kc_fp8,
                vc_fp8,
                0.85,
                k_descale,
                v_descale,
            ),
        ]

        config_str = f"B={batch}, Q={qlen}, KV={kvlen}, H={nqh}/{nkvh}, D={hs}"
        print(f"  {config_str}")

        n_iters = 200
        times = {}
        for label, q, kc, vc, qd, kd, vd in scenarios:
            # Warmup
            for _ in range(20):
                run(q, kc, vc, qd, kd, vd)
            torch.accelerator.synchronize()

            t0 = time.perf_counter()
            for _ in range(n_iters):
                run(q, kc, vc, qd, kd, vd)
            torch.accelerator.synchronize()
            t_ms = (time.perf_counter() - t0) / n_iters * 1000
            times[label] = t_ms

        t_bf16 = times["bf16 Q+KV"]
        for label, t_ms in times.items():
            speedup = t_bf16 / t_ms
            print(f"    {label:20s}: {t_ms:.3f} ms ({speedup:.2f}x vs bf16)")
        print()


if __name__ == "__main__":
    print("=" * 60)
    print("ACCURACY TESTS: FP8 query with various q_descale values")
    print("=" * 60)
    all_passed = True
    for val in [1.0, 0.5, 0.85, 1.5, 0.1]:
        passed = test_accuracy(val)
        all_passed = all_passed and passed
    print()

    if not all_passed:
        print("⚠ Some accuracy tests failed!")
    else:
        print("✓ All accuracy tests passed!")

    print()
    print("=" * 60)
    print("PERFORMANCE BENCHMARK: bf16 vs fp8 query")
    print("=" * 60)
    benchmark_fp8_query()
