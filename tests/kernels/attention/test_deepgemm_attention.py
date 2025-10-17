# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils import cdiv, has_deep_gemm
from vllm.utils.deep_gemm import (
    _ceil_to_ue8m0,
    calc_diff,
    fp8_mqa_logits,
    fp8_paged_mqa_logits,
    get_num_sms,
    get_paged_mqa_logits_metadata,
)


def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    # x: (num_blocks, block_size, 1, head_dim)
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4)),
        device=x.device,
        dtype=torch.uint8,
    )
    x_fp8[:, : block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim
    ).view(dtype=torch.uint8)
    x_fp8[:, block_size * head_dim :] = sf.view(num_blocks, block_size).view(
        dtype=torch.uint8
    )
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)


def per_custom_dims_cast_to_fp8(
    x: torch.Tensor, dims: tuple, use_ue8m0: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    excluded_dims = tuple([i for i in range(x.dim()) if i not in set(dims)])
    x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled, sf.squeeze()


def _generate_cp_test_data(seq_len: int, seq_len_kv: int):
    assert seq_len_kv % seq_len == 0 and seq_len % 2 == 0
    chunk_size = seq_len // 2
    cp_size = seq_len_kv // seq_len
    cp_id = cp_size // 3
    ks = torch.zeros(seq_len, dtype=torch.int, device="cuda")
    ke = torch.zeros(seq_len, dtype=torch.int, device="cuda")
    for i in range(chunk_size):
        ke[i] = cp_id * chunk_size + i
        ke[i + chunk_size] = (cp_size * 2 - 1 - cp_id) * chunk_size + i
    return ks, ke


def _ref_fp8_mqa_logits(
    q: torch.Tensor,             # [seq_len, heads, head_dim]
    kv: torch.Tensor,            # [seq_len_kv, head_dim]
    weights: torch.Tensor,       # [seq_len, heads]
    cu_seqlen_ks: torch.Tensor,  # [seq_len]
    cu_seqlen_ke: torch.Tensor,  # [seq_len]
):
    breakpoint()
    seq_len_kv = kv.shape[0]

    k = kv
    q = q.float()
    k = k.float()
    # mask_lo: [seq_len, seq_len_kv]
    # mask_hi: [seq_len, seq_len_kv]
    mask_lo = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
    )
    # mask: [seq_len, seq_len_kv]
    mask = mask_lo & mask_hi
    # q: [seq_len, heads, head_dim] -> [heads, seq_len, head_dim]
    # k: [seq_len_kv, head_dim]     -> [head_dim, seq_len_kv]
    # [heads, seq_len, head_dim] @ [head_dim, seq_len_kv] -> [heads, seq_len, seq_len_kv]
    score = torch.einsum("mhd,nd->hmn", q, k)
    # [seq_len, heads] -> [heads, seq_len, 1] -> [heads, seq_len, 1]
    cur_weight = weights.unsqueeze(-1).transpose(0, 1)
    # [heads, seq_len, seq_len_kv] * [heads, seq_len, 1] -> [heads, seq_len, seq_len_kv]
    # [heads, seq_len, seq_len_kv] -> [seq_len, seq_len_kv]
    logits = (score.relu() * cur_weight).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    return logits


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
# @pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
# @pytest.mark.skipif(
#     not current_platform.has_device_capability(90), reason="SM90 and SM100 only"
# )
def test_deepgemm_fp8_mqa_logits():
    torch.manual_seed(0)
    random.seed(0)
    num_heads, head_dim = 32, 128
    for seq_len in (512,):
        for seq_len_kv in (1024,):
            for disable_cp in (
                False,
                #   True,
                  ):
                q = torch.randn(
                    seq_len,
                    num_heads,
                    head_dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                kv = torch.randn(
                    seq_len_kv, head_dim, device="cuda", dtype=torch.bfloat16
                )
                weights = torch.randn(
                    seq_len, num_heads, device="cuda", dtype=torch.float32
                )

                if disable_cp:
                    ks = torch.zeros(seq_len, dtype=torch.int, device="cuda")
                    ke = torch.arange(seq_len, dtype=torch.int, device="cuda") + (
                        seq_len_kv - seq_len
                    )
                else:
                    ks, ke = _generate_cp_test_data(seq_len, seq_len_kv)

                q_fp8 = q.to(torch.float8_e4m3fn)
                kv_fp8 = per_custom_dims_cast_to_fp8(kv, (0,), False)

                ref_logits = _ref_fp8_mqa_logits(
                    q=q,
                    kv=kv,
                    weights=weights,
                    cu_seqlen_ks=ks,
                    cu_seqlen_ke=ke,
                )
                breakpoint()
                logits = fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke)

                ref_neginf_mask = ref_logits == float("-inf")
                neginf_mask = logits == float("-inf")
                assert torch.equal(neginf_mask, ref_neginf_mask)

                ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)
                logits = logits.masked_fill(neginf_mask, 0)
                diff = calc_diff(logits, ref_logits)
                assert diff < 1e-3, f"{diff=}"


def _ref_fp8_paged_mqa_logits(
    q: torch.Tensor,            # [batch_size, next_n, heads, head_dim], [batch_size, 1, heads, 128]
    kv_cache: torch.Tensor,     # [num_blocks, block_size, 1, head_dim], [num_blocks, 64, 1, 128]
    weights: torch.Tensor,      # [batch_size * next_n, heads]         , [batch_size * 1, heads]
    context_lens: torch.Tensor, # [batch_size]
    block_tables: torch.Tensor, # [batch_size, max_block_len]
    max_model_len: int,         
):
    """
    This is a reference implementation of the fp8_paged_mqa_logits function.
    The pseudo code is as follows:
    # For each sequence in the batch, update its logits
    for batch_idx in range(batch_size):
        cur_context_len = context_lens[batch_idx].item()
        block_nums = (cur_context_len + block_size - 1) // block_size
        cur_query = q[batch_idx]  # [1, heads, head_dim]
        # For each block in the current sequence, compute the attention scores
        for block_rk in range(block_nums):
            block_idx = block_tables[batch_idx][block_rk]
            cur_key_in_block = kv_cache[block_idx] # [block_size, 1, head_dim]
            # [heads, 1, head_dim] @ [1, head_dim, block_size] -> [heads, 1, block_size]
            cur_attn_score_temp = cur_query.transpose(0, 1) @ cur_key_in_block.transpose(0, 1).transpose(1, 2)
            # Apply ReLU
            act_cur_attn_score_temp = torch.relu(cur_attn_score_temp.to(torch.float32))
            # Apply the head weights, the weighted average over all heads
            act_cur_attn_score_temp_weighted = act_cur_attn_score_temp * weights[i].transpose(0, 1)[..., None]
            # Sum over all heads to get the final logits for the each key position in the block
            cur_attn_score = act_cur_attn_score_temp_weighted.sum(dim=0) # [1, block_size]
            # Update the logits for the current block, the formula ignores the mask
            logits[batch_idx][block_rk*block_size:(block_rk+1)*block_size] = cur_attn_score
    """
    
    batch_size, next_n, _, _ = q.size()
    _, block_size, _, _ = kv_cache.size()
    # [batch_size, max_model_len]
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    context_lens_list = context_lens.tolist()
    for i in range(batch_size):
        # breakpoint()
        context_len = context_lens_list[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device="cuda")
        # weights: [batch_size, heads] -> [1, heads] -> [heads, 1]
        # weight_slice: [heads, 1]
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        # For the current batch item, iterate over all blocks
        # Based on the current context length and block size, there are ceil(context_len / block_size) blocks
        # For example, if context_len=2447 and block_size=64, there are 39 blocks
        for block_rk in range(cdiv(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            # qx: [1, heads, head_dim]
            # kx: [block_size, 1, head_dim]
            qx, kx = q[i], kv_cache[block_idx]
            # k_offsets: [block_size]
            k_offsets = torch.arange(
                block_rk * block_size,
                (block_rk + 1) * block_size,
                device="cuda",
            )
            # mask: [1, block_size]
            mask = (k_offsets[None, :] < context_len) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            # qx: [1, heads, head_dim]      -> [heads, 1, head_dim]
            qx_temp = qx.transpose(0, 1).contiguous()
            # kx: [block_size, 1, head_dim] -> [1, block_size, head_dim] -> [1, head_dim, block_size]
            kx_temp = kx.transpose(0, 1).transpose(1, 2).contiguous()
            # [heads, 1, head_dim] @ [1, head_dim, block_size] -> [heads, 1, block_size]
            # For given query, compute attention scores with all keys in the block
            temp_s = (qx_temp @ kx_temp).to(torch.float32)  # [heads, 1, block_size]
            s = torch.where(
                mask[None, :, :],
                temp_s,
                float("-inf"),
            )
            # s = torch.where(
            #     mask[None, :, :],
            #     (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
            #         logits.dtype
            #     ),
            #     float("-inf"),
            # )
            # 
            # cur_weight: [heads, 1, 1]
            cur_weight = weight_slice[..., None]
            # [heads, 1, block_size] * [heads, 1, 1] -> [heads, 1, block_size]
            s = torch.relu(s) * cur_weight
            # sum the score over all heads
            # [heads, 1, block_size] -> [1, block_size]
            s = s.sum(dim=0)
            # Update the logits for the current batch item and current block
            # k_offsets: stands for the token positions of the keys in the current block
            # q_offsets: stands for the token positions of the queries in the current next_n
            cur_batch_start = i * next_n
            cur_batch_end = (i + 1) * next_n
            cur_block_start = block_rk * block_size
            cur_block_end = (block_rk + 1) * block_size
            print(f"Update logits for batch [{cur_batch_start}, {cur_batch_end}), block [{cur_block_start}, {cur_block_end})")
            if cur_block_end == 2496:
                breakpoint()
            cur_logist = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = cur_logist
    return logits


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
# @pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
# @pytest.mark.skipif(
#     not current_platform.has_device_capability(90), reason="SM90 and SM100 only"
# )
def test_deepgemm_fp8_paged_mqa_logits():
    torch.manual_seed(0)
    random.seed(0)

    max_model_len = 4096
    for batch_size, next_n in [
        (4, 1),
          (2, 2)
          ]:
        for heads, index_dim in [(32, 128)]:
            for avg_kv in (2048,):
                num_blocks, blocksize = max_model_len * 2, 64

                q = torch.randn(
                    (batch_size, next_n, heads, index_dim),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                kv_cache = torch.randn(
                    (num_blocks, blocksize, 1, index_dim),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                weights = torch.randn(
                    (batch_size * next_n, heads),
                    device="cuda",
                    dtype=torch.float32,
                )

                context_lens = (
                    torch.randint(int(0.8 * avg_kv), int(1.2 * avg_kv), (batch_size,))
                    .cuda()
                    .to(torch.int32)
                )
                max_block_len = (
                    (context_lens.max().item() + blocksize - 1) // blocksize * blocksize
                )
                block_tables = torch.zeros(
                    (batch_size, max_block_len),
                    device="cuda",
                    dtype=torch.int32,
                )

                counter = 0
                block_idx_pool = list(range(num_blocks))
                random.shuffle(block_idx_pool)
                for i in range(batch_size):
                    ctx_len = int(context_lens[i].item())
                    for j in range((ctx_len + blocksize - 1) // blocksize):
                        block_tables[i][j] = block_idx_pool[counter]
                        counter += 1

                q_fp8 = q.to(torch.float8_e4m3fn)
                kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache)


                ref_logits = _ref_fp8_paged_mqa_logits(
                    q,
                    kv_cache,
                    weights,
                    context_lens,
                    block_tables,
                    max_model_len,
                )
                
                schedule_metadata = get_paged_mqa_logits_metadata(
                    context_lens, blocksize, get_num_sms()
                )



                logits = fp8_paged_mqa_logits(
                    q_fp8,
                    kv_cache_fp8,
                    weights,
                    context_lens,
                    block_tables,
                    schedule_metadata,
                    max_model_len,
                )
                positions = (
                    torch.arange(max_model_len, device="cuda")
                    .unsqueeze(0)
                    .expand(batch_size * next_n, -1)
                )
                row_indices = torch.arange(batch_size * next_n, device="cuda") // next_n
                next_n_offset = (
                    torch.arange(batch_size * next_n, device="cuda") % next_n
                )
                mask = positions <= (
                    context_lens[row_indices] - next_n + next_n_offset
                ).unsqueeze(1)

                logits = logits.masked_fill(~mask, 0)
                ref_logits = ref_logits.masked_fill(~mask, 0)
                diff = calc_diff(logits, ref_logits)
                assert diff < 1e-3, f"{diff=}"
