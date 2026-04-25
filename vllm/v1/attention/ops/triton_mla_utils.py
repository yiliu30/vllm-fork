# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for merging attention outputs with LSE (log-sum-exp) scores.

Used to combine:
1. Compressed (top-k) + SWA (sliding window) attention outputs
2. Merged attention output with DeepSeek V4's learned attention sink bias
"""

import torch


def lse_merge(
    out1: torch.Tensor,
    lse1: torch.Tensor,
    out2: torch.Tensor,
    lse2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two attention outputs using their LSE scores.

    Args:
        out1: [num_tokens, num_heads, dv] bf16 — first attention output
        lse1: [num_tokens, num_heads] float32 — first LSE
        out2: [num_tokens, num_heads, dv] bf16 — second attention output
        lse2: [num_tokens, num_heads] float32 — second LSE

    Returns:
        merged_out: [num_tokens, num_heads, dv] bf16
        merged_lse: [num_tokens, num_heads] float32
    """
    # Numerically stable merge via online softmax
    merge_max = torch.maximum(lse1, lse2)
    w1 = torch.exp(lse1 - merge_max)
    w2 = torch.exp(lse2 - merge_max)
    denom = w1 + w2

    # Avoid 0/0 when both LSEs are -inf (no valid KV tokens)
    denom = torch.where(denom > 0, denom, torch.ones_like(denom))

    merged_out = (
        out1.float() * w1.unsqueeze(-1) + out2.float() * w2.unsqueeze(-1)
    ) / denom.unsqueeze(-1)
    merged_lse = merge_max + torch.log(denom)

    return merged_out.to(out1.dtype), merged_lse


def merge_attention_with_sink(
    out: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: torch.Tensor,
) -> torch.Tensor:
    """Merge attention output with DeepSeek V4's learned attention sink bias.

    The attention sink is a per-head learned bias that acts as a baseline
    attention score. It ensures DeepSeek V4's sink-aware denominator behavior
    is preserved.

    Formula:
        merge_max = max(lse, attn_sink)
        w_attn = exp(lse - merge_max)
        w_sink = exp(attn_sink - merge_max)
        output = (out * w_attn) / (w_attn + w_sink)

    Note: The sink contributes to the denominator but not the numerator
    (it has no associated value vector — it just "absorbs" attention mass).

    Args:
        out:       [num_tokens, num_heads, dv] bf16 — attention output
        lse:       [num_tokens, num_heads] float32 — attention LSE
        attn_sink: [num_heads] float32 — learned per-head sink bias

    Returns:
        output: [num_tokens, num_heads, dv] bf16
    """
    # attn_sink: [H] -> [1, H] for broadcasting
    sink = attn_sink.unsqueeze(0)  # [1, H]

    merge_max = torch.maximum(lse, sink)
    w_attn = torch.exp(lse - merge_max)
    w_sink = torch.exp(sink - merge_max)
    denom = w_attn + w_sink

    # Avoid 0/0
    denom = torch.where(denom > 0, denom, torch.ones_like(denom))

    # Sink has no value vector, so only w_attn scales the output
    result = out.float() * w_attn.unsqueeze(-1) / denom.unsqueeze(-1)
    return result.to(out.dtype)
