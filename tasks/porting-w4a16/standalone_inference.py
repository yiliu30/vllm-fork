"""Standalone Qwen2 inference with AWQ dequant — no vLLM, no kernel.
Uses only torch + safetensors + transformers tokenizer."""
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoConfig

MODEL_ID = "Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound"
MODEL_PATH = (
    "/home/yiliu7/.cache/huggingface/hub/"
    "models--Intel--Qwen2-0.5B-Instruct-int4-sym-AutoRound/"
    "snapshots/cb1835a214eaba028ca8f041172c0f9ddea78a3d/model.safetensors"
)
DEVICE = "cpu"
DTYPE = torch.float16  # model's declared dtype

# AWQ interleaved packing order
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def awq_dequant(qw, sc, qz_val=8, bits=4, group_size=128):
    """Dequant AWQ-packed weights with correct AWQ nibble ordering."""
    pack_factor = 32 // bits
    mask = (1 << bits) - 1
    in_size, out_packed = qw.shape
    out_size = out_packed * pack_factor

    # Step 1: Unpack columnwise (each int32 → 8 nibbles)
    shifts = torch.arange(0, 32, bits, device=qw.device)
    unpacked = torch.bitwise_right_shift(
        qw[:, :, None], shifts[None, None, :]
    ).to(torch.int32)
    unpacked = unpacked.view(unpacked.shape[0], -1)  # [in, out]

    # Step 2: Reverse AWQ interleaved order
    reverse_order = torch.arange(out_size, dtype=torch.int32, device=qw.device)
    reverse_order = reverse_order.view(-1, pack_factor)
    reverse_order = reverse_order[:, AWQ_REVERSE_ORDER]
    reverse_order = reverse_order.view(-1)
    unpacked = (unpacked[:, reverse_order]) & mask

    # Step 3: Dequantize
    num_groups = in_size // group_size
    dequant = torch.zeros((in_size, out_size), dtype=torch.float32, device=qw.device)
    for g in range(num_groups):
        rs, re = g * group_size, (g + 1) * group_size
        dequant[rs:re, :] = (unpacked[rs:re, :].float() - qz_val) * sc[g].float().unsqueeze(0)
    return dequant


def main():
    config = AutoConfig.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tensors = load_file(MODEL_PATH)

    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
    print(f"Input IDs: {input_ids.tolist()}")
    print(f"Config: hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}, "
          f"num_heads={config.num_attention_heads}, num_kv_heads={config.num_key_value_heads}")

    nl = config.num_hidden_layers
    hs = config.hidden_size
    nh = config.num_attention_heads
    nkv = config.num_key_value_heads
    hd = hs // nh
    gs = 128

    # Embed
    embed_w = tensors["model.embed_tokens.weight"].to(DTYPE).to(DEVICE)
    hidden = embed_w[input_ids.to(DEVICE)]  # [seq, hidden]
    print(f"After embed: {hidden.shape}, sum={hidden.sum().item():.4f}")

    for layer_idx in range(nl):
        prefix = f"model.layers.{layer_idx}"

        # Input layernorm (RMSNorm)
        ln_w = tensors[f"{prefix}.input_layernorm.weight"].to(DTYPE).to(DEVICE)
        normed = F.rms_norm(hidden.float(), (hs,)).to(DTYPE) * ln_w

        # QKV projection
        q_dequant = awq_dequant(
            tensors[f"{prefix}.self_attn.q_proj.qweight"].to(DEVICE),
            tensors[f"{prefix}.self_attn.q_proj.scales"].to(DEVICE),
        )
        k_dequant = awq_dequant(
            tensors[f"{prefix}.self_attn.k_proj.qweight"].to(DEVICE),
            tensors[f"{prefix}.self_attn.k_proj.scales"].to(DEVICE),
        )
        v_dequant = awq_dequant(
            tensors[f"{prefix}.self_attn.v_proj.qweight"].to(DEVICE),
            tensors[f"{prefix}.self_attn.v_proj.scales"].to(DEVICE),
        )

        q = (normed.float() @ q_dequant).to(DTYPE)
        k = (normed.float() @ k_dequant).to(DTYPE)
        v = (normed.float() @ v_dequant).to(DTYPE)

        # Add biases
        q = q + tensors[f"{prefix}.self_attn.q_proj.bias"].to(DTYPE).to(DEVICE)
        k = k + tensors[f"{prefix}.self_attn.k_proj.bias"].to(DTYPE).to(DEVICE)
        v = v + tensors[f"{prefix}.self_attn.v_proj.bias"].to(DTYPE).to(DEVICE)

        # Reshape for MHA: [seq, heads, head_dim]
        seq_len = q.shape[0]
        q = q.view(seq_len, nh, hd)
        k = k.view(seq_len, nkv, hd)
        v = v.view(seq_len, nkv, hd)

        # GQA: repeat k,v for each head group
        n_rep = nh // nkv
        if n_rep > 1:
            k = k.unsqueeze(2).expand(-1, -1, n_rep, -1).reshape(seq_len, nh, hd)
            v = v.unsqueeze(2).expand(-1, -1, n_rep, -1).reshape(seq_len, nh, hd)

        # RoPE (simplified: use torch's scaled_dot_product_attention with causal=True)
        # Skip RoPE for now — just test basic attention flow
        # Apply RoPE
        inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, hd, 2, dtype=torch.float32, device=DEVICE) / hd))
        positions = torch.arange(seq_len, dtype=torch.float32, device=DEVICE)
        freqs = torch.outer(positions, inv_freq)
        cos_f = torch.cos(freqs).to(DTYPE)
        sin_f = torch.sin(freqs).to(DTYPE)

        def apply_rope(x, cos, sin):
            x1 = x[..., :hd//2]
            x2 = x[..., hd//2:]
            return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

        q = apply_rope(q, cos_f.unsqueeze(1), sin_f.unsqueeze(1))
        k = apply_rope(k, cos_f.unsqueeze(1), sin_f.unsqueeze(1))

        # Attention: [seq, heads, head_dim] -> transpose to [heads, seq, head_dim]
        q = q.transpose(0, 1)  # [nh, seq, hd]
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(0, 1).contiguous().view(seq_len, hs)

        # O projection
        o_dequant = awq_dequant(
            tensors[f"{prefix}.self_attn.o_proj.qweight"].to(DEVICE),
            tensors[f"{prefix}.self_attn.o_proj.scales"].to(DEVICE),
        )
        attn_output = (attn_out.float() @ o_dequant).to(DTYPE)

        # Residual connection
        hidden = hidden + attn_output

        # Post-attention layernorm
        ln2_w = tensors[f"{prefix}.post_attention_layernorm.weight"].to(DTYPE).to(DEVICE)
        normed2 = F.rms_norm(hidden.float(), (hs,)).to(DTYPE) * ln2_w

        # MLP: gate_proj, up_proj, down_proj with SiLU activation
        gate_dequant = awq_dequant(
            tensors[f"{prefix}.mlp.gate_proj.qweight"].to(DEVICE),
            tensors[f"{prefix}.mlp.gate_proj.scales"].to(DEVICE),
        )
        up_dequant = awq_dequant(
            tensors[f"{prefix}.mlp.up_proj.qweight"].to(DEVICE),
            tensors[f"{prefix}.mlp.up_proj.scales"].to(DEVICE),
        )
        down_dequant = awq_dequant(
            tensors[f"{prefix}.mlp.down_proj.qweight"].to(DEVICE),
            tensors[f"{prefix}.mlp.down_proj.scales"].to(DEVICE),
        )

        gate_out = (normed2.float() @ gate_dequant).to(DTYPE)
        up_out = (normed2.float() @ up_dequant).to(DTYPE)
        mlp_out = F.silu(gate_out) * up_out
        mlp_proj = (mlp_out.float() @ down_dequant).to(DTYPE)

        # Residual connection
        hidden = hidden + mlp_proj

        if layer_idx == 0:
            print(f"After layer 0: hidden sum={hidden.sum().item():.4f}, "
                  f"first5={hidden[-1, :5].tolist()}")

    # Final layernorm
    final_ln_w = tensors["model.norm.weight"].to(DTYPE).to(DEVICE)
    hidden = F.rms_norm(hidden.float(), (hs,)).to(DTYPE) * final_ln_w

    # LM head (tied to embed_tokens)
    logits = hidden @ embed_w.t()  # [seq, vocab]

    # Get next token (greedy from last position)
    next_token_id = logits[-1].argmax().item()
    next_token = tokenizer.decode([next_token_id])
    print(f"\nLogits[-1] shape: {logits[-1].shape}")
    print(f"Top-5 tokens: {torch.topk(logits[-1], 5)}")
    print(f"Next token ID: {next_token_id}")
    print(f"Next token: {repr(next_token)}")

    # Generate a few more tokens
    generated = [next_token_id]
    for _ in range(9):
        # Simple: just append and re-run (very slow but correct)
        all_ids = torch.cat([input_ids.to(DEVICE),
                             torch.tensor(generated, device=DEVICE)])
        hidden = embed_w[all_ids]

        for layer_idx in range(nl):
            prefix = f"model.layers.{layer_idx}"
            ln_w = tensors[f"{prefix}.input_layernorm.weight"].to(DTYPE).to(DEVICE)
            normed = F.rms_norm(hidden.float(), (hs,)).to(DTYPE) * ln_w

            q_dequant = awq_dequant(
                tensors[f"{prefix}.self_attn.q_proj.qweight"].to(DEVICE),
                tensors[f"{prefix}.self_attn.q_proj.scales"].to(DEVICE))
            k_dequant = awq_dequant(
                tensors[f"{prefix}.self_attn.k_proj.qweight"].to(DEVICE),
                tensors[f"{prefix}.self_attn.k_proj.scales"].to(DEVICE))
            v_dequant = awq_dequant(
                tensors[f"{prefix}.self_attn.v_proj.qweight"].to(DEVICE),
                tensors[f"{prefix}.self_attn.v_proj.scales"].to(DEVICE))

            q = (normed.float() @ q_dequant).to(DTYPE) + tensors[f"{prefix}.self_attn.q_proj.bias"].to(DTYPE).to(DEVICE)
            k = (normed.float() @ k_dequant).to(DTYPE) + tensors[f"{prefix}.self_attn.k_proj.bias"].to(DTYPE).to(DEVICE)
            v = (normed.float() @ v_dequant).to(DTYPE) + tensors[f"{prefix}.self_attn.v_proj.bias"].to(DTYPE).to(DEVICE)

            cur_seq = q.shape[0]
            q = q.view(cur_seq, nh, hd)
            k = k.view(cur_seq, nkv, hd)
            v = v.view(cur_seq, nkv, hd)
            if n_rep > 1:
                k = k.unsqueeze(2).expand(-1, -1, n_rep, -1).reshape(cur_seq, nh, hd)
                v = v.unsqueeze(2).expand(-1, -1, n_rep, -1).reshape(cur_seq, nh, hd)

            inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, hd, 2, dtype=torch.float32, device=DEVICE) / hd))
            positions = torch.arange(cur_seq, dtype=torch.float32, device=DEVICE)
            freqs = torch.outer(positions, inv_freq)
            cos_f = torch.cos(freqs).to(DTYPE)
            sin_f = torch.sin(freqs).to(DTYPE)
            q = apply_rope(q, cos_f.unsqueeze(1), sin_f.unsqueeze(1))
            k = apply_rope(k, cos_f.unsqueeze(1), sin_f.unsqueeze(1))

            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn_out = attn_out.transpose(0, 1).contiguous().view(cur_seq, hs)

            o_dequant = awq_dequant(
                tensors[f"{prefix}.self_attn.o_proj.qweight"].to(DEVICE),
                tensors[f"{prefix}.self_attn.o_proj.scales"].to(DEVICE))
            hidden = hidden + (attn_out.float() @ o_dequant).to(DTYPE)

            ln2_w = tensors[f"{prefix}.post_attention_layernorm.weight"].to(DTYPE).to(DEVICE)
            normed2 = F.rms_norm(hidden.float(), (hs,)).to(DTYPE) * ln2_w
            gate_d = awq_dequant(tensors[f"{prefix}.mlp.gate_proj.qweight"].to(DEVICE),
                                  tensors[f"{prefix}.mlp.gate_proj.scales"].to(DEVICE))
            up_d = awq_dequant(tensors[f"{prefix}.mlp.up_proj.qweight"].to(DEVICE),
                                tensors[f"{prefix}.mlp.up_proj.scales"].to(DEVICE))
            down_d = awq_dequant(tensors[f"{prefix}.mlp.down_proj.qweight"].to(DEVICE),
                                  tensors[f"{prefix}.mlp.down_proj.scales"].to(DEVICE))
            gate_o = (normed2.float() @ gate_d).to(DTYPE)
            up_o = (normed2.float() @ up_d).to(DTYPE)
            hidden = hidden + (((F.silu(gate_o) * up_o).float()) @ down_d).to(DTYPE)

        hidden = F.rms_norm(hidden.float(), (hs,)).to(DTYPE) * final_ln_w
        logits = hidden @ embed_w.t()
        next_id = logits[-1].argmax().item()
        generated.append(next_id)

    output_text = tokenizer.decode(generated)
    print(f"\nGenerated: {repr(output_text)}")


if __name__ == "__main__":
    main()
