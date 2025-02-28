###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from vllm.attention.backends.mla.utils import MLACommonImpl
import vllm_hpu_extension.kernels as kernels
import vllm_hpu_extension.ops as ops
from vllm_hpu_extension.flags import enabled_flags
from vllm_hpu_extension.utils import (Matmul, ModuleFusedSDPA, Softmax,
                                      VLLMKVCache)

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType, AttentionLayer)
from vllm.attention.backends.utils import CommonAttentionState
from vllm.attention.ops.hpu_paged_attn import (HPUPagedAttention,
                                               HPUPagedAttentionMetadata)
from vllm.logger import init_logger

logger = init_logger(__name__)


class HPUAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "HPU_ATTN"

    @staticmethod
    def get_impl_cls() -> Type["HPUAttentionImpl"]:
        return HPUAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return HPUAttentionMetadata

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return HPUPagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                    num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dsts: torch.Tensor,
    ) -> None:
        HPUPagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dsts)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dsts: torch.Tensor,
    ) -> None:
        HPUPagedAttention.copy_blocks(kv_caches, src_to_dsts)


class HPUMLAAttentionBackend(HPUAttentionBackend):
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (num_blocks, block_size, head_size//9*1), (num_blocks, block_size, head_size//9*8)
    
    @staticmethod
    def get_impl_cls() -> Type["HPUAttentionImpl"]:
        return HPUMLAImpl

    @staticmethod
    def get_name() -> str:
        return "HPU_MLA"

def flat_pa_mla(query, key_cache, value_cache, block_list, block_mapping,
            block_bias, block_scales, block_groups, scale, matmul_qk_op,
            matmul_av_op, batch2block_matmul_op, block2batch_matmul_op,
            keys_fetch_func, values_fetch_func):
    batch_size = query.size(0)
    q_heads = query.size(1)
    kv_heads = key_cache.size(2)

    query = ops.batch2block(scale * query, block_mapping, batch2block_matmul_op).unsqueeze(-2)
    key = keys_fetch_func(key_cache, block_list).transpose(1, 2)
    value = values_fetch_func(value_cache, block_list).transpose(1, 2)
    # get concat key
    key = torch.concat((value, key), dim=-1)
    block_bias = block_bias.view(key.size(0), 1, 1, -1)
    if kv_heads != q_heads:
        block_bias = block_bias.unsqueeze(1)
        query = query.unflatten(1, (kv_heads, -1))
        key = key.unflatten(1, (kv_heads, 1))
        value = value.unflatten(1, (kv_heads, 1))
        key = key.transpose(3, 4)
    else:
        key = key.transpose(2, 3)

    attn = matmul_qk_op(query, key)
    attn = attn + block_bias
    attn = ops.pipelined_pa(attn, value, block_groups, block_mapping, block_scales=block_scales,
                        batch_size=batch_size, matmul_av_op=matmul_av_op,
                        batch2block_matmul_op=batch2block_matmul_op, block2batch_matmul_op=block2batch_matmul_op)
    attn = ops.block2batch(attn, block_mapping, block2batch_matmul_op)
    attn = attn.squeeze(-2)
    if kv_heads != q_heads:
        attn = attn.flatten(1, 2)
    return attn

@dataclass
class HPUAttentionMetadata(HPUPagedAttentionMetadata, AttentionMetadata):
    """Metadata for HPUAttentionbackend."""
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    attn_bias: Optional[torch.Tensor]
    seq_lens_tensor: Optional[torch.Tensor]
    context_lens_tensor: Optional[torch.Tensor]
    input_positions: torch.Tensor
    seq_lens: Optional[List[int]] = None
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None
    cross_block_indices: Optional[torch.Tensor] = None
    cross_block_offsets: Optional[torch.Tensor] = None
    cross_block_list: Optional[torch.Tensor] = None
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_mapping: Optional[torch.Tensor] = None
    cross_block_groups: Optional[torch.Tensor] = None
    cross_block_scales: Optional[torch.Tensor] = None
    cross_block_usage: Optional[torch.Tensor] = None
    cross_attn_bias: Optional[torch.Tensor] = None
    

class HPUMLAImpl(MLACommonImpl[HPUAttentionMetadata], torch.nn.Module):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[List[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            blocksparse_params: Optional[Dict[str, Any]],
            logits_soft_cap: Optional[float],
            attn_type: str,
            # MLA Specific Arguments
            **kwargs) -> None:
        torch.nn.Module.__init__(self)
        MLACommonImpl.__init__(self, num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         blocksparse_params, logits_soft_cap, attn_type,
                         **kwargs)
        self.matmul_qk = Matmul()
        self.softmax = Softmax()
        self.matmul_av = Matmul()
        self.batch2block_matmul = Matmul()
        self.block2batch_matmul = Matmul()
        self.latent_cache_k = VLLMKVCache()
        self.latent_cache_v = VLLMKVCache()
        HPUFusedSDPA = kernels.fsdpa()
        self.fused_scaled_dot_product_attention = None if HPUFusedSDPA is None \
            else ModuleFusedSDPA(HPUFusedSDPA)

        unsupported_features = [
            alibi_slopes, sliding_window, blocksparse_params, logits_soft_cap
        ]
        if any(unsupported_features):
            raise NotImplementedError(
                "TritonMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, blocksparse_params, "
                "logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "TritonMLAImpl")

    def forward(
        self,
        layer: AttentionLayer,
        hidden_states_or_q_c: torch.Tensor,  # query in unified attn
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: HPUAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if output is not None:
            raise NotImplementedError(
                "output is not yet supported for MLAImplBase")

        batch_size = hidden_states_or_q_c.shape[0]

        is_prefill = attn_metadata.is_prompt

        k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)

        # Restore head dim (for rotary embedding)
        # k_pe = k_pe.unsqueeze(1)
        assert hasattr(attn_metadata, "input_positions"), f"attn meta: {attn_metadata}"

        if not is_prefill:
            q_nope = self._q_proj_and_k_up_proj(hidden_states_or_q_c)
            q_pe = torch.matmul(hidden_states_or_q_c, self.W_QR)\
                .view(-1, self.num_heads, self.qk_rope_head_dim)
            input_positions = attn_metadata.input_positions.view(-1)
            q_pe, k_pe = \
                self.rotary_emb(input_positions, q_pe, k_pe)
        else:
            q = self.q_proj(hidden_states_or_q_c)[0]\
                .view(-1, self.num_heads, self.qk_head_dim)
            
            q_pe = q[..., self.qk_nope_head_dim:]

            input_positions = attn_metadata.input_positions.view(-1)
            # TODO(lucas): there must be a nicer way to write this line
            q[..., self.qk_nope_head_dim:], k_pe = \
                self.rotary_emb(input_positions, q_pe, k_pe)
        
        block_indices = attn_metadata.block_indices
        block_offsets = attn_metadata.block_offsets

        latent_vec_k = torch.concat(
                (k_c_normed, k_pe.view(batch_size, -1, self.qk_rope_head_dim)), dim=-1)
        # assert layer._k_scale == 0, f"got _k_scale={layer._k_scale}"
        latent_vec_k = latent_vec_k.view(-1, self.qk_rope_head_dim + self.kv_lora_rank)
        #latent_vec_v = k_c_normed.view(-1, self.kv_lora_rank)
        if is_prefill:
            latent_vec_k = latent_vec_k.unflatten(0, (block_indices.size(0), -1))
            #latent_vec_v = latent_vec_v.unflatten(0, (block_indices.size(0), -1))
        # print("latent_vec", latent_vec.shape)


        # write the latent and rope to kv cache
        if kv_cache is not None and len(kv_cache) == 2:
            # print(f"k cache shape: {kv_cache[0].shape}")
            # print(f"v cache shape: {kv_cache[1].shape}")
            # print(f"latent vec k shape: {latent_vec_k.shape}")
            # print(f"latent vec v shape: {latent_vec_v.shape}")
            latent_vec_v = latent_vec_k[..., :self.kv_lora_rank]
            latent_vec_k = latent_vec_k[..., self.kv_lora_rank:]
            k_cache = self.latent_cache_k(latent_vec_k, kv_cache[0], block_indices,
                                        block_offsets)
            v_cache = self.latent_cache_v(latent_vec_v, kv_cache[1], block_indices,
                                        block_offsets)
            kv_cache = (k_cache, v_cache)

#        if torch.distributed.get_rank() == 0:
#            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#            if kv_cache is not None and len(kv_cache) == 2:
#                print("latent_vec_k: " + str(latent_vec_k.shape))
#                print("latent_vec_v: " + str(latent_vec_v.shape))
#                print("k cache: " + str(k_cache.shape) + " v cache: " + str(v_cache.shape))


        if is_prefill:
            return self._forward_prefill(q, k_c_normed, k_pe, attn_metadata, batch_size)
        else:
            return self._forward_decode(q_nope, q_pe, kv_cache, attn_metadata, batch_size)
    
    def _forward_prefill(
        self,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: HPUAttentionMetadata,
        batch_size: int
    ) -> torch.Tensor:
        kv_nope = self.kv_b_proj(k_c_normed)[0]\
            .view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_nope\
            .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)

        # For MLA the v head dim is smaller than qk head dim so we pad out
        # v with 0s to match the qk head dim
        v_padded = torch.nn.functional.pad(v, [0, q.shape[-1] - v.shape[-1]],
                                           value=0)
        q = q.view(batch_size, -1, self.num_heads, self.qk_head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.qk_head_dim)
        v_padded = v_padded.view(batch_size, -1, self.num_heads, self.qk_head_dim)
#        print("q shape: " + str(q.shape) + " k shape: " + str(k.shape) + " v shape: " + str(v_padded.shape) + "  original v shape: " + str(v.shape))
        out = ops.prompt_attention(
                    q,
                    k,
                    v_padded,
                    attn_bias=None,
                    p=0.0,
                    scale=self.scale,
                    matmul_qk_op=self.matmul_qk,
                    softmax_op=self.softmax,
                    matmul_av_op=self.matmul_av,
                    valid_seq_lengths=attn_metadata.seq_lens_tensor,
                    fsdpa_op=self.fused_scaled_dot_product_attention,
                )
        attn_output = out\
            .view(batch_size, -1, self.num_heads, q.shape[-1])[..., :v.shape[-1]]\
                .reshape(batch_size, -1, self.num_heads * v.shape[-1])

        return self.o_proj(attn_output)[0]
    
    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: HPUAttentionMetadata,
        batch_size: int
    ) -> torch.Tensor:
        q = torch.cat([q_nope, q_pe], dim=-1)
        kv_c_and_k_pe_cache = kv_cache[0].unsqueeze(2)
        kv_c_cache = kv_cache[1].unsqueeze(2)

        output = flat_pa_mla(
            query=q,
            key_cache=kv_c_and_k_pe_cache,
            value_cache=kv_c_cache,
            block_list=attn_metadata.block_list,
            block_mapping=attn_metadata.block_mapping,
            block_bias=attn_metadata.attn_bias,
            block_scales=attn_metadata.block_scales,
            block_groups=attn_metadata.block_groups,
            scale=self.scale,
            matmul_qk_op=self.matmul_qk,
            matmul_av_op=self.matmul_av,
            batch2block_matmul_op=self.batch2block_matmul,
            block2batch_matmul_op=self.block2batch_matmul,
            keys_fetch_func=self.latent_cache_k.fetch_from_cache,
            values_fetch_func=self.latent_cache_v.fetch_from_cache)
        output = output.view(batch_size, 1, -1)
        result = self._v_up_proj_and_o_proj(output)
        result = result.view(batch_size, 1, -1)
        return result


class HPUAttentionImpl(AttentionImpl, torch.nn.Module):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:
    |<----------------- num_decode_tokens ------------------>|
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        max_seq_len: int = 4096,
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super(AttentionImpl, self).__init__()
        self.kv_cache_dtype = kv_cache_dtype
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.matmul_qk = Matmul()
        self.softmax = Softmax()
        self.matmul_av = Matmul()
        self.batch2block_matmul = Matmul()
        self.block2batch_matmul = Matmul()
        self.k_cache = VLLMKVCache()
        self.v_cache = VLLMKVCache()
        HPUFusedSDPA = kernels.fsdpa()
        self.fused_scaled_dot_product_attention = None if HPUFusedSDPA is None \
            else ModuleFusedSDPA(HPUFusedSDPA)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        self.alibi_slopes = alibi_slopes
        if alibi_slopes is not None:
            alibi_slopes_tensor = torch.tensor(alibi_slopes,
                                               dtype=torch.bfloat16)
            self.alibi_slopes = alibi_slopes_tensor
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.prefill_use_fusedsdpa = "fsdpa" in enabled_flags()
        if self.prefill_use_fusedsdpa:
            assert alibi_slopes is None, \
                'Prefill with FusedSDPA not supported with alibi slopes!'

        # suppored_head_sizes = HPUPagedAttention.get_supported_head_sizes()
        # if head_size not in suppored_head_sizes:
        #     raise ValueError(
        #         f"Head size {head_size} is not supported by PagedAttention. "
        #         f"Supported head sizes are: {suppored_head_sizes}.")

        self.attn_type = attn_type
        if (self.attn_type != AttentionType.DECODER
                and self.attn_type != AttentionType.ENCODER_DECODER):
            raise NotImplementedError("Encoder self-attention "
                                      "is not implemented for "
                                      "HPUAttentionImpl")

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: HPUAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with xFormers and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        if self.attn_type == AttentionType.ENCODER_DECODER:
            return self.forward_encoder_decoder(
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                k_scale=layer._k_scale_float,
                v_scale=layer._k_scale_float,
            )

        batch_size, seq_len, hidden_size = query.shape
        _, seq_len_kv, _ = key.shape

        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        block_indices = attn_metadata.block_indices
        block_offsets = attn_metadata.block_offsets
        if attn_metadata.is_prompt:
            key = key.unflatten(0, (block_indices.size(0), -1))
            value = value.unflatten(0, (block_indices.size(0), -1))
        if kv_cache is not None and isinstance(kv_cache, tuple):
            key_cache, value_cache = HPUPagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            key_cache = self.k_cache(key, key_cache, block_indices,
                                     block_offsets)
            value_cache = self.v_cache(value, value_cache, block_indices,
                                       block_offsets)

        if attn_metadata.is_prompt:
            # Prompt run.
            query_shape = (batch_size, seq_len, self.num_heads, self.head_size)
            kv_shape = (batch_size, seq_len_kv, self.num_kv_heads,
                        self.head_size)
            if attn_metadata is None or attn_metadata.block_list is None:
                if not self.prefill_use_fusedsdpa:
                    # TODO: move this outside of model
                    assert attn_metadata.attn_bias is not None, \
                            'attn_bias must be set before calling model.forward'
                    attn_bias = attn_metadata.attn_bias
                    if self.alibi_slopes is not None:
                        position_bias = _make_alibi_bias(
                            self.alibi_slopes, self.num_kv_heads,
                            attn_bias.dtype, attn_bias.shape[-1])
                        attn_bias = attn_bias.tile(
                            (1, self.num_kv_heads, 1, 1))
                        attn_bias.add_(position_bias)
                else:
                    attn_bias = None

                out = ops.prompt_attention(
                    query.view(query_shape),
                    key.view(kv_shape),
                    value.view(kv_shape),
                    attn_bias=attn_bias,
                    p=0.0,
                    scale=self.scale,
                    matmul_qk_op=self.matmul_qk,
                    softmax_op=self.softmax,
                    matmul_av_op=self.matmul_av,
                    valid_seq_lengths=attn_metadata.seq_lens_tensor,
                    fsdpa_op=self.fused_scaled_dot_product_attention,
                )
            else:
                # TODO: enable FusedSDPA
                out = HPUPagedAttention.forward_prefix(
                    query=query.view(query_shape),
                    key=key.view(kv_shape),
                    value=value.view(kv_shape),
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_list=attn_metadata.block_list,
                    attn_bias=attn_metadata.attn_bias,
                    scale=self.scale,
                    matmul_qk_op=self.matmul_qk,
                    matmul_av_op=self.matmul_av,
                    softmax_op=self.softmax,
                    keys_fetch_func=self.k_cache.fetch_from_cache,
                    values_fetch_func=self.v_cache.fetch_from_cache)
            output = out.reshape(batch_size, seq_len, hidden_size)
        else:
            # Decoding run.
            output = HPUPagedAttention.forward_decode(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                block_list=attn_metadata.block_list,
                block_mapping=attn_metadata.block_mapping,
                block_bias=attn_metadata.attn_bias,
                block_scales=attn_metadata.block_scales,
                block_groups=attn_metadata.block_groups,
                scale=self.scale,
                matmul_qk_op=self.matmul_qk,
                matmul_av_op=self.matmul_av,
                batch2block_matmul_op=self.batch2block_matmul,
                block2batch_matmul_op=self.block2batch_matmul,
                keys_fetch_func=self.k_cache.fetch_from_cache,
                values_fetch_func=self.v_cache.fetch_from_cache)
        # Reshape the output tensor.
        return output.view(batch_size, seq_len, hidden_size)

    def forward_encoder_decoder(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: HPUAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass with xFormers and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        batch_size, hidden_size = query.shape

        if attn_metadata.is_prompt:
            batch_size = attn_metadata.num_prefills
            batched_tokens, _ = query.shape
            batched_kv_tokens, _, _ = key.shape
            assert batch_size > 0, (
                "In prefill stage the num_prefills should be > 0")
            assert batched_tokens % batch_size == 0
            assert batched_kv_tokens % batch_size == 0
            seq_len = batched_tokens // batch_size

        query = query.view(-1, self.num_heads, self.head_size)
        if key is not None:
            assert value is not None
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
        else:
            assert value is None

        block_indices = attn_metadata.cross_block_indices
        block_offsets = attn_metadata.cross_block_offsets
        if kv_cache is not None and isinstance(kv_cache, tuple):
            key_cache, value_cache = HPUPagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            if (key is not None) and (value is not None):
                # During cross-attention decode, key & value will be None,
                # we don't need to cache them.
                key_cache = self.k_cache(key, key_cache, block_indices,
                                         block_offsets)
                value_cache = self.v_cache(value, value_cache, block_indices,
                                           block_offsets)

        if attn_metadata.is_prompt:
            # Prompt run.
            batch_size = attn_metadata.num_prefills

            query_shape = (batch_size, -1, self.num_heads, self.head_size)
            kv_shape = (batch_size, -1, self.num_kv_heads, self.head_size)
            # Just a workaround, to make ops.prompt_attention go into the
            # torch ops assembly path.
            # TODO: add new prompt_attention op in vllm_hpu_extension
            # which calls FusedSDPA with causal = False.
            attn_bias = torch.zeros((batch_size, 1, 1, 1),
                                    device=query.device,
                                    dtype=torch.bool)
            out = ops.prompt_attention(
                query.view(query_shape),
                key.view(kv_shape),
                value.view(kv_shape),
                attn_bias=attn_bias,
                p=0.0,
                scale=self.scale,
                matmul_qk_op=self.matmul_qk,
                softmax_op=self.softmax,
                matmul_av_op=self.matmul_av,
            )
            output = out.reshape(batch_size, seq_len, hidden_size)
        else:
            # Enc/dec cross-attention KVs match encoder sequence length;
            # cross-attention utilizes special "cross" block tables
            block_list = attn_metadata.cross_block_list
            block_mapping = attn_metadata.cross_block_mapping
            block_scales = attn_metadata.cross_block_scales
            block_groups = attn_metadata.cross_block_groups
            attn_bias = attn_metadata.cross_attn_bias
            # Decoding run.
            output = HPUPagedAttention.forward_decode(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                block_list=block_list,
                block_mapping=block_mapping,
                block_bias=attn_bias,
                block_scales=block_scales,
                block_groups=block_groups,
                scale=self.scale,
                matmul_qk_op=self.matmul_qk,
                matmul_av_op=self.matmul_av,
                batch2block_matmul_op=self.batch2block_matmul,
                block2batch_matmul_op=self.block2batch_matmul,
                keys_fetch_func=self.k_cache.fetch_from_cache,
                values_fetch_func=self.v_cache.fetch_from_cache)
        # Reshape the output tensor.
        return output.view(batch_size, -1, hidden_size)


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    dtype: torch.dtype,
    seq_len: int,
) -> torch.Tensor:
    bias = torch.arange(seq_len, dtype=dtype)
    # NOTE(zhuohan): HF uses
    #     `bias = bias[None, :].repeat(seq_len, 1)`
    # here. We find that both biases give the same results, but
    # the bias below more accurately follows the original ALiBi
    # paper.
    # Calculate a matrix where each element represents ith element- jth
    # element.
    bias = bias[None, :] - bias[:, None]

    padded_len = (seq_len + 7) // 8 * 8
    num_heads = alibi_slopes.shape[0]
    bias = torch.empty(
        1,  # batch size
        num_heads,
        seq_len,
        padded_len,
        device=alibi_slopes.device,
        dtype=dtype,
    )[:, :, :, :seq_len].copy_(bias)
    bias.mul_(alibi_slopes[:, None, None])
    if num_heads != num_kv_heads:
        bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))
    return bias
