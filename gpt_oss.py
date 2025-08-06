###############################################################################
# Patching GPT-OSS Experts and Router
###############################################################################

# if STATIC_MOE:
#     num_experts, intermediate_size, hidden_dim = self.gate_up_proj.shape
#     routing_weights = routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)
#     for expert_idx in range(num_experts):
#         local_gate_up_proj = self.gate_up_proj[expert_idx]
#         local_gate_up_proj_bias = self.gate_up_proj_bias[expert_idx]
#         local_down_proj = self.down_proj[expert_idx]
#         local_down_proj_bias = self.down_proj_bias[expert_idx]
#         local_gate_up = hidden_states @ local_gate_up_proj + local_gate_up_proj_bias
#         local_gate, local_up = local_gate_up[..., ::2], local_gate_up[..., 1::2]
#         local_gate = local_gate.clamp(min=None, max=self.limit)
#         local_up = local_up.clamp(min=-self.limit, max=self.limit)
#         local_glu = local_gate * torch.sigmoid(local_gate * self.alpha)
#         local_up_glu = (local_up + 1) * local_glu
#         local_next_states = local_up_glu @ local_down_proj + local_down_proj_bias
#         local_next_states = local_next_states.view(batch_size, -1, self.hidden_size)
#         local_routing_weights = routing_weights[expert_idx][..., None]
#         local_next_states = local_next_states * local_routing_weights
#         if expert_idx == 0:
#             final_next_states = local_next_states
#         else:
#             final_next_states += local_next_states
#     return final_next_states
        
                


import torch

import torch.nn as nn

from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssPreTrainedModel,
    GptOssExperts,
    GptOssTopKRouter,
)


class GptOssExpertMLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.expert_dim, bias=True)
        self.down_proj = nn.Linear(self.expert_dim, self.hidden_size, bias=True)
        self.alpha = 1.702
        self.limit = 7.0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): (batch_size, seq_len, hidden_size)
        Returns:
            torch.Tensor
        """
        # breakpoint()
        gate_up = self.gate_up_proj(hidden_states)
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        gated_output = (up + 1) * glu
        out = self.down_proj(gated_output)
        return out


from transformers.modeling_utils import TORCH_INIT_FUNCTIONS
import contextlib
from typing import Any


@contextlib.contextmanager
def patch_attr(base: object, attr: str, value: Any):
    """
    Patch the value of an object attribute. Original value is restored upon exit

    :param base: object which has the attribute to patch
    :param attr: name of the the attribute to patch
    :param value: used to replace original value

    Usage:
    >>> from types import SimpleNamespace
    >>> obj = SimpleNamespace()
    >>> with patch_attr(obj, "attribute", "value"):
    ...     assert obj.attribute == "value"
    >>> assert not hasattr(obj, "attribute")
    """
    _sentinel = object()
    original_value = getattr(base, attr, _sentinel)

    setattr(base, attr, value)
    try:
        yield
    finally:
        if original_value is not _sentinel:
            setattr(base, attr, original_value)
        else:
            delattr(base, attr)


@contextlib.contextmanager
def skip_weights_initialize(use_zeros: bool = False):
    """
    Very similar to `transformers.model_utils.no_init_weights`, except that torch.Tensor
    initialization functions are also patched to account for tensors which are
    initialized not on the meta device
    """

    def skip(tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if use_zeros:
            return tensor.fill_(0)
        return tensor

    with contextlib.ExitStack() as stack:
        for name in TORCH_INIT_FUNCTIONS.keys():
            stack.enter_context(patch_attr(torch.nn.init, name, skip))
            stack.enter_context(patch_attr(torch.Tensor, name, skip))
        yield


class PacthedGptOssExperts(torch.nn.ModuleList):
    def __init__(self, config):
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size

        with skip_weights_initialize():
            super().__init__([GptOssExpertMLP(config) for _ in range(self.num_experts)])

    def forward(
        self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): (batch_size, seq_len, hidden_size)
            router_indices (torch.Tensor): (batch_size * token_num, top_k)
            routing_weights (torch.Tensor): (batch_size * token_num, num_experts)
        Returns:
            torch.Tensor
        """
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(
            -1, self.hidden_size
        )  # (num_tokens, hidden_size)
        num_experts = routing_weights.shape[1]
        routing_weights = routing_weights.transpose(0, 1).view(
            num_experts, batch_size, -1
        )
        # logger.trace(f"expert routing weights shape: {routing_weights.shape}")
        for expert_idx in range(num_experts):
            local_expert = self[expert_idx]
            local_next_states = local_expert(hidden_states)
            local_next_states = local_next_states.reshape(
                batch_size, -1, self.hidden_size
            )
            local_routing_weights = routing_weights[expert_idx][..., None]
            # try:
            # logger.trace(f"start routing for expert {expert_idx}, "
            #              f"local_next_states shape: {local_next_states.shape}, "
            #              f"local_routing_weights shape: {local_routing_weights.shape}")
            local_next_states = local_next_states * local_routing_weights
            # except RuntimeError as e:
            #     breakpoint()
            if expert_idx == 0:
                final_next_states = local_next_states
            else:
                final_next_states += local_next_states
        return final_next_states

    @classmethod
    def from_original(cls, config, original_experts: "GptOssExperts"):
        """
        This method allows to copy the weights from the original GptOssExperts
        to the patched version.
        """
        new_experts = cls(config)
        new_experts = new_experts.to(original_experts.gate_up_proj.dtype)
        for expert_idx in range(new_experts.num_experts):
            gate_up_proj_weight = original_experts.gate_up_proj[expert_idx]
            gate_up_proj_bias = original_experts.gate_up_proj_bias[expert_idx]
            down_proj_weight = original_experts.down_proj[expert_idx]
            down_proj_bias = original_experts.down_proj_bias[expert_idx]
            new_experts[expert_idx].gate_up_proj.weight.data.copy_(
                gate_up_proj_weight.t()
            )
            new_experts[expert_idx].gate_up_proj.bias.data.copy_(gate_up_proj_bias)
            new_experts[expert_idx].down_proj.weight.data.copy_(down_proj_weight.t())
            new_experts[expert_idx].down_proj.bias.data.copy_(down_proj_bias)

        return new_experts


class PatchedGptOssTopKRouter(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        # self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        # self.bias = nn.Parameter(torch.empty(self.num_experts))
        self.mod = torch.nn.Linear(
            in_features=self.hidden_dim, out_features=self.num_experts, bias=True
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        # router_logits = F.linear(
        #     hidden_states, self.weight, self.bias
        # )  # (seq_len, num_experts)
        router_logits = self.mod(hidden_states)  # (seq_len, num_experts)
        router_top_value, router_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )  # (seq_len, top_k)
        router_top_value = torch.nn.functional.softmax(
            router_top_value, dim=1, dtype=router_top_value.dtype
        )
        router_scores = torch.zeros_like(router_logits).scatter_(
            1, router_indices, router_top_value
        )
        return router_scores, router_indices

    @classmethod
    def from_original(cls, config, original_router: GptOssTopKRouter):
        """
        This method allows to copy the weights from the original GptOssTopKRouter
        to the patched version.
        """
        new_rounter = cls(config)
        new_rounter.mod.weight.data = new_rounter.mod.weight.data.to(
            original_router.weight.dtype
        )
        new_rounter.mod.bias.data = new_rounter.mod.bias.data.to(
            original_router.bias.dtype
        )
        new_rounter.mod.weight.data.copy_(original_router.weight)
        new_rounter.mod.bias.data.copy_(original_router.bias)
        return new_rounter

from loguru import logger


def _replace_router_with_patched(
    mod: "GptOssPreTrainedModel", config, src_cls, dst_cls
):
    named_children_list = list(mod.named_children())
    for name, layer in named_children_list:
        if isinstance(layer, src_cls):
            new_layer = dst_cls.from_original(config, layer)
            setattr(mod, name, new_layer)
            logger.trace(f"Patched {name} with {new_layer.__class__.__name__}")
        elif isinstance(layer, nn.Module):
            _replace_router_with_patched(layer, config, src_cls, dst_cls)
    return mod


def patching_mod(mod: "GptOssPreTrainedModel"):
    config = mod.config
    # mod = _replace_router_with_patched(
    #     mod, config, GptOssTopKRouter, PatchedGptOssTopKRouter
    # )
    mod = _replace_router_with_patched(mod, config, GptOssExperts, PacthedGptOssExperts)
    return mod


import os

STATIC_MOE = os.environ.get("STATIC_MOE", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
    "y",
    "t",
)


###############################################################################
# Patching GPT-OSS Experts and Router End
###############################################################################

def reset_seed(seed=42):
    import torch
    import random
    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


reset_seed(0)
from transformers import pipeline
import torch

model_id = "openai/gpt-oss-20b"
model_id = "/home/yiliu7/models/openai/gpt-oss-20b"
from transformers import AutoModelForCausalLM, AutoTokenizer


model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
model = patching_mod(model)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
# breakpoint()

# breakpoint()

def simple_gen(model, tokenizer, prompt, max_new_tokens=64):
    with torch.device("cuda"):
        model.eval()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            # top_p=0.95,
            # temperature=0.8,
        )
        decode_res = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(decode_res)


# simple_gen(
#     model, tokenizer, "Hi, I am openai model oss-20b, how to use it?", max_new_tokens=64
# )

from auto_round import AutoRound
# from transformers import AutoModelForCausalLM, AutoTokenizer

# data_type fp8 act_data_type fp8 group_size -1 act_group_size 0
autoround = AutoRound(
    model,
    tokenizer,
    data_type="fp8",
    act_data_type="fp8",
    bits=8,
    group_size=-1,
    act_group_size=0,
    sym=True,
    # Use 0 after https://github.com/intel/auto-round/pull/662
    iters=0,
    seqlen=2,
    act_dynamic=False
    # nsamples=2,
)
# model, qconfig = autoround.quantize()
# autoround.save_quantized(output_dir="quantized_gpt_oss_20b")
autoround.quantize_and_save(output_dir="quantized_gpt_oss_20b")
# breakpoint()
assert model is not None, f"Expected q_model to be not None"

# messages = [
#     {"role": "user", "content": "Hi, explain Ai.."},
# ]


# outputs = pipe(
#     messages,
#     max_new_tokens=64,
# )

# print(outputs[0]["generated_text"][-1])

# Patched
# Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
# Hi, I am openai model oss-20b, how to use it?

# It sounds like you're referring to a large language model, possibly a variant of the GPT (Generative Pre-trained Transformer) family, named "oss-20b." If you're looking to use this model, here are some general steps and considerations you might find helpful:

# ### 1. Accessing the Model

# Hi, I am openai model oss-20b, how to use it?

# It sounds like you're referring to a large language model, possibly a variant of the GPT (Generative Pre-trained Transformer) series, named "oss-20b." If you're looking to use this model, here are some general steps and considerations you might find helpful:

# ### 1. Accessing the Model