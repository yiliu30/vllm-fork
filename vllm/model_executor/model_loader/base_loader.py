# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from vllm.config import LoadConfig, ModelConfig, VllmConfig
from vllm.model_executor.model_loader.utils import (
    initialize_model, process_weights_after_loading, set_default_torch_dtype)


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, model: nn.Module,
                     model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows 
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError

    def load_model(self, vllm_config: VllmConfig,
                   model_config: ModelConfig) -> nn.Module:
        """Load a model with the given configurations."""
        device_config = vllm_config.device_config
        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config,
                                         model_config=model_config)
                
            # Quantization does not happen in `load_weights` but after it
            self.load_weights(model, model_config)
            process_weights_after_loading(model, model_config, target_device)
            print_model_state_dict(model)
        return model.eval()



def print_model_state_dict(model):
    print("\n" + "="*100)
    print(f"{'Model State Dictionary':^100}")
    print("="*100)

    # 打印表头
    print(f"{'Parameter Name':<{60}} | {'Shape':<20} | {'Dtype':<20} | {'Device'}")
    print("-" * (60 + 20 + 20 + 10))

    # 打印每个参数信息
    for name, param in model.state_dict().items():
        param_shape = str(tuple(param.shape))
        param_dtype = str(param.dtype).replace("torch.", "")
        param_device = str(param.device)

        print(f"{name:<{60}} | {param_shape:<20} | {param_dtype:<20} | {param_device}")