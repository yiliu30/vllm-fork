# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class INCLinearScheme(ABC):
    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError

    @abstractmethod
    def create_weights(
        self,
        layer: "torch.nn.Module",
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: "torch.dtype",
        **extra_weight_attrs,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: "torch.nn.Module") -> None:
        raise NotImplementedError

    @abstractmethod
    def apply_weights(
        self,
        layer: "torch.nn.Module",
        x: "torch.Tensor",
        bias: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        raise NotImplementedError
