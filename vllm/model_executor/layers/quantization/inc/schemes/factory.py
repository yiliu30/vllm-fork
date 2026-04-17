# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..resolver import INCLayerConfig
    from .base import INCScheme


_SCHEME_REGISTRY: list[type["INCScheme"]] = []


def register_scheme(cls: type["INCScheme"]) -> type["INCScheme"]:
    _SCHEME_REGISTRY.append(cls)
    return cls


def resolve_scheme(layer_config: "INCLayerConfig") -> "INCScheme":
    for scheme_cls in _SCHEME_REGISTRY:
        if scheme_cls.can_handle(layer_config):
            return scheme_cls()

    raise NotImplementedError(
        f"No INC scheme found for layer config: {layer_config}"
    )
