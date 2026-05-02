# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from lionagi.service.connections.provider_config import ProviderConfig
from lionagi.service.connections.registry import EndpointType


class NvidiaNimConfigs(ProviderConfig, Enum):

    CHAT = (
        "chat/completions",
        ["chat"],
        EndpointType.API,
        None,
        "https://integrate.api.nvidia.com/v1",
        "bearer",
    )
    EMBED = (
        "embeddings",
        ["embed"],
        EndpointType.API,
        None,
        "https://integrate.api.nvidia.com/v1",
        "bearer",
    )


NvidiaNimConfigs._PROVIDER = "nvidia_nim"
NvidiaNimConfigs._PROVIDER_ALIASES = ["nvidia", "nim"]

__all__ = ("NvidiaNimConfigs",)
