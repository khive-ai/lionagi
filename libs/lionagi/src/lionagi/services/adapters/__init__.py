# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Provider adapters for hybrid registry system."""

from .openai_adapter import OpenAIAdapter
from .generic_adapter import GenericJSONAdapter

__all__ = ["OpenAIAdapter", "GenericJSONAdapter"]