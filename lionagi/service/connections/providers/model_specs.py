# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""
OpenAI Model Specifications and Context Window Mappings

This module contains comprehensive specifications for OpenAI models including
context windows, capabilities, and status information.

Updated: 2025-08-07
Sources: OpenAI Platform documentation, OpenAI API documentation
"""

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel


class ModelCapability(BaseModel):
    """Model capability specification."""
    
    text: bool = True
    vision: bool = False
    audio: bool = False
    reasoning: bool = False
    function_calling: bool = True
    structured_outputs: bool = False


class ModelSpec(BaseModel):
    """Comprehensive model specification."""
    
    context_window: int
    max_output_tokens: Optional[int] = None
    total_context: Optional[int] = None
    capabilities: ModelCapability
    status: Literal["active", "deprecated", "preview"] = "active"
    release_date: Optional[str] = None
    deprecation_date: Optional[str] = None
    retirement_date: Optional[str] = None
    notes: Optional[str] = None


# OpenAI Model Specifications
# Based on research as of 2025-08-07
OPENAI_MODEL_SPECS: Dict[str, ModelSpec] = {
    
    # GPT-5 Series (Released August 7, 2025)
    "gpt-5": ModelSpec(
        context_window=272000,
        max_output_tokens=128000,
        total_context=400000,
        capabilities=ModelCapability(
            text=True, vision=True, reasoning=True, 
            function_calling=True, structured_outputs=True
        ),
        status="active",
        release_date="2025-08-07",
        notes="Advanced reasoning with vision capabilities"
    ),
    "gpt-5-mini": ModelSpec(
        context_window=272000,
        max_output_tokens=128000,
        total_context=400000,
        capabilities=ModelCapability(
            text=True, vision=True, reasoning=True,
            function_calling=True, structured_outputs=True
        ),
        status="active",
        release_date="2025-08-07",
        notes="Faster and more cost-effective GPT-5 variant"
    ),
    "gpt-5-nano": ModelSpec(
        context_window=272000,
        max_output_tokens=128000,
        total_context=400000,
        capabilities=ModelCapability(
            text=True, vision=True, reasoning=True,
            function_calling=True, structured_outputs=True
        ),
        status="active",
        release_date="2025-08-07",
        notes="Most cost-effective GPT-5 variant"
    ),
    
    # GPT-4.1 Series (1 Million Token Context)
    "gpt-4.1": ModelSpec(
        context_window=1000000,
        capabilities=ModelCapability(
            text=True, vision=True, function_calling=True, structured_outputs=True
        ),
        status="active",
        release_date="2025-01-31",
        notes="1 million token context window breakthrough"
    ),
    "gpt-4.1-mini": ModelSpec(
        context_window=1000000,
        capabilities=ModelCapability(
            text=True, vision=True, function_calling=True, structured_outputs=True
        ),
        status="active",
        release_date="2025-01-31",
        notes="Fast and affordable with near GPT-4.1 performance"
    ),
    "gpt-4.1-nano": ModelSpec(
        context_window=1000000,
        capabilities=ModelCapability(
            text=True, vision=True, function_calling=True, structured_outputs=True
        ),
        status="active",
        release_date="2025-01-31",
        notes="Most cost-effective 1M context variant"
    ),
    
    # O-Series Reasoning Models
    "o1": ModelSpec(
        context_window=200000,
        max_output_tokens=100000,
        capabilities=ModelCapability(
            text=True, reasoning=True, function_calling=True, structured_outputs=True
        ),
        status="active",
        release_date="2024-12-17",
        notes="Uses internal reasoning tokens for complex problem solving"
    ),
    "o1-2024-12-17": ModelSpec(
        context_window=200000,
        max_output_tokens=100000,
        capabilities=ModelCapability(
            text=True, reasoning=True, function_calling=True, structured_outputs=True
        ),
        status="active",
        release_date="2024-12-17"
    ),
    "o1-preview": ModelSpec(
        context_window=128000,
        max_output_tokens=32768,
        capabilities=ModelCapability(text=True, reasoning=True),
        status="active",
        release_date="2024-09-12",
        notes="Preview version of O1 reasoning capabilities"
    ),
    "o1-preview-2024-09-12": ModelSpec(
        context_window=128000,
        max_output_tokens=32768,
        capabilities=ModelCapability(text=True, reasoning=True),
        status="active",
        release_date="2024-09-12"
    ),
    "o1-mini": ModelSpec(
        context_window=128000,
        max_output_tokens=65536,
        capabilities=ModelCapability(text=True, reasoning=True, structured_outputs=True),
        status="active",
        release_date="2024-09-12",
        notes="Faster, more cost-effective reasoning model"
    ),
    "o1-mini-2024-09-12": ModelSpec(
        context_window=128000,
        max_output_tokens=65536,
        capabilities=ModelCapability(text=True, reasoning=True, structured_outputs=True),
        status="active",
        release_date="2024-09-12"
    ),
    "o3": ModelSpec(
        context_window=200000,
        max_output_tokens=100000,
        capabilities=ModelCapability(
            text=True, vision=True, reasoning=True, 
            function_calling=True, structured_outputs=True
        ),
        status="active",
        release_date="2025-04-16",
        notes="Reasoning with vision, 20% fewer errors than o1"
    ),
    "o3-2025-04-16": ModelSpec(
        context_window=200000,
        max_output_tokens=100000,
        capabilities=ModelCapability(
            text=True, vision=True, reasoning=True,
            function_calling=True, structured_outputs=True
        ),
        status="active",
        release_date="2025-04-16"
    ),
    "o3-mini": ModelSpec(
        context_window=200000,
        max_output_tokens=100000,
        capabilities=ModelCapability(
            text=True, vision=True, reasoning=True,
            function_calling=True, structured_outputs=True
        ),
        status="active",
        release_date="2025-01-31",
        notes="Three reasoning levels: low, medium, high"
    ),
    "o3-mini-2025-01-31": ModelSpec(
        context_window=200000,
        max_output_tokens=100000,
        capabilities=ModelCapability(
            text=True, vision=True, reasoning=True,
            function_calling=True, structured_outputs=True
        ),
        status="active",
        release_date="2025-01-31"
    ),
    "o4-mini": ModelSpec(
        context_window=200000,
        max_output_tokens=100000,
        capabilities=ModelCapability(
            text=True, vision=True, reasoning=True,
            function_calling=True, structured_outputs=True
        ),
        status="active",
        release_date="2025-04-16",
        notes="Optimized for fast, cost-efficient reasoning"
    ),
    "o4-mini-2025-04-16": ModelSpec(
        context_window=200000,
        max_output_tokens=100000,
        capabilities=ModelCapability(
            text=True, vision=True, reasoning=True,
            function_calling=True, structured_outputs=True
        ),
        status="active",
        release_date="2025-04-16"
    ),
    
    # GPT-4o Series (Omni Models)
    "gpt-4o": ModelSpec(
        context_window=128000,
        max_output_tokens=16384,
        capabilities=ModelCapability(
            text=True, vision=True, audio=True,
            function_calling=True, structured_outputs=True
        ),
        status="active",
        release_date="2024-11-20",
        notes="Current flagship multimodal model"
    ),
    "gpt-4o-2024-11-20": ModelSpec(
        context_window=128000,
        max_output_tokens=16384,
        capabilities=ModelCapability(
            text=True, vision=True, audio=True,
            function_calling=True, structured_outputs=True
        ),
        status="active",
        release_date="2024-11-20"
    ),
    "gpt-4o-2024-08-06": ModelSpec(
        context_window=128000,
        max_output_tokens=16384,
        capabilities=ModelCapability(
            text=True, vision=True, function_calling=True, structured_outputs=True
        ),
        status="active",
        release_date="2024-08-06",
        notes="First model with structured outputs support"
    ),
    "gpt-4o-2024-05-13": ModelSpec(
        context_window=128000,
        max_output_tokens=4096,
        capabilities=ModelCapability(text=True, vision=True, function_calling=True),
        status="active",
        release_date="2024-05-13"
    ),
    "gpt-4o-mini": ModelSpec(
        context_window=128000,
        max_output_tokens=16384,
        capabilities=ModelCapability(
            text=True, vision=True, audio=True,
            function_calling=True, structured_outputs=True
        ),
        status="active",
        release_date="2024-07-18",
        notes="Cost-effective GPT-4o variant"
    ),
    "gpt-4o-mini-2024-07-18": ModelSpec(
        context_window=128000,
        max_output_tokens=16384,
        capabilities=ModelCapability(
            text=True, vision=True, audio=True,
            function_calling=True, structured_outputs=True
        ),
        status="active",
        release_date="2024-07-18"
    ),
    
    # GPT-4o Preview Models
    "gpt-4o-audio-preview": ModelSpec(
        context_window=128000,
        capabilities=ModelCapability(text=True, vision=True, audio=True),
        status="preview",
        release_date="2024-10-01",
        notes="Real-time speech input/output conversations"
    ),
    "gpt-4o-audio-preview-2024-10-01": ModelSpec(
        context_window=128000,
        capabilities=ModelCapability(text=True, vision=True, audio=True),
        status="preview",
        release_date="2024-10-01"
    ),
    "gpt-4o-audio-preview-2024-12-17": ModelSpec(
        context_window=128000,
        capabilities=ModelCapability(text=True, vision=True, audio=True),
        status="preview",
        release_date="2024-12-17"
    ),
    "gpt-4o-mini-audio-preview": ModelSpec(
        context_window=128000,
        capabilities=ModelCapability(text=True, vision=True, audio=True),
        status="preview",
        release_date="2024-12-17",
        notes="Cost-effective audio preview model"
    ),
    "gpt-4o-mini-audio-preview-2024-12-17": ModelSpec(
        context_window=128000,
        capabilities=ModelCapability(text=True, vision=True, audio=True),
        status="preview",
        release_date="2024-12-17"
    ),
    
    # Special GPT-4o Models
    "chatgpt-4o-latest": ModelSpec(
        context_window=128000,
        capabilities=ModelCapability(
            text=True, vision=True, audio=True,
            function_calling=True, structured_outputs=True
        ),
        status="active",
        release_date="2024-11-20",
        notes="Points to the latest GPT-4o version"
    ),
    
    # GPT-4 Turbo Series
    "gpt-4-turbo": ModelSpec(
        context_window=128000,
        capabilities=ModelCapability(text=True, vision=True, function_calling=True),
        status="active",
        release_date="2024-04-09",
        notes="Most capable GPT-4 model"
    ),
    "gpt-4-turbo-2024-04-09": ModelSpec(
        context_window=128000,
        capabilities=ModelCapability(text=True, vision=True, function_calling=True),
        status="active",
        release_date="2024-04-09"
    ),
    "gpt-4-0125-preview": ModelSpec(
        context_window=128000,
        capabilities=ModelCapability(text=True, function_calling=True),
        status="active",
        release_date="2024-01-25"
    ),
    "gpt-4-turbo-preview": ModelSpec(
        context_window=128000,
        capabilities=ModelCapability(text=True, function_calling=True),
        status="active",
        release_date="2024-01-25"
    ),
    "gpt-4-1106-preview": ModelSpec(
        context_window=128000,
        capabilities=ModelCapability(text=True, function_calling=True),
        status="active",
        release_date="2023-11-06"
    ),
    "gpt-4-vision-preview": ModelSpec(
        context_window=128000,
        capabilities=ModelCapability(text=True, vision=True, function_calling=True),
        status="active",
        release_date="2023-11-06"
    ),
    
    # Legacy GPT-4 Models (Deprecated)
    "gpt-4": ModelSpec(
        context_window=8192,
        capabilities=ModelCapability(text=True, function_calling=True),
        status="deprecated",
        release_date="2023-03-14",
        deprecation_date="2024-10-01",
        retirement_date="2025-06-06",
        notes="Original GPT-4, being phased out"
    ),
    "gpt-4-0314": ModelSpec(
        context_window=8192,
        capabilities=ModelCapability(text=True),
        status="deprecated",
        release_date="2023-03-14",
        deprecation_date="2024-10-01",
        retirement_date="2025-06-06"
    ),
    "gpt-4-0613": ModelSpec(
        context_window=8192,
        capabilities=ModelCapability(text=True, function_calling=True),
        status="deprecated",
        release_date="2023-06-13",
        deprecation_date="2024-10-01",
        retirement_date="2025-06-06"
    ),
    "gpt-4-32k": ModelSpec(
        context_window=32768,
        capabilities=ModelCapability(text=True, function_calling=True),
        status="deprecated",
        release_date="2023-03-14",
        deprecation_date="2024-10-01",
        retirement_date="2025-06-06",
        notes="Extended context version of GPT-4"
    ),
    "gpt-4-32k-0314": ModelSpec(
        context_window=32768,
        capabilities=ModelCapability(text=True),
        status="deprecated",
        release_date="2023-03-14",
        deprecation_date="2024-10-01",
        retirement_date="2025-06-06"
    ),
    "gpt-4-32k-0613": ModelSpec(
        context_window=32768,
        capabilities=ModelCapability(text=True, function_calling=True),
        status="deprecated",
        release_date="2023-06-13",
        deprecation_date="2024-10-01",
        retirement_date="2025-06-06"
    ),
    
    # GPT-3.5 Series (Deprecated)
    "gpt-3.5-turbo": ModelSpec(
        context_window=16385,
        capabilities=ModelCapability(text=True, function_calling=True),
        status="deprecated",
        release_date="2023-03-01",
        deprecation_date="2024-10-01",
        notes="Legacy model, use GPT-4o-mini instead"
    ),
    "gpt-3.5-turbo-16k": ModelSpec(
        context_window=16385,
        capabilities=ModelCapability(text=True, function_calling=True),
        status="deprecated",
        release_date="2023-06-13",
        deprecation_date="2024-10-01"
    ),
    "gpt-3.5-turbo-0301": ModelSpec(
        context_window=4097,
        capabilities=ModelCapability(text=True),
        status="deprecated",
        release_date="2023-03-01",
        deprecation_date="2024-10-01"
    ),
    "gpt-3.5-turbo-0613": ModelSpec(
        context_window=4097,
        capabilities=ModelCapability(text=True, function_calling=True),
        status="deprecated",
        release_date="2023-06-13",
        deprecation_date="2024-10-01"
    ),
    "gpt-3.5-turbo-1106": ModelSpec(
        context_window=16385,
        capabilities=ModelCapability(text=True, function_calling=True),
        status="deprecated",
        release_date="2023-11-06",
        deprecation_date="2024-10-01"
    ),
    "gpt-3.5-turbo-0125": ModelSpec(
        context_window=16385,
        capabilities=ModelCapability(text=True, function_calling=True),
        status="deprecated",
        release_date="2024-01-25",
        deprecation_date="2024-10-01"
    ),
    "gpt-3.5-turbo-16k-0613": ModelSpec(
        context_window=16385,
        capabilities=ModelCapability(text=True, function_calling=True),
        status="deprecated",
        release_date="2023-06-13",
        deprecation_date="2024-10-01"
    ),
}


def get_model_context_window(model_name: str) -> int:
    """
    Get the context window size for a given model.
    
    Args:
        model_name: The model identifier
        
    Returns:
        Context window size in tokens
        
    Raises:
        ValueError: If model is not found
    """
    spec = OPENAI_MODEL_SPECS.get(model_name)
    if not spec:
        # Return a reasonable default for unknown models
        return 128000
    return spec.context_window


def get_model_capabilities(model_name: str) -> ModelCapability:
    """
    Get the capabilities for a given model.
    
    Args:
        model_name: The model identifier
        
    Returns:
        Model capabilities
        
    Raises:
        ValueError: If model is not found
    """
    spec = OPENAI_MODEL_SPECS.get(model_name)
    if not spec:
        # Return basic text capabilities for unknown models
        return ModelCapability()
    return spec.capabilities


def is_reasoning_model(model_name: str) -> bool:
    """Check if a model has reasoning capabilities."""
    capabilities = get_model_capabilities(model_name)
    return capabilities.reasoning


def is_deprecated_model(model_name: str) -> bool:
    """Check if a model is deprecated."""
    spec = OPENAI_MODEL_SPECS.get(model_name)
    if not spec:
        return False
    return spec.status == "deprecated"


def get_active_models() -> List[str]:
    """Get list of active (non-deprecated) models."""
    return [
        model for model, spec in OPENAI_MODEL_SPECS.items()
        if spec.status == "active"
    ]


def get_models_by_capability(capability: str) -> List[str]:
    """
    Get models that have a specific capability.
    
    Args:
        capability: One of 'text', 'vision', 'audio', 'reasoning', 
                   'function_calling', 'structured_outputs'
    
    Returns:
        List of model names with the capability
    """
    models = []
    for model_name, spec in OPENAI_MODEL_SPECS.items():
        if getattr(spec.capabilities, capability, False):
            models.append(model_name)
    return models


__all__ = [
    "ModelCapability",
    "ModelSpec", 
    "OPENAI_MODEL_SPECS",
    "get_model_context_window",
    "get_model_capabilities", 
    "is_reasoning_model",
    "is_deprecated_model",
    "get_active_models",
    "get_models_by_capability",
]