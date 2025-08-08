"""
OpenAI Model Names extracted from generated models.

This module provides lists of allowed model names for different OpenAI services,
extracted from the auto-generated openai_models.py file.
"""

from typing import Literal, get_args

# Manually define the chat models from the ChatModel class in openai_models.py
# These are extracted from the Literal type definition
CHAT_MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-2025-08-07",
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano-2025-08-07",
    "gpt-5-chat-latest",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano-2025-04-14",
    "o4-mini",
    "o4-mini-2025-04-16",
    "o3",
    "o3-2025-04-16",
    "o3-mini",
    "o3-mini-2025-01-31",
    "o1",
    "o1-2024-12-17",
    "o1-preview",
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-mini-2024-09-12",
    "gpt-4o",
    "gpt-4o-2024-11-20",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
    "gpt-4o-audio-preview",
    "gpt-4o-audio-preview-2024-10-01",
    "gpt-4o-audio-preview-2024-12-17",
    "gpt-4o-audio-preview-2025-06-03",
    "gpt-4o-mini-audio-preview",
    "gpt-4o-mini-audio-preview-2024-12-17",
    "gpt-4o-search-preview",
    "gpt-4o-mini-search-preview",
    "gpt-4o-search-preview-2025-03-11",
    "gpt-4o-mini-search-preview-2025-03-11",
    "chatgpt-4o-latest",
    "codex-mini-latest",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-0125-preview",
    "gpt-4-turbo-preview",
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-16k-0613",
]

# Reasoning models (o1, o3, o4 series)
# Note: Add o1-pro models that may not be in the generated list yet
ADDITIONAL_REASONING_MODELS = [
    "o1-pro",
    "o1-pro-2025-03-19",
    "o3-pro",
    "o3-pro-2025-06-10",
]

REASONING_MODELS = [
    model
    for model in CHAT_MODELS
    if model.startswith(("o1", "o1-", "o3", "o3-", "o4", "o4-", "gpt-5"))
] + ADDITIONAL_REASONING_MODELS

# GPT models (excluding reasoning models)
GPT_MODELS = [
    model
    for model in CHAT_MODELS
    if model.startswith("gpt")
    or model.startswith("chatgpt")
    or model.startswith("codex")
]

# Embedding models
EMBEDDING_MODELS = [
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large",
]

# Audio models
AUDIO_MODELS = {
    "tts": ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"],
    "transcription": [
        "whisper-1",
        "gpt-4o-transcribe",
        "gpt-4o-mini-transcribe",
    ],
}

# Image models
IMAGE_MODELS = ["dall-e-2", "dall-e-3", "gpt-image-1"]

# Moderation models
MODERATION_MODELS = ["text-moderation-latest", "text-moderation-stable"]


def is_reasoning_model(model: str) -> bool:
    """Check if a model is a reasoning model (o1/o3/o4 series)."""
    return model in REASONING_MODELS


def is_valid_chat_model(model: str) -> bool:
    """Check if a model is a valid chat model."""
    return model in CHAT_MODELS


def is_valid_embedding_model(model: str) -> bool:
    """Check if a model is a valid embedding model."""
    return model in EMBEDDING_MODELS


def get_model_category(model: str) -> str:
    """Get the category of a model."""
    if model in REASONING_MODELS:
        return "reasoning"
    elif model in GPT_MODELS:
        return "gpt"
    elif model in EMBEDDING_MODELS:
        return "embedding"
    elif model in AUDIO_MODELS["tts"]:
        return "tts"
    elif model in AUDIO_MODELS["transcription"]:
        return "transcription"
    elif model in IMAGE_MODELS:
        return "image"
    elif model in MODERATION_MODELS:
        return "moderation"
    else:
        return "unknown"


def validate_model(model: str, category: str = None) -> bool:
    """
    Validate if a model is valid, optionally checking category.

    Args:
        model: The model name to validate
        category: Optional category to check against ('chat', 'embedding', etc.)

    Returns:
        True if model is valid (and in the specified category if provided)
    """
    if category == "chat":
        return model in CHAT_MODELS or model in ADDITIONAL_REASONING_MODELS
    elif category == "reasoning":
        return is_reasoning_model(model)
    elif category == "embedding":
        return is_valid_embedding_model(model)
    elif category:
        return get_model_category(model) == category
    else:
        # Check if model exists in any category
        return get_model_category(model) != "unknown"


# Export all model lists
__all__ = [
    "CHAT_MODELS",
    "REASONING_MODELS",
    "GPT_MODELS",
    "EMBEDDING_MODELS",
    "AUDIO_MODELS",
    "IMAGE_MODELS",
    "MODERATION_MODELS",
    "is_reasoning_model",
    "is_valid_chat_model",
    "is_valid_embedding_model",
    "get_model_category",
    "validate_model",
]
