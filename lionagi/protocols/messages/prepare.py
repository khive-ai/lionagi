# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

# Re-export the canonical implementation from the beta message prepare module.
# The implementation lives in lionagi.beta.core.message.prepare_msg and is now
# written against production types.  Importing it here keeps the production
# public API stable.
from lionagi.beta.core.message.prepare_msg import prepare_messages_for_chat

__all__ = ("prepare_messages_for_chat",)
