# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0


class LionError(Exception):
    pass


class ItemNotFoundError(LionError):
    pass


class ItemExistsError(LionError):
    pass


class IDError(LionError):
    pass


class RelationError(LionError):
    pass


class RateLimitError(LionError):
    pass


class OperationError(LionError):
    pass


class ExecutionError(LionError):
    pass


class MissingAdapterError(Exception):
    """Raised when an adapter is not found for a given type."""


class ClassNotFoundError(Exception):
    """Raised when a class cannot be found by name in the registry or dynamically."""


class EdgeError(Exception):
    """Raised when there is an issue with a edge relation"""


class EdgeConditionError(Exception):
    """Raised when there is an issue with a edge condition"""


class MissingEdgeConditionError(EdgeConditionError):
    """Raised when a condition is not found in the registry or dynamically."""


class EdgeEntityError(Exception):
    """Raised when there is an issue with a edge or its head/tail entities"""
