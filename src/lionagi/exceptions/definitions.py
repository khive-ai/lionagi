# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

# Base LionAGI Error
class LionError(Exception):
    """Base class for all LionAGI errors."""
    def __init__(self, message: str, **context: Any):
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        context_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        if context_str:
            return f"{self.message} ({context_str})"
        return self.message

# General Errors (from original _errors.py, now inheriting new LionError)
class ItemNotFoundError(LionError):
    def __init__(self, message: str = "Item not found.", **context: Any):
        super().__init__(message, **context)

class ItemExistsError(LionError):
    def __init__(self, message: str = "Item already exists.", **context: Any):
        super().__init__(message, **context)

class IDError(LionError): # Used by protocols.generic.element
    def __init__(self, message: str = "Invalid ID.", **context: Any):
        super().__init__(message, **context)

class RelationError(LionError):
    def __init__(self, message: str = "Error in relationship.", **context: Any):
        super().__init__(message, **context)

class OperationError(LionError):
    def __init__(self, message: str = "Operation failed.", **context: Any):
        super().__init__(message, **context)

class ExecutionError(LionError): # Broader than core_defs.Execution's error field
    def __init__(self, message: str = "Execution failed.", **context: Any):
        super().__init__(message, **context)


# Network/Connection Errors (inspired by pynector's errors.py)
class NetworkError(LionError):
    """Base class for network or connection related errors."""
    def __init__(self, message: str = "A network error occurred.", **context: Any):
        super().__init__(message, **context)

class TransportError(NetworkError):
    def __init__(self, message: str = "Transport operation failed.", **context: Any):
        super().__init__(message, **context)

class NetworkConfigurationError(NetworkError): # Renamed from pynector's ConfigurationError
    def __init__(self, message: str = "Network configuration error.", **context: Any):
        super().__init__(message, **context)

class ConnectionTimeoutError(NetworkError): # Renamed from pynector's TimeoutError
    def __init__(self, message: str = "Connection timed out.", **context: Any):
        super().__init__(message, **context)

class RateLimitError(NetworkError): # From original _errors.py, fits well here
    def __init__(self, message: str = "Rate limit exceeded.", **context: Any):
        super().__init__(message, **context)


# Adapter and Data Processing Errors (inspired by pydapter's exceptions.py)
class AdapterError(LionError):
    """Base exception for adapter related errors."""
    def __init__(self, message: str = "An adapter error occurred.", **context: Any):
        super().__init__(message, **context)

class ValidationError(AdapterError):
    """Exception raised when data validation fails."""
    def __init__(self, message: str = "Data validation failed.", data: Optional[Any] = None, **context: Any):
        super().__init__(message, **context)
        self.data = data

class TypeConversionError(ValidationError):
    """Exception raised when type conversion fails."""
    def __init__(
        self,
        message: str = "Type conversion failed.",
        source_type: Optional[type] = None,
        target_type: Optional[type] = None,
        field_name: Optional[str] = None,
        model_name: Optional[str] = None,
        **context: Any,
    ):
        super().__init__(message, **context)
        self.source_type = source_type
        self.target_type = target_type
        self.field_name = field_name
        self.model_name = model_name

class ParseError(AdapterError):
    """Exception raised when data parsing fails."""
    def __init__(self, message: str = "Data parsing failed.", source: Optional[str] = None, **context: Any):
        super().__init__(message, **context)
        self.source = source

class AdapterConnectionError(AdapterError): # Renamed from pydapter's ConnectionError
    """Exception raised when a connection to a data source via an adapter fails."""
    def __init__(
        self,
        message: str = "Adapter connection failed.",
        adapter: Optional[str] = None,
        url: Optional[str] = None,
        **context: Any,
    ):
        super().__init__(message, **context)
        self.adapter = adapter
        self.url = url

class QueryError(AdapterError):
    """Exception raised when a query to a data source via an adapter fails."""
    def __init__(
        self,
        message: str = "Query failed.",
        query: Optional[Any] = None,
        adapter: Optional[str] = None,
        **context: Any,
    ):
        super().__init__(message, **context)
        self.query = query
        self.adapter = adapter

class ResourceError(AdapterError):
    """Exception raised when a resource (file, database, etc.) cannot be accessed by an adapter."""
    def __init__(self, message: str = "Resource access error.", resource: Optional[str] = None, **context: Any):
        super().__init__(message, **context)
        self.resource = resource

class AdapterConfigurationError(AdapterError): # Renamed from pydapter's ConfigurationError
    """Exception raised when adapter configuration is invalid."""
    def __init__(
        self, message: str = "Invalid adapter configuration.", config: Optional[dict[str, Any]] = None, **context: Any
    ):
        super().__init__(message, **context)
        self.config = config

class AdapterNotFoundError(AdapterError):
    """Exception raised when an adapter is not found."""
    def __init__(self, message: str = "Adapter not found.", obj_key: Optional[str] = None, **context: Any):
        super().__init__(message, **context)
        self.obj_key = obj_key


PYDAPTER_PYTHON_ERRORS = (KeyError, ImportError, AttributeError, ValueError)