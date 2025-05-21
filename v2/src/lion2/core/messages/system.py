# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

from pydantic import field_validator

from lion2.core.types import MessageContent


class SystemContent(MessageContent):
    """
    Content of a system message, which sets the context or policy for the AI.
    """

    system: str
    system_datetime: str | None = None

    @field_validator("system_datetime", mode="before")
    def _validate_system_datetime(cls, v: bool | str | None) -> str | None:
        if v is None:
            return None
        if isinstance(v, bool):
            if v is True:
                return datetime.now().isoformat(timespec="seconds")
            else:
                return None
        if isinstance(v, str):
            v_stripped = v.strip()
            if not v_stripped:
                return None
            return v_stripped

        # Pydantic might catch this earlier based on type hints, but defensive check:
        raise TypeError(
            f"Unexpected type for system_datetime: {type(v)}. Expected bool, str, or None."
        )

    @property
    def rendered(self) -> str:
        """Human-readable message content"""
        
        str_ = "## System Message\n"
        if self.system_datetime:
            str_ += f"- Date: {self.system_datetime}\n"
        str_ += f"- Message:\n\n{self.system}\n"
        return str_
    
    def update(
        self,
        system: str | None = None,
        system_datetime: bool | str | None = None,
    ) -> None:
        if system is not None:
            self.system = system

        # Check if system_datetime was passed as an argument
        if ("system_datetime" in locals()):  
            self.system_datetime = type(self)._validate_system_datetime(
                system_datetime
            )
