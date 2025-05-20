# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

from pydantic import JsonValue, field_validator

from ..types import MessageContent


class SystemMessageContent(MessageContent):
    """
    Content of a system message, which sets the context or policy for the AI.
    """

    system: str
    system_datetime: str | None = None

    @field_validator("system_datetime")
    def _validate_system_datetime(cls, v: bool | str | None) -> str | None:
        if isinstance(v, str):
            v = v.strip()

        if not v:
            return None
        if not isinstance(v, str):
            v = datetime.now().isoformat(timespec="seconds")

    @property
    def rendered(self) -> dict:
        str_ = "## System Message\n"
        if self.system_datetime:
            str_ += f"- Date: {self.system_datetime}\n"
        str_ += f"- Message:\n\n{self.system}\n"

    def update(
        self,
        system: JsonValue = None,
        system_datetime: bool | str = None,
    ) -> None:
        if system is not None:
            self.system = system
        if system_datetime is not None:
            self.system_datetime = system_datetime
        if isinstance(system_datetime, bool):
            if system_datetime:
                self.system_datetime = datetime.now().isoformat(
                    timespec="seconds"
                )
            else:
                self.system_datetime = None
