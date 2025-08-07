# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from typing_extensions import TypedDict

from lionagi.config import settings
from lionagi.protocols._concepts import Manager
from lionagi.protocols.types import DataLogger

from .imodel import iModel


class iModelConfig(TypedDict, total=False):
    predict: iModel | None
    parse: iModel | None
    interpret: iModel | None
    symbolic: iModel | None
    analysis: iModel | None


class iModelManager(Manager):

    def __init__(
        self,
        api_log_config: dict,
        hook_log_config: dict,
        kw,
    ):
        super().__init__()
        self.registry: dict[str, iModel] = {}
        self._api_dlog: DataLogger | None = DataLogger(
            **(api_log_config or settings.API_LOG_CONFIG)
        )
        self._hook_dlog: DataLogger | None = DataLogger(
            **(hook_log_config or settings.HOOK_LOG_CONFIG)
        )
        if kw:
            for name, model in kw.items():
                self.register_imodel(name, model)

    def __getattr__(self, name: str) -> iModel:
        if name in iModelConfig.__optional_keys__:
            return self.registry.get(name)
        if name in self.registry:
            return self.registry[name]
        return super().__getattr__(name)

    def __setattr__(self, name, value):
        if isinstance(value, iModel):
            self.registry[name] = value
            return
        super().__setattr__(name, value)

    def register_imodel(self, name: str, model: iModel):
        if isinstance(model, iModel):
            self.registry[name] = model
        else:
            raise TypeError("Input model is not an instance of iModel")
