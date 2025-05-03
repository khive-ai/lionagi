"""Adapter that converts pandas.Series <-> single domain object."""

from __future__ import annotations

from typing import TypeVar

import pandas as pd
from pydantic import BaseModel

from ..adapter import Adapter

T = TypeVar("T", bound=BaseModel)


class SeriesAdapter(Adapter[T]):
    obj_key = "pd.Series"

    @classmethod
    def from_obj(
        cls,
        subj_cls: type[T],
        obj: pd.Series,
        /,
        *,
        many: bool = False,
        **kwargs,
    ):
        if many:
            raise ValueError("SeriesAdapter supports only single object.")
        return subj_cls.model_validate(obj.to_dict(), **kwargs)

    @classmethod
    def to_obj(
        cls,
        subj: T,
        /,
        *,
        many: bool = False,
        **kwargs,
    ) -> pd.Series:
        if many:
            raise ValueError("SeriesAdapter supports only single object.")
        return pd.Series(subj.model_dump())
