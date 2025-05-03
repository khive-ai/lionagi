"""Adapter that converts between pandas.DataFrame and domain objects."""

from __future__ import annotations

from typing import TypeVar

import pandas as pd
from pydantic import BaseModel

from ..adapter import Adapter

T = TypeVar("T", bound=BaseModel)


class DataFrameAdapter(Adapter[T]):
    obj_key = "pd.DataFrame"

    @classmethod
    def from_obj(
        cls,
        subj_cls: type[T],
        obj: pd.DataFrame,
        /,
        *,
        many: bool = True,
        **kwargs,
    ):
        if many:
            return [
                subj_cls.model_validate(rec, **kwargs)
                for rec in obj.to_dict(orient="records")
            ]
        # single row
        return subj_cls.model_validate(obj.iloc[0].to_dict(), **kwargs)

    @classmethod
    def to_obj(
        cls,
        subj: T | list[T],
        /,
        *,
        many: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        import pandas as pd

        if many:
            items = subj if isinstance(subj, list) else [subj]
        else:
            items = [subj]
        data = [s.model_dump() for s in items]
        return pd.DataFrame(data)
