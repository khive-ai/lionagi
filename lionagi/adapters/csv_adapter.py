"""CSV adapter built on top of pandas DataFrame adapter."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pandas as pd

from .adapter import Adapter
from .pd_dataframe_adapter import DataFrameAdapter

T = TypeVar("T")


class CsvAdapter(Adapter[T]):
    """External representation: CSV *file* path or CSV text."""

    obj_key = "csv"

    @classmethod
    def from_obj(
        cls,
        subj_cls: type[T],
        obj: str | Path,
        /,
        *,
        many: bool = True,
        **kwargs,
    ):
        # If obj is a path, read file; else assume CSV string
        if isinstance(obj, (str, Path)) and Path(obj).exists():
            df = pd.read_csv(obj, **kwargs)
        else:
            df = pd.read_csv(pd.compat.StringIO(str(obj)), **kwargs)
        return DataFrameAdapter.from_obj(subj_cls, df, many=many)

    @classmethod
    def to_obj(
        cls,
        subj: T | list[T],
        /,
        *,
        many: bool = True,
        **kwargs,
    ) -> str:
        df = DataFrameAdapter.to_obj(subj, many=many)
        return df.to_csv(index=False, **kwargs)
