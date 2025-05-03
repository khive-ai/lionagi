"""Excel adapter built on top of pandas DataFrame adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import pandas as pd

from .adapter import Adapter
from .pd_dataframe_adapter import DataFrameAdapter

T = TypeVar("T")


class ExcelAdapter(Adapter[T]):
    """External representation: Excel *file* path or bytes stream."""

    obj_key = "xlsx"

    @classmethod
    def from_obj(
        cls,
        subj_cls: type[T],
        obj: str | Path | bytes,
        /,
        *,
        many: bool = True,
        sheet_name: str | int = 0,
        **kwargs,
    ):
        if isinstance(obj, bytes):
            import io

            df = pd.read_excel(
                io.BytesIO(obj), sheet_name=sheet_name, **kwargs
            )
        else:
            df = pd.read_excel(obj, sheet_name=sheet_name, **kwargs)
        return DataFrameAdapter.from_obj(subj_cls, df, many=many)

    @classmethod
    def to_obj(
        cls,
        subj: T | list[T],
        /,
        *,
        many: bool = True,
        sheet_name: str = "Sheet1",
        path: str | Path | None = None,
        **kwargs,
    ) -> bytes:
        df = DataFrameAdapter.to_obj(subj, many=many)
        import io

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False, **kwargs)
        if path:
            with open(path, "wb") as f:
                f.write(buffer.getvalue())
        return buffer.getvalue()
