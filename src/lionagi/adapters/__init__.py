from .adapter import Adapter
from .json_adapter import JsonAdapter    # This is now the Pydantic-centric version
from .toml_adapter import TomlAdapter    # This is now the Pydantic-centric version
from .pydantic_csv_adapter import CsvAdapter # This is the Pydantic-centric CSV adapter

# Pandas specific adapters
from .pandas_.csv_adapter import CSVFileAdapter
from .pandas_.excel_adapter import ExcelFileAdapter
from .pandas_.pd_dataframe_adapter import PandasDataFrameAdapter
from .pandas_.pd_series_adapter import PandasSeriesAdapter

__all__ = [
    "Adapter",
    "JsonAdapter",
    "TomlAdapter",
    "CsvAdapter",  # Pydantic CSV
    "CSVFileAdapter",  # Pandas CSV
    "ExcelFileAdapter",
    "PandasDataFrameAdapter",
    "PandasSeriesAdapter",
]
