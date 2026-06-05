from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

CSV_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1251", "latin1")


class DataFrameReadError(ValueError):
    """Raised when a tabular file cannot be read with supported readers."""


def read_dataframe(path: str | Path, *, nrows: int | None = None) -> pd.DataFrame:
    data_path = Path(path)
    suffix = data_path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return _read_csv_with_encoding_policy(data_path, nrows=nrows)
    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(data_path, nrows=nrows)
        frame.attrs["source_format"] = suffix.lstrip(".")
        return frame
    if suffix == ".parquet":
        frame = pd.read_parquet(data_path)
        result = frame.head(nrows) if nrows is not None else frame
        result.attrs["source_format"] = "parquet"
        return result
    raise ValueError(f"Unsupported data format: {suffix}")


def _read_csv_with_encoding_policy(data_path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in CSV_ENCODINGS:
        try:
            frame = pd.read_csv(data_path, nrows=nrows, encoding=encoding)
            frame.attrs["source_format"] = data_path.suffix.lower().lstrip(".") or "csv"
            frame.attrs["source_encoding"] = encoding
            return frame
        except (UnicodeDecodeError, ParserError, EmptyDataError, OSError, ValueError) as exc:
            errors.append(f"{encoding}: {type(exc).__name__}: {exc}")
    joined_errors = " | ".join(errors)
    raise DataFrameReadError(f"Could not read CSV file '{data_path}' with encodings {CSV_ENCODINGS}. {joined_errors}")
