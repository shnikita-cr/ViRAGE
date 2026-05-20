from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.errors import ParserError


CSV_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1251", "latin1")


class DataFrameReadError(ValueError):
    """Raised when a tabular file cannot be read with supported readers."""


def read_dataframe(path: str | Path, *, nrows: int | None = None) -> pd.DataFrame:
    data_path = Path(path)
    suffix = data_path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return _read_csv_with_encoding_fallback(data_path, nrows=nrows)
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(data_path, nrows=nrows)
        df.attrs["source_format"] = suffix.lstrip(".")
        return df
    if suffix == ".parquet":
        df = pd.read_parquet(data_path)
        result = df.head(nrows) if nrows is not None else df
        result.attrs["source_format"] = "parquet"
        return result
    raise ValueError(f"Unsupported data format: {suffix}")


def _read_csv_with_encoding_fallback(data_path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in CSV_ENCODINGS:
        try:
            df = pd.read_csv(data_path, nrows=nrows, encoding=encoding)
            df.attrs["source_format"] = data_path.suffix.lower().lstrip(".") or "csv"
            df.attrs["source_encoding"] = encoding
            return df
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {type(exc).__name__}: {exc}")
        except ParserError as exc:
            # A parser error usually means the separator/quoting is invalid, not the encoding.
            # Still collect the failed attempt because latin1 can decode any byte sequence and
            # could otherwise hide the original context.
            errors.append(f"{encoding}: {type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - keep a compact diagnostic for all reader failures.
            errors.append(f"{encoding}: {type(exc).__name__}: {exc}")

    joined_errors = " | ".join(errors)
    raise DataFrameReadError(f"Could not read CSV file '{data_path}' with encodings {CSV_ENCODINGS}. {joined_errors}")
