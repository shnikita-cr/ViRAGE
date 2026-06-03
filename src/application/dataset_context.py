from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.errors import ParserError

CSV_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1251", "latin1")


class DataFrameReadError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetFingerprint:
    path: str
    exists: bool
    size_bytes: int
    mtime_ns: int
    sha256: str

    @property
    def cache_key(self) -> str:
        return f"{self.path}:{self.mtime_ns}:{self.size_bytes}:{self.sha256}"


@dataclass
class DatasetContext:
    fingerprint: DatasetFingerprint
    dataframe: pd.DataFrame | None = None
    profile: object | None = None
    prepared_path: str | None = None


def fingerprint_dataset(path: str | Path) -> DatasetFingerprint:
    data_path = Path(path)
    if not data_path.exists():
        return DatasetFingerprint(data_path.as_posix(), False, 0, 0, "missing")
    stat = data_path.stat()
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    return DatasetFingerprint(data_path.as_posix(), True, stat.st_size, stat.st_mtime_ns, digest)


def read_dataframe_cached(cache: dict[str, pd.DataFrame], path: str | Path, *,
                          nrows: int | None = None) -> pd.DataFrame:
    fingerprint = fingerprint_dataset(path)
    full_key = fingerprint.cache_key + ":full"
    if full_key in cache:
        df = cache[full_key]
        return df.head(nrows).copy() if nrows is not None else df.copy()
    if nrows is None:
        df = _read_dataframe(path)
        cache[full_key] = df.copy()
        return df
    partial_key = fingerprint.cache_key + f":nrows={nrows}"
    if partial_key not in cache:
        cache[partial_key] = _read_dataframe(path, nrows=nrows).copy()
    return cache[partial_key].copy()


def _read_dataframe(path: str | Path, *, nrows: int | None = None) -> pd.DataFrame:
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
            errors.append(f"{encoding}: {type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{encoding}: {type(exc).__name__}: {exc}")
    raise DataFrameReadError(
        f"Could not read CSV file '{data_path}' with encodings {CSV_ENCODINGS}. {' | '.join(errors)}"
    )
