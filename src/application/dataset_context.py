from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.services.data.frame_reader import DataFrameReadError, read_dataframe


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


def read_dataframe_cached(cache: dict[str, pd.DataFrame], path: str | Path, *, nrows: int | None = None) -> pd.DataFrame:
    fingerprint = fingerprint_dataset(path)
    full_key = f"{fingerprint.cache_key}:full"
    if full_key not in cache:
        cache[full_key] = read_dataframe(path).copy()
    frame = cache[full_key]
    return frame.head(nrows).copy() if nrows is not None else frame.copy()
