from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_dataframe(path: str | Path, *, nrows: int | None = None) -> pd.DataFrame:
    data_path = Path(path)
    suffix = data_path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(data_path, nrows=nrows)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(data_path, nrows=nrows)
    if suffix == ".parquet":
        df = pd.read_parquet(data_path)
        return df.head(nrows) if nrows is not None else df
    raise ValueError(f"Unsupported data format: {suffix}")
