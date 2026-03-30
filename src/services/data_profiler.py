from pathlib import Path

import pandas as pd
from src.domain.models import DataColumnProfile, DataProfile
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService


class DataProfilerService(BaseService):
    def invoke(self, data_path: str, runtime: RuntimeContext) -> DataProfile:
        path = Path(data_path)
        df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        categorical = [c for c in df.columns if c not in numeric]
        time_like = [c for c in categorical if any(t in c.lower() for t in ["date", "time", "year", "month", "day"])]
        columns = [
            DataColumnProfile(
                name=column,
                dtype=str(df[column].dtype),
                missing_ratio=float(df[column].isna().mean()),
                unique_count=int(df[column].nunique(dropna=True)),
            )
            for column in df.columns
        ]
        notes = []
        if df.isna().any().any():
            notes.append("Dataset contains missing values.")
        if df.duplicated().any():
            notes.append("Dataset contains duplicated rows.")
        if not numeric:
            notes.append("No numeric columns detected; chart options may be limited.")
        return DataProfile(
            row_count=int(len(df)),
            col_count=int(len(df.columns)),
            columns=columns,
            likely_numeric_columns=numeric,
            likely_categorical_columns=categorical,
            likely_time_columns=time_like,
            quality_notes=notes,
        )
