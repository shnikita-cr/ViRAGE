from pathlib import Path

import pandas as pd

from src.domain.models import DataPreparationResult, DataProfile, RequestAnalysisResult
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService


class DataPreparationService(BaseService):
    def invoke(
            self,
            data_path: str,
            data_profile: DataProfile,
            request_analysis: RequestAnalysisResult,
            run_id: str,
            runtime: RuntimeContext,
    ) -> DataPreparationResult:
        path = Path(data_path)
        df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        operations = []
        selected_fields = [field for field in request_analysis.selected_fields if field in df.columns]
        if selected_fields:
            df = df[selected_fields].copy()
            operations.append(f"select_fields:{','.join(selected_fields)}")
        before = len(df)
        df = df.drop_duplicates()
        if len(df) != before:
            operations.append("drop_duplicates")
        for column in [col for col in data_profile.likely_time_columns if col in df.columns]:
            try:
                df[column] = pd.to_datetime(df[column])
                operations.append(f"to_datetime:{column}")
            except Exception:
                pass
        for column in [col for col in data_profile.likely_numeric_columns if col in df.columns]:
            if df[column].isna().any():
                df[column] = df[column].fillna(df[column].median())
                operations.append(f"fill_numeric_median:{column}")
        run_dir = runtime.ensure_run_dir(run_id)
        cleaned_path = run_dir / "cleaned_data.csv"
        df.to_csv(cleaned_path, index=False)
        return DataPreparationResult(output_path=cleaned_path.as_posix(), operations=operations, row_count=int(len(df)),
                                     col_count=int(len(df.columns)))
