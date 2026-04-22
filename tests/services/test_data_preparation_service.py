from __future__ import annotations

import pandas as pd

from src.domain.models import DataProfile, RequestAnalysisResult
from src.services.data_preparation import DataPreparationService


def test_data_preparation_service_creates_cleaned_csv(tmp_path, runtime) -> None:
    data_path = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "sales": [10, 10, None],
            "region": ["A", "A", "B"],
        }
    ).to_csv(data_path, index=False)
    profile = DataProfile(
        row_count=3,
        col_count=3,
        columns=[],
        likely_numeric_columns=["sales"],
        likely_categorical_columns=["region"],
        likely_time_columns=["date"],
        quality_notes=[],
    )
    service = DataPreparationService()

    result = service.invoke(
        data_path=data_path.as_posix(),
        data_profile=profile,
        request_analysis=RequestAnalysisResult(selected_fields=["date", "sales", "region"]),
        run_id="prepare-data",
        runtime=runtime,
    )

    cleaned = pd.read_csv(result.output_path)
    assert result.row_count == 2
    assert "drop_duplicates" in result.operations
    assert "fill_numeric_median:sales" in result.operations
    assert cleaned["sales"].isna().sum() == 0
