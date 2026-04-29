from __future__ import annotations

import pandas as pd

from src.application.settings import ViRAGESettings
from src.domain.models import DataColumnProfile, DataProfile, QueryUnderstandingResult, RequestAnalysisResult
from src.infrastructure.runtime import RuntimeContext
from src.services.data_preparation import DataPreparationService


def test_data_preparation_preserves_duplicate_rows_by_default(tmp_path):
    data_path = tmp_path / "sales.csv"
    pd.DataFrame({"Region": ["A", "A", "B"], "Sales": [10, 10, 20]}).to_csv(data_path, index=False)
    profile = DataProfile(
        row_count=3,
        col_count=2,
        columns=[
            DataColumnProfile(name="Region", dtype="categorical", missing_ratio=0, unique_count=2),
            DataColumnProfile(name="Sales", dtype="numeric", missing_ratio=0, unique_count=2),
        ],
        likely_numeric_columns=["Sales"],
        likely_categorical_columns=["Region"],
    )
    runtime = RuntimeContext(settings=ViRAGESettings(artifact_root=tmp_path / "artifacts"))
    runtime.current_run_id = "test"

    result = DataPreparationService().invoke(
        str(data_path),
        profile,
        RequestAnalysisResult(selected_fields=["Region", "Sales"]),
        "test",
        runtime,
        QueryUnderstandingResult(intent="sum sales by region"),
    )

    assert result.row_count == 3
    assert "preserve_row_multiplicity" in result.operations
