from __future__ import annotations

import pandas as pd

from src.domain.models import DataProfile, QueryUnderstandingResult, RequestAnalysisResult
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


def test_data_preparation_preserves_row_multiplicity_for_distribution_queries(tmp_path, runtime) -> None:
    data_path = tmp_path / "iris_species.csv"
    pd.DataFrame({"Species": ["setosa", "setosa", "virginica", "versicolor", "setosa"]}).to_csv(data_path, index=False)
    profile = DataProfile(
        row_count=5,
        col_count=1,
        columns=[],
        likely_numeric_columns=[],
        likely_categorical_columns=["Species"],
        likely_time_columns=[],
        quality_notes=[],
    )
    understanding = QueryUnderstandingResult(
        intent="Analyse Species distribution",
        requested_operations=["count", "distribution"],
        candidate_charts=["bar"],
        constraints=[],
        confidence=0.9,
        task_type="distribution",
        user_goal="understand class balance",
        analysis_goal="count category frequency",
    )
    request_analysis = RequestAnalysisResult(selected_fields=["Species"], grounded_fields=["Species"])

    result = DataPreparationService().invoke(
        data_path=data_path.as_posix(),
        data_profile=profile,
        request_analysis=request_analysis,
        run_id="prepare-iris",
        runtime=runtime,
        query_understanding=understanding,
    )

    cleaned = pd.read_csv(result.output_path)
    assert result.row_count == 5
    assert "preserve_row_multiplicity" in result.operations
    assert "drop_duplicates" not in result.operations
    assert cleaned["Species"].tolist().count("setosa") == 3
