from __future__ import annotations

import pandas as pd

from src.application.config.settings import ViRAGESettings
from src.domain.models import DataColumnProfile, DataProfile, QueryRequestAnalysisResult
from src.infrastructure.runtime import RuntimeContext
from src.services.data.preparation.data_preparation import DataPreparationService


def test_data_preparation_preserves_duplicate_rows_by_default(tmp_path):
    data_path = tmp_path / "sales.csv"
    pd.DataFrame({"Region": ["A", "A", "B"], "Sales": [10, 10, 20]}).to_csv(data_path, index=False)
    profile = DataProfile(
        row_count=3,
        col_count=2,
        columns=[
            DataColumnProfile(name="Region", dtype="categorical", role="dimension", missing_ratio=0, unique_count=2),
            DataColumnProfile(name="Sales", dtype="numeric", role="measure", missing_ratio=0, unique_count=2),
        ],
    )
    runtime = RuntimeContext(settings=ViRAGESettings(artifact_root=tmp_path / "artifacts"))
    runtime.current_run_id = "test"

    result = DataPreparationService().invoke(
        str(data_path),
        profile,
        QueryRequestAnalysisResult(normalized_query="sum sales by region", selected_fields=["Region", "Sales"]),
        "test",
        runtime,
    )

    assert result.row_count == 3
    assert "preserve_row_multiplicity" in result.operations


def test_data_preparation_writes_safe_column_mapping_and_csv_columns(tmp_path):
    data_path = tmp_path / "unsafe.csv"
    pd.DataFrame(
        {
            "Metric Value (%)": [1.0, 2.0],
            "Region.Name": ["A", "B"],
        }
    ).to_csv(data_path, index=False)
    profile = DataProfile(
        row_count=2,
        col_count=2,
        columns=[
            DataColumnProfile(
                name="Metric Value (%)",
                safe_name="Metric_Value",
                dtype="numeric",
                role="measure",
                missing_ratio=0,
                unique_count=2,
            ),
            DataColumnProfile(
                name="Region.Name",
                safe_name="Region_Name",
                dtype="categorical",
                role="dimension",
                missing_ratio=0,
                unique_count=2,
            ),
        ],
    )
    runtime = RuntimeContext(settings=ViRAGESettings(artifact_root=tmp_path / "artifacts"))
    runtime.current_run_id = "safe-map"
    runtime.reset_artifact_indices(run_id="safe-map")

    result = DataPreparationService().invoke(
        str(data_path),
        profile,
        QueryRequestAnalysisResult(
            normalized_query="compare metric by region",
            selected_fields=["Metric Value (%)", "Region.Name"],
        ),
        "safe-map",
        runtime,
    )

    prepared = pd.read_csv(result.output_path)
    assert list(prepared.columns) == ["Metric_Value", "Region_Name"]
    assert result.column_name_map == {"Metric Value (%)": "Metric_Value", "Region.Name": "Region_Name"}
    assert result.reverse_column_name_map == {"Metric_Value": "Metric Value (%)", "Region_Name": "Region.Name"}
    assert result.renamed_column_count == 2
    assert any("safe_column_mapping:2" == op for op in result.operations)
    assert result.column_name_map
