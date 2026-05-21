from __future__ import annotations

from src.application.settings import ViRAGESettings
from src.domain.models import CandidateSpec, CandidateSpecSet, DataPreparationResult
from src.infrastructure.runtime import RuntimeContext
from src.services.chart_generator import ChartGeneratorService


def test_chart_generator_maps_original_fields_to_safe_prepared_fields(tmp_path):
    candidate = CandidateSpec(
        spec_id="test:unsafe-fields",
        chart_family="bar",
        summary="Compare metric by region.",
        spec_template={
            "mark": "bar",
            "encoding": {
                "x": {"field": "Region.Name", "type": "nominal"},
                "y": {"field": "Metric Value (%)", "type": "quantitative", "aggregate": "mean"},
            },
        },
    )
    artifact = ChartGeneratorService().invoke(
        prepared=DataPreparationResult(
            output_path="prepared.csv",
            row_count=2,
            col_count=2,
            column_name_map={"Region.Name": "Region_Name", "Metric Value (%)": "Metric_Value"},
            reverse_column_name_map={"Region_Name": "Region.Name", "Metric_Value": "Metric Value (%)"},
            original_columns=["Region.Name", "Metric Value (%)"],
            safe_columns=["Region_Name", "Metric_Value"],
            renamed_column_count=2,
        ),
        candidate_spec_set=CandidateSpecSet(candidate_specs=[candidate], selected_candidate_spec=candidate),
        runtime=RuntimeContext(
            settings=ViRAGESettings(artifact_root=tmp_path / "artifacts", spec_generation_backend="template")),
    )

    encoding = artifact.spec_json["encoding"]
    assert encoding["x"]["field"] == "Region_Name"
    assert encoding["y"]["field"] == "Metric_Value"
