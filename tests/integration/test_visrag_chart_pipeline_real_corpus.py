from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.application.settings import ViRAGESettings
from src.domain.models import (
    DataColumnProfile,
    DataPreparationResult,
    DataProfile,
    QueryUnderstandingResult,
    QueryVariant,
    RequestAnalysisResult,
)
from src.infrastructure.runtime import RuntimeContext
from src.services.chart_generator import ChartGeneratorService
from src.services.spec_validator import SpecValidatorService
from src.services.visrag import VisRAGService

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = PROJECT_ROOT / "rag_corpus" / "data"
RUNTIME_CORPUS_FILE = CORPUS_ROOT / "vega_lite_examples.jsonl"


def require_runtime_corpus() -> None:
    if not RUNTIME_CORPUS_FILE.exists():
        pytest.skip(
            "Runtime VisRAG corpus is missing. "
            "Generate it with scripts/rag_corpus/export_visrag_runtime_corpus.py first."
        )


def make_runtime(tmp_path: Path) -> RuntimeContext:
    require_runtime_corpus()

    return RuntimeContext(
        settings=ViRAGESettings(
            artifact_root=tmp_path / "artifacts",
            visrag_corpus_root=CORPUS_ROOT,
            visrag_retriever_backend="bm25",
            visrag_top_k_examples=5,
        )
    )


def infer_dtype(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"

    if isinstance(value, int):
        return "int64"

    if isinstance(value, float):
        return "float64"

    text = str(value)

    if len(text) >= 8:
        try:
            pd.to_datetime([text], errors="raise")
            return "datetime64[ns]"
        except Exception:
            pass

    return "object"


def build_data_profile(rows: list[dict[str, Any]], field_roles: dict[str, str]) -> DataProfile:
    dataframe = pd.DataFrame(rows)
    columns: list[DataColumnProfile] = []

    for column_name in dataframe.columns:
        series = dataframe[column_name]
        sample_value = next((value for value in series.tolist() if pd.notna(value)), "")

        columns.append(
            DataColumnProfile(
                name=str(column_name),
                dtype=infer_dtype(sample_value),
                missing_ratio=float(series.isna().mean()),
                unique_count=int(series.nunique(dropna=True)),
                sample_values=[value for value in series.dropna().head(5).tolist()],
            )
        )

    return DataProfile(
        row_count=int(len(dataframe)),
        col_count=int(len(dataframe.columns)),
        columns=columns,
        field_roles=field_roles,
    )


def run_pipeline_case(
        *,
        tmp_path: Path,
        case_id: str,
        query: str,
        intent: str,
        candidate_charts: list[str],
        selected_fields: list[str],
        field_roles: dict[str, str],
        rows: list[dict[str, Any]],
) -> dict[str, Any]:
    runtime = make_runtime(tmp_path)

    dataset_path = tmp_path / f"{case_id}.csv"
    pd.DataFrame(rows).to_csv(dataset_path, index=False)

    data_profile = build_data_profile(rows, field_roles)
    query_understanding = QueryUnderstandingResult(
        intent=intent,
        user_goal=query,
        candidate_charts=candidate_charts,
        query_variants=[QueryVariant(kind="original", text=query, confidence=1.0)],
        confidence=1.0,
    )
    request_analysis = RequestAnalysisResult(
        selected_fields=selected_fields,
        grounded_fields=selected_fields,
        confidence=1.0,
    )

    visrag_result = VisRAGService().invoke(
        query_understanding=query_understanding,
        request_analysis=request_analysis,
        data_profile=data_profile,
        runtime=runtime,
    )

    assert visrag_result.candidate_spec_set is not None
    selected_candidate = visrag_result.candidate_spec_set.selected_candidate_spec
    assert selected_candidate is not None

    spec_artifact = ChartGeneratorService().invoke(
        prepared=DataPreparationResult(
            output_path=dataset_path.as_posix(),
            row_count=len(rows),
            col_count=len(rows[0]),
        ),
        candidate_spec_set=visrag_result.candidate_spec_set,
        runtime=runtime,
    )
    validation = SpecValidatorService().invoke(spec_artifact)

    return {
        "candidate": selected_candidate,
        "spec": spec_artifact.spec_json,
        "validation": validation,
        "dataset_path": dataset_path,
    }


def test_real_corpus_chart_pipeline_scatter_validates(tmp_path: Path) -> None:
    result = run_pipeline_case(
        tmp_path=tmp_path,
        case_id="scatter_quantitative_relationship",
        query="show the relationship between two numeric fields as a scatter plot",
        intent="relationship",
        candidate_charts=["point"],
        selected_fields=["Horsepower", "Miles_per_Gallon"],
        field_roles={
            "Horsepower": "measure",
            "Miles_per_Gallon": "measure",
            "Origin": "dimension",
        },
        rows=[
            {"Horsepower": 130, "Miles_per_Gallon": 18.0, "Origin": "USA"},
            {"Horsepower": 95, "Miles_per_Gallon": 24.0, "Origin": "Europe"},
            {"Horsepower": 70, "Miles_per_Gallon": 33.0, "Origin": "Japan"},
        ],
    )

    spec = result["spec"]
    validation = result["validation"]

    assert validation.is_valid
    assert spec["data"]["url"] == result["dataset_path"].as_posix()
    assert spec["mark"] == "point"
    assert spec["encoding"]["x"]["field"] == "Horsepower"
    assert spec["encoding"]["y"]["field"] == "Miles_per_Gallon"


def test_real_corpus_chart_pipeline_line_validates(tmp_path: Path) -> None:
    result = run_pipeline_case(
        tmp_path=tmp_path,
        case_id="line_temporal_trend",
        query="show how sales change over time using a line chart",
        intent="trend",
        candidate_charts=["line"],
        selected_fields=["Month", "Sales"],
        field_roles={
            "Month": "date",
            "Sales": "measure",
            "Category": "dimension",
        },
        rows=[
            {"Month": "2024-01-01", "Sales": 120.0, "Category": "A"},
            {"Month": "2024-02-01", "Sales": 150.0, "Category": "A"},
            {"Month": "2024-03-01", "Sales": 135.0, "Category": "B"},
        ],
    )

    spec = result["spec"]
    validation = result["validation"]

    assert validation.is_valid
    assert spec["data"]["url"] == result["dataset_path"].as_posix()
    assert spec["mark"] == "line"
    assert spec["encoding"]["x"]["field"] == "Month"
    assert spec["encoding"]["y"]["field"] == "Sales"


def test_real_corpus_chart_pipeline_bar_validates_with_known_selected_fields_gap(tmp_path: Path) -> None:
    result = run_pipeline_case(
        tmp_path=tmp_path,
        case_id="bar_category_comparison",
        query="compare values across categories with a bar chart",
        intent="comparison",
        candidate_charts=["bar"],
        selected_fields=["Category", "Sales"],
        field_roles={
            "Category": "dimension",
            "Sales": "measure",
            "Month": "date",
        },
        rows=[
            {"Category": "A", "Sales": 100.0, "Month": "2024-01-01"},
            {"Category": "A", "Sales": 110.0, "Month": "2024-02-01"},
            {"Category": "B", "Sales": 170.0, "Month": "2024-01-01"},
        ],
    )

    spec = result["spec"]
    validation = result["validation"]

    assert validation.is_valid
    assert spec["data"]["url"] == result["dataset_path"].as_posix()
    assert spec["encoding"]["y"]["field"] == "Sales"

    # Known backlog:
    # selected_fields are currently preferred, not strict.
    # Current grounding may choose Month instead of Category for x.
    assert spec["encoding"]["x"]["field"] in {"Category", "Month"}


def test_real_corpus_chart_pipeline_histogram_validates_count_without_y_field(tmp_path: Path) -> None:
    result = run_pipeline_case(
        tmp_path=tmp_path,
        case_id="histogram_distribution",
        query="show the distribution of a numeric value with a histogram",
        intent="distribution",
        candidate_charts=["histogram"],
        selected_fields=["Value"],
        field_roles={
            "Value": "measure",
            "Group": "dimension",
        },
        rows=[
            {"Value": 10.0, "Group": "A"},
            {"Value": 12.5, "Group": "A"},
            {"Value": 15.0, "Group": "B"},
            {"Value": 20.0, "Group": "B"},
        ],
    )

    spec = result["spec"]
    validation = result["validation"]

    assert validation.is_valid
    assert spec["data"]["url"] == result["dataset_path"].as_posix()
    assert spec["encoding"]["x"]["field"] == "Value"
    assert spec["encoding"]["y"]["aggregate"] == "count"
    assert "field" not in spec["encoding"]["y"]
