from __future__ import annotations

import json
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
CASES_FILE = PROJECT_ROOT / "tests" / "fixtures" / "visrag_intent_cases.json"


def require_runtime_corpus() -> None:
    if not RUNTIME_CORPUS_FILE.exists():
        pytest.skip(
            "Runtime VisRAG corpus is missing. "
            "Generate it with scripts/rag_corpus/09_export_visrag_runtime_corpus.py first."
        )


def load_cases() -> list[dict[str, Any]]:
    if not CASES_FILE.exists():
        pytest.fail(f"Missing intent cases fixture: {CASES_FILE}")

    payload = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    cases = payload.get("cases")

    if not isinstance(cases, list) or not cases:
        pytest.fail(f"Fixture must contain non-empty `cases` list: {CASES_FILE}")

    return cases


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


def run_case(case: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    rows = case["rows"]
    dataset_path = tmp_path / f"{case['case_id']}.csv"
    pd.DataFrame(rows).to_csv(dataset_path, index=False)

    runtime = make_runtime(tmp_path)
    data_profile = build_data_profile(rows, case["field_roles"])
    query_understanding = QueryUnderstandingResult(
        intent=case["intent"],
        user_goal=case["query"],
        candidate_charts=case["candidate_charts"],
        query_variants=[
            QueryVariant(kind="original", text=case["query"], confidence=1.0),
        ],
        confidence=1.0,
    )
    request_analysis = RequestAnalysisResult(
        selected_fields=case["selected_fields"],
        grounded_fields=case["selected_fields"],
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
        "selected_candidate": selected_candidate,
        "spec": spec_artifact.spec_json,
        "validation": validation,
        "dataset_path": dataset_path,
    }


def assert_field_mapping(
        actual_mapping: dict[str, str],
        expected_mapping: dict[str, str],
) -> None:
    for channel, expected_field in expected_mapping.items():
        assert actual_mapping.get(channel) == expected_field


def assert_encoding(spec: dict[str, Any], expected_encoding: dict[str, Any]) -> None:
    encoding = spec.get("encoding")

    assert isinstance(encoding, dict)

    for channel, expected_channel in expected_encoding.items():
        assert channel in encoding
        actual_channel = encoding[channel]

        assert isinstance(actual_channel, dict)

        if expected_channel.get("field_absent"):
            assert "field" not in actual_channel

        for key, expected_value in expected_channel.items():
            if key == "field_absent":
                continue

            if key == "bin":
                assert "bin" in actual_channel
                continue

            assert actual_channel.get(key) == expected_value


@pytest.mark.parametrize(
    "case",
    load_cases(),
    ids=lambda case: str(case["case_id"]),
)
def test_visrag_intent_quality_real_corpus(case: dict[str, Any], tmp_path: Path) -> None:
    xfail = case.get("xfail")

    if isinstance(xfail, dict):
        pytest.xfail(str(xfail.get("reason") or "known intent-quality gap"))

    result = run_case(case, tmp_path)
    selected_candidate = result["selected_candidate"]
    spec = result["spec"]
    validation = result["validation"]
    expected = case["expected"]

    assert validation.is_valid
    assert spec["data"]["url"] == result["dataset_path"].as_posix()

    assert selected_candidate.chart_family == expected["chart_family"]

    expected_mapping = expected.get("field_mapping")
    if expected_mapping:
        assert_field_mapping(selected_candidate.field_mapping, expected_mapping)

    expected_encoding = expected.get("encoding")
    if expected_encoding:
        assert_encoding(spec, expected_encoding)
