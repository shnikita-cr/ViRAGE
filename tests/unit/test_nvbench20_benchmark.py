from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.benchmark.datasets.datasets import load_benchmark_cases
from src.benchmark.datasets.nvbench20 import convert_nvbench20
from src.benchmark.evaluation.evaluator import VegaChatBenchmarkEvaluator


def test_convert_nvbench20_exports_single_table_cases(tmp_path: Path) -> None:
    source = tmp_path / "train.parquet"
    csv_dir = tmp_path / "database_csv"
    output_dir = tmp_path / "out"
    csv_dir.mkdir()
    (csv_dir / "demo@items.csv").write_text("category,value\nA,1\nB,2\n", encoding="utf-8")
    frame = pd.DataFrame([
        {
            "nl_query": "show value by category",
            "table_schema": json.dumps({
                "table_columns": ["category", "value"],
                "column_examples": {"category": ["A", "B"], "value": [1, 2]},
            }),
            "steps": json.dumps({"leak": "hidden"}),
            "gold_answer": json.dumps([
                {"mark": "bar", "encoding": {"x": {"field": "category"}, "y": {"field": "value"}}},
            ]),
        }
    ])
    frame.to_parquet(source, index=False)

    report = convert_nvbench20(
        input_path=source,
        database_csv_dir=csv_dir,
        output_dir=output_dir,
        limit=None,
        seed=42,
        single_table_only=True,
    )

    assert report["exported_single_table_cases"] == 1
    case = json.loads((output_dir / "cases.jsonl").read_text(encoding="utf-8").strip())
    assert case["query"] == "show value by category"
    assert case["metadata"]["steps_hidden"] is True
    assert "steps" not in case
    assert case["reference_specs"][0]["encoding"]["y"]["type"] == "quantitative"


def test_load_benchmark_cases_reads_reference_specs(tmp_path: Path) -> None:
    data_path = tmp_path / "data.csv"
    data_path.write_text("category,value\nA,1\n", encoding="utf-8")
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(json.dumps({
        "case_id": "case-a",
        "query": "show value",
        "data_path": data_path.name,
        "reference_specs": [
            {"mark": "line", "encoding": {"x": {"field": "category"}, "y": {"field": "value"}}},
            {"mark": "bar", "encoding": {"x": {"field": "category"}, "y": {"field": "value"}}},
        ],
    }), encoding="utf-8")

    case = load_benchmark_cases(cases_path)[0]

    assert len(case.reference_specs) == 2
    assert case.reference_spec["mark"] == "line"


def test_evaluator_selects_best_reference_by_spec_score(tmp_path: Path) -> None:
    data_path = tmp_path / "data.csv"
    data_path.write_text("category,value\nA,1\nB,2\n", encoding="utf-8")
    cases_path = tmp_path / "cases.jsonl"
    common_encoding = {
        "x": {"field": "category", "type": "nominal"},
        "y": {"field": "value", "type": "quantitative"},
    }
    cases_path.write_text(json.dumps({
        "case_id": "case-a",
        "query": "show a bar chart",
        "data_path": data_path.name,
        "reference_specs": [
            {"mark": "line", "encoding": common_encoding},
            {"mark": "bar", "encoding": common_encoding},
        ],
    }), encoding="utf-8")
    case = load_benchmark_cases(cases_path)[0]

    result = VegaChatBenchmarkEvaluator().evaluate_spec_and_image(
        case=case,
        case_root=tmp_path,
        generated_spec={"mark": "bar", "encoding": common_encoding},
        generated_image_path=None,
        runtime=None,
        output_dir=tmp_path / "out",
    )

    assert result.best_reference_index == 1
    assert result.reference_selection_method == "argmax_spec_score"
    assert result.reference_count == 2
    assert result.spec_score is not None
    assert result.spec_score > 0.95
