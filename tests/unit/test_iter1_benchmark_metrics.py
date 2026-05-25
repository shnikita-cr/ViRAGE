from __future__ import annotations

import json
from pathlib import Path

from src.benchmark.analysis_models import AnalysisBenchmarkReport, AnalysisBenchmarkResult
from src.benchmark.datasets import load_benchmark_cases
from src.benchmark.models import BenchmarkAggregateReport, BenchmarkCaseResult
from src.benchmark.chart_text_metrics import chart_text_consistency_score


def test_nlv_loader_keeps_only_single_turn_cases_and_uses_stable_ids(tmp_path: Path) -> None:
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "cars.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "vlSpecs.json").write_text(
        json.dumps({"cars-one": {"mark": "bar"}, "cars-two": {"mark": "line"}}),
        encoding="utf-8",
    )
    (tmp_path / "NLV_Corpus.csv").write_text(
        "dataset,visId,sequential,Utterance Set\n"
        "cars,one,n,show one\n"
        "cars,two,y,show two|then change it\n",
        encoding="utf-8",
    )

    first = load_benchmark_cases(tmp_path)
    second = load_benchmark_cases(tmp_path)

    assert len(first) == 1
    assert first[0].utterance_type == "single_turn"
    assert first[0].case_id == second[0].case_id
    assert first[0].case_id.startswith("nlv_cars_one_")


def test_failure_as_zero_metrics_do_not_drop_failed_cases() -> None:
    report = BenchmarkAggregateReport.from_results([
        BenchmarkCaseResult(
            case_id="ok",
            query="q",
            data_path="data.csv",
            spec_score=1.0,
            vision_score=1.0,
        ),
        BenchmarkCaseResult(
            case_id="failed",
            query="q",
            data_path="data.csv",
            error="RuntimeError: failed",
        ),
    ])

    assert report.mean_spec_score == 1.0
    assert report.mean_spec_score_failure_as_zero == 0.5
    assert report.mean_vision_score_failure_as_zero == 0.5


def test_infiagent_accepted_chart_rate_uses_all_cases_as_denominator() -> None:
    report = AnalysisBenchmarkReport.from_results([
        AnalysisBenchmarkResult(case_id="accepted", question="q", data_path="d.csv", chart_was_accepted=True),
        AnalysisBenchmarkResult(case_id="rejected", question="q", data_path="d.csv", chart_was_accepted=False),
        AnalysisBenchmarkResult(case_id="technical", question="q", data_path="d.csv", chart_was_accepted=False, error="RuntimeError: failed"),
    ])

    assert report.accepted_chart_rate == 0.333333
    assert report.technical_failure_rate == 0.333333


def test_chart_text_consistency_is_structural_not_synonym_dictionary() -> None:
    spec = {
        "encoding": {
            "y": {"field": "psnr", "aggregate": "mean", "axis": {"title": "Mean PSNR"}},
            "tooltip": [{"field": "psnr", "aggregate": "mean", "title": "Average PSNR"}],
        }
    }

    assert chart_text_consistency_score(spec) == 0.0
