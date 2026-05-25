from __future__ import annotations

import json
from pathlib import Path

from src.benchmark.datasets import load_benchmark_cases
from src.benchmark.evaluator import VegaChatBenchmarkEvaluator
from src.benchmark.models import BenchmarkAggregateReport


def test_load_benchmark_cases_supports_flexible_vegachat_keys(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("category,value\nA,1\nB,2\n", encoding="utf-8")
    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text(
        json.dumps({
            "id": "case-a",
            "utterance": "show value by category",
            "csv_path": csv_path.name,
            "gt_spec": {"mark": "bar", "encoding": {"x": {"field": "category"}, "y": {"field": "value"}}},
            "difficulty": "easy",
        }) + "\n",
        encoding="utf-8",
    )

    cases = load_benchmark_cases(dataset_path)

    assert len(cases) == 1
    assert cases[0].case_id == "case-a"
    assert cases[0].query == "show value by category"
    assert cases[0].data_path == csv_path.name
    assert cases[0].reference_spec["mark"] == "bar"
    assert cases[0].difficulty == "easy"


def test_aggregate_report_computes_vegachat_rates() -> None:
    from src.benchmark.models import BenchmarkCaseResult

    report = BenchmarkAggregateReport.from_results([
        BenchmarkCaseResult(
            case_id="ok",
            query="q1",
            data_path="data.csv",
            is_valid_spec=True,
            is_empty_chart=False,
            visualization_error_rate_item=False,
            empty_chart_rate_item=False,
            spec_score=1.0,
            vision_score=0.8,
            total_tokens=10,
        ),
        BenchmarkCaseResult(
            case_id="bad",
            query="q2",
            data_path="data.csv",
            is_valid_spec=False,
            is_empty_chart=True,
            visualization_error_rate_item=True,
            empty_chart_rate_item=True,
            spec_score=0.0,
            total_tokens=5,
            error="RuntimeError: failed",
        ),
    ])

    assert report.total_cases == 2
    assert report.successful_cases == 1
    assert report.failed_cases == 1
    assert report.visualization_error_rate == 0.5
    assert report.empty_chart_rate == 0.5
    assert report.mean_spec_score == 0.5
    assert report.mean_vision_score == 0.8
    assert report.total_tokens == 15


def test_offline_benchmark_evaluator_scores_generated_spec(tmp_path: Path) -> None:
    data_path = tmp_path / "data.csv"
    data_path.write_text("category,value\nA,1\nB,2\n", encoding="utf-8")
    reference_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "mark": "bar",
        "encoding": {
            "x": {"field": "category", "type": "nominal"},
            "y": {"field": "value", "type": "quantitative"},
        },
    }
    generated_spec = {
        "mark": "bar",
        "encoding": {
            "x": {"field": "category", "type": "nominal"},
            "y": {"field": "value", "type": "quantitative"},
        },
    }
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps([{"id": "case-a", "query": "show value by category", "data_path": data_path.name,
                     "reference_spec": reference_spec}], ensure_ascii=False),
        encoding="utf-8",
    )
    case = load_benchmark_cases(cases_path)[0]

    result = VegaChatBenchmarkEvaluator().evaluate_spec_and_image(
        case=case,
        case_root=tmp_path,
        generated_spec=generated_spec,
        generated_image_path=None,
        runtime=None,
        output_dir=tmp_path / "out",
    )

    assert result.is_valid_spec is True
    assert result.visualization_error_rate_item is False
    assert result.spec_score is not None
    assert result.spec_score > 0.95


def test_vegachat_spec_score_matches_core_formula_for_empty_chart() -> None:
    from src.services.vegachat_spec_metrics import compute_vegachat_spec_score

    reference_spec = {
        "mark": "bar",
        "encoding": {"x": {"field": "category"}, "y": {"field": "value"}},
    }
    generated_spec = {
        "mark": "bar",
        "encoding": {"x": {"field": "category"}, "y": {"field": "value"}},
    }

    normal = compute_vegachat_spec_score(
        reference_spec,
        generated_spec,
        utterance="show a bar chart",
        hyp_is_drawable=True,
        hyp_is_empty_chart=False,
        hyp_is_valid_schema=True,
    )
    empty = compute_vegachat_spec_score(
        reference_spec,
        generated_spec,
        utterance="show a bar chart",
        hyp_is_drawable=True,
        hyp_is_empty_chart=True,
        hyp_is_valid_schema=True,
    )

    assert normal.spec_score > 0.99
    assert 0.0 < empty.spec_score < 0.01
    assert normal.encoding.f1 == 1.0
    assert normal.mark.f1 == 1.0


def test_load_nlv_corpus_directory(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    (datasets_dir / "cars.csv").write_text("Horsepower,Miles_per_Gallon\n100,20\n", encoding="utf-8")
    (tmp_path / "vlSpecs.json").write_text(
        json.dumps({
            "cars-bar": {
                "mark": "bar",
                "encoding": {"x": {"field": "Horsepower"}, "y": {"field": "Miles_per_Gallon"}},
            }
        }),
        encoding="utf-8",
    )
    (tmp_path / "NLV_Corpus.csv").write_text(
        "dataset,visId,sequential,Utterance Set\n"
        "cars,bar,n,show mpg by horsepower\n",
        encoding="utf-8",
    )

    cases = load_benchmark_cases(tmp_path)

    assert len(cases) == 1
    assert cases[0].dataset_name == "nlv_corpus"
    assert cases[0].query == "show mpg by horsepower"
    assert Path(cases[0].data_path).is_absolute()
    assert cases[0].reference_spec["mark"] == "bar"


def test_load_chart_llm_directory(tmp_path: Path) -> None:
    gold_dir = tmp_path / "exp" / "gold" / "result"
    specs_dir = tmp_path / "docs" / "data" / "chart_48_in"
    csv_dir = tmp_path / "docs" / "data" / "csv_48_process"
    png_dir = tmp_path / "docs" / "data" / "chart_48_img" / "1.simple"
    gold_dir.mkdir(parents=True)
    specs_dir.mkdir(parents=True)
    csv_dir.mkdir(parents=True)
    png_dir.mkdir(parents=True)
    (gold_dir / "gold.csv").write_text(
        "Chart #,level,interaction,composite,command,query,question\n"
        "0,simple,False,False,draw line,price by date,how price changes\n",
        encoding="utf-8",
    )
    (specs_dir / "vl_00.vl.json").write_text(
        json.dumps({"mark": "line", "encoding": {"x": {"field": "date"}, "y": {"field": "price"}}}),
        encoding="utf-8",
    )
    (csv_dir / "d_00.csv").write_text("date,price\n2024-01-01,10\n", encoding="utf-8")
    (png_dir / "visualization (0).png").write_bytes(b"png")

    cases = load_benchmark_cases(tmp_path)

    assert len(cases) == 3
    assert {case.utterance_type for case in cases} == {"command", "query", "question"}
    assert all(case.dataset_name == "chart_llm_gold" for case in cases)
    assert all(Path(case.data_path).is_absolute() for case in cases)
    assert all(case.reference_image_path for case in cases)


def test_benchmark_runner_resume_skips_existing_successful_cases(tmp_path: Path) -> None:
    from src.benchmark.models import BenchmarkCase, BenchmarkCaseResult
    from src.benchmark.runner import VegaChatBenchmarkRunner

    data_path = tmp_path / "data.csv"
    data_path.write_text("category,value\nA,1\n", encoding="utf-8")
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        "\n".join([
            json.dumps({"id": "case-a", "query": "show a", "data_path": data_path.name}),
            json.dumps({"id": "case-b", "query": "show b", "data_path": data_path.name}),
        ]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    existing_dir = output_dir / "cases" / "case-a"
    existing_dir.mkdir(parents=True)
    existing = BenchmarkCaseResult(
        case_id="case-a",
        query="show a",
        data_path=data_path.as_posix(),
        is_valid_spec=True,
        is_empty_chart=False,
        visualization_error_rate_item=False,
        empty_chart_rate_item=False,
        spec_score=1.0,
        total_tokens=10,
    )
    (existing_dir / "result.json").write_text(
        json.dumps(existing.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    class RecordingRunner(VegaChatBenchmarkRunner):
        def __init__(self) -> None:
            super().__init__(pipeline=object())  # type: ignore[arg-type]
            self.ran_case_ids: list[str] = []

        def run_case(self, *, case: BenchmarkCase, case_root: Path, output_dir: Path) -> BenchmarkCaseResult:
            self.ran_case_ids.append(case.case_id)
            result = BenchmarkCaseResult(
                case_id=case.case_id,
                query=case.query,
                data_path=case.resolved_data_path(case_root),
                is_valid_spec=True,
                is_empty_chart=False,
                visualization_error_rate_item=False,
                empty_chart_rate_item=False,
                spec_score=0.5,
                total_tokens=5,
            )
            self._write_case_artifacts(result, output_dir)
            return result

    runner = RecordingRunner()
    report = runner.run_dataset(cases_path=cases_path, output_dir=output_dir, resume=True)

    assert runner.ran_case_ids == ["case-b"]
    assert report.total_cases == 2
    assert report.total_tokens == 15
    assert (output_dir / "benchmark_results.csv").exists()
    assert (output_dir / "benchmark_report.json").exists()


def test_benchmark_runner_resume_can_retry_failed_cases(tmp_path: Path) -> None:
    from src.benchmark.models import BenchmarkCase, BenchmarkCaseResult
    from src.benchmark.runner import VegaChatBenchmarkRunner

    data_path = tmp_path / "data.csv"
    data_path.write_text("category,value\nA,1\n", encoding="utf-8")
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps({"id": "case-a", "query": "show a", "data_path": data_path.name}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    existing_dir = output_dir / "cases" / "case-a"
    existing_dir.mkdir(parents=True)
    failed = BenchmarkCaseResult(
        case_id="case-a",
        query="show a",
        data_path=data_path.as_posix(),
        is_valid_spec=False,
        is_empty_chart=True,
        visualization_error_rate_item=True,
        empty_chart_rate_item=True,
        error="RuntimeError: failed",
    )
    (existing_dir / "result.json").write_text(
        json.dumps(failed.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    class RecordingRunner(VegaChatBenchmarkRunner):
        def __init__(self) -> None:
            super().__init__(pipeline=object())  # type: ignore[arg-type]
            self.ran_case_ids: list[str] = []

        def run_case(self, *, case: BenchmarkCase, case_root: Path, output_dir: Path) -> BenchmarkCaseResult:
            self.ran_case_ids.append(case.case_id)
            result = BenchmarkCaseResult(
                case_id=case.case_id,
                query=case.query,
                data_path=case.resolved_data_path(case_root),
                is_valid_spec=True,
                is_empty_chart=False,
                visualization_error_rate_item=False,
                empty_chart_rate_item=False,
                spec_score=0.9,
            )
            self._write_case_artifacts(result, output_dir)
            return result

    runner = RecordingRunner()
    report = runner.run_dataset(cases_path=cases_path, output_dir=output_dir, resume=True, retry_failed=True)

    assert runner.ran_case_ids == ["case-a"]
    assert report.successful_cases == 1
    assert report.failed_cases == 0
