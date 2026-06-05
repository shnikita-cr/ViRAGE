from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmarks.run_virage_e2e_test_cases import BenchmarkCase, load_cases, main
from scripts.rag_corpus.loading import load_eda_best_practices as eda_loader
from scripts.rag_corpus.export_guidance_chunks import source_kind_for


def _write_config(path: Path, artifact_root: Path) -> None:
    path.write_text(
        f'''
mode = "pipeline"

[settings]
artifact_root = "{artifact_root.as_posix()}"
visrag_enabled = true

[reasoning_model]
provider = "ollama"
model = "dummy"

[spec_model]
provider = "ollama"
model = "dummy"

[vlm_model]
provider = "ollama"
model = "dummy"

[vision_judge_model]
provider = "ollama"
model = "dummy"
'''.strip(),
        encoding="utf-8",
    )


def test_load_cases_filters_by_suite_and_case_id(tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    payload = {
        "case_id": "case_a",
        "suite": "eda_manual",
        "input_modality": "table",
        "query_specificity": "open_ended",
        "analysis_task": "eda_overview",
        "chart_family": "multi_chart",
        "output_target": "eda",
        "expected_charts": "max_3_charts",
        "evaluation_mode": "orchestrator_execute",
        "known_risks": ["missed_missingness"],
        "data_path": "data.csv",
        "query": "Проанализируй данные",
        "expected_checks": ["valid_spec_rate"],
    }
    (cases_dir / "eda_manual.jsonl").write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    cases = load_cases(cases_dir, suites={"eda_manual"}, case_ids={"case_a"})

    assert len(cases) == 1
    assert isinstance(cases[0], BenchmarkCase)
    assert cases[0].case_id == "case_a"
    assert cases[0].analysis_task == "eda_overview"


def test_e2e_runner_plan_only_writes_reports(tmp_path: Path) -> None:
    data_path = tmp_path / "data.csv"
    data_path.write_text("group,value\na,1\nb,2\na,3\n", encoding="utf-8")
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    case = {
        "case_id": "mini_case",
        "suite": "manual_vulnerability",
        "input_modality": "table",
        "query_specificity": "specific",
        "analysis_task": "group_comparison",
        "chart_family": "boxplot",
        "output_target": "scientific_article",
        "expected_charts": "single_chart",
        "evaluation_mode": "orchestrator_execute",
        "known_risks": ["compact_layout"],
        "data_path": data_path.as_posix(),
        "query": "Сравни группы",
        "expected_checks": ["layout_compactness_score"],
    }
    (cases_dir / "manual_vulnerability.jsonl").write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
    suites_path = tmp_path / "suites.json"
    suites_path.write_text('{"manual_vulnerability": {"description": "test"}}', encoding="utf-8")
    artifact_root = tmp_path / "artifacts"
    config_path = tmp_path / "project.toml"
    _write_config(config_path, artifact_root)

    exit_code = main([
        "--config",
        config_path.as_posix(),
        "--cases-dir",
        cases_dir.as_posix(),
        "--suites-file",
        suites_path.as_posix(),
        "--run-id",
        "bench_plan",
        "--suite",
        "manual_vulnerability",
        "--no-progress",
    ])

    assert exit_code == 0
    report_dir = artifact_root / "bench_plan" / "e2e_cases_report"
    assert (report_dir / "per_case_results.csv").exists()
    assert (report_dir / "per_case_results.jsonl").exists()
    assert (report_dir / "benchmark_summary.json").exists()
    assert (report_dir / "benchmark_report.md").exists()
    summary = json.loads((report_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert summary["total_cases"] == 1
    assert summary["completed_cases"] == 1


def test_eda_best_practices_loader_and_source_kind(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def fake_download_html_pages(root, config, *, refresh, timeout_seconds):
        captured["root"] = root
        captured["config"] = config
        captured["refresh"] = refresh
        captured["timeout_seconds"] = timeout_seconds
        return {"source_id": config.source_id, "downloaded_pages": len(config.seed_urls)}

    monkeypatch.setattr(eda_loader, "download_html_pages", fake_download_html_pages)

    report = eda_loader.load(tmp_path, refresh=True, timeout_seconds=12.0)

    config = captured["config"]
    assert report["source_id"] == "eda_best_practices"
    assert report["downloaded_pages"] == 2
    assert "www.itl.nist.gov" in config.allowed_hosts
    assert "r4ds.had.co.nz" in config.allowed_hosts
    assert source_kind_for("eda_best_practices") == "eda_guidance"
