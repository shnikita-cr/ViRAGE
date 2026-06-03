from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmarks.run_virage_e2e_test_cases import (
    VirageE2ETestCase,
    load_cases,
    resolve_case_data_path,
)


def test_load_cases_reads_jsonl(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps({
            "case_id": "case_1",
            "input_type": "table",
            "data_path": "demo_data/Iris.csv",
            "query": "Проанализируй данные",
            "focus": ["axis_domain_score"],
            "expected_checks": ["check axis"],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    cases = load_cases(cases_path)

    assert len(cases) == 1
    assert cases[0].case_id == "case_1"
    assert cases[0].focus == ["axis_domain_score"]


def test_resolve_case_data_path_uses_image_folder_override(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    case = VirageE2ETestCase(
        case_id="image_case",
        input_type="image_folder",
        data_path="{image_folder}",
        query="Проанализируй изображения",
    )

    resolved = resolve_case_data_path(case, image_folder=image_dir.as_posix(), project_root=tmp_path)

    assert resolved == image_dir


def test_resolve_case_data_path_resolves_relative_to_project_root(tmp_path: Path) -> None:
    case = VirageE2ETestCase(
        case_id="table_case",
        input_type="table",
        data_path="demo/data.csv",
        query="Проанализируй таблицу",
    )

    resolved = resolve_case_data_path(case, image_folder=None, project_root=tmp_path)

    assert resolved == tmp_path / "demo" / "data.csv"
