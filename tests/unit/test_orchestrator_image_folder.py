from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts.run_orchestrator import main
from tests.unit.test_run_orchestrator_cli import _fake_build_chat_model, _write_config


def test_run_orchestrator_image_folder_plan_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("scripts.run_orchestrator.build_chat_model", _fake_build_chat_model)
    image_dir = tmp_path / "images"
    (image_dir / "set_a").mkdir(parents=True)
    Image.new("L", (12, 8), color=80).save(image_dir / "set_a" / "dark.png")
    Image.new("L", (12, 8), color=220).save(image_dir / "set_a" / "bright.png")
    artifact_root = tmp_path / "artifacts"
    config_path = tmp_path / "project.toml"
    _write_config(config_path, artifact_root)

    exit_code = main([
        "--config",
        config_path.as_posix(),
        "--query",
        "Проанализируй качество изображений и найди проблемные файлы",
        "--data-path",
        image_dir.as_posix(),
        "--input-type",
        "image_folder",
        "--run-id",
        "orch_images",
    ])

    assert exit_code == 0
    run_dir = artifact_root / "orch_images"
    metrics_path = run_dir / "input" / "image_folder" / "image_quality_metrics.csv"
    failed_path = run_dir / "input" / "image_folder" / "failed_images.csv"
    assert metrics_path.exists()
    assert failed_path.exists()
    plan = json.loads((run_dir / "chart_plan.json").read_text(encoding="utf-8"))
    assert plan["input_type"] == "image_folder"
    assert plan["data_path"] == metrics_path.as_posix()
    assert 1 <= len(plan["subtasks"]) <= 3
    assert any(item["task_type"] == "image_quality_analysis" for item in plan["subtasks"])
    request = json.loads((run_dir / "orchestrator_request.json").read_text(encoding="utf-8"))
    assert request["preprocessing"]["processed_images"] == 2
