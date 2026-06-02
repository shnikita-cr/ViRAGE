from __future__ import annotations

import json
from pathlib import Path

from scripts.run_orchestrator import main


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


def test_run_orchestrator_plan_only_writes_artifacts(tmp_path: Path) -> None:
    data_path = tmp_path / "data.csv"
    data_path.write_text("condition,score,age\na,1,10\nb,3,20\na,2,15\n", encoding="utf-8")
    artifact_root = tmp_path / "artifacts"
    config_path = tmp_path / "project.toml"
    _write_config(config_path, artifact_root)

    exit_code = main([
        "--config",
        config_path.as_posix(),
        "--query",
        "Проанализируй данные",
        "--data-path",
        data_path.as_posix(),
        "--run-id",
        "orch_test",
        "--max-charts",
        "3",
    ])

    assert exit_code == 0
    run_dir = artifact_root / "orch_test"
    assert (run_dir / "orchestrator_request.json").exists()
    assert (run_dir / "data_profile.json").exists()
    assert (run_dir / "chart_plan.json").exists()
    assert (run_dir / "orchestrator_report.json").exists()
    assert (run_dir / "final_summary.md").exists()
    plan = json.loads((run_dir / "chart_plan.json").read_text(encoding="utf-8"))
    assert 1 <= len(plan["subtasks"]) <= 3
    assert plan["max_charts"] == 3
    report = json.loads((run_dir / "orchestrator_report.json").read_text(encoding="utf-8"))
    assert report["executed"] is False
