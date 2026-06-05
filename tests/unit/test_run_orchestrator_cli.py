from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from scripts.run_orchestrator import main


class _FakeLLMResult:
    def __init__(self, content: str) -> None:
        self.content = content
        self.usage_metadata = {"input_tokens": 1, "output_tokens": 1}


class _FakePlannerLLM:
    model = "fake-planner"

    def invoke(self, messages: Any) -> _FakeLLMResult:
        text = str(messages)
        fields_match = re.search(r'"available_fields"\s*:\s*\[(.*?)\]', text, flags=re.DOTALL)
        fields: list[str] = []
        if fields_match:
            fields = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', fields_match.group(1))
        fields = fields or ["field"]
        is_image = re.search(r'"input_type"\s*:\s*"image_folder"', text) is not None
        if is_image:
            required = [field for field in ["laplacian_variance", "contrast_rms", "mean_brightness"] if field in fields]
            required = required or fields[:1]
            task_type = "image_quality_analysis"
            query = "Find problematic generated image quality records."
        else:
            required = [field for field in ["condition", "score"] if field in fields] or fields[:1]
            task_type = "group_comparison" if len(required) >= 2 else "overview"
            query = "Analyze the selected dataset fields."
        payload = {
            "user_query": "test",
            "data_path": "data.csv",
            "input_type": "image_folder" if is_image else "table",
            "max_charts": 3,
            "subtasks": [
                {
                    "id": "analysis_001",
                    "task_type": task_type,
                    "query": query,
                    "purpose": "Create the highest-priority analytical view.",
                    "required_fields": required,
                    "optional_fields": [],
                    "priority": 1,
                    "constraints": {"output_target": "scientific_figure"},
                    "metric_semantics": {field: "higher_is_better" for field in required if field in {"laplacian_variance"}} if is_image else {},
                    "ranking_strategy": "top_n_highest_severity" if is_image else None,
                    "scale_strategy": "normalized_severity" if is_image else None,
                    "visual_constraints": ["use_overall_severity_for_problematic_items"] if is_image else [],
                    "rationale": "The fields are present in the DataProfile.",
                }
            ],
            "skipped_candidates": [
                {"task_type": "correlation", "reason": "Not selected for this plan.", "required_fields": []}
            ],
            "rationale": ["Strict LLM planner test payload."],
        }
        return _FakeLLMResult(json.dumps(payload, ensure_ascii=False))


def _fake_build_chat_model(_config: Any) -> _FakePlannerLLM:
    return _FakePlannerLLM()


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

'''.strip(),
        encoding="utf-8",
    )


def test_run_orchestrator_plan_only_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("scripts.run_orchestrator.build_chat_model", _fake_build_chat_model)
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
