from __future__ import annotations

from pathlib import Path

from src.application.config.settings import ViRAGESettings
from src.infrastructure.runtime import RuntimeContext


def test_numbered_artifacts_are_sequential_per_run(tmp_path: Path) -> None:
    runtime = RuntimeContext(settings=ViRAGESettings(artifact_root=tmp_path))
    runtime.current_run_id = "run-a"
    runtime.reset_artifact_indices()

    first = runtime.save_json_artifact("nodes/query_request_analysis.json", {"ok": True}, numbered=True)
    second = runtime.save_text_artifact("errors/fatal_error.txt", "boom", numbered=True)
    third_path = runtime.next_artifact_path("plot.png")
    third_path.write_bytes(b"png")

    assert first.endswith("/nodes/001_query_request_analysis.json")
    assert second.endswith("/errors/002_fatal_error.txt")
    assert third_path.as_posix().endswith("/003_plot.png")


def test_numbered_artifacts_reset_for_new_run(tmp_path: Path) -> None:
    runtime = RuntimeContext(settings=ViRAGESettings(artifact_root=tmp_path))

    runtime.current_run_id = "run-a"
    runtime.reset_artifact_indices()
    first_run_path = runtime.save_json_artifact("nodes/a.json", {}, numbered=True)

    runtime.current_run_id = "run-b"
    runtime.reset_artifact_indices()
    second_run_path = runtime.save_json_artifact("nodes/b.json", {}, numbered=True)

    assert first_run_path.endswith("/run-a/nodes/001_a.json")
    assert second_run_path.endswith("/run-b/nodes/001_b.json")


def test_unnumbered_artifacts_keep_existing_path(tmp_path: Path) -> None:
    runtime = RuntimeContext(settings=ViRAGESettings(artifact_root=tmp_path))
    runtime.current_run_id = "run-a"

    path = runtime.save_json_artifact("input/user_context.json", {"x": 1})

    assert path.endswith("/run-a/input/user_context.json")
