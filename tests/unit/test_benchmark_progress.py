from __future__ import annotations

from src.benchmark.core.progress import ConsoleProgressBar


def test_console_progress_bar_prints_shared_status_fields(capsys) -> None:
    progress = ConsoleProgressBar(total=4, title="Test benchmark", width=10)

    progress.update(2, ok=1, errors=1, reused=0, stage="vlm_judge", label="case_002")
    progress.close()

    captured = capsys.readouterr().out
    assert "Test benchmark" in captured
    assert "2/4" in captured
    assert "ok=1" in captured
    assert "errors=1" in captured
    assert "reused=0" in captured
    assert "stage=vlm_judge" in captured
    assert "eta=" in captured
    assert "case_002" in captured


def test_console_progress_bar_can_be_disabled(capsys) -> None:
    progress = ConsoleProgressBar(total=2, title="Silent", enabled=False)

    progress.update(1, ok=1, errors=0, label="case_001")
    progress.close()

    assert capsys.readouterr().out == ""
