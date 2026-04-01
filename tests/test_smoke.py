from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.application.contracts import PipelineRequest
from src.application.pipeline import ViRAGEPipeline
from src.application.settings import ViRAGESettings
from src.domain.enums import ChartCaseType


def test_smoke_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    data_path = tmp_path / "demo.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "sales": [10, 14, 13, 18],
            "region": ["A", "A", "B", "B"],
        }
    ).to_csv(data_path, index=False)

    pipeline = ViRAGEPipeline(settings=ViRAGESettings(artifact_root=tmp_path / "artifacts"))
    result = pipeline.invoke(PipelineRequest(query="Show the sales trend over time", data_path=data_path.as_posix()))

    assert result.case_type is ChartCaseType.CANONICAL
    assert result.query_understanding.intent
    assert result.planning.steps
    assert result.data_profile.row_count == 4
    assert "sales" in result.data_profile.likely_numeric_columns
    assert "date" in result.data_profile.likely_time_columns
    assert result.execution.success is True
    assert result.artifact_bundle.artifacts
    assert result.chart_read.chart_type
    assert result.facts.facts
    assert result.reasoning.statements
    assert result.verification.findings


def test_non_canonical_routing_uses_non_canonical_plan(tmp_path: Path) -> None:
    data_path = tmp_path / "demo.csv"
    pd.DataFrame(
        {
            "source": ["A", "A", "B", "C"],
            "target": ["B", "C", "C", "D"],
            "weight": [1, 2, 1, 3],
        }
    ).to_csv(data_path, index=False)

    pipeline = ViRAGEPipeline(settings=ViRAGESettings(artifact_root=tmp_path / "artifacts_noncanonical"))
    result = pipeline.invoke(PipelineRequest(query="Build a network-like diagram of flows between nodes", data_path=data_path.as_posix()))

    assert result.case_type is ChartCaseType.NON_CANONICAL
    assert result.planning.mode is ChartCaseType.NON_CANONICAL
    assert any("assumption" in step.description.lower() or "non-canonical" in step.description.lower() or "simpler" in step.description.lower() for step in result.planning.steps)
