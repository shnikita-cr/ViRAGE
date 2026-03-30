from pathlib import Path

from src.application.contracts import PipelineRequest
from src.application.pipeline import ViRAGEPipeline
from src.application.settings import ViRAGESettings


def test_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    settings = ViRAGESettings(artifact_root=root / "artifacts_test", allow_code_execution=True)
    pipeline = ViRAGEPipeline(settings=settings)
    result = pipeline.invoke(PipelineRequest(query="Analyze trend of sales over time",
                                             data_path=(root / "examples" / "demo.csv").as_posix()))
    assert result.execution_result.success is True
    assert result.verification_result.findings
