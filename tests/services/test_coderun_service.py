from __future__ import annotations

from src.domain.models import CodegenResult
from src.services.coderun import CodeRunService


def test_coderun_service_executes_generated_code(runtime) -> None:
    service = CodeRunService()
    code = """
from pathlib import Path
import json

def main(output_dir: str = \".\") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / \"plot.png\").write_bytes(b\"png\")
    (out / \"metrics.json\").write_text(json.dumps({\"row_count\": 2}), encoding=\"utf-8\")
    (out / \"chart_metadata.json\").write_text(
        json.dumps({\"chart_type\": \"bar\", \"y_column\": \"sales\"}),
        encoding=\"utf-8\",
    )

if __name__ == \"__main__\":
    main()
"""

    result = service.invoke(
        codegen=CodegenResult(chart_type="bar", code=code),
        run_id="coderun",
        runtime=runtime,
    )

    assert result.success is True
    assert result.chart_metadata["chart_type"] == "bar"
    assert any(metric.name == "row_count" for metric in result.metrics)
    assert any(artifact.artifact_type.value == "plot" for artifact in result.artifacts)
