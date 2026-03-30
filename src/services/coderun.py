import json
import subprocess
import sys

from src.domain.enums import ArtifactType
from src.domain.models import ArtifactRef, CodeRunResult, CodegenResult, ExecutionMetric
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService


class CodeRunService(BaseService):
    def invoke(self, codegen: CodegenResult, run_id: str, runtime: RuntimeContext) -> CodeRunResult:
        execution_dir = runtime.ensure_run_dir(run_id) / "execution"
        execution_dir.mkdir(parents=True, exist_ok=True)
        code_path = execution_dir / "generated_plot.py"
        code_path.write_text(codegen.code, encoding="utf-8")
        artifacts = [ArtifactRef(artifact_type=ArtifactType.GENERATED_CODE, path=code_path.as_posix(),
                                 description="Generated plotting code.")]
        if not runtime.settings.allow_code_execution:
            return CodeRunResult(success=False, stderr="Code execution disabled.", artifacts=artifacts)
        proc = subprocess.run([sys.executable, code_path.as_posix()], cwd=execution_dir.as_posix(), capture_output=True,
                              text=True)
        metrics_path = execution_dir / "metrics.json"
        metadata_path = execution_dir / "chart_metadata.json"
        plot_path = execution_dir / "plot.png"
        metrics = []
        metadata = {}
        if metrics_path.exists():
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics = [ExecutionMetric(name=k, value=v) for k, v in data.items()]
            artifacts.append(ArtifactRef(artifact_type=ArtifactType.METRICS, path=metrics_path.as_posix(),
                                         description="Execution metrics."))
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            artifacts.append(ArtifactRef(artifact_type=ArtifactType.CHART_METADATA, path=metadata_path.as_posix(),
                                         description="Chart metadata."))
        if plot_path.exists():
            artifacts.append(ArtifactRef(artifact_type=ArtifactType.PLOT, path=plot_path.as_posix(),
                                         description="Generated chart image."))
        return CodeRunResult(success=proc.returncode == 0, stdout=proc.stdout, stderr=proc.stderr, metrics=metrics,
                             artifacts=artifacts, chart_metadata=metadata)
