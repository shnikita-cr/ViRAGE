import json

from src.domain.enums import ArtifactType
from src.domain.models import ArtifactBundle, ArtifactRef, CodeRunResult, DataPreparationResult
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService


class ArtifactStoreService(BaseService):
    def invoke(self, prepared: DataPreparationResult, execution: CodeRunResult, run_id: str,
               runtime: RuntimeContext) -> ArtifactBundle:
        run_dir = runtime.ensure_run_dir(run_id)
        artifacts = [ArtifactRef(artifact_type=ArtifactType.CLEAN_DATA, path=prepared.output_path,
                                 description="Prepared dataset."), *execution.artifacts]
        manifest_path = run_dir / "artifact_manifest.json"
        manifest_path.write_text(
            json.dumps([a.model_dump(mode="json") for a in artifacts], ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts.append(ArtifactRef(artifact_type=ArtifactType.MANIFEST, path=manifest_path.as_posix(),
                                     description="Artifact manifest."))
        return ArtifactBundle(artifacts=artifacts, manifest_path=manifest_path.as_posix())
