from __future__ import annotations

import json
from pathlib import Path

from src.services.artifact_store import ArtifactStoreService


def test_artifact_store_service_creates_manifest(
    runtime,
    prepared_result,
    code_run_result,
) -> None:
    service = ArtifactStoreService()

    result = service.invoke(
        prepared=prepared_result,
        execution=code_run_result,
        run_id="artifact-store",
        runtime=runtime,
    )

    manifest_path = Path(result.manifest_path)
    assert manifest_path.exists()
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result.artifacts[-1].artifact_type.value == "manifest"
    assert any(item["artifact_type"] == "clean_data" for item in manifest_payload)
