from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_benchmark_manifest(
        *,
        output_dir: str | Path,
        cases_path: str | Path,
        config_path: str | Path | None = None,
        run_options: dict[str, Any] | None = None,
        corpus_root: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    manifest = build_benchmark_manifest(
        cases_path=cases_path,
        config_path=config_path,
        run_options=run_options or {},
        corpus_root=corpus_root,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def build_benchmark_manifest(
        *,
        cases_path: str | Path,
        config_path: str | Path | None = None,
        run_options: dict[str, Any] | None = None,
        corpus_root: str | Path | None = None,
) -> dict[str, Any]:
    cases = Path(cases_path)
    config = Path(config_path) if config_path else None
    corpus = Path(corpus_root) if corpus_root else None
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cases_path": cases.as_posix(),
        "dataset_hash": hash_path(cases) if cases.exists() else None,
        "config_path": config.as_posix() if config else None,
        "config_hash": hash_path(config) if config and config.exists() else None,
        "corpus_root": corpus.as_posix() if corpus else None,
        "corpus_hash": hash_path(corpus) if corpus and corpus.exists() else None,
        "run_options": run_options or {},
    }


def hash_path(path: str | Path) -> str:
    source = Path(path)
    if source.is_file():
        return _hash_file(source)
    if source.is_dir():
        digest = hashlib.sha256()
        for child in sorted(item for item in source.rglob("*") if item.is_file()):
            rel = child.relative_to(source).as_posix()
            digest.update(rel.encode("utf-8"))
            digest.update(_hash_file(child).encode("utf-8"))
        return digest.hexdigest()
    return ""


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
