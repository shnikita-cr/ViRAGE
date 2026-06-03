from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
from pathlib import Path

_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next(
    (parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / "src").exists() and (parent / "scripts").exists()),
    _PathForImports.cwd(),
)
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from scripts.rag_corpus.common.io import write_json
from scripts.rag_corpus.loading.common import raw_dir_for, run_loader_cli
from scripts.rag_corpus.loading.loader_models import SourceDownloadError

_MANUAL_CACHE_ROOT = Path("rag_corpus") / "manual_sources" / "eda_guidance"
_ALLOWED_EXTENSIONS = {".txt", ".md"}


def _manual_seed_paths(root: Path) -> list[Path]:
    manual_root = root / _MANUAL_CACHE_ROOT
    if not manual_root.exists():
        return []
    return sorted(
        path
        for path in manual_root.rglob("*")
        if path.is_file() and path.suffix.lower() in _ALLOWED_EXTENSIONS
    )


def _copy_manual_sources(root: Path, target_dir: Path) -> list[dict[str, object]]:
    paths = _manual_seed_paths(root)
    if not paths:
        raise SourceDownloadError(
            "No EDA guidance manual sources found. Expected .txt/.md files under "
            f"{root / _MANUAL_CACHE_ROOT}."
        )

    items: list[dict[str, object]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="strict").strip()
        if len(text) < 120:
            raise SourceDownloadError(f"EDA guidance source is too small: {path}")
        rel = path.relative_to(root / _MANUAL_CACHE_ROOT)
        output_path = target_dir / "manual_seed" / rel.with_suffix(".txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        items.append(
            {
                "status": "manual_seed",
                "source_label": path.stem,
                "url": "internal:eda_guidance",
                "path": str(output_path),
                "bytes": output_path.stat().st_size,
                "raw_bytes": path.stat().st_size,
                "content_type": "text/plain; charset=utf-8; source=manual_seed",
            }
        )
    return items


def load(root: Path, *, refresh: bool = False, timeout_seconds: float = 60.0) -> dict[str, object]:  # noqa: ARG001
    target_dir = raw_dir_for(root, "eda_guidance")
    if target_dir.exists() and refresh:
        import shutil

        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = target_dir / "download_manifest.json"
    if manifest_path.exists() and not refresh:
        try:
            import json

            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise SourceDownloadError(f"Invalid cached manifest for eda_guidance: {manifest_path}: {exc}") from exc

    items = _copy_manual_sources(root, target_dir)
    manifest = {
        "source_id": "eda_guidance",
        "status": "ok",
        "items": items,
        "downloaded_pages": len(items),
        "target_dir": str(target_dir),
        "manual_cache_root": str(root / _MANUAL_CACHE_ROOT),
        "saved_format": "clean_txt",
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    run_loader_cli(load, "Prepare EDA guidance manual seeds into rag_corpus/raw_external_rules.")


if __name__ == "__main__":
    main()
