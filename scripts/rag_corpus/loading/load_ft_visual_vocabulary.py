from __future__ import annotations

from pathlib import Path as _PathForImports
import sys
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next(
    (parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / "src").exists() and (parent / "scripts").exists()),
    _PathForImports.cwd(),
)
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from pathlib import Path

from scripts.rag_corpus.loading.common import download_text_file, run_loader_cli


def load(root: Path, *, refresh: bool = False, timeout_seconds: float = 60.0) -> dict[str, object]:
    return download_text_file(
        root,
        source_id="ft_visual_vocabulary",
        url="https://raw.githubusercontent.com/Financial-Times/chart-doctor/main/visual-vocabulary/README.md",
        relative_path="visual-vocabulary/README.md",
        refresh=refresh,
        timeout_seconds=timeout_seconds,
        min_bytes=1000,
    )


def main() -> None:
    run_loader_cli(load, "Download FT Visual Vocabulary README into rag_corpus/raw_external_rules.")


if __name__ == "__main__":
    main()
