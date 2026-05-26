from __future__ import annotations

from pathlib import Path
from pathlib import Path as _PathForImports
import sys
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next(
    (
        parent
        for parent in _CURRENT_FILE_FOR_IMPORTS.parents
        if (parent / "src").exists() and (parent / "scripts").exists()
    ),
    _PathForImports.cwd(),
)
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))


from scripts.rag_corpus.loading.common import TextFileSeed, download_text_files, run_loader_cli

SOURCE_ID = "ft_visual_vocabulary"
FILES = (
    TextFileSeed(
        "https://raw.githubusercontent.com/Financial-Times/chart-doctor/main/visual-vocabulary/README.md",
        "visual-vocabulary/README.md",
        1_000,
    ),
)


def load(root: Path, *, refresh: bool = False, timeout_seconds: float = 120.0) -> dict[str, object]:
    return download_text_files(
        root=root,
        source_id=SOURCE_ID,
        files=FILES,
        refresh=refresh,
        timeout_seconds=timeout_seconds,
    )


if __name__ == "__main__":
    run_loader_cli("Download FT Visual Vocabulary source files for ViRAGE visrag corpus.", load)
