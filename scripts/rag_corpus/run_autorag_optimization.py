from __future__ import annotations

import sys
from pathlib import Path as _PathForImports

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

import argparse
import subprocess

from scripts.rag_corpus.common.io import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoRAG optimization for ViRAGE rule corpus.")
    parser.add_argument("--project-root", default="rag_corpus/autorag/virage_rules")
    parser.add_argument("--config", default="configs/virage_rules_all.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = project_root()
    project = root / args.project_root
    config = project / args.config
    cmd = [sys.executable, "-m", "autorag.cli", "--config", str(config)]
    if args.dry_run:
        print(" ".join(cmd))
        return
    subprocess.run(cmd, cwd=project, check=True)


if __name__ == "__main__":
    main()
