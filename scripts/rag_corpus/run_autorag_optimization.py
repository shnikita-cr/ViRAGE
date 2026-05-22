from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

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
