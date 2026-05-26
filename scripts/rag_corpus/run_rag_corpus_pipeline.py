from __future__ import annotations

import runpy
from pathlib import Path

ROOT_PIPELINE = Path(__file__).resolve().parents[2] / "rag_corpus" / "run_rag_corpus_pipeline.py"
runpy.run_path(str(ROOT_PIPELINE), run_name="__main__")
