from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

_CURRENT = Path(__file__).resolve()
ROOT = next((parent for parent in _CURRENT.parents if (parent / "src").exists() and (parent / "scripts").exists()), Path.cwd())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = "rag_corpus/autorag/visrag_chunks/configs/visrag_chunks_ollama_all.yaml"
DEFAULT_TRAIN_QA = "rag_corpus/autorag/visrag_chunks/splits/train/qa.parquet"
DEFAULT_TRAIN_CORPUS = "rag_corpus/autorag/visrag_chunks/splits/train/corpus.parquet"
DEFAULT_PROJECT_DIR = "rag_corpus/autorag/runs/visrag_chunks_train"


class AutoRAGRunnerError(RuntimeError):
    pass


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _require_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise AutoRAGRunnerError(f"{label} not found: {path}")


def _prepare_autorag_env() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")


def _import_validator():
    try:
        from autorag.validator import Validator
    except Exception as exc:  # pragma: no cover - depends on external AutoRAG installation
        raise AutoRAGRunnerError(
            "Cannot import autorag.validator.Validator. "
            "Check AutoRAG installation and avoid using the 'autorag' CLI when it imports deploy/gradio/fastapi."
        ) from exc
    return Validator


def _import_evaluator():
    try:
        from autorag.evaluator import Evaluator
    except Exception as exc:  # pragma: no cover - depends on external AutoRAG installation
        raise AutoRAGRunnerError(
            "Cannot import autorag.evaluator.Evaluator. "
            "Check AutoRAG installation and avoid using the 'autorag' CLI when it imports deploy/gradio/fastapi."
        ) from exc
    return Evaluator


def run_validate(*, config: Path, qa_path: Path, corpus_path: Path) -> dict[str, Any]:
    _require_file(config, "AutoRAG config")
    _require_file(qa_path, "AutoRAG QA parquet")
    _require_file(corpus_path, "AutoRAG corpus parquet")
    _prepare_autorag_env()
    Validator = _import_validator()
    validator = Validator(qa_data_path=str(qa_path), corpus_data_path=str(corpus_path))
    validator.validate(str(config))
    return {
        "stage": "validate",
        "config": str(config),
        "qa_data_path": str(qa_path),
        "corpus_data_path": str(corpus_path),
        "status": "ok",
    }


def run_evaluate(*, config: Path, qa_path: Path, corpus_path: Path, project_dir: Path, skip_validation: bool, clean_project_dir: bool) -> dict[str, Any]:
    _require_file(config, "AutoRAG config")
    _require_file(qa_path, "AutoRAG QA parquet")
    _require_file(corpus_path, "AutoRAG corpus parquet")
    if clean_project_dir and project_dir.exists():
        shutil.rmtree(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    _prepare_autorag_env()
    Evaluator = _import_evaluator()
    evaluator = Evaluator(
        qa_data_path=str(qa_path),
        corpus_data_path=str(corpus_path),
        project_dir=str(project_dir),
    )
    evaluator.start_trial(str(config), skip_validation=skip_validation)
    return {
        "stage": "evaluate",
        "config": str(config),
        "qa_data_path": str(qa_path),
        "corpus_data_path": str(corpus_path),
        "project_dir": str(project_dir),
        "skip_validation": skip_validation,
        "status": "ok",
    }


def _print_report(report: dict[str, Any]) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run AutoRAG for VisRAG chunks through the Python API, without importing autorag.cli/deploy/gradio."
        )
    )
    parser.add_argument("mode", choices=["validate", "evaluate", "both"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--qa-data-path", default=DEFAULT_TRAIN_QA)
    parser.add_argument("--corpus-data-path", default=DEFAULT_TRAIN_CORPUS)
    parser.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--clean-project-dir", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _resolve(args.config)
    qa_path = _resolve(args.qa_data_path)
    corpus_path = _resolve(args.corpus_data_path)
    project_dir = _resolve(args.project_dir)
    reports: list[dict[str, Any]] = []
    if args.mode in {"validate", "both"}:
        reports.append(run_validate(config=config, qa_path=qa_path, corpus_path=corpus_path))
    if args.mode in {"evaluate", "both"}:
        reports.append(
            run_evaluate(
                config=config,
                qa_path=qa_path,
                corpus_path=corpus_path,
                project_dir=project_dir,
                skip_validation=args.skip_validation,
                clean_project_dir=args.clean_project_dir,
            )
        )
    _print_report({"reports": reports})


if __name__ == "__main__":
    main()
