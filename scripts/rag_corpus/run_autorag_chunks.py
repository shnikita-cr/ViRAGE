from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_CURRENT = Path(__file__).resolve()
ROOT = next((parent for parent in _CURRENT.parents if (parent / "src").exists() and (parent / "scripts").exists()), Path.cwd())

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


def _prepare_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    return env


def _validate_parquet(path: Path, *, required_columns: set[str], label: str) -> dict[str, Any]:
    _require_file(path, label)
    frame = pd.read_parquet(path)
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise AutoRAGRunnerError(f"{label} misses required columns {missing}: {path}")
    if frame.empty:
        raise AutoRAGRunnerError(f"{label} is empty: {path}")
    null_counts = {column: int(count) for column, count in frame.isna().sum().items() if int(count) > 0}
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "null_counts": null_counts,
    }


def _preflight_autorag_cli(autorag_command: str, *, cwd: Path, env: dict[str, str]) -> None:
    try:
        result = subprocess.run(
            [autorag_command, "--help"],
            cwd=cwd,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError as exc:
        raise AutoRAGRunnerError(
            "AutoRAG CLI command was not found. Install API AutoRAG without GPU extras:\n"
            "pip install AutoRAG fastapi gradio chromadb pyarrow pandas scikit-learn llama-index-embeddings-ollama"
        ) from exc
    output = result.stdout or ""
    if result.returncode != 0:
        raise AutoRAGRunnerError(
            "AutoRAG CLI is installed but cannot start. This project uses the API AutoRAG package, "
            "not AutoRAG[gpu] and not vLLM.\n"
            "Install/repair the API dependencies, for example:\n"
            "pip install --upgrade AutoRAG fastapi gradio chromadb pyarrow pandas scikit-learn llama-index-embeddings-ollama\n\n"
            f"AutoRAG output:\n{output}"
        )


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def run_validate(*, autorag_command: str, config: Path, qa_path: Path, corpus_path: Path) -> dict[str, Any]:
    _require_file(config, "AutoRAG YAML config")
    qa_report = _validate_parquet(qa_path, required_columns={"qid", "query", "retrieval_gt"}, label="AutoRAG QA parquet")
    corpus_report = _validate_parquet(corpus_path, required_columns={"doc_id", "contents"}, label="AutoRAG corpus parquet")
    env = _prepare_env()
    _preflight_autorag_cli(autorag_command, cwd=ROOT, env=env)
    _run(
        [
            autorag_command,
            "validate",
            "--config",
            str(config),
            "--qa_data_path",
            str(qa_path),
            "--corpus_data_path",
            str(corpus_path),
        ],
        cwd=ROOT,
        env=env,
    )
    return {
        "stage": "validate",
        "status": "ok",
        "config": str(config),
        "qa": qa_report,
        "corpus": corpus_report,
    }


def run_evaluate(
    *,
    autorag_command: str,
    config: Path,
    qa_path: Path,
    corpus_path: Path,
    project_dir: Path,
    skip_validation: bool,
    clean_project_dir: bool,
) -> dict[str, Any]:
    _require_file(config, "AutoRAG YAML config")
    qa_report = _validate_parquet(qa_path, required_columns={"qid", "query", "retrieval_gt"}, label="AutoRAG QA parquet")
    corpus_report = _validate_parquet(corpus_path, required_columns={"doc_id", "contents"}, label="AutoRAG corpus parquet")
    if clean_project_dir and project_dir.exists():
        shutil.rmtree(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    env = _prepare_env()
    _preflight_autorag_cli(autorag_command, cwd=ROOT, env=env)
    command = [
        autorag_command,
        "evaluate",
        "--config",
        str(config),
        "--qa_data_path",
        str(qa_path),
        "--corpus_data_path",
        str(corpus_path),
        "--project_dir",
        str(project_dir),
    ]
    if skip_validation:
        command.append("--skip_validation")
    _run(command, cwd=ROOT, env=env)
    return {
        "stage": "evaluate",
        "status": "ok",
        "config": str(config),
        "project_dir": str(project_dir),
        "skip_validation": skip_validation,
        "qa": qa_report,
        "corpus": corpus_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run AutoRAG for VisRAG chunks through the AutoRAG API CLI and YAML config. "
            "This runner does not require AutoRAG[gpu], vLLM, or local model extras."
        )
    )
    parser.add_argument("mode", choices=["validate", "evaluate", "both"])
    parser.add_argument("--autorag-command", default="autorag")
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
        reports.append(
            run_validate(
                autorag_command=args.autorag_command,
                config=config,
                qa_path=qa_path,
                corpus_path=corpus_path,
            )
        )
    if args.mode in {"evaluate", "both"}:
        reports.append(
            run_evaluate(
                autorag_command=args.autorag_command,
                config=config,
                qa_path=qa_path,
                corpus_path=corpus_path,
                project_dir=project_dir,
                skip_validation=args.skip_validation,
                clean_project_dir=args.clean_project_dir,
            )
        )
    print(json.dumps({"reports": reports}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
