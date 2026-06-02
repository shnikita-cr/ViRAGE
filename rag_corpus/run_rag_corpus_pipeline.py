from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PRACTICAL_SOURCES = [
    "wilke_fundamentals",
    "from_data_to_viz",
    "ft_visual_vocabulary",
    "uk_analysis_colours",
    "uk_charts_checklist",
    "urban_institute_style_guide",
    "chartability",
]

CLEAN_PATHS = [
    "rag_corpus/runtime/guidance_chunks.jsonl",
    "rag_corpus/runtime/guidance_chunk_embeddings.jsonl",
    "rag_corpus/runtime/runtime_export_report.json",
    "rag_corpus/autorag/visrag_chunks",
    "rag_corpus/autorag/runs/visrag_chunks_train",
    "rag_corpus/autorag/runs/visrag_chunks_test",
]


def run_step(title: str, command: list[str], *, cwd: Path) -> None:
    print(f"\n=== {title} ===", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def clean_outputs(root: Path) -> None:
    print("\n=== Clean previous VisRAG chunk outputs ===", flush=True)
    for rel in CLEAN_PATHS:
        path = root / rel
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    (root / "rag_corpus/runtime").mkdir(parents=True, exist_ok=True)
    (root / "rag_corpus/autorag").mkdir(parents=True, exist_ok=True)


def python_cmd(*args: str) -> list[str]:
    return [sys.executable, *args]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the chunk-based VisRAG corpus pipeline.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--embedding-provider", default="ollama")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--embedding-base-url", default="http://localhost:11434")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--chunk-min-chars", type=int, default=220)
    parser.add_argument("--chunk-max-chars", type=int, default=1000)
    parser.add_argument("--chunk-overlap-chars", type=int, default=120)
    parser.add_argument("--skip-environment-check", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--refresh-sources", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-autorag-export", action="store_true")
    parser.add_argument("--skip-autorag", action="store_true")
    parser.add_argument("--run-autorag-validate", action="store_true")
    parser.add_argument("--run-autorag-internal-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()

    if not args.skip_environment_check:
        run_step("Compile check", python_cmd("-Wdefault", "-m", "compileall", "-q", "src", "ui", "scripts", "tests", "rag_corpus/run_rag_corpus_pipeline.py"), cwd=root)
        run_step("Unit tests", python_cmd("-m", "pytest", "-q", "tests/unit/test_visrag_chunks_autorag_export.py", "tests/unit/test_visrag_retrieved_chunk_conversion.py"), cwd=root)

    if not args.skip_download:
        command = python_cmd("scripts/rag_corpus/loading/download_sources.py", "--sources", *PRACTICAL_SOURCES)
        if args.refresh_sources:
            command.append("--refresh")
        run_step("Download practical VisRAG sources", command, cwd=root)

    if not args.skip_clean:
        clean_outputs(root)

    run_step(
        "Export VisRAG guidance chunks",
        python_cmd(
            "scripts/rag_corpus/export_guidance_chunks.py",
            "--min-chars", str(args.chunk_min_chars),
            "--max-chars", str(args.chunk_max_chars),
            "--overlap-chars", str(args.chunk_overlap_chars),
        ),
        cwd=root,
    )

    if not args.skip_embeddings:
        run_step(
            "Build VisRAG chunk embeddings",
            python_cmd(
                "scripts/rag_corpus/build_visrag_embeddings.py",
                "--provider", args.embedding_provider,
                "--model", args.embedding_model,
                "--base-url", args.embedding_base_url,
            ),
            cwd=root,
        )

    if not args.skip_autorag_export:
        run_step(
            "Export AutoRAG chunk data and train/test split",
            python_cmd(
                "scripts/rag_corpus/run_export_autorag.py",
                "--train-ratio", str(args.train_ratio),
                "--split-seed", str(args.split_seed),
            ),
            cwd=root,
        )

    if not args.skip_autorag:
        config = "rag_corpus/autorag/visrag_chunks/configs/visrag_chunks_ollama_all.yaml"
        train_qa = "rag_corpus/autorag/visrag_chunks/splits/train/qa.parquet"
        train_corpus = "rag_corpus/autorag/visrag_chunks/splits/train/corpus.parquet"
        if args.run_autorag_validate:
            run_step(
                "AutoRAG validate on train via Python API",
                python_cmd(
                    "scripts/rag_corpus/run_autorag_chunks.py",
                    "validate",
                    "--config", config,
                    "--qa-data-path", train_qa,
                    "--corpus-data-path", train_corpus,
                ),
                cwd=root,
            )
        evaluate_command = python_cmd(
            "scripts/rag_corpus/run_autorag_chunks.py",
            "evaluate",
            "--config", config,
            "--qa-data-path", train_qa,
            "--corpus-data-path", train_corpus,
            "--project-dir", "rag_corpus/autorag/runs/visrag_chunks_train",
            "--clean-project-dir",
        )
        if not args.run_autorag_internal_validation:
            evaluate_command.append("--skip-validation")
        run_step("AutoRAG evaluate on train via Python API", evaluate_command, cwd=root)
        run_step(
            "Collect AutoRAG summary",
            python_cmd("scripts/rag_corpus/collect_autorag_summary.py", "--runs-root", "rag_corpus/autorag/runs", "--output-dir", "rag_corpus/autorag/runs/summary"),
            cwd=root,
        )

    print("\nVisRAG chunk corpus pipeline completed.")


if __name__ == "__main__":
    main()
