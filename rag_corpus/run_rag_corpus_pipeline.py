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
    "rag_corpus/extracted",
    "rag_corpus/processed/llm_normalized",
    "rag_corpus/processed/all_rules.jsonl",
    "rag_corpus/processed/all_rules.deduped.jsonl",
    "rag_corpus/processed/all_rules.filtered.jsonl",
    "rag_corpus/processed/all_rules.embedding_deduped.jsonl",
    "rag_corpus/processed/all_rules.validated.jsonl",
    "rag_corpus/processed/normalization_failures.jsonl",
    "rag_corpus/runtime",
    "rag_corpus/autorag/virage_rules",
    "rag_corpus/autorag/runs",
]
CREATE_DIRS = [
    "rag_corpus/extracted",
    "rag_corpus/processed/llm_normalized",
    "rag_corpus/runtime",
    "rag_corpus/autorag/virage_rules",
]


def run_step(title: str, command: list[str], *, cwd: Path) -> None:
    print(f"\n=== {title} ===", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def clean_outputs(root: Path) -> None:
    print("\n=== Clean previous RAG outputs ===", flush=True)
    for rel in CLEAN_PATHS:
        path = root / rel
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    for rel in CREATE_DIRS:
        (root / rel).mkdir(parents=True, exist_ok=True)


def first_trial_path(run_root: Path) -> Path:
    trials = sorted(path for path in run_root.iterdir() if path.is_dir()) if run_root.exists() else []
    if not trials:
        raise RuntimeError(f"No AutoRAG trial directory found in {run_root}")
    return trials[0]


def python_cmd(*args: str) -> list[str]:
    return [sys.executable, *args]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ViRAGE practical visrag corpus pipeline.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--llm-model", default="qwen2.5-coder:7b")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--embedding-threshold", type=float, default=0.95)
    parser.add_argument("--export-profile", choices=["validated", "filtered", "embedding_deduped"], default="embedding_deduped")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--nlv-smoke-limit", type=int, default=20)
    parser.add_argument("--skip-environment-check", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--refresh-sources", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-autorag", action="store_true")
    parser.add_argument("--skip-runtime-config-apply", action="store_true")
    parser.add_argument("--skip-nlv-smoke", action="store_true")
    parser.add_argument("--skip-infiagent-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    sources = [*PRACTICAL_SOURCES]

    if not args.skip_environment_check:
        run_step("Environment check", python_cmd("-Wdefault", "-m", "compileall", "-q", "src", "ui", "scripts", "tests"), cwd=root)
        run_step("Unit tests", python_cmd("-m", "pytest", "-q", "tests"), cwd=root)

    if not args.skip_download:
        command = python_cmd("scripts/rag_corpus/loading/download_sources.py", "--sources", *sources)
        if args.refresh_sources:
            command.append("--refresh")
        run_step("Download practical visrag corpus sources", command, cwd=root)

    if not args.skip_clean:
        clean_outputs(root)

    run_step(
        "Prepare quality corpus",
        python_cmd(
            "scripts/rag_corpus/run_prepare_corpus.py",
            "--provider", "ollama",
            "--model", args.llm_model,
            "--sources", *sources,
            "--clean-processed",
            "--skip-source-download",
        ),
        cwd=root,
    )
    run_step("Corpus quality report", python_cmd("scripts/rag_corpus/report_corpus_quality.py", "--input", "rag_corpus/processed/all_rules.validated.jsonl"), cwd=root)
    run_step("Embedding deduplication test", python_cmd("scripts/rag_corpus/normalize/test_embedding_dedup.py", "--model", args.embedding_model), cwd=root)
    run_step("Embedding deduplication", python_cmd("scripts/rag_corpus/normalize/deduplicate_by_embeddings.py", "--model", args.embedding_model, "--threshold", str(args.embedding_threshold)), cwd=root)
    run_step("Embedding deduplication report", python_cmd("scripts/rag_corpus/report_corpus_quality.py", "--input", "rag_corpus/processed/all_rules.embedding_deduped.jsonl", "--output", "rag_corpus/reports/corpus_quality_embedding_deduped_report.md"), cwd=root)
    run_step("Export runtime corpus", python_cmd("scripts/rag_corpus/run_export_runtime.py", "--profile", args.export_profile), cwd=root)
    run_step("Export AutoRAG data and train/test split", python_cmd("scripts/rag_corpus/run_export_autorag.py", "--profile", args.export_profile, "--train-ratio", str(args.train_ratio), "--split-seed", str(args.split_seed)), cwd=root)

    if not args.skip_autorag:
        config = "rag_corpus/autorag/virage_rules/configs/virage_rules_ollama_all.yaml"
        train_qa = "rag_corpus/autorag/virage_rules/splits/train/qa.parquet"
        train_corpus = "rag_corpus/autorag/virage_rules/splits/train/corpus.parquet"
        test_qa = "rag_corpus/autorag/virage_rules/splits/test/qa.parquet"
        test_corpus = "rag_corpus/autorag/virage_rules/splits/test/corpus.parquet"
        run_step("AutoRAG validate on train", ["autorag", "validate", "--config", config, "--qa_data_path", train_qa, "--corpus_data_path", train_corpus], cwd=root)
        shutil.rmtree(root / "rag_corpus/autorag/runs/ollama_all_train", ignore_errors=True)
        (root / "rag_corpus/autorag/runs/ollama_all_train").mkdir(parents=True, exist_ok=True)
        run_step("AutoRAG evaluate on train", ["autorag", "evaluate", "--config", config, "--qa_data_path", train_qa, "--corpus_data_path", train_corpus, "--project_dir", "rag_corpus/autorag/runs/ollama_all_train"], cwd=root)
        trial_path = first_trial_path(root / "rag_corpus/autorag/runs/ollama_all_train")
        run_step("Extract best AutoRAG config", ["autorag", "extract_best_config", "--trial_path", str(trial_path), "--output_path", "rag_corpus/autorag/runs/ollama_all_best_config.yaml"], cwd=root)
        shutil.rmtree(root / "rag_corpus/autorag/runs/ollama_all_test", ignore_errors=True)
        (root / "rag_corpus/autorag/runs/ollama_all_test").mkdir(parents=True, exist_ok=True)
        run_step("AutoRAG evaluate on test", ["autorag", "evaluate", "--config", "rag_corpus/autorag/runs/ollama_all_best_config.yaml", "--qa_data_path", test_qa, "--corpus_data_path", test_corpus, "--project_dir", "rag_corpus/autorag/runs/ollama_all_test"], cwd=root)
        run_step("Collect AutoRAG summary", python_cmd("scripts/rag_corpus/collect_autorag_summary.py", "--runs-root", "rag_corpus/autorag/runs", "--output-dir", "rag_corpus/autorag/runs/summary"), cwd=root)

    if not args.skip_runtime_config_apply:
        run_step("Apply runtime retrieval config", python_cmd("scripts/rag_corpus/apply_runtime_retrieval_config.py", "--base-config", "ui/config/benchmark/project-gemma4-bench_rag.toml", "--output-config", "ui/config/benchmark/project-gemma4-bench_rag_autorag.toml"), cwd=root)

    if not args.skip_nlv_smoke:
        run_step("NLV smoke no-RAG", python_cmd("scripts/benchmark/run_vegachat_compatible_benchmark.py", "--cases", "datasets/nlv_corpus/", "--config", "ui/config/benchmark/project-gemma4-bench_norag.toml", "--output-dir", "artifacts/benchmarks/nlv_no_rag_smoke", "--limit", str(args.nlv_smoke_limit), "--disable-analytics-tail"), cwd=root)
        run_step("NLV smoke RAG", python_cmd("scripts/benchmark/run_vegachat_compatible_benchmark.py", "--cases", "datasets/nlv_corpus/", "--config", "ui/config/benchmark/project-gemma4-bench_rag_autorag.toml", "--output-dir", "artifacts/benchmarks/nlv_rag_autorag_smoke", "--limit", str(args.nlv_smoke_limit), "--disable-analytics-tail"), cwd=root)
        run_step("Compare NLV smoke runs", python_cmd("scripts/benchmark/compare_runs.py", "--left", "artifacts/benchmarks/nlv_no_rag_smoke", "--right", "artifacts/benchmarks/nlv_rag_autorag_smoke", "--output", "artifacts/benchmarks/nlv_compare_autorag_smoke"), cwd=root)

    if not args.skip_infiagent_smoke:
        run_step("InfiAgent scan", python_cmd("scripts/benchmark/infiagent_scan.py", "--source-root", "Datasets/InfiAgent"), cwd=root)
        run_step("InfiAgent smoke benchmark", python_cmd("scripts/benchmark/run_infiagent_chart_grounded.py", "--source-root", "Datasets/InfiAgent", "--config", "ui/config/benchmark/project-gemma4-bench_rag_autorag.toml", "--output-dir", "artifacts/benchmarks/infiagent_rag_autorag_20", "--limit", "20"), cwd=root)

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()
