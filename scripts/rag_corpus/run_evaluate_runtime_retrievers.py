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
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.rag_corpus.common.io import ensure_dir, project_root, write_json, write_text
from src.domain.models import QueryRequestAnalysisResult, VisRAGRuleDocument
from src.visrag_core.rule_retrieval import RuleRetrieverOptions, build_rule_retriever

DEFAULT_CORPUS = "rag_corpus/autorag/virage_rules/corpus.parquet"
DEFAULT_QA = "rag_corpus/autorag/virage_rules/qa.parquet"
DEFAULT_OUTPUT_DIR = "rag_corpus/autorag/virage_rules/runtime_retriever_eval"
DEFAULT_BACKENDS = ["keyword", "bm25", "tfidf"]
DEFAULT_TOP_K = [1, 2, 3, 5, 8]
SUPPORTED_RUNTIME_TOP_K_KEYS: dict[str, str] = {}


@dataclass(frozen=True)
class RetrievalQuestion:
    qid: str
    query: str
    retrieval_gt: list[str]
    record_type: str | None


def _safe_json_loads(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def _metadata(value: Any) -> dict[str, Any]:
    data = _safe_json_loads(value, {})
    return data if isinstance(data, dict) else {}


def _load_corpus(path: Path) -> list[VisRAGRuleDocument]:
    df = pd.read_parquet(path)
    required = {"doc_id", "contents", "metadata"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Corpus parquet is missing columns: {sorted(missing)}")
    documents: list[VisRAGRuleDocument] = []
    for row in df.to_dict(orient="records"):
        metadata = _metadata(row.get("metadata"))
        record_type = str(metadata.get("record_type") or "readability_rule")
        documents.append(
            VisRAGRuleDocument(
                doc_id=str(row["doc_id"]),
                record_type=record_type,  # type: ignore[arg-type]
                title=str(metadata.get("title") or row["doc_id"]),
                retrieval_text=str(row.get("contents") or ""),
                prompt_text=str(row.get("contents") or ""),
                metadata=metadata,
            )
        )
    return documents


def _load_questions(path: Path) -> list[RetrievalQuestion]:
    df = pd.read_parquet(path)
    required = {"qid", "query", "retrieval_gt", "metadata"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"QA parquet is missing columns: {sorted(missing)}")
    questions: list[RetrievalQuestion] = []
    for row in df.to_dict(orient="records"):
        metadata = _metadata(row.get("metadata"))
        retrieval_gt = _safe_json_loads(row.get("retrieval_gt"), [])
        if isinstance(retrieval_gt, str):
            retrieval_gt = [retrieval_gt]
        if not isinstance(retrieval_gt, list):
            retrieval_gt = []
        questions.append(
            RetrievalQuestion(
                qid=str(row["qid"]),
                query=str(row["query"]),
                retrieval_gt=[str(item) for item in retrieval_gt],
                record_type=str(metadata.get("record_type") or "") or None,
            )
        )
    return questions


def _precision_at_k(ranked_ids: list[str], gt: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(ranked_ids[:k]) & gt) / k


def _recall_at_k(ranked_ids: list[str], gt: set[str], k: int) -> float:
    if not gt:
        return 0.0
    return len(set(ranked_ids[:k]) & gt) / len(gt)


def _hit_at_k(ranked_ids: list[str], gt: set[str], k: int) -> float:
    return 1.0 if set(ranked_ids[:k]) & gt else 0.0


def _mrr_at_k(ranked_ids: list[str], gt: set[str], k: int) -> float:
    for index, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in gt:
            return 1.0 / index
    return 0.0


def _ndcg_at_k(ranked_ids: list[str], gt: set[str], k: int) -> float:
    dcg = 0.0
    for index, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in gt:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(gt), k)
    if ideal_hits <= 0:
        return 0.0
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _query_analysis(query: str) -> QueryRequestAnalysisResult:
    return QueryRequestAnalysisResult(
        normalized_query=query,
        analysis_task="retrieval_evaluation",
        confidence=1.0,
    )


def _evaluate_backend(
        *,
        backend: str,
        documents: list[VisRAGRuleDocument],
        questions: list[RetrievalQuestion],
        top_k_values: list[int],
        embedding_provider: str | None,
        embedding_model: str | None,
        embedding_base_url: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    options = RuleRetrieverOptions(
        backend=backend,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_base_url=embedding_base_url,
    )
    retriever = build_rule_retriever(options)
    max_k = max(top_k_values)
    case_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    documents_by_type: dict[str, list[VisRAGRuleDocument]] = {}
    for document in documents:
        documents_by_type.setdefault(document.record_type, []).append(document)

    for question in questions:
        gt = set(question.retrieval_gt)
        candidate_documents = documents_by_type.get(question.record_type or "", documents)
        ranked = retriever.rank(
            candidate_documents,
            query=question.query,
            query_analysis=_query_analysis(question.query),
            corpus_key=f"autorag_eval:{backend}:{question.record_type or 'all'}",
        )
        ranked_ids = [item.doc_id for item in ranked[:max_k]]
        row: dict[str, Any] = {
            "backend": backend,
            "qid": question.qid,
            "record_type": question.record_type or "",
            "gt": json.dumps(question.retrieval_gt, ensure_ascii=False),
            "top_docs": json.dumps(ranked_ids, ensure_ascii=False),
        }
        for k in top_k_values:
            row[f"hit@{k}"] = _hit_at_k(ranked_ids, gt, k)
            row[f"recall@{k}"] = _recall_at_k(ranked_ids, gt, k)
            row[f"precision@{k}"] = _precision_at_k(ranked_ids, gt, k)
            row[f"mrr@{k}"] = _mrr_at_k(ranked_ids, gt, k)
            row[f"ndcg@{k}"] = _ndcg_at_k(ranked_ids, gt, k)
        case_rows.append(row)

    for record_type in ["__all__", *sorted({question.record_type or "" for question in questions})]:
        subset = [row for row in case_rows if record_type == "__all__" or row["record_type"] == record_type]
        if not subset:
            continue
        for k in top_k_values:
            metric_rows.append({
                "backend": backend,
                "record_type": record_type,
                "top_k": k,
                "cases": len(subset),
                "hit": _mean(row[f"hit@{k}"] for row in subset),
                "recall": _mean(row[f"recall@{k}"] for row in subset),
                "precision": _mean(row[f"precision@{k}"] for row in subset),
                "mrr": _mean(row[f"mrr@{k}"] for row in subset),
                "ndcg": _mean(row[f"ndcg@{k}"] for row in subset),
            })
    return case_rows, metric_rows


def _mean(values) -> float:
    items = list(values)
    return round(sum(float(value) for value in items) / len(items), 6) if items else 0.0


def _best_rows(metric_rows: list[dict[str, Any]], select_by: str) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in metric_rows:
        record_type = str(row["record_type"])
        current = best.get(record_type)
        candidate_key = (
            float(row.get(select_by, 0.0)),
            float(row.get("mrr", 0.0)),
            float(row.get("hit", 0.0)),
            -int(row.get("top_k", 0)),
        )
        if current is None:
            best[record_type] = row
            continue
        current_key = (
            float(current.get(select_by, 0.0)),
            float(current.get("mrr", 0.0)),
            float(current.get("hit", 0.0)),
            -int(current.get("top_k", 0)),
        )
        if candidate_key > current_key:
            best[record_type] = row
    return best


def _runtime_config_toml(best: dict[str, dict[str, Any]]) -> str:
    overall = best.get("__all__") or {}
    backend = str(overall.get("backend") or "bm25")
    top_k = max(1, int(overall.get("top_k") or 8))
    if backend in {"bm25", "keyword", "tfidf"}:
        retrieval_backend = "lexical"
    else:
        retrieval_backend = "semantic"
    lines = [
        "[settings]",
        f'visrag_retrieval_backend = "{retrieval_backend}"',
        f"visrag_top_k_chunks = {top_k}",
        'visrag_embedding_provider = "ollama"',
        'visrag_embedding_model = "nomic-embed-text:latest"',
        'visrag_embedding_base_url = "http://localhost:11434"',
        'visrag_chroma_persist_dir = "./resources/chroma/virage_guidance_chunks_nomic_embed_text_latest"',
        'visrag_chroma_collection_name = "virage_guidance_chunks_nomic_embed_text_latest"',
    ]
    return "\n".join(lines) + "\n"


def _markdown_summary(metric_rows: list[dict[str, Any]], best: dict[str, dict[str, Any]], select_by: str) -> str:
    lines = [
        "# Runtime retrieval evaluation",
        "",
        f"Selection metric: `{select_by}`",
        "",
        "## Recommended runtime settings",
        "",
        "```toml",
        _runtime_config_toml(best).strip(),
        "```",
        "",
        "## Best rows",
        "",
        "| record_type | backend | top_k | hit | recall | precision | mrr | ndcg | cases |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record_type, row in sorted(best.items()):
        lines.append(
            f"| {record_type} | {row.get('backend')} | {row.get('top_k')} | {row.get('hit')} | "
            f"{row.get('recall')} | {row.get('precision')} | {row.get('mrr')} | {row.get('ndcg')} | {row.get('cases')} |"
        )
    lines.extend(["", "## All metrics", ""])
    if metric_rows:
        headers = ["backend", "record_type", "top_k", "cases", "hit", "recall", "precision", "mrr", "ndcg"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join("---" for _ in headers) + "|")
        for row in metric_rows:
            lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ViRAGE runtime retrievers on AutoRAG QA/corpus parquet files.")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--qa", default=DEFAULT_QA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS)
    parser.add_argument("--top-k", nargs="+", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--select-by", choices=["hit", "recall", "precision", "mrr", "ndcg"], default="mrr")
    parser.add_argument("--embedding-provider", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-base-url", default=None)
    args = parser.parse_args()

    root = project_root()
    corpus_path = root / args.corpus
    qa_path = root / args.qa
    output_dir = root / args.output_dir
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus parquet not found: {corpus_path}")
    if not qa_path.is_file():
        raise FileNotFoundError(f"QA parquet not found: {qa_path}")

    documents = _load_corpus(corpus_path)
    questions = _load_questions(qa_path)
    top_k_values = sorted({value for value in args.top_k if value > 0})
    if not top_k_values:
        raise ValueError("At least one positive --top-k value is required.")

    all_case_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []
    for backend in args.backends:
        case_rows, metric_rows = _evaluate_backend(
            backend=backend,
            documents=documents,
            questions=questions,
            top_k_values=top_k_values,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            embedding_base_url=args.embedding_base_url,
        )
        all_case_rows.extend(case_rows)
        all_metric_rows.extend(metric_rows)

    best = _best_rows(all_metric_rows, args.select_by)
    ensure_dir(output_dir)
    pd.DataFrame(all_case_rows).to_csv(output_dir / "runtime_retrieval_cases.csv", index=False)
    pd.DataFrame(all_metric_rows).to_csv(output_dir / "runtime_retrieval_metrics.csv", index=False)
    write_json(output_dir / "runtime_retrieval_metrics.json", {"rows": all_metric_rows, "best": best})
    write_text(output_dir / "recommended_runtime_config.toml", _runtime_config_toml(best))
    write_text(output_dir / "runtime_retrieval_report.md", _markdown_summary(all_metric_rows, best, args.select_by))
    print(json.dumps({"cases": len(questions), "documents": len(documents), "output_dir": str(output_dir), "best": best.get("__all__")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
