from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))


import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.rag_corpus.common.hashing import stable_hash
from scripts.rag_corpus.common.io import ensure_dir, project_root, read_jsonl, write_json, write_text
from scripts.rag_corpus.common.schemas import AutoRAGQuestion, RagRuleRecord

DEFAULT_INPUT = "rag_corpus/processed/all_rules.validated.jsonl"
DEFAULT_OUT = "rag_corpus/autorag/virage_rules/qa.parquet"
DEFAULT_REPORT_JSON = "rag_corpus/autorag/virage_rules/qa_export_report.json"
DEFAULT_REPORT_MD = "rag_corpus/autorag/virage_rules/qa_export_report.md"

QUESTION_TEMPLATES = {
    "chart_pattern": "A user asks for a {task} visualization with chart family {chart_family}. Which chart pattern rule should be retrieved?",
    "readability_rule": "A generated chart needs better readability for {chart_family} / {task}. Which readability rule should be retrieved?",
    "scale_plot_area_rule": "A chart has poor plot area utilization or outlier-compressed marks. Which scale or plot-area rule should be retrieved?",
    "vlm_readability_rule": "A static PNG chart must be readable by a VLM judge. Which VLM readability rule should be retrieved?",
    "domain_semantics_rule": "A dataset or user request contains domain-specific terms. Which domain semantics rule should be retrieved?",
}


def _question_for(record: RagRuleRecord) -> AutoRAGQuestion:
    task = record.task or "the requested task"
    family = record.chart_family or "the suitable chart"
    template = QUESTION_TEMPLATES.get(record.record_type, "Which ViRAGE rule should be retrieved?")
    query = template.format(task=task, chart_family=family)
    if record.applies_when:
        query += " Context: " + "; ".join(record.applies_when[:3])
    generation_gt = record.prompt_text
    return AutoRAGQuestion(
        qid=f"qa__{record.record_type}__{stable_hash(record.doc_id)}",
        query=query,
        generation_gt=generation_gt,
        retrieval_gt=[record.doc_id],
        metadata={"record_type": record.record_type, "source_doc_id": record.doc_id},
    )


def export_autorag_qa(input_path: Path, out_path: Path) -> dict[str, Any]:
    records = [RagRuleRecord.model_validate(raw) for raw in read_jsonl(input_path)]
    if not records:
        raise RuntimeError(f"Cannot export AutoRAG QA from empty input: {input_path}")
    questions = [_question_for(record) for record in records]
    rows = []
    for item in questions:
        rows.append({
            "qid": item.qid,
            "query": item.query,
            "generation_gt": item.generation_gt,
            # AutoRAG accepts retrieval_gt as a list; keeping it as a list avoids
            # an unnecessary JSON-string round trip before validation/evaluation.
            "retrieval_gt": item.retrieval_gt,
            "metadata": item.metadata,
        })
    ensure_dir(out_path.parent)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return {"questions": len(rows), "output": str(out_path)}


def _md(report: dict[str, Any]) -> str:
    return f"# AutoRAG QA export\n\nQuestions: **{report['questions']}**\n\nOutput: `{report['output']}`\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export validated ViRAGE rule records to AutoRAG qa.parquet.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--report-json", default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", default=DEFAULT_REPORT_MD)
    args = parser.parse_args()
    root = project_root()
    report = export_autorag_qa(root / args.input, root / args.output)
    write_json(root / args.report_json, report)
    write_text(root / args.report_md, _md(report))
    print(report)


if __name__ == "__main__":
    main()
