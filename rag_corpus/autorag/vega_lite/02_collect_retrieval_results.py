from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


AUTORAG_ROOT = Path("rag_corpus/autorag/vega_lite")
TRIALS_ROOT = AUTORAG_ROOT / "trials"
REPORTS_ROOT = AUTORAG_ROOT / "reports"
OUT_CSV = REPORTS_ROOT / "retrieval_comparison.csv"
OUT_MD = REPORTS_ROOT / "retrieval_comparison.md"
OUT_JSON = REPORTS_ROOT / "retrieval_comparison.json"

PREFERRED_SORT_COLUMNS = [
    "retrieval_mrr",
    "retrieval_ndcg",
    "retrieval_recall",
    "retrieval_f1",
    "retrieval_precision",
]


def parse_mapping_like(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value

    if not isinstance(value, str) or not value.strip():
        return {}

    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return {"raw": value}

    if isinstance(parsed, dict):
        return parsed

    return {"raw": parsed}


def find_summary_files() -> list[Path]:
    if not TRIALS_ROOT.exists():
        return []

    return sorted(TRIALS_ROOT.rglob("summary.csv"))


def infer_top_k(item: dict[str, Any], module_params: dict[str, Any]) -> Any:
    for key in ("top_k", "k"):
        if key in module_params and module_params[key] not in (None, ""):
            return module_params[key]

    for key in ("top_k", "node_top_k", "retrieval_top_k"):
        value = item.get(key)
        if value not in (None, ""):
            return value

    node_params = parse_mapping_like(item.get("node_params"))
    for key in ("top_k", "k"):
        if key in node_params and node_params[key] not in (None, ""):
            return node_params[key]

    searchable_text = " ".join(
        str(item.get(key, ""))
        for key in (
            "node_line_name",
            "node_name",
            "config_name",
            "summary_path",
            "trial_path",
        )
    )
    match = re.search(r"topk[_-]?(\d+)", searchable_text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None


def load_summary(path: Path) -> list[dict[str, Any]]:
    relative_parts = path.relative_to(TRIALS_ROOT).parts

    qa_name = relative_parts[0] if len(relative_parts) >= 3 else "unknown_qa"
    config_name = relative_parts[1] if len(relative_parts) >= 3 else "unknown_config"

    dataframe = pd.read_csv(path)
    rows: list[dict[str, Any]] = []

    for _, row in dataframe.iterrows():
        item = row.to_dict()
        module_params = parse_mapping_like(item.get("module_params"))
        node_params = parse_mapping_like(item.get("node_params"))

        item["qa_name"] = qa_name
        item["config_name"] = config_name
        item["summary_path"] = path.as_posix()
        item["node_line_name"] = item.get("node_line_name") or item.get("node_line") or item.get("line_name")
        item["node_type"] = item.get("node_type") or item.get("node")
        item["top_k"] = infer_top_k(item, module_params)
        item["bm25_tokenizer"] = module_params.get("bm25_tokenizer")
        item["vectordb"] = module_params.get("vectordb")
        item["normalize_method"] = module_params.get("normalize_method")
        item["weight_range"] = module_params.get("weight_range")
        item["test_weight_size"] = module_params.get("test_weight_size")
        item["module_params_json"] = json.dumps(module_params, ensure_ascii=False, sort_keys=True)
        item["node_params_json"] = json.dumps(node_params, ensure_ascii=False, sort_keys=True)

        rows.append(item)

    return rows


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    available_sort_columns = [column for column in PREFERRED_SORT_COLUMNS if any(column in row for row in rows)]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("qa_name", "unknown_qa")), []).append(row)

    result: list[dict[str, Any]] = []
    for qa_name in sorted(grouped):
        group = grouped[qa_name]
        group.sort(
            key=lambda row: tuple(
                float(row.get(column, float("-inf"))) if not pd.isna(row.get(column, float("-inf"))) else float("-inf")
                for column in available_sort_columns
            ),
            reverse=True,
        )
        result.extend(group)

    return result


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "qa_name",
        "config_name",
        "node_line_name",
        "node_type",
        "module_name",
        "top_k",
        "bm25_tokenizer",
        "vectordb",
        "normalize_method",
        "retrieval_recall",
        "retrieval_mrr",
        "retrieval_ndcg",
        "retrieval_f1",
        "retrieval_precision",
        "execution_time",
        "is_best",
        "module_params_json",
        "node_params_json",
        "summary_path",
    ]

    extra_columns = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key not in columns and not key.startswith("Unnamed")
        }
    )
    all_columns = columns + [column for column in extra_columns if column not in columns]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=all_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")

    return str(value)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# AutoRAG Vega-Lite retrieval comparison")
    lines.append("")
    lines.append(f"Rows: **{len(rows)}**")
    lines.append("")

    if not rows:
        lines.append("No `summary.csv` files found under `rag_corpus/autorag/vega_lite/trials`.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("qa_name", "unknown_qa")), []).append(row)

    for qa_name in sorted(grouped):
        group = grouped[qa_name]
        lines.append(f"## `{qa_name}`")
        lines.append("")
        lines.append("| Rank | Node line | Node type | Module | top_k | tokenizer | vectordb | recall | mrr | ndcg | f1 | precision | time |")
        lines.append("|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|")

        for index, row in enumerate(group, start=1):
            lines.append(
                "| "
                f"{index} | "
                f"`{format_value(row.get('node_line_name'))}` | "
                f"`{format_value(row.get('node_type'))}` | "
                f"`{format_value(row.get('module_name'))}` | "
                f"{format_value(row.get('top_k'))} | "
                f"`{format_value(row.get('bm25_tokenizer'))}` | "
                f"`{format_value(row.get('vectordb'))}` | "
                f"{format_value(row.get('retrieval_recall'))} | "
                f"{format_value(row.get('retrieval_mrr'))} | "
                f"{format_value(row.get('retrieval_ndcg'))} | "
                f"{format_value(row.get('retrieval_f1'))} | "
                f"{format_value(row.get('retrieval_precision'))} | "
                f"{format_value(row.get('execution_time'))} |"
            )

        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    summary_files = find_summary_files()
    rows: list[dict[str, Any]] = []

    for path in summary_files:
        rows.extend(load_summary(path))

    rows = sort_rows(rows)

    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(rows, OUT_CSV)
    write_markdown(rows, OUT_MD)
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"Summary files: {len(summary_files)}")
    print(f"Rows: {len(rows)}")
    print(f"CSV: {OUT_CSV}")
    print(f"Markdown: {OUT_MD}")
    print(f"JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
