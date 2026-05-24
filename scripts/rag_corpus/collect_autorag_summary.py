from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.rag_corpus.common.io import ensure_dir, project_root, write_json, write_text


def _read_csv_safely(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _candidate_metric_columns(df: pd.DataFrame) -> list[str]:
    candidates = []
    for col in df.columns:
        lower = col.lower()
        if any(key in lower for key in ["retrieval", "recall", "precision", "f1", "mrr", "ndcg", "hit"]):
            if pd.api.types.is_numeric_dtype(df[col]):
                candidates.append(col)
    return candidates


def collect_summary(runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(runs_root.rglob("*.csv")):
        if "summary" not in csv_path.name.lower() and "trial" not in csv_path.name.lower() and "result" not in csv_path.name.lower():
            continue
        df = _read_csv_safely(csv_path)
        if df is None or df.empty:
            continue
        metric_cols = _candidate_metric_columns(df)
        if not metric_cols:
            rows.append({
                "run": _run_name(runs_root, csv_path),
                "file": str(csv_path),
                "rows": int(len(df)),
                "note": "no numeric retrieval metric columns detected",
            })
            continue
        for metric in metric_cols:
            best_idx = df[metric].astype(float).idxmax()
            best_row = df.loc[best_idx].to_dict()
            row: dict[str, Any] = {
                "run": _run_name(runs_root, csv_path),
                "file": str(csv_path),
                "rows": int(len(df)),
                "selected_metric": metric,
                "selected_metric_value": float(best_row.get(metric)),
            }
            for key, value in best_row.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    row[f"best.{key}"] = value
            rows.append(row)
    return rows


def _run_name(runs_root: Path, csv_path: Path) -> str:
    try:
        rel = csv_path.relative_to(runs_root)
    except ValueError:
        return ""
    return rel.parts[0] if rel.parts else ""


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# AutoRAG summary", "", f"Rows: **{len(rows)}**", ""]
    if not rows:
        return "\n".join(lines) + "\n"
    preferred = ["run", "selected_metric", "selected_metric_value", "file", "rows"]
    extra = sorted({key for row in rows for key in row.keys()} - set(preferred))
    headers = preferred + extra[:20]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")).replace("|", "/") for key in headers) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect AutoRAG run CSV files into one summary table.")
    parser.add_argument("--runs-root", default="rag_corpus/autorag/runs")
    parser.add_argument("--output-dir", default="rag_corpus/autorag/runs/summary")
    args = parser.parse_args()
    root = project_root()
    runs_root = root / args.runs_root
    output_dir = ensure_dir(root / args.output_dir)
    rows = collect_summary(runs_root)
    pd.DataFrame(rows).to_csv(output_dir / "autorag_summary.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "autorag_summary.json", {"rows": rows})
    write_text(output_dir / "autorag_summary.md", _markdown(rows))
    print(json.dumps({"rows": len(rows), "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
