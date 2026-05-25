from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_csv_safely(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def score_columns(df: pd.DataFrame) -> list[str]:
    keys = [
        "retrieval_f1",
        "retrieval_recall",
        "retrieval_precision",
        "mrr",
        "ndcg",
        "hit",
        "score",
        "execution_time",
        "latency",
    ]
    cols: list[str] = []
    for col in df.columns:
        low = col.lower()
        if any(key in low for key in keys):
            cols.append(col)
    return cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect AutoRAG CSV outputs into one summary table.")
    parser.add_argument("--runs-root", default="rag_corpus/autorag/runs")
    parser.add_argument("--output-dir", default="rag_corpus/autorag/runs/summary")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[pd.DataFrame] = []
    for csv_path in sorted(runs_root.rglob("*.csv")):
        if output_dir in csv_path.parents:
            continue
        df = read_csv_safely(csv_path)
        if df is None or df.empty:
            continue

        run_name = csv_path.relative_to(runs_root).parts[0]
        selected_cols = list(dict.fromkeys([*score_columns(df), *df.columns[:12].tolist()]))
        slim = df[selected_cols].copy()
        slim.insert(0, "run_name", run_name)
        slim.insert(1, "source_csv", str(csv_path))
        rows.append(slim)

    if not rows:
        print("No AutoRAG CSV files found.")
        return

    summary = pd.concat(rows, ignore_index=True, sort=False)
    summary_csv = output_dir / "autorag_summary.csv"
    summary_json = output_dir / "autorag_summary.json"
    summary_md = output_dir / "autorag_summary.md"

    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary.to_json(summary_json, orient="records", force_ascii=False, indent=2)

    metric_cols = [col for col in summary.columns if any(k in col.lower() for k in ["retrieval_f1", "retrieval_recall", "retrieval_precision", "mrr", "ndcg", "hit", "score"])]
    sort_col = metric_cols[0] if metric_cols else None
    view = summary.copy()
    if sort_col:
        view = view.sort_values(sort_col, ascending=False)

    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# AutoRAG summary\n\n")
        f.write(f"Rows: {len(summary)}\n\n")
        if sort_col:
            f.write(f"Sorted by: `{sort_col}` descending.\n\n")
        f.write(view.head(50).to_markdown(index=False))
        f.write("\n")

    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")
    print(view.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
