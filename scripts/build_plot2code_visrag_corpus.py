from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

SUPPORTED_SUFFIXES = {".json", ".jsonl", ".ndjson", ".parquet", ".csv", ".tsv"}
TOKEN_RE = re.compile(r"[a-zA-Zа-яА-Я0-9_]+")

CHART_ALIASES = {
    "line": "line",
    "linechart": "line",
    "line_chart": "line",
    "timeseries": "line",
    "time_series": "line",
    "step": "line",
    "stem": "line",
    "bar": "bar",
    "barchart": "bar",
    "bar_chart": "bar",
    "column": "bar",
    "column_chart": "bar",
    "broken_barh": "bar",
    "scatter": "scatter",
    "scatterplot": "scatter",
    "scatter_plot": "scatter",
    "hist": "histogram",
    "histogram": "histogram",
    "distribution": "histogram",
    "box": "boxplot",
    "boxplot": "boxplot",
    "box_plot": "boxplot",
    "violin": "boxplot",
    "area": "line",
    "area_chart": "line",
    "fill_between": "line",
    "heatmap": "bar",
    "matshow": "bar",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized Plot2Code corpus for local VisRAG retrieval.")
    parser.add_argument("--src", required=True, help="Path to raw Plot2Code directory")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of records to write")
    parser.add_argument("--verbose", action="store_true", help="Print debug information")
    args = parser.parse_args()

    src_root = Path(args.src).resolve()
    out_path = Path(args.out).resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"Source path does not exist: {src_root}")

    rows = discover_raw_rows(src_root, verbose=args.verbose)
    normalized = normalize_rows(rows, verbose=args.verbose)

    if args.limit > 0:
        normalized = normalized[: args.limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, normalized)

    chart_counter = Counter(row["chart_type"] for row in normalized)
    print(f"Source root: {src_root}")
    print(f"Output:      {out_path}")
    print(f"Records:     {len(normalized)}")
    print("Chart types:")
    for chart_type, count in sorted(chart_counter.items()):
        print(f"  - {chart_type}: {count}")


def discover_raw_rows(src_root: Path, *, verbose: bool) -> list[dict[str, Any]]:
    files = [
        path for path in src_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    files.sort()

    if verbose:
        print(f"Discovered {len(files)} candidate raw files")

    all_rows: list[dict[str, Any]] = []
    for path in files:
        try:
            rows = list(read_file_as_rows(path))
            for index, row in enumerate(rows, start=1):
                row = dict(row)
                row["_source_file"] = path.as_posix()
                row["_source_index"] = index
                all_rows.append(row)
        except Exception as exc:
            if verbose:
                print(f"[WARN] skip file {path}: {exc}")

    if verbose:
        print(f"Collected {len(all_rows)} raw row candidates")
    return all_rows


def read_file_as_rows(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()

    if suffix in {".jsonl", ".ndjson"}:
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if isinstance(payload, dict):
                yield payload
        return

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        yield from unpack_json_payload(payload)
        return

    if suffix == ".parquet":
        try:
            df = pd.read_parquet(path)
        except ImportError as exc:
            raise RuntimeError(
                "Reading parquet requires pyarrow or fastparquet. "
                "Install one of them, for example: pip install pyarrow"
            ) from exc
        for row in df.to_dict(orient="records"):
            yield row
        return

    if suffix == ".csv":
        df = pd.read_csv(path)
        for row in df.to_dict(orient="records"):
            yield row
        return

    if suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
        for row in df.to_dict(orient="records"):
            yield row
        return


def unpack_json_payload(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(payload, dict):
        list_like_keys = ("records", "items", "data", "samples", "examples", "rows")
        for key in list_like_keys:
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return
        yield payload


def normalize_rows(rows: list[dict[str, Any]], *, verbose: bool) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for row in rows:
        item = normalize_single_row(row)
        if item is None:
            continue
        if item["id"] in seen_ids:
            continue
        seen_ids.add(item["id"])
        normalized.append(item)

    normalized.sort(key=lambda item: (item["chart_type"], item["id"]))

    if verbose:
        print(f"Normalized {len(normalized)} unique records")
    return normalized


def normalize_single_row(row: dict[str, Any]) -> dict[str, Any] | None:
    instruction = first_text(
        row,
        "instruction",
        "prompt",
        "query",
        "question",
        "text",
        "description",
        "detailed_description",
        "caption",
        "summary",
    )
    description = first_text(
        row,
        "description",
        "detailed_description",
        "caption",
        "summary",
        "text",
    )
    code = first_text(
        row,
        "code",
        "python_code",
        "source_code",
        "pyplot_code",
        "reference_code",
        "answer",
    )
    chart_type_raw = first_text(
        row,
        "chart_type",
        "plot_type",
        "chart_family",
        "type",
        "plot_kind",
    )

    chart_type = canonicalize_chart_type(
        chart_type_raw
        or infer_chart_type(
            instruction=instruction,
            description=description,
            code=code,
            source_file=str(row.get("_source_file", "")),
        )
    )

    if not chart_type:
        return None
    if not instruction and not description:
        return None

    example_id = (
        first_text(row, "id", "example_id", "sample_id", "uid", "plot_id")
        or build_fallback_id(row)
    )

    code_language = infer_code_language(code=code, source_file=str(row.get("_source_file", "")))
    domain = first_text(row, "domain", "topic", "category", "subject") or "general"
    tags = extract_tags(row, instruction=instruction, description=description, chart_type=chart_type)

    result = {
        "id": example_id,
        "chart_type": chart_type,
        "instruction": instruction or description or "",
        "description": description,
        "tags": tags,
        "code": code,
        "code_language": code_language,
        "domain": domain,
        "source": "Plot2Code",
        "source_file": str(row.get("_source_file", "")),
    }
    return result


def first_text(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def build_fallback_id(row: dict[str, Any]) -> str:
    base = f"{row.get('_source_file', '')}|{row.get('_source_index', '')}|{row.get('instruction', '')}|{row.get('description', '')}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    return f"plot2code-{digest}"


def canonicalize_chart_type(value: str | None) -> str:
    if not value:
        return ""
    normalized = re.sub(r"[^a-z0-9_]+", "_", value.strip().lower()).strip("_")
    if normalized in {"line", "bar", "scatter", "histogram", "boxplot"}:
        return normalized
    return CHART_ALIASES.get(normalized, "")


def infer_chart_type(
    *,
    instruction: str | None,
    description: str | None,
    code: str | None,
    source_file: str,
) -> str:
    text = " ".join(part for part in [instruction, description, code, source_file] if part).lower()

    checks = [
        ("scatter", ["scatter", "correlation", "relationship"]),
        ("histogram", ["histogram", "hist ", "distribution", "density"]),
        ("boxplot", ["boxplot", "box plot", "violin"]),
        ("bar", ["bar chart", "barplot", "bar ", "column chart", "broken_barh"]),
        ("line", ["line chart", "line ", "trend", "time series", "timeseries", "step", "stem"]),
    ]
    for chart_type, hints in checks:
        if any(hint in text for hint in hints):
            return chart_type
    return ""


def infer_code_language(*, code: str | None, source_file: str) -> str | None:
    text = f"{source_file}\n{code or ''}".lower()
    if ".r" in source_file.lower() or "ggplot" in text or "<-" in text:
        return "r"
    if ".py" in source_file.lower() or "matplotlib" in text or "plotly" in text or "plt." in text:
        return "python"
    return None


def extract_tags(
    row: dict[str, Any],
    *,
    instruction: str | None,
    description: str | None,
    chart_type: str,
) -> list[str]:
    explicit = row.get("tags") or row.get("keywords")
    if isinstance(explicit, list):
        return [str(item).strip() for item in explicit if str(item).strip()]
    if isinstance(explicit, str) and explicit.strip():
        return [part.strip() for part in re.split(r"[;,|]", explicit) if part.strip()]

    text = " ".join(part for part in [instruction, description] if part).lower()
    tags = [chart_type]

    if any(token in text for token in ["trend", "time", "timeline", "date", "series", "динам", "тренд"]):
        tags.append("trend analysis")
    if any(token in text for token in ["compare", "comparison", "rank", "top", "bottom", "сравн"]):
        tags.append("comparison")
    if any(token in text for token in ["distribution", "spread", "density", "распредел"]):
        tags.append("distribution analysis")
    if any(token in text for token in ["relationship", "correlation", "scatter", "связ", "завис"]):
        tags.append("relationship analysis")

    deduped: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        key = tag.lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(tag)
    return deduped


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()