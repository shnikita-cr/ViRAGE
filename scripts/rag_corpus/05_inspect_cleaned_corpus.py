from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_CLEANED_ROOT = "rag_corpus/cleaned"
DEFAULT_OUT_DIR = "rag_corpus/reports"

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
    "env",
}

TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".txt",
    ".md",
    ".markdown",
    ".yaml",
    ".yml",
    ".vl.json",
}


def file_kind(path: Path) -> str:
    name = path.name.lower()

    if name.endswith(".vl.json"):
        return ".vl.json"

    suffix = path.suffix.lower()
    return suffix if suffix else "[no_ext]"


def is_text_like(path: Path) -> bool:
    return file_kind(path) in TEXT_SUFFIXES


def iter_files(root: Path) -> list[Path]:
    files: list[Path] = []

    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue

        if path.is_file():
            files.append(path)

    return sorted(files)


def source_name(path: Path, root: Path) -> str:
    rel_parts = path.relative_to(root).parts

    if not rel_parts:
        return "__root__"

    if len(rel_parts) == 1:
        return "__root__"

    return rel_parts[0]


def size_mb(size_bytes: int) -> float:
    return round(size_bytes / 1024 / 1024, 3)


def read_json_sample(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "type": "json",
            "valid": False,
            "error": str(exc),
        }

    summary: dict[str, Any] = {
        "type": "json",
        "valid": True,
        "top_level_type": type(value).__name__,
    }

    if isinstance(value, dict):
        summary["top_level_keys"] = sorted(str(key) for key in value.keys())[:100]
        summary["top_level_key_count"] = len(value)

    if isinstance(value, list):
        summary["items_count"] = len(value)

        if value and isinstance(value[0], dict):
            keys = sorted(str(key) for key in value[0].keys())
            summary["first_item_keys"] = keys[:100]

    return summary


def read_jsonl_summary(path: Path, max_scan: int) -> dict[str, Any]:
    total_lines = 0
    scanned_lines = 0
    valid_lines = 0
    invalid_lines = 0
    key_counter: Counter[str] = Counter()
    corpus_type_counter: Counter[str] = Counter()
    mark_type_counter: Counter[str] = Counter()
    chart_pattern_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    sample_ids: list[str] = []
    errors: list[str] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            total_lines += 1
            stripped = line.strip()

            if not stripped:
                continue

            if scanned_lines >= max_scan:
                continue

            scanned_lines += 1

            try:
                value = json.loads(stripped)
            except Exception as exc:
                invalid_lines += 1

                if len(errors) < 10:
                    errors.append(f"line {line_number}: {exc}")

                continue

            if not isinstance(value, dict):
                invalid_lines += 1

                if len(errors) < 10:
                    errors.append(f"line {line_number}: expected JSON object")

                continue

            valid_lines += 1
            key_counter.update(str(key) for key in value.keys())

            doc_id = value.get("id")
            if isinstance(doc_id, str) and len(sample_ids) < 10:
                sample_ids.append(doc_id)

            corpus_type = value.get("corpus_type")
            if isinstance(corpus_type, str):
                corpus_type_counter[corpus_type] += 1

            mark_type = value.get("mark_type")
            if isinstance(mark_type, str):
                mark_type_counter[mark_type] += 1

            chart_pattern = value.get("chart_pattern")
            if isinstance(chart_pattern, str):
                chart_pattern_counter[chart_pattern] += 1

            source = value.get("source")
            if isinstance(source, str):
                source_counter[source] += 1

    return {
        "type": "jsonl",
        "total_lines": total_lines,
        "scanned_lines": scanned_lines,
        "valid_json_objects": valid_lines,
        "invalid_lines_in_scan": invalid_lines,
        "sample_ids": sample_ids,
        "keys": dict(key_counter.most_common()),
        "corpus_type": dict(corpus_type_counter.most_common()),
        "mark_type": dict(mark_type_counter.most_common()),
        "chart_pattern": dict(chart_pattern_counter.most_common()),
        "source": dict(source_counter.most_common()),
        "errors": errors,
    }


def read_delimited_summary(path: Path, delimiter: str, max_scan: int) -> dict[str, Any]:
    rows_scanned = 0
    header: list[str] = []
    error: str | None = None

    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file, delimiter=delimiter)

            for row_index, row in enumerate(reader):
                if row_index == 0:
                    header = row

                rows_scanned += 1

                if rows_scanned >= max_scan:
                    break
    except Exception as exc:
        error = str(exc)

    return {
        "type": "csv" if delimiter == "," else "tsv",
        "valid": error is None,
        "rows_scanned": rows_scanned,
        "header": header[:100],
        "column_count": len(header),
        "error": error,
    }


def inspect_file(path: Path, root: Path, max_scan: int) -> dict[str, Any]:
    stat = path.stat()
    rel_path = path.relative_to(root).as_posix()
    kind = file_kind(path)

    result: dict[str, Any] = {
        "path": rel_path,
        "name": path.name,
        "source": source_name(path, root),
        "kind": kind,
        "size_bytes": stat.st_size,
        "size_mb": size_mb(stat.st_size),
        "text_like": is_text_like(path),
    }

    if kind == ".jsonl":
        result["content_summary"] = read_jsonl_summary(path, max_scan)

    elif kind in {".json", ".vl.json"}:
        result["content_summary"] = read_json_sample(path)

    elif kind == ".csv":
        result["content_summary"] = read_delimited_summary(path, ",", max_scan)

    elif kind == ".tsv":
        result["content_summary"] = read_delimited_summary(path, "\t", max_scan)

    return result


def build_tree(root: Path, max_depth: int) -> list[dict[str, Any]]:
    directories: dict[str, dict[str, Any]] = {}

    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue

        rel = path.relative_to(root)
        depth = len(rel.parts)

        if depth > max_depth:
            continue

        if path.is_dir():
            directories[rel.as_posix()] = {
                "path": rel.as_posix(),
                "depth": depth,
                "file_count_direct": 0,
                "dir_count_direct": 0,
            }

    directories["."] = {
        "path": ".",
        "depth": 0,
        "file_count_direct": 0,
        "dir_count_direct": 0,
    }

    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue

        rel_parent = path.parent.relative_to(root).as_posix()
        rel_parent = "." if rel_parent == "." else rel_parent

        if rel_parent not in directories:
            continue

        if path.is_file():
            directories[rel_parent]["file_count_direct"] += 1

        elif path.is_dir():
            directories[rel_parent]["dir_count_direct"] += 1

    return sorted(
        directories.values(),
        key=lambda item: (item["depth"], item["path"]),
    )


def build_inventory(cleaned_root: Path, max_scan: int, tree_depth: int) -> dict[str, Any]:
    files = iter_files(cleaned_root)
    inspected_files = [inspect_file(path, cleaned_root, max_scan) for path in files]

    total_bytes = sum(item["size_bytes"] for item in inspected_files)
    ext_counter = Counter(item["kind"] for item in inspected_files)
    source_counter = Counter(item["source"] for item in inspected_files)

    sources: dict[str, dict[str, Any]] = {}

    for item in inspected_files:
        source = item["source"]

        if source not in sources:
            sources[source] = {
                "source": source,
                "file_count": 0,
                "total_bytes": 0,
                "total_mb": 0.0,
                "extensions": Counter(),
                "files": [],
            }

        sources[source]["file_count"] += 1
        sources[source]["total_bytes"] += item["size_bytes"]
        sources[source]["extensions"][item["kind"]] += 1
        sources[source]["files"].append(item)

    source_list: list[dict[str, Any]] = []

    for source in sorted(sources):
        source_item = sources[source]
        source_item["total_mb"] = size_mb(source_item["total_bytes"])
        source_item["extensions"] = dict(source_item["extensions"].most_common())
        source_item["largest_files"] = sorted(
            source_item["files"],
            key=lambda item: item["size_bytes"],
            reverse=True,
        )[:15]

        source_list.append(source_item)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cleaned_root": cleaned_root.as_posix(),
        "total_sources": len(source_counter),
        "total_files": len(inspected_files),
        "total_bytes": total_bytes,
        "total_mb": size_mb(total_bytes),
        "extensions": dict(ext_counter.most_common()),
        "sources": source_list,
        "directory_tree": build_tree(cleaned_root, tree_depth),
    }


def md_table_row(values: list[Any]) -> str:
    return "| " + " | ".join(str(value) for value in values) + " |"


def format_counter_table(title: str, counter: dict[str, int]) -> list[str]:
    lines: list[str] = []

    lines.append(f"### {title}")
    lines.append("")

    if not counter:
        lines.append("_No data._")
        lines.append("")
        return lines

    lines.append("| Value | Count |")
    lines.append("|---|---:|")

    for key, value in counter.items():
        lines.append(md_table_row([f"`{key}`", value]))

    lines.append("")
    return lines


def build_markdown_report(inventory: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append("# Cleaned corpus inventory")
    lines.append("")
    lines.append(f"Generated at: `{inventory['generated_at']}`")
    lines.append(f"Cleaned root: `{inventory['cleaned_root']}`")
    lines.append(f"Total sources: **{inventory['total_sources']}**")
    lines.append(f"Total files: **{inventory['total_files']}**")
    lines.append(f"Total size: **{inventory['total_mb']} MB**")
    lines.append("")

    lines.extend(format_counter_table("Extensions", inventory["extensions"]))

    lines.append("## Sources")
    lines.append("")
    lines.append("| Source | Files | Size MB | Extensions |")
    lines.append("|---|---:|---:|---|")

    for source in inventory["sources"]:
        extensions = ", ".join(
            f"`{ext}`:{count}"
            for ext, count in source["extensions"].items()
        )
        lines.append(
            md_table_row(
                [
                    f"`{source['source']}`",
                    source["file_count"],
                    source["total_mb"],
                    extensions,
                ]
            )
        )

    lines.append("")

    lines.append("## Directory tree")
    lines.append("")
    lines.append("| Path | Direct dirs | Direct files |")
    lines.append("|---|---:|---:|")

    for item in inventory["directory_tree"]:
        lines.append(
            md_table_row(
                [
                    f"`{item['path']}`",
                    item["dir_count_direct"],
                    item["file_count_direct"],
                ]
            )
        )

    lines.append("")

    for source in inventory["sources"]:
        lines.append(f"## Source: `{source['source']}`")
        lines.append("")
        lines.append(f"- Files: **{source['file_count']}**")
        lines.append(f"- Size: **{source['total_mb']} MB**")
        lines.append("")

        lines.extend(format_counter_table("Extensions", source["extensions"]))

        lines.append("### Largest files")
        lines.append("")
        lines.append("| Path | Size MB |")
        lines.append("|---|---:|")

        for file_item in source["largest_files"]:
            lines.append(
                md_table_row(
                    [
                        f"`{file_item['path']}`",
                        file_item["size_mb"],
                    ]
                )
            )

        lines.append("")

        lines.append("### Structured file summaries")
        lines.append("")

        structured_files = [
            item
            for item in source["files"]
            if "content_summary" in item
        ]

        if not structured_files:
            lines.append("_No structured files found._")
            lines.append("")
            continue

        for file_item in structured_files:
            summary = file_item["content_summary"]
            lines.append(f"#### `{file_item['path']}`")
            lines.append("")
            lines.append(f"- Type: `{summary.get('type')}`")
            lines.append(f"- Size MB: `{file_item['size_mb']}`")

            if summary.get("type") == "jsonl":
                lines.append(f"- Total lines: `{summary.get('total_lines')}`")
                lines.append(f"- Scanned lines: `{summary.get('scanned_lines')}`")
                lines.append(f"- Valid JSON objects: `{summary.get('valid_json_objects')}`")
                lines.append(f"- Invalid lines in scan: `{summary.get('invalid_lines_in_scan')}`")

                if summary.get("sample_ids"):
                    ids = ", ".join(f"`{item}`" for item in summary["sample_ids"])
                    lines.append(f"- Sample ids: {ids}")

                for key in ("corpus_type", "mark_type", "chart_pattern", "source"):
                    values = summary.get(key)

                    if values:
                        formatted = ", ".join(
                            f"`{name}`:{count}"
                            for name, count in values.items()
                        )
                        lines.append(f"- {key}: {formatted}")

                if summary.get("keys"):
                    keys = ", ".join(
                        f"`{name}`:{count}"
                        for name, count in list(summary["keys"].items())[:30]
                    )
                    lines.append(f"- Keys: {keys}")

                if summary.get("errors"):
                    errors = "; ".join(summary["errors"][:3])
                    lines.append(f"- Errors: `{errors}`")

            elif summary.get("type") in {"json", "csv", "tsv"}:
                for key, value in summary.items():
                    if key == "type":
                        continue

                    if isinstance(value, list):
                        value = ", ".join(f"`{item}`" for item in value[:30])

                    lines.append(f"- {key}: {value}")

            lines.append("")

    return "\n".join(lines)


def write_outputs(inventory: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "cleaned_inventory.json"
    md_path = out_dir / "cleaned_inventory.md"

    json_path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_path.write_text(
        build_markdown_report(inventory),
        encoding="utf-8",
    )

    return json_path, md_path


def resolve_cleaned_root(path_arg: str) -> Path:
    path = Path(path_arg)

    if path.exists():
        return path

    fallback = Path("ra_corpus/cleaned")

    if path_arg == DEFAULT_CLEANED_ROOT and fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Cleaned root does not exist: {path}. "
        f"Pass --cleaned-root explicitly if your path is different."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect cleaned RAG corpus files and generate JSON/Markdown reports."
    )

    parser.add_argument(
        "--cleaned-root",
        default=DEFAULT_CLEANED_ROOT,
        help="Path to cleaned corpus directory.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Path to reports directory.",
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        default=5000,
        help="Max records/rows to scan per JSONL/CSV/TSV file.",
    )
    parser.add_argument(
        "--tree-depth",
        type=int,
        default=4,
        help="Max directory depth for tree report.",
    )

    args = parser.parse_args()

    cleaned_root = resolve_cleaned_root(args.cleaned_root)
    out_dir = Path(args.out_dir)

    inventory = build_inventory(
        cleaned_root=cleaned_root,
        max_scan=args.max_scan,
        tree_depth=args.tree_depth,
    )

    json_path, md_path = write_outputs(inventory, out_dir)

    print(f"Cleaned root: {cleaned_root}")
    print(f"Sources: {inventory['total_sources']}")
    print(f"Files: {inventory['total_files']}")
    print(f"Size MB: {inventory['total_mb']}")
    print(f"Saved JSON report: {json_path}")
    print(f"Saved Markdown report: {md_path}")


if __name__ == "__main__":
    main()
