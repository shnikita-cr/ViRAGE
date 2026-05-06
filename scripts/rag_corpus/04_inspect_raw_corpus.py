from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

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

INTERESTING_SUFFIXES = {
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".txt",
    ".md",
    ".markdown",
    ".yaml",
    ".yml",
    ".ipynb",
    ".vl.json",
}


def file_kind(path: Path) -> str:
    name = path.name.lower()

    if name.endswith(".vl.json"):
        return ".vl.json"

    suffix = path.suffix.lower()
    return suffix if suffix else "[no_ext]"


def is_interesting(path: Path) -> bool:
    name = path.name.lower()

    if name.endswith(".vl.json"):
        return True

    return path.suffix.lower() in INTERESTING_SUFFIXES


def walk_files(root: Path):
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue

        if path.is_file():
            yield path


def inspect_source(source_dir: Path, raw_root: Path) -> dict[str, Any]:
    files = []
    ext_counter: Counter[str] = Counter()
    interesting_counter: Counter[str] = Counter()
    total_bytes = 0

    interesting_samples_by_type: dict[str, list[str]] = defaultdict(list)
    largest_files: list[dict[str, Any]] = []

    for path in walk_files(source_dir):
        try:
            size = path.stat().st_size
        except OSError:
            continue

        rel_path = path.relative_to(raw_root).as_posix()
        kind = file_kind(path)

        files.append(path)
        ext_counter[kind] += 1
        total_bytes += size

        largest_files.append(
            {
                "path": rel_path,
                "size_bytes": size,
            }
        )

        if is_interesting(path):
            interesting_counter[kind] += 1

            if len(interesting_samples_by_type[kind]) < 10:
                interesting_samples_by_type[kind].append(rel_path)

    largest_files = sorted(
        largest_files,
        key=lambda item: item["size_bytes"],
        reverse=True,
    )[:15]

    return {
        "source": source_dir.name,
        "path": source_dir.relative_to(raw_root).as_posix(),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / 1024 / 1024, 3),
        "extensions": dict(ext_counter.most_common()),
        "interesting_files": dict(interesting_counter.most_common()),
        "interesting_samples_by_type": dict(interesting_samples_by_type),
        "largest_files": largest_files,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append("# Raw corpus inventory")
    lines.append("")
    lines.append(f"Raw root: `{report['raw_root']}`")
    lines.append(f"Total sources: **{report['total_sources']}**")
    lines.append(f"Total files: **{report['total_files']}**")
    lines.append(f"Total size: **{report['total_mb']} MB**")
    lines.append("")

    lines.append("## Sources")
    lines.append("")
    lines.append("| Source | Files | Size MB | Interesting files |")
    lines.append("|---|---:|---:|---:|")

    for source in report["sources"]:
        interesting_count = sum(source["interesting_files"].values())
        lines.append(
            f"| `{source['source']}` | "
            f"{source['file_count']} | "
            f"{source['total_mb']} | "
            f"{interesting_count} |"
        )

    lines.append("")

    for source in report["sources"]:
        lines.append(f"## `{source['source']}`")
        lines.append("")
        lines.append(f"- Files: **{source['file_count']}**")
        lines.append(f"- Size: **{source['total_mb']} MB**")
        lines.append("")

        lines.append("### Extensions")
        lines.append("")
        if source["extensions"]:
            lines.append("| Extension | Count |")
            lines.append("|---|---:|")
            for ext, count in list(source["extensions"].items())[:25]:
                lines.append(f"| `{ext}` | {count} |")
        else:
            lines.append("_No files found._")

        lines.append("")

        lines.append("### Interesting files")
        lines.append("")
        if source["interesting_files"]:
            lines.append("| Type | Count |")
            lines.append("|---|---:|")
            for ext, count in source["interesting_files"].items():
                lines.append(f"| `{ext}` | {count} |")
        else:
            lines.append("_No interesting files found._")

        lines.append("")

        lines.append("### Samples")
        lines.append("")
        if source["interesting_samples_by_type"]:
            for ext, samples in source["interesting_samples_by_type"].items():
                lines.append(f"#### `{ext}`")
                lines.append("")
                for sample in samples:
                    lines.append(f"- `{sample}`")
                lines.append("")
        else:
            lines.append("_No samples._")
            lines.append("")

        lines.append("### Largest files")
        lines.append("")
        if source["largest_files"]:
            lines.append("| Path | Size MB |")
            lines.append("|---|---:|")
            for item in source["largest_files"]:
                size_mb = round(item["size_bytes"] / 1024 / 1024, 3)
                lines.append(f"| `{item['path']}` | {size_mb} |")
        else:
            lines.append("_No files._")

        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        default="rag_corpus/raw",
        help="Path to raw corpus directory.",
    )
    parser.add_argument(
        "--out-dir",
        default="rag_corpus/reports",
        help="Path to reports directory.",
    )

    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root does not exist: {raw_root}")

    out_dir.mkdir(parents=True, exist_ok=True)

    sources = []

    for source_dir in sorted(raw_root.iterdir()):
        if not source_dir.is_dir():
            continue

        if source_dir.name in SKIP_DIRS:
            continue

        sources.append(inspect_source(source_dir, raw_root))

    total_files = sum(source["file_count"] for source in sources)
    total_bytes = sum(source["total_bytes"] for source in sources)

    report = {
        "raw_root": raw_root.as_posix(),
        "total_sources": len(sources),
        "total_files": total_files,
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / 1024 / 1024, 3),
        "sources": sources,
    }

    json_path = out_dir / "raw_inventory.json"
    md_path = out_dir / "raw_inventory.md"

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_path.write_text(
        build_markdown(report),
        encoding="utf-8",
    )

    print(f"Saved JSON report: {json_path}")
    print(f"Saved Markdown report: {md_path}")
    print(f"Sources: {len(sources)}")
    print(f"Files: {total_files}")
    print(f"Size MB: {round(total_bytes / 1024 / 1024, 3)}")


if __name__ == "__main__":
    main()
