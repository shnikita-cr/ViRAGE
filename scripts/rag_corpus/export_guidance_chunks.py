from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

ROOT = next((parent for parent in Path(__file__).resolve().parents if (parent / "src").exists()), Path.cwd())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MIN_CHARS = 220
DEFAULT_MAX_CHARS = 1000
DEFAULT_OVERLAP_CHARS = 120
SKIP_FILE_RE = re.compile(r"download_manifest|bibliography|references|image-file-formats|choosing-visualization-software", re.I)


def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="strict")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n{2,}|(?<=\.)\s+(?=[A-ZА-Я])", text)
    return [part.strip() for part in parts if part.strip()]


def _validate_chunking_params(*, min_chars: int, max_chars: int, overlap_chars: int) -> None:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive.")
    if min_chars < 0:
        raise ValueError("min_chars must be non-negative.")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be non-negative.")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars.")
    if min_chars > max_chars:
        raise ValueError("min_chars must not be greater than max_chars.")


def _best_cut(text: str, start: int, max_chars: int) -> int:
    hard_end = min(len(text), start + max_chars)
    if hard_end >= len(text):
        return len(text)
    window = text[start:hard_end]
    lower_bound = max(1, int(len(window) * 0.55))
    candidates = [
        window.rfind("\n"),
        window.rfind(". "),
        window.rfind("; "),
        window.rfind(", "),
        window.rfind(" "),
    ]
    valid = [candidate for candidate in candidates if candidate >= lower_bound]
    if valid:
        return start + max(valid) + 1
    return hard_end


def _split_oversized_text(text: str, *, max_chars: int, overlap_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        cut = _best_cut(text, start, max_chars)
        if cut <= start:
            cut = min(len(text), start + max_chars)
        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)
        if cut >= len(text):
            break
        next_start = max(cut - overlap_chars, start + 1)
        if next_start <= start:
            next_start = cut
        start = next_start
    return chunks


def chunk_text(
    text: str,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    _validate_chunking_params(min_chars=min_chars, max_chars=max_chars, overlap_chars=overlap_chars)
    chunks: list[str] = []
    buffer = ""

    def flush_buffer() -> None:
        nonlocal buffer
        if not buffer:
            return
        if len(buffer) > max_chars:
            chunks.extend(_split_oversized_text(buffer, max_chars=max_chars, overlap_chars=overlap_chars))
        elif len(buffer) >= min_chars or not chunks:
            chunks.append(buffer.strip())
        elif chunks and len(chunks[-1]) + 1 + len(buffer) <= max_chars:
            chunks[-1] = f"{chunks[-1]}\n{buffer}".strip()
        else:
            chunks.append(buffer.strip())
        buffer = ""

    for part in split_paragraphs(text):
        if len(part) > max_chars:
            flush_buffer()
            chunks.extend(_split_oversized_text(part, max_chars=max_chars, overlap_chars=overlap_chars))
            continue

        candidate = f"{buffer}\n{part}".strip() if buffer else part
        if len(candidate) <= max_chars:
            buffer = candidate
            continue

        flush_buffer()
        buffer = part

    flush_buffer()

    normalized = [chunk.strip() for chunk in chunks if chunk.strip()]
    too_long = [len(chunk) for chunk in normalized if len(chunk) > max_chars]
    if too_long:
        raise RuntimeError(f"Chunking produced {len(too_long)} oversized chunks. Longest length: {max(too_long)}; limit: {max_chars}.")
    return normalized


def source_kind_for(source_id: str) -> str:
    if source_id == "scientific_figure_guidance":
        return "scientific_figure_guidance"
    if source_id == "eda_guidance":
        return "eda_guidance"
    if "feedback" in source_id:
        return "manual_feedback"
    if source_id in {"vlat", "massvis", "previs"}:
        return "dataset_pattern"
    return "web_guidance"


def iter_source_files(raw_root: Path, sources: set[str] | None) -> list[Path]:
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)
    paths = sorted(path for path in raw_root.rglob("*.txt") if path.is_file())
    if sources:
        paths = [path for path in paths if path.relative_to(raw_root).parts[0] in sources]
    return [path for path in paths if not SKIP_FILE_RE.search(path.name)]


def build_chunk(
    path: Path,
    raw_root: Path,
    text: str,
    index: int,
    *,
    original_text_chars: int,
    max_chars: int,
    overlap_chars: int,
) -> dict[str, object]:
    rel = path.relative_to(raw_root).as_posix()
    source_id = rel.split("/", 1)[0]
    stable = uuid5(NAMESPACE_URL, f"{rel}:{index}:{hashlib.sha1(text.encode('utf-8')).hexdigest()}").hex
    title = path.stem.replace("dataviz_", "").replace("_", " ").replace("-", " ")
    return {
        "chunk_id": f"chunk_{stable}",
        "source_id": source_id,
        "source_name": source_id.replace("_", " "),
        "source_kind": source_kind_for(source_id),
        "title": title,
        "text": text,
        "source_path": rel,
        "url": None,
        "metadata": {
            "chunk_index": index,
            "source_file": rel,
            "text_sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(),
            "chunk_char_length": len(text),
            "original_text_chars": original_text_chars,
            "chunk_max_chars": max_chars,
            "chunk_overlap_chars": overlap_chars,
        },
    }


def _runtime_length_report(records: list[dict[str, object]], *, max_chars: int) -> dict[str, object]:
    lengths = [len(str(record.get("text") or "")) for record in records]
    oversized = [record for record in records if len(str(record.get("text") or "")) > max_chars]
    by_source: dict[str, dict[str, int]] = {}
    for record, length in zip(records, lengths):
        source_id = str(record.get("source_id") or "")
        stats = by_source.setdefault(source_id, {"chunks": 0, "max_chars": 0})
        stats["chunks"] += 1
        stats["max_chars"] = max(stats["max_chars"], length)
    longest = sorted(
        [
            {
                "chunk_id": str(record.get("chunk_id") or ""),
                "source_id": str(record.get("source_id") or ""),
                "source_path": str(record.get("source_path") or ""),
                "title": str(record.get("title") or ""),
                "char_length": len(str(record.get("text") or "")),
            }
            for record in records
        ],
        key=lambda item: item["char_length"],
        reverse=True,
    )[:20]
    return {
        "max_allowed_chars": max_chars,
        "total_chunks": len(records),
        "min_chunk_chars": min(lengths) if lengths else 0,
        "max_chunk_chars": max(lengths) if lengths else 0,
        "avg_chunk_chars": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        "oversized_chunks": len(oversized),
        "by_source": by_source,
        "longest_chunks": longest,
    }


def export_chunks(
    raw_root: Path,
    output: Path,
    sources: set[str] | None,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> dict[str, object]:
    _validate_chunking_params(min_chars=min_chars, max_chars=max_chars, overlap_chars=overlap_chars)
    records: list[dict[str, object]] = []
    source_file_count = 0
    for path in iter_source_files(raw_root, sources):
        source_file_count += 1
        text = read_text(path)
        chunks = chunk_text(text, min_chars=min_chars, max_chars=max_chars, overlap_chars=overlap_chars)
        for index, chunk in enumerate(chunks, start=1):
            records.append(
                build_chunk(
                    path,
                    raw_root,
                    chunk,
                    index,
                    original_text_chars=len(text),
                    max_chars=max_chars,
                    overlap_chars=overlap_chars,
                )
            )
    if not records:
        raise RuntimeError("No guidance chunks exported. Check raw corpus input and source filters.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    by_source: dict[str, int] = {}
    for record in records:
        by_source[str(record["source_id"])] = by_source.get(str(record["source_id"]), 0) + 1
    length_report = _runtime_length_report(records, max_chars=max_chars)
    return {
        "output": output.as_posix(),
        "source_files": source_file_count,
        "by_source": by_source,
        "total_chunks": sum(by_source.values()),
        "chunking": {
            "min_chars": min_chars,
            "max_chars": max_chars,
            "overlap_chars": overlap_chars,
        },
        "length_report": length_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export cleaned raw corpus text to VisRAG guidance chunks.")
    parser.add_argument("--raw-root", default="rag_corpus/raw_external_rules")
    parser.add_argument("--output", default="rag_corpus/runtime/guidance_chunks.jsonl")
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--overlap-chars", type=int, default=DEFAULT_OVERLAP_CHARS)
    args = parser.parse_args()
    report = export_chunks(
        ROOT / args.raw_root,
        ROOT / args.output,
        set(args.sources or []) or None,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
    )
    runtime_dir = ROOT / "rag_corpus" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    report_path = runtime_dir / "runtime_export_report.json"
    length_report_path = runtime_dir / "runtime_chunk_length_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    length_report_path.write_text(json.dumps(report["length_report"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
