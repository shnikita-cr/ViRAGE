from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_CLEANED_ROOT = "rag_corpus/cleaned"
DEFAULT_SPECS_DIR = "rag_corpus/cleaned/vega-lite/examples/specs"
DEFAULT_OUT_FILE = "rag_corpus/normalized/jsonl/official_vega_lite_examples.jsonl"
DEFAULT_REPORT_FILE = "rag_corpus/reports/official_vega_lite_examples_report.md"

EXCLUDED_DIR_NAMES = {
    "normalized",
    "compiled",
    "__pycache__",
    ".git",
    "node_modules",
}

DATA_KEYS_TO_REMOVE = {
    "data",
    "datasets",
}

PATTERN_PRIORITY = [
    "bar_chart",
    "line_chart",
    "scatter_plot",
    "histogram",
    "heatmap",
    "area_chart",
    "pie_chart",
    "layered_chart",
    "faceted_chart",
    "point_chart",
    "tick_plot",
    "rule_chart",
    "text_chart",
    "map_chart",
    "rect_chart",
    "unknown",
]

CHANNELS = [
    "x",
    "y",
    "color",
    "size",
    "shape",
    "opacity",
    "theta",
    "radius",
    "row",
    "column",
    "facet",
    "tooltip",
]

KNOWN_MARKS = {
    "arc",
    "area",
    "bar",
    "boxplot",
    "circle",
    "errorband",
    "errorbar",
    "geoshape",
    "image",
    "line",
    "point",
    "rect",
    "rule",
    "square",
    "text",
    "tick",
    "trail",
}


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {path} | {exc}") from exc

    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return value


def is_excluded_path(path: Path) -> bool:
    return any(part in EXCLUDED_DIR_NAMES for part in path.parts)


def list_vega_lite_specs(specs_dir: Path) -> list[Path]:
    if not specs_dir.exists():
        raise FileNotFoundError(f"Specs directory does not exist: {specs_dir}")

    result: list[Path] = []

    for path in specs_dir.rglob("*.vl.json"):
        if not path.is_file():
            continue

        rel_path = path.relative_to(specs_dir)

        if is_excluded_path(rel_path):
            continue

        result.append(path)

    return sorted(result)


def file_id(path: Path) -> str:
    if path.name.endswith(".vl.json"):
        return path.name[: -len(".vl.json")]

    return path.stem


def make_title(file_stem: str) -> str:
    return file_stem.replace("_", " ").replace("-", " ").strip()


def relative_source_path(path: Path, cleaned_root: Path) -> str:
    try:
        return path.relative_to(cleaned_root).as_posix()
    except ValueError:
        return path.as_posix()


def remove_data_sections(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: remove_data_sections(child)
            for key, child in value.items()
            if key not in DATA_KEYS_TO_REMOVE
        }

    if isinstance(value, list):
        return [remove_data_sections(item) for item in value]

    return value


def count_removed_data_sections(value: Any) -> int:
    if isinstance(value, dict):
        own_count = sum(1 for key in value if key in DATA_KEYS_TO_REMOVE)
        child_count = sum(count_removed_data_sections(child) for child in value.values())
        return own_count + child_count

    if isinstance(value, list):
        return sum(count_removed_data_sections(item) for item in value)

    return 0


def get_mark_type(mark_def: Any) -> str | None:
    if isinstance(mark_def, str):
        return mark_def

    if isinstance(mark_def, dict):
        mark_type = mark_def.get("type")

        if isinstance(mark_type, str):
            return mark_type

    return None


def collect_marks(spec: dict[str, Any]) -> list[str]:
    marks: list[str] = []

    mark = get_mark_type(spec.get("mark"))

    if mark:
        marks.append(mark)

    for key in ("layer", "hconcat", "vconcat", "concat"):
        children = spec.get(key)

        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    marks.extend(collect_marks(child))

    child_spec = spec.get("spec")

    if isinstance(child_spec, dict):
        marks.extend(collect_marks(child_spec))

    return marks


def collect_encodings(spec: dict[str, Any]) -> list[dict[str, Any]]:
    encodings: list[dict[str, Any]] = []

    encoding = spec.get("encoding")

    if isinstance(encoding, dict):
        encodings.append(encoding)

    for key in ("layer", "hconcat", "vconcat", "concat"):
        children = spec.get(key)

        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    encodings.extend(collect_encodings(child))

    child_spec = spec.get("spec")

    if isinstance(child_spec, dict):
        encodings.extend(collect_encodings(child_spec))

    return encodings


def normalize_mark_type(mark: str | None) -> str:
    if not mark:
        return "unknown"

    mark = mark.lower()

    if mark in KNOWN_MARKS:
        return mark

    return "unknown"


def primary_mark_type(spec: dict[str, Any]) -> str:
    marks = [normalize_mark_type(mark) for mark in collect_marks(spec)]

    if not marks:
        return "unknown"

    for preferred in (
            "bar",
            "line",
            "point",
            "circle",
            "rect",
            "area",
            "arc",
            "tick",
            "rule",
            "text",
            "geoshape",
    ):
        if preferred in marks:
            return preferred

    return marks[0]


def is_layered(spec: dict[str, Any]) -> bool:
    layer = spec.get("layer")
    return isinstance(layer, list) and len(layer) > 0


def is_faceted(spec: dict[str, Any]) -> bool:
    if "facet" in spec or "repeat" in spec:
        return True

    encoding = spec.get("encoding")

    if isinstance(encoding, dict):
        return "row" in encoding or "column" in encoding or "facet" in encoding

    return False


def first_channel_def(
        encodings: list[dict[str, Any]],
        channel: str,
) -> dict[str, Any] | None:
    for encoding in encodings:
        value = encoding.get(channel)

        if isinstance(value, dict):
            return value

    return None


def channel_type(channel_def: dict[str, Any] | None) -> str | None:
    if not isinstance(channel_def, dict):
        return None

    value = channel_def.get("type")

    if isinstance(value, str):
        return value

    return None


def is_binned(channel_def: dict[str, Any] | None) -> bool:
    return isinstance(channel_def, dict) and bool(channel_def.get("bin"))


def is_count(channel_def: dict[str, Any] | None) -> bool:
    return isinstance(channel_def, dict) and channel_def.get("aggregate") == "count"


def detect_chart_pattern(spec: dict[str, Any], mark_type: str) -> str:
    encodings = collect_encodings(spec)

    x_def = first_channel_def(encodings, "x")
    y_def = first_channel_def(encodings, "y")
    color_def = first_channel_def(encodings, "color")
    theta_def = first_channel_def(encodings, "theta")

    if is_faceted(spec):
        return "faceted_chart"

    if is_layered(spec):
        return "layered_chart"

    if mark_type == "arc" or theta_def is not None:
        return "pie_chart"

    if mark_type == "geoshape":
        return "map_chart"

    if mark_type == "rect":
        if color_def is not None:
            return "heatmap"

        return "rect_chart"

    if mark_type == "bar":
        if is_binned(x_def) or is_binned(y_def) or is_count(x_def) or is_count(y_def):
            return "histogram"

        return "bar_chart"

    if mark_type == "line":
        return "line_chart"

    if mark_type == "area":
        return "area_chart"

    if mark_type in {"point", "circle", "square"}:
        if channel_type(x_def) == "quantitative" and channel_type(y_def) == "quantitative":
            return "scatter_plot"

        return "point_chart"

    if mark_type == "tick":
        return "tick_plot"

    if mark_type == "rule":
        return "rule_chart"

    if mark_type == "text":
        return "text_chart"

    return "unknown"


def normalize_channel_role(channel_def: Any) -> str:
    if not isinstance(channel_def, dict):
        return "unknown"

    aggregate = channel_def.get("aggregate")
    field_type = channel_def.get("type")
    has_bin = bool(channel_def.get("bin"))
    has_time_unit = bool(channel_def.get("timeUnit"))

    if aggregate == "count":
        return "count"

    role = field_type if isinstance(field_type, str) else "unknown"

    if has_time_unit:
        role = f"{role}_timeunit"

    if has_bin:
        role = f"{role}_binned"

    if isinstance(aggregate, str):
        role = f"{role}_aggregate"

    return role


def extract_field_roles(spec: dict[str, Any]) -> dict[str, str]:
    roles: dict[str, str] = {}

    for encoding in collect_encodings(spec):
        for channel in CHANNELS:
            if channel not in encoding:
                continue

            value = encoding[channel]

            if isinstance(value, list):
                if value:
                    roles[channel] = normalize_channel_role(value[0])
                continue

            roles[channel] = normalize_channel_role(value)

    return roles


def make_instruction(
        chart_pattern: str,
        mark_type: str,
        field_roles: dict[str, str],
) -> str:
    if chart_pattern == "bar_chart":
        if "aggregate" in field_roles.get("x", "") or "aggregate" in field_roles.get("y", ""):
            return "Create a bar chart that compares aggregated quantitative values across categories."

        return "Create a bar chart that compares values across categories."

    if chart_pattern == "line_chart":
        return "Create a line chart that shows how a quantitative value changes over an ordered or temporal field."

    if chart_pattern == "scatter_plot":
        return "Create a scatter plot that shows the relationship between two quantitative fields."

    if chart_pattern == "histogram":
        return "Create a histogram that shows the distribution of a quantitative field using bins and counts."

    if chart_pattern == "heatmap":
        return "Create a heatmap that uses color intensity to compare values across two dimensions."

    if chart_pattern == "area_chart":
        return "Create an area chart that shows a quantitative trend over an ordered or temporal field."

    if chart_pattern == "pie_chart":
        return "Create a pie or arc chart that shows part-to-whole composition."

    if chart_pattern == "layered_chart":
        return "Create a layered chart that combines multiple marks or views in one visualization."

    if chart_pattern == "faceted_chart":
        return "Create a faceted chart that splits the visualization into small multiples by category."

    if chart_pattern == "point_chart":
        return "Create a point chart that displays individual records or values."

    if chart_pattern == "tick_plot":
        return "Create a tick plot that shows individual values along an axis."

    if chart_pattern == "rule_chart":
        return "Create a rule chart that shows reference lines, ranges, or intervals."

    if chart_pattern == "text_chart":
        return "Create a text-based chart that displays values as labels."

    if chart_pattern == "map_chart":
        return "Create a geographic visualization using geospatial shapes or coordinates."

    if chart_pattern == "rect_chart":
        return "Create a rectangular mark chart for binned, gridded, or interval-based visual encodings."

    return f"Create a Vega-Lite chart using a {mark_type} mark."


def format_field_roles_for_retrieval(field_roles: dict[str, str]) -> str:
    if not field_roles:
        return "none"

    return ", ".join(
        f"{channel} {role}"
        for channel, role in sorted(field_roles.items())
    )


def build_retrieval_text(
        title: str,
        instruction: str,
        chart_pattern: str,
        mark_type: str,
        field_roles: dict[str, str],
) -> str:
    field_roles_text = format_field_roles_for_retrieval(field_roles)

    return (
        f"Title: {title}. "
        f"Instruction: {instruction} "
        f"Chart pattern: {chart_pattern}. "
        f"Mark type: {mark_type}. "
        f"Field roles: {field_roles_text}. "
        f"Source: Vega-Lite official example."
    )


def normalize_spec(path: Path, cleaned_root: Path) -> dict[str, Any]:
    raw_spec = load_json(path)
    spec_template = remove_data_sections(copy.deepcopy(raw_spec))
    removed_data_sections = count_removed_data_sections(raw_spec)

    mark_type = primary_mark_type(raw_spec)
    chart_pattern = detect_chart_pattern(raw_spec, mark_type)
    field_roles = extract_field_roles(raw_spec)

    file_name = path.name
    file_stem = file_id(path)
    title = make_title(file_stem)
    instruction = make_instruction(chart_pattern, mark_type, field_roles)
    retrieval_text = build_retrieval_text(
        title=title,
        instruction=instruction,
        chart_pattern=chart_pattern,
        mark_type=mark_type,
        field_roles=field_roles,
    )

    return {
        "id": f"vega_lite:example:{file_stem}",
        "source": "vega-lite",
        "source_path": relative_source_path(path, cleaned_root),
        "file_name": file_name,
        "file_stem": file_stem,
        "title": title,
        "source_split": None,
        "benchmark_group": None,
        "is_eval_leak_sensitive": False,
        "corpus_type": "example",
        "instruction": instruction,
        "mark_type": mark_type,
        "chart_pattern": chart_pattern,
        "field_roles": field_roles,
        "retrieval_text": retrieval_text,
        "spec_template": spec_template,
        "raw_spec": raw_spec,
        "data_policy": "removed_from_spec_template",
        "removed_data_sections": removed_data_sections,
        "notes": (
            "Official Vega-Lite example from cleaned corpus. "
            "The spec_template has data/datasets removed. "
            "ViRAGE runtime must attach the data section deterministically. "
            "Use this record as chart structure guidance and do not copy example field names blindly."
        ),
    }


def balanced_sample(
        records: list[dict[str, Any]],
        total: int,
        seed: int,
        include_unknown: bool,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        pattern = record["chart_pattern"]

        if pattern == "unknown" and not include_unknown:
            continue

        groups[pattern].append(record)

    for group_records in groups.values():
        rng.shuffle(group_records)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    available_patterns = [
        pattern
        for pattern in PATTERN_PRIORITY
        if pattern in groups and groups[pattern]
    ]

    # First pass: take one record from each available chart pattern.
    for pattern in available_patterns:
        if len(selected) >= total:
            break

        record = groups[pattern].pop()
        selected.append(record)
        selected_ids.add(record["id"])

    # Second pass: fill remaining slots in round-robin order.
    while len(selected) < total:
        made_progress = False

        for pattern in available_patterns:
            if len(selected) >= total:
                break

            while groups[pattern]:
                record = groups[pattern].pop()

                if record["id"] in selected_ids:
                    continue

                selected.append(record)
                selected_ids.add(record["id"])
                made_progress = True
                break

        if not made_progress:
            break

    rng.shuffle(selected)
    return selected


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")


def build_report(
        selected: list[dict[str, Any]],
        all_records: list[dict[str, Any]],
        skipped_errors: list[str],
        seed: int,
        cleaned_root: Path,
        specs_dir: Path,
) -> str:
    corpus_pattern_counts = Counter(record["chart_pattern"] for record in all_records)
    corpus_mark_counts = Counter(record["mark_type"] for record in all_records)

    selected_pattern_counts = Counter(record["chart_pattern"] for record in selected)
    selected_mark_counts = Counter(record["mark_type"] for record in selected)

    removed_total = sum(record["removed_data_sections"] for record in selected)

    total_records = max(len(all_records), 1)
    selected_total = max(len(selected), 1)

    lines: list[str] = []

    lines.append("# Official Vega-Lite examples normalization report")
    lines.append("")
    lines.append(f"Cleaned root: `{cleaned_root.as_posix()}`")
    lines.append(f"Specs dir: `{specs_dir.as_posix()}`")
    lines.append(f"Seed: `{seed}`")
    lines.append(f"Total parsed candidates: **{len(all_records)}**")
    lines.append(f"Selected records: **{len(selected)}**")
    lines.append(f"Skipped parse errors: **{len(skipped_errors)}**")
    lines.append(f"Removed data/datasets sections from selected templates: **{removed_total}**")
    lines.append("")

    lines.append("## Corpus chart pattern distribution")
    lines.append("")
    lines.append("Detected chart patterns across all parsed Vega-Lite candidates before sampling.")
    lines.append("")
    lines.append("| Chart pattern | Count | Share |")
    lines.append("|---|---:|---:|")

    for pattern, count in corpus_pattern_counts.most_common():
        share = round(count / total_records * 100, 2)
        lines.append(f"| `{pattern}` | {count} | {share}% |")

    lines.append("")

    lines.append("## Corpus mark type distribution")
    lines.append("")
    lines.append("| Mark type | Count | Share |")
    lines.append("|---|---:|---:|")

    for mark_type, count in corpus_mark_counts.most_common():
        share = round(count / total_records * 100, 2)
        lines.append(f"| `{mark_type}` | {count} | {share}% |")

    lines.append("")

    lines.append("## Selected chart pattern coverage")
    lines.append("")
    lines.append("Chart patterns in the sampled normalized JSONL output.")
    lines.append("")
    lines.append("| Chart pattern | Count | Share |")
    lines.append("|---|---:|---:|")

    for pattern, count in selected_pattern_counts.most_common():
        share = round(count / selected_total * 100, 2)
        lines.append(f"| `{pattern}` | {count} | {share}% |")

    lines.append("")

    lines.append("## Selected mark type coverage")
    lines.append("")
    lines.append("| Mark type | Count | Share |")
    lines.append("|---|---:|---:|")

    for mark_type, count in selected_mark_counts.most_common():
        share = round(count / selected_total * 100, 2)
        lines.append(f"| `{mark_type}` | {count} | {share}% |")

    lines.append("")

    lines.append("## Selected records")
    lines.append("")
    lines.append("| ID | Title | Pattern | Mark | Removed data sections | Source path |")
    lines.append("|---|---|---|---|---:|---|")

    for record in sorted(selected, key=lambda item: item["id"]):
        lines.append(
            f"| `{record['id']}` | "
            f"`{record['title']}` | "
            f"`{record['chart_pattern']}` | "
            f"`{record['mark_type']}` | "
            f"{record['removed_data_sections']} | "
            f"`{record['source_path']}` |"
        )

    lines.append("")

    if skipped_errors:
        lines.append("## Skipped parse errors")
        lines.append("")

        for error in skipped_errors[:50]:
            lines.append(f"- `{error}`")

        lines.append("")

    return "\n".join(lines)


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def print_counter(title: str, counter: Counter[str]) -> None:
    print(title)

    for key, value in counter.most_common():
        print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize a balanced random sample of official Vega-Lite examples from cleaned corpus."
    )

    parser.add_argument(
        "--cleaned-root",
        default=DEFAULT_CLEANED_ROOT,
        help="Path to rag_corpus/cleaned.",
    )
    parser.add_argument(
        "--specs-dir",
        default=DEFAULT_SPECS_DIR,
        help="Path to cleaned vega-lite/examples/specs.",
    )
    parser.add_argument(
        "--out-file",
        default=DEFAULT_OUT_FILE,
        help="Output normalized JSONL file.",
    )
    parser.add_argument(
        "--report-file",
        default=DEFAULT_REPORT_FILE,
        help="Output Markdown report file.",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=30,
        help="Number of normalized records to select.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Allow records with chart_pattern=unknown in the sample.",
    )

    args = parser.parse_args()

    cleaned_root = Path(args.cleaned_root)
    specs_dir = Path(args.specs_dir)
    out_file = Path(args.out_file)
    report_file = Path(args.report_file)

    if not cleaned_root.exists():
        raise FileNotFoundError(f"Cleaned root does not exist: {cleaned_root}")

    spec_paths = list_vega_lite_specs(specs_dir)

    records: list[dict[str, Any]] = []
    skipped_errors: list[str] = []

    for path in spec_paths:
        try:
            records.append(normalize_spec(path, cleaned_root))
        except Exception as exc:
            skipped_errors.append(f"{path.as_posix()} | {exc}")

    selected = balanced_sample(
        records=records,
        total=args.total,
        seed=args.seed,
        include_unknown=args.include_unknown,
    )

    write_jsonl(out_file, selected)

    report = build_report(
        selected=selected,
        all_records=records,
        skipped_errors=skipped_errors,
        seed=args.seed,
        cleaned_root=cleaned_root,
        specs_dir=specs_dir,
    )
    write_report(report_file, report)

    corpus_pattern_counts = Counter(record["chart_pattern"] for record in records)
    selected_pattern_counts = Counter(record["chart_pattern"] for record in selected)
    corpus_mark_counts = Counter(record["mark_type"] for record in records)
    selected_mark_counts = Counter(record["mark_type"] for record in selected)
    removed_total = sum(record["removed_data_sections"] for record in selected)

    print(f"Cleaned root: {cleaned_root}")
    print(f"Specs dir: {specs_dir}")
    print(f"Parsed candidates: {len(records)}")
    print(f"Selected records: {len(selected)}")
    print(f"Skipped parse errors: {len(skipped_errors)}")
    print(f"Removed data/datasets sections: {removed_total}")
    print(f"Saved JSONL: {out_file}")
    print(f"Saved report: {report_file}")

    print_counter("Corpus chart patterns:", corpus_pattern_counts)
    print_counter("Selected chart patterns:", selected_pattern_counts)
    print_counter("Corpus mark types:", corpus_mark_counts)
    print_counter("Selected mark types:", selected_mark_counts)


if __name__ == "__main__":
    main()
