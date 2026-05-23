from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from src.benchmark.models import BenchmarkCase

_CASE_LIST_KEYS = ("cases", "examples", "data", "items", "records")
_QUERY_KEYS = ("query", "prompt", "utterance", "nl_query", "question", "instruction")
_DATA_PATH_KEYS = ("data_path", "dataset_path", "csv_path", "table_path", "data")
_REFERENCE_SPEC_KEYS = ("reference_spec", "ground_truth_spec", "gt_spec", "vl_spec", "spec", "vega_lite_spec")
_REFERENCE_IMAGE_KEYS = ("reference_image_path", "gt_image_path", "image_path", "reference_png", "reference_image")
_ID_KEYS = ("case_id", "id", "example_id", "uid", "index")


def load_benchmark_cases(path: str | Path, *, nlv_mode: str = "single_turn") -> list[BenchmarkCase]:
    """Load JSON/JSONL benchmark cases with flexible VegaChat/NLV/ChartLLM-style keys."""

    source = Path(path)
    root = source.parent if source.is_file() else source
    if source.is_dir():
        nlv_cases = _try_load_nlv_corpus_cases(source, nlv_mode=nlv_mode)
        if nlv_cases is not None:
            return nlv_cases
        chart_llm_cases = _try_load_chart_llm_cases(source)
        if chart_llm_cases is not None:
            return chart_llm_cases
    payloads = list(_iter_payloads(source))
    cases: list[BenchmarkCase] = []
    for index, item in enumerate(payloads):
        if not isinstance(item, dict):
            continue
        cases.append(_case_from_payload(item, index=index, root=root))
    return cases


def _try_load_nlv_corpus_cases(root: Path, *, nlv_mode: str = "single_turn") -> list[BenchmarkCase] | None:
    corpus_path = root / "NLV_Corpus.csv"
    specs_path = root / "vlSpecs.json"
    datasets_dir = root / "datasets"
    if not corpus_path.exists() or not specs_path.exists() or not datasets_dir.exists():
        return None
    specs = json.loads(specs_path.read_text(encoding="utf-8"))
    rows = _read_csv_dicts(corpus_path)
    cases: list[BenchmarkCase] = []
    for index, row in enumerate(rows):
        dataset = str(row.get("dataset") or "").lower().strip()
        vis_id = str(row.get("visId") or "").strip()
        spec_id = f"{dataset}-{vis_id}"
        reference_spec = specs.get(spec_id)
        if not dataset or not vis_id or not isinstance(reference_spec, dict):
            continue
        utterance_set = str(row.get("Utterance Set") or "").strip()
        if not utterance_set:
            continue
        prompt_parts = [part.strip() for part in utterance_set.split("|") if part.strip()]
        sequential_flag = str(row.get("sequential") or "").lower().strip()
        utterance_type = "sequential" if sequential_flag == "y" or len(prompt_parts) > 1 else "single_turn"
        if nlv_mode != "single_turn":
            raise ValueError("Only nlv_mode='single_turn' is supported for the main ViRAGE benchmark.")
        if utterance_type != "single_turn":
            continue
        query = prompt_parts[0] if prompt_parts else utterance_set
        stable_id = _stable_case_id("nlv", dataset, vis_id, query)
        cases.append(BenchmarkCase(
            case_id=stable_id,
            query=query,
            data_path=(datasets_dir / f"{dataset}.csv").resolve().as_posix(),
            reference_spec=reference_spec,
            dataset_name="nlv_corpus",
            utterance_type=utterance_type,
            metadata={
                "dataset": dataset,
                "visId": vis_id,
                "sequential": sequential_flag,
                "prompt_parts": prompt_parts,
                "nlv_mode": nlv_mode,
                "source_row_index": index,
            },
        ))
    return cases


def _try_load_chart_llm_cases(root: Path) -> list[BenchmarkCase] | None:
    gold_path = root / "exp" / "gold" / "result" / "gold.csv"
    specs_dir = root / "docs" / "data" / "chart_48_in"
    csv_dir = root / "docs" / "data" / "csv_48_process"
    png_dir = root / "docs" / "data" / "chart_48_img"
    if not gold_path.exists() or not specs_dir.exists() or not csv_dir.exists():
        return None
    cases: list[BenchmarkCase] = []
    rows = _read_csv_dicts(gold_path)
    utterance_fields = ("command", "query", "question")
    for index, row in enumerate(rows):
        if index == 37:
            continue
        spec_path = specs_dir / f"vl_{index:02}.vl.json"
        data_path = csv_dir / f"d_{index:02}.csv"
        if not spec_path.exists() or not data_path.exists():
            continue
        reference_spec = json.loads(spec_path.read_text(encoding="utf-8"))
        level = str(row.get("level") or "").strip()
        reference_image = _chart_llm_reference_image_path(png_dir, index, level)
        for utterance_field in utterance_fields:
            query = str(row.get(utterance_field) or "").strip()
            if not query:
                continue
            cases.append(BenchmarkCase(
                case_id=f"chart_llm_{index}_{utterance_field}",
                query=query,
                data_path=data_path.resolve().as_posix(),
                reference_spec=reference_spec,
                reference_image_path=reference_image.resolve().as_posix() if reference_image and reference_image.exists() else None,
                dataset_name="chart_llm_gold",
                difficulty=level or None,
                utterance_type=utterance_field,
                metadata={
                    "chart_index": index,
                    "chart_number": row.get("Chart #"),
                    "interaction": row.get("interaction"),
                    "composite": row.get("composite"),
                },
            ))
    return cases


def _chart_llm_reference_image_path(root: Path, index: int, level: str) -> Path | None:
    level_int_by_name = {"simple": 1, "medium": 2, "complex": 3, "extracomplex": 4}
    level_int = level_int_by_name.get(level)
    if level_int is None:
        return None
    return root / f"{level_int}.{level}" / f"visualization ({index}).png"


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _iter_payloads(path: Path) -> Iterable[dict[str, Any]]:
    if path.is_dir():
        for child in sorted(path.iterdir()):
            if child.suffix.lower() in {".json", ".jsonl"}:
                yield from _iter_payloads(child)
        return
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                cleaned = line.strip()
                if not cleaned:
                    continue
                try:
                    payload = json.loads(cleaned)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
                if isinstance(payload, dict):
                    yield payload
        return
    if path.suffix.lower() != ".json":
        raise ValueError(f"Unsupported benchmark dataset format: {path.suffix}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(data, dict):
        for key in _CASE_LIST_KEYS:
            value = data.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return
        yield data
        return
    raise ValueError(f"Unsupported JSON payload in {path}")


def _case_from_payload(payload: dict[str, Any], *, index: int, root: Path) -> BenchmarkCase:
    query = _first_string(payload, _QUERY_KEYS)
    data_path = _extract_data_path(payload, root)
    reference_spec = _first_dict(payload, _REFERENCE_SPEC_KEYS)
    if not query:
        raise ValueError(f"Benchmark case #{index} has no query/prompt/utterance field.")
    if not data_path:
        raise ValueError(f"Benchmark case #{index} has no data_path/dataset_path/csv_path field.")
    case_id = _first_string(payload, _ID_KEYS) or _stable_case_id("case", root.name or "dataset", str(index), query)
    reference_image_path = _first_string(payload, _REFERENCE_IMAGE_KEYS)
    return BenchmarkCase(
        case_id=case_id,
        query=query,
        data_path=data_path,
        reference_spec=reference_spec,
        reference_image_path=reference_image_path,
        dataset_name=_first_string(payload, ("dataset_name", "dataset", "source")) or root.name or "unknown",
        difficulty=_first_string(payload, ("difficulty", "level")),
        utterance_type=_first_string(payload, ("utterance_type", "query_type", "type")),
        metadata={key: value for key, value in payload.items() if key not in _known_keys()},
    )


def _extract_data_path(payload: dict[str, Any], root: Path) -> str:
    for key in _DATA_PATH_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            url = value.get("url") or value.get("path") or value.get("name")
            if isinstance(url, str) and url.strip() and Path(url).suffix.lower() in {".csv", ".xlsx", ".xls",
                                                                                     ".parquet"}:
                return url.strip()
    # Vega-Lite specs often carry the data path inside reference spec.data.url.
    reference_spec = _first_dict(payload, _REFERENCE_SPEC_KEYS)
    data = reference_spec.get("data") if isinstance(reference_spec, dict) else None
    data_url = data.get("url") if isinstance(data, dict) else None
    if isinstance(data_url, str) and data_url.strip():
        candidate = Path(data_url)
        return candidate.as_posix() if candidate.is_absolute() else (root / candidate).as_posix()
    return ""


def _first_string(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
    return None


def _first_dict(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip().startswith("{"):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return {}


def _known_keys() -> set[str]:
    return set(_QUERY_KEYS + _DATA_PATH_KEYS + _REFERENCE_SPEC_KEYS + _REFERENCE_IMAGE_KEYS + _ID_KEYS + (
        "dataset_name", "dataset", "source", "difficulty", "level", "utterance_type", "query_type", "type",
    ))


def _stable_case_id(prefix: str, *parts: str) -> str:
    cleaned = [str(part).strip().lower().replace(" ", "_") for part in parts if str(part).strip()]
    digest = hashlib.sha1("|".join(cleaned).encode("utf-8")).hexdigest()[:12]
    readable = "_".join(cleaned[:2])[:80].strip("_")
    return f"{prefix}_{readable}_{digest}" if readable else f"{prefix}_{digest}"
