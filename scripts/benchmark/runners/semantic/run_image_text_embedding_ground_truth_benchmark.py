from __future__ import annotations

import argparse
import site
import csv
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
site.addsitedir(PROJECT_ROOT.as_posix())
from statistics import mean
from typing import Any

from src.benchmark.core.sampling import SAMPLING_RANDOM, SAMPLING_STRATIFIED_CHART_TYPE, chart_type_from_case, select_benchmark_cases
from src.benchmark.datasets.datasets import load_benchmark_cases
from src.benchmark.evaluation.evaluator import VegaChatBenchmarkEvaluator
from src.benchmark.evaluation.image_text_cosine import ImageTextCosineEvaluator
from src.benchmark.evaluation.semantic_match import normalize_cosine_to_unit


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.cases)
    case_root = source.parent if source.is_file() else source
    loaded_cases = load_benchmark_cases(source, nlv_mode=args.nlv_mode)
    cases, sampling = select_benchmark_cases(
        loaded_cases,
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
        sampling_strategy=args.sampling,
        max_per_chart_type=args.max_per_chart_type,
        sampling_allocation=args.sampling_allocation,
        cases_per_chart_type=args.cases_per_chart_type,
    )
    scorer = ImageTextCosineEvaluator(model_names=args.models, device=args.device, dtype=args.dtype)
    renderer = VegaChatBenchmarkEvaluator()
    rows: list[dict[str, Any]] = []
    rendered = _render_reference_images(renderer, cases, case_root=case_root, output_dir=output)
    for case in cases:
        image_path = rendered[case.case_id]
        rows.extend(_score_pairs(case=case, cases=cases, image_path=image_path, scorer=scorer))
    _write_csv(output / "image_text_embedding_results.csv", rows)
    report = _report(rows, sampling.as_report_payload())
    (output / "image_text_embedding_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "image_text_embedding_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "pairs": len(rows), "report": (output / "image_text_embedding_report.json").as_posix()}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CLIP/SigLIP image-text scores on NLV reference plots.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output-dir", default="artifacts/benchmarks/image_text_ground_truth")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling", choices=[SAMPLING_RANDOM, SAMPLING_STRATIFIED_CHART_TYPE], default=SAMPLING_RANDOM)
    parser.add_argument("--max-per-chart-type", type=int, default=None)
    parser.add_argument("--sampling-allocation", choices=["round_robin", "balanced"], default="round_robin")
    parser.add_argument("--cases-per-chart-type", type=int, default=None)
    parser.add_argument("--nlv-mode", choices=["single_turn"], default="single_turn")
    parser.add_argument("--models", nargs="+", default=["openai/clip-vit-base-patch32", "google/siglip-so400m-patch14-384"])
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    return parser.parse_args()


def _render_reference_images(renderer: VegaChatBenchmarkEvaluator, cases, *, case_root: Path, output_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for case in cases:
        if not case.reference_spec:
            raise ValueError(f"Case has no reference_spec: {case.case_id}")
        out[case.case_id] = renderer.render_reference_image(
            reference_spec=case.reference_spec,
            data_path=case.resolved_data_path(case_root),
            output_path=output_dir / "reference_images" / f"{case.case_id}.png",
        )
    return out


def _score_pairs(*, case, cases, image_path: str, scorer: ImageTextCosineEvaluator) -> list[dict[str, Any]]:
    pairs = [("positive", case.case_id, case.query)]
    same = _negative_case(cases, case, same_chart_type=True)
    other = _negative_case(cases, case, same_chart_type=False)
    if same is not None:
        pairs.append(("negative_same_stratum", same.case_id, same.query))
    if other is not None:
        pairs.append(("negative_other_stratum", other.case_id, other.query))
    rows: list[dict[str, Any]] = []
    for pair_type, text_case_id, text in pairs:
        batch = scorer.score(image_path=image_path, task_text=text)
        model_scores = {item.model_name: normalize_cosine_to_unit(item.cosine) for item in batch.results}
        row = {
            "case_id": case.case_id,
            "chart_type": chart_type_from_case(case),
            "image_path": image_path,
            "pair_type": pair_type,
            "text_case_id": text_case_id,
            "query": text,
            "embedding_score": mean(model_scores.values()) if model_scores else None,
        }
        row.update({f"model_score.{name}": score for name, score in model_scores.items()})
        rows.append(row)
    return rows


def _negative_case(cases, case, *, same_chart_type: bool):
    own_type = chart_type_from_case(case)
    for candidate in cases:
        if candidate.case_id == case.case_id:
            continue
        candidate_same = chart_type_from_case(candidate) == own_type
        if candidate_same == same_chart_type:
            return candidate
    return None


def _report(rows: list[dict[str, Any]], sampling: dict[str, Any]) -> dict[str, Any]:
    positives = [float(row["embedding_score"]) for row in rows if row["pair_type"] == "positive" and row["embedding_score"] is not None]
    negatives = [float(row["embedding_score"]) for row in rows if row["pair_type"].startswith("negative") and row["embedding_score"] is not None]
    return {
        "total_pairs": len(rows),
        "positive_pairs": len(positives),
        "negative_pairs": len(negatives),
        "mean_positive_score": mean(positives) if positives else None,
        "mean_negative_score": mean(negatives) if negatives else None,
        "margin": (mean(positives) - mean(negatives)) if positives and negatives else None,
        "accuracy_positive_above_negative_mean": _accuracy_positive_above_negative_mean(positives, negatives),
        "sampling": sampling,
    }


def _accuracy_positive_above_negative_mean(positives: list[float], negatives: list[float]) -> float | None:
    if not positives or not negatives:
        return None
    threshold = mean(negatives)
    return sum(1 for value in positives if value > threshold) / len(positives)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
