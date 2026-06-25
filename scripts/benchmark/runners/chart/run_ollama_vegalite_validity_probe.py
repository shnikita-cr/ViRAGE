from __future__ import annotations

import argparse
import json
import logging
import site
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
site.addsitedir(PROJECT_ROOT.as_posix())

from src.application.config.project_config import ModelRoleConfig, ProjectConfig, load_project_config
from src.benchmark.external.ollama import DEFAULT_OLLAMA_HOST, OllamaSettings, assert_ollama_model_available
from src.domain.models import VegaLiteSpecArtifact
from src.llm.factory import build_chat_model
from src.llm.helpers import invoke_text
from src.services.spec.validator import SpecValidatorService
from src.services.spec_generation.vegachat_parser import VegaChatResponseParseError, parse_vegachat_response

logger = logging.getLogger(__name__)

ModelRole = Literal["reasoning", "spec", "vlm"]


@dataclass(frozen=True, slots=True)
class ProbeRunRecord:
    run_index: int
    raw_response_path: str
    duration_ms: int
    json_valid: bool
    vega_lite_valid: bool
    parse_error: str | None
    validation_errors: list[str]
    repair_hints: list[str]
    spec: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class ProbeSummary:
    model_role: str
    provider: str
    model: str
    dataset: str
    runs_requested: int
    total: int
    json_valid: int
    json_invalid: int
    vega_lite_valid: int
    vega_lite_invalid: int
    json_valid_rate: float
    vega_lite_valid_rate_total: float
    vega_lite_valid_rate_among_json: float
    output_dir: str


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file was not found: {dataset_path}")

    config = load_project_config(args.config)
    model_config = _role_config(config, args.model_role)
    if args.model:
        model_config = model_config.model_copy(update={"model": args.model})
    if args.temperature is not None:
        model_config = model_config.model_copy(update={"temperature": args.temperature})

    _check_ollama_if_required(model_config)
    llm = build_chat_model(model_config, config.settings, role=args.model_role)
    validator = SpecValidatorService()
    prompt = build_prompt(dataset_path)

    records: list[ProbeRunRecord] = []
    runs_path = output_dir / "runs.jsonl"
    with runs_path.open("w", encoding="utf-8") as handle:
        for run_index in range(1, args.runs + 1):
            record = run_once(
                run_index=run_index,
                llm=llm,
                prompt=prompt,
                validator=validator,
                raw_dir=raw_dir,
                model_role=args.model_role,
            )
            records.append(record)
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            handle.flush()
            logger.info(
                "%03d/%03d json=%s vega_lite=%s",
                run_index,
                args.runs,
                "ok" if record.json_valid else "fail",
                "ok" if record.vega_lite_valid else "fail",
            )

    summary = build_summary(
        records=records,
        model_config=model_config,
        model_role=args.model_role,
        dataset_path=dataset_path,
        runs_requested=args.runs,
        output_dir=output_dir,
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("JSON valid: %s/%s", summary.json_valid, summary.total)
    logger.info("Vega-Lite valid: %s/%s", summary.vega_lite_valid, summary.total)
    logger.info("Report directory: %s", output_dir.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated direct Ollama Vega-Lite generation probes and count JSON/Vega-Lite validity."
    )
    parser.add_argument("--config", default="ui/config/app/project.toml", help="Path to ViRAGE project config TOML.")
    parser.add_argument(
        "--model-role",
        default="spec",
        choices=["reasoning", "spec", "vlm"],
        help="Project model role used for direct generation.",
    )
    parser.add_argument("--model", default=None, help="Optional model name override for the selected role.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Optional temperature override for the selected role.")
    parser.add_argument("--runs", type=int, default=200, help="Number of independent model calls.")
    parser.add_argument("--dataset", default="demo_data/Iris.csv",
                        help="Dataset path that generated specs must reference.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/benchmarks/ollama_vegalite_validity",
        help="Directory for raw responses, per-run JSONL and summary.",
    )
    return parser.parse_args()


def _role_config(config: ProjectConfig, role: ModelRole) -> ModelRoleConfig:
    if role == "reasoning":
        return config.reasoning_model
    if role == "spec":
        return config.spec_model
    if role == "vlm":
        return config.vlm_model
    raise ValueError(f"Unsupported model role: {role}")


def _check_ollama_if_required(model_config: ModelRoleConfig) -> None:
    if model_config.provider.lower().strip() != "ollama":
        return
    settings = OllamaSettings(
        host=model_config.base_url or DEFAULT_OLLAMA_HOST,
        model=model_config.model,
    )
    assert_ollama_model_available(settings)


def build_prompt(dataset_path: Path) -> str:
    df = pd.read_csv(dataset_path, nrows=5)
    columns = list(df.columns)
    sample_rows = df.to_dict(orient="records")
    return f"""
Generate one valid Vega-Lite v5 JSON specification for the prepared dataset.

Task:
Build a scatter plot that shows PetalLengthCm on the x-axis and PetalWidthCm on the y-axis, colored by Species.

Dataset path for data.url:
{dataset_path.as_posix()}

Available columns:
{json.dumps(columns, ensure_ascii=False)}

Sample rows:
{json.dumps(sample_rows, ensure_ascii=False)}

Strict output rules:
- Return only one Vega-Lite JSON object inside <json>...</json>.
- Do not include markdown fences.
- Do not include explanations outside JSON.
- Use "$schema": "https://vega.github.io/schema/vega-lite/v5.json".
- Use "data": {{"url": "{dataset_path.as_posix()}"}}.
- Use only fields from the available columns.
""".strip()


def run_once(
        *,
        run_index: int,
        llm: Any,
        prompt: str,
        validator: SpecValidatorService,
        raw_dir: Path,
        model_role: str,
) -> ProbeRunRecord:
    started = time.perf_counter()
    raw_response = invoke_text(llm, prompt, stage="ollama_vegalite_validity_probe", role=model_role)
    duration_ms = int((time.perf_counter() - started) * 1000)

    raw_path = raw_dir / f"{run_index:03d}.txt"
    raw_path.write_text(raw_response, encoding="utf-8")

    try:
        _, spec = parse_vegachat_response(raw_response, mode="tolerant")
    except VegaChatResponseParseError as exc:
        return ProbeRunRecord(
            run_index=run_index,
            raw_response_path=raw_path.as_posix(),
            duration_ms=duration_ms,
            json_valid=False,
            vega_lite_valid=False,
            parse_error=str(exc),
            validation_errors=[],
            repair_hints=[],
            spec=None,
        )

    validation = validator.invoke(
        VegaLiteSpecArtifact(
            spec_json=spec,
            spec_without_runtime_data={k: v for k, v in spec.items() if k != "data"},
            version="vega-lite-v5",
            generation_backend="ollama_direct_probe",
        )
    )
    return ProbeRunRecord(
        run_index=run_index,
        raw_response_path=raw_path.as_posix(),
        duration_ms=duration_ms,
        json_valid=True,
        vega_lite_valid=validation.is_valid,
        parse_error=None,
        validation_errors=validation.validation_errors,
        repair_hints=validation.repair_hints,
        spec=validation.validated_spec,
    )


def build_summary(
        *,
        records: list[ProbeRunRecord],
        model_config: ModelRoleConfig,
        model_role: str,
        dataset_path: Path,
        runs_requested: int,
        output_dir: Path,
) -> ProbeSummary:
    total = len(records)
    json_valid = sum(1 for record in records if record.json_valid)
    vega_lite_valid = sum(1 for record in records if record.vega_lite_valid)
    vega_lite_invalid = json_valid - vega_lite_valid
    return ProbeSummary(
        model_role=model_role,
        provider=model_config.provider,
        model=model_config.model,
        dataset=dataset_path.as_posix(),
        runs_requested=runs_requested,
        total=total,
        json_valid=json_valid,
        json_invalid=total - json_valid,
        vega_lite_valid=vega_lite_valid,
        vega_lite_invalid=vega_lite_invalid,
        json_valid_rate=_safe_ratio(json_valid, total),
        vega_lite_valid_rate_total=_safe_ratio(vega_lite_valid, total),
        vega_lite_valid_rate_among_json=_safe_ratio(vega_lite_valid, json_valid),
        output_dir=output_dir.as_posix(),
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


if __name__ == "__main__":
    main()
