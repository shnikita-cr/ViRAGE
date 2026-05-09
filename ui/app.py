from __future__ import annotations

import shutil
import sys
import tempfile
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.bootstrap import bootstrap_project_environment
from src.application.contracts import PipelineRequest
from src.application.pipeline import ViRAGEPipeline
from src.application.project_config import DEFAULT_CONFIG_PATH, ProjectConfig, load_project_config
from src.domain.models import ModelCallLog, StepLog


CONFIG_DIR = PROJECT_ROOT / "ui" / "config"

CHART_MODE_INTERACTIVE = "Interactive plot"
CHART_MODE_PNG = "PNG plot"
CHART_MODE_OPTIONS = [CHART_MODE_INTERACTIVE, CHART_MODE_PNG]

METRICS_ENABLED = "Enabled"
METRICS_DISABLED = "Disabled"
METRICS_OPTIONS = [METRICS_ENABLED, METRICS_DISABLED]

EXPECTED_STAGE_COUNT = 13


def resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value.resolve()
    return (PROJECT_ROOT / value).resolve()


def discover_config_files() -> list[Path]:
    files = sorted(CONFIG_DIR.glob("*.toml"))

    default_path = resolve_project_path(DEFAULT_CONFIG_PATH)
    if default_path.exists() and default_path not in files:
        files.insert(0, default_path)

    return files


def config_label(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def path_from_config_label(label: str, config_files: list[Path]) -> Path:
    by_label = {config_label(path): path for path in config_files}
    if label not in by_label:
        raise ValueError(f"Unknown config label: {label}")
    return by_label[label]


def apply_streamlit_run_overrides(
    project_config: ProjectConfig,
    *,
    compute_metrics: bool,
    spec_generation_max_attempts: int,
    semantic_feedback_loop_enabled: bool,
    semantic_feedback_max_attempts: int,
    semantic_feedback_min_accept_confidence: float,
    semantic_feedback_save_rejected_specs: bool,
) -> ProjectConfig:
    streamlit_config = project_config.streamlit.model_copy(
        update={"compute_metrics": compute_metrics},
    )
    settings = project_config.settings.model_copy(
        update={
            "spec_generation_max_attempts": spec_generation_max_attempts,
            "semantic_feedback_loop_enabled": semantic_feedback_loop_enabled,
            "semantic_feedback_max_attempts": semantic_feedback_max_attempts,
            "semantic_feedback_min_accept_confidence": semantic_feedback_min_accept_confidence,
            "semantic_feedback_save_rejected_specs": semantic_feedback_save_rejected_specs,
        },
    )
    return project_config.model_copy(
        deep=True,
        update={"streamlit": streamlit_config, "settings": settings},
    )


def collapse_steps(steps: list[StepLog]) -> list[tuple[StepLog, int]]:
    """Keep the latest log per stage while preserving first-seen order."""

    grouped: OrderedDict[str, tuple[StepLog, int]] = OrderedDict()

    for step in steps:
        previous = grouped.get(step.stage)

        if previous is None:
            grouped[step.stage] = (step, 1)
            continue

        _, count = previous
        grouped[step.stage] = (step, count + 1)

    return list(grouped.values())


def short_json_preview(value: Any, max_items: int = 6) -> Any:
    if isinstance(value, dict):
        return dict(list(value.items())[:max_items])
    if isinstance(value, list):
        return value[:max_items]
    return value


def render_reasoning_trace(steps: list[StepLog], model_calls: list[ModelCallLog]) -> None:
    if not steps:
        st.info("Waiting for the first completed step...")
        return

    collapsed_steps = collapse_steps(steps)
    completed = len(collapsed_steps)
    progress = min(completed / EXPECTED_STAGE_COUNT, 1.0)

    st.progress(progress)
    st.caption(
        f"{completed} pipeline stages · "
        f"{len(steps)} updates · "
        f"{len(model_calls)} model calls"
    )

    with st.expander("Reasoning trace", expanded=True):
        for index, (log, update_count) in enumerate(collapsed_steps, start=1):
            is_latest = index == len(collapsed_steps)
            status_icon = "●" if is_latest else "✓"
            update_suffix = f" · {update_count} updates" if update_count > 1 else ""

            st.markdown(
                f"""
<div style="
    border-left: 3px solid {'#ffb020' if is_latest else '#29a36a'};
    padding: 0.35rem 0 0.35rem 0.85rem;
    margin: 0.45rem 0;
">
  <div style="font-size: 0.9rem; opacity: 0.72;">
    {status_icon} Step {index:02d} · <code>{log.stage}</code>{update_suffix}
  </div>
  <div style="font-weight: 650; margin-top: 0.1rem;">
    {log.title}
  </div>
  <div style="margin-top: 0.1rem;">
    {log.summary}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

    latest_step = collapsed_steps[-1][0]

    if latest_step.inputs or latest_step.outputs or latest_step.details:
        with st.expander("Latest step evidence", expanded=False):
            if latest_step.inputs:
                st.caption("Inputs")
                st.write(latest_step.inputs)

            if latest_step.outputs:
                st.caption("Outputs")
                st.write(latest_step.outputs)

            if latest_step.details:
                st.caption("Details")
                st.json(short_json_preview(latest_step.details))


def render_model_calls(calls: list[ModelCallLog]) -> None:
    st.subheader("Model calls")

    if not calls:
        st.info("No model calls completed yet.")
        return

    summary_rows = []

    for index, call in enumerate(calls, start=1):
        summary_rows.append(
            {
                "idx": f"{call.call_index or index:03d}",
                "stage": call.stage,
                "role": call.model_role,
                "model": call.model_name,
                "tokens": call.token_usage.total_tokens,
                "duration_s": round(call.duration_seconds, 2),
                "errors": len(call.parser_errors),
            }
        )

    st.dataframe(summary_rows, use_container_width=True, hide_index=True)

    with st.expander("Detailed model calls", expanded=False):
        for index, call in enumerate(calls, start=1):
            call_number = call.call_index or index

            st.markdown(f"### {call_number:03d}. {call.stage}")
            st.caption(f"{call.model_role} · {call.model_name}")

            st.caption("Token usage")
            st.json(call.token_usage.model_dump())

            st.caption("Prompt")
            st.code(call.prompt)

            st.caption("Raw response")
            st.code(call.raw_response)

            if call.parsed_preview is not None:
                st.caption("Parsed preview")
                st.json(call.parsed_preview)

            if call.parser_errors:
                st.caption("Parser errors")
                st.write(call.parser_errors)

            st.divider()


def render_live(
    progress_slot: Any,
    model_calls_slot: Any,
    steps: list[StepLog],
    model_calls: list[ModelCallLog],
) -> None:
    with progress_slot.container():
        render_reasoning_trace(steps, model_calls)

    with model_calls_slot.container():
        render_model_calls(model_calls)


def get_result_vegalite_spec(result: Any) -> dict[str, Any] | None:
    if result.spec_validation and result.spec_validation.validated_spec:
        return deepcopy(result.spec_validation.validated_spec)

    if result.vega_spec and result.vega_spec.spec_json:
        return deepcopy(result.vega_spec.spec_json)

    return None


def resolve_existing_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None

    path = Path(path_value)
    candidates = [path]

    if not path.is_absolute():
        candidates.append(PROJECT_ROOT / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return None


def read_chart_data(spec: dict[str, Any], result: Any) -> pd.DataFrame | None:
    data_path: Path | None = None

    if result.data_preparation and result.data_preparation.output_path:
        data_path = resolve_existing_path(result.data_preparation.output_path)

    if data_path is None:
        data = spec.get("data")
        if isinstance(data, dict):
            data_path = resolve_existing_path(data.get("url"))

    if data_path is None:
        return None

    if data_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(data_path)

    return pd.read_csv(data_path)


def render_interactive_vegalite_chart(result: Any) -> bool:
    spec = get_result_vegalite_spec(result)

    if not spec:
        st.warning("No Vega-Lite spec was produced.")
        return False

    try:
        data = read_chart_data(spec, result)
        spec_for_streamlit = deepcopy(spec)
        spec_for_streamlit.pop("data", None)

        if data is None:
            st.vega_lite_chart(spec_for_streamlit, use_container_width=True)
        else:
            st.vega_lite_chart(data, spec_for_streamlit, use_container_width=True)

        with st.expander("Vega-Lite spec", expanded=False):
            st.json(spec)

        return True

    except Exception as exc:
        st.warning("Interactive Vega-Lite rendering failed. Falling back to PNG if available.")
        st.exception(exc)
        return False


def render_png_chart(result: Any) -> bool:
    image_path = result.plot_image.get("image_path") if result.plot_image else None

    if not image_path:
        st.warning("No plot image was produced.")
        return False

    st.image(image_path, use_container_width=True)
    return True


def render_chart(result: Any, chart_mode: str) -> None:
    st.subheader("Rendered chart")

    if chart_mode == CHART_MODE_INTERACTIVE:
        rendered = render_interactive_vegalite_chart(result)
        if not rendered:
            render_png_chart(result)
        return

    render_png_chart(result)



def payload_from_model(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if hasattr(value, "model_dump"):
        payload = value.model_dump()
    else:
        payload = value

    if isinstance(payload, dict):
        return payload

    return {"value": payload}


def numeric_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    return None


def normalize_score(value: Any) -> float | None:
    score = numeric_score(value)

    if score is None:
        return None

    if score > 1.0:
        return max(0.0, min(score / 100.0, 1.0))

    return max(0.0, min(score, 1.0))


def format_score(value: Any) -> str:
    score = numeric_score(value)

    if score is None:
        return "—"

    if score <= 1.0:
        return f"{score * 100:.0f}%"

    return f"{score:.2f}"


def score_badge(score: Any) -> tuple[str, str]:
    normalized = normalize_score(score)

    if normalized is None:
        return "Not available", "⚪"

    if normalized >= 0.85:
        return "Strong", "🟢"

    if normalized >= 0.65:
        return "Acceptable", "🟡"

    return "Needs attention", "🔴"


def render_score_card(
    *,
    title: str,
    score: Any,
    description: str,
) -> None:
    normalized = normalize_score(score)
    status, icon = score_badge(score)

    with st.container(border=True):
        st.markdown(f"#### {icon} {title}")
        st.metric("Score", format_score(score))
        st.caption(description)

        if normalized is not None:
            st.progress(normalized)

        st.caption(status)


def render_component_scores(payload: dict[str, Any], fields: list[tuple[str, str]]) -> None:
    available_fields = [(label, key) for label, key in fields if key in payload]

    if not available_fields:
        return

    columns = st.columns(len(available_fields))

    for column, (label, key) in zip(columns, available_fields):
        with column:
            normalized = normalize_score(payload.get(key))
            st.caption(label)
            st.metric("", format_score(payload.get(key)))

            if normalized is not None:
                st.progress(normalized)


def render_metric_details(title: str, details: Any) -> None:
    if not details:
        st.caption("No details reported.")
        return

    st.markdown(f"**{title}**")

    if isinstance(details, list):
        for item in details:
            st.markdown(f"- {item}")
        return

    if isinstance(details, dict):
        for key, value in details.items():
            st.markdown(f"- **{key}:** {value}")
        return

    st.write(details)


def render_structural_metric(metric: Any) -> None:
    payload = payload_from_model(metric)

    if not payload:
        return

    render_score_card(
        title="Structural specification",
        score=payload.get("score"),
        description="Checks whether the generated Vega-Lite structure matches the requested chart intent.",
    )

    render_component_scores(
        payload,
        [
            ("Mark", "mark_score"),
            ("Encoding", "encoding_score"),
            ("Transform", "transform_score"),
            ("Task alignment", "task_alignment_score"),
        ],
    )

    details = payload.get("details")
    if details:
        with st.expander("Structural findings", expanded=False):
            render_metric_details("Checks", details)


def render_visual_quality_metric(metric: Any) -> None:
    payload = payload_from_model(metric)

    if not payload:
        return

    render_score_card(
        title="Visual quality",
        score=payload.get("score"),
        description="Estimates readability, prompt compliance and whether the image supports useful insight extraction.",
    )

    render_component_scores(
        payload,
        [
            ("Prompt compliance", "prompt_compliance"),
            ("Readability", "readability"),
            ("Insight support", "insight_supportiveness"),
        ],
    )

    details = payload.get("details")
    if details:
        with st.expander("Visual quality findings", expanded=False):
            render_metric_details("Checks", details)


def render_summary_status_card(label: str, value: Any) -> None:
    display_value = "—" if value is None or value == "" else str(value)

    with st.container(border=True):
        st.caption(label)
        st.markdown(f"**{display_value}**")


def render_evaluation_summary(summary: Any) -> None:
    payload = payload_from_model(summary)

    if not payload:
        return

    st.markdown("### Evaluation summary")

    score_cols = st.columns(3)

    with score_cols[0]:
        render_summary_status_card(
            "Structural score",
            format_score(payload.get("structural_spec_metric")),
        )

    with score_cols[1]:
        render_summary_status_card(
            "Visual score",
            format_score(payload.get("visual_quality_metric")),
        )

    with score_cols[2]:
        empty_status = payload.get("empty_chart_status") or "unknown"
        render_summary_status_card("Empty chart status", empty_status)

    verification_summary = payload.get("insight_verification_summary")
    if verification_summary:
        with st.container(border=True):
            st.markdown("#### Insight verification")
            st.write(verification_summary)

    benchmark_report = payload.get("benchmark_report")
    if isinstance(benchmark_report, dict) and benchmark_report:
        compact_report = {
            key: value
            for key, value in benchmark_report.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }

        if compact_report:
            st.markdown("#### Benchmark report")
            report_columns = st.columns(min(len(compact_report), 4))

            for column, (key, value) in zip(report_columns, compact_report.items()):
                with column:
                    label = key.replace("_", " ").title()
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        st.metric(label, format_score(value) if "score" in key or "metric" in key else value)
                    else:
                        st.caption(label)
                        st.markdown(f"**{value}**")

        complex_report = {
            key: value
            for key, value in benchmark_report.items()
            if key not in compact_report
        }

        if complex_report:
            with st.expander("Benchmark details", expanded=False):
                st.json(complex_report)



def read_text_artifact(path_value: str, *, max_chars: int = 30000) -> str:
    try:
        path = Path(path_value)
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Could not read artifact {path_value!r}: {exc}"
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[artifact truncated in UI]"
    return text


def render_spec_generation_validation_details(result: Any) -> None:
    artifact_paths = getattr(result, "artifact_paths", {}) or {}
    if not isinstance(artifact_paths, dict):
        return

    prompt_items = sorted(
        (key, value)
        for key, value in artifact_paths.items()
        if key.startswith("spec_generation_attempt_") and key.endswith("_prompt")
    )
    validation_report_items = sorted(
        (key, value)
        for key, value in artifact_paths.items()
        if key.startswith("spec_validation_attempt_") and key.endswith("_report")
    )

    if not prompt_items and not validation_report_items and not getattr(result, "vega_spec", None):
        return

    with st.expander("Spec generation and validation details", expanded=False):
        if getattr(result, "vega_spec", None):
            st.markdown("#### Final Vega-Lite spec")
            st.json(result.vega_spec.spec_json)

        if prompt_items:
            st.markdown("#### Generation prompts")
            for key, path in prompt_items:
                st.markdown(f"**{key}**")
                st.caption(path)
                st.text_area(
                    label=f"Prompt: {key}",
                    value=read_text_artifact(str(path)),
                    height=280,
                    label_visibility="collapsed",
                )

        if validation_report_items:
            st.markdown("#### Validation reports")
            for key, path in validation_report_items:
                st.markdown(f"**{key}**")
                st.caption(path)
                st.markdown(read_text_artifact(str(path)))

        semantic_summary = getattr(result, "semantic_feedback_loop_summary", None)
        if semantic_summary:
            st.markdown("#### Semantic VLM loop summary")
            st.json(semantic_summary.model_dump())

        for title, payload in [
            ("PNG-only VLM description", getattr(result, "vlm_chart_description", None)),
            ("Chart fact summary", getattr(result, "chart_fact_summary", None)),
            ("Chart answer judge", getattr(result, "chart_answer_judge", None)),
        ]:
            if payload:
                st.markdown(f"#### {title}")
                st.json(payload.model_dump())

def render_metrics(result: Any, compute_metrics: bool) -> None:
    st.subheader("Metrics")

    if not compute_metrics:
        st.info("Metric calculation was disabled for this run.")
        return

    structural_metric = result.structural_spec_metric
    visual_metric = result.visual_quality_metric
    evaluation_summary = result.evaluation_summary

    if not any([structural_metric, visual_metric, evaluation_summary]):
        st.info("No metrics were produced.")
        return

    metric_count = sum(1 for item in [structural_metric, visual_metric, evaluation_summary] if item is not None)

    overview_columns = st.columns(3)

    with overview_columns[0]:
        st.metric("Metric groups", metric_count)

    with overview_columns[1]:
        structural_payload = payload_from_model(structural_metric)
        st.metric("Structural", format_score(structural_payload.get("score")))

    with overview_columns[2]:
        visual_payload = payload_from_model(visual_metric)
        st.metric("Visual", format_score(visual_payload.get("score")))

    st.markdown("---")

    if structural_metric and visual_metric:
        left, right = st.columns(2)

        with left:
            render_structural_metric(structural_metric)

        with right:
            render_visual_quality_metric(visual_metric)

    elif structural_metric:
        render_structural_metric(structural_metric)

    elif visual_metric:
        render_visual_quality_metric(visual_metric)

    if evaluation_summary:
        st.markdown("---")
        render_evaluation_summary(evaluation_summary)

    with st.expander("Raw metric payloads", expanded=False):
        if structural_metric:
            st.markdown("#### Structural spec")
            st.json(payload_from_model(structural_metric))

        if visual_metric:
            st.markdown("#### Visual quality")
            st.json(payload_from_model(visual_metric))

        if evaluation_summary:
            st.markdown("#### Evaluation summary")
            st.json(payload_from_model(evaluation_summary))


def init_session_state() -> None:
    st.session_state.setdefault("pipeline_running", False)
    st.session_state.setdefault("pending_run", None)
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_run_settings", None)


def build_pending_run_payload(
    *,
    selected_config_label: str,
    chart_mode: str,
    compute_metrics: bool,
    spec_generation_max_attempts: int,
    semantic_feedback_loop_enabled: bool,
    semantic_feedback_max_attempts: int,
    semantic_feedback_min_accept_confidence: float,
    semantic_feedback_save_rejected_specs: bool,
    uploaded_file: Any,
    query: str,
) -> dict[str, Any]:
    return {
        "config_label": selected_config_label,
        "chart_mode": chart_mode,
        "compute_metrics": compute_metrics,
        "spec_generation_max_attempts": spec_generation_max_attempts,
        "semantic_feedback_loop_enabled": semantic_feedback_loop_enabled,
        "semantic_feedback_max_attempts": semantic_feedback_max_attempts,
        "semantic_feedback_min_accept_confidence": semantic_feedback_min_accept_confidence,
        "semantic_feedback_save_rejected_specs": semantic_feedback_save_rejected_specs,
        "uploaded_file_name": uploaded_file.name,
        "uploaded_file_bytes": uploaded_file.getvalue(),
        "query": query,
    }


bootstrap_project_environment()

st.set_page_config(page_title="ViRAGE", layout="wide")
st.title("ViRAGE")

init_session_state()

config_files = discover_config_files()

if not config_files:
    st.error(f"No .toml configs found in {CONFIG_DIR}")
    st.stop()

labels = [config_label(path) for path in config_files]
default_path = resolve_project_path(DEFAULT_CONFIG_PATH)

try:
    default_index = config_files.index(default_path)
except ValueError:
    default_index = 0

pending_run = st.session_state.pending_run
controls_disabled = bool(st.session_state.pipeline_running)

if pending_run:
    selected_config_index = labels.index(pending_run["config_label"])
    selected_chart_index = CHART_MODE_OPTIONS.index(pending_run["chart_mode"])
    selected_metrics_index = 0 if pending_run["compute_metrics"] else 1
    selected_spec_attempts = int(pending_run.get("spec_generation_max_attempts", 3))
    selected_semantic_enabled = bool(pending_run.get("semantic_feedback_loop_enabled", False))
    selected_semantic_attempts = int(pending_run.get("semantic_feedback_max_attempts", 2))
    selected_semantic_confidence = float(pending_run.get("semantic_feedback_min_accept_confidence", 0.75))
    selected_semantic_save = bool(pending_run.get("semantic_feedback_save_rejected_specs", True))
else:
    selected_config_index = default_index
    selected_chart_index = 0
    selected_metrics_index = 0
    selected_spec_attempts = 3
    selected_semantic_enabled = False
    selected_semantic_attempts = 2
    selected_semantic_confidence = 0.75
    selected_semantic_save = True

with st.sidebar:
    st.header("Run configuration")

    selected_config_label = st.radio(
        "Configuration file",
        options=labels,
        index=selected_config_index,
        disabled=controls_disabled,
        help="Choose a TOML config from ui/config before running the pipeline.",
    )
    selected_config_path = path_from_config_label(selected_config_label, config_files)
    try:
        selected_config_defaults = load_project_config(selected_config_path)
        configured_spec_attempts = int(selected_config_defaults.settings.spec_generation_max_attempts)
        configured_semantic_enabled = bool(selected_config_defaults.settings.semantic_feedback_loop_enabled)
        configured_semantic_attempts = int(selected_config_defaults.settings.semantic_feedback_max_attempts)
        configured_semantic_confidence = float(selected_config_defaults.settings.semantic_feedback_min_accept_confidence)
        configured_semantic_save = bool(selected_config_defaults.settings.semantic_feedback_save_rejected_specs)
    except Exception:
        configured_spec_attempts = 3
        configured_semantic_enabled = False
        configured_semantic_attempts = 2
        configured_semantic_confidence = 0.75
        configured_semantic_save = True
    if not pending_run:
        selected_spec_attempts = configured_spec_attempts
        selected_semantic_enabled = configured_semantic_enabled
        selected_semantic_attempts = configured_semantic_attempts
        selected_semantic_confidence = configured_semantic_confidence
        selected_semantic_save = configured_semantic_save

    chart_mode = st.radio(
        "Chart output",
        options=CHART_MODE_OPTIONS,
        index=selected_chart_index,
        disabled=controls_disabled,
        help="Interactive plot uses Streamlit Vega-Lite rendering. PNG plot uses the rendered pipeline image.",
    )

    metrics_mode = st.radio(
        "Compute metrics",
        options=METRICS_OPTIONS,
        index=selected_metrics_index,
        disabled=controls_disabled,
        help="Overrides Streamlit metrics flag from TOML for this run.",
    )
    compute_metrics = metrics_mode == METRICS_ENABLED

    spec_generation_max_attempts = st.slider(
        "Spec generation attempts",
        min_value=1,
        max_value=8,
        value=max(1, min(8, int(selected_spec_attempts))),
        step=1,
        disabled=controls_disabled,
        help="Maximum number of graph-level generate → spec validation attempts. Default from config is 3.",
    )

    semantic_feedback_loop_enabled = st.checkbox(
        "Enable semantic VLM loop",
        value=bool(selected_semantic_enabled),
        disabled=controls_disabled,
        help="If enabled, a PNG-only VLM description and semantic judge can trigger spec regeneration.",
    )

    semantic_feedback_max_attempts = st.slider(
        "Semantic VLM attempts",
        min_value=1,
        max_value=5,
        value=max(1, min(5, int(selected_semantic_attempts))),
        step=1,
        disabled=controls_disabled or not semantic_feedback_loop_enabled,
        help="Maximum semantic attempts. Each rejected attempt saves comments and loops back to spec generation.",
    )

    semantic_feedback_min_accept_confidence = st.slider(
        "Semantic accept confidence",
        min_value=0.0,
        max_value=1.0,
        value=max(0.0, min(1.0, float(selected_semantic_confidence))),
        step=0.05,
        disabled=controls_disabled or not semantic_feedback_loop_enabled,
    )

    semantic_feedback_save_rejected_specs = st.checkbox(
        "Save rejected specs to feedback corpus",
        value=bool(selected_semantic_save),
        disabled=controls_disabled or not semantic_feedback_loop_enabled,
    )

    st.markdown("---")

    if controls_disabled and pending_run:
        st.info(
            "Run settings are locked:\n\n"
            f"- `{pending_run['config_label']}`\n"
            f"- `{pending_run['chart_mode']}`\n"
            f"- metrics: `{METRICS_ENABLED if pending_run['compute_metrics'] else METRICS_DISABLED}`\n"
            f"- spec attempts: `{pending_run.get('spec_generation_max_attempts', 3)}`\n"
            f"- semantic loop: `{pending_run.get('semantic_feedback_loop_enabled', False)}`\n"
            f"- semantic attempts: `{pending_run.get('semantic_feedback_max_attempts', 2)}`"
        )
    else:
        st.caption("Settings are locked after pressing Run pipeline.")

uploaded_file = st.file_uploader(
    "Upload a table",
    type=["csv", "xlsx"],
    disabled=controls_disabled,
)

query = st.text_area(
    "Request",
    height=120,
    placeholder="Например: Покажи тренд продаж по датам и дай основные инсайты",
    disabled=controls_disabled,
)

run_clicked = st.button(
    "Run pipeline",
    type="primary",
    disabled=controls_disabled,
)

if run_clicked:
    if uploaded_file is None:
        st.error("Upload a dataset first.")
        st.stop()

    if not query.strip():
        st.error("Enter a query first.")
        st.stop()

    st.session_state.pending_run = build_pending_run_payload(
        selected_config_label=selected_config_label,
        chart_mode=chart_mode,
        compute_metrics=compute_metrics,
        spec_generation_max_attempts=spec_generation_max_attempts,
        semantic_feedback_loop_enabled=semantic_feedback_loop_enabled,
        semantic_feedback_max_attempts=semantic_feedback_max_attempts,
        semantic_feedback_min_accept_confidence=semantic_feedback_min_accept_confidence,
        semantic_feedback_save_rejected_specs=semantic_feedback_save_rejected_specs,
        uploaded_file=uploaded_file,
        query=query,
    )
    st.session_state.pipeline_running = True
    st.rerun()

if not pending_run:
    if st.session_state.last_result and st.session_state.last_run_settings:
        st.success("Last pipeline run completed.")
        result = st.session_state.last_result
        run_settings = st.session_state.last_run_settings

        top_left, top_right = st.columns([1.2, 1])

        with top_left:
            render_chart(result, run_settings["chart_mode"])

            st.subheader("Insights")
            if result.insights and result.insights.final_insights:
                for item in result.insights.final_insights:
                    st.markdown(f"- {item}")
            else:
                st.info("No final insights were produced.")

        with top_right:
            st.subheader("Run settings")
            st.json(run_settings)

            st.subheader("Token usage summary")
            st.json(result.token_usage_summary.model_dump())

        render_metrics(result, run_settings["compute_metrics"])

    st.stop()

# Pipeline execution starts only after a rerun with locked pending_run.
locked_config_label = pending_run["config_label"]
locked_config_path = path_from_config_label(locked_config_label, config_files)
locked_chart_mode = pending_run["chart_mode"]
locked_compute_metrics = bool(pending_run["compute_metrics"])
locked_spec_generation_max_attempts = int(pending_run.get("spec_generation_max_attempts", 3))
locked_semantic_feedback_loop_enabled = bool(pending_run.get("semantic_feedback_loop_enabled", False))
locked_semantic_feedback_max_attempts = int(pending_run.get("semantic_feedback_max_attempts", 2))
locked_semantic_feedback_min_accept_confidence = float(pending_run.get("semantic_feedback_min_accept_confidence", 0.75))
locked_semantic_feedback_save_rejected_specs = bool(pending_run.get("semantic_feedback_save_rejected_specs", True))
locked_query = pending_run["query"]
uploaded_file_name = pending_run["uploaded_file_name"]
uploaded_file_bytes = pending_run["uploaded_file_bytes"]

try:
    project_config = load_project_config(locked_config_path)
    project_config = apply_streamlit_run_overrides(
        project_config=project_config,
        compute_metrics=locked_compute_metrics,
        spec_generation_max_attempts=locked_spec_generation_max_attempts,
        semantic_feedback_loop_enabled=locked_semantic_feedback_loop_enabled,
        semantic_feedback_max_attempts=locked_semantic_feedback_max_attempts,
        semantic_feedback_min_accept_confidence=locked_semantic_feedback_min_accept_confidence,
        semantic_feedback_save_rejected_specs=locked_semantic_feedback_save_rejected_specs,
    )
    pipeline = ViRAGEPipeline.from_project_config(project_config)

except Exception as exc:
    st.session_state.pipeline_running = False
    st.session_state.pending_run = None
    st.exception(exc)
    st.stop()

suffix = Path(uploaded_file_name).suffix or ".csv"
temp_dir = Path(tempfile.mkdtemp(prefix="virage_streamlit_"))
data_path = temp_dir / f"uploaded{suffix}"
data_path.write_bytes(uploaded_file_bytes)

top_left, top_right = st.columns([1.2, 1])
chart_slot = top_left.empty()
insights_slot = top_left.empty()
status_slot = st.empty()

progress_slot = st.empty()
model_calls_slot = st.empty()

live_steps: list[StepLog] = []
live_model_calls: list[ModelCallLog] = []

with status_slot.container():
    st.info(
        "Pipeline started with locked settings: "
        f"`{locked_config_label}` · `{locked_chart_mode}` · "
        f"metrics `{METRICS_ENABLED if locked_compute_metrics else METRICS_DISABLED}` · "
        f"spec attempts `{locked_spec_generation_max_attempts}` · "
        f"semantic loop `{locked_semantic_feedback_loop_enabled}`"
    )


def on_step(step: StepLog) -> None:
    live_steps.append(step)
    status_slot.info(f"Current step: {step.stage} — {step.title}")
    render_live(
        progress_slot=progress_slot,
        model_calls_slot=model_calls_slot,
        steps=live_steps,
        model_calls=live_model_calls,
    )


def on_model_call(call: ModelCallLog) -> None:
    live_model_calls.append(call)
    render_live(
        progress_slot=progress_slot,
        model_calls_slot=model_calls_slot,
        steps=live_steps,
        model_calls=live_model_calls,
    )


try:
    result = pipeline.invoke(
        PipelineRequest(
            query=locked_query,
            data_path=data_path.as_posix(),
        ),
        step_callback=on_step,
        model_call_callback=on_model_call,
    )

except Exception as exc:
    render_live(
        progress_slot=progress_slot,
        model_calls_slot=model_calls_slot,
        steps=live_steps,
        model_calls=live_model_calls,
    )
    st.session_state.pipeline_running = False
    st.session_state.pending_run = None
    st.exception(exc)
    shutil.rmtree(temp_dir, ignore_errors=True)
    st.stop()

finally:
    shutil.rmtree(temp_dir, ignore_errors=True)

st.session_state.pipeline_running = False
st.session_state.pending_run = None
st.session_state.last_result = result
st.session_state.last_run_settings = {
    "config_path": locked_config_label,
    "chart_mode": locked_chart_mode,
    "compute_metrics": locked_compute_metrics,
    "spec_generation_max_attempts": locked_spec_generation_max_attempts,
    "semantic_feedback_loop_enabled": locked_semantic_feedback_loop_enabled,
    "semantic_feedback_max_attempts": locked_semantic_feedback_max_attempts,
    "semantic_feedback_min_accept_confidence": locked_semantic_feedback_min_accept_confidence,
    "semantic_feedback_save_rejected_specs": locked_semantic_feedback_save_rejected_specs,
}

status_slot.success("Pipeline completed successfully.")

with chart_slot.container():
    render_chart(result, locked_chart_mode)

with insights_slot.container():
    st.subheader("Insights")

    if result.insights and result.insights.final_insights:
        for item in result.insights.final_insights:
            st.markdown(f"- {item}")
    else:
        st.info("No final insights were produced.")

with top_right:
    st.subheader("Run settings")
    st.json(st.session_state.last_run_settings)

    render_spec_generation_validation_details(result)

    st.subheader("Token usage summary")
    st.json(result.token_usage_summary.model_dump())

render_metrics(result, locked_compute_metrics)
