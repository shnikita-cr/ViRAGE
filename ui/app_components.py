from __future__ import annotations

import json
import shutil
import sys
import tempfile
from io import BytesIO
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
from src.infrastructure.runtime import RuntimeContext
from src.services.visual_feedback.feedback_corpus_writer import FeedbackCorpusWriterService


CONFIG_DIR = PROJECT_ROOT / "ui" / "config"
APP_CONFIG_DIR = CONFIG_DIR / "app"

CHART_MODE_INTERACTIVE = "Interactive plot"
CHART_MODE_PNG = "PNG plot"
CHART_MODE_OPTIONS = [CHART_MODE_INTERACTIVE, CHART_MODE_PNG]

METRICS_ENABLED = "Enabled"
METRICS_DISABLED = "Disabled"
METRICS_OPTIONS = [METRICS_ENABLED, METRICS_DISABLED]

EXPECTED_STAGE_COUNT = 24

RUN_SETTING_DEFAULTS = {
    "visrag_enabled": True,
    "analytics_tail_enabled": True,
    "spec_generation_max_attempts": 3,
    "semantic_feedback_loop_enabled": False,
    "semantic_feedback_max_attempts": 2,
    "semantic_feedback_min_accept_confidence": 0.75,
    "semantic_feedback_save_rejected_specs": True,
}


def run_setting_defaults_from_config(project_config: ProjectConfig | None) -> dict[str, Any]:
    if project_config is None:
        return dict(RUN_SETTING_DEFAULTS)
    settings = project_config.settings
    return {
        "visrag_enabled": bool(settings.visrag_enabled),
        "analytics_tail_enabled": bool(settings.analytics_tail_enabled),
        "spec_generation_max_attempts": int(settings.spec_generation_max_attempts),
        "semantic_feedback_loop_enabled": bool(settings.semantic_feedback_loop_enabled),
        "semantic_feedback_max_attempts": int(settings.semantic_feedback_max_attempts),
        "semantic_feedback_min_accept_confidence": float(settings.semantic_feedback_min_accept_confidence),
        "semantic_feedback_save_rejected_specs": bool(settings.semantic_feedback_save_rejected_specs),
    }


def run_setting_defaults_from_pending(pending_run: dict[str, Any] | None) -> dict[str, Any]:
    values = dict(RUN_SETTING_DEFAULTS)
    if pending_run:
        for key in RUN_SETTING_DEFAULTS:
            if key in pending_run:
                values[key] = pending_run[key]
    return values


def run_settings_summary(settings: dict[str, Any]) -> str:
    return (
        f"RAG `{settings.get('visrag_enabled', True)}` · "
        f"analytics tail `{settings.get('analytics_tail_enabled', True)}` · "
        f"spec attempts `{settings.get('spec_generation_max_attempts', 3)}` · "
        f"semantic loop `{settings.get('semantic_feedback_loop_enabled', False)}`"
    )


def render_loading_status(slot, message: str) -> None:
    slot.markdown(
        f"""
<style>
.virage-spinner {{
  display: inline-block;
  width: 0.9rem;
  height: 0.9rem;
  border: 2px solid rgba(49, 130, 206, 0.25);
  border-top-color: rgba(49, 130, 206, 1);
  border-radius: 50%;
  animation: virage-spin 0.85s linear infinite;
  margin-right: 0.45rem;
  vertical-align: -0.12rem;
}}
@keyframes virage-spin {{
  to {{ transform: rotate(360deg); }}
}}
</style>
<div style="padding: 0.75rem 1rem; border: 1px solid rgba(49,130,206,.25); border-radius: .5rem; background: rgba(49,130,206,.08);">
  <span class="virage-spinner"></span>{message}
</div>
""",
        unsafe_allow_html=True,
    )


def resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value.resolve()
    return (PROJECT_ROOT / value).resolve()


def discover_config_files() -> list[Path]:
    search_dir = APP_CONFIG_DIR if APP_CONFIG_DIR.exists() else CONFIG_DIR
    files = sorted(search_dir.glob("*.toml"))

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


def read_table_preview_from_bytes(file_name: str, payload: bytes, *, max_rows: int = 20) -> pd.DataFrame:
    suffix = Path(file_name).suffix.lower()
    buffer = BytesIO(payload)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(buffer, nrows=max_rows)
    return pd.read_csv(buffer, nrows=max_rows)


def read_table_preview_from_path(path_value: str | Path, *, max_rows: int = 20) -> pd.DataFrame:
    path = Path(path_value)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, nrows=max_rows)
    return pd.read_csv(path, nrows=max_rows)


def render_table_preview(title: str, data: pd.DataFrame) -> None:
    st.subheader(title)
    left, middle, right = st.columns(3)
    with left:
        st.metric("Preview rows", len(data))
    with middle:
        st.metric("Columns", len(data.columns))
    with right:
        st.metric("Missing values", int(data.isna().sum().sum()))
    st.dataframe(data, width="stretch", hide_index=True)
    with st.expander("Column types", expanded=False):
        st.dataframe(
            [{"column": name, "dtype": str(dtype)} for name, dtype in data.dtypes.items()],
            width="stretch",
            hide_index=True,
        )


def final_data_path(result: Any) -> str:
    if getattr(result, "data_preparation", None) and result.data_preparation.output_path:
        return result.data_preparation.output_path
    return result.data_path


def save_manual_feedback_artifact(result: Any, example: Any, corpus_path: str) -> str:
    run_dir = PROJECT_ROOT / "artifacts" / result.run_id
    manual_dir = run_dir / "manual_feedback"
    manual_dir.mkdir(parents=True, exist_ok=True)
    index = len(sorted(manual_dir.glob("*_user_feedback.json"))) + 1
    path = manual_dir / f"{index:03d}_user_feedback.json"
    payload = example.model_dump() if hasattr(example, "model_dump") else dict(example)
    payload["corpus_path"] = corpus_path
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path.as_posix()


def save_manual_feedback_for_result(
    *,
    result: Any,
    project_config: ProjectConfig,
    comment: str,
    needs_regeneration: bool,
) -> tuple[str, str]:
    writer = FeedbackCorpusWriterService()
    runtime = RuntimeContext(settings=project_config.settings)
    runtime.current_run_id = result.run_id
    image_path = result.plot_image.get("image_path") if result.plot_image else ""
    if result.vega_spec is None:
        raise ValueError("Cannot save chart feedback because the run has no Vega-Lite specification.")
    example = writer.build_user_feedback_example(
        run_id=result.run_id,
        query=result.query,
        comment=comment,
        needs_regeneration=needs_regeneration,
        vega_spec=result.vega_spec,
        rendered_png_path=image_path or "",
        request_analysis=result.query_request_analysis,
        attempt_number=(result.semantic_feedback_loop_summary.attempt_count if result.semantic_feedback_loop_summary else 1),
    )
    corpus_path = writer.append_to_corpus(example, runtime)
    artifact_path = save_manual_feedback_artifact(result, example, corpus_path)
    return corpus_path, artifact_path


def build_manual_rerun_payload(
    *,
    previous_result: Any,
    previous_settings: dict[str, Any],
    manual_feedback: str,
) -> dict[str, Any]:
    return {
        "config_label": previous_settings["config_path"],
        "chart_mode": previous_settings["chart_mode"],
        "compute_metrics": bool(previous_settings["compute_metrics"]),
        "visrag_enabled": bool(previous_settings.get("visrag_enabled", True)),
        "analytics_tail_enabled": bool(previous_settings.get("analytics_tail_enabled", True)),
        "spec_generation_max_attempts": int(previous_settings.get("spec_generation_max_attempts", 3)),
        "semantic_feedback_loop_enabled": bool(previous_settings.get("semantic_feedback_loop_enabled", False)),
        "semantic_feedback_max_attempts": int(previous_settings.get("semantic_feedback_max_attempts", 2)),
        "semantic_feedback_min_accept_confidence": float(previous_settings.get("semantic_feedback_min_accept_confidence", 0.75)),
        "semantic_feedback_save_rejected_specs": bool(previous_settings.get("semantic_feedback_save_rejected_specs", True)),
        "data_path": final_data_path(previous_result),
        "query": previous_result.query,
        "manual_feedback": manual_feedback,
    }


def render_manual_feedback_form(result: Any, run_settings: dict[str, Any], config_files: list[Path]) -> None:
    st.subheader("Manual chart feedback")
    with st.form("manual_chart_feedback_form", clear_on_submit=False):
        comment = st.text_area(
            "Comment or correction",
            height=120,
            placeholder="Например: Сделай горизонтальные столбцы и раздели метрики по независимым шкалам.",
        )
        needs_regeneration = st.checkbox("Regenerate chart using this comment", value=False)
        submitted = st.form_submit_button("Save feedback")

    if not submitted:
        return

    if not comment.strip():
        st.warning("Write a comment before saving feedback.")
        return

    config_path = path_from_config_label(run_settings["config_path"], config_files)
    project_config = load_project_config(config_path)
    project_config = apply_streamlit_run_overrides(
        project_config=project_config,
        compute_metrics=bool(run_settings["compute_metrics"]),
        visrag_enabled=bool(run_settings.get("visrag_enabled", True)),
        analytics_tail_enabled=bool(run_settings.get("analytics_tail_enabled", True)),
        spec_generation_max_attempts=int(run_settings.get("spec_generation_max_attempts", 3)),
        semantic_feedback_loop_enabled=bool(run_settings.get("semantic_feedback_loop_enabled", False)),
        semantic_feedback_max_attempts=int(run_settings.get("semantic_feedback_max_attempts", 2)),
        semantic_feedback_min_accept_confidence=float(run_settings.get("semantic_feedback_min_accept_confidence", 0.75)),
        semantic_feedback_save_rejected_specs=bool(run_settings.get("semantic_feedback_save_rejected_specs", True)),
    )
    corpus_path, artifact_path = save_manual_feedback_for_result(
        result=result,
        project_config=project_config,
        comment=comment,
        needs_regeneration=needs_regeneration,
    )
    st.success(f"Feedback saved to corpus: {corpus_path}")
    st.caption(f"Run artifact: {artifact_path}")

    if needs_regeneration:
        st.session_state.pending_run = build_manual_rerun_payload(
            previous_result=result,
            previous_settings=run_settings,
            manual_feedback=comment.strip(),
        )
        st.session_state.pipeline_running = True
        st.rerun()


def apply_streamlit_run_overrides(
    project_config: ProjectConfig,
    *,
    compute_metrics: bool,
    visrag_enabled: bool,
    analytics_tail_enabled: bool,
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
            "visrag_enabled": visrag_enabled,
            "spec_generation_include_visrag_context": visrag_enabled,
            "analytics_tail_enabled": analytics_tail_enabled,
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

            duration_suffix = ""
            if getattr(log, "duration_seconds", 0.0):
                duration_suffix = f" · {log.duration_seconds:.2f}s"

            st.markdown(
                f"""
<div style="
    border-left: 3px solid {'#ffb020' if is_latest else '#29a36a'};
    padding: 0.35rem 0 0.35rem 0.85rem;
    margin: 0.45rem 0;
">
  <div style="font-size: 0.9rem; opacity: 0.72;">
    {status_icon} Step {index:02d} · <code>{log.stage}</code>{update_suffix}{duration_suffix}
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


def render_token_usage_cards(token_usage: Any) -> None:
    input_tokens = int(getattr(token_usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(token_usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(token_usage, "total_tokens", 0) or 0)

    col_input, col_output, col_total = st.columns(3)
    col_input.metric("Input tokens", input_tokens)
    col_output.metric("Output tokens", output_tokens)
    col_total.metric("Total tokens", total_tokens)


def render_model_calls(calls: list[ModelCallLog]) -> None:
    st.subheader("Model calls")

    if not calls:
        st.info("No model calls completed yet.")
        return

    summary_rows = []

    for index, call in enumerate(calls, start=1):
        usage = call.token_usage
        summary_rows.append(
            {
                "idx": f"{call.call_index or index:03d}",
                "stage": call.stage,
                "role": call.model_role,
                "model": call.model_name,
                "input_tokens": int(usage.prompt_tokens or 0),
                "output_tokens": int(usage.completion_tokens or 0),
                "total_tokens": int(usage.total_tokens or 0),
                "duration_s": round(call.duration_seconds, 2),
                "errors": len(call.parser_errors),
            }
        )

    st.dataframe(summary_rows, width="stretch", hide_index=True)

    with st.expander("Detailed model calls", expanded=False):
        for index, call in enumerate(calls, start=1):
            call_number = call.call_index or index

            st.markdown(f"### {call_number:03d}. {call.stage}")
            st.caption(f"{call.model_role} · {call.model_name}")

            st.caption("Token usage")
            render_token_usage_cards(call.token_usage)

            with st.expander("Token usage JSON", expanded=False):
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
            st.vega_lite_chart(spec_for_streamlit, width="stretch")
        else:
            st.vega_lite_chart(data, spec_for_streamlit, width="stretch")

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

    st.image(image_path, width="stretch")
    return True


def render_chart(result: Any, chart_mode: str) -> None:
    st.subheader("Rendered chart")

    if chart_mode == CHART_MODE_INTERACTIVE:
        rendered = render_interactive_vegalite_chart(result)
        if not rendered:
            render_png_chart(result)
        return

    render_png_chart(result)


def live_chart_preview_from_step(step: StepLog) -> dict[str, Any] | None:
    details = getattr(step, "details", {}) or {}
    if not isinstance(details, dict):
        return None

    preview = details.get("live_chart_preview")
    if not isinstance(preview, dict):
        return None

    image_path = preview.get("image_path")
    if not image_path:
        return None

    preview_id = str(preview.get("artifact") or image_path)
    return {**preview, "_preview_id": preview_id}


def append_live_chart_preview_once(previews: list[dict[str, Any]], preview: dict[str, Any]) -> None:
    preview_id = str(preview.get("_preview_id") or "")
    if preview_id and any(str(item.get("_preview_id") or "") == preview_id for item in previews):
        return
    previews.append(preview)


def render_live_chart_previews(chart_slot: Any, previews: list[dict[str, Any]]) -> None:
    if not previews:
        return

    latest = previews[-1]
    image_path = resolve_existing_path(str(latest.get("image_path") or ""))
    semantic_attempt = int(latest.get("semantic_attempt_number") or 1)
    technical_attempt = int(latest.get("technical_attempt_number") or 1)
    empty_status = (latest.get("empty_chart_check") or {}).get("empty_chart_status", "unknown")
    scenegraph = latest.get("scenegraph_check") or {}

    with chart_slot.container():
        st.subheader("Current chart preview")
        st.caption(
            "Shown after successful technical validation, before semantic/VLM validation. "
            f"Semantic attempt {semantic_attempt}, technical attempt {technical_attempt}."
        )

        if image_path is not None:
            st.image(image_path.as_posix(), width="stretch")
        else:
            st.warning(f"Preview image was produced, but the file was not found: {latest.get('image_path')}")

        left, middle, right = st.columns(3)
        with left:
            st.metric("Empty check", empty_status)
        with middle:
            st.metric("Marks", "yes" if scenegraph.get("has_marks") else "unknown")
        with right:
            st.metric("Axes", "yes" if scenegraph.get("has_axes") else "unknown")

        if len(previews) > 1:
            with st.expander("Preview history", expanded=False):
                rows = []
                for index, item in enumerate(previews, start=1):
                    rows.append({
                        "preview": index,
                        "semantic_attempt": item.get("semantic_attempt_number"),
                        "technical_attempt": item.get("technical_attempt_number"),
                        "empty_status": (item.get("empty_chart_check") or {}).get("empty_chart_status", "unknown"),
                        "image_path": item.get("image_path"),
                        "artifact": item.get("artifact"),
                    })
                st.dataframe(rows, width="stretch", hide_index=True)

        spec = latest.get("validated_spec")
        if isinstance(spec, dict) and spec:
            with st.expander("Current validated Vega-Lite spec", expanded=False):
                st.json(spec)


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

    score_cols = st.columns(4)

    with score_cols[0]:
        render_summary_status_card(
            "Structural score",
            format_score(payload.get("structural_spec_metric")),
        )

    with score_cols[1]:
        render_summary_status_card("Technical", payload.get("technical_status") or "unknown")

    with score_cols[2]:
        render_summary_status_card("Semantic", payload.get("semantic_status") or "unknown")

    with score_cols[3]:
        render_summary_status_card("Chart accepted", payload.get("chart_accepted"))

    empty_status = payload.get("empty_chart_status") or "unknown"
    render_summary_status_card("Empty chart status", empty_status)

    verification_summary = payload.get("insight_summary")
    if verification_summary:
        with st.container(border=True):
            st.markdown("#### Insight summary")
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
    evaluation_summary = result.evaluation_summary

    if not any([structural_metric, evaluation_summary]):
        st.info("No metrics were produced.")
        return

    metric_count = sum(1 for item in [structural_metric, evaluation_summary] if item is not None)

    overview_columns = st.columns(3)

    with overview_columns[0]:
        st.metric("Metric groups", metric_count)

    with overview_columns[1]:
        structural_payload = payload_from_model(structural_metric)
        st.metric("Structural", format_score(structural_payload.get("score")))

    with overview_columns[2]:
        evaluation_payload = payload_from_model(evaluation_summary)
        st.metric("Chart accepted", str(evaluation_payload.get("chart_accepted", "—")))

    st.markdown("---")

    if structural_metric:
        render_structural_metric(structural_metric)

    if evaluation_summary:
        st.markdown("---")
        render_evaluation_summary(evaluation_summary)

    with st.expander("Raw metric payloads", expanded=False):
        if structural_metric:
            st.markdown("#### Structural spec")
            st.json(payload_from_model(structural_metric))

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
    visrag_enabled: bool,
    analytics_tail_enabled: bool,
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
        "visrag_enabled": visrag_enabled,
        "analytics_tail_enabled": analytics_tail_enabled,
        "spec_generation_max_attempts": spec_generation_max_attempts,
        "semantic_feedback_loop_enabled": semantic_feedback_loop_enabled,
        "semantic_feedback_max_attempts": semantic_feedback_max_attempts,
        "semantic_feedback_min_accept_confidence": semantic_feedback_min_accept_confidence,
        "semantic_feedback_save_rejected_specs": semantic_feedback_save_rejected_specs,
        "uploaded_file_name": uploaded_file.name,
        "uploaded_file_bytes": uploaded_file.getvalue(),
        "query": query,
        "manual_feedback": "",
    }


