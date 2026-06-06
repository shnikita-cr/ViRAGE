from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from src.application.config.bootstrap import bootstrap_project_environment
from src.application.contracts import PipelineRequest
from src.application.config.project_config import load_project_config
from src.application.pipeline import ViRAGEPipeline
from src.domain.models import ModelCallLog, StepLog
from ui.app_components import (
    CHART_MODE_OPTIONS,
    CONFIG_DIR,
    DEFAULT_CONFIG_PATH,
    METRICS_DISABLED,
    METRICS_ENABLED,
    METRICS_OPTIONS,
    apply_streamlit_run_overrides,
    append_live_chart_preview_once,
    build_pending_run_payload,
    config_label,
    discover_config_files,
    final_data_path,
    init_session_state,
    live_chart_preview_from_step,
    path_from_config_label,
    read_table_preview_from_bytes,
    read_table_preview_from_path,
    render_chart,
    render_live,
    render_live_chart_previews,
    render_loading_status,
    render_manual_feedback_form,
    render_metrics,
    render_spec_generation_validation_details,
    render_table_preview,
    render_token_usage_cards,
    resolve_project_path,
    run_setting_defaults_from_config,
    run_setting_defaults_from_pending,
)

RecoverableUiError = (RuntimeError, ValueError, TypeError, OSError, KeyError, IndexError, AttributeError, ImportError)


def run_app() -> None:
    bootstrap_project_environment()
    _configure_page()
    init_session_state()

    config_files = _load_config_files()
    pending_run = st.session_state.pending_run
    controls_disabled = bool(st.session_state.pipeline_running)
    controls = _render_sidebar_controls(config_files, pending_run, controls_disabled)
    uploaded_file, query, run_clicked = _render_input_controls(controls_disabled)

    if run_clicked:
        _lock_pending_run(controls, uploaded_file, query)

    if not pending_run:
        _render_previous_result(config_files)
        st.stop()

    locked = _locked_run_context(pending_run, config_files)
    pipeline = _load_pipeline(locked)
    data_path, temp_dir = _resolve_data_path(locked)
    _render_input_preview(data_path)
    result = _execute_pipeline(pipeline, locked, data_path, temp_dir)
    _store_completed_run(result, locked)
    _render_completed_run(result, locked, config_files)


def _configure_page() -> None:
    st.set_page_config(page_title="ViRAGE", layout="wide")
    st.title("ViRAGE")


def _load_config_files() -> list[Path]:
    config_files = discover_config_files()
    if not config_files:
        st.error(f"No .toml configs found in {CONFIG_DIR}")
        st.stop()
    return config_files


def _default_config_index(config_files: list[Path]) -> int:
    default_path = resolve_project_path(DEFAULT_CONFIG_PATH)
    try:
        return config_files.index(default_path)
    except ValueError:
        return 0


def _selected_indices(config_files: list[Path], pending_run: dict[str, Any] | None) -> tuple[int, int, int]:
    if not pending_run:
        return _default_config_index(config_files), 0, 0
    labels = [config_label(path) for path in config_files]
    return (
        labels.index(pending_run["config_label"]),
        CHART_MODE_OPTIONS.index(pending_run["chart_mode"]),
        0 if pending_run["compute_metrics"] else 1,
    )


def _render_sidebar_controls(
    config_files: list[Path],
    pending_run: dict[str, Any] | None,
    controls_disabled: bool,
) -> dict[str, Any]:
    labels = [config_label(path) for path in config_files]
    selected_config_index, selected_chart_index, selected_metrics_index = _selected_indices(config_files, pending_run)
    selected_settings = run_setting_defaults_from_pending(pending_run)
    with st.sidebar:
        st.header("Run configuration")
        selected_label = st.radio("Configuration file", labels, index=selected_config_index, disabled=controls_disabled)
        selected_path = path_from_config_label(selected_label, config_files)
        selected_settings = _settings_for_sidebar(selected_path, pending_run, selected_settings)
        controls = _render_run_controls(
            selected_label,
            selected_chart_index,
            selected_metrics_index,
            selected_settings,
            controls_disabled,
        )
        _render_locked_settings_notice(controls_disabled, pending_run)
    return controls


def _settings_for_sidebar(
    selected_config_path: Path,
    pending_run: dict[str, Any] | None,
    selected_settings: dict[str, Any],
) -> dict[str, Any]:
    if pending_run:
        return selected_settings
    try:
        selected_config_defaults = load_project_config(selected_config_path)
    except RecoverableUiError:
        selected_config_defaults = None
    return run_setting_defaults_from_config(selected_config_defaults)


def _render_run_controls(
    selected_config_label: str,
    selected_chart_index: int,
    selected_metrics_index: int,
    settings: dict[str, Any],
    controls_disabled: bool,
) -> dict[str, Any]:
    chart_mode = st.radio("Chart output", CHART_MODE_OPTIONS, index=selected_chart_index, disabled=controls_disabled)
    metrics_mode = st.radio("Compute metrics", METRICS_OPTIONS, index=selected_metrics_index, disabled=controls_disabled)
    semantic_loop = st.checkbox("Enable semantic VLM loop", value=bool(settings["semantic_feedback_loop_enabled"]), disabled=controls_disabled)
    return {
        "selected_config_label": selected_config_label,
        "chart_mode": chart_mode,
        "compute_metrics": metrics_mode == METRICS_ENABLED,
        "visrag_enabled": st.checkbox("Enable RAG / VisRAG context", value=bool(settings["visrag_enabled"]), disabled=controls_disabled),
        "analytics_tail_enabled": st.checkbox("Enable analytics tail", value=bool(settings["analytics_tail_enabled"]), disabled=controls_disabled),
        "spec_generation_max_attempts": st.slider("Spec generation attempts", 1, 8, max(1, min(8, int(settings["spec_generation_max_attempts"]))), 1, disabled=controls_disabled),
        "semantic_feedback_loop_enabled": semantic_loop,
        "semantic_feedback_max_attempts": st.slider("Semantic VLM attempts", 1, 5, max(1, min(5, int(settings["semantic_feedback_max_attempts"]))), 1, disabled=controls_disabled or not semantic_loop),
        "semantic_feedback_min_accept_confidence": st.slider("Semantic accept confidence", 0.0, 1.0, max(0.0, min(1.0, float(settings["semantic_feedback_min_accept_confidence"]))), 0.05, disabled=controls_disabled or not semantic_loop),
        "semantic_feedback_save_rejected_specs": st.checkbox("Save rejected specs to feedback corpus", value=bool(settings["semantic_feedback_save_rejected_specs"]), disabled=controls_disabled or not semantic_loop),
    }


def _render_locked_settings_notice(controls_disabled: bool, pending_run: dict[str, Any] | None) -> None:
    st.markdown("---")
    if controls_disabled and pending_run:
        st.info(_locked_settings_text(pending_run))
        return
    st.caption("Settings are locked after pressing Run pipeline.")


def _locked_settings_text(pending_run: dict[str, Any]) -> str:
    metrics = METRICS_ENABLED if pending_run["compute_metrics"] else METRICS_DISABLED
    return (
        "Run settings are locked:\n\n"
        f"- `{pending_run['config_label']}`\n"
        f"- `{pending_run['chart_mode']}`\n"
        f"- metrics: `{metrics}`\n"
        f"- RAG enabled: `{pending_run.get('visrag_enabled', True)}`\n"
        f"- analytics tail: `{pending_run.get('analytics_tail_enabled', True)}`\n"
        f"- spec attempts: `{pending_run.get('spec_generation_max_attempts', 3)}`\n"
        f"- semantic loop: `{pending_run.get('semantic_feedback_loop_enabled', False)}`\n"
        f"- semantic attempts: `{pending_run.get('semantic_feedback_max_attempts', 2)}`"
    )


def _render_input_controls(controls_disabled: bool) -> tuple[Any, str, bool]:
    uploaded_file = st.file_uploader("Upload a table", type=["csv", "xlsx"], disabled=controls_disabled)
    if uploaded_file is not None and not controls_disabled:
        _render_uploaded_preview(uploaded_file)
    query = st.text_area("Request", height=120, placeholder="Например: Покажи тренд продаж по датам и дай основные инсайты", disabled=controls_disabled)
    run_clicked = st.button("Run pipeline", type="primary", disabled=controls_disabled)
    return uploaded_file, query, run_clicked


def _render_uploaded_preview(uploaded_file: Any) -> None:
    try:
        preview_df = read_table_preview_from_bytes(uploaded_file.name, uploaded_file.getvalue())
        with st.expander("Table preview", expanded=True):
            render_table_preview("Uploaded table preview", preview_df)
    except RecoverableUiError as exc:
        st.warning(f"Could not preview the uploaded table: {exc}")


def _lock_pending_run(controls: dict[str, Any], uploaded_file: Any, query: str) -> None:
    if uploaded_file is None:
        st.error("Upload a dataset first.")
        st.stop()
    if not query.strip():
        st.error("Enter a query first.")
        st.stop()
    st.session_state.pending_run = build_pending_run_payload(uploaded_file=uploaded_file, query=query, **controls)
    st.session_state.pipeline_running = True
    st.rerun()


def _render_previous_result(config_files: list[Path]) -> None:
    if not st.session_state.last_result or not st.session_state.last_run_settings:
        return
    st.success("Last pipeline run completed.")
    result = st.session_state.last_result
    run_settings = st.session_state.last_run_settings
    _render_result_main_columns(result, run_settings)
    _render_prepared_preview(result)
    render_manual_feedback_form(result, run_settings, config_files)
    render_metrics(result, run_settings["compute_metrics"])


def _render_result_main_columns(result: Any, run_settings: dict[str, Any]) -> None:
    top_left, top_right = st.columns([1.2, 1])
    with top_left:
        render_chart(result, run_settings["chart_mode"])
        _render_insights(result)
    with top_right:
        st.subheader("Run settings")
        st.json(run_settings)
        st.subheader("Token usage summary")
        render_token_usage_cards(result.token_usage_summary)
        with st.expander("Token usage summary JSON", expanded=False):
            st.json(result.token_usage_summary.model_dump())


def _render_insights(result: Any) -> None:
    st.subheader("Insights")
    if result.insights and result.insights.final_insights:
        for item in result.insights.final_insights:
            st.markdown(f"- {item}")
        return
    st.info("No final insights were produced.")


def _render_prepared_preview(result: Any) -> None:
    try:
        with st.expander("Prepared table preview", expanded=False):
            render_table_preview("Prepared table preview", read_table_preview_from_path(final_data_path(result)))
    except RecoverableUiError as exc:
        st.warning(f"Could not preview prepared table: {exc}")


def _locked_run_context(pending_run: dict[str, Any], config_files: list[Path]) -> dict[str, Any]:
    return {
        "config_label": pending_run["config_label"],
        "config_path": path_from_config_label(pending_run["config_label"], config_files),
        "chart_mode": pending_run["chart_mode"],
        "compute_metrics": bool(pending_run["compute_metrics"]),
        "visrag_enabled": bool(pending_run.get("visrag_enabled", True)),
        "analytics_tail_enabled": bool(pending_run.get("analytics_tail_enabled", True)),
        "spec_generation_max_attempts": int(pending_run.get("spec_generation_max_attempts", 3)),
        "semantic_feedback_loop_enabled": bool(pending_run.get("semantic_feedback_loop_enabled", False)),
        "semantic_feedback_max_attempts": int(pending_run.get("semantic_feedback_max_attempts", 2)),
        "semantic_feedback_min_accept_confidence": float(pending_run.get("semantic_feedback_min_accept_confidence", 0.75)),
        "semantic_feedback_save_rejected_specs": bool(pending_run.get("semantic_feedback_save_rejected_specs", True)),
        "query": pending_run["query"],
        "uploaded_file_name": pending_run.get("uploaded_file_name", "uploaded.csv"),
        "uploaded_file_bytes": pending_run.get("uploaded_file_bytes"),
        "data_path": pending_run.get("data_path"),
        "manual_feedback": str(pending_run.get("manual_feedback") or "").strip(),
    }


def _load_pipeline(locked: dict[str, Any]) -> ViRAGEPipeline:
    try:
        project_config = load_project_config(locked["config_path"])
        project_config = apply_streamlit_run_overrides(project_config=project_config, **_override_values(locked))
        return ViRAGEPipeline.from_project_config(project_config)
    except RecoverableUiError as exc:
        st.session_state.pipeline_running = False
        st.session_state.pending_run = None
        st.exception(exc)
        st.stop()


def _override_values(locked: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "compute_metrics",
        "visrag_enabled",
        "analytics_tail_enabled",
        "spec_generation_max_attempts",
        "semantic_feedback_loop_enabled",
        "semantic_feedback_max_attempts",
        "semantic_feedback_min_accept_confidence",
        "semantic_feedback_save_rejected_specs",
    ]
    return {key: locked[key] for key in keys}


def _resolve_data_path(locked: dict[str, Any]) -> tuple[Path, Path | None]:
    if locked["uploaded_file_bytes"] is not None:
        suffix = Path(locked["uploaded_file_name"]).suffix or ".csv"
        temp_dir = Path(tempfile.mkdtemp(prefix="virage_streamlit_"))
        data_path = temp_dir / f"uploaded{suffix}"
        data_path.write_bytes(locked["uploaded_file_bytes"])
        return data_path, temp_dir
    if locked["data_path"]:
        return resolve_project_path(locked["data_path"]), None
    st.session_state.pipeline_running = False
    st.session_state.pending_run = None
    st.error("No dataset is available for this run.")
    st.stop()


def _render_input_preview(data_path: Path) -> None:
    try:
        with st.expander("Input table preview", expanded=False):
            render_table_preview("Input table preview", read_table_preview_from_path(data_path))
    except RecoverableUiError as exc:
        st.warning(f"Could not preview input table: {exc}")


def _execute_pipeline(pipeline: ViRAGEPipeline, locked: dict[str, Any], data_path: Path, temp_dir: Path | None) -> Any:
    top_left, _ = st.columns([1.2, 1])
    slots = _live_slots(top_left)
    live_steps: list[StepLog] = []
    live_model_calls: list[ModelCallLog] = []
    live_chart_previews: list[dict[str, Any]] = []
    _render_start_status(slots["status"], locked)
    try:
        return pipeline.invoke(
            PipelineRequest(query=locked["query"], data_path=data_path.as_posix(), user_context=_user_context(locked)),
            step_callback=_step_callback(slots, live_steps, live_model_calls, live_chart_previews),
            model_call_callback=_model_call_callback(slots, live_steps, live_model_calls),
        )
    except RecoverableUiError as exc:
        _render_live_state(slots, live_steps, live_model_calls)
        st.session_state.pipeline_running = False
        st.session_state.pending_run = None
        st.exception(exc)
        _cleanup_temp_dir(temp_dir)
        st.stop()
    finally:
        _cleanup_temp_dir(temp_dir)


def _live_slots(top_left: Any) -> dict[str, Any]:
    return {
        "chart": top_left.empty(),
        "insights": top_left.empty(),
        "status": st.empty(),
        "progress": st.empty(),
        "model_calls": st.empty(),
    }


def _render_start_status(status_slot: Any, locked: dict[str, Any]) -> None:
    with status_slot.container():
        render_loading_status(status_slot, _start_status_text(locked))


def _start_status_text(locked: dict[str, Any]) -> str:
    return (
        "Pipeline started with locked settings: "
        f"<code>{locked['config_label']}</code> · <code>{locked['chart_mode']}</code> · "
        f"metrics <code>{METRICS_ENABLED if locked['compute_metrics'] else METRICS_DISABLED}</code> · "
        f"RAG <code>{locked['visrag_enabled']}</code> · analytics tail <code>{locked['analytics_tail_enabled']}</code> · "
        f"spec attempts <code>{locked['spec_generation_max_attempts']}</code> · semantic loop <code>{locked['semantic_feedback_loop_enabled']}</code>"
    )


def _step_callback(slots: dict[str, Any], steps: list[StepLog], calls: list[ModelCallLog], previews: list[dict[str, Any]]) -> Any:
    def on_step(step: StepLog) -> None:
        steps.append(step)
        duration = f" · {step.duration_seconds:.2f}s" if getattr(step, "duration_seconds", 0.0) else ""
        render_loading_status(slots["status"], f"Current step: <code>{step.stage}</code> — {step.title}{duration}")
        preview = live_chart_preview_from_step(step)
        if preview is not None:
            append_live_chart_preview_once(previews, preview)
            render_live_chart_previews(slots["chart"], previews)
        _render_live_state(slots, steps, calls)
    return on_step


def _model_call_callback(slots: dict[str, Any], steps: list[StepLog], calls: list[ModelCallLog]) -> Any:
    def on_model_call(call: ModelCallLog) -> None:
        calls.append(call)
        _render_live_state(slots, steps, calls)
    return on_model_call


def _render_live_state(slots: dict[str, Any], steps: list[StepLog], calls: list[ModelCallLog]) -> None:
    render_live(progress_slot=slots["progress"], model_calls_slot=slots["model_calls"], steps=steps, model_calls=calls)


def _user_context(locked: dict[str, Any]) -> dict[str, Any]:
    feedback = locked["manual_feedback"]
    return {
        "manual_semantic_feedback": [feedback] if feedback else [],
        "feedback_source": "manual_user_comment" if feedback else "",
        "rerun_from_user_feedback": bool(feedback),
    }


def _cleanup_temp_dir(temp_dir: Path | None) -> None:
    if temp_dir is not None:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _store_completed_run(result: Any, locked: dict[str, Any]) -> None:
    st.session_state.pipeline_running = False
    st.session_state.pending_run = None
    st.session_state.last_result = result
    st.session_state.last_run_settings = _last_run_settings(locked)


def _last_run_settings(locked: dict[str, Any]) -> dict[str, Any]:
    return {"config_path": locked["config_label"], **_override_values(locked)} | {"chart_mode": locked["chart_mode"]}


def _render_completed_run(result: Any, locked: dict[str, Any], config_files: list[Path]) -> None:
    st.success("Pipeline completed successfully.")
    top_left, top_right = st.columns([1.2, 1])
    with top_left:
        render_chart(result, locked["chart_mode"])
        _render_insights(result)
    with top_right:
        st.subheader("Run settings")
        st.json(st.session_state.last_run_settings)
        render_spec_generation_validation_details(result)
        _render_prepared_preview(result)
        st.subheader("Token usage summary")
        render_token_usage_cards(result.token_usage_summary)
        with st.expander("Token usage summary JSON", expanded=False):
            st.json(result.token_usage_summary.model_dump())
    render_manual_feedback_form(result, st.session_state.last_run_settings, config_files)
    render_metrics(result, locked["compute_metrics"])
