from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import streamlit as st

from src.application.bootstrap import bootstrap_project_environment
from src.application.contracts import PipelineRequest
from src.application.pipeline import ViRAGEPipeline
from src.application.project_config import load_project_config
from src.domain.models import ModelCallLog, StepLog
from ui.app_components import (
    CHART_MODE_OPTIONS,
    CONFIG_DIR,
    DEFAULT_CONFIG_PATH,
    EXPECTED_STAGE_COUNT,
    METRICS_DISABLED,
    METRICS_ENABLED,
    METRICS_OPTIONS,
    apply_streamlit_run_overrides,
    build_pending_run_payload,
    collapse_steps,
    config_label,
    discover_config_files,
    final_data_path,
    init_session_state,
    live_chart_preview_from_step,
    append_live_chart_preview_once,
    path_from_config_label,
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
)


def run_app() -> None:
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
        selected_visrag_enabled = bool(pending_run.get("visrag_enabled", True))
        selected_spec_attempts = int(pending_run.get("spec_generation_max_attempts", 3))
        selected_semantic_enabled = bool(pending_run.get("semantic_feedback_loop_enabled", False))
        selected_semantic_attempts = int(pending_run.get("semantic_feedback_max_attempts", 2))
        selected_semantic_confidence = float(pending_run.get("semantic_feedback_min_accept_confidence", 0.75))
        selected_semantic_save = bool(pending_run.get("semantic_feedback_save_rejected_specs", True))
    else:
        selected_config_index = default_index
        selected_chart_index = 0
        selected_metrics_index = 0
        selected_visrag_enabled = True
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
            configured_visrag_enabled = bool(getattr(selected_config_defaults.settings, "visrag_enabled", True))
            configured_spec_attempts = int(selected_config_defaults.settings.spec_generation_max_attempts)
            configured_semantic_enabled = bool(selected_config_defaults.settings.semantic_feedback_loop_enabled)
            configured_semantic_attempts = int(selected_config_defaults.settings.semantic_feedback_max_attempts)
            configured_semantic_confidence = float(selected_config_defaults.settings.semantic_feedback_min_accept_confidence)
            configured_semantic_save = bool(selected_config_defaults.settings.semantic_feedback_save_rejected_specs)
        except Exception:
            configured_visrag_enabled = True
            configured_spec_attempts = 3
            configured_semantic_enabled = False
            configured_semantic_attempts = 2
            configured_semantic_confidence = 0.75
            configured_semantic_save = True
        if not pending_run:
            selected_visrag_enabled = configured_visrag_enabled
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

        visrag_enabled = st.checkbox(
            "Enable RAG / VisRAG context",
            value=bool(selected_visrag_enabled),
            disabled=controls_disabled,
            help="If disabled, VisRAG retrieval is skipped and the spec generator works from query + data profile only.",
        )

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
                f"- RAG enabled: `{pending_run.get('visrag_enabled', True)}`\n"
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

    if uploaded_file is not None and not controls_disabled:
        try:
            preview_df = read_table_preview_from_bytes(uploaded_file.name, uploaded_file.getvalue())
            with st.expander("Table preview", expanded=True):
                render_table_preview("Uploaded table preview", preview_df)
        except Exception as exc:
            st.warning(f"Could not preview the uploaded table: {exc}")

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
            visrag_enabled=visrag_enabled,
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
                render_token_usage_cards(result.token_usage_summary)
                with st.expander("Token usage summary JSON", expanded=False):
                    st.json(result.token_usage_summary.model_dump())

            try:
                with st.expander("Prepared table preview", expanded=False):
                    render_table_preview("Prepared table preview", read_table_preview_from_path(final_data_path(result)))
            except Exception as exc:
                st.warning(f"Could not preview prepared table: {exc}")

            render_manual_feedback_form(result, run_settings, config_files)

            render_metrics(result, run_settings["compute_metrics"])

        st.stop()

    # Pipeline execution starts only after a rerun with locked pending_run.
    locked_config_label = pending_run["config_label"]
    locked_config_path = path_from_config_label(locked_config_label, config_files)
    locked_chart_mode = pending_run["chart_mode"]
    locked_compute_metrics = bool(pending_run["compute_metrics"])
    locked_visrag_enabled = bool(pending_run.get("visrag_enabled", True))
    locked_spec_generation_max_attempts = int(pending_run.get("spec_generation_max_attempts", 3))
    locked_semantic_feedback_loop_enabled = bool(pending_run.get("semantic_feedback_loop_enabled", False))
    locked_semantic_feedback_max_attempts = int(pending_run.get("semantic_feedback_max_attempts", 2))
    locked_semantic_feedback_min_accept_confidence = float(pending_run.get("semantic_feedback_min_accept_confidence", 0.75))
    locked_semantic_feedback_save_rejected_specs = bool(pending_run.get("semantic_feedback_save_rejected_specs", True))
    locked_query = pending_run["query"]
    uploaded_file_name = pending_run.get("uploaded_file_name", "uploaded.csv")
    uploaded_file_bytes = pending_run.get("uploaded_file_bytes")
    locked_data_path = pending_run.get("data_path")
    locked_manual_feedback = str(pending_run.get("manual_feedback") or "").strip()

    try:
        project_config = load_project_config(locked_config_path)
        project_config = apply_streamlit_run_overrides(
            project_config=project_config,
            compute_metrics=locked_compute_metrics,
            visrag_enabled=locked_visrag_enabled,
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

    temp_dir: Path | None = None
    if uploaded_file_bytes is not None:
        suffix = Path(uploaded_file_name).suffix or ".csv"
        temp_dir = Path(tempfile.mkdtemp(prefix="virage_streamlit_"))
        data_path = temp_dir / f"uploaded{suffix}"
        data_path.write_bytes(uploaded_file_bytes)
    elif locked_data_path:
        data_path = resolve_project_path(locked_data_path)
    else:
        st.session_state.pipeline_running = False
        st.session_state.pending_run = None
        st.error("No dataset is available for this run.")
        st.stop()

    try:
        with st.expander("Input table preview", expanded=False):
            render_table_preview("Input table preview", read_table_preview_from_path(data_path))
    except Exception as exc:
        st.warning(f"Could not preview input table: {exc}")

    top_left, top_right = st.columns([1.2, 1])
    chart_slot = top_left.empty()
    insights_slot = top_left.empty()
    status_slot = st.empty()

    progress_slot = st.empty()
    model_calls_slot = st.empty()

    live_steps: list[StepLog] = []
    live_model_calls: list[ModelCallLog] = []
    live_chart_previews: list[dict[str, Any]] = []

    with status_slot.container():
        render_loading_status(
            status_slot,
            "Pipeline started with locked settings: "
            f"<code>{locked_config_label}</code> · <code>{locked_chart_mode}</code> · "
            f"metrics <code>{METRICS_ENABLED if locked_compute_metrics else METRICS_DISABLED}</code> · "
            f"RAG <code>{locked_visrag_enabled}</code> · "
            f"spec attempts <code>{locked_spec_generation_max_attempts}</code> · "
            f"semantic loop <code>{locked_semantic_feedback_loop_enabled}</code>"
        )


    def on_step(step: StepLog) -> None:
        live_steps.append(step)
        duration = f" · {step.duration_seconds:.2f}s" if getattr(step, "duration_seconds", 0.0) else ""
        render_loading_status(status_slot, f"Current step: <code>{step.stage}</code> — {step.title}{duration}")

        preview = live_chart_preview_from_step(step)
        if preview is not None:
            append_live_chart_preview_once(live_chart_previews, preview)
            render_live_chart_previews(chart_slot, live_chart_previews)

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
                user_context={
                    "manual_semantic_feedback": [locked_manual_feedback] if locked_manual_feedback else [],
                    "feedback_source": "manual_user_comment" if locked_manual_feedback else "",
                    "rerun_from_user_feedback": bool(locked_manual_feedback),
                },
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
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
        st.stop()

    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)

    st.session_state.pipeline_running = False
    st.session_state.pending_run = None
    st.session_state.last_result = result
    st.session_state.last_run_settings = {
        "config_path": locked_config_label,
        "chart_mode": locked_chart_mode,
        "compute_metrics": locked_compute_metrics,
        "visrag_enabled": locked_visrag_enabled,
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

        try:
            with st.expander("Prepared table preview", expanded=False):
                render_table_preview("Prepared table preview", read_table_preview_from_path(final_data_path(result)))
        except Exception as exc:
            st.warning(f"Could not preview prepared table: {exc}")

        st.subheader("Token usage summary")
        render_token_usage_cards(result.token_usage_summary)
        with st.expander("Token usage summary JSON", expanded=False):
            st.json(result.token_usage_summary.model_dump())

    render_manual_feedback_form(result, st.session_state.last_run_settings, config_files)

    render_metrics(result, locked_compute_metrics)
