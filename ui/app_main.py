from __future__ import annotations

import shutil
import tempfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

import streamlit as st

from src.application.config.bootstrap import bootstrap_project_environment
from src.application.config.project_config import ProjectConfig, load_project_config
from src.application.contracts import PipelineRequest
from src.application.pipeline import ViRAGEPipeline
from src.domain.models import ModelCallLog, StepLog
from src.infrastructure.runtime import RuntimeContext
from src.orchestrator.analysis_planner import AnalysisPlanner
from src.orchestrator.image_folder_preprocessor import IMAGE_EXTENSIONS, ImageFolderPreprocessor
from src.services.data.profile.data_profiler import DataProfilerService
from ui.app_components import (
    CHART_MODE_OPTIONS,
    CONFIG_DIR,
    DEFAULT_CONFIG_PATH,
    METRICS_DISABLED,
    METRICS_ENABLED,
    METRICS_OPTIONS,
    RUN_MODE_OPTIONS,
    RUN_MODE_ORCHESTRATOR,
    RUN_MODE_SINGLE_PIPELINE,
    apply_streamlit_run_overrides,
    build_pending_run_payload,
    config_label,
    discover_config_files,
    init_session_state,
    path_from_config_label,
    read_table_preview_from_bytes,
    read_table_preview_from_path,
    render_chart,
    render_insight_summary,
    render_metrics,
    render_model_calls,
    render_spec_generation_validation_details,
    render_table_preview,
    render_token_usage_cards,
    resolve_project_path,
    run_setting_defaults_from_config,
    run_setting_defaults_from_pending,
)

RecoverableUiError = (RuntimeError, ValueError, TypeError, OSError, KeyError, IndexError, AttributeError, ImportError)

SHOW_MODEL_CALLS_IN_UI = False
INPUT_MODE_TABLE = "Таблица CSV/XLSX"
INPUT_MODE_IMAGE_FILES = "Изображения"
INPUT_MODE_IMAGE_DIRECTORY = "Папка изображений"
INPUT_MODE_OPTIONS = [INPUT_MODE_TABLE, INPUT_MODE_IMAGE_FILES, INPUT_MODE_IMAGE_DIRECTORY]
IMAGE_UPLOAD_TYPES = sorted(extension.lstrip(".") for extension in IMAGE_EXTENSIONS)



def run_app() -> None:
    bootstrap_project_environment()
    _configure_page()
    init_session_state()
    config_files = _load_config_files()
    pending_run = st.session_state.pending_run
    controls_disabled = bool(st.session_state.pipeline_running)
    controls = _render_sidebar_controls(config_files, pending_run, controls_disabled)
    input_payload, query, run_clicked = _render_input_controls(controls_disabled)
    if run_clicked:
        _lock_pending_run(controls, input_payload, query)
    if not pending_run:
        _render_previous_orchestrator_result()
        st.stop()
    locked = _locked_run_context(pending_run, config_files)
    data_path, temp_dir = _resolve_data_path(locked)
    _render_input_preview(data_path, locked)
    outcome = _execute_run(locked, data_path, temp_dir)
    _store_completed_run(outcome, locked)
    _render_completed_outcome(outcome, locked)


def _configure_page() -> None:
    st.set_page_config(page_title="ViRAGE Orchestrator", layout="wide")
    st.title("ViRAGE Orchestrator")
    st.caption("Один запрос проекта → до трёх аналитических подзадач → графики, спецификации и проверяемые результаты по мере готовности.")


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


def _selected_indices(config_files: list[Path], pending_run: dict[str, Any] | None) -> tuple[int, int, int, int]:
    if not pending_run:
        return _default_config_index(config_files), 0, 0, 0
    labels = [config_label(path) for path in config_files]
    return (
        labels.index(pending_run["config_label"]),
        RUN_MODE_OPTIONS.index(pending_run.get("run_mode", RUN_MODE_ORCHESTRATOR)),
        CHART_MODE_OPTIONS.index(pending_run["chart_mode"]),
        0 if pending_run["compute_metrics"] else 1,
    )


def _render_sidebar_controls(
    config_files: list[Path],
    pending_run: dict[str, Any] | None,
    controls_disabled: bool,
) -> dict[str, Any]:
    labels = [config_label(path) for path in config_files]
    config_index, run_mode_index, chart_index, metrics_index = _selected_indices(config_files, pending_run)
    selected_settings = run_setting_defaults_from_pending(pending_run)
    with st.sidebar:
        st.header("Project run")
        selected_label = st.radio("Configuration file", labels, index=config_index, disabled=controls_disabled)
        selected_path = path_from_config_label(selected_label, config_files)
        selected_settings = _settings_for_sidebar(selected_path, pending_run, selected_settings)
        controls = _render_run_controls(selected_label, run_mode_index, chart_index, metrics_index, selected_settings, controls_disabled)
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
        return run_setting_defaults_from_config(load_project_config(selected_config_path))
    except RecoverableUiError:
        return selected_settings


def _render_run_controls(
    selected_config_label: str,
    selected_run_mode_index: int,
    selected_chart_index: int,
    selected_metrics_index: int,
    settings: dict[str, Any],
    controls_disabled: bool,
) -> dict[str, Any]:
    run_mode = st.radio("Run target", RUN_MODE_OPTIONS, index=selected_run_mode_index, disabled=controls_disabled)
    chart_mode = st.radio("Chart output", CHART_MODE_OPTIONS, index=selected_chart_index, disabled=controls_disabled)
    metrics_mode = st.radio("Compute metrics", METRICS_OPTIONS, index=selected_metrics_index, disabled=controls_disabled)
    semantic_loop = st.checkbox(
        "Enable semantic VLM loop",
        value=bool(settings["semantic_feedback_loop_enabled"]),
        disabled=controls_disabled,
    )
    return {
        "selected_config_label": selected_config_label,
        "chart_mode": chart_mode,
        "compute_metrics": metrics_mode == METRICS_ENABLED,
        "run_mode": run_mode,
        "max_charts": _render_max_charts_control(run_mode, controls_disabled),
        "visrag_enabled": st.checkbox("Enable RAG / VisRAG context", value=bool(settings["visrag_enabled"]), disabled=controls_disabled),
        "analytics_tail_enabled": st.checkbox("Enable analytics tail", value=bool(settings["analytics_tail_enabled"]), disabled=controls_disabled),
        "spec_generation_max_attempts": st.slider("Spec generation attempts", 1, 20, max(1, min(20, int(settings["spec_generation_max_attempts"]))), 1, disabled=controls_disabled),
        "semantic_feedback_loop_enabled": semantic_loop,
        "semantic_feedback_max_attempts": st.slider("Semantic VLM attempts", 1, 20, max(1, min(20, int(settings["semantic_feedback_max_attempts"]))), 1, disabled=controls_disabled or not semantic_loop),
        "semantic_feedback_min_accept_confidence": st.slider("Semantic accept confidence", 0.0, 1.0, max(0.0, min(1.0, float(settings["semantic_feedback_min_accept_confidence"]))), 0.05, disabled=controls_disabled or not semantic_loop),
        "semantic_feedback_save_rejected_specs": st.checkbox("Save rejected specs to feedback corpus", value=bool(settings["semantic_feedback_save_rejected_specs"]), disabled=controls_disabled or not semantic_loop),
    }


def _render_max_charts_control(run_mode: str, controls_disabled: bool) -> int:
    if run_mode == RUN_MODE_SINGLE_PIPELINE:
        st.caption("Single pipeline mode runs exactly one chart request without orchestration.")
        return 1
    return st.slider("Maximum subtasks", 1, 3, 3, 1, disabled=controls_disabled)


def _render_locked_settings_notice(controls_disabled: bool, pending_run: dict[str, Any] | None) -> None:
    st.markdown("---")
    if controls_disabled and pending_run:
        st.info(_locked_settings_text(pending_run))
        return
    st.caption("Settings are locked after pressing Run.")


def _locked_settings_text(pending_run: dict[str, Any]) -> str:
    metrics = METRICS_ENABLED if pending_run["compute_metrics"] else METRICS_DISABLED
    return (
        "Run settings are locked:\n\n"
        f"- mode: `{pending_run.get('run_mode', RUN_MODE_ORCHESTRATOR)}`\n"
        f"- `{pending_run['config_label']}`\n"
        f"- subtasks: `{pending_run.get('max_charts', 3)}`\n"
        f"- metrics: `{metrics}`\n"
        f"- RAG enabled: `{pending_run.get('visrag_enabled', True)}`\n"
        f"- spec attempts: `{pending_run.get('spec_generation_max_attempts', 3)}`\n"
        f"- semantic attempts: `{pending_run.get('semantic_feedback_max_attempts', 2)}`"
    )


def _render_input_controls(controls_disabled: bool) -> tuple[dict[str, Any] | None, str, bool]:
    input_mode = st.radio("Input data", INPUT_MODE_OPTIONS, horizontal=True, disabled=controls_disabled)
    input_payload = _render_dataset_uploader(input_mode, controls_disabled)
    query = st.text_area(
        "Project task",
        height=140,
        placeholder="Например: Проанализируй качество изображений, покажи основные проблемные случаи и сравни методы.",
        disabled=controls_disabled,
    )
    run_clicked = st.button("Run", type="primary", disabled=controls_disabled)
    return input_payload, query, run_clicked


def _render_dataset_uploader(input_mode: str, controls_disabled: bool) -> dict[str, Any] | None:
    if input_mode == INPUT_MODE_TABLE:
        uploaded_file = st.file_uploader("Upload a table", type=["csv", "xlsx"], disabled=controls_disabled)
        if uploaded_file is not None and not controls_disabled:
            _render_uploaded_preview(uploaded_file)
            return {
                "input_type": "table",
                "uploaded_file_name": uploaded_file.name,
                "uploaded_file_bytes": uploaded_file.getvalue(),
            }
        return None

    if input_mode == INPUT_MODE_IMAGE_FILES:
        files = st.file_uploader(
            "Drag and drop images",
            type=IMAGE_UPLOAD_TYPES,
            accept_multiple_files=True,
            disabled=controls_disabled,
        )
        return _image_upload_payload(files or [], input_mode=input_mode, controls_disabled=controls_disabled)

    files = st.file_uploader(
        "Drag and drop an image folder",
        type=IMAGE_UPLOAD_TYPES,
        accept_multiple_files="directory",
        disabled=controls_disabled,
    )
    return _image_upload_payload(files or [], input_mode=input_mode, controls_disabled=controls_disabled)


def _render_uploaded_preview(uploaded_file: Any) -> None:
    try:
        preview_df = read_table_preview_from_bytes(uploaded_file.name, uploaded_file.getvalue())
        with st.expander("Table preview", expanded=True):
            render_table_preview("Uploaded table preview", preview_df)
    except RecoverableUiError as exc:
        st.warning(f"Could not preview the uploaded table: {exc}")


def _image_upload_payload(files: list[Any], *, input_mode: str, controls_disabled: bool) -> dict[str, Any] | None:
    if not files:
        return None
    images = [{"name": file.name, "bytes": file.getvalue()} for file in files]
    if not controls_disabled:
        _render_uploaded_images_preview(images, input_mode=input_mode)
    return {
        "input_type": "image_folder",
        "image_upload_mode": input_mode,
        "uploaded_images": images,
    }


def _render_uploaded_images_preview(images: list[dict[str, Any]], *, input_mode: str) -> None:
    with st.expander("Image input preview", expanded=True):
        st.caption(f"{input_mode}: {len(images)} image file(s) selected.")
        st.dataframe(
            [
                {
                    "idx": index,
                    "name": item["name"],
                    "size_kb": round(len(item["bytes"]) / 1024, 1),
                }
                for index, item in enumerate(images, start=1)
            ],
            width="stretch",
            hide_index=True,
        )
        preview_columns = st.columns(min(4, len(images)))
        for column, item in zip(preview_columns, images[:4]):
            column.image(item["bytes"], caption=Path(item["name"]).name, width="stretch")


def _lock_pending_run(controls: dict[str, Any], input_payload: dict[str, Any] | None, query: str) -> None:
    if input_payload is None:
        st.error("Upload a dataset first.")
        st.stop()
    if not query.strip():
        st.error("Enter a project task first.")
        st.stop()
    st.session_state.pending_run = build_pending_run_payload(input_payload=input_payload, query=query, **controls)
    st.session_state.pipeline_running = True
    st.rerun()


def _locked_run_context(pending_run: dict[str, Any], config_files: list[Path]) -> dict[str, Any]:
    return {
        "config_label": pending_run["config_label"],
        "config_path": path_from_config_label(pending_run["config_label"], config_files),
        "chart_mode": pending_run["chart_mode"],
        "compute_metrics": bool(pending_run["compute_metrics"]),
        "run_mode": pending_run.get("run_mode", RUN_MODE_ORCHESTRATOR),
        "max_charts": max(1, min(3, int(pending_run.get("max_charts", 3)))),
        "visrag_enabled": bool(pending_run.get("visrag_enabled", True)),
        "analytics_tail_enabled": bool(pending_run.get("analytics_tail_enabled", True)),
        "spec_generation_max_attempts": int(pending_run.get("spec_generation_max_attempts", 3)),
        "semantic_feedback_loop_enabled": bool(pending_run.get("semantic_feedback_loop_enabled", False)),
        "semantic_feedback_max_attempts": int(pending_run.get("semantic_feedback_max_attempts", 2)),
        "semantic_feedback_min_accept_confidence": float(pending_run.get("semantic_feedback_min_accept_confidence", 0.75)),
        "semantic_feedback_save_rejected_specs": bool(pending_run.get("semantic_feedback_save_rejected_specs", True)),
        "query": pending_run["query"],
        "input_type": pending_run.get("input_type", "table"),
        "uploaded_file_name": pending_run.get("uploaded_file_name", "uploaded.csv"),
        "uploaded_file_bytes": pending_run.get("uploaded_file_bytes"),
        "uploaded_images": pending_run.get("uploaded_images", []),
        "image_upload_mode": pending_run.get("image_upload_mode"),
    }


def _override_values(locked: dict[str, Any]) -> dict[str, Any]:
    return {
        key: locked[key]
        for key in [
            "compute_metrics",
            "visrag_enabled",
            "analytics_tail_enabled",
            "spec_generation_max_attempts",
            "semantic_feedback_loop_enabled",
            "semantic_feedback_max_attempts",
            "semantic_feedback_min_accept_confidence",
            "semantic_feedback_save_rejected_specs",
        ]
    }


def _resolve_data_path(locked: dict[str, Any]) -> tuple[Path, Path | None]:
    temp_dir = Path(tempfile.mkdtemp(prefix="virage_orchestrator_"))
    if locked.get("input_type") == "image_folder":
        return _prepare_uploaded_images(locked, temp_dir), temp_dir

    suffix = Path(locked["uploaded_file_name"]).suffix or ".csv"
    data_path = temp_dir / f"uploaded{suffix}"
    data_path.write_bytes(locked["uploaded_file_bytes"])
    return data_path, temp_dir


def _prepare_uploaded_images(locked: dict[str, Any], temp_dir: Path) -> Path:
    image_dir = temp_dir / "uploaded_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    images = list(locked.get("uploaded_images") or [])
    if not images:
        raise ValueError("Image input mode requires at least one uploaded image.")

    for index, item in enumerate(images, start=1):
        relative_path = _safe_uploaded_image_relative_path(str(item.get("name") or ""), index=index)
        output_path = _unique_uploaded_image_path(image_dir / relative_path, index=index)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(bytes(item.get("bytes") or b""))

    preprocessing_dir = temp_dir / "image_preprocessing"
    preprocessing = ImageFolderPreprocessor().process(input_dir=image_dir, output_dir=preprocessing_dir)
    if preprocessing.processed_images < 1:
        raise ValueError("No uploaded images could be processed.")

    locked["original_input_path"] = image_dir.as_posix()
    locked["preprocessed_data_path"] = preprocessing.output_csv_path
    locked["failed_images_path"] = preprocessing.failed_csv_path
    locked["preprocessing_report_path"] = preprocessing.report_path
    locked["found_images"] = preprocessing.found_images
    locked["processed_images"] = preprocessing.processed_images
    locked["failed_images"] = preprocessing.failed_images
    locked["feature_columns"] = list(preprocessing.feature_columns)
    return Path(preprocessing.output_csv_path)


def _safe_uploaded_image_relative_path(file_name: str, *, index: int) -> Path:
    normalized = file_name.replace("\\", "/").strip()
    raw_path = PurePosixPath(normalized) if normalized else PurePosixPath(f"image_{index:04d}.png")
    if raw_path.is_absolute() or ".." in raw_path.parts:
        raise ValueError(f"Unsafe uploaded image path: {file_name}")

    safe_parts = [_safe_path_component(part) for part in raw_path.parts if part not in {"", "."}]
    if not safe_parts:
        safe_parts = [f"image_{index:04d}.png"]

    relative_path = Path(*safe_parts)
    suffix = relative_path.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {file_name}")
    return relative_path


def _safe_path_component(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {".", "_", "-", " "} else "_" for char in value).strip()
    if cleaned in {"", ".", ".."}:
        raise ValueError(f"Unsafe uploaded image path component: {value}")
    return cleaned


def _unique_uploaded_image_path(path: Path, *, index: int) -> Path:
    if not path.exists():
        return path
    return path.with_name(f"{path.stem}_{index:04d}{path.suffix}")


def _render_input_preview(data_path: Path, locked: dict[str, Any]) -> None:
    try:
        with st.expander("Input table preview", expanded=False):
            if locked.get("input_type") == "image_folder":
                st.caption(
                    "Uploaded images were converted into an image-quality metrics table. "
                    f"Processed: {locked.get('processed_images', 0)} / {locked.get('found_images', 0)}; "
                    f"failed: {locked.get('failed_images', 0)}."
                )
            render_table_preview("Input table preview", read_table_preview_from_path(data_path))
    except RecoverableUiError as exc:
        st.warning(f"Could not preview input table: {exc}")


def _run_user_context(locked: dict[str, Any], *, source: str) -> dict[str, Any]:
    context: dict[str, Any] = {
        "source": source,
        "input_type": locked.get("input_type", "table"),
    }
    if locked.get("input_type") == "image_folder":
        context.update(
            {
                "image_upload_mode": locked.get("image_upload_mode"),
                "original_input_path": locked.get("original_input_path"),
                "preprocessed_data_path": locked.get("preprocessed_data_path"),
                "failed_images_path": locked.get("failed_images_path"),
                "preprocessing_report_path": locked.get("preprocessing_report_path"),
                "found_images": locked.get("found_images"),
                "processed_images": locked.get("processed_images"),
                "failed_images": locked.get("failed_images"),
                "feature_columns": locked.get("feature_columns", []),
            }
        )
    return context


def _execute_run(locked: dict[str, Any], data_path: Path, temp_dir: Path | None) -> dict[str, Any]:
    if locked["run_mode"] == RUN_MODE_SINGLE_PIPELINE:
        return _execute_single_pipeline(locked, data_path, temp_dir)
    return _execute_orchestrator(locked, data_path, temp_dir)


def _execute_single_pipeline(locked: dict[str, Any], data_path: Path, temp_dir: Path | None) -> dict[str, Any]:
    status_slot = st.empty()
    steps_slot = st.empty()
    model_calls_slot = st.empty()
    result_area = st.container()
    steps: list[StepLog] = []
    model_calls: list[ModelCallLog] = []
    try:
        config = _load_project_config(locked)
        pipeline = ViRAGEPipeline.from_project_config(config)
        run_id = _single_pipeline_run_id()
        status_slot.info("Running single ViRAGE pipeline...")
        result = pipeline.invoke(
            PipelineRequest(
                query=locked["query"],
                data_path=data_path.as_posix(),
                run_id=run_id,
                user_context=_run_user_context(locked, source="streamlit_single_pipeline"),
            ),
            step_callback=_single_step_callback(steps, steps_slot),
            model_call_callback=_single_call_callback(model_calls, model_calls_slot),
        )
        status_slot.success("Single pipeline completed.")
        _render_project_steps(steps_slot, steps or list(result.step_logs))
        if SHOW_MODEL_CALLS_IN_UI:
            _render_model_call_summary(model_calls_slot, model_calls or list(result.model_call_logs))
        _render_single_pipeline_result(result_area, result, locked, model_calls or list(result.model_call_logs))
        return {
            "mode": RUN_MODE_SINGLE_PIPELINE,
            "run_id": run_id,
            "result": result,
            "model_calls": model_calls or list(result.model_call_logs),
            "steps": steps or list(result.step_logs),
        }
    except RecoverableUiError as exc:
        st.session_state.pipeline_running = False
        st.session_state.pending_run = None
        st.exception(exc)
        st.stop()
    finally:
        _cleanup_temp_dir(temp_dir)


def _single_step_callback(steps: list[StepLog], slot: Any) -> Any:
    def on_step(step: StepLog) -> None:
        steps.append(step)
        _render_project_steps(slot, steps)
    return on_step


def _single_call_callback(model_calls: list[ModelCallLog], slot: Any) -> Any:
    def on_call(call: ModelCallLog) -> None:
        model_calls.append(call)
        if SHOW_MODEL_CALLS_IN_UI:
            _render_model_call_summary(slot, model_calls)
    return on_call


def _render_single_pipeline_result(area: Any, result: Any, locked: dict[str, Any], calls: list[ModelCallLog]) -> None:
    with area.container():
        st.markdown("## Single ViRAGE pipeline result")
        left, right = st.columns([1.25, 1])
        with left:
            render_chart(result, locked["chart_mode"])
        with right:
            if SHOW_MODEL_CALLS_IN_UI:
                st.markdown("#### Model calls")
                render_model_calls(calls)
            st.markdown("#### Token usage")
            render_token_usage_cards(result.token_usage_summary)
        render_insight_summary(result, analytics_tail_enabled=locked["analytics_tail_enabled"])
        render_spec_generation_validation_details(result)
        render_metrics(result, locked["compute_metrics"])



def _execute_orchestrator(locked: dict[str, Any], data_path: Path, temp_dir: Path | None) -> dict[str, Any]:
    status_slot = st.empty()
    steps_slot = st.empty()
    aggregate_calls_slot = st.empty()
    result_area = st.container()
    steps: list[StepLog] = []
    aggregate_calls: list[ModelCallLog] = []
    results: list[dict[str, Any]] = []
    try:
        config = _load_project_config(locked)
        pipeline = ViRAGEPipeline.from_project_config(config)
        run_id = _orchestrator_run_id()
        status_slot.info("Profiling dataset and planning analytical subtasks...")
        plan = _plan_subtasks(pipeline, locked, data_path, run_id, aggregate_calls)
        _append_project_step(steps, "planning", "Analytical plan built", f"{len(plan.subtasks)} subtask(s) selected.")
        _render_project_steps(steps_slot, steps)
        for index, subtask in enumerate(plan.subtasks, start=1):
            result = _run_subtask(
                pipeline=pipeline,
                locked=locked,
                data_path=data_path,
                parent_run_id=run_id,
                index=index,
                subtask=subtask,
                steps=steps,
                aggregate_calls=aggregate_calls,
                steps_slot=steps_slot,
                aggregate_calls_slot=aggregate_calls_slot,
                result_area=result_area,
            )
            results.append(result)
        status_slot.success("Orchestrator completed.")
        if SHOW_MODEL_CALLS_IN_UI:
            _render_model_call_summary(aggregate_calls_slot, aggregate_calls)
        return {"mode": RUN_MODE_ORCHESTRATOR, "run_id": run_id, "plan": plan, "subtasks": results, "model_calls": aggregate_calls, "steps": steps}
    except RecoverableUiError as exc:
        st.session_state.pipeline_running = False
        st.session_state.pending_run = None
        st.exception(exc)
        st.stop()
    finally:
        _cleanup_temp_dir(temp_dir)


def _load_project_config(locked: dict[str, Any]) -> ProjectConfig:
    config = load_project_config(locked["config_path"])
    return apply_streamlit_run_overrides(project_config=config, **_override_values(locked))


def _plan_subtasks(
    pipeline: ViRAGEPipeline,
    locked: dict[str, Any],
    data_path: Path,
    run_id: str,
    aggregate_calls: list[ModelCallLog],
) -> Any:
    pipeline.runtime.current_run_id = run_id
    data_profile = DataProfilerService().invoke(data_path.as_posix(), pipeline.runtime)
    plan = AnalysisPlanner(max_charts=locked["max_charts"], reasoning_llm=pipeline.runtime.reasoning_llm).plan(
        user_query=locked["query"],
        data_path=data_path.as_posix(),
        data_profile=data_profile,
        runtime=pipeline.runtime,
        user_context=_run_user_context(locked, source="streamlit_orchestrator"),
        input_type=str(locked.get("input_type") or "table"),
        original_input_path=locked.get("original_input_path"),
        preprocessing_report_path=locked.get("preprocessing_report_path"),
    )
    aggregate_calls.extend(list(pipeline.runtime.model_call_logs))
    return plan


def _run_subtask(
    *,
    pipeline: ViRAGEPipeline,
    locked: dict[str, Any],
    data_path: Path,
    parent_run_id: str,
    index: int,
    subtask: Any,
    steps: list[StepLog],
    aggregate_calls: list[ModelCallLog],
    steps_slot: Any,
    aggregate_calls_slot: Any,
    result_area: Any,
) -> dict[str, Any]:
    local_steps: list[StepLog] = []
    local_calls: list[ModelCallLog] = []
    subrun_id = f"{parent_run_id}/subruns/{index:02d}_{_safe_slug(subtask.id)}"
    _append_project_step(steps, f"subtask_{index}", f"Started: {subtask.id}", subtask.purpose)
    _render_project_steps(steps_slot, steps)
    result = pipeline.invoke(
        PipelineRequest(
            query=subtask.query,
            data_path=data_path.as_posix(),
            run_id=subrun_id,
            user_context={
                **_run_user_context(locked, source="streamlit_orchestrator_subtask"),
                "orchestrator_parent_run_id": parent_run_id,
                "analysis_subtask": subtask.model_dump(),
                "original_user_query": locked["query"],
            },
        ),
        step_callback=_subtask_step_callback(index, steps, local_steps, steps_slot),
        model_call_callback=_subtask_call_callback(local_calls, aggregate_calls, aggregate_calls_slot),
    )
    _append_project_step(steps, f"subtask_{index}", f"Completed: {subtask.id}", "Chart and specification are available.")
    _render_project_steps(steps_slot, steps)
    _render_subtask_result(result_area, index, subtask, result, locked, local_calls)
    return {"subtask": subtask, "result": result, "model_calls": local_calls, "steps": local_steps}


def _subtask_step_callback(index: int, global_steps: list[StepLog], local_steps: list[StepLog], slot: Any) -> Any:
    def on_step(step: StepLog) -> None:
        local_steps.append(step)
        global_steps.append(step.model_copy(update={"stage": f"subtask_{index}.{step.stage}"}))
        _render_project_steps(slot, global_steps)
    return on_step


def _subtask_call_callback(local_calls: list[ModelCallLog], aggregate_calls: list[ModelCallLog], slot: Any) -> Any:
    def on_call(call: ModelCallLog) -> None:
        local_calls.append(call)
        aggregate_calls.append(call)
        if SHOW_MODEL_CALLS_IN_UI:
            _render_model_call_summary(slot, aggregate_calls)
    return on_call


def _render_subtask_result(area: Any, index: int, subtask: Any, result: Any, locked: dict[str, Any], calls: list[ModelCallLog]) -> None:
    with area.container():
        st.markdown(f"## Subtask {index}: {subtask.id}")
        st.caption(subtask.purpose)
        left, right = st.columns([1.25, 1])
        with left:
            render_chart(result, locked["chart_mode"])
        with right:
            if SHOW_MODEL_CALLS_IN_UI:
                st.markdown("#### Model calls")
                render_model_calls(calls)
            st.markdown("#### Token usage")
            render_token_usage_cards(result.token_usage_summary)
        render_insight_summary(result, analytics_tail_enabled=locked["analytics_tail_enabled"])
        render_spec_generation_validation_details(result)
        render_metrics(result, locked["compute_metrics"])
        st.divider()


def _render_project_steps(slot: Any, steps: list[StepLog]) -> None:
    with slot.container():
        st.subheader("Project steps")
        if not steps:
            st.info("No completed project steps yet.")
            return
        rows = [
            {
                "idx": index,
                "stage": step.stage,
                "title": step.title,
                "summary": step.summary,
                "duration_s": round(float(step.duration_seconds or 0.0), 2),
            }
            for index, step in enumerate(steps, start=1)
        ]
        st.dataframe(rows, width="stretch", hide_index=True)


def _render_model_call_summary(slot: Any, calls: list[ModelCallLog]) -> None:
    if not SHOW_MODEL_CALLS_IN_UI:
        slot.empty()
        return
    with slot.container():
        st.subheader("Total model calls")
        if not calls:
            st.info("No model calls completed yet.")
            return
        rows = []
        total_prompt = 0
        total_completion = 0
        total_duration = 0.0
        for index, call in enumerate(calls, start=1):
            usage = call.token_usage
            prompt_tokens = int(usage.prompt_tokens or 0)
            completion_tokens = int(usage.completion_tokens or 0)
            total_prompt += prompt_tokens
            total_completion += completion_tokens
            total_duration += float(call.duration_seconds or 0.0)
            rows.append({
                "idx": index,
                "stage": call.stage,
                "role": call.model_role,
                "model": call.model_name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": int(usage.total_tokens or prompt_tokens + completion_tokens),
                "duration_s": round(float(call.duration_seconds or 0.0), 2),
                "parser_errors": len(call.parser_errors),
            })
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Calls", len(calls))
        c2.metric("Input tokens", total_prompt)
        c3.metric("Output tokens", total_completion)
        c4.metric("Duration, s", round(total_duration, 2))
        st.dataframe(rows, width="stretch", hide_index=True)


def _append_project_step(steps: list[StepLog], stage: str, title: str, summary: str) -> None:
    steps.append(StepLog(stage=stage, title=title, summary=summary))


def _store_completed_run(outcome: dict[str, Any], locked: dict[str, Any]) -> None:
    st.session_state.pipeline_running = False
    st.session_state.pending_run = None
    st.session_state.last_orchestrator_results = outcome
    st.session_state.last_result = outcome.get("result")
    st.session_state.last_run_settings = {
        "config_path": locked["config_label"],
        "run_mode": locked["run_mode"],
        **_override_values(locked),
        "chart_mode": locked["chart_mode"],
        "input_type": locked.get("input_type", "table"),
        "original_input_path": locked.get("original_input_path"),
        "preprocessing_report_path": locked.get("preprocessing_report_path"),
    }


def _render_completed_outcome(outcome: dict[str, Any], locked: dict[str, Any]) -> None:
    if outcome.get("mode") == RUN_MODE_SINGLE_PIPELINE:
        _render_single_pipeline_outcome(outcome)
        return
    _render_orchestrator_outcome(outcome)


def _render_orchestrator_outcome(outcome: dict[str, Any]) -> None:
    st.success("Project task completed through orchestrator.")
    with st.expander("Analytical plan", expanded=False):
        st.json(outcome["plan"].model_dump())
    if SHOW_MODEL_CALLS_IN_UI:
        _render_model_call_summary(st.empty(), outcome["model_calls"])


def _render_single_pipeline_outcome(outcome: dict[str, Any]) -> None:
    st.success("Single ViRAGE pipeline completed.")
    _render_project_steps(st.empty(), outcome.get("steps", []))
    if SHOW_MODEL_CALLS_IN_UI:
        _render_model_call_summary(st.empty(), outcome.get("model_calls", []))


def _render_previous_orchestrator_result() -> None:
    outcome = st.session_state.get("last_orchestrator_results")
    if not outcome:
        return
    if outcome.get("mode") == RUN_MODE_SINGLE_PIPELINE:
        st.success("Last single pipeline run completed.")
        _render_project_steps(st.empty(), outcome.get("steps", []))
        if SHOW_MODEL_CALLS_IN_UI:
            _render_model_call_summary(st.empty(), outcome.get("model_calls", []))
        return
    st.success("Last orchestrator run completed.")
    with st.expander("Last analytical plan", expanded=False):
        st.json(outcome["plan"].model_dump())
    _render_project_steps(st.empty(), outcome.get("steps", []))
    if SHOW_MODEL_CALLS_IN_UI:
        _render_model_call_summary(st.empty(), outcome.get("model_calls", []))


def _orchestrator_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + "_orchestrator_" + uuid4().hex


def _single_pipeline_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + "_single_" + uuid4().hex


def _safe_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value).strip("_") or "subtask"


def _cleanup_temp_dir(temp_dir: Path | None) -> None:
    if temp_dir is not None:
        shutil.rmtree(temp_dir, ignore_errors=True)
