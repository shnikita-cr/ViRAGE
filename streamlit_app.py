from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import streamlit as st

from src.application.bootstrap import bootstrap_project_environment
from src.application.contracts import PipelineRequest
from src.application.pipeline import ViRAGEPipeline
from src.application.project_config import DEFAULT_CONFIG_PATH, load_project_config

bootstrap_project_environment()
st.set_page_config(page_title="ViRAGE", layout="wide")
st.title("ViRAGE — NL2VIS + RAG + Image-only Insights")

with st.sidebar:
    st.header("Project configuration")
    config_path = st.text_input("Config path", value=str(DEFAULT_CONFIG_PATH))
    st.caption("Copy config/project.example.toml to config/project.toml and edit model roles there.")
    st.markdown("---")
    st.caption("Metrics in Streamlit are computed only if enabled in the config.")

uploaded_file = st.file_uploader("Upload a table", type=["csv", "xlsx"])
query = st.text_area("Request", height=120, placeholder="Например: Покажи тренд продаж по датам и дай основные инсайты")
run_clicked = st.button("Run pipeline", type="primary")

if run_clicked:
    if uploaded_file is None:
        st.error("Upload a dataset first.")
        st.stop()
    if not query.strip():
        st.error("Enter a query first.")
        st.stop()

    try:
        project_config = load_project_config(config_path)
    except Exception as exc:
        st.exception(exc)
        st.stop()

    try:
        pipeline = ViRAGEPipeline.from_project_config(project_config)
    except Exception as exc:
        st.exception(exc)
        st.stop()

    suffix = Path(uploaded_file.name).suffix or ".csv"
    temp_dir = Path(tempfile.mkdtemp(prefix="virage_streamlit_"))
    data_path = temp_dir / f"uploaded{suffix}"
    data_path.write_bytes(uploaded_file.read())

    try:
        result = pipeline.invoke(PipelineRequest(query=query, data_path=data_path.as_posix()))
    except Exception as exc:
        st.exception(exc)
        shutil.rmtree(temp_dir, ignore_errors=True)
        st.stop()

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Rendered chart")
        image_path = None
        if result.plot_image:
            image_path = result.plot_image.get("image_path") if isinstance(result.plot_image, dict) else result.plot_image.image_path
        if image_path:
            st.image(image_path, use_container_width=True)
        else:
            st.warning("No plot image was produced.")

        st.subheader("Insights")
        if result.insights and result.insights.final_insights:
            for item in result.insights.final_insights:
                st.markdown(f"- {item}")
        else:
            st.info("No final insights were produced.")

    with col2:
        st.subheader("Intermediate steps")
        for log in result.step_logs:
            with st.expander(f"{log.stage}: {log.title}", expanded=False):
                st.write(log.summary)
                if log.inputs:
                    st.caption("Inputs")
                    st.write(log.inputs)
                if log.outputs:
                    st.caption("Outputs")
                    st.write(log.outputs)

        st.subheader("Key artifacts")
        if result.query_understanding:
            st.json(result.query_understanding.model_dump())
        if result.request_analysis:
            st.json(result.request_analysis.model_dump())
        if result.candidate_spec_set:
            st.json(result.candidate_spec_set.model_dump())
        if result.spec_validation:
            st.json(result.spec_validation.model_dump())

        if project_config.streamlit.compute_metrics:
            st.subheader("Metrics")
            if result.structural_spec_metric:
                st.json(result.structural_spec_metric.model_dump())
            if result.visual_quality_metric:
                st.json(result.visual_quality_metric.model_dump())
            if result.evaluation_summary:
                st.json(result.evaluation_summary.model_dump())

    shutil.rmtree(temp_dir, ignore_errors=True)
