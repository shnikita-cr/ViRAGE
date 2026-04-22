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
st.set_page_config(page_title='ViRAGE', layout='wide')
st.title('ViRAGE — NL2VIS + RAG + Image-only Insights')

with st.sidebar:
    st.header('Project configuration')
    config_path = st.text_input('Config path', value=str(DEFAULT_CONFIG_PATH))
    st.caption('Copy config/project.example.toml to config/project.toml and edit model roles there.')
    st.markdown('---')
    st.caption('Metrics in Streamlit are computed only if enabled in the config.')

uploaded_file = st.file_uploader('Upload a table', type=['csv', 'xlsx'])
query = st.text_area('Request', height=120, placeholder='Например: Покажи тренд продаж по датам и дай основные инсайты')
run_clicked = st.button('Run pipeline', type='primary')

if run_clicked:
    if uploaded_file is None:
        st.error('Upload a dataset first.')
        st.stop()
    if not query.strip():
        st.error('Enter a query first.')
        st.stop()

    try:
        project_config = load_project_config(config_path)
        pipeline = ViRAGEPipeline.from_project_config(project_config)
    except Exception as exc:
        st.exception(exc)
        st.stop()

    suffix = Path(uploaded_file.name).suffix or '.csv'
    temp_dir = Path(tempfile.mkdtemp(prefix='virage_streamlit_'))
    data_path = temp_dir / f'uploaded{suffix}'
    data_path.write_bytes(uploaded_file.read())

    try:
        result = pipeline.invoke(PipelineRequest(query=query, data_path=data_path.as_posix()))
    except Exception as exc:
        st.exception(exc)
        shutil.rmtree(temp_dir, ignore_errors=True)
        st.stop()

    top_left, top_right = st.columns([1.2, 1])
    with top_left:
        st.subheader('Rendered chart')
        image_path = None
        if result.plot_image:
            image_path = result.plot_image.get('image_path') if isinstance(result.plot_image, dict) else result.plot_image.image_path
        if image_path:
            st.image(image_path, use_container_width=True)
        else:
            st.warning('No plot image was produced.')

        st.subheader('Insights')
        if result.insights and result.insights.final_insights:
            for item in result.insights.final_insights:
                st.markdown(f'- {item}')
        else:
            st.info('No final insights were produced.')

    with top_right:
        st.subheader('Token usage summary')
        st.json(result.token_usage_summary.model_dump())
        st.subheader('Model call log')
        for idx, call in enumerate(result.model_call_logs, start=1):
            with st.expander(f"{idx}. {call.stage} | {call.model_role} | {call.model_name}", expanded=False):
                st.caption('Token usage')
                st.json(call.token_usage.model_dump())
                st.caption('Prompt')
                st.code(call.prompt)
                st.caption('Raw response')
                st.code(call.raw_response)
                if call.parsed_preview is not None:
                    st.caption('Parsed preview')
                    st.json(call.parsed_preview)
                if call.parser_errors:
                    st.caption('Parser errors')
                    st.write(call.parser_errors)

    st.subheader('Intermediate thinking-like steps')
    for log in result.step_logs:
        with st.expander(f"{log.stage}: {log.title}", expanded=False):
            st.write(log.summary)
            if log.inputs:
                st.caption('Inputs')
                st.write(log.inputs)
            if log.outputs:
                st.caption('Outputs')
                st.write(log.outputs)
            if log.details:
                st.caption('Details')
                st.json(log.details)

    tabs = st.tabs(['Full result', 'Artifacts', 'Metrics'])
    with tabs[0]:
        st.json(result.model_dump())
    with tabs[1]:
        for name in ['query_understanding', 'request_analysis', 'candidate_spec_set', 'vega_spec', 'spec_validation', 'plot_rendering', 'vlm_analysis', 'visual_facts', 'insight_reasoning', 'insight_verification']:
            value = getattr(result, name, None)
            st.markdown(f'### {name}')
            if value is None:
                st.write('None')
            elif hasattr(value, 'model_dump'):
                st.json(value.model_dump())
            else:
                st.json(value)
    with tabs[2]:
        if project_config.streamlit.compute_metrics:
            if result.structural_spec_metric:
                st.markdown('### structural_spec_metric')
                st.json(result.structural_spec_metric.model_dump())
            if result.visual_quality_metric:
                st.markdown('### visual_quality_metric')
                st.json(result.visual_quality_metric.model_dump())
            if result.evaluation_summary:
                st.markdown('### evaluation_summary')
                st.json(result.evaluation_summary.model_dump())
        else:
            st.info('Metric calculation is disabled by config for Streamlit mode.')

    shutil.rmtree(temp_dir, ignore_errors=True)
