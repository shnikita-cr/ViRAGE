from enum import Enum


class PipelineStage(str, Enum):
    INITIALIZED = "initialized"
    QUERY_UNDERSTANDING = "query_understanding"
    DATA_PROFILING = "data_profiling"
    REQUEST_ANALYSIS = "request_analysis"
    DATA_PREPARATION = "data_preparation"
    VISRAG = "visrag"
    CHART_GENERATION = "chart_generation"
    SPEC_VALIDATION = "spec_validation"
    PLOT_RENDERING = "plot_rendering"
    SCENEGRAPH_CHECK = "scenegraph_check"
    EMPTY_CHART_CHECK = "empty_chart_check"
    VLM_ANALYSIS = "vlm_analysis"
    FACT_EXTRACTION = "fact_extraction"
    REASONING = "reasoning"
    VERIFICATION = "verification"
    INSIGHTS = "insights"
    EVALUATION = "evaluation"
    COMPLETED = "completed"
    FAILED = "failed"


class ArtifactType(str, Enum):
    CLEAN_DATA = "clean_data"
    PLOT = "plot"
    METRICS = "metrics"
    MANIFEST = "manifest"
    JSON = "json"
    TEXT = "text"
