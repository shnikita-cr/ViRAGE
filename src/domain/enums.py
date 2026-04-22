from enum import Enum


class ChartCaseType(str, Enum):
    CANONICAL = "canonical"
    NON_CANONICAL = "non_canonical"


class PipelineStage(str, Enum):
    INITIALIZED = "initialized"
    QUERY_UNDERSTANDING = "query_understanding"
    REQUEST_ANALYSIS = "request_analysis"
    PLANNING = "planning"
    DATA_PROFILING = "data_profiling"
    DATA_PREPARATION = "data_preparation"
    VISRAG = "visrag"
    CANDIDATE_SPEC_SELECTION = "candidate_spec_selection"
    CHART_GENERATION = "chart_generation"
    SPEC_VALIDATION = "spec_validation"
    SCENEGRAPH_CHECK = "scenegraph_check"
    EMPTY_CHART_CHECK = "empty_chart_check"
    PLOT_RENDERING = "plot_rendering"
    VLM_ANALYSIS = "vlm_analysis"
    EVALUATION = "evaluation"
    CODEGEN = "codegen"
    CODERUN = "coderun"
    ARTIFACT_STORE = "artifact_store"
    CHART_READING = "chart_reading"
    FACT_EXTRACTION = "fact_extraction"
    REASONING = "reasoning"
    VERIFICATION = "verification"
    COMPLETED = "completed"
    FAILED = "failed"


class ArtifactType(str, Enum):
    CLEAN_DATA = "clean_data"
    GENERATED_CODE = "generated_code"
    PLOT = "plot"
    METRICS = "metrics"
    CHART_METADATA = "chart_metadata"
    MANIFEST = "manifest"
