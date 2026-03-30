from enum import Enum


class ChartCaseType(str, Enum):
    CANONICAL = "canonical"
    NON_CANONICAL = "non_canonical"


class PipelineStage(str, Enum):
    INITIALIZED = "initialized"
    QUERY_UNDERSTANDING = "query_understanding"
    PLANNING = "planning"
    DATA_PROFILING = "data_profiling"
    DATA_PREPARATION = "data_preparation"
    VISRAG = "visrag"
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
