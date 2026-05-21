from __future__ import annotations

import pandas as pd

from src.application.settings import ViRAGESettings
from src.infrastructure.runtime import RuntimeContext
from src.services.data_profiler import DataProfilerService


def test_boolean_column_is_profiled_as_dimension_without_outlier_quantile_error(tmp_path):
    data_path = tmp_path / "flags.csv"
    pd.DataFrame(
        {
            "IsActive": [True, False, True, True, False, True, False, True],
            "Sales": [10, 12, 14, 16, 18, 20, 22, 24],
        }
    ).to_csv(data_path, index=False)

    runtime = RuntimeContext(settings=ViRAGESettings(artifact_root=tmp_path / "artifacts"))
    runtime.current_run_id = "test_boolean"
    runtime.reset_artifact_indices(run_id="test_boolean")

    profile = DataProfilerService().invoke(str(data_path), runtime)

    flag = next(column for column in profile.columns if column.name == "IsActive")

    assert profile.profile_status == "ok"
    assert profile.errors == []
    assert flag.dtype == "boolean"
    assert flag.outlier_count == 0
    assert flag.outlier_ratio == 0.0
    assert flag.role == "dimension"
    assert flag.role == "dimension"
    sales = next(column for column in profile.columns if column.name == "Sales")
    assert sales.role == "measure"


def test_boolean_like_strings_are_profiled_as_boolean_dimension(tmp_path):
    data_path = tmp_path / "flags.csv"
    pd.DataFrame(
        {
            "Flag": ["true", "false", "true", "false"],
            "Value": [1, 2, 3, 4],
        }
    ).to_csv(data_path, index=False)

    runtime = RuntimeContext(settings=ViRAGESettings(artifact_root=tmp_path / "artifacts"))
    runtime.current_run_id = "test_boolean_strings"
    runtime.reset_artifact_indices(run_id="test_boolean_strings")

    profile = DataProfilerService().invoke(str(data_path), runtime)
    flag = next(column for column in profile.columns if column.name == "Flag")

    assert flag.dtype == "boolean"
    assert flag.role == "dimension"
    assert flag.role == "dimension"


def test_column_level_profile_error_degrades_without_failing_pipeline(tmp_path, monkeypatch):
    data_path = tmp_path / "bad_column.csv"
    pd.DataFrame(
        {
            "Broken": ["a", "b", "c"],
            "Value": [1, 2, 3],
        }
    ).to_csv(data_path, index=False)

    service = DataProfilerService()
    original_semantic_dtype = service._semantic_dtype

    def fail_for_broken(column, series):
        if column == "Broken":
            raise ValueError("synthetic profiling failure")
        return original_semantic_dtype(column, series)

    monkeypatch.setattr(service, "_semantic_dtype", fail_for_broken)

    runtime = RuntimeContext(settings=ViRAGESettings(artifact_root=tmp_path / "artifacts"))
    runtime.current_run_id = "test_degraded"
    runtime.reset_artifact_indices(run_id="test_degraded")

    profile = service.invoke(str(data_path), runtime)

    broken = next(column for column in profile.columns if column.name == "Broken")

    assert profile.profile_status == "degraded"
    assert len(profile.errors) == 1
    assert profile.errors[0]["failure_class"] == "data_profile_error"
    assert profile.errors[0]["recoverable"] is True
    assert broken.dtype == "categorical"
    assert broken.role == "dimension"
    value = next(column for column in profile.columns if column.name == "Value")
    assert value.role == "measure"

    error_artifacts = list((tmp_path / "artifacts" / "test_degraded" / "artifacts").glob("*_data_profile_*error*.json"))
    assert not error_artifacts


def test_data_profiler_preserves_detected_csv_encoding(tmp_path):
    data_path = tmp_path / "cp1251.csv"
    data_path.write_bytes("Метод,Значение\nА,1\nБ,2\n".encode("cp1251"))

    runtime = RuntimeContext(settings=ViRAGESettings(artifact_root=tmp_path / "artifacts"))
    runtime.current_run_id = "test_encoding"
    runtime.reset_artifact_indices(run_id="test_encoding")

    profile = DataProfilerService().invoke(str(data_path), runtime)

    assert profile.source_format == "csv"
    assert profile.source_encoding == "cp1251"
    assert profile.row_count == 2
