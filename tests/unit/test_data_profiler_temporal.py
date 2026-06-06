from __future__ import annotations

import pandas as pd

from src.application.config.settings import ViRAGESettings
from src.infrastructure.runtime import RuntimeContext
from src.services.data.profile.data_profiler import DataProfilerService


def test_numeric_year_column_is_temporal(tmp_path):
    data_path = tmp_path / "cars.csv"
    pd.DataFrame({"Year": [70, 73, 82], "Weight": [1000, 1200, 900]}).to_csv(data_path, index=False)
    runtime = RuntimeContext(settings=ViRAGESettings(artifact_root=tmp_path / "artifacts"))
    runtime.current_run_id = "test"

    profile = DataProfilerService().invoke(str(data_path), runtime)

    year_profile = next(column for column in profile.columns if column.name == "Year")
    assert year_profile.role == "temporal"
    assert year_profile.dtype == "datetime"
    assert year_profile.min_value.startswith("1970")
    assert year_profile.max_value.startswith("1982")
