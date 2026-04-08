from __future__ import annotations

import pandas as pd

from src.services.data_profiler import DataProfilerService


def test_data_profiler_service_detects_basic_column_types(tmp_path, runtime) -> None:
    data_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "sales": [10, 12, 14],
            "region": ["A", "B", "C"],
        }
    ).to_csv(data_path, index=False)
    service = DataProfilerService()

    result = service.invoke(data_path=data_path.as_posix(), runtime=runtime)

    assert result.row_count == 3
    assert "sales" in result.likely_numeric_columns
    assert "date" in result.likely_time_columns
    assert "region" in result.likely_categorical_columns
