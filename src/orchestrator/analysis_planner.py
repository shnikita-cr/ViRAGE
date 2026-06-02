from __future__ import annotations

import re
from dataclasses import dataclass

from src.domain.models import DataColumnProfile, DataProfile
from src.orchestrator.models import AnalysisPlan, AnalysisSubtask, AnalysisTaskType, SkippedAnalysisCandidate

_GENERAL_QUERY_RE = re.compile(
    r"\b(analy[sz]e|analysis|overview|summary|explore|insight|patterns?)\b|"
    r"(проанализ|анализ|обзор|закономерност|исследуй|покажи основн|выяви)",
    re.IGNORECASE,
)
_DISTRIBUTION_RE = re.compile(r"distribution|histogram|density|распредел|гистограмм", re.IGNORECASE)
_GROUP_RE = re.compile(r"group|category|compare|comparison|различ|сравн|категор|групп", re.IGNORECASE)
_CORRELATION_RE = re.compile(r"correlation|relationship|scatter|связ|коррел|зависим", re.IGNORECASE)
_TEMPORAL_RE = re.compile(r"trend|time|date|temporal|динамик|тренд|времен", re.IGNORECASE)
_OUTLIER_RE = re.compile(r"outlier|anomal|выброс|аномал", re.IGNORECASE)
_MISSING_RE = re.compile(r"missing|null|пропуск|пуст", re.IGNORECASE)
_RANKING_RE = re.compile(r"ranking|top|bottom|rank|рейтинг|топ|лидер", re.IGNORECASE)


@dataclass(frozen=True)
class _ColumnSets:
    measures: list[DataColumnProfile]
    dimensions: list[DataColumnProfile]
    temporal: list[DataColumnProfile]
    identifiers: list[DataColumnProfile]


class AnalysisPlanner:
    """Create a compact, deterministic analysis plan for a single table.

    The planner deliberately does not choose a final chart type. It selects up to
    three analytical subtasks that the existing single-chart ViRAGE pipeline can
    execute. RAG is later invoked per subtask to provide task-specific guidance.
    """

    def __init__(self, *, max_charts: int = 3) -> None:
        if max_charts < 1 or max_charts > 3:
            raise ValueError("max_charts must be between 1 and 3.")
        self.max_charts = max_charts

    def plan(
        self,
        *,
        user_query: str,
        data_path: str,
        data_profile: DataProfile,
        input_type: str = "table",
        original_input_path: str | None = None,
        preprocessing_report_path: str | None = None,
    ) -> AnalysisPlan:
        columns = self._column_sets(data_profile)
        candidates: list[AnalysisSubtask] = []
        skipped: list[SkippedAnalysisCandidate] = []
        query = user_query.strip()
        query_lower = query.lower()
        is_general = bool(_GENERAL_QUERY_RE.search(query))
        if input_type == "image_folder":
            return self._image_folder_plan(
                query=query,
                data_path=data_path,
                data_profile=data_profile,
                columns=columns,
                original_input_path=original_input_path,
                preprocessing_report_path=preprocessing_report_path,
            )

        def add(candidate: AnalysisSubtask | None, task_type: AnalysisTaskType, reason: str, required: list[str]) -> None:
            if candidate is None:
                skipped.append(SkippedAnalysisCandidate(task_type=task_type, reason=reason, required_fields=required))
                return
            candidates.append(candidate)

        if _TEMPORAL_RE.search(query) or (is_general and columns.temporal and columns.measures):
            add(
                self._temporal_task(query, columns),
                "temporal_trend",
                "No temporal field with numeric measure was found.",
                ["temporal", "numeric"],
            )
        if _GROUP_RE.search(query) or (is_general and columns.dimensions and columns.measures):
            add(
                self._group_comparison_task(query, columns),
                "group_comparison",
                "No categorical dimension with numeric measure was found.",
                ["categorical", "numeric"],
            )
        if _CORRELATION_RE.search(query) or (is_general and len(columns.measures) >= 2):
            add(
                self._correlation_task(query, columns),
                "correlation",
                "Fewer than two numeric measures were found.",
                ["numeric", "numeric"],
            )
        if _DISTRIBUTION_RE.search(query) or is_general or columns.measures:
            add(
                self._distribution_task(query, columns),
                "distribution",
                "No numeric measure was found.",
                ["numeric"],
            )
        if _OUTLIER_RE.search(query) or (is_general and any((column.outlier_count or 0) > 0 for column in columns.measures)):
            add(
                self._outlier_task(query, columns),
                "outlier_detection",
                "No numeric measure with detected outliers was found.",
                ["numeric"],
            )
        if _MISSING_RE.search(query) or (is_general and self._has_missingness(data_profile)):
            add(
                self._missingness_task(query, data_profile),
                "missingness_analysis",
                "No fields with missing values were found.",
                ["field_with_missing_values"],
            )
        if _RANKING_RE.search(query):
            add(
                self._ranking_task(query, columns),
                "ranking",
                "No dimension and numeric measure suitable for ranking were found.",
                ["categorical", "numeric"],
            )

        selected = self._select_tasks(candidates)
        if not selected and columns.measures:
            default_task = self._distribution_task(query, columns)
            if default_task is not None:
                selected = [default_task]
        if not selected:
            selected = [
                AnalysisSubtask(
                    id="overview_001",
                    task_type="overview",
                    query=f"Summarize the available dataset structure for: {query}",
                    purpose="Provide a compact overview when no numeric analytical subtask can be safely selected.",
                    required_fields=[column.name for column in data_profile.columns[: min(5, len(data_profile.columns))]],
                    priority=90,
                    rationale="No standard numeric, categorical, or temporal pattern was available.",
                )
            ]

        return AnalysisPlan(
            user_query=query,
            data_path=data_path,
            input_type=input_type,
            original_input_path=original_input_path,
            preprocessing_report_path=preprocessing_report_path,
            max_charts=self.max_charts,
            subtasks=selected,
            skipped_candidates=skipped,
            rationale=self._plan_rationale(query_lower, data_profile, selected, is_general),
        )


    def _image_folder_plan(
        self,
        *,
        query: str,
        data_path: str,
        data_profile: DataProfile,
        columns: _ColumnSets,
        original_input_path: str | None,
        preprocessing_report_path: str | None,
    ) -> AnalysisPlan:
        candidates: list[AnalysisSubtask] = []
        skipped: list[SkippedAnalysisCandidate] = []
        available = {column.name for column in data_profile.columns}

        def has(*names: str) -> bool:
            return all(name in available for name in names)

        if has("brisque_score", "niqe_score", "piqe_score", "laplacian_variance"):
            candidates.append(
                AnalysisSubtask(
                    id="image_quality_analysis_001",
                    task_type="image_quality_analysis",
                    query=(
                        "Analyse image quality using BRISQUE, NIQE, PIQE, exposure, sharpness, contrast, and brightness metrics. "
                        f"Original request: {query}"
                    ),
                    purpose="Identify potential quality problems in the image collection using standard no-reference IQA metrics and interpretable image features.",
                    required_fields=["brisque_score", "niqe_score", "piqe_score", "laplacian_variance"],
                    optional_fields=[
                        field
                        for field in [
                            "file_name",
                            "relative_path",
                            "group",
                            "contrast_rms",
                            "mean_brightness",
                            "clipping_ratio",
                            "exposure_balance_score",
                        ]
                        if field in available
                    ],
                    priority=10,
                    constraints={
                        "output_target": "scientific_figure",
                        "input_type": "image_folder",
                        "quality_metrics": [
                            "brisque_score",
                            "niqe_score",
                            "piqe_score",
                            "laplacian_variance",
                            "contrast_rms",
                            "mean_brightness",
                        ],
                        "iqa_score_direction": "lower_is_better",
                    },
                    rationale="Selected standard no-reference IQA metrics from pyiqa plus interpretable image-quality features.",
                )
            )
        elif has("laplacian_variance", "contrast_rms", "mean_brightness"):
            candidates.append(
                AnalysisSubtask(
                    id="image_quality_analysis_001",
                    task_type="image_quality_analysis",
                    query=(
                        "Analyse image quality using sharpness, contrast, and brightness metrics. "
                        f"Original request: {query}"
                    ),
                    purpose="Identify potential quality problems in the image collection using no-reference metrics.",
                    required_fields=["laplacian_variance", "contrast_rms", "mean_brightness"],
                    optional_fields=[field for field in ["file_name", "relative_path", "group"] if field in available],
                    priority=10,
                    constraints={
                        "output_target": "scientific_figure",
                        "input_type": "image_folder",
                        "quality_metrics": ["laplacian_variance", "contrast_rms", "mean_brightness"],
                    },
                    rationale="Selected no-reference image-quality metrics computed during image-folder preprocessing.",
                )
            )
        else:
            skipped.append(
                SkippedAnalysisCandidate(
                    task_type="image_quality_analysis",
                    reason="Required image-quality metrics were not found in the generated metrics table.",
                    required_fields=["brisque_score", "niqe_score", "piqe_score", "laplacian_variance"],
                )
            )

        sharpness = self._field_by_name(data_profile, "laplacian_variance")
        if sharpness is not None:
            candidates.append(
                AnalysisSubtask(
                    id="image_sharpness_distribution_001",
                    task_type="distribution",
                    query=f"Show the distribution of image sharpness using laplacian_variance. Original request: {query}",
                    purpose="Assess whether the image collection contains blurred or low-sharpness files.",
                    required_fields=["laplacian_variance"],
                    optional_fields=[field for field in ["group", "file_name"] if field in available],
                    priority=20,
                    constraints={"output_target": "scientific_figure", "input_type": "image_folder"},
                    rationale="Laplacian variance is available as a no-reference sharpness indicator.",
                )
            )

        if has("mean_brightness", "contrast_rms"):
            candidates.append(
                AnalysisSubtask(
                    id="brightness_contrast_relationship_001",
                    task_type="correlation",
                    query=f"Show the relationship between mean_brightness and contrast_rms. Original request: {query}",
                    purpose="Check whether brightness and contrast reveal problematic image groups or acquisition conditions.",
                    required_fields=["mean_brightness", "contrast_rms"],
                    optional_fields=[field for field in ["group", "file_name"] if field in available],
                    priority=30,
                    constraints={"output_target": "scientific_figure", "input_type": "image_folder"},
                    rationale="Brightness and RMS contrast were computed for every processed image.",
                )
            )

        problem_metric = self._image_problem_ranking_metric(available)
        if problem_metric and (has("file_name", problem_metric) or has("relative_path", problem_metric)):
            id_field = "relative_path" if "relative_path" in available else "file_name"
            prefer_low_values = problem_metric == "laplacian_variance"
            candidates.append(
                AnalysisSubtask(
                    id="problem_image_ranking_001",
                    task_type="ranking",
                    query=f"Rank images by potential quality problems using {problem_metric}. Original request: {query}",
                    purpose="Surface files with high no-reference IQA scores or low sharpness for manual inspection.",
                    required_fields=[id_field, problem_metric],
                    optional_fields=[
                        field
                        for field in ["contrast_rms", "mean_brightness", "clipping_ratio", "exposure_balance_score", "group"]
                        if field in available
                    ],
                    priority=40,
                    constraints={
                        "output_target": "scientific_figure",
                        "input_type": "image_folder",
                        "sort_values": True,
                        "prefer_low_values": prefer_low_values,
                    },
                    rationale=f"Selected '{id_field}' as image identifier and '{problem_metric}' as quality ranking metric.",
                )
            )

        selected = self._select_tasks(candidates)
        if not selected:
            selected = [
                AnalysisSubtask(
                    id="image_overview_001",
                    task_type="overview",
                    query=f"Summarize the generated image metrics table. Original request: {query}",
                    purpose="Provide a compact overview of generated image metadata and quality metrics.",
                    required_fields=[column.name for column in data_profile.columns[: min(5, len(data_profile.columns))]],
                    priority=90,
                    constraints={"output_target": "scientific_figure", "input_type": "image_folder"},
                    rationale="No standard image quality metric pattern was available.",
                )
            ]

        return AnalysisPlan(
            user_query=query,
            data_path=data_path,
            input_type="image_folder",
            original_input_path=original_input_path,
            preprocessing_report_path=preprocessing_report_path,
            max_charts=self.max_charts,
            subtasks=selected,
            skipped_candidates=skipped,
            rationale=[
                f"Image-folder input was converted to a metrics table with {data_profile.row_count} rows and {data_profile.col_count} columns.",
                f"Selected {len(selected)} image-analysis subtask(s), limit is {self.max_charts}.",
                "The planner uses no-reference image metrics only; it does not infer domain labels or diagnose image content.",
                "Chart types are not fixed by the planner; RAG and spec generation decide how to visualize each selected task.",
            ],
        )

    @staticmethod
    def _image_problem_ranking_metric(available: set[str]) -> str | None:
        for metric in ("brisque_score", "niqe_score", "piqe_score"):
            if metric in available:
                return metric
        if "laplacian_variance" in available:
            return "laplacian_variance"
        return None

    @staticmethod
    def _field_by_name(profile: DataProfile, name: str) -> DataColumnProfile | None:
        for column in profile.columns:
            if column.name == name:
                return column
        return None

    def _select_tasks(self, candidates: list[AnalysisSubtask]) -> list[AnalysisSubtask]:
        unique: list[AnalysisSubtask] = []
        seen_types: set[str] = set()
        seen_field_sets: set[tuple[str, ...]] = set()
        for candidate in sorted(candidates, key=lambda item: (item.priority, item.id)):
            field_key = tuple(candidate.required_fields)
            if candidate.task_type in seen_types:
                continue
            if field_key and field_key in seen_field_sets:
                continue
            seen_types.add(candidate.task_type)
            if field_key:
                seen_field_sets.add(field_key)
            unique.append(candidate)
            if len(unique) >= self.max_charts:
                break
        return unique

    def _distribution_task(self, query: str, columns: _ColumnSets) -> AnalysisSubtask | None:
        measure = self._best_measure(columns.measures)
        if measure is None:
            return None
        return AnalysisSubtask(
            id="distribution_001",
            task_type="distribution",
            query=f"Show the distribution of {measure.name}. Original request: {query}",
            purpose="Inspect the shape, spread, and possible skew of the main numeric measure.",
            required_fields=[measure.name],
            priority=30,
            constraints={"output_target": "scientific_figure", "do_not_change_task_type": True},
            rationale=f"Selected numeric measure '{measure.name}' for an initial distribution view.",
        )

    def _group_comparison_task(self, query: str, columns: _ColumnSets) -> AnalysisSubtask | None:
        dimension = self._best_dimension(columns.dimensions)
        measure = self._best_measure(columns.measures)
        if dimension is None or measure is None:
            return None
        return AnalysisSubtask(
            id="group_comparison_001",
            task_type="group_comparison",
            query=f"Compare {measure.name} across {dimension.name}. Original request: {query}",
            purpose="Compare a numeric measurement between meaningful groups or experimental conditions.",
            required_fields=[dimension.name, measure.name],
            priority=20,
            constraints={"output_target": "scientific_figure", "show_distribution_or_uncertainty": True},
            rationale=f"Selected categorical dimension '{dimension.name}' and measure '{measure.name}'.",
        )

    def _correlation_task(self, query: str, columns: _ColumnSets) -> AnalysisSubtask | None:
        measures = [column for column in columns.measures if not column.is_identifier][:2]
        if len(measures) < 2:
            return None
        return AnalysisSubtask(
            id="correlation_001",
            task_type="correlation",
            query=f"Show the relationship between {measures[0].name} and {measures[1].name}. Original request: {query}",
            purpose="Check whether two numeric measurements are associated.",
            required_fields=[measures[0].name, measures[1].name],
            priority=40,
            constraints={"output_target": "scientific_figure"},
            rationale=f"Selected first two suitable numeric measures: '{measures[0].name}', '{measures[1].name}'.",
        )

    def _temporal_task(self, query: str, columns: _ColumnSets) -> AnalysisSubtask | None:
        if not columns.temporal or not columns.measures:
            return None
        time_field = columns.temporal[0]
        measure = self._best_measure(columns.measures)
        if measure is None:
            return None
        return AnalysisSubtask(
            id="temporal_trend_001",
            task_type="temporal_trend",
            query=f"Show how {measure.name} changes over {time_field.name}. Original request: {query}",
            purpose="Inspect temporal dynamics of a numeric measurement.",
            required_fields=[time_field.name, measure.name],
            priority=10,
            constraints={"output_target": "scientific_figure", "preserve_time_order": True},
            rationale=f"Selected temporal field '{time_field.name}' and measure '{measure.name}'.",
        )

    def _outlier_task(self, query: str, columns: _ColumnSets) -> AnalysisSubtask | None:
        outlier_measures = [column for column in columns.measures if (column.outlier_count or 0) > 0]
        measure = self._best_measure(outlier_measures or columns.measures)
        if measure is None:
            return None
        return AnalysisSubtask(
            id="outlier_detection_001",
            task_type="outlier_detection",
            query=f"Highlight potential outliers in {measure.name}. Original request: {query}",
            purpose="Identify extreme values that may require scientific interpretation or data-quality review.",
            required_fields=[measure.name],
            optional_fields=[columns.dimensions[0].name] if columns.dimensions else [],
            priority=50,
            constraints={"output_target": "scientific_figure", "make_extreme_values_visible": True},
            rationale=f"Selected measure '{measure.name}' for outlier inspection.",
        )

    def _missingness_task(self, query: str, profile: DataProfile) -> AnalysisSubtask | None:
        fields = [column.name for column in profile.columns if (column.missing_ratio or 0.0) > 0.0]
        if not fields:
            return None
        return AnalysisSubtask(
            id="missingness_analysis_001",
            task_type="missingness_analysis",
            query=f"Show missing data patterns for fields with missing values. Original request: {query}",
            purpose="Assess whether missing values can affect downstream visual interpretation.",
            required_fields=fields[: min(8, len(fields))],
            priority=60,
            constraints={"output_target": "scientific_figure"},
            rationale=f"Detected {len(fields)} fields with missing values.",
        )

    def _ranking_task(self, query: str, columns: _ColumnSets) -> AnalysisSubtask | None:
        dimension = self._best_dimension(columns.dimensions)
        measure = self._best_measure(columns.measures)
        if dimension is None or measure is None:
            return None
        return AnalysisSubtask(
            id="ranking_001",
            task_type="ranking",
            query=f"Rank {dimension.name} by {measure.name}. Original request: {query}",
            purpose="Identify the highest or lowest categories by a numeric measure.",
            required_fields=[dimension.name, measure.name],
            priority=25,
            constraints={"output_target": "scientific_figure", "sort_values": True},
            rationale=f"Selected dimension '{dimension.name}' and measure '{measure.name}' for ranking.",
        )

    def _plan_rationale(
        self,
        query_lower: str,
        profile: DataProfile,
        selected: list[AnalysisSubtask],
        is_general: bool,
    ) -> list[str]:
        rationale = [
            f"Dataset profile: {profile.row_count} rows, {profile.col_count} columns.",
            f"Selected {len(selected)} analytical subtask(s), limit is {self.max_charts}.",
        ]
        if is_general:
            rationale.append("The user request is broad; the plan prioritizes complementary overview, comparison, and diagnostic tasks.")
        else:
            rationale.append(f"The user request was matched against analytical task keywords: {query_lower[:160]}.")
        rationale.append("Chart types are not fixed by the planner; RAG and spec generation decide how to visualize each selected task.")
        return rationale

    @staticmethod
    def _column_sets(profile: DataProfile) -> _ColumnSets:
        measures = [column for column in profile.columns if column.role == "measure" and not column.is_identifier]
        dimensions = [
            column
            for column in profile.columns
            if column.role == "dimension" and not column.is_identifier and not column.is_high_cardinality
        ]
        temporal = [column for column in profile.columns if column.role == "temporal" and not column.is_identifier]
        identifiers = [column for column in profile.columns if column.is_identifier or column.role == "identifier"]
        return _ColumnSets(measures=measures, dimensions=dimensions, temporal=temporal, identifiers=identifiers)

    @staticmethod
    def _best_measure(columns: list[DataColumnProfile]) -> DataColumnProfile | None:
        if not columns:
            return None
        return sorted(
            columns,
            key=lambda column: (
                bool(column.is_identifier),
                bool(column.missing_ratio >= 0.5),
                -(column.outlier_count or 0),
                column.name.lower(),
            ),
        )[0]

    @staticmethod
    def _best_dimension(columns: list[DataColumnProfile]) -> DataColumnProfile | None:
        if not columns:
            return None
        return sorted(
            columns,
            key=lambda column: (
                bool(column.is_high_cardinality),
                bool(column.missing_ratio >= 0.5),
                column.unique_count if column.unique_count > 0 else 10**9,
                column.name.lower(),
            ),
        )[0]

    @staticmethod
    def _has_missingness(profile: DataProfile) -> bool:
        return any((column.missing_ratio or 0.0) > 0.0 for column in profile.columns)
