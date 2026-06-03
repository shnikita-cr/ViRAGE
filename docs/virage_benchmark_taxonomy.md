# ViRAGE benchmark taxonomy

This document defines the benchmark suites used to test ViRAGE after the RAG, orchestrator, image-folder, feedback, and VLM judge updates.

## Case classification fields

Each benchmark case is stored as one JSON object in `benchmarks/cases/*.jsonl`.

Required fields:

- `case_id` — stable case identifier.
- `suite` — benchmark suite name.
- `input_modality` — `table`, `image_folder`, or reserved `chart_image_only`.
- `query_specificity` — `specific`, `semi_open`, or `open_ended`.
- `analysis_task` — expected analytical task family.
- `chart_family` — expected or stress-tested visualization family.
- `output_target` — `eda`, `scientific_article`, `presentation`, or `debug_analysis`.
- `expected_charts` — `single_chart`, `max_3_charts`, or `multi_panel`.
- `evaluation_mode` — how the case is evaluated.
- `known_risks` — vulnerabilities this case is intended to expose.
- `data_path` — table path or image folder path.
- `query` — user request passed to the orchestrator.
- `expected_checks` — metrics or artifacts to inspect after execution.

## Suites

| Suite | Purpose |
|---|---|
| `manual_vulnerability` | Known weaknesses: axis domain, empty plot area, compact layout, repeat labels, publication readiness. |
| `eda_manual` | Open-ended EDA requests that should produce complementary diagnostic subtasks. |
| `chart_type_coverage` | Coverage for bar, line, scatter, heatmap, boxplot/histogram, repeat/facet behavior. |
| `analysis_task_coverage` | Distribution, group comparison, correlation, temporal trend, ranking, outlier detection, missingness analysis. |
| `image_folder_quality` | Image-folder preprocessing and analysis using PNG/JPG/TIFF samples and IQA metrics. |
| `nlv_comparison` | Reserved suite for NLV/VegaChat-compatible cases. |
| `external_agents_image_only` | Reserved suite for comparing only chart images from external agents. |

## Main runner

Plan-only mode:

    python scripts/benchmarks/run_virage_e2e_test_cases.py --config ui/config/benchmark/project-gemma4-bench_rag.toml --run-id e2e_plan_001

Execute mode:

    python scripts/benchmarks/run_virage_e2e_test_cases.py --config ui/config/benchmark/project-gemma4-bench_rag.toml --run-id e2e_execute_001 --execute

Run one suite:

    python scripts/benchmarks/run_virage_e2e_test_cases.py --config ui/config/benchmark/project-gemma4-bench_rag.toml --run-id e2e_eda_001 --suite eda_manual --execute

Run one case:

    python scripts/benchmarks/run_virage_e2e_test_cases.py --config ui/config/benchmark/project-gemma4-bench_rag.toml --run-id e2e_axis_001 --case-id axis_domain_narrow_range --execute

Run image-folder suite with custom folder:

    python scripts/benchmarks/run_virage_e2e_test_cases.py --config ui/config/benchmark/project-gemma4-bench_rag.toml --run-id e2e_images_001 --suite image_folder_quality --image-folder path/to/images --execute

## Output artifacts

The runner writes:

    artifacts/<run_id>/e2e_cases_report/benchmark_request.json
    artifacts/<run_id>/e2e_cases_report/per_case_results.csv
    artifacts/<run_id>/e2e_cases_report/per_case_results.jsonl
    artifacts/<run_id>/e2e_cases_report/benchmark_summary.json
    artifacts/<run_id>/e2e_cases_report/benchmark_report.md

Each case also creates the usual orchestrator artifacts under:

    artifacts/<run_id>/cases/<case_id>/

If `--execute` is used, subrun artifacts are created under:

    artifacts/<run_id>/cases/<case_id>/subruns/

## Metrics collected from subruns

When `--execute` is enabled, the runner aggregates metrics from each subrun `run_report.json`:

- `valid_spec_rate`
- `render_success_rate`
- `empty_chart_rate`
- `mean_spec_score`
- `mean_vision_score`
- `mean_plot_area_usage_score`
- `mean_axis_domain_score`
- `mean_layout_compactness_score`
- `mean_repeat_axis_label_score`
- `mean_publication_layout_score`
- `mean_duration_seconds`

Plan-only mode still records planned subtasks, suite metadata, and artifact paths, but does not produce chart-quality metrics.
