# AutoRAG QA export report

Generated at: `2026-05-05T22:55:07.023371+00:00`
Input JSONL: `rag_corpus/normalized/jsonl/official_vega_lite_examples.jsonl`
Corpus parquet: `rag_corpus/autorag/vega_lite/corpus.parquet`
QA parquet: `rag_corpus/autorag/vega_lite/qa_mixed_technical.parquet`
Normalized records: **100**
QA rows: **300**
Query modes: `query_field, title_query, chart_pattern`

## Field mapping

- id_field: `id`
- query_field: `instruction`
- title_field: `title`
- generation_field: `None`
- chart_pattern_field: `chart_pattern`
- mark_type_field: `mark_type`
- field_roles_field: `field_roles`

## Query length

- Min query length: `57`
- Max query length: `172`
- Avg query length: `97.55`

## Chart pattern distribution from normalized records

| Value | Count |
|---|---:|
| `bar_chart` | 9 |
| `scatter_plot` | 8 |
| `layered_chart` | 8 |
| `line_chart` | 8 |
| `heatmap` | 8 |
| `histogram` | 8 |
| `tick_plot` | 8 |
| `area_chart` | 8 |
| `faceted_chart` | 8 |
| `point_chart` | 8 |
| `pie_chart` | 7 |
| `map_chart` | 4 |
| `rule_chart` | 3 |
| `text_chart` | 3 |
| `rect_chart` | 2 |

## Mark type distribution from normalized records

| Value | Count |
|---|---:|
| `bar` | 22 |
| `point` | 15 |
| `line` | 11 |
| `area` | 11 |
| `rect` | 10 |
| `tick` | 8 |
| `arc` | 7 |
| `geoshape` | 5 |
| `circle` | 5 |
| `rule` | 3 |
| `text` | 3 |

## Query mode distribution

| Value | Count |
|---|---:|
| `query_field` | 100 |
| `title_query` | 100 |
| `chart_pattern` | 100 |

## Sample rows

| qid | query | retrieval_gt | generation_gt |
|---|---|---|---|
| `vega_lite_mixed_technical_qa_000001` | Create a scatter plot that shows the relationship between two quantitative fields. | `[["vega_lite:example:selection_project_multi"]]` | `["scatter_plot using point mark with field roles: color unknown, size unknown, x quantitative, y quantitative"]` |
| `vega_lite_mixed_technical_qa_000002` | selection project multi. Create a scatter plot that shows the relationship between two quantitative fields. | `[["vega_lite:example:selection_project_multi"]]` | `["scatter_plot using point mark with field roles: color unknown, size unknown, x quantitative, y quantitative"]` |
| `vega_lite_mixed_technical_qa_000003` | Find a relevant example for scatter_plot using point mark with field roles: color unknown, size unknown, x quantitative, y quantitative. | `[["vega_lite:example:selection_project_multi"]]` | `["scatter_plot using point mark with field roles: color unknown, size unknown, x quantitative, y quantitative"]` |
| `vega_lite_mixed_technical_qa_000004` | Create a rule chart that shows reference lines, ranges, or intervals. | `[["vega_lite:example:rule_extent"]]` | `["rule_chart using rule mark with field roles: x unknown, y unknown_aggregate"]` |
| `vega_lite_mixed_technical_qa_000005` | rule extent. Create a rule chart that shows reference lines, ranges, or intervals. | `[["vega_lite:example:rule_extent"]]` | `["rule_chart using rule mark with field roles: x unknown, y unknown_aggregate"]` |
| `vega_lite_mixed_technical_qa_000006` | Find a relevant example for rule_chart using rule mark with field roles: x unknown, y unknown_aggregate. | `[["vega_lite:example:rule_extent"]]` | `["rule_chart using rule mark with field roles: x unknown, y unknown_aggregate"]` |
| `vega_lite_mixed_technical_qa_000007` | Create a layered chart that combines multiple marks or views in one visualization. | `[["vega_lite:example:layer_ranged_dot"]]` | `["layered_chart using line mark with field roles: color ordinal, opacity unknown, size unknown, x quantitative, y nominal"]` |
| `vega_lite_mixed_technical_qa_000008` | layer ranged dot. Create a layered chart that combines multiple marks or views in one visualization. | `[["vega_lite:example:layer_ranged_dot"]]` | `["layered_chart using line mark with field roles: color ordinal, opacity unknown, size unknown, x quantitative, y nominal"]` |
| `vega_lite_mixed_technical_qa_000009` | Find a relevant example for layered_chart using line mark with field roles: color ordinal, opacity unknown, size unknown, x quantitative, y nominal. | `[["vega_lite:example:layer_ranged_dot"]]` | `["layered_chart using line mark with field roles: color ordinal, opacity unknown, size unknown, x quantitative, y nominal"]` |
| `vega_lite_mixed_technical_qa_000010` | Create a line chart that shows how a quantitative value changes over an ordered or temporal field. | `[["vega_lite:example:line_skip_invalid_mid"]]` | `["line_chart using line mark with field roles: x quantitative, y quantitative"]` |
