# AutoRAG QA export report

Generated at: `2026-05-05T21:34:09.478794+00:00`
Input JSONL: `rag_corpus/normalized/jsonl/official_vega_lite_examples.jsonl`
Corpus parquet: `rag_corpus/autorag/vega_lite/corpus.parquet`
QA parquet: `rag_corpus/autorag/vega_lite/qa.parquet`
Normalized records: **100**
QA rows: **100**
Query modes: `query_field`

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
- Max query length: `98`
- Avg query length: `78.25`

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

## Sample rows

| qid | query | retrieval_gt | generation_gt |
|---|---|---|---|
| `vega_lite_qa_000001` | Create a scatter plot that shows the relationship between two quantitative fields. | `[["vega_lite:example:selection_project_multi"]]` | `["scatter_plot using point mark with field roles: color unknown, size unknown, x quantitative, y quantitative"]` |
| `vega_lite_qa_000002` | Create a rule chart that shows reference lines, ranges, or intervals. | `[["vega_lite:example:rule_extent"]]` | `["rule_chart using rule mark with field roles: x unknown, y unknown_aggregate"]` |
| `vega_lite_qa_000003` | Create a layered chart that combines multiple marks or views in one visualization. | `[["vega_lite:example:layer_ranged_dot"]]` | `["layered_chart using line mark with field roles: color ordinal, opacity unknown, size unknown, x quantitative, y nominal"]` |
| `vega_lite_qa_000004` | Create a line chart that shows how a quantitative value changes over an ordered or temporal field. | `[["vega_lite:example:line_skip_invalid_mid"]]` | `["line_chart using line mark with field roles: x quantitative, y quantitative"]` |
| `vega_lite_qa_000005` | Create a geographic visualization using geospatial shapes or coordinates. | `[["vega_lite:example:geo_graticule_object"]]` | `["map_chart using geoshape mark with field roles: none"]` |
| `vega_lite_qa_000006` | Create a heatmap that uses color intensity to compare values across two dimensions. | `[["vega_lite:example:selection_type_single_dblclick"]]` | `["heatmap using rect mark with field roles: color unknown, x unknown, y unknown"]` |
| `vega_lite_qa_000007` | Create a scatter plot that shows the relationship between two quantitative fields. | `[["vega_lite:example:point_color_shape_constant"]]` | `["scatter_plot using point mark with field roles: color unknown, shape unknown, x quantitative, y quantitative"]` |
| `vega_lite_qa_000008` | Create a histogram that shows the distribution of a quantitative field using bins and counts. | `[["vega_lite:example:bar_sort_by_count"]]` | `["histogram using bar mark with field roles: x unknown, y count"]` |
| `vega_lite_qa_000009` | Create a scatter plot that shows the relationship between two quantitative fields. | `[["vega_lite:example:selection_clear_brush"]]` | `["scatter_plot using point mark with field roles: color unknown, x quantitative, y quantitative"]` |
| `vega_lite_qa_000010` | Create a line chart that shows how a quantitative value changes over an ordered or temporal field. | `[["vega_lite:example:time_output_utc_scale"]]` | `["line_chart using line mark with field roles: x unknown_timeunit, y quantitative"]` |
