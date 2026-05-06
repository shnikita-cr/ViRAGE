# Official Vega-Lite examples normalization report

Cleaned root: `rag_corpus/cleaned`
Specs dir: `rag_corpus/cleaned/vega-lite/examples/specs`
Seed: `42`
Total parsed candidates: **627**
Selected records: **100**
Skipped parse errors: **0**
Removed data/datasets sections from selected templates: **105**

## Corpus chart pattern distribution

Detected chart patterns across all parsed Vega-Lite candidates before sampling.

| Chart pattern | Count | Share |
|---|---:|---:|
| `bar_chart` | 124 | 19.78% |
| `layered_chart` | 113 | 18.02% |
| `scatter_plot` | 85 | 13.56% |
| `faceted_chart` | 63 | 10.05% |
| `line_chart` | 62 | 9.89% |
| `point_chart` | 43 | 6.86% |
| `histogram` | 39 | 6.22% |
| `area_chart` | 22 | 3.51% |
| `unknown` | 22 | 3.51% |
| `heatmap` | 21 | 3.35% |
| `tick_plot` | 14 | 2.23% |
| `pie_chart` | 7 | 1.12% |
| `map_chart` | 4 | 0.64% |
| `rule_chart` | 3 | 0.48% |
| `text_chart` | 3 | 0.48% |
| `rect_chart` | 2 | 0.32% |

## Corpus mark type distribution

| Mark type | Count | Share |
|---|---:|---:|
| `bar` | 223 | 35.57% |
| `point` | 130 | 20.73% |
| `line` | 100 | 15.95% |
| `circle` | 53 | 8.45% |
| `area` | 28 | 4.47% |
| `rect` | 26 | 4.15% |
| `boxplot` | 16 | 2.55% |
| `tick` | 14 | 2.23% |
| `arc` | 11 | 1.75% |
| `geoshape` | 8 | 1.28% |
| `text` | 5 | 0.8% |
| `errorband` | 3 | 0.48% |
| `rule` | 3 | 0.48% |
| `errorbar` | 2 | 0.32% |
| `square` | 2 | 0.32% |
| `trail` | 2 | 0.32% |
| `image` | 1 | 0.16% |

## Selected chart pattern coverage

Chart patterns in the sampled normalized JSONL output.

| Chart pattern | Count | Share |
|---|---:|---:|
| `bar_chart` | 9 | 9.0% |
| `scatter_plot` | 8 | 8.0% |
| `layered_chart` | 8 | 8.0% |
| `line_chart` | 8 | 8.0% |
| `heatmap` | 8 | 8.0% |
| `histogram` | 8 | 8.0% |
| `tick_plot` | 8 | 8.0% |
| `area_chart` | 8 | 8.0% |
| `faceted_chart` | 8 | 8.0% |
| `point_chart` | 8 | 8.0% |
| `pie_chart` | 7 | 7.0% |
| `map_chart` | 4 | 4.0% |
| `rule_chart` | 3 | 3.0% |
| `text_chart` | 3 | 3.0% |
| `rect_chart` | 2 | 2.0% |

## Selected mark type coverage

| Mark type | Count | Share |
|---|---:|---:|
| `bar` | 22 | 22.0% |
| `point` | 15 | 15.0% |
| `line` | 11 | 11.0% |
| `area` | 11 | 11.0% |
| `rect` | 10 | 10.0% |
| `tick` | 8 | 8.0% |
| `arc` | 7 | 7.0% |
| `geoshape` | 5 | 5.0% |
| `circle` | 5 | 5.0% |
| `rule` | 3 | 3.0% |
| `text` | 3 | 3.0% |

## Selected records

| ID | Title | Pattern | Mark | Removed data sections | Source path |
|---|---|---|---|---:|---|
| `vega_lite:example:arc_color_mappings` | `arc color mappings` | `pie_chart` | `arc` | 1 | `vega-lite/examples/specs/arc_color_mappings.vl.json` |
| `vega_lite:example:arc_donut` | `arc donut` | `pie_chart` | `arc` | 1 | `vega-lite/examples/specs/arc_donut.vl.json` |
| `vega_lite:example:arc_ordinal_theta` | `arc ordinal theta` | `pie_chart` | `arc` | 1 | `vega-lite/examples/specs/arc_ordinal_theta.vl.json` |
| `vega_lite:example:arc_params` | `arc params` | `pie_chart` | `arc` | 2 | `vega-lite/examples/specs/arc_params.vl.json` |
| `vega_lite:example:arc_pie` | `arc pie` | `pie_chart` | `arc` | 1 | `vega-lite/examples/specs/arc_pie.vl.json` |
| `vega_lite:example:arc_pie_normalize_tooltip` | `arc pie normalize tooltip` | `pie_chart` | `arc` | 1 | `vega-lite/examples/specs/arc_pie_normalize_tooltip.vl.json` |
| `vega_lite:example:arc_pie_pyramid` | `arc pie pyramid` | `pie_chart` | `arc` | 1 | `vega-lite/examples/specs/arc_pie_pyramid.vl.json` |
| `vega_lite:example:area` | `area` | `area_chart` | `area` | 1 | `vega-lite/examples/specs/area.vl.json` |
| `vega_lite:example:area_cumulative_freq` | `area cumulative freq` | `area_chart` | `area` | 1 | `vega-lite/examples/specs/area_cumulative_freq.vl.json` |
| `vega_lite:example:area_density` | `area density` | `area_chart` | `area` | 1 | `vega-lite/examples/specs/area_density.vl.json` |
| `vega_lite:example:area_density_facet` | `area density facet` | `faceted_chart` | `area` | 1 | `vega-lite/examples/specs/area_density_facet.vl.json` |
| `vega_lite:example:area_gradient` | `area gradient` | `area_chart` | `area` | 1 | `vega-lite/examples/specs/area_gradient.vl.json` |
| `vega_lite:example:area_horizon` | `area horizon` | `layered_chart` | `area` | 1 | `vega-lite/examples/specs/area_horizon.vl.json` |
| `vega_lite:example:area_invalid_null` | `area invalid null` | `area_chart` | `area` | 1 | `vega-lite/examples/specs/area_invalid_null.vl.json` |
| `vega_lite:example:area_overlay` | `area overlay` | `area_chart` | `area` | 1 | `vega-lite/examples/specs/area_overlay.vl.json` |
| `vega_lite:example:bar_aggregate_count` | `bar aggregate count` | `histogram` | `bar` | 1 | `vega-lite/examples/specs/bar_aggregate_count.vl.json` |
| `vega_lite:example:bar_axis_orient` | `bar axis orient` | `bar_chart` | `bar` | 1 | `vega-lite/examples/specs/bar_axis_orient.vl.json` |
| `vega_lite:example:bar_diverging_stack_population_pyramid` | `bar diverging stack population pyramid` | `bar_chart` | `bar` | 1 | `vega-lite/examples/specs/bar_diverging_stack_population_pyramid.vl.json` |
| `vega_lite:example:bar_fit` | `bar fit` | `bar_chart` | `bar` | 1 | `vega-lite/examples/specs/bar_fit.vl.json` |
| `vega_lite:example:bar_grouped_facet_independent_scale` | `bar grouped facet independent scale` | `faceted_chart` | `bar` | 1 | `vega-lite/examples/specs/bar_grouped_facet_independent_scale.vl.json` |
| `vega_lite:example:bar_grouped_horizontal` | `bar grouped horizontal` | `bar_chart` | `bar` | 1 | `vega-lite/examples/specs/bar_grouped_horizontal.vl.json` |
| `vega_lite:example:bar_month_band` | `bar month band` | `bar_chart` | `bar` | 1 | `vega-lite/examples/specs/bar_month_band.vl.json` |
| `vega_lite:example:bar_sort_by_count` | `bar sort by count` | `histogram` | `bar` | 1 | `vega-lite/examples/specs/bar_sort_by_count.vl.json` |
| `vega_lite:example:bar_tooltip_aggregate` | `bar tooltip aggregate` | `tick_plot` | `tick` | 1 | `vega-lite/examples/specs/bar_tooltip_aggregate.vl.json` |
| `vega_lite:example:bar_tooltip_groupby` | `bar tooltip groupby` | `tick_plot` | `tick` | 1 | `vega-lite/examples/specs/bar_tooltip_groupby.vl.json` |
| `vega_lite:example:boxplot_preaggregated` | `boxplot preaggregated` | `layered_chart` | `bar` | 1 | `vega-lite/examples/specs/boxplot_preaggregated.vl.json` |
| `vega_lite:example:circle_binned_maxbins_20` | `circle binned maxbins 20` | `point_chart` | `circle` | 1 | `vega-lite/examples/specs/circle_binned_maxbins_20.vl.json` |
| `vega_lite:example:circle_labelangle_orient_signal` | `circle labelangle orient signal` | `point_chart` | `circle` | 1 | `vega-lite/examples/specs/circle_labelangle_orient_signal.vl.json` |
| `vega_lite:example:circle_scale_quantize` | `circle scale quantize` | `point_chart` | `circle` | 1 | `vega-lite/examples/specs/circle_scale_quantize.vl.json` |
| `vega_lite:example:circle_scale_threshold` | `circle scale threshold` | `point_chart` | `circle` | 1 | `vega-lite/examples/specs/circle_scale_threshold.vl.json` |
| `vega_lite:example:concat_layer_voyager_result` | `concat layer voyager result` | `point_chart` | `point` | 2 | `vega-lite/examples/specs/concat_layer_voyager_result.vl.json` |
| `vega_lite:example:concat_marginal_histograms` | `concat marginal histograms` | `histogram` | `bar` | 1 | `vega-lite/examples/specs/concat_marginal_histograms.vl.json` |
| `vega_lite:example:config_numberFormatType_test` | `config numberFormatType test` | `faceted_chart` | `point` | 1 | `vega-lite/examples/specs/config_numberFormatType_test.vl.json` |
| `vega_lite:example:config_numberFormatType_tooltip` | `config numberFormatType tooltip` | `bar_chart` | `bar` | 1 | `vega-lite/examples/specs/config_numberFormatType_tooltip.vl.json` |
| `vega_lite:example:errorbar_aggregate` | `errorbar aggregate` | `layered_chart` | `point` | 1 | `vega-lite/examples/specs/errorbar_aggregate.vl.json` |
| `vega_lite:example:facet_grid_bar` | `facet grid bar` | `faceted_chart` | `bar` | 1 | `vega-lite/examples/specs/facet_grid_bar.vl.json` |
| `vega_lite:example:facet_independent_scale` | `facet independent scale` | `faceted_chart` | `bar` | 1 | `vega-lite/examples/specs/facet_independent_scale.vl.json` |
| `vega_lite:example:geo_choropleth` | `geo choropleth` | `map_chart` | `geoshape` | 2 | `vega-lite/examples/specs/geo_choropleth.vl.json` |
| `vega_lite:example:geo_graticule` | `geo graticule` | `map_chart` | `geoshape` | 1 | `vega-lite/examples/specs/geo_graticule.vl.json` |
| `vega_lite:example:geo_graticule_object` | `geo graticule object` | `map_chart` | `geoshape` | 1 | `vega-lite/examples/specs/geo_graticule_object.vl.json` |
| `vega_lite:example:geo_params_projections` | `geo params projections` | `map_chart` | `geoshape` | 1 | `vega-lite/examples/specs/geo_params_projections.vl.json` |
| `vega_lite:example:histogram_bin_transform` | `histogram bin transform` | `histogram` | `bar` | 1 | `vega-lite/examples/specs/histogram_bin_transform.vl.json` |
| `vega_lite:example:histogram_no_spacing` | `histogram no spacing` | `histogram` | `bar` | 1 | `vega-lite/examples/specs/histogram_no_spacing.vl.json` |
| `vega_lite:example:histogram_nonlinear` | `histogram nonlinear` | `bar_chart` | `bar` | 1 | `vega-lite/examples/specs/histogram_nonlinear.vl.json` |
| `vega_lite:example:interactive_area_brush` | `interactive area brush` | `layered_chart` | `area` | 1 | `vega-lite/examples/specs/interactive_area_brush.vl.json` |
| `vega_lite:example:interactive_bin_extent_bottom` | `interactive bin extent bottom` | `histogram` | `bar` | 1 | `vega-lite/examples/specs/interactive_bin_extent_bottom.vl.json` |
| `vega_lite:example:interactive_geo_facet_species` | `interactive geo facet species` | `faceted_chart` | `geoshape` | 2 | `vega-lite/examples/specs/interactive_geo_facet_species.vl.json` |
| `vega_lite:example:interactive_histogram_full_height_hover` | `interactive histogram full height hover` | `layered_chart` | `bar` | 1 | `vega-lite/examples/specs/interactive_histogram_full_height_hover.vl.json` |
| `vega_lite:example:interactive_multi_line_label` | `interactive multi line label` | `layered_chart` | `line` | 1 | `vega-lite/examples/specs/interactive_multi_line_label.vl.json` |
| `vega_lite:example:layer_point_errorbar_2d_horizontal_color_encoding` | `layer point errorbar 2d horizontal color encoding` | `layered_chart` | `point` | 1 | `vega-lite/examples/specs/layer_point_errorbar_2d_horizontal_color_encoding.vl.json` |
| `vega_lite:example:layer_ranged_dot` | `layer ranged dot` | `layered_chart` | `line` | 1 | `vega-lite/examples/specs/layer_ranged_dot.vl.json` |
| `vega_lite:example:line` | `line` | `line_chart` | `line` | 1 | `vega-lite/examples/specs/line.vl.json` |
| `vega_lite:example:line_default_format` | `line default format` | `line_chart` | `line` | 1 | `vega-lite/examples/specs/line_default_format.vl.json` |
| `vega_lite:example:line_encoding_impute_keyvals_sequence` | `line encoding impute keyvals sequence` | `line_chart` | `line` | 1 | `vega-lite/examples/specs/line_encoding_impute_keyvals_sequence.vl.json` |
| `vega_lite:example:line_inside_domain_using_transform` | `line inside domain using transform` | `line_chart` | `line` | 1 | `vega-lite/examples/specs/line_inside_domain_using_transform.vl.json` |
| `vega_lite:example:line_mean_month` | `line mean month` | `line_chart` | `line` | 1 | `vega-lite/examples/specs/line_mean_month.vl.json` |
| `vega_lite:example:line_outside_domain` | `line outside domain` | `line_chart` | `line` | 1 | `vega-lite/examples/specs/line_outside_domain.vl.json` |
| `vega_lite:example:line_skip_invalid_mid` | `line skip invalid mid` | `line_chart` | `line` | 1 | `vega-lite/examples/specs/line_skip_invalid_mid.vl.json` |
| `vega_lite:example:lookup` | `lookup` | `bar_chart` | `bar` | 2 | `vega-lite/examples/specs/lookup.vl.json` |
| `vega_lite:example:param_search_input` | `param search input` | `scatter_plot` | `point` | 1 | `vega-lite/examples/specs/param_search_input.vl.json` |
| `vega_lite:example:point_2d` | `point 2d` | `scatter_plot` | `point` | 1 | `vega-lite/examples/specs/point_2d.vl.json` |
| `vega_lite:example:point_color_shape_constant` | `point color shape constant` | `scatter_plot` | `point` | 1 | `vega-lite/examples/specs/point_color_shape_constant.vl.json` |
| `vega_lite:example:point_colorramp_size` | `point colorramp size` | `scatter_plot` | `point` | 1 | `vega-lite/examples/specs/point_colorramp_size.vl.json` |
| `vega_lite:example:point_dot_timeunit_color` | `point dot timeunit color` | `point_chart` | `point` | 1 | `vega-lite/examples/specs/point_dot_timeunit_color.vl.json` |
| `vega_lite:example:point_offset_random` | `point offset random` | `point_chart` | `point` | 1 | `vega-lite/examples/specs/point_offset_random.vl.json` |
| `vega_lite:example:point_ordinal_bin_offset_random` | `point ordinal bin offset random` | `point_chart` | `point` | 1 | `vega-lite/examples/specs/point_ordinal_bin_offset_random.vl.json` |
| `vega_lite:example:rect_binned_heatmap` | `rect binned heatmap` | `heatmap` | `rect` | 1 | `vega-lite/examples/specs/rect_binned_heatmap.vl.json` |
| `vega_lite:example:rect_heatmap_weather` | `rect heatmap weather` | `heatmap` | `rect` | 1 | `vega-lite/examples/specs/rect_heatmap_weather.vl.json` |
| `vega_lite:example:rect_heatmap_weather_temporal_center_band_config` | `rect heatmap weather temporal center band config` | `heatmap` | `rect` | 1 | `vega-lite/examples/specs/rect_heatmap_weather_temporal_center_band_config.vl.json` |
| `vega_lite:example:rect_mosaic_labelled` | `rect mosaic labelled` | `heatmap` | `rect` | 1 | `vega-lite/examples/specs/rect_mosaic_labelled.vl.json` |
| `vega_lite:example:rect_mosaic_labelled_with_offset` | `rect mosaic labelled with offset` | `heatmap` | `rect` | 1 | `vega-lite/examples/specs/rect_mosaic_labelled_with_offset.vl.json` |
| `vega_lite:example:rect_params` | `rect params` | `rect_chart` | `rect` | 1 | `vega-lite/examples/specs/rect_params.vl.json` |
| `vega_lite:example:repeat_line_weather` | `repeat line weather` | `faceted_chart` | `line` | 1 | `vega-lite/examples/specs/repeat_line_weather.vl.json` |
| `vega_lite:example:repeat_splom_cars` | `repeat splom cars` | `faceted_chart` | `point` | 1 | `vega-lite/examples/specs/repeat_splom_cars.vl.json` |
| `vega_lite:example:rule_color_mean` | `rule color mean` | `rule_chart` | `rule` | 1 | `vega-lite/examples/specs/rule_color_mean.vl.json` |
| `vega_lite:example:rule_extent` | `rule extent` | `rule_chart` | `rule` | 1 | `vega-lite/examples/specs/rule_extent.vl.json` |
| `vega_lite:example:rule_params` | `rule params` | `rule_chart` | `rule` | 1 | `vega-lite/examples/specs/rule_params.vl.json` |
| `vega_lite:example:selection_clear_brush` | `selection clear brush` | `scatter_plot` | `point` | 1 | `vega-lite/examples/specs/selection_clear_brush.vl.json` |
| `vega_lite:example:selection_composition_and` | `selection composition and` | `heatmap` | `rect` | 1 | `vega-lite/examples/specs/selection_composition_and.vl.json` |
| `vega_lite:example:selection_heatmap` | `selection heatmap` | `rect_chart` | `rect` | 1 | `vega-lite/examples/specs/selection_heatmap.vl.json` |
| `vega_lite:example:selection_project_interval_x_y` | `selection project interval x y` | `heatmap` | `rect` | 1 | `vega-lite/examples/specs/selection_project_interval_x_y.vl.json` |
| `vega_lite:example:selection_project_multi` | `selection project multi` | `scatter_plot` | `point` | 1 | `vega-lite/examples/specs/selection_project_multi.vl.json` |
| `vega_lite:example:selection_translate_scatterplot_drag` | `selection translate scatterplot drag` | `scatter_plot` | `circle` | 1 | `vega-lite/examples/specs/selection_translate_scatterplot_drag.vl.json` |
| `vega_lite:example:selection_type_single_dblclick` | `selection type single dblclick` | `heatmap` | `rect` | 1 | `vega-lite/examples/specs/selection_type_single_dblclick.vl.json` |
| `vega_lite:example:stacked_area_normalize` | `stacked area normalize` | `area_chart` | `area` | 1 | `vega-lite/examples/specs/stacked_area_normalize.vl.json` |
| `vega_lite:example:stacked_area_stream` | `stacked area stream` | `area_chart` | `area` | 1 | `vega-lite/examples/specs/stacked_area_stream.vl.json` |
| `vega_lite:example:stacked_bar_count_corner_radius_config` | `stacked bar count corner radius config` | `histogram` | `bar` | 1 | `vega-lite/examples/specs/stacked_bar_count_corner_radius_config.vl.json` |
| `vega_lite:example:stacked_bar_weather` | `stacked bar weather` | `histogram` | `bar` | 1 | `vega-lite/examples/specs/stacked_bar_weather.vl.json` |
| `vega_lite:example:test_invalid_color_size_mark_show_only` | `test invalid color size mark show only` | `scatter_plot` | `point` | 1 | `vega-lite/examples/specs/test_invalid_color_size_mark_show_only.vl.json` |
| `vega_lite:example:text_format` | `text format` | `text_chart` | `text` | 1 | `vega-lite/examples/specs/text_format.vl.json` |
| `vega_lite:example:text_scatterplot_colored` | `text scatterplot colored` | `text_chart` | `text` | 1 | `vega-lite/examples/specs/text_scatterplot_colored.vl.json` |
| `vega_lite:example:text_tooltip_image` | `text tooltip image` | `text_chart` | `text` | 1 | `vega-lite/examples/specs/text_tooltip_image.vl.json` |
| `vega_lite:example:tick_sort` | `tick sort` | `tick_plot` | `tick` | 1 | `vega-lite/examples/specs/tick_sort.vl.json` |
| `vega_lite:example:tick_strip` | `tick strip` | `tick_plot` | `tick` | 1 | `vega-lite/examples/specs/tick_strip.vl.json` |
| `vega_lite:example:tick_strip_1D_with_height` | `tick strip 1D with height` | `tick_plot` | `tick` | 1 | `vega-lite/examples/specs/tick_strip_1D_with_height.vl.json` |
| `vega_lite:example:tick_strip_tick_band` | `tick strip tick band` | `tick_plot` | `tick` | 1 | `vega-lite/examples/specs/tick_strip_tick_band.vl.json` |
| `vega_lite:example:tick_strip_with_height` | `tick strip with height` | `tick_plot` | `tick` | 1 | `vega-lite/examples/specs/tick_strip_with_height.vl.json` |
| `vega_lite:example:tick_width_band` | `tick width band` | `tick_plot` | `tick` | 1 | `vega-lite/examples/specs/tick_width_band.vl.json` |
| `vega_lite:example:time_output_utc_scale` | `time output utc scale` | `line_chart` | `line` | 1 | `vega-lite/examples/specs/time_output_utc_scale.vl.json` |
| `vega_lite:example:window_top_k_others` | `window top k others` | `bar_chart` | `bar` | 1 | `vega-lite/examples/specs/window_top_k_others.vl.json` |
