# ViRAGE

ViRAGE строит аналитические графики по пользовательскому запросу к таблице или папке изображений. Рабочий контур проекта: подготовка данных, профиль данных, RAG-планирование, генерация Vega-Lite, проверка спецификации, построение PNG, проверка качества графика, VLM-оценка и сохранение артефактов.

## Роли моделей

В TOML указываются только переопределения. Если параметр не задан в TOML, используется значение из `ViRAGESettings`.

| Роль | Назначение |
|---|---|
| `reasoning_model` | Планирование анализа, анализ запроса, выбор подзадач, структурированные решения без изображения. |
| `spec_model` | Генерация Vega-Lite спецификаций. |
| `vlm_model` | Единая мультимодальная модель для проверки графика как изображения, решения `accept/retry`, визуального анализа PNG и итогового описания результата. |
| `visrag_embedding_model` | Эмбеддинги RAG-корпуса для Chroma. |
| `image_text_alignment_models` | Необязательные англоязычные модели для метрики `PNG + TaskText -> cosine`; задаются только при включении или переопределении этой оценки. |

## Основной запуск

```powershell
python -m compileall -q src scripts tests
pytest -q tests
```

```powershell
python scripts/rag_corpus/loading/download_sources.py --sources wilke_fundamentals from_data_to_viz ft_visual_vocabulary uk_analysis_colours uk_charts_checklist urban_institute_style_guide chartability image_quality_metrics eda_best_practices --refresh
python scripts/rag_corpus/export_guidance_chunks.py --raw-root rag_corpus/raw_external_rules --output rag_corpus/runtime/guidance_chunks.jsonl --min-chars 220 --max-chars 1000 --overlap-chars 120
python scripts/rag_corpus/build_runtime_chroma_index.py --chunks rag_corpus/runtime/guidance_chunks.jsonl --persist-dir resources/chroma/virage_guidance_chunks_nomic_embed_text_latest --collection virage_guidance_chunks_nomic_embed_text_latest --embedding-provider ollama --embedding-model nomic-embed-text:latest --embedding-base-url http://localhost:11434 --batch-size 8 --max-input-chars 1600 --recreate
```

```powershell
python scripts/benchmark/run_virage_e2e_test_cases.py --config ui/config/benchmark/project-gemma4-bench_rag.toml --suite image_folder_widefield_bpae --run-id image_folder_control --execute
python scripts/benchmark/run_virage_e2e_test_cases.py --config ui/config/benchmark/project-gemma4-bench_rag.toml --run-id smoke_cloud_control --max-cases 5 --execute
```

## Корпус знаний

Основной runtime-корпус:

```text
rag_corpus/runtime/guidance_chunks.jsonl
```

Chroma-индекс строится перед запуском режимов `semantic` и `hybrid`. AutoRAG используется как отдельный этап подбора retrieval-компонентов. Он не является обязательным runtime-шагом.

## Артефакты

Каждый запуск сохраняет:

```text
chart_plan.json
run_report.json
plot.png
plot_rendering.json
scenegraph_check.json
visrag.json
```

Статус `completed` допустим только при успешной спецификации, построенном PNG, отсутствии критических ошибок качества и согласованной смысловой оценке.

## Минимальные критерии готовности

- runtime-корпус строится без LLM-предобработки;
- Chroma manifest соответствует `guidance_chunks.jsonl`;
- план содержит строгие стратегии и семантику метрик;
- запросы поиска проблемных объектов используют `overall_severity` и top-N;
- плохой PNG не получает `completed`;
- benchmark summary согласован с `run_report.json` subrun-ов.
