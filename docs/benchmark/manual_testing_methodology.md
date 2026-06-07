# Методика ручного тестирования ViRAGE

## 1. Ручная проверка семантических метрик

| Параметр | Значение |
|---|---|
| Цель | проверить, насколько `vlm_judge_score`, `embedding_score`, `semantic_match_score` совпадают с человеческой оценкой |
| Объём | 50 графиков |
| Выборка | `bar=12`, `histogram=13`, `line=12`, `scatter=13` |
| Вход | `query`, `plot.png`, `case_id`, `run_id`, автоматические метрики |
| Шкала | `0` — не соответствует; `1` — частично соответствует; `2` — соответствует |
| Результат | CSV с ручной оценкой и отчёт корреляции/точности |

### Действия

1. Сформировать набор для ручной проверки из `benchmark_report.json`.
2. Открыть каждую строку: прочитать `query`, открыть `plot.png`.
3. Поставить `manual_score`: `0`, `1` или `2`.
4. Заполнить `manual_comment` только при ошибке или частичном соответствии.
5. Запустить evaluator ручной разметки.

### Команды

```powershell
python scripts\benchmark\runners\manual\export_semantic_review_set.py --benchmark-report artifacts\benchmarks\nlv_main_200\benchmark_report.json --output artifacts\benchmarks\manual_semantic_review\review_set.csv --limit 50
python scripts\benchmark\runners\manual\evaluate_semantic_review.py --input artifacts\benchmarks\manual_semantic_review\review_set.csv --output-dir artifacts\benchmarks\manual_semantic_review --threshold 0.7
```

---

## 2. Сравнение со сторонними проектами по `query + plot.png`

| Параметр | Значение |
|---|---|
| Проекты | `VegaChat`, `LIDA`, `Data Formulator`, `Chat2VIS`, `NL4DV` |
| Объём | 20 кейсов |
| Вход | одинаковые запросы и таблицы, итоговая картинка графика от каждого проекта |
| Основной критерий | график решает ту же аналитическую задачу |
| Метрики | `embedding_score`, `vlm_judge_score`, `semantic_match_score`, ручная `0/1/2` |

### Формат входного CSV/JSONL

| Колонка | Обязательность | Значение |
|---|---|---|
| `project` | да | имя проекта |
| `case_id` | да | идентификатор кейса |
| `query` | да | исходный запрос |
| `image_path` | да | путь к итоговому графику PNG/JPEG |
| `notes` | нет | комментарий |

### Команда

```powershell
python scripts\benchmark\runners\external\evaluate_query_image_artifacts.py --input artifacts\external_projects\query_image_artifacts.csv --config ui\config\benchmark\local_test_gemma3_4b_safe.toml --output-dir artifacts\benchmarks\external_query_image --models openai/clip-vit-base-patch32 google/siglip-so400m-patch14-384 --device cuda --dtype float16
```

---

## 3. Прикладной кейс: таблица метрик моделей

| Параметр | Значение |
|---|---|
| Вход | CSV/Excel с результатами benchmark моделей |
| Цель | проверить, строит ли ViRAGE аналитически верные графики по таблице метрик |
| Скрипт | `scripts/benchmark/runners/manual/run_applied_case_benchmark.py` |
| Статус | новый сценарий |

### Запросы

| `case_id` | Запрос | Ожидаемый смысл графика |
|---|---|---|
| `model_metrics_quality_by_chart_type` | Сравни качество моделей по типам графиков. | сравнение моделей в разрезе типа графика |
| `model_metrics_quality_tokens_pareto` | Покажи компромисс качество-токены для моделей. | Pareto/рассеяние качества и стоимости |
| `model_metrics_errors_by_chart_type` | Покажи ошибки VER и ECR по типам графиков. | техническая надёжность по стратам |

### Команда

```powershell
python scripts\benchmark\runners\manual\run_applied_case_benchmark.py --config ui\config\benchmark\local_test_gemma3_4b_safe.toml --run-id applied_model_metrics_001 --metrics-table artifacts\model_nlv\summary.csv --case-group applied_model_metrics --execute --enable-semantic-scoring --image-text-embedding-models openai/clip-vit-base-patch32 google/siglip-so400m-patch14-384 --image-text-device cuda --image-text-dtype float16
```

---

## 4. Прикладной кейс: папка изображений

| Параметр | Значение |
|---|---|
| Вход | `image_folder` |
| Цель | проверить контур `папка изображений -> таблица метрик -> аналитические графики` |
| Код | используется существующий `image_folder` контур через `run_virage_e2e_test_cases.py`/orchestrator |
| Новый runner | `scripts/benchmark/runners/manual/run_applied_case_benchmark.py` как обёртка |

### Запросы

| `case_id` | Запрос | Ожидаемый смысл графика |
|---|---|---|
| `image_folder_quality_distribution` | Покажи распределение качества изображений. | histogram по метрике качества |
| `image_folder_quality_outliers` | Найди выбросы по метрикам качества изображений. | boxplot/scatter для выбросов |
| `image_folder_group_comparison` | Сравни группы изображений по метрикам качества. | grouped bar/boxplot по группам |

### Команда

```powershell
python scripts\benchmark\runners\manual\run_applied_case_benchmark.py --config ui\config\benchmark\local_test_gemma3_4b_safe.toml --run-id applied_image_folder_001 --image-folder demo_data\denoise --case-group applied_image_folder --execute --enable-semantic-scoring --image-text-embedding-models openai/clip-vit-base-patch32 google/siglip-so400m-patch14-384 --image-text-device cuda --image-text-dtype float16
```

---

## 5. Критерии ручной оценки

| Оценка | Критерий |
|---:|---|
| `2` | график отвечает на тот же аналитический вопрос; тип графика, поля и агрегация соответствуют запросу |
| `1` | график частично полезен, но есть ошибка типа графика, агрегации, поля, сортировки или группировки |
| `0` | график не отвечает запросу, пустой, технически ошибочный или анализирует другую задачу |

## 6. Обязательные ошибки для фиксации

| Ошибка | Как фиксировать |
|---|---|
| `scatter -> bar` | `manual_score=0` или `1`, комментарий `wrong_chart_type` |
| `count -> average` | `manual_score=0` или `1`, комментарий `wrong_aggregation` |
| `histogram -> average by field` | `manual_score=0`, комментарий `wrong_task` |
| пустой график | `manual_score=0`, комментарий `empty_chart` |
| нечитаемый график | `manual_score=0` или `1`, комментарий `unreadable` |
