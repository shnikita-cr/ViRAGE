Ниже — конкретный сценарий загрузки, обработки и подготовки корпусов для нового VisRAG-стандарта: **runtime RAG хранит rule/guidance-документы, а не Vega-Lite spec**.

Команды ниже предполагают, что ты находишься в корне проекта:

```
D:\programming\projects\ViRAGE
```

---

# 0. Что получится на выходе

После полного пайплайна должны появиться:

```
rag_corpus/processed/all_rules.validated.jsonl

rag_corpus/autorag/virage_rules/corpus.parquet

rag_corpus/autorag/virage_rules/qa.parquet

rag_corpus/autorag/virage_rules/configs/virage_rules_all.yaml

rag_corpus/runtime/virage_rules.jsonl
```

`corpus.parquet` для AutoRAG должен иметь колонки `doc_id`, `contents`, `metadata`; это соответствует формату AutoRAG corpus dataset. `qa.parquet` должен содержать `retrieval_gt`, который ссылается на реальные `doc_id` из `corpus.parquet`, иначе retrieval evaluation будет некорректным. ([Marker Inc.][1])

---

# 1. Проверить, что проект запускается

Из корня ViRAGE:

```
python -Wdefault -m compileall -q src ui scripts tests

pytest -vvv
```

Если здесь есть ошибки — сначала чиним их, потом готовим корпус.

---

# 2. Установить зависимости для corpus pipeline

Минимально нужны:

```
pip install pandas pyarrow requests pydantic pyyaml
```

Если будешь запускать AutoRAG:

```
pip install AutoRAG
```

AutoRAG предназначен для автоматического перебора и оценки разных RAG-конфигураций на твоих данных. ([GitHub][2])

---

# 3. Подготовить Ollama или OpenAI

## Вариант A. Ollama

Проверь, что Ollama работает:

```
ollama list
```

Скачай модель, если ещё нет:

```
ollama pull qwen2.5-coder:7b
```

Проверь HTTP API:

```
python -c "import requests; print(requests.get('http://localhost:11434/api/tags').json().keys())"
```

## Вариант B. OpenAI

В PowerShell:

```
$env:OPENAI_API_KEY="твой_ключ"
```

Проверка:

```
python -c "import os; print(bool(os.getenv('OPENAI_API_KEY')))"
```

---

# 4. Очистить старые результаты, если они остались

Не удаляй `rag_corpus/raw/manual_rules`, если там уже есть новые seed rules.

Удалить можно:

```
Remove-Item -Recurse -Force rag_corpus\extracted -ErrorAction SilentlyContinue

Remove-Item -Recurse -Force rag_corpus\processed\llm_normalized -ErrorAction SilentlyContinue

Remove-Item -Force rag_corpus\processed\all_rules.jsonl -ErrorAction SilentlyContinue

Remove-Item -Force rag_corpus\processed\all_rules.deduped.jsonl -ErrorAction SilentlyContinue

Remove-Item -Force rag_corpus\processed\all_rules.validated.jsonl -ErrorAction SilentlyContinue

Remove-Item -Force rag_corpus\processed\normalization_failures.jsonl -ErrorAction SilentlyContinue

Remove-Item -Recurse -Force rag_corpus\autorag\virage_rules -ErrorAction SilentlyContinue

Remove-Item -Recurse -Force rag_corpus\runtime -ErrorAction SilentlyContinue
```

Заново создать базовые папки:

```
New-Item -ItemType Directory -Force rag_corpus\extracted

New-Item -ItemType Directory -Force rag_corpus\processed\llm_normalized

New-Item -ItemType Directory -Force rag_corpus\autorag\virage_rules

New-Item -ItemType Directory -Force rag_corpus\runtime
```

---

# 5. Загрузить внешние источники

## 5.1. Vega-Lite examples

Официальная галерея Vega-Lite содержит спецификации разных типов графиков; в ViRAGE они используются только как сырьё для извлечения паттернов, а не как runtime spec templates. ([Vega][3])

Команды:

```
git clone https://github.com/vega/vega-lite.git rag_corpus\raw\vega_lite_examples\vega-lite
```

Проверить:

```
dir rag_corpus\raw\vega_lite_examples\vega-lite\examples\specs
```

---

## 5.2. ChartSquared / C-2

C² полезен как источник feedback/readability/VLM-readability идей: проект описывает reference-free automatic feedback для LLM-based chart generation; GitHub-репозиторий содержит ChartUIE-8K и prompts/criteria. ([chartsquared.github.io][4])

Команда:

```
git clone https://github.com/chartsquared/C-2.git rag_corpus\raw\chartsquared\C-2
```

Проверить:

```
dir rag_corpus\raw\chartsquared\C-2
```

---

## 5.3. NLV Corpus

NLV Corpus полезен для пользовательских формулировок запросов к визуализациям; на сайте указано, что датасеты и Vega-Lite specifications доступны в GitHub repo. ([nlvcorpus.github.io][5])

Команда:

```
git clone https://github.com/nlvcorpus/nlvcorpus.github.io.git rag_corpus\raw\nlv\nlvcorpus.github.io
```

Проверить:

```
dir rag_corpus\raw\nlv\nlvcorpus.github.io
```

На текущей версии скриптов NLV extractor ещё не подключён к `run_prepare_corpus.py`, поэтому этот источник пока будет лежать как raw-заготовка. Его можно подключить следующей итерацией.

---

## 5.4. ChartLLM / VL2NL

ChartLLM полезен как источник связки Vega-Lite спецификаций и естественно-языковых формулировок. ([GitHub][6])

Команда:

```
git clone https://github.com/hyungkwonko/chart-llm.git rag_corpus\raw\chartllm\chart-llm
```

Проверить:

```
dir rag_corpus\raw\chartllm\chart-llm
```

Как и NLV, в текущем наборе скриптов это пока raw-заготовка для следующего extractor-а.

---

## 5.5. nvBench

nvBench — крупный NL2VIS benchmark: репозиторий описывает 25 750 NL/VIS пар, 105 доменов и Vega-Lite формат. ([GitHub][7])

Команда:

```
git clone https://github.com/TsinghuaDatabaseGroup/nvBench.git rag_corpus\raw\nvbench\nvBench
```

Проверить:

```
dir rag_corpus\raw\nvbench\nvBench
```

Сейчас extractor для nvBench ещё не подключён; источник нужно положить сейчас, а обработку добавить отдельной итерацией.

---

# 6. Перенести feedback ViRAGE в raw-зону

Если у тебя есть старые feedback-файлы, перенеси их в новую структуру:

```
New-Item -ItemType Directory -Force rag_corpus\raw\virage_feedback
```

Пример, если файлы лежат в старом месте:

```
Copy-Item rag_corpus\feedback\visual_feedback.jsonl rag_corpus\raw\virage_feedback\visual_feedback.jsonl -ErrorAction SilentlyContinue

Copy-Item rag_corpus\feedback\visual_feedback_nlv.jsonl rag_corpus\raw\virage_feedback\visual_feedback_nlv.jsonl -ErrorAction SilentlyContinue
```

Если старой папки уже нет — ничего делать не нужно.

---

# 7. Проверить raw-источники

```
python scripts/rag_corpus/sources/scan_sources.py
```

После этого должны появиться:

```
rag_corpus/manifests/raw_inventory.json

rag_corpus/manifests/source_inventory.md
```

Открой отчёт:

```
notepad rag_corpus\manifests\source_inventory.md
```

---

# 8. Сделать полный prepare corpus через Ollama

Это основной шаг: extractors создают `rag_corpus/extracted/*.jsonl`, затем LLM нормализует записи в rule/guidance records.

```
python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --clean-processed
```

Если Ollama не на стандартном порту:

```
python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --base-url http://localhost:11434 --clean-processed
```

Для быстрого smoke-теста:

```
python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --limit 10 --clean-processed
```

Ожидаемые файлы:

```
rag_corpus/extracted/manual_rules.jsonl

rag_corpus/extracted/virage_feedback.jsonl

rag_corpus/extracted/chartsquared.jsonl

rag_corpus/extracted/vega_lite_examples.jsonl

rag_corpus/processed/llm_normalized/*.jsonl

rag_corpus/processed/all_rules.jsonl

rag_corpus/processed/all_rules.deduped.jsonl

rag_corpus/processed/all_rules.validated.jsonl

rag_corpus/processed/processing_report.json

rag_corpus/processed/processing_report.md
```

---

# 9. Если LLM-normalization оборвалась

Продолжить:

```
python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --resume
```

Повторить только упавшие записи:

```
python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --retry-failed
```

Ограничить конкретными типами документов:

```
python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --record-types chart_pattern readability_rule scale_plot_area_rule vlm_readability_rule
```

Для доменных правил тоже:

```
python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --record-types domain_semantics_rule
```

---

# 10. Альтернатива: prepare corpus через OpenAI

```
python scripts/rag_corpus/run_prepare_corpus.py --provider openai --model gpt-4.1-mini --clean-processed
```

С кастомным endpoint:

```
python scripts/rag_corpus/run_prepare_corpus.py --provider openai --model gpt-4.1-mini --base-url https://api.openai.com/v1 --clean-processed
```

---

# 11. Проверить качество processed-корпуса

Открыть отчёт:

```
notepad rag_corpus\processed\processing_report.md
```

Посмотреть первые записи:

```
python -c "import json,itertools; f=open('rag_corpus/processed/all_rules.validated.jsonl',encoding='utf-8'); [print(json.dumps(json.loads(x),ensure_ascii=False,indent=2)[:1200]) for x in itertools.islice(f,3)]"
```

Проверить, что в processed-записях **нет** Vega-Lite spec-полей:

```
python -c "from pathlib import Path; text=Path('rag_corpus/processed/all_rules.validated.jsonl').read_text(encoding='utf-8'); print(any(x in text for x in ['\"mark\"','\"encoding\"','\"$schema\"','\"spec_template\"']))"
```

Ожидаемый вывод:

```
False
```

---

# 12. Экспортировать корпус для AutoRAG

```
python scripts/rag_corpus/run_export_autorag.py
```

Ожидаемые файлы:

```
rag_corpus/autorag/virage_rules/corpus.parquet

rag_corpus/autorag/virage_rules/qa.parquet

rag_corpus/autorag/virage_rules/configs/virage_rules_all.yaml
```

Проверить parquet:

```
python -c "import pandas as pd; print(pd.read_parquet('rag_corpus/autorag/virage_rules/corpus.parquet').head()); print(pd.read_parquet('rag_corpus/autorag/virage_rules/qa.parquet').head())"
```

---

# 13. Запустить AutoRAG optimization

Сначала dry-run, чтобы увидеть команду:

```
python scripts/rag_corpus/run_autorag_optimization.py --dry-run
```

Потом запуск:

```
python scripts/rag_corpus/run_autorag_optimization.py
```

Важно: текущий скрипт вызывает:

```
python -m autorag.cli --config ...
```

Если установленная версия AutoRAG использует другую CLI-команду, запуск может потребовать корректировки под твою версию. Сам конфиг лежит здесь:

```
rag_corpus/autorag/virage_rules/configs/virage_rules_all.yaml
```

После завершения собрать результаты:

```
python scripts/rag_corpus/autorag/collect_results.py
```

Ожидаемые отчёты:

```
rag_corpus/autorag/virage_rules/reports/retrieval_comparison.csv

rag_corpus/autorag/virage_rules/reports/retrieval_comparison.json

rag_corpus/autorag/virage_rules/reports/retrieval_comparison.md
```

---

# 14. Экспортировать runtime-корпус для ViRAGE

```
python scripts/rag_corpus/run_export_runtime.py
```

Ожидаемый файл:

```
rag_corpus/runtime/virage_rules.jsonl
```

Проверить:

```
python -c "import json,itertools; f=open('rag_corpus/runtime/virage_rules.jsonl',encoding='utf-8'); [print(json.dumps(json.loads(x),ensure_ascii=False,indent=2)[:1000]) for x in itertools.islice(f,3)]"
```

---

# 15. Проверить, что runtime VisRAG видит новый корпус

Запусти один обычный сценарий ViRAGE или notebook. Если нужно быстро проверить файл:

```
python -c "from pathlib import Path; p=Path('rag_corpus/runtime/virage_rules.jsonl'); print(p.exists(), p.stat().st_size)"
```

Если файл есть и не пустой, `VisRAGService` сможет читать runtime rules через JSONL backend.

---

# 16. Минимальный полный сценарий одной командной последовательностью

Для первого smoke-прогона:

```
python -Wdefault -m compileall -q src ui scripts tests

pytest -vvv

python scripts/rag_corpus/sources/scan_sources.py

python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --limit 10 --clean-processed

python scripts/rag_corpus/run_export_autorag.py

python scripts/rag_corpus/run_export_runtime.py

python -c "from pathlib import Path; print(Path('rag_corpus/runtime/virage_rules.jsonl').exists())"
```

Для полного прогона:

```
python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --clean-processed

python scripts/rag_corpus/run_export_autorag.py

python scripts/rag_corpus/run_autorag_optimization.py

python scripts/rag_corpus/autorag/collect_results.py

python scripts/rag_corpus/run_export_runtime.py
```

---

# 17. Что сейчас реально обрабатывается скриптами

В текущей версии pipeline автоматически обрабатывает:

```
rag_corpus/raw/manual_rules

rag_corpus/raw/virage_feedback

rag_corpus/raw/chartsquared

rag_corpus/raw/vega_lite_examples
```

Эти источники извлекаются в:

```
rag_corpus/extracted/*.jsonl
```

Затем LLM-normalization превращает их в:

```
chart_pattern
readability_rule
scale_plot_area_rule
vlm_readability_rule
domain_semantics_rule
```

Источники `nlv`, `chartllm`, `nvbench` сейчас лучше уже скачать и положить в `raw/`, но их extractors нужно добавить отдельной итерацией. Это нормально: мы не смешиваем загрузку источников и интеграцию каждого датасета в один большой небезопасный скрипт.

---

# 18. Как понять, что корпус готов

Готовность минимального корпуса:

* `rag_corpus/processed/all_rules.validated.jsonl` существует;
* `rag_corpus/runtime/virage_rules.jsonl` существует;
* в runtime JSONL нет `mark`, `encoding`, `$schema`, `spec_template`;
* есть записи минимум четырёх типов:

  * `chart_pattern`;
  * `readability_rule`;
  * `scale_plot_area_rule`;
  * `vlm_readability_rule`;
* AutoRAG export создаёт `corpus.parquet` и `qa.parquet`;
* обычный ViRAGE-запуск показывает, что `VisRAGService` возвращает `generation_guidance.prompt_text`.

[1]: https://marker-inc-korea.github.io/AutoRAG/data_creation/data_format.html?utm_source=chatgpt.com "Dataset Format - AutoRAG documentation"
[2]: https://github.com/Marker-Inc-Korea/AutoRAG?utm_source=chatgpt.com "AutoRAG: An Open-Source Framework for Retrieval ..."
[3]: https://vega.github.io/vega-lite/examples/?utm_source=chatgpt.com "Example Gallery | Vega-Lite"
[4]: https://chartsquared.github.io/?utm_source=chatgpt.com "Scalable Auto-Feedback for LLM-based Chart Generation: C²"
[5]: https://nlvcorpus.github.io/?utm_source=chatgpt.com "NLV: Natural Language Utterances for Specifying Data ..."
[6]: https://github.com/hyungkwonko/chart-llm?utm_source=chatgpt.com "hyungkwonko/chart-llm: Vega-Lite Chart Dataset and NL ..."
[7]: https://github.com/TsinghuaDatabaseGroup/nvBench?utm_source=chatgpt.com "TsinghuaDatabaseGroup/nvBench"
