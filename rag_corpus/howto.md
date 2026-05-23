# ViRAGE: внешний корпус правил, RAG и benchmark

Команды запускать из корня проекта:

    D:\programming\projects\ViRAGE

NLV используется только для оценки качества. В RAG-корпус NLV, его запросы, эталонные спецификации и правила, извлечённые из NLV, не добавляются.

Проверка проекта

    python -Wdefault -m compileall -q src ui scripts tests
    pytest -q

Установка зависимостей

    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    pip install pandas pyarrow requests pydantic pyyaml

Если нужен AutoRAG:

    pip install AutoRAG

Проверка Ollama

    ollama list
    python -c "import requests; print(requests.get('http://localhost:11434/api/tags').json().keys())"

OpenAI вместо Ollama

    $env:OPENAI_API_KEY="твой_ключ"
    python -c "import os; print(bool(os.getenv('OPENAI_API_KEY')))"

Скачать источники для внешнего корпуса правил

    New-Item -ItemType Directory -Force rag_corpus\raw_external_rules

    git clone https://github.com/ShenLeixian/TaskVis.git rag_corpus\raw_external_rules\taskvis
    Test-Path rag_corpus\raw_external_rules\taskvis

    git clone https://github.com/uwdata/draco.git rag_corpus\raw_external_rules\draco
    Test-Path rag_corpus\raw_external_rules\draco

    git clone https://github.com/holtzy/data_to_viz.git rag_corpus\raw_external_rules\from_data_to_viz
    Test-Path rag_corpus\raw_external_rules\from_data_to_viz

    git clone https://github.com/Financial-Times/chart-doctor.git rag_corpus\raw_external_rules\ft_visual_vocabulary
    Test-Path rag_corpus\raw_external_rules\ft_visual_vocabulary

    git clone https://github.com/vega/compassql.git rag_corpus\raw_external_rules\compassql
    Test-Path rag_corpus\raw_external_rules\compassql

    git clone https://github.com/chartsquared/C-2.git rag_corpus\raw_external_rules\chartsquared
    Test-Path rag_corpus\raw_external_rules\chartsquared

Скачать NLV только для benchmark

    New-Item -ItemType Directory -Force datasets
    git clone https://github.com/giahy2507/nlvcorpus.github.io.git .\datasets\nlv_corpus
    Invoke-WebRequest "https://docs.google.com/spreadsheets/d/1GMWktNGJCwC8U1dvT0gMggVRRYqN3uL28zjVDbxYJOg/export?format=csv&gid=0" -OutFile .\datasets\nlv_corpus\NLV_Corpus.csv
    Test-Path .\datasets\nlv_corpus\NLV_Corpus.csv
    Test-Path .\datasets\nlv_corpus\vlSpecs.json
    Test-Path .\datasets\nlv_corpus\datasets

InfiAgent должен лежать здесь

    Datasets\InfiAgent

Проверка InfiAgent

    python scripts\benchmark\infiagent_scan.py --source-root Datasets\InfiAgent

Очистить сгенерированные RAG-результаты

    Remove-Item -Recurse -Force rag_corpus\extracted -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force rag_corpus\processed\llm_normalized -ErrorAction SilentlyContinue
    Remove-Item -Force rag_corpus\processed\all_rules.jsonl -ErrorAction SilentlyContinue
    Remove-Item -Force rag_corpus\processed\all_rules.deduped.jsonl -ErrorAction SilentlyContinue
    Remove-Item -Force rag_corpus\processed\all_rules.validated.jsonl -ErrorAction SilentlyContinue
    Remove-Item -Force rag_corpus\processed\normalization_failures.jsonl -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force rag_corpus\runtime -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force rag_corpus\autorag\virage_rules -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force rag_corpus\autorag\runs\semantic_rules_eval -ErrorAction SilentlyContinue

    New-Item -ItemType Directory -Force rag_corpus\extracted
    New-Item -ItemType Directory -Force rag_corpus\processed\llm_normalized
    New-Item -ItemType Directory -Force rag_corpus\runtime
    New-Item -ItemType Directory -Force rag_corpus\autorag\virage_rules

Извлечь внешние источники без LLM-нормализации

    python scripts\rag_corpus\sources\extract_taskvis.py
    python scripts\rag_corpus\sources\extract_draco.py
    python scripts\rag_corpus\sources\extract_from_data_to_viz.py
    python scripts\rag_corpus\sources\extract_ft_visual_vocabulary.py
    python scripts\rag_corpus\sources\extract_compassql.py
    python scripts\rag_corpus\sources\extract_chartsquared_rules.py

Проверить количество извлечённых записей

    python -c "from pathlib import Path; [print(p.name, sum(1 for _ in p.open(encoding='utf-8'))) for p in Path('rag_corpus/extracted').glob('*.jsonl')]"

Ожидаемо должны быть файлы:

    rag_corpus\extracted\taskvis.jsonl
    rag_corpus\extracted\draco.jsonl
    rag_corpus\extracted\from_data_to_viz.jsonl
    rag_corpus\extracted\ft_visual_vocabulary.jsonl
    rag_corpus\extracted\compassql.jsonl
    rag_corpus\extracted\chartsquared_rules.jsonl

Подготовить внешний корпус правил через Ollama

    python scripts\rag_corpus\run_prepare_corpus.py --provider ollama --model gemma4:31b-cloud --sources taskvis draco from_data_to_viz ft_visual_vocabulary compassql chartsquared_rules --clean-processed

Подготовить внешний корпус без ChartSquared

    python scripts\rag_corpus\run_prepare_corpus.py --provider ollama --model gemma4:31b-cloud --sources taskvis draco from_data_to_viz ft_visual_vocabulary compassql --clean-processed

Продолжить после обрыва

    python scripts\rag_corpus\run_prepare_corpus.py --provider ollama --model gemma4:31b-cloud --resume

Повторить только упавшие записи

    python scripts\rag_corpus\run_prepare_corpus.py --provider ollama --model gemma4:31b-cloud --retry-failed

Проверить обработанный корпус

    Test-Path rag_corpus\processed\all_rules.deduped.jsonl
    Test-Path rag_corpus\processed\all_rules.validated.jsonl
    notepad rag_corpus\processed\processing_report.md

Проверить состав по источникам

    python -c "import json,collections; c=collections.Counter(); f=open('rag_corpus/processed/all_rules.deduped.jsonl',encoding='utf-8'); [c.update([json.loads(x).get('source_dataset') or json.loads(x).get('source') or json.loads(x).get('metadata',{}).get('source_dataset')]) for x in f]; print(c)"

Проверить состав по типам записей

    python -c "import json,collections; c=collections.Counter(); f=open('rag_corpus/processed/all_rules.deduped.jsonl',encoding='utf-8'); [c.update([json.loads(x).get('record_type')]) for x in f]; print(c)"

Проверить, что NLV не попал в корпус

    python -c "from pathlib import Path; text=Path('rag_corpus/processed/all_rules.deduped.jsonl').read_text(encoding='utf-8').lower(); print('nlv' in text, 'nlv_corpus' in text)"

Экспорт runtime-корпуса

    python scripts\rag_corpus\run_export_runtime.py
    Test-Path rag_corpus\runtime\virage_rules.jsonl

Проверить runtime-корпус

    python -c "from pathlib import Path; p=Path('rag_corpus/runtime/virage_rules.jsonl'); print(p.exists(), p.stat().st_size if p.exists() else 0)"

Экспорт для AutoRAG

    python scripts\rag_corpus\run_export_autorag.py
    Test-Path rag_corpus\autorag\virage_rules\corpus.parquet
    Test-Path rag_corpus\autorag\virage_rules\qa.parquet
    Test-Path rag_corpus\autorag\virage_rules\configs\virage_rules_all.yaml

Посмотреть parquet

    python -c "import pandas as pd; pd.set_option('display.max_columns', None); pd.set_option('display.max_colwidth', 300); df=pd.read_parquet('rag_corpus/autorag/virage_rules/corpus.parquet'); print(df.head(10).to_string(index=False)); print(df.shape)"
    python -c "import pandas as pd; pd.set_option('display.max_columns', None); pd.set_option('display.max_colwidth', 300); df=pd.read_parquet('rag_corpus/autorag/virage_rules/qa.parquet'); print(df.head(10).to_string(index=False)); print(df.shape)"

AutoRAG напрямую

    New-Item -ItemType Directory -Force rag_corpus\autorag\runs\semantic_rules_eval

    autorag evaluate --config rag_corpus\autorag\virage_rules\configs\virage_rules_all.yaml --qa_data_path rag_corpus\autorag\virage_rules\qa.parquet --corpus_data_path rag_corpus\autorag\virage_rules\corpus.parquet --project_dir rag_corpus\autorag\runs\semantic_rules_eval

Если команда autorag недоступна

    python -m autorag.cli evaluate --config rag_corpus\autorag\virage_rules\configs\virage_rules_all.yaml --qa_data_path rag_corpus\autorag\virage_rules\qa.parquet --corpus_data_path rag_corpus\autorag\virage_rules\corpus.parquet --project_dir rag_corpus\autorag\runs\semantic_rules_eval

Оценить runtime-извлекатели ViRAGE

    python scripts\rag_corpus\run_evaluate_runtime_retrievers.py --backends keyword bm25 tfidf --top-k 1 2 3 5 8

Применить рекомендованную настройку RAG

    python scripts\rag_corpus\apply_runtime_retrieval_config.py --base-config ui\config\project-gemma4-bench_rag.toml --output-config ui\config\project-gemma4-bench_rag_autorag.toml
    Test-Path ui\config\project-gemma4-bench_rag_autorag.toml

Jupyter-проверка RAG вручную

    New-Item -ItemType Directory -Force notebooks
    jupyter notebook notebooks\manual_rag_check.ipynb

В блокноте вручную проверить начало цепочки: загрузить config, создать VisRAGService, вызвать retrieve_generation_guidance для собственного запроса и посмотреть retrieved rules. Длинный код проверки держать в отдельном notebook, не в этом howto.

NLV без RAG, smoke

    python scripts\benchmark\run_vegachat_compatible_benchmark.py --cases .\datasets\nlv_corpus\ --config ui\config\project-gemma4-bench_norag.toml --output-dir artifacts\benchmarks\nlv_no_rag_smoke --limit 20 --disable-analytics-tail

NLV без RAG, полный запуск

    python scripts\benchmark\run_vegachat_compatible_benchmark.py --cases .\datasets\nlv_corpus\ --config ui\config\project-gemma4-bench_norag.toml --output-dir artifacts\benchmarks\nlv_no_rag --disable-analytics-tail

Текущий RAG до нового корпуса, если нужно сохранить baseline

    python scripts\benchmark\run_vegachat_compatible_benchmark.py --cases .\datasets\nlv_corpus\ --config ui\config\project-gemma4-bench_rag.toml --output-dir artifacts\benchmarks\nlv_rag_current --disable-analytics-tail

Новый внешний корпус без AutoRAG-настройки

    python scripts\benchmark\run_vegachat_compatible_benchmark.py --cases .\datasets\nlv_corpus\ --config ui\config\project-gemma4-bench_rag.toml --output-dir artifacts\benchmarks\nlv_rag_semantic_rules --disable-analytics-tail

Новый внешний корпус после AutoRAG-настройки

    python scripts\benchmark\run_vegachat_compatible_benchmark.py --cases .\datasets\nlv_corpus\ --config ui\config\project-gemma4-bench_rag_autorag.toml --output-dir artifacts\benchmarks\nlv_rag_semantic_rules_autorag --disable-analytics-tail

Сравнить no RAG и новый RAG

    python scripts\benchmark\compare_runs.py --left artifacts\benchmarks\nlv_no_rag --right artifacts\benchmarks\nlv_rag_semantic_rules_autorag --output artifacts\benchmarks\nlv_compare_no_rag_vs_semantic_rag_autorag

Красивая таблица NLV-метрик

    python scripts\benchmark\nlv_compare.py --no-rag-report artifacts\benchmarks\nlv_no_rag\benchmark_report.json --rag-report artifacts\benchmarks\nlv_rag_semantic_rules_autorag\benchmark_report.json --output-dir artifacts\benchmarks\nlv_compare_metrics

InfiAgent, smoke

    python scripts\benchmark\run_infiagent_chart_grounded.py --source-root Datasets\InfiAgent --config ui\config\project-gemma4-bench_rag_autorag.toml --output-dir artifacts\benchmarks\infiagent_semantic_rag_autorag_3 --limit 3

InfiAgent, 20 кейсов

    python scripts\benchmark\run_infiagent_chart_grounded.py --source-root Datasets\InfiAgent --config ui\config\project-gemma4-bench_rag_autorag.toml --output-dir artifacts\benchmarks\infiagent_semantic_rag_autorag_20 --limit 20

InfiAgent, полный запуск

    python scripts\benchmark\run_infiagent_chart_grounded.py --source-root Datasets\InfiAgent --config ui\config\project-gemma4-bench_rag_autorag.toml --output-dir artifacts\benchmarks\infiagent_semantic_rag_autorag

InfiAgent, пересобрать отчёт

    python scripts\benchmark\evaluate_infiagent_results.py --output-dir artifacts\benchmarks\infiagent_semantic_rag_autorag

InfiAgent, инспекция кейса

    python scripts\benchmark\inspect_infiagent_case.py --output-dir artifacts\benchmarks\infiagent_semantic_rag_autorag --case-id infiagent_0

Проверка нужных скриптов

    Test-Path scripts\rag_corpus\sources\extract_taskvis.py
    Test-Path scripts\rag_corpus\sources\extract_draco.py
    Test-Path scripts\rag_corpus\sources\extract_from_data_to_viz.py
    Test-Path scripts\rag_corpus\sources\extract_ft_visual_vocabulary.py
    Test-Path scripts\rag_corpus\sources\extract_compassql.py
    Test-Path scripts\rag_corpus\sources\extract_chartsquared_rules.py
    Test-Path scripts\rag_corpus\run_prepare_corpus.py
    Test-Path scripts\rag_corpus\run_export_runtime.py
    Test-Path scripts\rag_corpus\run_export_autorag.py
    Test-Path scripts\rag_corpus\run_evaluate_runtime_retrievers.py
    Test-Path scripts\rag_corpus\apply_runtime_retrieval_config.py
    Test-Path scripts\benchmark\run_vegachat_compatible_benchmark.py
    Test-Path scripts\benchmark\run_infiagent_chart_grounded.py
    Test-Path scripts\benchmark\compare_runs.py
    Test-Path scripts\benchmark\nlv_compare.py

Что должно быть готово перед финальным benchmark

    Test-Path .\datasets\nlv_corpus\NLV_Corpus.csv
    Test-Path .\datasets\nlv_corpus\vlSpecs.json
    Test-Path .\datasets\nlv_corpus\datasets
    Test-Path rag_corpus\runtime\virage_rules.jsonl
    Test-Path ui\config\project-gemma4-bench_rag_autorag.toml
