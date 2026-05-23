# ViRAGE: RAG-корпус и benchmark

Команды запускать из корня проекта:

    C:\Users\Nikita\projects\ViRAGE

Прирост от RAG измеряется не по самому корпусу, а по итоговому качеству построения графиков. Основное сравнение: один и тот же NLV benchmark без RAG и с RAG. При изменении/дополнении корпуса заново собирается `rag_corpus\runtime\virage_rules.jsonl`, затем повторяется тот же benchmark. Матрица по корпусам не нужна.

Основные метрики: `mean_spec_score`, `mean_vision_score`, `median_spec_score`, `median_vision_score`, `visualization_error_rate`, `empty_chart_rate`, `mean_duration_seconds`, `total_tokens`.

Техническая проверка проекта

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
    ollama pull qwen2.5-coder:7b
    python -c "import requests; print(requests.get('http://localhost:11434/api/tags').json().keys())"

OpenAI вместо Ollama

    $env:OPENAI_API_KEY="твой_ключ"
    python -c "import os; print(bool(os.getenv('OPENAI_API_KEY')))"

Скачать внешние источники без лишних вложенных папок

    New-Item -ItemType Directory -Force rag_corpus\raw

    git clone https://github.com/vega/vega-lite.git rag_corpus\raw\vega_lite_examples
    Test-Path rag_corpus\raw\vega_lite_examples\examples\specs

    git clone https://github.com/chartsquared/C-2.git rag_corpus\raw\chartsquared
    Test-Path rag_corpus\raw\chartsquared

    New-Item -ItemType Directory -Force datasets
    git clone https://github.com/giahy2507/nlvcorpus.github.io.git .\datasets\nlv_corpus
    Invoke-WebRequest "https://docs.google.com/spreadsheets/d/1GMWktNGJCwC8U1dvT0gMggVRRYqN3uL28zjVDbxYJOg/export?format=csv&gid=0" -OutFile .\datasets\nlv_corpus\NLV_Corpus.csv
    Test-Path .\datasets\nlv_corpus\NLV_Corpus.csv
    Test-Path .\datasets\nlv_corpus\vlSpecs.json
    Test-Path .\datasets\nlv_corpus\datasets

InfiAgent должен лежать здесь:

    Datasets\InfiAgent

Проверка InfiAgent:

    python scripts/benchmark/infiagent_scan.py --source-root Datasets\InfiAgent

Нужные файлы InfiAgent:

    Datasets\InfiAgent\examples\DA-Agent\data\da-dev-questions.jsonl
    Datasets\InfiAgent\examples\DA-Agent\data\da-dev-labels.jsonl
    Datasets\InfiAgent\examples\DA-Agent\data\da-dev-tables\

Очистить сгенерированные RAG-результаты

    Remove-Item -Recurse -Force rag_corpus\extracted -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force rag_corpus\processed\llm_normalized -ErrorAction SilentlyContinue
    Remove-Item -Force rag_corpus\processed\all_rules.jsonl -ErrorAction SilentlyContinue
    Remove-Item -Force rag_corpus\processed\all_rules.deduped.jsonl -ErrorAction SilentlyContinue
    Remove-Item -Force rag_corpus\processed\all_rules.validated.jsonl -ErrorAction SilentlyContinue
    Remove-Item -Force rag_corpus\processed\normalization_failures.jsonl -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force rag_corpus\runtime -ErrorAction SilentlyContinue

    New-Item -ItemType Directory -Force rag_corpus\extracted
    New-Item -ItemType Directory -Force rag_corpus\processed\llm_normalized
    New-Item -ItemType Directory -Force rag_corpus\runtime

Проверить raw-источники

    python scripts/rag_corpus/sources/scan_sources.py
    notepad rag_corpus\manifests\source_inventory.md

Smoke-подготовка RAG-корпуса

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --limit 10 --sources manual_rules virage_feedback chartsquared vega_lite_examples --chartsquared-mode prompts_only --clean-processed

Полная подготовка RAG-корпуса

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --sources manual_rules virage_feedback chartsquared vega_lite_examples --chartsquared-mode prompts_only --clean-processed

Продолжить после обрыва

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --resume

Повторить только упавшие записи

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --retry-failed

Экспорт runtime-корпуса

    python scripts/rag_corpus/run_export_runtime.py
    Test-Path rag_corpus\runtime\virage_rules.jsonl

Проверить runtime-корпус

    python -c "from pathlib import Path; p=Path('rag_corpus/runtime/virage_rules.jsonl'); print(p.exists(), p.stat().st_size if p.exists() else 0)"

NLV без RAG, smoke

    python scripts/benchmark/run_vegachat_compatible_benchmark.py --cases .\datasets\nlv_corpus\ --config ui\config\project-gemma4-bench_norag.toml --output-dir artifacts\benchmarks\nlv_no_rag_smoke --limit 20

NLV без RAG, полный запуск

    python scripts/benchmark/run_vegachat_compatible_benchmark.py --cases .\datasets\nlv_corpus\ --config ui\config\project-gemma4-bench_norag.toml --output-dir artifacts\benchmarks\nlv_no_rag

NLV без RAG, продолжить

    python scripts/benchmark/run_vegachat_compatible_benchmark.py --cases .\datasets\nlv_corpus\ --config ui\config\project-gemma4-bench_norag.toml --output-dir artifacts\benchmarks\nlv_no_rag --resume

NLV без RAG, перезапустить failed/missing

    python scripts/benchmark/run_vegachat_compatible_benchmark.py --cases .\datasets\nlv_corpus\ --config ui\config\project-gemma4-bench_norag.toml --output-dir artifacts\benchmarks\nlv_no_rag --retry-failed

NLV с RAG, smoke

    python scripts/benchmark/run_vegachat_compatible_benchmark.py --cases .\datasets\nlv_corpus\ --config ui\config\project-gemma4-bench_rag.toml --output-dir artifacts\benchmarks\nlv_rag_smoke --limit 20

NLV с RAG, полный запуск

    python scripts/benchmark/run_vegachat_compatible_benchmark.py --cases .\datasets\nlv_corpus\ --config ui\config\project-gemma4-bench_rag.toml --output-dir artifacts\benchmarks\nlv_rag

NLV с RAG, продолжить

    python scripts/benchmark/run_vegachat_compatible_benchmark.py --cases .\datasets\nlv_corpus\ --config ui\config\project-gemma4-bench_rag.toml --output-dir artifacts\benchmarks\nlv_rag --resume

NLV с RAG, перезапустить failed/missing

    python scripts/benchmark/run_vegachat_compatible_benchmark.py --cases .\datasets\nlv_corpus\ --config ui\config\project-gemma4-bench_rag.toml --output-dir artifacts\benchmarks\nlv_rag --retry-failed

Сравнить NLV no RAG и RAG

    @'
    import json
    from pathlib import Path

    no_rag = json.loads(Path('artifacts/benchmarks/nlv_no_rag/benchmark_report.json').read_text(encoding='utf-8'))
    rag = json.loads(Path('artifacts/benchmarks/nlv_rag/benchmark_report.json').read_text(encoding='utf-8'))

    metrics = [
        'total_cases',
        'successful_cases',
        'failed_cases',
        'visualization_error_rate',
        'empty_chart_rate',
        'mean_spec_score',
        'mean_vision_score',
        'median_spec_score',
        'median_vision_score',
        'mean_duration_seconds',
        'total_tokens',
    ]

    print('metric,no_rag,rag,delta_rag_minus_no_rag')
    for metric in metrics:
        a = no_rag.get(metric)
        b = rag.get(metric)
        delta = None if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) else round(b - a, 6)
        print(f'{metric},{a},{b},{delta}')
    '@ | python -

InfiAgent chart-answerable, smoke

    python scripts/benchmark/run_infiagent_chart_grounded.py --source-root Datasets\InfiAgent --config ui\config\project-gemma4-bench_rag.toml --output-dir artifacts\benchmarks\infiagent_chartable_rag_smoke --limit 3

InfiAgent chart-answerable, 20 кейсов

    python scripts/benchmark/run_infiagent_chart_grounded.py --source-root Datasets\InfiAgent --config ui\config\project-gemma4-bench_rag.toml --output-dir artifacts\benchmarks\infiagent_chartable_rag_20 --limit 20

InfiAgent chart-answerable, полный запуск

    python scripts/benchmark/run_infiagent_chart_grounded.py --source-root Datasets\InfiAgent --config ui\config\project-gemma4-bench_rag.toml --output-dir artifacts\benchmarks\infiagent_chartable_rag

InfiAgent chart-answerable, продолжить

    python scripts/benchmark/run_infiagent_chart_grounded.py --source-root Datasets\InfiAgent --config ui\config\project-gemma4-bench_rag.toml --output-dir artifacts\benchmarks\infiagent_chartable_rag --resume

InfiAgent chart-answerable, перезапустить failed/missing

    python scripts/benchmark/run_infiagent_chart_grounded.py --source-root Datasets\InfiAgent --config ui\config\project-gemma4-bench_rag.toml --output-dir artifacts\benchmarks\infiagent_chartable_rag --retry-failed

InfiAgent, инспекция кейса

    python scripts/benchmark/inspect_infiagent_case.py --output-dir artifacts\benchmarks\infiagent_chartable_rag --case-id infiagent_0

InfiAgent, пересобрать отчёт

    python scripts/benchmark/evaluate_infiagent_results.py --output-dir artifacts\benchmarks\infiagent_chartable_rag

AutoRAG, опционально

    python scripts/rag_corpus/run_export_autorag.py
    python scripts/rag_corpus/run_autorag_optimization.py --dry-run
    python scripts/rag_corpus/run_autorag_optimization.py
    python scripts/rag_corpus/autorag/collect_results.py

Проверка, что нужные скрипты и конфиги есть

    Test-Path scripts\benchmark\run_vegachat_compatible_benchmark.py
    Test-Path scripts\benchmark\run_infiagent_chart_grounded.py
    Test-Path scripts\benchmark\infiagent_scan.py
    Test-Path ui\config\project-gemma4-bench_norag.toml
    Test-Path ui\config\project-gemma4-bench_rag.toml
    Test-Path scripts\rag_corpus\run_prepare_corpus.py
    Test-Path scripts\rag_corpus\run_export_runtime.py
    Test-Path scripts\rag_corpus\run_autorag_optimization.py

Что должно быть готово для запуска benchmark

    Test-Path .\datasets\nlv_corpus\NLV_Corpus.csv
    Test-Path .\datasets\nlv_corpus\vlSpecs.json
    Test-Path .\datasets\nlv_corpus\datasets
    Test-Path rag_corpus\runtime\virage_rules.jsonl
