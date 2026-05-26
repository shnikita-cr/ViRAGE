# Скрипты корпуса ViRAGE

## Назначение

Скрипты готовят практический корпус правил для этапа `visrag`: выбор графика по задаче, типовые ошибки, цвет, подписи, читаемость и доступность. Корпус не содержит Vega-Lite-спецификаций и не использует NLV как источник знаний.

Основной корпус строится из источников:

    wilke_fundamentals
    from_data_to_viz
    ft_visual_vocabulary
    uk_analysis_colours
    uk_charts_checklist
    urban_institute_style_guide
    chartability

## Папки

- `loading/` — загрузка исходных HTML/текстовых страниц.
- `exporters/` — извлечение исходных записей из загруженных источников.
- `normalize/` — LLM-нормализация, фильтрация, дедупликация, валидация.
- `runtime/` — экспорт компактного корпуса для приложения.
- `autorag/` — экспорт корпуса и вопросов для AutoRAG.

## Загрузка источников

    python scripts/rag_corpus/loading/download_sources.py

Обновить сохранённые страницы:

    python scripts/rag_corpus/loading/download_sources.py --refresh


## Мини-загрузчики источников

Каждый внешний источник можно загрузить отдельно:

    python scripts\rag_corpus\loading\load_wilke_fundamentals.py --refresh
    python scripts\rag_corpus\loading\load_from_data_to_viz.py --refresh
    python scripts\rag_corpus\loading\load_ft_visual_vocabulary.py --refresh
    python scripts\rag_corpus\loading\load_uk_analysis_colours.py --refresh
    python scripts\rag_corpus\loading\load_uk_charts_checklist.py --refresh
    python scripts\rag_corpus\loading\load_urban_institute_style_guide.py --refresh
    python scripts\rag_corpus\loading\load_chartability.py --refresh

Загрузчики используют BeautifulSoup для HTML-страниц, обходят только полезные внутренние HTML-ссылки и не используют fallback-тексты.

## Полный pipeline

    python rag_corpus/run_rag_corpus_pipeline.py

## Подготовка корпуса

С загрузкой источников:

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --clean-processed

Без повторной загрузки источников, только если `rag_corpus/raw_external_rules/*` уже реально загружены:

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --clean-processed --skip-source-download

## Запуск отдельных извлекателей

    python scripts/rag_corpus/exporters/extract_wilke_fundamentals.py
    python scripts/rag_corpus/exporters/extract_from_data_to_viz.py
    python scripts/rag_corpus/exporters/extract_ft_visual_vocabulary.py
    python scripts/rag_corpus/exporters/extract_uk_analysis_colours.py
    python scripts/rag_corpus/exporters/extract_uk_charts_checklist.py
    python scripts/rag_corpus/exporters/extract_urban_institute_style_guide.py
    python scripts/rag_corpus/exporters/extract_chartability.py

## Поведение при ошибках

Загрузка и извлечение работают строго: если источник не скачался, файл подозрительно маленький, raw-папка отсутствует или из реальных данных не извлечено ни одной записи, pipeline падает с ошибкой. Подстановочные тексты не создаются.

## Проверка

Проверить загрузку:

    notepad rag_corpus\reports\source_download_report.json

Проверить извлечение:

    notepad rag_corpus\reports\extraction_report.json

Проверить количество извлечённых записей:

    python -c "from pathlib import Path; [print(p.name, sum(1 for _ in p.open(encoding='utf-8'))) for p in Path('rag_corpus/extracted').glob('*.jsonl')]"

## Старые извлекатели

Извлекатели для IBM Carbon, W3C WAI, VisText, Data Visualisation Catalogue и других прежних источников оставлены для обратной совместимости и тестов, но не входят в основной practical `visrag`-корпус.
