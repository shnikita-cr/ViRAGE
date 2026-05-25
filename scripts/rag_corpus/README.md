# Скрипты корпуса ViRAGE

## Назначение

Скрипты готовят корпус правил качества графиков. Основной корпус строится из 9 читаемых источников:

    ft_visual_vocabulary
    from_data_to_viz
    data_visualisation_catalogue
    ibm_carbon_chart_anatomy
    ibm_carbon_legends
    uswds_data_visualizations
    urban_institute_style_guide
    w3c_wai_complex_images
    vistext

## Папки

- `sources/` — извлечение исходных записей из локально скачанных источников.
- `normalize/` — LLM-нормализация, фильтрация, дедупликация, валидация.
- `runtime/` — экспорт компактного корпуса для приложения.
- `autorag/` — экспорт корпуса и вопросов для AutoRAG.

## Основной запуск

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --clean-processed

## Запуск отдельных источников

    python scripts/rag_corpus/sources/extract_ft_visual_vocabulary.py
    python scripts/rag_corpus/sources/extract_from_data_to_viz.py
    python scripts/rag_corpus/sources/extract_data_visualisation_catalogue.py
    python scripts/rag_corpus/sources/extract_ibm_carbon_chart_anatomy.py
    python scripts/rag_corpus/sources/extract_ibm_carbon_legends.py
    python scripts/rag_corpus/sources/extract_uswds_data_visualizations.py
    python scripts/rag_corpus/sources/extract_urban_institute_style_guide.py
    python scripts/rag_corpus/sources/extract_w3c_wai_complex_images.py
    python scripts/rag_corpus/sources/extract_vistext.py

## Старые извлекатели

Файлы для `draco`, `compassql`, `chartsquared`, `taskvis`, `vega_lite_examples` оставлены только для истории и обратной совместимости. Они не вызываются основным пайплайном. Список кандидатов на ручное удаление лежит в `rag_corpus/manual_delete_candidates.md`.

Проверить ошибки извлечения:

    notepad rag_corpus\reports\extraction_report.json

## Примечание по извлечению источников качества графиков

Пайплайн дополнительно сохраняет несколько прямых HTML-страниц From Data to Viz, IBM Carbon и USWDS. Это нужно, чтобы извлечение не зависело только от текущей структуры репозиториев. VisText обрабатывается только как структурированный набор подписей и таблиц; файлы метрик, предсказаний и результатов моделей исключаются.

