# Architecture diagrams

Актуальные UML-схемы проекта ViRAGE. Старые экспериментальные схемы VegaChat/EDAPlot и исследовательские черновики исключены из рабочей папки архитектуры.

## Файлы

- `01_runtime_pipeline.puml` — основной runtime pipeline.
- `02_visrag_runtime.puml` — runtime-часть VisRAG, поиск правил и совместимость с анализом запроса.
- `03_corpus_autorag_pipeline.puml` — подготовка корпуса, фильтрация, AutoRAG train/test.
- `04_benchmark_protocol.puml` — протокол NLV/InfiAgent benchmark.

## Правило поддержки

Если меняется граф pipeline, corpus flow или benchmark-протокол, сначала обновляется соответствующий `.puml`, затем документация и команды запуска.
