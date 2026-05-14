# VegaChat-compatible benchmark protocol in ViRAGE

## Цель

В режиме VegaChat-compatible benchmark проект ViRAGE повторяет постановку оценки VegaChat для задачи NL2VIS: на вход подаются табличные данные и естественно-языковой запрос, система генерирует Vega-Lite specification и изображение графика, после чего результат сравнивается с эталонной Vega-Lite specification и эталонным изображением.

Данный файл фиксирует формальные метрики и наборы данных, используемые в проекте для академического сравнения с VegaChat.

## Бенчмарки

### NLV Corpus

Используется как основной NL2VIS-бенчмарк с естественно-языковыми запросами, табличными данными и эталонными Vega-Lite specifications.

Для каждого примера используются:

- `query` — естественно-языковой запрос;
- `data_path` — CSV-таблица;
- `reference_spec` — эталонная Vega-Lite specification;
- `reference_image` — эталонное изображение, если оно доступно; иначе оно генерируется из `reference_spec`.

### ChartLLM / VL2NL

Используется как второй NL2VIS-бенчмарк для проверки разных формулировок одного визуального намерения.

Для каждого исходного графика используются варианты запроса:

- `command`;
- `query`;
- `question`.

Для каждого варианта считается тот же набор метрик: Spec Score, Vision Score, Visualization Error Rate и Empty Chart Rate.

## Обозначения

Пусть:

- $S_{ref}$ — эталонная Vega-Lite specification;
- $S_{hyp}$ — сгенерированная Vega-Lite specification;
- $I_{ref}$ — эталонное изображение;
- $I_{hyp}$ — сгенерированное изображение;
- $u$ — пользовательский запрос;
- $D$ — таблица данных;
- $valid(S_{hyp})$ — признак валидности сгенерированной спецификации;
- $drawable(S_{hyp})$ — признак того, что спецификация может быть отрисована;
- $empty(I_{hyp})$ — признак пустого или неинформативного графика.

## Visualization Error Rate

Для одного примера:

$$VER_i = 1 - \mathbb{1}[valid(S_{hyp})]$$

Для набора из $N$ примеров:

$$VER = \frac{1}{N}\sum_{i=1}^{N} VER_i$$

В ViRAGE это поле сохраняется как `visualization_error_rate`.

## Empty Chart Rate

Для одного примера:

$$ECR_i = \mathbb{1}[\neg valid(S_{hyp}) \lor empty(I_{hyp})]$$

Для набора из $N$ примеров:

$$ECR = \frac{1}{N}\sum_{i=1}^{N} ECR_i$$

В ViRAGE это поле сохраняется как `empty_plot_rate` и агрегируется как `empty_chart_rate`.

## Precision, Recall и F-score

Для сравнения компонентов спецификаций используются мультимножества эталонных и сгенерированных элементов.

Пусть:

- $R$ — мультимножество эталонных элементов;
- $H$ — мультимножество сгенерированных элементов;
- $TP = |R \cap H|$;
- $P = |H|$;
- $T = |R|$.

Тогда:

$$Precision = \frac{TP}{P}$$

$$Recall = \frac{TP}{T}$$

Обобщённый F-score:

$$F_{\beta} = \frac{(1 + \beta^2) \cdot Precision \cdot Recall}{\beta^2 \cdot Precision + Recall}$$

Для mark и transform используется $F_1$:

$$F_1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$$

Для encoding используется $F_2$, так как полнота правильных кодирований важнее лишних допустимых деталей:

$$F_2 = \frac{5 \cdot Precision \cdot Recall}{4 \cdot Precision + Recall}$$

## Mark Score

Из спецификаций извлекаются значения `mark`.

$$MarkScore = F_1(M_{ref}, M_{hyp})$$

Для совместимости с VegaChat дополнительно применяется частичная эквивалентность:

$$circle \sim point \sim square$$

Итоговый mark score считается как среднее между строгим совпадением и совпадением с учётом этой эквивалентности:

$$MarkScore = \frac{F_1(M_{ref}, M_{hyp}) + F_1(E(M_{ref}), E(M_{hyp}))}{2}$$

где $E$ заменяет `circle`, `point` и `square` на общий класс `circle-point-square`.

## Encoding Score

Из `encoding` извлекаются пары вида:

$$e = (channel, property, value)$$

Учитываются свойства:

- `field`;
- `type`;
- `aggregate`;
- `bin`;
- `timeUnit`.

Для совместимости с VegaChat допускается перестановка осей $x/y$:

$$x \sim y$$

Также допускается перестановка faceting-каналов:

$$row \sim column$$

Свойства `type` и `timeUnit` имеют пониженный вес:

$$w(type) = 0.5$$

$$w(timeUnit) = 0.5$$

Остальные свойства имеют вес:

$$w(other) = 1.0$$

Encoding score считается через weighted $F_2$:

$$EncodingScore = F_2^{weighted}(E_{ref}, E_{hyp})$$

## Transform Score

Из спецификаций извлекаются пути внутри `transform`.

$$TransformScore = F_1(T_{ref}, T_{hyp})$$

Порядок элементов списков нормализуется через замену индексов списков на общий индекс, чтобы не штрафовать эквивалентные transform-структуры только из-за порядка.

## Spec Score

Если спецификация не может быть отрисована, итоговый Spec Score равен нулю:

$$SpecScore = 0, \quad \text{если } drawable(S_{hyp}) = 0$$

Иначе используется взвешенная сумма компонент.

Базовый вклад отрисовываемости:

$$w_{drawable} = 0.005$$

$$x_{drawable} = w_{drawable}$$

Вклад валидности схемы:

$$w_{schema} = \begin{cases} 0.005, & valid(S_{hyp}) = 1 \\ 1.0, & valid(S_{hyp}) = 0 \end{cases}$$

$$x_{schema} = \begin{cases} w_{schema}, & valid(S_{hyp}) = 1 \\ 0, & valid(S_{hyp}) = 0 \end{cases}$$

Вклад непустого графика:

$$w_{not\_empty} = \begin{cases} 1000, & empty(I_{hyp}) = 1 \\ 0.005, & empty(I_{hyp}) = 0 \end{cases}$$

$$x_{not\_empty} = \begin{cases} 0, & empty(I_{hyp}) = 1 \\ w_{not\_empty}, & empty(I_{hyp}) = 0 \end{cases}$$

Вклад transform:

$$x_{transform} = w_{transform} \cdot TransformScore$$

Если в эталоне нет transform и в гипотезе тоже нет transform, используется малый положительный вклад:

$$w_{transform} = 0.005, \quad x_{transform} = 0.005$$

Если в эталоне нет transform, но в гипотезе есть лишний transform, применяется штраф к encoding:

$$w_{encoding\_penalty} = 0.25$$

Вклад mark:

$$w_{mark} = \begin{cases} 1.0, & \text{если тип графика явно упомянут в } u \\ 0.5, & \text{иначе} \end{cases}$$

$$x_{mark} = w_{mark} \cdot MarkScore$$

Вклад encoding:

$$w_{encoding} = 3.0 \cdot (1 - w_{encoding\_penalty})$$

$$x_{encoding} = w_{encoding} \cdot EncodingScore$$

Итоговая формула:

$$SpecScore = \frac{x_{drawable} + x_{schema} + x_{not\_empty} + x_{encoding} + x_{mark} + x_{transform}}{w_{drawable} + w_{schema} + w_{not\_empty} + w_{encoding} + w_{mark} + w_{transform}}$$

В ViRAGE это поле сохраняется как `spec_score`.

## Дополнительные spec-level метрики

Для диагностики сохраняются дополнительные метрики VegaChat-compatible сравнения:

- `mark_f1`;
- `mark_precision`;
- `mark_recall`;
- `encoding_f1`;
- `encoding_precision`;
- `encoding_recall`;
- `transform_f1`;
- `transform_precision`;
- `transform_recall`;
- `full_f1`;
- `keys_f1`;
- `kvs_f1`;
- `keys_jaccard`.

Jaccard similarity по ключам:

$$JaccardKeys = \frac{|K_{ref} \cap K_{hyp}|}{|K_{ref} \cup K_{hyp}|}$$

## Vision Score

Vision Score считается в reference-mode. VLM получает:

- первое изображение: $I_{hyp}$;
- второе изображение: $I_{ref}$;
- пользовательский запрос $u$.

VLM возвращает дискретные оценки $0$, $1$ или $2$ по критериям:

- `visualization_type`;
- `data_encoding`;
- `data_transformation`;
- `aesthetics`;
- `prompt_compliance`;
- `is_blank`.

Веса критериев:

$$w_{type} = 1.0$$

$$w_{encoding} = 2.0$$

$$w_{transformation} = 1.0$$

$$w_{aesthetics} = 0.75$$

$$w_{prompt} = 1.5$$

Каждая VLM-оценка нормализуется делением на $2$:

$$\hat{c}_j = \frac{c_j}{2}$$

Если график пустой, в знаменатель добавляется штраф:

$$w_{blank} = 1000$$

Итоговая формула:

$$VisionScore = \frac{w_{type}\hat{c}_{type} + w_{encoding}\hat{c}_{encoding} + w_{transformation}\hat{c}_{transformation} + w_{aesthetics}\hat{c}_{aesthetics} + w_{prompt}\hat{c}_{prompt}}{w_{type} + w_{encoding} + w_{transformation} + w_{aesthetics} + w_{prompt} + \mathbb{1}[is\_blank] \cdot w_{blank}}$$

В ViRAGE это поле сохраняется как `vision_judge` и агрегируется как `mean_vision_score`.

## Aggregation over benchmark

Для любой метрики $m$ итоговое значение по benchmark считается как среднее по всем примерам, где метрика определена:

$$\overline{m} = \frac{1}{|Q_m|}\sum_{i \in Q_m} m_i$$

где $Q_m$ — множество примеров, для которых метрика $m$ была посчитана.

Для Spec Score и Vision Score дополнительно сохраняются медианные значения:

$$MedianSpecScore = median(SpecScore_1, ..., SpecScore_N)$$

$$MedianVisionScore = median(VisionScore_1, ..., VisionScore_N)$$

## Отчёты ViRAGE

Benchmark runner сохраняет:

- `benchmark_results.csv` — построчные результаты;
- `benchmark_report.json` — агрегированный отчёт;
- `benchmark_report.md` — человекочитаемый отчёт;
- `cases/<case_id>/result.json` — результат конкретного примера;
- `cases/<case_id>/generated_spec.json` — сгенерированная спецификация.

Ключевые поля отчёта:

- `visualization_error_rate`;
- `empty_chart_rate`;
- `mean_spec_score`;
- `median_spec_score`;
- `mean_vision_score`;
- `median_vision_score`;
- `vegachat_metrics`;
- `total_tokens`;
- `mean_duration_seconds`.

## Сравниваемые системы

Для академического сравнения используются одинаковые датасеты и одинаковые метрики для следующих систем:

- VegaChat;
- Plain LLM baseline;
- ViRAGE.

Ключевое условие сравнения: для всех систем используются одинаковые входные запросы, таблицы, эталонные спецификации, эталонные изображения и одинаковый evaluator.
