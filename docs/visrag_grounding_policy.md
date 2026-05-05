# VisRAG Grounding Policy

Документ фиксирует policy для выбора режима field grounding в VisRAG: когда `selected_fields` должны быть строгим контрактом, а когда только предпочтением.

## 1. Контекст

Текущий pipeline уже подтверждён integration-тестами:

```text
runtime corpus loading
→ core retrieval/search
→ VisRAGService
→ ChartGeneratorService
→ SpecValidatorService
→ VegaLitePlotDrawingService
→ ScenegraphCheckService
→ EmptyChartCheckService
→ intent-level quality baseline
```

Intent-level baseline сейчас показывает:

```text
scatter_relationship_numeric  passed
line_trend_over_time          passed
histogram_distribution        passed
bar_compare_categories        xfailed
heatmap_two_dimensions        xfailed
```

Этот документ касается первого xfail:

```text
bar_compare_categories
```

---

## 2. Текущая проблема

Кейс:

```text
query:
  compare sales across product categories with a bar chart

selected_fields:
  Category, Sales

data profile:
  Category = dimension
  Sales = measure
  Month = date

expected:
  x = Category
  y = Sales

current:
  x = Month
  y = Sales
```

Текущий результат технически валиден, но семантически неверен: пользователь просит сравнение по категориям, а график строится по месяцам.

---

## 3. Почему так происходит

Текущий grounding использует `selected_fields` как предпочтение:

```text
selected_fields используются первыми, если подходят по semantic role;
если не подходят — grounding может выбрать другое поле из DataProfile.
```

Поэтому candidate с role:

```text
x = temporal
y = quantitative
```

может выбрать:

```text
x = Month
y = Sales
```

даже если `selected_fields = Category, Sales`.

---

## 4. Почему policy должна вычисляться автоматически

Ручной режим `strict/prefer` недостаточен, потому что разные запросы требуют разного поведения.

Пример строгого запроса:

```text
compare sales across product categories
selected_fields = Category, Sales
confidence = high
```

Ожидаем:

```text
policy = strict
```

Пример размытого запроса:

```text
show me a useful chart
selected_fields = []
confidence = low
```

Ожидаем:

```text
policy = prefer
```

Пример среднего доверия:

```text
show distribution
selected_fields = Value
confidence = medium
```

Ожидаем:

```text
policy = soft_fail
```

Итог:

```text
VisRAG должен уметь вычислять policy сам по query/request/data_profile,
а не требовать ручного выбора в каждом вызове.
```

---

## 5. Итоговый контракт

Добавляем enum:

```text
SelectedFieldsPolicy:
  auto
  prefer
  strict
  soft_fail
```

Где:

```text
auto       = вычислить policy детерминированным resolver-ом
prefer     = текущее поведение: selected_fields как предпочтение
strict     = если selected_fields заданы, использовать только selected_fields
soft_fail  = сначала strict, затем fallback на prefer с caveat
```

Runtime должен работать в два этапа:

```text
selected_fields_policy_mode:
  auto | prefer | strict | soft_fail

resolved policy:
  prefer | strict | soft_fail
```

---

## 6. GroundingPolicyResolver

Новый компонент:

```text
src/visrag_core/grounding_policy.py
```

Назначение:

```text
VisRAGRequest
+ request_confidence
+ ambiguity_notes
+ DataProfile
→ GroundingPolicyDecision
```

Decision:

```python
class GroundingPolicyDecision(BaseModel):
    selected_fields_policy: SelectedFieldsPolicy
    confidence: float
    reasons: list[str]
```

Resolver должен быть:

```text
детерминированным
без LLM
легко тестируемым
объяснимым через reasons
```

---

## 7. Auto rules v1

### `prefer`, если:

```text
selected_fields пустые
```

Пример:

```text
query = show me a useful chart
selected_fields = []
```

Reason:

```text
no_selected_fields
```

---

### `strict`, если:

```text
selected_fields не пустые
request_confidence >= 0.75
selected_fields существуют в DataProfile
selected_fields не только identifier/id поля
нет ambiguity_notes
есть явный query signal или preferred_chart_types
```

Явные query signals:

```text
by
across
over time
relationship between
distribution of
compare / comparing
trend
correlation
scatter
histogram
line chart
bar chart
```

Пример:

```text
compare sales across product categories with a bar chart
selected_fields = Category, Sales
confidence = 0.92
preferred_chart_types = bar
```

Expected:

```text
policy = strict
```

---

### `soft_fail`, если:

```text
selected_fields есть, но confidence средний
selected_fields частично отсутствуют в DataProfile
selected_fields похожи только на identifier/id поля
есть ambiguity_notes
selected_fields слишком много
```

Пример:

```text
query = show distribution
selected_fields = Sales
confidence = 0.61
```

Expected:

```text
policy = soft_fail
```

---

## 8. Первый инкремент

Сначала добавляем только resolver и unit-тесты.

Добавляем:

```text
src/visrag_core/grounding_policy.py
tests/unit/test_visrag_grounding_policy.py
```

На этом шаге не меняем:

```text
field_grounding.py
VisRAGCoreService
VisRAGService
intent-level xfail статусы
```

Цель:

```text
зафиксировать контракт и правила без риска сломать зелёный baseline.
```

---

## 9. Unit-test expectations

Тесты resolver-а должны проверять:

```text
no selected_fields → prefer
high confidence explicit category comparison → strict
high confidence trend with preferred chart → strict
medium confidence selected fields → soft_fail
missing selected field → soft_fail
identifier-only selected fields → soft_fail
ambiguity notes present → soft_fail
explicit override → respected
```

---

## 10. Следующий кодовый инкремент после resolver-а

После зелёных unit-тестов:

```text
1. Добавить selected_fields_policy в VisRAGRequest.
2. Подключить GroundingPolicyResolver в VisRAGService.
3. Передавать resolved policy в VisRAGCoreService / field_grounding.
4. Реализовать strict в map_fields.
5. Реализовать soft_fail на уровне search/service.
6. Обновить intent fixture.
7. Перевести bar_compare_categories из xfail в pass.
```

---

## 11. Expected final result после подключения policy

Intent-level suite:

```text
before:
  3 passed, 2 xfailed

after:
  4 passed, 1 xfailed
```

Должен перейти в pass:

```text
bar_compare_categories
```

Останется xfail:

```text
heatmap_two_dimensions
```

---

## 12. Acceptance criteria для resolver-инкремента

```text
[ ] src/visrag_core/grounding_policy.py добавлен.
[ ] tests/unit/test_visrag_grounding_policy.py добавлен.
[ ] Resolver не меняет runtime-поведение.
[ ] Все новые unit-тесты зелёные.
[ ] Все старые integration-тесты остаются зелёными.
[ ] Документ обновлён под auto-resolver.
```

---

## 13. Acceptance criteria для следующего runtime-инкремента

```text
[ ] selected_fields_policy поддерживается в VisRAGRequest.
[ ] default mode = auto или prefer, без регрессии текущих тестов.
[ ] strict запрещает использовать поля вне selected_fields.
[ ] soft_fail делает strict-first, prefer-fallback и добавляет caveat.
[ ] bar_compare_categories получает x=Category, y=Sales.
[ ] scatter / line / histogram остаются зелёными.
[ ] heatmap остаётся xfail.
```
