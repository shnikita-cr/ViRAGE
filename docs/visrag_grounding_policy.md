# VisRAG Grounding Policy

Документ фиксирует текущую политику сопоставления `selected_fields` с ролями полей в VisRAG, фактический статус внедрения `auto`/`strict`, known issues и следующие возможные итерации.

## 1. Текущий статус

На текущем этапе реализовано:

```text
[done] GroundingPolicyResolver
[done] selected_fields_policy mode
[done] auto policy resolution
[done] strict selected_fields grounding
[done] prefer backward-compatible behavior
[done] pytest baseline green
```

Текущий стабильный результат тестов:

```text
green pytest suite
bar_compare_categories = xfail
heatmap_two_dimensions = xfail
```

Важно: `bar_compare_categories` сейчас остаётся `xfail` **не потому, что strict policy не работает**, а потому что в текущем runtime corpus нет подходящего categorical bar template для:

```text
x = Category / nominal
y = Sales / quantitative
```

---

## 2. Контекст проблемы

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
```

До внедрения strict policy VisRAG мог выбрать технически валидный, но семантически неверный mapping:

```text
x = Month
y = Sales
```

Это происходило потому, что `selected_fields` работали как предпочтение, а не как ограничение.

---

## 3. Что изменилось после strict policy

Теперь при `selected_fields_policy = strict`:

```text
если selected_fields заданы,
field grounding может использовать только selected_fields.
```

То есть для кейса:

```text
selected_fields = Category, Sales
```

VisRAG больше не должен выбирать:

```text
x = Month
```

Это корректное поведение.

---

## 4. Почему `bar_compare_categories` всё ещё xfail

После strict policy старый temporal bar template больше не может быть использован для category-query.

Пример старого template:

```text
bar_month_band
```

Он ожидает:

```text
x = temporal
y = quantitative
```

При strict selected fields:

```text
Category = nominal
Sales = quantitative
```

temporal field среди `selected_fields` нет. Поэтому такой candidate корректно отбрасывается.

Итог:

```text
strict policy работает,
но совместимый categorical bar candidate не найден.
```

Поэтому кейс остаётся known issue:

```text
bar_compare_categories:
  xfail
  reason: current runtime corpus has no compatible categorical bar template available for strict selected_fields.
```

---

## 5. Policy modes

Поддерживаемые режимы:

```text
auto
prefer
strict
soft_fail
```

---

## 6. `auto`

Режим по умолчанию.

VisRAG вычисляет итоговую policy автоматически на основании:

```text
query_understanding
request_analysis
data_profile
selected_fields
request confidence
ambiguity notes
```

`auto` возвращает одну из effective policies:

```text
prefer
strict
soft_fail
```

---

## 7. `prefer`

Backward-compatible режим.

Поведение:

```text
selected_fields используются как предпочтение,
но grounding может использовать другие поля из DataProfile,
если selected_fields не подходят по field_roles.
```

Использование:

```text
размытые запросы
exploratory mode
нет selected_fields
низкая уверенность RequestAnalyzer
```

---

## 8. `strict`

Строгий режим.

Поведение:

```text
если selected_fields заданы,
grounding использует только selected_fields.
```

Использование:

```text
пользователь явно указал поля или аналитический смысл
RequestAnalyzer уверен
нужно избежать semantic drift
```

Пример:

```text
query: compare sales across categories
selected_fields: Category, Sales
```

В strict режиме недопустимо заменить `Category` на `Month`.

---

## 9. `soft_fail`

Промежуточный режим.

Целевое поведение:

```text
1. сначала попытаться strict
2. если candidates не найдены — fallback на prefer
3. добавить caveat
```

Текущий статус:

```text
soft_fail fallback не является основной завершённой итерацией.
```

Рекомендуемый caveat:

```text
selected_fields_fallback: no compatible candidates using only selected fields; used broader DataProfile.
```

---

## 10. GroundingPolicyResolver

Компонент:

```text
src/visrag_core/grounding_policy.py
```

Основные сущности:

```text
SelectedFieldsPolicy
GroundingPolicyDecision
GroundingPolicyResolver
```

Решение policy должно быть:

```text
детерминированным
тестируемым
объяснимым через reasons
без LLM внутри resolver
```

---

## 11. Тесты

### Unit tests

```text
tests/unit/test_visrag_grounding_policy.py
```

Проверяют:

```text
no selected_fields → prefer
high confidence category comparison → strict
high confidence trend → strict
medium confidence → soft_fail
missing selected field → soft_fail
identifier-only fields → soft_fail
ambiguity notes → soft_fail
explicit override → respected
```

### Field grounding policy tests

```text
tests/unit/test_visrag_field_grounding_policy.py
```

Проверяют:

```text
strict использует только selected_fields
prefer сохраняет старое поведение
```

### Intent quality tests

```text
tests/integration/test_visrag_intent_quality_real_corpus.py
```

Текущий ожидаемый статус:

```text
scatter_relationship_numeric = pass
line_trend_over_time = pass
histogram_distribution = pass
bar_compare_categories = xfail
heatmap_two_dimensions = xfail
```

---

## 12. Почему categorical bar template не добавляем сейчас

Решение текущей итерации:

```text
не добавлять curated local categorical bar template в MVP
```

Причина:

```text
хотим сохранить текущий runtime corpus как результат official vega-lite subset,
а отсутствие categorical bar template зафиксировать как known issue.
```

Это значит, что strict policy остаётся внедрённой, но конкретный intent-case пока не переводится в pass.

---

## 13. Known issues

### 13.1. `bar_compare_categories`

Статус:

```text
xfail
```

Причина:

```text
strict selected_fields работает,
но в текущем runtime corpus нет подходящего categorical bar template.
```

Возможное будущее решение:

```text
добавить curated local template:
  chart_type = bar
  field_roles.x = nominal
  field_roles.y = quantitative
```

Но в текущем MVP это не делаем.

---

### 13.2. `heatmap_two_dimensions`

Статус:

```text
xfail
```

Причина:

```text
heatmap/rect runtime support пока не стабилен.
```

Возможное будущее решение:

```text
стабилизировать rect templates:
  x = nominal
  y = nominal
  color = quantitative
```

---

### 13.3. Over-specific line templates

Некоторые line templates могут быть слишком специфичными.

Пример:

```text
time_output_utc_scale
```

Может содержать:

```text
timeUnit = yearmonthdatehoursminutes
scale.type = utc
```

Возможное будущее решение:

```text
ranking penalty за overly-specific templates
```

---

## 14. Следующие возможные итерации

### Итерация A. Soft-fail fallback

Реализовать полный fallback:

```text
strict first
prefer fallback
caveat
```

### Итерация B. Categorical bar template

Добавить curated local runtime-safe template, если решим закрывать `bar_compare_categories`.

### Итерация C. Heatmap support

Отдельно стабилизировать `rect/heatmap` корпус и grounding.

### Итерация D. Ranking penalties

Добавить штрафы за слишком специфичные templates.

---

## 15. Current DoD

Текущий grounding-policy MVP считается готовым, если:

```text
[done] GroundingPolicyResolver добавлен
[done] strict selected_fields работает
[done] prefer backward-compatible
[done] тесты зелёные
[done] bar_compare_categories остаётся documented xfail
[done] heatmap_two_dimensions остаётся documented xfail
```
