import time
import json
import io
import ollama
from PIL import Image, ImageDraw, ImageFont

# Списки моделей строго по вашим компонентам
COMPONENTS = {
    "1. Оркестратор / Планирование": [
        "deepseek-r1:8b", "qwen3.5:latest", "gemma3:4b", "llama3.1:8b", "hermes3:8b"
    ],
    "2. Кодогенерация JSON": [
        "qwen2.5-coder:7b", "deepseek-coder:6.7b", "qwen2.5-coder:3b", "codegemma:7b", "granite-code:8b"
    ],
    "3. Критик / Валидация (VLM)": [
        "qwen2.5vl:7b", "gemma3:12b", "openbmb/minicpm-v4.5", "phi3.5-vision:latest", "moondream:latest"
    ]
}

# Тестовые данные для каждого компонента
PROMPTS = {
    "1. Оркестратор / Планирование": {
        "system": "Вы — управляющий агент. Проанализируй запрос и вызови инструмент, ответив строго в формате JSON: {'action': 'render_chart', 'parameters': {'type': 'bar', 'sort': 'descending'}}.",
        "user": "Мне нужно визуализировать продажи по убыванию в виде столбчатой диаграммы."
    },
    "2. Кодогенерация JSON": {
        "system": "Вы — кодер Vega-Lite. Сгенерируй чистый, строго валидный JSON спецификации Vega-Lite (блок $schema, mark, encoding). Не используй markdown-обертки ```json.",
        "user": "Построй bar chart. Данные: [{'x': 'A', 'y': 10}, {'x': 'B', 'y': 20}]. Ось X — x, ось Y — y."
    },
    "3. Критик / Валидация (VLM)": {
        "system": "Вы — визуальный критик графиков. Посмотри на картинку. На ней изображен тестовый график. Скажи, видишь ли ты оси координат X и Y, и нет ли ошибок в рендеринге? Ответь кратко.",
        "user": "Проверь этот график на наличие ошибок."
    }
}


def clean_json(text: str) -> str:
    """Удаляет теги рассуждения R1 <think> и markdown-обертки кода."""
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1]
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"): text = text[:-3]
    return text.strip()


def create_mock_chart() -> bytes:
    """Генерирует фейковый график в памяти для тестирования VLM."""
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    # Рисуем оси
    draw.line([(50, 250), (350, 250)], fill='black', width=2)  # X
    draw.line([(50, 50), (50, 250)], fill='black', width=2)  # Y
    # Рисуем тестовые столбцы
    draw.rectangle([80, 150, 130, 250], fill='blue')
    draw.rectangle([180, 100, 230, 250], fill='blue')

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


def run_test():
    mock_chart_bytes = create_mock_chart()
    final_report = []

    print("🚀 Запуск всестороннего бенчмарка компонентов агента...")
    print("=" * 90)

    for component_name, models in COMPONENTS.items():
        print(f"\nТестируем компонент: {component_name}")
        print("-" * 60)

        prompt_data = PROMPTS[component_name]

        for model in models:
            print(f" -> Вызов модели: {model} ... ", end="", flush=True)

            # Подготовка параметров запроса
            chat_args = {
                "model": model,
                "messages": [{"role": "system", "content": prompt_data["system"]},
                             {"role": "user", "content": prompt_data["user"]}],
                "options": {"temperature": 0.1}
            }

            # Если это VLM-компонент, прикрепляем картинку
            if "VLM" in component_name:
                chat_args["messages"][1]["images"] = [mock_chart_bytes]

            start_time = time.time()
            try:
                response = ollama.chat(**chat_args)
                elapsed = round(time.time() - start_time, 2)
                output = response['message']['content']

                # Валидация для текстовых/кодовых компонентов (проверка на JSON)
                status = "Выполнено"
                if "VLM" not in component_name:
                    cleaned = clean_json(output)
                    try:
                        json.loads(cleaned)
                        status = "Valid JSON ✅"
                    except json.JSONDecodeError:
                        status = "Invalid JSON ❌"
                else:
                    status = "Ответ получен 👁️"

                eval_count = response.get('eval_count', 0)
                tps = round(eval_count / elapsed, 1) if eval_count > 0 else "Н/Д"

                print(f"Успешно за {elapsed}с | Скорость: {tps} токенов/сек | Статус: {status}")
                final_report.append((component_name, model, f"{elapsed} сек", tps, status))

            except Exception as e:
                print(f"ОШИБКА: {str(e)[:40]}... (Возможно, модель еще не скачана)")
                final_report.append((component_name, model, "Ошибка", "Н/Д", "Недоступна"))

    # Итоговый вывод результатов в красивом виде
    print("\n" + "=" * 95)
    print(f"{'Компонент':<30} | {'Модель':<25} | {'Время':<10} | {'Ток/сек':<8} | {'Статус проверки'}")
    print("=" * 95)
    for comp, mod, t_sec, tps, stat in final_report:
        print(f"{comp:<30} | {mod:<25} | {t_sec:<10} | {tps:<8} | {stat}")
    print("=" * 95)


if __name__ == "__main__":
    run_test()
