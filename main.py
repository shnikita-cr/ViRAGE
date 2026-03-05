import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

from rag import VisualizationRAG, initialize_rag


def main():
    print("=" * 60)
    print("RAG система по визуализации данных")
    print("=" * 60)

    print("\nИнициализация RAG...")
    rag = initialize_rag()

    print("\nДоступные команды:")
    print("  - Введите вопрос о визуализации данных")
    print("  - 'search <запрос>' - просто найти релевантные записи")
    print("  - 'quit' или 'exit' - выход")

    while True:
        print("\n" + "-" * 40)
        try:
            user_input = input("Ваш вопрос: ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "q"]:
            print("До свидания!")
            break

        if user_input.lower().startswith("search "):
            query = user_input[7:]
            print(f"\nПоиск по запросу: {query}")
            results = rag.search(query)
            for i, result in enumerate(results, 1):
                print(f"\n--- Результат {i} ---")
                print(f"Задача: {result['task']}")
                print(f"Рекомендуемые графики: {', '.join(result['best_charts'])}")
                print(f"Библиотеки: {result['library']}")
        else:
            print("\nОтвет:")
            answer = rag.ask(user_input)
            print(answer)


if __name__ == "__main__":
    main()
