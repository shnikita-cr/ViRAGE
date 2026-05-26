import os

# Фикс для Windows против критического падения процесса (0xC0000005)
# Настраивает потоки OpenMP и предотвращает конфликты библиотек математических вычислений
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Переключаемся на современные и поддерживаемые пакеты
from langchain_huggingface import HuggingFaceEmbeddings
# Импортируем FAISS без использования устаревшего общего корня langchain_community
from langchain_community.vectorstores.faiss import FAISS

# Настройки путей (должны совпадать со скриптом создания базы)
FAISS_DB_DIR = "faiss_dataviz_index"


def load_retriever():
    """Загружает модель эмбеддингов и локальную базу FAISS."""
    if not os.path.exists(FAISS_DB_DIR):
        raise FileNotFoundError(
            f"Векторная база '{FAISS_DB_DIR}' не найдена. "
            f"Сначала запустите скрипт индексации книги."
        )

    print("Загрузка модели эмбеддингов...")
    # Использование современного пакета langchain_huggingface
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    print("Загрузка базы данных FAISS...")
    # allow_dangerous_deserialization=True необходим для загрузки локальных файлов FAISS
    db = FAISS.load_local(FAISS_DB_DIR, embeddings, allow_dangerous_deserialization=True)
    return db


def main():
    try:
        db = load_retriever()
        print("\n" + "=" * 50)
        print(" РЕТРИВЕР ГОТОВ К РАБОТЕ ")
        print("Введите свой вопрос по книге Клауса Вильке.")
        print("Для выхода из программы введите 'exit' или 'выход'.")
        print("=" * 50 + "\n")
    except Exception as e:
        print(f"Ошибка при инициализации: {e}")
        return

    while True:
        # Ручной ввод вопроса пользователем
        query = input("\nВаш вопрос: ").strip()

        # Проверка на выход из цикла
        if query.lower() in ['exit', 'выход', 'quit', 'q']:
            print("Завершение работы ретривера. До свидания!")
            break

        if not query:
            print("Вопрос не может быть пустым. Попробуйте еще раз.")
            continue

        print(f"Поиск релевантных фрагментов для: '{query}'...\n")

        try:
            # similarity_search_with_score возвращает кортеж (Document, score)
            # В FAISS score — это L2-расстояние (чем МЕНЬШЕ значение, тем БЛИЖЕ и релевантнее текст)
            k = 3  # Количество возвращаемых фрагментов
            results = db.similarity_search_with_score(query, k=k)

            print(f"--- НАЙДЕНО ТОП-{k} СОВПАДЕНИЙ ---")
            for idx, (doc, score) in enumerate(results, 1):
                print(f"\n[Фрагмент №{idx}] [L2 Distance (Score): {score:.4f}]")
                print("-" * 60)
                # Выводим текст фрагмента
                print(doc.page_content)
                print("-" * 60)

        except Exception as e:
            print(f"Произошла ошибка при поиске: {e}")


if __name__ == "__main__":
    main()
