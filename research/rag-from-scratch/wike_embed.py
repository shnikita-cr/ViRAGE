import os

# Фикс для Windows против критического падения процесса (0xC0000005)
# Настраивает потоки OpenMP и предотвращает конфликты библиотек математических вычислений
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Переключаемся на современные и поддерживаемые пакеты
from langchain_huggingface import HuggingFaceEmbeddings
# Импортируем FAISS без использования устаревшего общего корня langchain_community
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Настройки путей
INPUT_FILE = "../../rag_corpus/raw_external_rules/wike/claus_wilke_dataviz_rag.txt"
FAISS_DB_DIR = "faiss_dataviz_index"


def main():
    # 1. Проверяем наличие текстового файла книги
    if not os.path.exists(INPUT_FILE):
        print(f"Ошибка: Файл {INPUT_FILE} не найден. Сначала запустите скрипт скачивания.")
        return

    print("Читаем текст книги...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        book_text = f.read()

    # 2. Нарезаем текст на чанки
    print("Разбиваем текст на смысловые фрагменты (chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # оптимальный размер для этой модели эмбеддингов
        chunk_overlap=150,  # перекрытие, чтобы не терять контекст на стыках
        length_function=len
    )

    # create_documents принимает список строк и возвращает список объектов Document
    documents = text_splitter.create_documents([book_text])
    print(f"Успешно создано {len(documents)} фрагментов.")

    # 3. Инициализируем модель эмбеддингов
    print("Загружаем модель эмбеддингов (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # используем процессор
    )

    # 4. Создаем базу данных FAISS и индексируем фрагменты
    print("Создаем векторный индекс FAISS (это может занять некоторое время)...")
    db = FAISS.from_documents(documents, embeddings)

    # 5. Сохраняем базу локально на диск
    print(f"Сохраняем векторную базу в папку: {FAISS_DB_DIR}")
    db.save_local(FAISS_DB_DIR)
    print("Готово! Векторная база данных успешно создана и сохранена.")

    # 6. Тестовый поиск для проверки работы
    print("\n--- Проверка работоспособности (Тестовый поиск) ---")
    query = "How to choose a color palette for data visualization?"
    print(f"Запрос: '{query}'")

    # Ищем 3 самых похожих фрагмента
    k = 3
    docs = db.similarity_search(query, k=k)

    print(f"\nНайдено совпадений: {len(docs)}")
    print("Результаты поиска:")
    # Итерируемся напрямую по найденным документам (безопасно, если len(docs) < k)
    for idx, doc in enumerate(docs, 1):
        print(f"\n[Фрагмент №{idx}]")
        print("-" * 50)
        print(doc.page_content[:400] + "...")
        print("-" * 50)


if __name__ == "__main__":
    main()
