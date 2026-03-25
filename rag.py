import json
import os
from typing import List, Dict, Any

import ollama
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VisualizationRAG:
    def __init__(
            self,
            knowledge_path: str = "data/visualization_knowledge.json",
            embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
            llm_model: str = "llama3.2",
            vector_store_path: str = "vector_store"
    ):
        self.knowledge_path = knowledge_path
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.vector_store_path = vector_store_path
        self.documents = []
        self.embeddings_model = None
        self.index = None

    def load_knowledge(self) -> List[Dict]:
        with open(self.knowledge_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def create_vector_store(self, force_recreate: bool = False):
        if os.path.exists(self.vector_store_path) and not force_recreate:
            print(f"Загрузка существующего векторного хранилища из {self.vector_store_path}")
            self.index = faiss.read_index(os.path.join(self.vector_store_path, "index.faiss"))
            with open(os.path.join(self.vector_store_path, "documents.json"), "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            print(f"Загрузка модели эмбеддингов ({self.embedding_model})...")
            self.embeddings_model = SentenceTransformer(self.embedding_model)
        else:
            print("Создание векторного хранилища...")
            data = self.load_knowledge()

            self.documents = []
            for item in data:
                content = f"""
Задача визуализации: {item['task']}
Описание: {item['description']}
Лучшие типы графиков: {', '.join(item['best_charts'])}
Когда использовать: {item['when_use']}
Библиотеки: {item['library']}
Пример кода: {item['example_code']}
"""
                self.documents.append({
                    "content": content,
                    "task": item["task"],
                    "best_charts": item["best_charts"],
                    "library": item["library"]
                })

            self._create_embeddings()

            os.makedirs(self.vector_store_path, exist_ok=True)
            faiss.write_index(self.index, os.path.join(self.vector_store_path, "index.faiss"))
            with open(os.path.join(self.vector_store_path, "documents.json"), "w", encoding="utf-8") as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            print(f"Векторное хранилище сохранено в {self.vector_store_path}")

    def _create_embeddings(self):
        print(f"Создание эмбеддингов через SentenceTransformer ({self.embedding_model})...")
        self.embeddings_model = SentenceTransformer(self.embedding_model)
        texts = [doc["content"] for doc in self.documents]

        embeddings = self.embeddings_model.encode(texts, convert_to_numpy=True)

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        query_embedding = self.embeddings_model.encode([query], convert_to_numpy=True)

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        return results

    def ask(self, question: str) -> str:
        results = self.search(question, k=3)

        context = "\n\n".join([r["content"] for r in results])

        prompt = f"""Ты - эксперт по визуализации данных. Используй контекст ниже, чтобы ответить на вопрос пользователя о выборе типа графика.

Контекст из базы знаний:
{context}

Вопрос: {question}

Отвечай на русском языке. Если в контексте есть подходящие рекомендации, обязательно укажи:
- Какой тип графика лучше использовать
- Почему он подходит для этой задачи
- Пример кода на Python

Если точного ответа нет в контексте, дай рекомендацию на основе твоих знаний.
"""

        response = ollama.generate(model=self.llm_model, prompt=prompt)
        return response["response"]


def initialize_rag() -> VisualizationRAG:
    rag = VisualizationRAG()
    rag.create_vector_store()
    return rag
