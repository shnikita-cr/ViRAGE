import os
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# Основные настройки
BASE_URL = "https://clauswilke.com/dataviz/"
OUTPUT_FILE = "../../../rag_corpus/raw_external_rules/wike/claus_wilke_dataviz_rag.txt"


def get_chapter_links():
    """Получает упорядоченный список ссылок на все главы книги."""
    print(f"Подключение к главной странице: {BASE_URL}")
    response = requests.get(BASE_URL)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    # Книга собрана на bookdown, ссылки на главы лежат в боковом меню (ul.summary)
    nav_menu = soup.find('ul', class_='summary')
    if not nav_menu:
        # Резервный поиск, если структура bookdown минимально изменилась
        nav_menu = soup.find('nav') or soup

    links = []
    for a in nav_menu.find_all('a', href=True):
        href = a['href']
        # Игнорируем внешние ссылки и якорные ссылки внутри одной страницы
        if href.startswith('http') or href.startswith('#') or href == './':
            continue

        full_url = urljoin(BASE_URL, href)
        if full_url not in links:
            links.append(full_url)

    # Добавляем главную страницу в начало, так как там находится Введение/Предисловие
    if BASE_URL not in links:
        links.insert(0, BASE_URL)

    return links


def extract_clean_text(html_content):
    """Очищает HTML и извлекает только смысловой текст главы."""
    soup = BeautifulSoup(html_content, 'html.parser')

    # В bookdown контент главы обычно обернут в класс 'page-inner', 'normal' или тег 'main'
    content_area = soup.find('div', class_='page-inner') or soup.find('section', class_='normal') or soup.find('main')

    if not content_area:
        content_area = soup.body  # Резервный вариант

    # Удаляем ненужные интерактивные элементы, если они есть внутри контента
    for element in content_area.find_all(['nav', 'script', 'style', 'noscript']):
        element.decompose()

    # Извлекаем текст с сохранением логических переносов строк
    text = content_area.get_text(separator="\n")

    # Базовая очистка от лишних пустых строк
    lines = [line.strip() for line in text.splitlines()]
    clean_lines = [line for line in lines if line]

    return "\n".join(clean_lines)


def main():
    try:
        chapter_urls = get_chapter_links()
        print(f"Найдено глав для скачивания: {len(chapter_urls)}")
    except Exception as e:
        print(f"Не удалось получить структуру книги: {e}")
        return

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for idx, url in enumerate(chapter_urls, 1):
            print(f"[{idx}/{len(chapter_urls)}] Скачивание: {url}")
            try:
                response = requests.get(url)
                response.raise_for_status()

                # Извлекаем чистый текст главы
                chapter_text = extract_clean_text(response.text)

                # Записываем в файл с четкими разделителями (полезно для Chunking в RAG)
                f.write(f"\n\n--- НАЧАЛО ГЛАВЫ: {url} ---\n\n")
                f.write(chapter_text)
                f.write(f"\n\n--- КОНЕЦ ГЛАВЫ ---\n")

                # Небольшая пауза, чтобы не нагружать сервер
                time.sleep(0.5)

            except Exception as e:
                print(f"Ошибка при обработке {url}: {e}")

    print(f"\nГотово! Полный текст книги сохранен в файл: {os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    main()
