import logging
logger = logging.getLogger(__name__)
import os
import time
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
BASE_URL = 'https://clauswilke.com/dataviz/'
OUTPUT_FILE = '../../../rag_corpus/raw_external_rules/wike/claus_wilke_dataviz_rag.txt'

def get_chapter_links():
    """Получает упорядоченный список ссылок на все главы книги."""
    logger.info(f'Подключение к главной странице: {BASE_URL}')
    response = requests.get(BASE_URL)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    nav_menu = soup.find('ul', class_='summary')
    if not nav_menu:
        nav_menu = soup.find('nav') or soup
    links = []
    for a in nav_menu.find_all('a', href=True):
        href = a['href']
        if href.startswith('http') or href.startswith('#') or href == './':
            continue
        full_url = urljoin(BASE_URL, href)
        if full_url not in links:
            links.append(full_url)
    if BASE_URL not in links:
        links.insert(0, BASE_URL)
    return links

def extract_clean_text(html_content):
    """Очищает HTML и извлекает только смысловой текст главы."""
    soup = BeautifulSoup(html_content, 'html.parser')
    content_area = soup.find('div', class_='page-inner') or soup.find('section', class_='normal') or soup.find('main')
    if not content_area:
        content_area = soup.body
    for element in content_area.find_all(['nav', 'script', 'style', 'noscript']):
        element.decompose()
    text = content_area.get_text(separator='\n')
    lines = [line.strip() for line in text.splitlines()]
    clean_lines = [line for line in lines if line]
    return '\n'.join(clean_lines)

def main():
    try:
        chapter_urls = get_chapter_links()
        logger.info(f'Найдено глав для скачивания: {len(chapter_urls)}')
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, IndexError, AttributeError, ImportError) as e:
        logger.info(f'Не удалось получить структуру книги: {e}')
        return
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for idx, url in enumerate(chapter_urls, 1):
            logger.info(f'[{idx}/{len(chapter_urls)}] Скачивание: {url}')
            try:
                response = requests.get(url)
                response.raise_for_status()
                chapter_text = extract_clean_text(response.text)
                f.write(f'\n\n--- НАЧАЛО ГЛАВЫ: {url} ---\n\n')
                f.write(chapter_text)
                f.write(f'\n\n--- КОНЕЦ ГЛАВЫ ---\n')
                time.sleep(0.5)
            except (RuntimeError, ValueError, TypeError, OSError, KeyError, IndexError, AttributeError, ImportError) as e:
                logger.info(f'Ошибка при обработке {url}: {e}')
    logger.info(f'\nГотово! Полный текст книги сохранен в файл: {os.path.abspath(OUTPUT_FILE)}')
if __name__ == '__main__':
    main()
