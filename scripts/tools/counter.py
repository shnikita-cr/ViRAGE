from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd


def count_lines_in_file(file_path: Path, encoding: str = 'utf-8') -> int:
    """Возвращает количество строк в текстовом файле."""
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return sum(1 for _ in f)
    except (UnicodeDecodeError, IsADirectoryError, PermissionError):
        return 0


def build_tree(root_dir: Union[str, Path],
               extensions: Optional[List[str]] = None,
               skip_hidden: bool = True) -> Dict:
    """
    Рекурсивно строит дерево файлов и директорий со статистикой строк.

    Возвращает словарь вида:
    {
        'name': 'src',
        'type': 'dir',
        'path': 'абсолютный_путь',
        'lines': 1234,
        'children': { ... }
    }
    """
    root_path = Path(root_dir).resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"{root_path} не является директорией")

    tree = {
        'name': root_path.name,
        'type': 'dir',
        'path': str(root_path),
        'lines': 0,
        'children': {}
    }

    for item in root_path.iterdir():
        if skip_hidden and item.name.startswith('.'):
            continue

        if item.is_dir():
            sub_tree = build_tree(item, extensions, skip_hidden)
            tree['children'][item.name] = sub_tree
            tree['lines'] += sub_tree['lines']

        elif item.is_file():
            if extensions is not None and item.suffix.lower() not in extensions:
                continue
            lines = count_lines_in_file(item)
            tree['children'][item.name] = {
                'type': 'file',
                'name': item.name,
                'path': str(item),
                'lines': lines
            }
            tree['lines'] += lines

    return tree


def tree_to_dataframe(tree: Dict) -> pd.DataFrame:
    """
    Преобразует дерево статистики в плоский DataFrame с колонками:
        - path (полный путь)
        - name (имя элемента)
        - type ('dir' или 'file')
        - lines (сумма строк для директорий, число строк для файлов)
    """
    rows = []

    def dfs(node, parent_path=None):
        # Сохраняем текущий узел
        rows.append({
            'path': node['path'],
            'name': node['name'],
            'type': node['type'],
            'lines': node['lines']
        })
        # Рекурсивно обходим детей
        if node['type'] == 'dir':
            for child_node in node['children'].values():
                dfs(child_node)

    dfs(tree)
    return pd.DataFrame(rows)


def print_tree(tree: Dict, indent: int = 0) -> None:
    """Красиво выводит дерево статистики."""
    prefix = '  ' * indent
    if tree['type'] == 'dir':
        print(f"{prefix}📁 {tree['name']} (total lines: {tree['lines']})")
        for child in tree['children'].values():
            print_tree(child, indent + 1)
    else:
        print(f"{prefix}📄 {tree['name']} - {tree['lines']} lines")


# ========== ПРИМЕР ИСПОЛЬЗОВАНИЯ ==========
if __name__ == '__main__':
    # 1. Строим дерево (например, для папки 'src', только .py файлы)
    stats = build_tree('src', extensions=['.py'])

    # 2. Выводим дерево в консоль
    print_tree(stats)

    # 3. Преобразуем в DataFrame
    df = tree_to_dataframe(stats)

    # 4. Показываем результат
    print("\nDataFrame со статистикой:")
    print(df.to_string(index=False))

    # 5. Пример агрегации: общее количество строк в src
    total_lines = df[df['type'] == 'dir'].iloc[0]['lines']  # корневой элемент
    print(f"\nОбщее количество строк в src: {total_lines}")

    # 6. Сохранить в CSV при желании
    df.to_csv('code_stats.csv', index=False, encoding='utf-8')
