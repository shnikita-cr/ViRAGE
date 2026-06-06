from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any
from scripts.rag_corpus.common.io import ensure_dir, project_root, write_json, write_text
DEFAULT_RAW_ROOT = 'rag_corpus/raw'
DEFAULT_OUT_JSON = 'rag_corpus/manifests/raw_inventory.json'
DEFAULT_OUT_MD = 'rag_corpus/manifests/source_inventory.md'

def scan_sources(raw_root: Path) -> dict[str, Any]:
    inventory: dict[str, Any] = {'raw_root': str(raw_root), 'sources': {}}
    if not raw_root.exists():
        return inventory
    for source_dir in sorted((path for path in raw_root.iterdir() if path.is_dir())):
        by_suffix: dict[str, int] = defaultdict(int)
        files: list[dict[str, Any]] = []
        for file_path in sorted((path for path in source_dir.rglob('*') if path.is_file())):
            suffix = file_path.suffix.lower() or '<none>'
            by_suffix[suffix] += 1
            files.append({'path': str(file_path), 'relative_path': str(file_path.relative_to(raw_root)), 'suffix': suffix, 'size_bytes': file_path.stat().st_size})
        inventory['sources'][source_dir.name] = {'file_count': len(files), 'by_suffix': dict(sorted(by_suffix.items())), 'files': files[:200], 'truncated': len(files) > 200}
    return inventory

def inventory_markdown(inventory: dict[str, Any]) -> str:
    lines = ['# RAG raw source inventory', '', f"Raw root: `{inventory.get('raw_root')}`", '']
    sources = inventory.get('sources', {})
    if not sources:
        lines.append('No raw sources found.')
        return '\n'.join(lines) + '\n'
    lines.extend(['| Source | Files | Types |', '|---|---:|---|'])
    for name, info in sources.items():
        types = ', '.join((f'{key}: {value}' for key, value in info.get('by_suffix', {}).items())) or '-'
        lines.append(f"| `{name}` | {info.get('file_count', 0)} | {types} |")
    lines.append('')
    return '\n'.join(lines) + '\n'

def main() -> None:
    parser = argparse.ArgumentParser(description='Scan rag_corpus/raw and create source inventory reports.')
    parser.add_argument('--raw-root', default=DEFAULT_RAW_ROOT)
    parser.add_argument('--out-json', default=DEFAULT_OUT_JSON)
    parser.add_argument('--out-md', default=DEFAULT_OUT_MD)
    args = parser.parse_args()
    root = project_root()
    raw_root = (root / args.raw_root).resolve()
    inventory = scan_sources(raw_root)
    write_json(root / args.out_json, inventory)
    write_text(root / args.out_md, inventory_markdown(inventory))
    logger.info(f'Wrote {args.out_json} and {args.out_md}')
if __name__ == '__main__':
    main()
