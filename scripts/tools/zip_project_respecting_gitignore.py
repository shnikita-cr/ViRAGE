from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
import argparse
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

def run_git_ls_files(project_root: Path, include_untracked: bool) -> list[Path]:
    command = ['git', 'ls-files', '--cached', '--exclude-standard', '-z']
    if include_untracked:
        command.append('--others')
    result = subprocess.run(command, cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode('utf-8', errors='replace').strip()
        raise RuntimeError(f'Failed to read project files via git ls-files. Make sure the target folder is a Git repository and Git is installed.\nGit error: {stderr}')
    raw_paths = result.stdout.split(b'\x00')
    files: list[Path] = []
    for raw_path in raw_paths:
        if not raw_path:
            continue
        relative_path = Path(raw_path.decode('utf-8', errors='replace'))
        absolute_path = project_root / relative_path
        if absolute_path.is_file():
            files.append(relative_path)
    return files

def is_same_or_inside(path: Path, possible_parent: Path) -> bool:
    try:
        path.resolve().relative_to(possible_parent.resolve())
        return True
    except ValueError:
        return False

def create_zip(project_root: Path, output_zip: Path, files: list[Path], compression_level: int, dry_run: bool) -> None:
    output_zip = output_zip.resolve()
    project_root = project_root.resolve()
    files_to_zip: list[Path] = []
    for relative_path in files:
        absolute_path = project_root / relative_path
        if absolute_path.resolve() == output_zip:
            continue
        if absolute_path.is_file():
            files_to_zip.append(relative_path)
    if dry_run:
        logger.info(f'Project root: {project_root}')
        logger.info(f'Output zip:   {output_zip}')
        logger.info(f'Files:        {len(files_to_zip)}')
        logger.info('')
        for relative_path in files_to_zip:
            logger.info(relative_path.as_posix())
        return
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, mode='w', compression=zipfile.ZIP_DEFLATED, compresslevel=compression_level) as archive:
        for relative_path in files_to_zip:
            absolute_path = project_root / relative_path
            if not absolute_path.exists():
                logger.error(f'Warning: skipped missing file: {relative_path}')
                continue
            archive.write(filename=absolute_path, arcname=relative_path.as_posix())
    logger.info(f'Created zip: {output_zip}')
    logger.info(f'Files added: {len(files_to_zip)}')

def default_output_path(project_root: Path) -> Path:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    project_name = project_root.resolve().name
    return project_root.parent / f'{project_name}_{timestamp}.zip'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create a ZIP archive of a Git project while respecting .gitignore, nested .gitignore files, .git/info/exclude, and global git excludes.')
    parser.add_argument('project_root', nargs='?', default='.', help='Project root directory. Default: current directory.')
    parser.add_argument('-o', '--output', default=None, help='Output ZIP path. Default: ../<project_name>_<timestamp>.zip')
    parser.add_argument('--tracked-only', action='store_true', help='Include only tracked Git files. By default, tracked + untracked non-ignored files are included.')
    parser.add_argument('--dry-run', action='store_true', help='Print files that would be archived without creating ZIP.')
    parser.add_argument('--compression-level', type=int, default=6, choices=range(0, 10), metavar='0-9', help='ZIP compression level. 0 = no compression, 9 = maximum. Default: 6.')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        raise FileNotFoundError(f'Project root does not exist: {project_root}')
    if not project_root.is_dir():
        raise NotADirectoryError(f'Project root is not a directory: {project_root}')
    output_zip = Path(args.output).resolve() if args.output else default_output_path(project_root)
    include_untracked = not args.tracked_only
    files = run_git_ls_files(project_root=project_root, include_untracked=include_untracked)
    create_zip(project_root=project_root, output_zip=output_zip, files=files, compression_level=args.compression_level, dry_run=args.dry_run)
if __name__ == '__main__':
    main()
