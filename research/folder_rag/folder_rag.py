from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import chromadb
from ollama import Client, ResponseError


SUPPORTED_EXTENSIONS = {
    ".py", ".java", ".kt", ".kts", ".js", ".jsx", ".ts", ".tsx",
    ".html", ".css", ".scss", ".sass", ".json", ".yaml", ".yml",
    ".md", ".txt", ".sql", ".xml", ".properties", ".gradle", ".toml",
    ".ini", ".cfg", ".conf", ".sh", ".bash", ".zsh", ".bat", ".ps1",
    ".dockerfile", ".graphql", ".gql", ".proto", ".env.example",
}

SUPPORTED_FILENAMES = {
    "Dockerfile", "Containerfile", "Makefile", "README", "LICENSE", ".gitignore",
    ".dockerignore", ".env.example", "pom.xml", "build.gradle", "settings.gradle",
    "docker-compose.yml", "docker-compose.yaml",
}

IGNORED_DIR_NAMES = {
    ".git", ".hg", ".svn", ".idea", ".vscode", "node_modules", "target",
    "build", "dist", "out", ".next", ".nuxt", ".cache", ".gradle", ".mvn",
    ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".mypy_cache",
    "coverage", ".terraform", "temp",
}

IGNORED_FILE_EXTENSIONS = {
    ".csv", ".tsv", ".xlsx", ".xls", ".ods", ".parquet", ".db", ".sqlite",
    ".sqlite3", ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".zip", ".rar",
    ".7z", ".tar", ".gz", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
    ".ico", ".mp3", ".mp4", ".mov", ".avi", ".jar", ".war", ".class", ".exe",
    ".dll", ".so", ".dylib", ".bin", ".lock",
}


@dataclass(frozen=True)
class TextFile:
    abs_path: Path
    rel_path: str
    text: str
    sha256: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    rel_path: str
    start_line: int
    end_line: int
    text: str
    sha256: str


@dataclass(frozen=True)
class RetrievedChunk:
    rel_path: str
    start_line: int
    end_line: int
    text: str
    distance: float | None

    @property
    def source(self) -> str:
        return f"{self.rel_path}:{self.start_line}-{self.end_line}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local folder RAG over code/text files with Ollama and ChromaDB."
    )
    parser.add_argument("--folder", required=True, help="Folder to index and query.")
    parser.add_argument("--query", required=True, help="User question.")
    parser.add_argument("--mode", choices=["fast", "deep"], default="fast")
    parser.add_argument("--reindex", action="store_true", help="Rebuild Chroma index.")
    parser.add_argument("--model", default="qwen2.5-coder:7b", help="Ollama LLM model.")
    parser.add_argument("--embedding-model", default="nomic-embed-text", help="Ollama embedding model.")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--chunk-chars", type=int, default=3500)
    parser.add_argument("--overlap-lines", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-file-mb", type=float, default=2.0)
    parser.add_argument("--collection", default="folder_rag_chunks")
    parser.add_argument("--deep-file-limit", type=int, default=80)
    parser.add_argument("--deep-max-file-chars", type=int, default=12000)
    parser.add_argument("--memory-max-chars", type=int, default=60000)
    return parser.parse_args()


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def temp_dir() -> Path:
    path = script_dir() / "temp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def memory_path() -> Path:
    return temp_dir() / "memory.md"


def chroma_dir() -> Path:
    return temp_dir() / "chroma"


def manifest_path() -> Path:
    return temp_dir() / "index_manifest.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def ensure_memory(folder: Path) -> None:
    path = memory_path()
    if path.exists():
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(
        "# Working memory\n\n"
        f"Created: {now}\n\n"
        f"Folder: `{folder}`\n\n"
        "This file is a compressed working memory for recursive analysis. "
        "It is not treated as a source of truth. Final answers must be grounded "
        "in retrieved source chunks with file paths and line ranges.\n\n",
        encoding="utf-8",
    )


def append_memory(section_title: str, body: str) -> None:
    path = memory_path()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as file:
        file.write(f"\n## {section_title}\n\n")
        file.write(f"Timestamp: {timestamp}\n\n")
        file.write(body.strip() + "\n")


def read_memory_tail(max_chars: int = 12000) -> str:
    path = memory_path()
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[-max_chars:]


def normalize_rel_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def is_ignored_dir(path: Path) -> bool:
    return path.name in IGNORED_DIR_NAMES


def is_supported_file(path: Path) -> bool:
    if path.name in SUPPORTED_FILENAMES:
        return True
    lower_name = path.name.lower()
    suffix = path.suffix.lower()
    if lower_name.endswith(".env.example"):
        return True
    if suffix in IGNORED_FILE_EXTENSIONS:
        return False
    return suffix in SUPPORTED_EXTENSIONS


def looks_binary(data: bytes) -> bool:
    if b"\x00" in data[:4096]:
        return True
    if not data:
        return False
    sample = data[:4096]
    control_chars = sum(1 for byte in sample if byte < 9 or (13 < byte < 32))
    return control_chars / max(len(sample), 1) > 0.20


def decode_text(data: bytes) -> str | None:
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def collect_text_files(root: Path, max_file_mb: float) -> list[TextFile]:
    max_bytes = int(max_file_mb * 1024 * 1024)
    files: list[TextFile] = []
    for current_root, dir_names, file_names in os.walk(root):
        current_path = Path(current_root)
        dir_names[:] = [name for name in dir_names if not is_ignored_dir(current_path / name)]
        for file_name in sorted(file_names):
            abs_path = current_path / file_name
            if not is_supported_file(abs_path):
                continue
            try:
                size = abs_path.stat().st_size
            except OSError:
                continue
            if size > max_bytes:
                continue
            try:
                data = abs_path.read_bytes()
            except OSError:
                continue
            if looks_binary(data):
                continue
            text = decode_text(data)
            if not text or not text.strip():
                continue
            rel_path = normalize_rel_path(abs_path, root)
            files.append(TextFile(abs_path=abs_path, rel_path=rel_path, text=text, sha256=sha256_bytes(data)))
    return files


def split_file_to_chunks(file: TextFile, max_chars: int, overlap_lines: int) -> list[Chunk]:
    lines = file.text.splitlines()
    chunks: list[Chunk] = []
    start = 0
    while start < len(lines):
        current: list[str] = []
        end = start
        current_chars = 0
        while end < len(lines):
            line = lines[end]
            line_chars = len(line) + 1
            if current and current_chars + line_chars > max_chars:
                break
            current.append(line)
            current_chars += line_chars
            end += 1
        text = "\n".join(current).strip()
        if text:
            start_line = start + 1
            end_line = end
            raw_id = f"{file.rel_path}:{start_line}-{end_line}:{file.sha256}"
            chunk_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    rel_path=file.rel_path,
                    start_line=start_line,
                    end_line=end_line,
                    text=text,
                    sha256=file.sha256,
                )
            )
        if end >= len(lines):
            break
        start = max(end - overlap_lines, start + 1)
    return chunks


def build_chunks(files: Iterable[TextFile], max_chars: int, overlap_lines: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for file in files:
        chunks.extend(split_file_to_chunks(file, max_chars=max_chars, overlap_lines=overlap_lines))
    return chunks


def batch_items(items: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(items), size):
        yield items[index:index + size]


def get_response_content(response: Any) -> str:
    try:
        return response["message"]["content"]
    except Exception:
        return response.message.content


def get_embeddings(response: Any) -> list[list[float]]:
    try:
        return response["embeddings"]
    except Exception:
        return response.embeddings


def embed_texts(client: Client, model: str, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    response = client.embed(model=model, input=texts)
    embeddings = get_embeddings(response)
    if len(embeddings) != len(texts):
        raise RuntimeError(f"Embedding count mismatch: {len(embeddings)} != {len(texts)}")
    return embeddings


def chat(client: Client, model: str, messages: list[dict[str, str]]) -> str:
    response = client.chat(model=model, messages=messages, stream=False)
    return get_response_content(response).strip()


def reset_chroma(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def open_collection(collection_name: str):
    client = chromadb.PersistentClient(path=str(chroma_dir()))
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def load_manifest() -> dict[str, Any] | None:
    path = manifest_path()
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(folder: Path, args: argparse.Namespace, files_count: int, chunks_count: int) -> None:
    payload = {
        "folder": str(folder.resolve()),
        "collection": args.collection,
        "embedding_model": args.embedding_model,
        "chunk_chars": args.chunk_chars,
        "overlap_lines": args.overlap_lines,
        "files_count": files_count,
        "chunks_count": chunks_count,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path().write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_manifest(folder: Path, args: argparse.Namespace) -> None:
    manifest = load_manifest()
    if manifest is None:
        return
    problems: list[str] = []
    if manifest.get("folder") != str(folder.resolve()):
        problems.append("folder differs from the folder used to build the index")
    if manifest.get("embedding_model") != args.embedding_model:
        problems.append("embedding model differs from the indexed embedding model")
    if manifest.get("collection") != args.collection:
        problems.append("Chroma collection name differs from the saved collection name")
    if problems:
        joined = "; ".join(problems)
        raise RuntimeError(f"The index cannot be safely reused: {joined}. Run with --reindex.")


def index_chunks(collection: Any, chunks: list[Chunk], client: Client, args: argparse.Namespace) -> None:
    for batch in batch_items(chunks, args.batch_size):
        texts = [chunk.text for chunk in batch]
        embeddings = embed_texts(client=client, model=args.embedding_model, texts=texts)
        collection.add(
            ids=[chunk.chunk_id for chunk in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {
                    "rel_path": chunk.rel_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "sha256": chunk.sha256,
                }
                for chunk in batch
            ],
        )


def build_or_reuse_index(folder: Path, client: Client, args: argparse.Namespace) -> tuple[Any, list[TextFile], bool]:
    should_build = args.reindex or not chroma_dir().exists() or not manifest_path().exists()
    if should_build:
        reset_chroma(chroma_dir())
    else:
        validate_manifest(folder, args)

    collection = open_collection(args.collection)
    if not should_build and collection.count() > 0:
        return collection, [], False

    files = collect_text_files(folder, max_file_mb=args.max_file_mb)
    chunks = build_chunks(files, max_chars=args.chunk_chars, overlap_lines=args.overlap_lines)
    if not chunks:
        raise RuntimeError("No suitable text/code files were found for indexing.")

    index_chunks(collection=collection, chunks=chunks, client=client, args=args)
    save_manifest(folder=folder, args=args, files_count=len(files), chunks_count=len(chunks))
    append_memory(
        "Index build",
        f"Indexed folder: `{folder}`\n\nFiles: {len(files)}\n\nChunks: {len(chunks)}\n\n"
        f"Embedding model: `{args.embedding_model}`\n",
    )
    return collection, files, True


def compact_memory_if_needed(client: Client, args: argparse.Namespace) -> None:
    path = memory_path()
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8", errors="ignore")
    if len(text) <= args.memory_max_chars:
        return
    prompt = (
        "Compress the project working memory. Keep only verifiable facts, architectural conclusions, "
        "important decisions, risks, questions, and file/line references if present. "
        "Do not add new facts. Write in English only. If the existing memory contains non-English text, translate the retained facts to English.\n\n"
        f"CURRENT MEMORY:\n{text[-args.memory_max_chars * 2:]}"
    )
    compacted = chat(
        client=client,
        model=args.model,
        messages=[
            {"role": "system", "content": "You carefully compress the working memory of a RAG analysis."},
            {"role": "user", "content": prompt},
        ],
    )
    path.write_text("# Working memory\n\n" + compacted.strip() + "\n", encoding="utf-8")


def deep_analyze_files(files: list[TextFile], client: Client, args: argparse.Namespace, query: str) -> None:
    if not files:
        append_memory(
            "Deep mode note",
            "Index was reused, so full recursive file analysis was not repeated. "
            "The current session still uses retrieved source chunks for the final answer.",
        )
        return

    selected = sorted(files, key=lambda file: (len(file.rel_path), file.rel_path))[:args.deep_file_limit]
    append_memory("Deep analysis start", f"Files selected for deep analysis: {len(selected)} of {len(files)}")

    for index, file in enumerate(selected, start=1):
        memory_tail = read_memory_tail(max_chars=7000)
        file_text = file.text[:args.deep_max_file_chars]
        prompt = (
            "Analyze this file for the working memory of a local folder RAG system. "
            "Extract only important and verifiable facts. "
            "Consider the user question, but do not invent an answer.\n\n"
            f"USER QUESTION:\n{query}\n\n"
            f"CURRENT MEMORY, MAY BE USED AS CONTEXT:\n{memory_tail}\n\n"
            f"FILE: {file.rel_path}\n"
            f"SHA256: {file.sha256}\n\n"
            f"FILE CONTENT, POSSIBLY TRUNCATED:\n{file_text}\n\n"
            "Return briefly:\n"
            "- file purpose;\n"
            "- key classes/functions/contracts;\n"
            "- important dependencies;\n"
            "- facts useful for the user question;\n"
            "- potential issues if they directly follow from the code.\n"
            "Each item must reference the file, for example `path/to/file.py`."
        )
        try:
            result = chat(
                client=client,
                model=args.model,
                messages=[
                    {"role": "system", "content": "You are a senior developer analyzing a codebase for RAG memory. Write in English only."},
                    {"role": "user", "content": prompt},
                ],
            )
            append_memory(f"File analysis {index}: {file.rel_path}", result)
            compact_memory_if_needed(client, args)
        except ResponseError as error:
            append_memory(f"File analysis failed: {file.rel_path}", f"Ollama error: {error}")


def retrieve(collection: Any, client: Client, args: argparse.Namespace, query: str) -> list[RetrievedChunk]:
    query_embedding = embed_texts(client=client, model=args.embedding_model, texts=[query])[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=args.top_k,
        include=["documents", "metadatas", "distances"],
    )
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    retrieved: list[RetrievedChunk] = []
    seen: set[str] = set()
    for document, metadata, distance in zip(documents, metadatas, distances):
        rel_path = str(metadata["rel_path"])
        start_line = int(metadata["start_line"])
        end_line = int(metadata["end_line"])
        key = f"{rel_path}:{start_line}-{end_line}"
        if key in seen:
            continue
        seen.add(key)
        retrieved.append(
            RetrievedChunk(
                rel_path=rel_path,
                start_line=start_line,
                end_line=end_line,
                text=document,
                distance=float(distance) if distance is not None else None,
            )
        )
    return retrieved


def format_context(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[S{index}] {chunk.source}\n"
            f"<source_text>\n{chunk.text}\n</source_text>"
        )
    return "\n\n".join(parts)


def build_detailed_answer(client: Client, args: argparse.Namespace, query: str, chunks: list[RetrievedChunk]) -> str:
    memory_tail = read_memory_tail(max_chars=14000)
    context = format_context(chunks)
    prompt = (
        "Answer in English only, regardless of the user question language. "
        "Use the retrieved sources as the primary source of truth. "
        "Use the working memory only as additional guidance. "
        "Do not invent facts that are not present in the sources. "
        "Support every important claim with a source reference in the format `[path/to/file.ext:start-end]`. "
        "If the available data is insufficient, clearly state what is missing.\n\n"
        "Answer format:\n"
        "# Short answer\n"
        "# Detailed answer\n"
        "# Source evidence\n"
        "# Answer limitations\n\n"
        f"QUESTION:\n{query}\n\n"
        f"WORKING MEMORY.MD, NOT A SOURCE OF TRUTH:\n{memory_tail}\n\n"
        f"RETRIEVED SOURCES:\n{context}\n"
    )
    return chat(
        client=client,
        model=args.model,
        messages=[
            {"role": "system", "content": "You are a senior developer and codebase analyst. Answer strictly from the sources. Write in English only."},
            {"role": "user", "content": prompt},
        ],
    )


def build_short_answer(client: Client, args: argparse.Namespace, query: str, detailed_answer: str) -> str:
    prompt = (
        "Create a short plain-text answer in English only, regardless of the user question language. "
        "Length: 1-3 sentences. Keep the main conclusion and do not add new facts.\n\n"
        f"QUESTION:\n{query}\n\n"
        f"DETAILED ANSWER:\n{detailed_answer}"
    )
    return chat(
        client=client,
        model=args.model,
        messages=[
            {"role": "system", "content": "You compress an answer without losing meaning."},
            {"role": "user", "content": prompt},
        ],
    )


def save_result(
    folder: Path,
    args: argparse.Namespace,
    query: str,
    short_answer: str,
    detailed_answer: str,
    chunks: list[RetrievedChunk],
) -> Path:
    base_name = f"result_{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
    output_path = script_dir() / f"{base_name}.md"
    counter = 2
    while output_path.exists():
        output_path = script_dir() / f"{base_name}_{counter}.md"
        counter += 1
    sources = "\n".join(
        f"- `{chunk.source}`"
        + (f"; distance={chunk.distance:.4f}" if chunk.distance is not None else "")
        for chunk in chunks
    )
    content = (
        f"# Folder RAG result\n\n"
        f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"Folder: `{folder}`\n\n"
        f"Mode: `{args.mode}`\n\n"
        f"LLM model: `{args.model}`\n\n"
        f"Embedding model: `{args.embedding_model}`\n\n"
        f"Query: {query}\n\n"
        f"## Short answer\n\n{short_answer}\n\n"
        f"## Detailed answer\n\n{detailed_answer}\n\n"
        f"## Retrieved sources\n\n{sources}\n"
    )
    output_path.write_text(content, encoding="utf-8")
    return output_path


def update_memory_after_answer(query: str, short_answer: str, result_path: Path, chunks: list[RetrievedChunk]) -> None:
    source_lines = "\n".join(f"- `{chunk.source}`" for chunk in chunks)
    append_memory(
        "Answered query",
        f"Query: {query}\n\nShort answer:\n{short_answer}\n\nResult file: `{result_path.name}`\n\nSources:\n{source_lines}",
    )


def main() -> int:
    args = parse_args()
    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"Error: folder was not found or is not a directory: {folder}", file=sys.stderr)
        return 2

    ensure_memory(folder)
    client = Client(host=args.ollama_host)

    try:
        collection, files, built_now = build_or_reuse_index(folder=folder, client=client, args=args)
        print(f"Index: {'rebuilt' if built_now else 'reused'}, chunks={collection.count()}")

        if args.mode == "deep":
            deep_analyze_files(files=files, client=client, args=args, query=args.query)

        chunks = retrieve(collection=collection, client=client, args=args, query=args.query)
        if not chunks:
            raise RuntimeError("ChromaDB did not return any relevant chunks.")

        detailed_answer = build_detailed_answer(client=client, args=args, query=args.query, chunks=chunks)
        short_answer = build_short_answer(client=client, args=args, query=args.query, detailed_answer=detailed_answer)
        result_path = save_result(
            folder=folder,
            args=args,
            query=args.query,
            short_answer=short_answer,
            detailed_answer=detailed_answer,
            chunks=chunks,
        )
        update_memory_after_answer(
            query=args.query,
            short_answer=short_answer,
            result_path=result_path,
            chunks=chunks,
        )
        compact_memory_if_needed(client, args)

        print("\nShort answer:")
        print(short_answer)
        print(f"\nDetailed answer saved to: {result_path}")
        print(f"Working memory: {memory_path()}")
        return 0
    except ResponseError as error:
        print(f"Ollama error: {error}", file=sys.stderr)
        print("Check that Ollama is running and the models are available:", file=sys.stderr)
        print(f"  ollama pull {args.model}", file=sys.stderr)
        print(f"  ollama pull {args.embedding_model}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
