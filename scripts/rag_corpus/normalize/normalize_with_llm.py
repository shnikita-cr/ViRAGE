from __future__ import annotations

from collections.abc import Iterable

from scripts.rag_corpus.common.schemas import SourceRecord


def build_prompt(record: SourceRecord, allowed_record_types: Iterable[str]) -> str:
    allowed = ", ".join(sorted(str(item) for item in allowed_record_types))
    is_wilke = str(record.source_dataset).strip().lower() == "wilke_fundamentals"
    count_instruction = "Return a JSON array with 0 to 5 rule records" if is_wilke else "Return a JSON array with 1 to 4 rule records"
    zero_instruction = "Return [] if the source text is bibliography, citation-only, navigation, or contains no actionable visualization guidance."
    return "\n".join([
        "Convert the source visualization text into atomic ViRAGE RAG rule records.",
        count_instruction + ".",
        "If multiple recommendations are present, split them into separate atomic records.",
        zero_instruction if is_wilke else "Do not invent rules that are not supported by the source text.",
        f"Allowed record_type values: {allowed}.",
        f"Preferred record_type: {record.metadata.get('preferred_record_type', '')}",
        f"Title: {record.title}",
        f"Source dataset: {record.source_dataset}",
        "Source text:",
        record.text,
    ])
