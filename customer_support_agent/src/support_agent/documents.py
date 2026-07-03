from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocumentChunk:
    document_id: str
    title: str
    source_path: str
    chunk_index: int
    text: str


def load_markdown_documents(sop_dir: Path) -> list[tuple[Path, str, str]]:
    if not sop_dir.exists():
        raise FileNotFoundError(f"SOP directory not found: {sop_dir}")

    docs: list[tuple[Path, str, str]] = []
    for path in sorted(sop_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        title = _extract_title(text) or path.stem.replace("_", " ").title()
        docs.append((path, title, _clean_text(text)))
    if not docs:
        raise ValueError(f"No Markdown SOP files found in {sop_dir}")
    return docs


def chunk_documents(
    sop_dir: Path,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for path, title, text in load_markdown_documents(sop_dir):
        parts = _split_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
        document_id = path.stem
        for idx, part in enumerate(parts):
            chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    title=title,
                    source_path=str(path),
                    chunk_index=idx,
                    text=part,
                )
            )
    return chunks


def _extract_title(text: str) -> str | None:
    match = re.search(r"^\s*#\s+(.+?)\s*$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _clean_text(text: str) -> str:
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    sections = re.split(r"(?=^##\s+)", text, flags=re.MULTILINE)
    raw_chunks: list[str] = []
    current = ""
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if current and len(current) + len(section) + 2 > chunk_size:
            raw_chunks.append(current.strip())
            current = _tail(current, overlap)
        current = f"{current}\n\n{section}".strip()
    if current:
        raw_chunks.append(current.strip())

    final_chunks: list[str] = []
    for chunk in raw_chunks:
        if len(chunk) <= chunk_size * 1.3:
            final_chunks.append(chunk)
            continue
        for start in range(0, len(chunk), max(1, chunk_size - overlap)):
            final_chunks.append(chunk[start : start + chunk_size].strip())
    return [chunk for chunk in final_chunks if chunk]


def _tail(text: str, overlap: int) -> str:
    if overlap <= 0:
        return ""
    return text[-overlap:].strip()
