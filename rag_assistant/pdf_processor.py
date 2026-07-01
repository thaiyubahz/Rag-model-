"""
pdf_processor.py
-----------------
Extracts text from an uploaded PDF and splits it into overlapping chunks
suitable for embedding. Uses pypdf only (pure Python, no external service).
"""

from typing import List, Dict, Any
from io import BytesIO

from pypdf import PdfReader


DEFAULT_CHUNK_WORDS = 180
DEFAULT_OVERLAP_WORDS = 30


def extract_pages(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Returns a list of {"page": int, "text": str} for every page with
    extractable text. Pages that fail to extract (scanned/image-only)
    are skipped rather than raising, so one bad page doesn't kill ingestion.
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages


def chunk_text(
    text: str,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> List[str]:
    """
    Splits text into overlapping word-count windows. Word-count chunking
    (rather than character count) keeps chunks semantically coherent
    enough for MiniLM embeddings without needing a sentence tokenizer.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(chunk_words - overlap_words, 1)
    for start in range(0, len(words), step):
        window = words[start:start + chunk_words]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_words >= len(words):
            break
    return chunks


def process_pdf(pdf_bytes: bytes, source_title: str) -> List[Dict[str, Any]]:
    """
    Full pipeline: PDF bytes -> list of chunk dicts ready for
    VectorStore.add_chunks(), each tagged with its source paper and page.
    """
    pages = extract_pages(pdf_bytes)
    all_chunks: List[Dict[str, Any]] = []

    for page in pages:
        for chunk in chunk_text(page["text"]):
            all_chunks.append({
                "text": chunk,
                "source": source_title,
                "page": page["page"],
            })

    return all_chunks
