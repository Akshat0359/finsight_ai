"""
FinSight AI — Section-Aware Text Chunker
Detects 10-K/10-Q/8-K sections by headers and chunks each separately.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# SEC 10-K section header patterns
SECTION_PATTERNS: list[tuple[str, str]] = [
    (r"(?i)item\s+1[.\s]+business", "ITEM_1_BUSINESS"),
    (r"(?i)item\s+1a[.\s]+risk\s+factors", "ITEM_1A_RISK_FACTORS"),
    (r"(?i)item\s+1b[.\s]+unresolved\s+staff", "ITEM_1B_STAFF_COMMENTS"),
    (r"(?i)item\s+2[.\s]+properties", "ITEM_2_PROPERTIES"),
    (r"(?i)item\s+3[.\s]+legal\s+proceedings", "ITEM_3_LEGAL"),
    (r"(?i)item\s+4[.\s]+mine\s+safety", "ITEM_4_MINE_SAFETY"),
    (r"(?i)item\s+5[.\s]+market\s+for", "ITEM_5_MARKET"),
    (r"(?i)item\s+6[.\s]+selected\s+financial", "ITEM_6_SELECTED"),
    (r"(?i)item\s+7[.\s]+management", "ITEM_7_MDA"),
    (r"(?i)item\s+7a[.\s]+quantitative", "ITEM_7A_QUANT"),
    (r"(?i)item\s+8[.\s]+financial\s+statements", "ITEM_8_FINANCIALS"),
    (r"(?i)item\s+9[.\s]+changes", "ITEM_9_CHANGES"),
    (r"(?i)item\s+9a[.\s]+controls", "ITEM_9A_CONTROLS"),
    (r"(?i)item\s+10[.\s]+directors", "ITEM_10_DIRECTORS"),
    (r"(?i)item\s+11[.\s]+executive\s+compensation", "ITEM_11_COMPENSATION"),
    (r"(?i)item\s+12[.\s]+security\s+ownership", "ITEM_12_SECURITY"),
    (r"(?i)item\s+13[.\s]+certain\s+relationships", "ITEM_13_RELATIONSHIPS"),
    (r"(?i)item\s+14[.\s]+principal\s+accountant", "ITEM_14_ACCOUNTANT"),
    (r"(?i)item\s+15[.\s]+exhibits", "ITEM_15_EXHIBITS"),
    # 10-Q sections
    (r"(?i)part\s+i[.\s]+financial\s+information", "PART_1_FINANCIAL"),
    (r"(?i)part\s+ii[.\s]+other\s+information", "PART_2_OTHER"),
    # 8-K sections
    (r"(?i)item\s+\d+\.\d+", "ITEM_OTHER"),
]


def detect_sections(text: str) -> list[dict[str, Any]]:
    """
    Split text into sections based on SEC filing section headers.
    Returns list of {section: str, text: str, start: int, end: int}.
    """
    if not text:
        return [{"section": "FULL_DOCUMENT", "text": text, "start": 0, "end": 0}]

    # Find all section boundaries
    boundaries: list[tuple[int, str]] = []
    for pattern, section_name in SECTION_PATTERNS:
        for match in re.finditer(pattern, text):
            boundaries.append((match.start(), section_name))

    if not boundaries:
        return [{"section": "FULL_DOCUMENT", "text": text, "start": 0, "end": len(text)}]

    # Sort by position
    boundaries.sort(key=lambda x: x[0])

    sections: list[dict[str, Any]] = []
    for i, (start, name) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        section_text = text[start:end].strip()
        if len(section_text) > 50:  # Skip empty/tiny sections
            sections.append({
                "section": name,
                "text": section_text,
                "start": start,
                "end": end,
            })

    # Add preamble if there's content before first section
    if boundaries[0][0] > 200:
        preamble = text[:boundaries[0][0]].strip()
        if preamble:
            sections.insert(0, {
                "section": "PREAMBLE",
                "text": preamble,
                "start": 0,
                "end": boundaries[0][0],
            })

    return sections


def chunk_text(
    text: str,
    section: str = "UNKNOWN",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict[str, Any]]:
    """
    Chunk a single section of text using RecursiveCharacterTextSplitter.
    Returns list of {text, section, chunk_index}.
    """
    _chunk_size = chunk_size or settings.CHUNK_SIZE
    _chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_chunk_size,
        chunk_overlap=_chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks_text = splitter.split_text(text)
    return [
        {"text": chunk_text, "section": section, "chunk_index": idx}
        for idx, chunk_text in enumerate(chunks_text)
        if chunk_text.strip()
    ]


def chunk_filing(
    text: str,
    form_type: str = "10-K",
    max_chars: int | None = None,
) -> list[dict[str, Any]]:
    """
    Full pipeline: section detection → per-section chunking.
    Limits input to max_chars to handle very large filings.
    Returns list of {text, section, chunk_index, form_type}.
    """
    _max_chars = max_chars or settings.MAX_FILING_CHARS

    # Trim very large filings
    if len(text) > _max_chars:
        logger.info(
            "Trimming filing from %d to %d chars", len(text), _max_chars
        )
        text = text[:_max_chars]

    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {3,}", " ", text)

    sections = detect_sections(text)
    all_chunks: list[dict[str, Any]] = []

    for section_info in sections:
        section_chunks = chunk_text(
            text=section_info["text"],
            section=section_info["section"],
        )
        for chunk in section_chunks:
            chunk["form_type"] = form_type
        all_chunks.extend(section_chunks)

    logger.info(
        "Chunked %s filing: %d sections → %d chunks",
        form_type, len(sections), len(all_chunks),
    )
    return all_chunks


def chunk_news_article(
    text: str,
    title: str = "",
    source: str = "",
    date: str = "",
) -> list[dict[str, Any]]:
    """
    Chunk a news article with metadata preserved.
    News articles are typically short, often fit in one chunk.
    """
    full_text = f"{title}\n\n{text}".strip() if title else text
    chunks = chunk_text(full_text, section="NEWS_ARTICLE")
    for chunk in chunks:
        chunk["source"] = source
        chunk["date"] = date
        chunk["title"] = title
    return chunks
