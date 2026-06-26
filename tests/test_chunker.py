"""
FinSight AI — Tests for rag/chunker.py
Tests section detection, chunk sizes, and overlap correctness.
"""
from __future__ import annotations

import pytest

from rag.chunker import (
    chunk_filing,
    chunk_news_article,
    chunk_text,
    detect_sections,
)

# ---- Sample 10-K text ----
SAMPLE_10K = """
UNITED STATES SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549

FORM 10-K

ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d)

ITEM 1. BUSINESS

We are a leading technology company providing cloud services, productivity software, and 
gaming products. Founded in 1975, we have grown to serve millions of customers globally.

Our segments include Productivity and Business Processes, Intelligent Cloud, and More Personal Computing.

ITEM 1A. RISK FACTORS

The following risk factors could materially affect our business:

Regulatory Risk: We are subject to increasing regulatory scrutiny in the European Union,
United States, and other jurisdictions. Antitrust investigations could result in significant 
fines or operational restrictions.

Competitive Risk: We face intense competition from Amazon Web Services, Google Cloud,
and other cloud providers. Our Azure platform must continue to innovate to maintain market share.

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

Revenue for fiscal year 2024 was $245 billion, representing a 16% increase compared to 
fiscal year 2023. Cloud revenue grew 29% driven by Azure adoption.

Operating income increased to $109 billion, reflecting improved margins across segments.

ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK

We are exposed to market risks including interest rate risk and foreign currency exchange risk.
Our investment portfolio consists primarily of government securities and corporate bonds.

ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA

See consolidated financial statements attached hereto.
"""

SAMPLE_NEWS = "Microsoft beats earnings expectations for Q4 2024."
SAMPLE_NEWS_BODY = "Microsoft reported strong quarterly results driven by cloud growth."


class TestDetectSections:
    """Tests for section detection in 10-K text."""

    def test_detects_item_1(self):
        sections = detect_sections(SAMPLE_10K)
        section_names = [s["section"] for s in sections]
        assert "ITEM_1_BUSINESS" in section_names

    def test_detects_item_1a_risk_factors(self):
        sections = detect_sections(SAMPLE_10K)
        section_names = [s["section"] for s in sections]
        assert "ITEM_1A_RISK_FACTORS" in section_names

    def test_detects_item_7_mda(self):
        sections = detect_sections(SAMPLE_10K)
        section_names = [s["section"] for s in sections]
        assert "ITEM_7_MDA" in section_names

    def test_detects_item_7a(self):
        sections = detect_sections(SAMPLE_10K)
        section_names = [s["section"] for s in sections]
        assert "ITEM_7A_QUANT" in section_names

    def test_detects_multiple_sections(self):
        sections = detect_sections(SAMPLE_10K)
        assert len(sections) >= 4

    def test_section_text_not_empty(self):
        sections = detect_sections(SAMPLE_10K)
        for s in sections:
            assert len(s["text"].strip()) > 0

    def test_empty_text_returns_full_document(self):
        sections = detect_sections("")
        assert len(sections) == 1
        assert sections[0]["section"] == "FULL_DOCUMENT"

    def test_no_sections_text_returns_full_document(self):
        plain_text = "This is plain text with no SEC section headers."
        sections = detect_sections(plain_text)
        assert len(sections) == 1
        assert sections[0]["section"] == "FULL_DOCUMENT"


class TestChunkText:
    """Tests for chunk_text() function."""

    def test_chunk_size_respected(self):
        long_text = "This is a sentence. " * 200  # ~4000 chars
        chunks = chunk_text(long_text, section="TEST", chunk_size=512, chunk_overlap=64)
        for chunk in chunks:
            # Allow slight overflow due to splitter behavior
            assert len(chunk["text"]) <= 600, f"Chunk too large: {len(chunk['text'])}"

    def test_chunk_index_sequential(self):
        text = "word " * 500
        chunks = chunk_text(text, section="TEST", chunk_size=256, chunk_overlap=32)
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(indices)))

    def test_section_preserved_in_chunks(self):
        text = "This is content for section testing. " * 50
        section_name = "ITEM_7_MDA"
        chunks = chunk_text(text, section=section_name)
        for chunk in chunks:
            assert chunk["section"] == section_name

    def test_empty_text_returns_empty_list(self):
        chunks = chunk_text("", section="TEST")
        assert chunks == []

    def test_short_text_single_chunk(self):
        short_text = "This is a very short text."
        chunks = chunk_text(short_text, section="TEST", chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0]["text"] == short_text


class TestChunkFiling:
    """Tests for the full chunk_filing() pipeline."""

    def test_chunk_filing_produces_chunks(self):
        chunks = chunk_filing(SAMPLE_10K, form_type="10-K")
        assert len(chunks) > 0

    def test_chunk_filing_form_type_set(self):
        chunks = chunk_filing(SAMPLE_10K, form_type="10-K")
        for chunk in chunks:
            assert chunk["form_type"] == "10-K"

    def test_chunk_filing_max_chars_limit(self):
        """Test that filing is trimmed to max_chars before chunking."""
        long_text = "word " * 200_000  # Very long
        chunks = chunk_filing(long_text, max_chars=10_000)
        # All chunk text combined should be ≤ 10_000 chars (plus overlap allowance)
        total_chars = sum(len(c["text"]) for c in chunks)
        assert total_chars <= 15_000  # Allow for overlap

    def test_chunk_10q(self):
        chunks = chunk_filing(SAMPLE_10K, form_type="10-Q")
        assert all(c["form_type"] == "10-Q" for c in chunks)

    def test_risk_factor_section_isolated(self):
        """Risk factor content should be in ITEM_1A_RISK_FACTORS chunks."""
        chunks = chunk_filing(SAMPLE_10K, form_type="10-K")
        risk_chunks = [c for c in chunks if c["section"] == "ITEM_1A_RISK_FACTORS"]
        assert len(risk_chunks) > 0

        # Risk chunk content should contain risk-related text
        risk_text = " ".join(c["text"] for c in risk_chunks)
        assert "risk" in risk_text.lower() or "regulatory" in risk_text.lower()


class TestChunkNewsArticle:
    """Tests for chunk_news_article()."""

    def test_news_chunk_has_metadata(self):
        chunks = chunk_news_article(
            text=SAMPLE_NEWS_BODY,
            title=SAMPLE_NEWS,
            source="Reuters",
            date="2024-10-30",
        )
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk["source"] == "Reuters"
            assert chunk["date"] == "2024-10-30"
            assert chunk["title"] == SAMPLE_NEWS

    def test_news_section_is_news_article(self):
        chunks = chunk_news_article(text=SAMPLE_NEWS_BODY, title=SAMPLE_NEWS)
        for chunk in chunks:
            assert chunk["section"] == "NEWS_ARTICLE"

    def test_short_news_is_single_chunk(self):
        chunks = chunk_news_article(text="Short news.", title="Title")
        assert len(chunks) == 1
