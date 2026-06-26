"""
FinSight AI — Ingestor Agent Node
Fetches SEC filings, market data, and news → chunks → embeds → stores in ChromaDB.
"""
from __future__ import annotations

import logging
from typing import Any

from agents.state import FinSightState
from mcp_server.client import get_mcp_client
from rag.chunker import chunk_filing, chunk_news_article

logger = logging.getLogger(__name__)

# Collection name constants
COL_FILINGS = "sec_filings"
COL_NEWS = "news"


def ingestor_node(state: FinSightState) -> FinSightState:
    """
    Ingestor node: fetch all data sources and embed into ChromaDB.
    Returns updated state with ingestor_result and news_articles.
    """
    ticker = state.get("ticker", "")
    cik = state.get("cik", "")
    errors: list[str] = []
    client = get_mcp_client()

    logger.info("[ingestor] Starting data ingestion for %s (CIK=%s)", ticker, cik)

    result: dict[str, Any] = {
        "filings_ingested": 0,
        "chunks_stored_filings": 0,
        "chunks_stored_news": 0,
        "news_articles_count": 0,
        "price_data_fetched": False,
        "financials_fetched": False,
        "filing_details": [],
    }

    # ------------------------------------------------------------------ #
    # 1. Fetch & ingest SEC filings
    # ------------------------------------------------------------------ #
    if cik:
        try:
            filings = client.get_recent_filings(
                cik=cik,
                form_types=["10-K", "10-Q", "8-K"],
                limit=11,  # 2x10-K + 4x10-Q + 5x8-K
            )

            form_limits = {"10-K": 2, "10-Q": 4, "8-K": 5}
            form_counts: dict[str, int] = {"10-K": 0, "10-Q": 0, "8-K": 0}

            for filing in filings:
                form_type = filing.get("form_type", "")
                accession = filing.get("accession_number", "")

                if form_limits.get(form_type, 0) <= form_counts.get(form_type, 99):
                    continue

                if not accession:
                    continue

                try:
                    logger.info("[ingestor] Downloading %s %s", form_type, accession)
                    text = client.download_filing_text(cik=cik, accession_number=accession)

                    if not text:
                        continue

                    # Chunk the filing
                    chunks = chunk_filing(text, form_type=form_type)

                    if chunks:
                        texts = [c["text"] for c in chunks]
                        metadatas = [
                            {
                                "section": c["section"],
                                "chunk_index": c["chunk_index"],
                                "form_type": form_type,
                                "filing_date": filing.get("filing_date", ""),
                                "ticker": ticker,
                                "accession_number": accession,
                            }
                            for c in chunks
                        ]

                        stored = client.embed_and_store(
                            texts=texts,
                            metadatas=metadatas,
                            collection=COL_FILINGS,
                            ticker=ticker,
                        )

                        result["chunks_stored_filings"] += stored
                        result["filing_details"].append({
                            "form_type": form_type,
                            "accession": accession,
                            "date": filing.get("filing_date", ""),
                            "chunks": stored,
                        })

                    form_counts[form_type] = form_counts.get(form_type, 0) + 1
                    result["filings_ingested"] += 1

                except Exception as exc:
                    msg = f"Failed to ingest filing {accession}: {exc}"
                    logger.warning(msg)
                    errors.append(msg)

        except Exception as exc:
            msg = f"Failed to fetch filings for CIK {cik}: {exc}"
            logger.error(msg)
            errors.append(msg)

    # ------------------------------------------------------------------ #
    # 2. Fetch market data
    # ------------------------------------------------------------------ #
    try:
        client.get_price_history(ticker, period="1y")
        result["price_data_fetched"] = True
        logger.info("[ingestor] Price history fetched for %s", ticker)
    except Exception as exc:
        msg = f"Failed to fetch price history: {exc}"
        logger.warning(msg)
        errors.append(msg)

    try:
        client.get_financial_statements(ticker)
        result["financials_fetched"] = True
        logger.info("[ingestor] Financial statements fetched for %s", ticker)
    except Exception as exc:
        msg = f"Failed to fetch financial statements: {exc}"
        logger.warning(msg)
        errors.append(msg)

    # ------------------------------------------------------------------ #
    # 3. Fetch & ingest news
    # ------------------------------------------------------------------ #
    company_name = state.get("company_name", ticker)
    news_articles: list[dict[str, Any]] = []
    try:
        news_articles = client.get_news_articles(
            query=company_name, ticker=ticker, days=60
        )
        result["news_articles_count"] = len(news_articles)

        if news_articles:
            # Chunk and embed news articles
            all_news_texts: list[str] = []
            all_news_metas: list[dict[str, Any]] = []

            for article in news_articles[:50]:  # Limit to 50 articles
                chunks = chunk_news_article(
                    text=article.get("summary", ""),
                    title=article.get("title", ""),
                    source=article.get("source", ""),
                    date=article.get("date", ""),
                )
                for chunk in chunks:
                    all_news_texts.append(chunk["text"])
                    all_news_metas.append({
                        "section": "NEWS_ARTICLE",
                        "chunk_index": chunk["chunk_index"],
                        "form_type": "NEWS",
                        "filing_date": article.get("date", ""),
                        "ticker": ticker,
                        "source": article.get("source", ""),
                        "title": article.get("title", ""),
                    })

            if all_news_texts:
                stored = client.embed_and_store(
                    texts=all_news_texts,
                    metadatas=all_news_metas,
                    collection=COL_NEWS,
                    ticker=ticker,
                )
                result["chunks_stored_news"] = stored

        logger.info("[ingestor] Ingested %d news articles for %s", len(news_articles), ticker)

    except Exception as exc:
        msg = f"Failed to fetch/ingest news: {exc}"
        logger.warning(msg)
        errors.append(msg)

    logger.info(
        "[ingestor] Complete: %d filings, %d filing chunks, %d news chunks",
        result["filings_ingested"],
        result["chunks_stored_filings"],
        result["chunks_stored_news"],
    )

    return {
        **state,
        "ingestor_result": result,
        "news_articles": news_articles,
        "errors": errors,
    }
