"""
FinSight AI — Sentiment Analysis Agent Node
Batch news sentiment classification + earnings tone analysis via Gemini.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import google.generativeai as genai

from agents._llm import get_cached_or_call
from agents.state import FinSightState
from analysis.sentiment_scoring import aggregate_sentiment, earnings_tone_from_text
from mcp_server.client import get_mcp_client
from prompts.sentiment import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

# Gemini schema for per-batch sentiment
_ARTICLE_SENTIMENT_SCHEMA = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "sentiment": genai.protos.Schema(type=genai.protos.Type.STRING),
        "significance": genai.protos.Schema(type=genai.protos.Type.STRING),
        "key_topic": genai.protos.Schema(type=genai.protos.Type.STRING),
    },
)

_NEWS_EVENT_SCHEMA = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "headline": genai.protos.Schema(type=genai.protos.Type.STRING),
        "date": genai.protos.Schema(type=genai.protos.Type.STRING),
        "sentiment": genai.protos.Schema(type=genai.protos.Type.STRING),
        "significance": genai.protos.Schema(type=genai.protos.Type.STRING),
    },
)

_BATCH_SCHEMA = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "article_sentiments": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=_ARTICLE_SENTIMENT_SCHEMA,
        ),
        "top_events": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=_NEWS_EVENT_SCHEMA,
        ),
    },
    required=["article_sentiments", "top_events"],
)


def _classify_batch(
    ticker: str,
    company_name: str,
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Classify sentiment for a batch of articles."""
    user_prompt = build_user_prompt(
        ticker=ticker,
        company_name=company_name,
        articles_batch=batch,
    )
    return get_cached_or_call(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=_BATCH_SCHEMA,
    )


def sentiment_node(state: FinSightState) -> FinSightState:
    """
    Sentiment analysis node:
    1. Use news articles from state (already fetched by ingestor)
    2. Batch classify with Gemini (10 articles per batch)
    3. Aggregate scores using sentiment_scoring.py
    4. Detect earnings tone from 8-K/earnings call text
    """
    ticker = state.get("ticker", "")
    company_name = state.get("company_name", ticker)
    news_articles = state.get("news_articles", [])
    errors: list[str] = []

    logger.info(
        "[sentiment] Analyzing %d articles for %s", len(news_articles), ticker
    )

    classified_articles: list[dict[str, Any]] = []
    all_top_events: list[dict[str, Any]] = []
    batch_size = 10

    try:
        if news_articles:
            # Process in batches of 10
            for batch_start in range(0, min(len(news_articles), 100), batch_size):
                batch = news_articles[batch_start: batch_start + batch_size]
                try:
                    batch_result = _classify_batch(ticker, company_name, batch)
                    sentiments = batch_result.get("article_sentiments", [])
                    top_events = batch_result.get("top_events", [])

                    # Merge sentiment back into articles
                    for i, article in enumerate(batch):
                        merged = dict(article)
                        if i < len(sentiments):
                            merged["sentiment"] = sentiments[i].get("sentiment", "NEUTRAL")
                            merged["significance"] = sentiments[i].get("significance", "MEDIUM")
                            merged["key_topic"] = sentiments[i].get("key_topic", "")
                        else:
                            merged["sentiment"] = "NEUTRAL"
                            merged["significance"] = "LOW"
                        classified_articles.append(merged)

                    all_top_events.extend(top_events)

                except Exception as exc:
                    msg = f"Sentiment batch error (offset {batch_start}): {exc}"
                    logger.warning(msg)
                    errors.append(msg)
                    # Add articles with NEUTRAL default
                    for article in batch:
                        classified_articles.append({**article, "sentiment": "NEUTRAL", "significance": "LOW"})

    except Exception as exc:
        msg = f"Sentiment agent error: {exc}"
        logger.error(msg)
        errors.append(msg)

    # Aggregate scores
    agg = aggregate_sentiment(classified_articles)

    # Deduplicate top events (take top 5 by significance)
    seen = set()
    deduped_events: list[dict[str, Any]] = []
    for event in all_top_events:
        key = event.get("headline", "")[:50]
        if key not in seen:
            seen.add(key)
            deduped_events.append(event)
    top_events_final = deduped_events[:5]

    # Try to get earnings tone from 8-K text in ChromaDB
    earnings_tone = "N/A"
    try:
        client = get_mcp_client()
        earnings_chunks = client.semantic_search(
            query="earnings results guidance revenue beat miss",
            collection="sec_filings",
            ticker=ticker,
            top_k=3,
            where={"form_type": {"$eq": "8-K"}},
        )
        if earnings_chunks:
            combined_text = " ".join(c.get("text", "") for c in earnings_chunks)
            earnings_tone = earnings_tone_from_text(combined_text)
    except Exception as exc:
        logger.debug("Could not determine earnings tone: %s", exc)

    sentiment_result: dict[str, Any] = {
        "aggregate_score": agg["aggregate_score"],
        "trend": agg["trend"],
        "article_count": agg["article_count"],
        "positive_count": agg["positive_count"],
        "negative_count": agg["negative_count"],
        "neutral_count": agg["neutral_count"],
        "top_events": top_events_final,
        "earnings_tone": earnings_tone,
    }

    logger.info(
        "[sentiment] Complete: score=%.2f trend=%s for %s",
        agg["aggregate_score"], agg["trend"], ticker,
    )

    return {**state, "sentiment_result": sentiment_result, "errors": errors}
