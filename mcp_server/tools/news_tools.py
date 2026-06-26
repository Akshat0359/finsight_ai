"""
FinSight AI — News RSS MCP Tools
Fetches news from Google News RSS and Yahoo Finance RSS (no API key needed).
Optionally uses NewsAPI if key is configured.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus

import feedparser
import httpx

from app.config import get_settings
from cache.disk_cache import TTL_MEDIUM, build_cache_key, cache_get, cache_set

logger = logging.getLogger(__name__)
settings = get_settings()


def _parse_date(date_str: str) -> str:
    """Parse various date formats to YYYY-MM-DD string."""
    if not date_str:
        return datetime.utcnow().strftime("%Y-%m-%d")
    try:
        dt = parsedate_to_datetime(date_str)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    try:
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(date_str[:19], fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    except Exception:
        pass
    return datetime.utcnow().strftime("%Y-%m-%d")


def _is_within_days(date_str: str, days: int) -> bool:
    """Check if a date string is within the last N days."""
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return date >= cutoff
    except Exception:
        return True  # Include by default if we can't parse


def get_google_news_rss(query: str, days: int = 30) -> list[dict[str, Any]]:
    """
    Fetch news from Google News RSS feed.
    Returns list of {title, summary, date, source, url}.
    """
    encoded_query = quote_plus(f"{query} stock")
    url = (
        f"https://news.google.com/rss/search?"
        f"q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    )

    cache_key = build_cache_key("gnews_rss", query, str(days))
    cached = cache_get(cache_key)
    if cached:
        return cached

    try:
        feed = feedparser.parse(url)
        articles: list[dict[str, Any]] = []

        for entry in feed.entries:
            pub_date = _parse_date(entry.get("published", ""))
            if not _is_within_days(pub_date, days):
                continue

            # Extract source from title (Google News format: "Title - Source")
            title = entry.get("title", "")
            source = ""
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title = parts[0].strip()
                source = parts[1].strip()

            summary = entry.get("summary", "")
            # Clean HTML from summary
            import re
            summary = re.sub(r"<[^>]+>", " ", summary).strip()

            articles.append({
                "title": title,
                "summary": summary[:500],
                "date": pub_date,
                "source": source or "Google News",
                "url": entry.get("link", ""),
            })

        cache_set(cache_key, articles, ttl=TTL_MEDIUM)
        return articles

    except Exception as exc:
        logger.warning("Google News RSS error for '%s': %s", query, exc)
        return []


def get_yahoo_finance_rss(ticker: str, days: int = 30) -> list[dict[str, Any]]:
    """
    Fetch news from Yahoo Finance RSS feed for a ticker.
    Returns list of {title, summary, date, source, url}.
    """
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

    cache_key = build_cache_key("yahoo_rss", ticker, str(days))
    cached = cache_get(cache_key)
    if cached:
        return cached

    try:
        feed = feedparser.parse(url)
        articles: list[dict[str, Any]] = []

        for entry in feed.entries:
            pub_date = _parse_date(entry.get("published", ""))
            if not _is_within_days(pub_date, days):
                continue

            articles.append({
                "title": entry.get("title", ""),
                "summary": entry.get("summary", "")[:500],
                "date": pub_date,
                "source": "Yahoo Finance",
                "url": entry.get("link", ""),
            })

        cache_set(cache_key, articles, ttl=TTL_MEDIUM)
        return articles

    except Exception as exc:
        logger.warning("Yahoo Finance RSS error for '%s': %s", ticker, exc)
        return []


def get_newsapi_articles(
    query: str, ticker: str, days: int = 30
) -> list[dict[str, Any]]:
    """
    Fetch news from NewsAPI (requires NEWSAPI_KEY in .env).
    Falls back to empty list if no key configured.
    """
    if not settings.NEWSAPI_KEY:
        return []

    cache_key = build_cache_key("newsapi", query, ticker, str(days))
    cached = cache_get(cache_key)
    if cached:
        return cached

    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    articles: list[dict[str, Any]] = []

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": f"{query} OR {ticker}",
                    "from": from_date,
                    "sortBy": "publishedAt",
                    "language": "en",
                    "pageSize": 50,
                    "apiKey": settings.NEWSAPI_KEY,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        for item in data.get("articles", []):
            pub = item.get("publishedAt", "")
            articles.append({
                "title": item.get("title", ""),
                "summary": (item.get("description") or "")[:500],
                "date": _parse_date(pub),
                "source": item.get("source", {}).get("name", "NewsAPI"),
                "url": item.get("url", ""),
            })

        cache_set(cache_key, articles, ttl=TTL_MEDIUM)
        return articles

    except Exception as exc:
        logger.warning("NewsAPI error: %s", exc)
        return []


def get_news_articles(
    query: str,
    ticker: str,
    days: int = 30,
) -> list[dict[str, Any]]:
    """
    Aggregate news from all available sources.
    Returns deduplicated list sorted by date (newest first).
    """
    cache_key = build_cache_key("news_all", query, ticker, str(days))
    cached = cache_get(cache_key)
    if cached:
        return cached

    google_articles = get_google_news_rss(query, days)
    yahoo_articles = get_yahoo_finance_rss(ticker, days)
    newsapi_articles = get_newsapi_articles(query, ticker, days)

    all_articles = google_articles + yahoo_articles + newsapi_articles

    # Deduplicate by title similarity
    seen_titles: set[str] = set()
    unique: list[dict[str, Any]] = []
    for article in all_articles:
        title_key = article["title"].lower()[:60]
        if title_key and title_key not in seen_titles:
            seen_titles.add(title_key)
            unique.append(article)

    # Sort newest first
    unique.sort(key=lambda x: x.get("date", ""), reverse=True)

    logger.info(
        "Fetched %d unique news articles for %s (google=%d, yahoo=%d, newsapi=%d)",
        len(unique), ticker, len(google_articles), len(yahoo_articles), len(newsapi_articles),
    )

    cache_set(cache_key, unique, ttl=TTL_MEDIUM)
    return unique
