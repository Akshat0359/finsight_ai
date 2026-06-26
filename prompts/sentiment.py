"""
FinSight AI — Sentiment Analysis Prompts
"""
from __future__ import annotations

SYSTEM_PROMPT = """You are a financial sentiment analyst. Classify news article sentiment and extract key events relevant to stock performance. Be objective and data-driven. Output valid JSON matching the required schema exactly."""


def build_user_prompt(ticker: str, company_name: str, articles_batch: list[dict]) -> str:
    articles_text = "\n\n".join(
        f"[{i+1}] {a.get('date', '')} | {a.get('source', '')} | {a.get('title', '')}\n{a.get('summary', '')[:300]}"
        for i, a in enumerate(articles_batch)
    )

    return f"""Analyze the sentiment of these news articles about {company_name} ({ticker}).

NEWS ARTICLES:
{articles_text}

For each article, classify:
- sentiment: POSITIVE, NEUTRAL, or NEGATIVE (impact on stock/company)
- significance: HIGH, MEDIUM, or LOW (importance to investors)
- key_topic: 1-3 words describing the main topic

Also identify the top 3 most significant events from all articles with:
- headline: the article title
- date: publication date
- sentiment: POSITIVE/NEUTRAL/NEGATIVE
- significance: why this matters to investors (1 sentence)

Output JSON with: article_sentiments (list), top_events (list of 3)."""


def build_earnings_tone_prompt(ticker: str, company_name: str, earnings_text: str) -> str:
    return f"""Analyze the tone of this earnings-related text for {company_name} ({ticker}):

{earnings_text[:1500]}

Classify the overall earnings tone as one of:
- POSITIVE: beats estimates, raised guidance, strong growth language
- NEUTRAL: met estimates, maintained guidance, balanced language  
- CAUTIOUS: missed estimates, lowered guidance, risk-heavy language
- N/A: no earnings-related content found

Provide: earnings_tone (str), key_phrases (list of 3 supporting phrases). Output JSON."""
