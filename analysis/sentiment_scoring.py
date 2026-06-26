"""
FinSight AI — Sentiment Scoring Aggregation
Pure math functions to aggregate article-level sentiment into composite scores.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Sentiment label → numeric score mapping
SENTIMENT_MAP: dict[str, float] = {
    "POSITIVE": 1.0,
    "positive": 1.0,
    "BULLISH": 1.0,
    "bullish": 1.0,
    "NEUTRAL": 0.0,
    "neutral": 0.0,
    "NEGATIVE": -1.0,
    "negative": -1.0,
    "BEARISH": -1.0,
    "bearish": -1.0,
}


def label_to_score(label: str) -> float:
    """Convert a string sentiment label to a numeric score [-1, 1]."""
    return SENTIMENT_MAP.get(label.strip().upper(), 0.0)


def score_to_label(score: float) -> str:
    """Convert numeric score to a sentiment label."""
    if score >= 0.25:
        return "POSITIVE"
    if score <= -0.25:
        return "NEGATIVE"
    return "NEUTRAL"


def aggregate_sentiment(
    articles: list[dict[str, Any]],
    recency_weight: bool = True,
) -> dict[str, Any]:
    """
    Aggregate a list of article sentiment dicts into composite metrics.

    Each article dict should have:
      - sentiment: str (POSITIVE | NEUTRAL | NEGATIVE)
      - significance: str (HIGH | MEDIUM | LOW) — optional
      - date: str — optional, used for recency weighting

    Returns:
      aggregate_score: float in [-1.0, 1.0]
      positive_count: int
      negative_count: int
      neutral_count: int
      article_count: int
      trend: str (IMPROVING | DECLINING | STABLE)
    """
    if not articles:
        return {
            "aggregate_score": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "article_count": 0,
            "trend": "STABLE",
        }

    scores: list[float] = []
    weights: list[float] = []
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    significance_weights = {"HIGH": 2.0, "MEDIUM": 1.0, "LOW": 0.5}

    for i, article in enumerate(articles):
        raw_label = article.get("sentiment", "NEUTRAL")
        score = label_to_score(raw_label)

        # Significance weighting
        sig = str(article.get("significance", "MEDIUM")).upper()
        sig_weight = significance_weights.get(sig, 1.0)

        # Recency weighting — more recent articles get higher weight
        if recency_weight:
            # Assume articles are ordered newest first
            recency = max(0.1, 1.0 - (i / max(len(articles), 1)) * 0.5)
        else:
            recency = 1.0

        w = sig_weight * recency
        scores.append(score)
        weights.append(w)

        if score > 0:
            positive_count += 1
        elif score < 0:
            negative_count += 1
        else:
            neutral_count += 1

    # Weighted average
    scores_arr = np.array(scores)
    weights_arr = np.array(weights)
    aggregate_score = float(np.average(scores_arr, weights=weights_arr))
    aggregate_score = round(max(-1.0, min(1.0, aggregate_score)), 4)

    # Trend: compare first half vs second half of articles
    trend = "STABLE"
    if len(scores) >= 4:
        mid = len(scores) // 2
        first_half = float(np.mean(scores_arr[:mid]))
        second_half = float(np.mean(scores_arr[mid:]))
        delta = second_half - first_half
        if delta > 0.15:
            trend = "IMPROVING"
        elif delta < -0.15:
            trend = "DECLINING"

    return {
        "aggregate_score": aggregate_score,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "article_count": len(articles),
        "trend": trend,
    }


def detect_sentiment_spike(
    current_score: float,
    baseline_score: float,
    threshold: float = 0.3,
) -> tuple[bool, str]:
    """
    Detect if current sentiment has spiked vs baseline.
    Returns (is_spike, direction) where direction is POSITIVE or NEGATIVE.
    """
    delta = current_score - baseline_score
    if abs(delta) >= threshold:
        direction = "POSITIVE" if delta > 0 else "NEGATIVE"
        return True, direction
    return False, "NONE"


def earnings_tone_from_text(text: str) -> str:
    """
    Simple rule-based earnings tone detection from transcript/8-K text.
    Returns: POSITIVE | NEUTRAL | CAUTIOUS | N/A
    """
    if not text:
        return "N/A"

    text_lower = text.lower()

    positive_signals = [
        "record revenue", "strong growth", "exceeded expectations",
        "beat estimates", "raised guidance", "record earnings",
        "strong demand", "robust", "outperformed",
    ]
    cautious_signals = [
        "headwinds", "uncertain", "challenging environment",
        "softening", "below expectations", "missed estimates",
        "lowered guidance", "macro pressures", "risk factors",
        "decline", "weakness",
    ]

    pos_hits = sum(1 for s in positive_signals if s in text_lower)
    cau_hits = sum(1 for s in cautious_signals if s in text_lower)

    if pos_hits > cau_hits + 1:
        return "POSITIVE"
    if cau_hits > pos_hits + 1:
        return "CAUTIOUS"
    if pos_hits == 0 and cau_hits == 0:
        return "N/A"
    return "NEUTRAL"
