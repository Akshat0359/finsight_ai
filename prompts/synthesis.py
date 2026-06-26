"""
FinSight AI — Synthesis Prompts
"""
from __future__ import annotations

SYSTEM_PROMPT = """You are FinSight AI's chief investment analyst. Synthesize financial, risk, and sentiment analysis into a comprehensive, actionable investment report. Be concise but thorough. Support every claim with data. Output valid JSON matching the required schema exactly."""


def build_user_prompt(
    ticker: str,
    company_name: str,
    financial_analysis: dict,
    risk_analysis: dict,
    sentiment_analysis: dict,
    context_chunks: list[dict],
) -> str:
    context_text = "\n".join(
        f"- {c.get('text', '')[:200]}" for c in context_chunks[:5]
    )

    fin_score = financial_analysis.get("health_score", "N/A")
    fin_rationale = financial_analysis.get("health_rationale", "")
    risk_score = risk_analysis.get("overall_risk_score", "N/A")
    sentiment_score = sentiment_analysis.get("aggregate_score", 0)
    sentiment_trend = sentiment_analysis.get("trend", "STABLE")

    return f"""Synthesize a complete investment analysis report for {company_name} ({ticker}).

FINANCIAL ANALYSIS:
- Health Score: {fin_score}/10
- Rationale: {fin_rationale}
- Strengths: {", ".join(financial_analysis.get("strengths", [])[:3])}
- Weaknesses: {", ".join(financial_analysis.get("weaknesses", [])[:3])}

RISK ANALYSIS:
- Overall Risk Score: {risk_score}/10
- Top Risks: {"; ".join([f"{r.get('category','')}: {r.get('description','')[:80]}" for r in risk_analysis.get("risk_factors", [])[:3]])}

SENTIMENT ANALYSIS:
- Aggregate Score: {sentiment_score:.2f} (-1=bearish, +1=bullish)
- Trend: {sentiment_trend}
- Article Count: {sentiment_analysis.get("article_count", 0)}

ADDITIONAL CONTEXT FROM FILINGS:
{context_text}

Generate a comprehensive investment report with:
1. overall_signal: BULLISH, NEUTRAL, or BEARISH
2. signal confidence: HIGH, MEDIUM, or LOW
3. signal rationale: 2-3 sentence explanation
4. executive_summary: 3-4 sentence overview
5. investment_thesis: exactly 3 bullet points (reasons to invest or avoid)
6. key_risks: exactly 3 bullet points (top risk items)
7. conclusion: 2-3 sentence closing assessment

The signal should reflect the weighted combination:
- Financial health (40% weight)
- Risk profile (30% weight, inverted — higher risk = more bearish)
- Sentiment (30% weight)

Output JSON."""
