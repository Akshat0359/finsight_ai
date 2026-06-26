"""
FinSight AI — Risk Analysis Prompts
"""
from __future__ import annotations

SYSTEM_PROMPT = """You are a risk analyst specializing in SEC filing risk factor analysis. Extract, categorize, and score risk factors from the provided text. Be specific and cite the exact risk language. Output valid JSON matching the required schema exactly."""


def build_user_prompt(
    ticker: str,
    company_name: str,
    risk_chunks: list[dict],
) -> str:
    risk_text = "\n\n".join(
        f"[RISK FACTOR {i+1}]: {c.get('text', '')[:600]}"
        for i, c in enumerate(risk_chunks[:8])
    )

    return f"""Extract and score risk factors for {company_name} ({ticker}) from their SEC filings.

RISK FACTOR EXCERPTS FROM SEC FILINGS:
{risk_text}

For each distinct risk factor identified, provide:
- category: one of (REGULATORY, COMPETITIVE, FINANCIAL, OPERATIONAL, MACRO, LEGAL, TECHNOLOGY, ESG)  
- description: concise 1-2 sentence description
- severity: 1-5 (5=catastrophic, 1=minor)
- likelihood: 1-5 (5=very likely, 1=very unlikely)
- is_new: false (set true if language suggests this is a newly disclosed risk)
- source_filing: form type (10-K, 10-Q, or 8-K)

Also provide overall_risk_score (1-10, where 10=extremely high risk).
Identify at least 3 and up to 8 risks. Output JSON."""
