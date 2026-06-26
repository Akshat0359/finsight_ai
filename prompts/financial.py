"""
FinSight AI — Financial Analysis Prompts
"""
from __future__ import annotations

SYSTEM_PROMPT = """You are a senior financial analyst. Analyze the provided financial ratios and company filings context to assess financial health. Be precise, data-driven, and cite specific metrics. Output valid JSON matching the required schema exactly."""


def build_user_prompt(
    ticker: str,
    company_name: str,
    ratios: dict,
    context_chunks: list[dict],
) -> str:
    context_text = "\n\n".join(
        f"[{c.get('metadata', {}).get('section', 'SEC')}]: {c.get('text', '')[:400]}"
        for c in context_chunks[:5]
    )

    ratios_text = "\n".join(
        f"  {k}: {v}" for k, v in ratios.items() if v is not None
    )

    return f"""Analyze {company_name} ({ticker}) financial health.

FINANCIAL RATIOS:
{ratios_text}

RELEVANT SEC FILING CONTEXT:
{context_text}

Assess the company's financial health on a scale of 1-10 where:
- 8-10: Excellent — strong balance sheet, growing revenues, high margins
- 5-7: Moderate — stable but with some concerns
- 1-4: Poor — significant financial stress

Provide health_score (float 1-10), health_rationale (str), 
strengths (list of 3-5 specific financial strengths), 
weaknesses (list of 3-5 specific concerns).
Output JSON."""
