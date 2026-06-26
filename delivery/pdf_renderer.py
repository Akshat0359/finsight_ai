"""
FinSight AI — WeasyPrint PDF Renderer
Generates professional PDF reports from FinSightReport data.
"""
from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Template directory
_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _ratio_row(name: str, value: Any, note: str = "", low_is_good: bool = False) -> dict:
    """Build a ratio table row with color-coding."""
    if value is None:
        return {"name": name, "value": "N/A", "css_class": "ratio-neutral", "note": note}

    try:
        fval = float(value)
        if low_is_good:
            css = "ratio-positive" if fval < 1.5 else ("ratio-negative" if fval > 3 else "ratio-neutral")
        else:
            css = "ratio-positive" if fval > 0 else "ratio-negative"
        formatted = f"{fval:.2f}"
    except (TypeError, ValueError):
        formatted = str(value)
        css = "ratio-neutral"

    return {"name": name, "value": formatted, "css_class": css, "note": note}


def _generate_returns_chart(price_performance: dict[str, Any]) -> str:
    """Generate a bar chart for price returns and return as base64 PNG."""
    try:
        import plotly.graph_objects as go

        labels = ["1M", "3M", "6M", "1Y", "YTD"]
        keys = ["return_1m", "return_3m", "return_6m", "return_1y", "return_ytd"]
        values = [price_performance.get(k) for k in keys]

        # Filter out None values
        valid_labels = [l for l, v in zip(labels, values) if v is not None]
        valid_values = [v for v in values if v is not None]

        if not valid_values:
            return ""

        colors = ["#00b894" if v >= 0 else "#d63031" for v in valid_values]

        fig = go.Figure(
            go.Bar(
                x=valid_labels,
                y=valid_values,
                marker_color=colors,
                text=[f"{v:.1f}%" for v in valid_values],
                textposition="outside",
            )
        )
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=5, r=5, t=5, b=5),
            height=200,
            width=500,
            font=dict(size=10),
            yaxis_title="Return (%)",
            showlegend=False,
        )

        img_bytes = fig.to_image(format="png", scale=1.5)
        return base64.b64encode(img_bytes).decode("utf-8")

    except Exception as exc:
        logger.warning("Returns chart generation failed: %s", exc)
        return ""


def _generate_sentiment_chart(sentiment: dict[str, Any]) -> str:
    """Generate a donut chart for sentiment distribution."""
    try:
        import plotly.graph_objects as go

        pos = sentiment.get("positive_count", 0)
        neg = sentiment.get("negative_count", 0)
        neu = sentiment.get("neutral_count", 0)

        if pos + neg + neu == 0:
            return ""

        fig = go.Figure(
            go.Pie(
                labels=["Positive", "Neutral", "Negative"],
                values=[pos, neu, neg],
                hole=0.55,
                marker_colors=["#00b894", "#636e72", "#d63031"],
                textfont_size=10,
            )
        )
        fig.update_layout(
            paper_bgcolor="white",
            margin=dict(l=5, r=5, t=5, b=5),
            height=180,
            width=300,
            showlegend=True,
            legend=dict(font=dict(size=9)),
        )

        img_bytes = fig.to_image(format="png", scale=1.5)
        return base64.b64encode(img_bytes).decode("utf-8")

    except Exception as exc:
        logger.warning("Sentiment chart generation failed: %s", exc)
        return ""


def render_pdf(
    report_data: dict[str, Any],
    run_id: str,
    ticker: str,
) -> str:
    """
    Render a FinSightReport to PDF using WeasyPrint + Jinja2.
    Returns the absolute path to the generated PDF file.
    """
    from weasyprint import HTML

    # Ensure output directory exists
    output_dir = Path(settings.PDF_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = str(output_dir / f"{run_id}.pdf")

    # Load Jinja2 template
    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)))
    template = env.get_template("report.html")

    # Extract data
    overall = report_data.get("overall_signal", {})
    financial = report_data.get("financial_analysis", {})
    risk = report_data.get("risk_analysis", {})
    sentiment = report_data.get("sentiment_analysis", {})
    metadata = report_data.get("metadata", {})

    # Build ratio rows
    ratios = financial.get("ratios", {})
    price_perf = financial.get("price_performance", {})

    ratio_rows = [
        _ratio_row("P/E Ratio", ratios.get("pe_ratio"), "Price / Earnings"),
        _ratio_row("P/B Ratio", ratios.get("pb_ratio"), "Price / Book Value", low_is_good=True),
        _ratio_row("EV/EBITDA", ratios.get("ev_ebitda"), "Enterprise Value / EBITDA", low_is_good=True),
        _ratio_row("Debt/Equity", ratios.get("debt_to_equity"), "Total Debt / Equity", low_is_good=True),
        _ratio_row("Current Ratio", ratios.get("current_ratio"), ">1 = liquid"),
        _ratio_row("ROE (%)", ratios.get("roe"), "Return on Equity"),
        _ratio_row("ROA (%)", ratios.get("roa"), "Return on Assets"),
        _ratio_row("Gross Margin (%)", ratios.get("gross_margin"), "Revenue − COGS / Revenue"),
        _ratio_row("Operating Margin (%)", ratios.get("operating_margin"), "Operating Income / Revenue"),
        _ratio_row("FCF TTM ($B)", ratios.get("free_cash_flow_ttm") and ratios.get("free_cash_flow_ttm", 0) / 1e9, "Free Cash Flow TTM"),
        _ratio_row("Revenue CAGR 3Y (%)", ratios.get("revenue_cagr_3y"), "3-Year Revenue Growth"),
    ]
    ratio_rows = [r for r in ratio_rows if r["value"] != "N/A"]

    # Generate charts
    returns_chart = _generate_returns_chart(price_perf)
    sentiment_chart = _generate_sentiment_chart(sentiment)

    # Generated at date
    generated_at = metadata.get("generated_at", "")[:10]

    # Render template
    html_content = template.render(
        ticker=ticker,
        company_name=metadata.get("company_name", ticker),
        signal=overall.get("signal", "NEUTRAL"),
        confidence=overall.get("confidence", "LOW"),
        signal_rationale=overall.get("rationale", ""),
        generated_at=generated_at,
        executive_summary=report_data.get("executive_summary", ""),
        investment_thesis=report_data.get("investment_thesis", []),
        key_risks=report_data.get("key_risks", []),
        conclusion=report_data.get("conclusion", ""),
        # Scores
        financial_score=financial.get("health_score", 5.0),
        risk_score=risk.get("overall_risk_score", 5.0),
        sentiment_score=sentiment.get("aggregate_score", 0.0),
        # Financial
        ratio_rows=ratio_rows,
        returns_chart=returns_chart,
        strengths=financial.get("strengths", []),
        weaknesses=financial.get("weaknesses", []),
        # Risk
        risk_factors=risk.get("risk_factors", []),
        # Sentiment
        sentiment_trend=sentiment.get("trend", "STABLE"),
        article_count=sentiment.get("article_count", 0),
        positive_count=sentiment.get("positive_count", 0),
        negative_count=sentiment.get("negative_count", 0),
        neutral_count=sentiment.get("neutral_count", 0),
        top_events=sentiment.get("top_events", []),
        earnings_tone=sentiment.get("earnings_tone", "N/A"),
        sentiment_chart=sentiment_chart,
        # Sources
        sources=report_data.get("sources", []),
    )

    # Generate PDF
    HTML(string=html_content, base_url=str(_TEMPLATE_DIR)).write_pdf(pdf_path)
    logger.info("PDF generated: %s (%d bytes)", pdf_path, os.path.getsize(pdf_path))
    return pdf_path
