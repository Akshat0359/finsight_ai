"""
FinSight AI — Full FinSightReport Pydantic Schema
All models use Pydantic v2.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class OverallSignal(BaseModel):
    signal: Literal["BULLISH", "NEUTRAL", "BEARISH"]
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    rationale: str


class FinancialRatios(BaseModel):
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    ev_ebitda: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    roe: float | None = None
    roa: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    free_cash_flow_ttm: float | None = None
    revenue_cagr_3y: float | None = None


class PricePerformance(BaseModel):
    return_1m: float | None = None
    return_3m: float | None = None
    return_6m: float | None = None
    return_1y: float | None = None
    return_ytd: float | None = None
    volatility_30d: float | None = None


class FinancialAnalysis(BaseModel):
    health_score: float = Field(ge=1, le=10, description="Financial health score 1-10")
    health_rationale: str
    ratios: FinancialRatios = Field(default_factory=FinancialRatios)
    price_performance: PricePerformance = Field(default_factory=PricePerformance)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


class RiskFactor(BaseModel):
    category: str
    description: str
    severity: int = Field(ge=1, le=5, description="Risk severity 1-5")
    likelihood: int = Field(ge=1, le=5, description="Risk likelihood 1-5")
    is_new: bool = False
    source_filing: str = ""


class RiskAnalysis(BaseModel):
    overall_risk_score: float = Field(ge=1, le=10, description="Overall risk score 1-10")
    risk_factors: list[RiskFactor] = Field(default_factory=list)


class NewsEvent(BaseModel):
    headline: str
    date: str
    sentiment: str
    significance: str


class SentimentAnalysis(BaseModel):
    aggregate_score: float = Field(ge=-1.0, le=1.0, description="Sentiment score -1 to 1")
    trend: Literal["IMPROVING", "DECLINING", "STABLE"]
    article_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    top_events: list[NewsEvent] = Field(default_factory=list)
    earnings_tone: Literal["POSITIVE", "NEUTRAL", "CAUTIOUS", "N/A"] = "N/A"


class Source(BaseModel):
    type: str
    title: str
    date: str
    url: str = ""


class ReportMetadata(BaseModel):
    ticker: str
    company_name: str
    run_id: str
    generated_at: str
    data_sources: list[str] = Field(default_factory=list)


class FinSightReport(BaseModel):
    metadata: ReportMetadata
    overall_signal: OverallSignal
    executive_summary: str
    investment_thesis: list[str] = Field(max_length=3)
    key_risks: list[str] = Field(max_length=3)
    financial_analysis: FinancialAnalysis
    risk_analysis: RiskAnalysis
    sentiment_analysis: SentimentAnalysis
    sources: list[Source] = Field(default_factory=list)
    conclusion: str
    pdf_path: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "FinSightReport":
        """Create from raw dict, handling nested conversions."""
        # Convert metadata dict
        meta = data.get("metadata", {})
        if not isinstance(meta, ReportMetadata):
            meta = ReportMetadata(**meta) if meta else ReportMetadata(
                ticker=data.get("ticker", ""),
                company_name=data.get("company_name", ""),
                run_id=data.get("run_id", ""),
                generated_at="",
            )

        return cls(**{**data, "metadata": meta})
