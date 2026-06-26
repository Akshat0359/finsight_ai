"""
FinSight AI — SQLAlchemy ORM Models
All database tables are defined here.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.database import Base


def _now() -> datetime:
    return datetime.utcnow()


def _uuid() -> str:
    return str(uuid.uuid4())


# ------------------------------------------------------------------ #
# Company
# ------------------------------------------------------------------ #
class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    cik: Mapped[str] = mapped_column(String(20), nullable=True, index=True)
    sector: Mapped[str] = mapped_column(String(100), nullable=True)
    industry: Mapped[str] = mapped_column(String(100), nullable=True)
    exchange: Mapped[str] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_now, server_default=func.now()
    )

    runs: Mapped[list["Run"]] = relationship("Run", back_populates="company_ref", foreign_keys="Run.ticker", primaryjoin="Company.ticker == Run.ticker")
    filings: Mapped[list["Filing"]] = relationship("Filing", back_populates="company_ref", foreign_keys="Filing.ticker", primaryjoin="Company.ticker == Filing.ticker")

    def __repr__(self) -> str:
        return f"<Company {self.ticker} ({self.name})>"


# ------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------ #
class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="PENDING"
    )  # PENDING, RUNNING, COMPLETE, FAILED
    triggered_by: Mapped[str] = mapped_column(String(50), nullable=True, default="api")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    completed_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    total_tokens_used: Mapped[int] = mapped_column(Integer, nullable=True, default=0)
    error_message: Mapped[str] = mapped_column(Text, nullable=True)

    company_ref: Mapped["Company"] = relationship(
        "Company",
        back_populates="runs",
        foreign_keys=[ticker],
        primaryjoin="Run.ticker == Company.ticker",
    )
    report: Mapped["Report"] = relationship("Report", back_populates="run", uselist=False)

    def __repr__(self) -> str:
        return f"<Run {self.id} [{self.status}] ticker={self.ticker}>"


# ------------------------------------------------------------------ #
# Report
# ------------------------------------------------------------------ #
class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.id"), nullable=False, unique=True, index=True
    )
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    report_json: Mapped[str] = mapped_column(Text, nullable=False)
    pdf_path: Mapped[str] = mapped_column(String(500), nullable=True)
    overall_signal: Mapped[str] = mapped_column(String(20), nullable=True)
    signal_confidence: Mapped[str] = mapped_column(String(20), nullable=True)
    financial_score: Mapped[float] = mapped_column(Float, nullable=True)
    risk_score: Mapped[float] = mapped_column(Float, nullable=True)
    sentiment_score: Mapped[float] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_now, server_default=func.now()
    )

    run: Mapped["Run"] = relationship("Run", back_populates="report")

    def __repr__(self) -> str:
        return f"<Report run_id={self.run_id} signal={self.overall_signal}>"


# ------------------------------------------------------------------ #
# Filing
# ------------------------------------------------------------------ #
class Filing(Base):
    __tablename__ = "filings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    cik: Mapped[str] = mapped_column(String(20), nullable=False)
    form_type: Mapped[str] = mapped_column(String(20), nullable=False)
    filing_date: Mapped[str] = mapped_column(String(20), nullable=True)
    period_of_report: Mapped[str] = mapped_column(String(20), nullable=True)
    accession_number: Mapped[str] = mapped_column(
        String(50), nullable=False, unique=True, index=True
    )
    document_url: Mapped[str] = mapped_column(String(500), nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=True, default=0)

    company_ref: Mapped["Company"] = relationship(
        "Company",
        back_populates="filings",
        foreign_keys=[ticker],
        primaryjoin="Filing.ticker == Company.ticker",
    )

    __table_args__ = (
        UniqueConstraint("accession_number", name="uq_filing_accession"),
    )

    def __repr__(self) -> str:
        return f"<Filing {self.form_type} {self.filing_date} ticker={self.ticker}>"


# ------------------------------------------------------------------ #
# AlertConfig
# ------------------------------------------------------------------ #
class AlertConfig(Base):
    __tablename__ = "alert_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    alert_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # NEW_FILING, SENTIMENT_SPIKE, SIGNAL_CHANGE
    threshold: Mapped[float] = mapped_column(Float, nullable=True)
    delivery_method: Mapped[str] = mapped_column(
        String(20), nullable=False, default="dashboard"
    )  # dashboard, log
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_now, server_default=func.now()
    )

    events: Mapped[list["AlertEvent"]] = relationship("AlertEvent", back_populates="config")

    def __repr__(self) -> str:
        return f"<AlertConfig {self.alert_type} ticker={self.ticker}>"


# ------------------------------------------------------------------ #
# AlertEvent
# ------------------------------------------------------------------ #
class AlertEvent(Base):
    __tablename__ = "alert_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    config_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("alert_configs.id"), nullable=False
    )
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    triggered_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    data_json: Mapped[str] = mapped_column(Text, nullable=True)
    acknowledged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    config: Mapped["AlertConfig"] = relationship("AlertConfig", back_populates="events")

    def __repr__(self) -> str:
        return f"<AlertEvent {self.ticker} at {self.triggered_at}>"


# ------------------------------------------------------------------ #
# LLMCache
# ------------------------------------------------------------------ #
class LLMCache(Base):
    __tablename__ = "llm_cache"

    prompt_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    response_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_now, server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<LLMCache {self.prompt_hash[:8]}...>"
