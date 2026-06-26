"""
FinSight AI — Market Data MCP Tools
Uses yfinance for market data and financial statements.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from analysis.financial_ratios import compute_ratios
from cache.disk_cache import TTL_SHORT, build_cache_key, cache_get, cache_set

logger = logging.getLogger(__name__)


def _df_to_dict(df: pd.DataFrame | None) -> dict[str, Any]:
    """Convert a DataFrame with DatetimeIndex columns to JSON-serializable dict."""
    if df is None or df.empty:
        return {}
    try:
        df_copy = df.copy()
        df_copy.columns = [str(c)[:10] for c in df_copy.columns]
        df_copy = df_copy.where(pd.notnull(df_copy), None)
        return df_copy.to_dict()
    except Exception as exc:
        logger.warning("DataFrame conversion error: %s", exc)
        return {}


def _safe_float(value: Any) -> float | None:
    """Return float or None."""
    try:
        if value is None:
            return None
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return None


def get_price_history(ticker: str, period: str = "1y") -> dict[str, Any]:
    """
    Fetch OHLCV price history for a ticker.
    period options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

    Returns dict with dates, opens, highs, lows, closes, volumes.
    """
    cache_key = build_cache_key("price_history", ticker, period)
    cached = cache_get(cache_key)
    if cached:
        return cached

    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, auto_adjust=True)

        if hist.empty:
            return {"ticker": ticker, "period": period, "data": []}

        records: list[dict[str, Any]] = []
        for date, row in hist.iterrows():
            records.append({
                "date": str(date)[:10],
                "open": _safe_float(row.get("Open")),
                "high": _safe_float(row.get("High")),
                "low": _safe_float(row.get("Low")),
                "close": _safe_float(row.get("Close")),
                "volume": int(row.get("Volume", 0)) if row.get("Volume") else None,
            })

        result = {
            "ticker": ticker,
            "period": period,
            "data": records,
            "latest_close": _safe_float(hist["Close"].iloc[-1]) if not hist.empty else None,
            "data_points": len(records),
        }

        cache_set(cache_key, result, ttl=TTL_SHORT)
        return result

    except Exception as exc:
        logger.error("Error fetching price history for %s: %s", ticker, exc)
        return {"ticker": ticker, "period": period, "data": [], "error": str(exc)}


def get_financial_statements(ticker: str) -> dict[str, Any]:
    """
    Fetch income statement, balance sheet, and cash flow statement.
    Returns nested dict with all three statements.
    """
    cache_key = build_cache_key("financials", ticker)
    cached = cache_get(cache_key)
    if cached:
        return cached

    try:
        tk = yf.Ticker(ticker)
        result: dict[str, Any] = {
            "ticker": ticker,
            "income_statement": _df_to_dict(tk.financials),
            "balance_sheet": _df_to_dict(tk.balance_sheet),
            "cash_flow": _df_to_dict(tk.cashflow),
        }
        cache_set(cache_key, result, ttl=TTL_SHORT)
        return result

    except Exception as exc:
        logger.error("Error fetching financial statements for %s: %s", ticker, exc)
        return {
            "ticker": ticker,
            "income_statement": {},
            "balance_sheet": {},
            "cash_flow": {},
            "error": str(exc),
        }


def get_company_info(ticker: str) -> dict[str, Any]:
    """
    Fetch company profile info: sector, industry, description, market cap, etc.
    """
    cache_key = build_cache_key("company_info", ticker)
    cached = cache_get(cache_key)
    if cached:
        return cached

    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}

        keep_fields = [
            "longName", "shortName", "sector", "industry", "country",
            "exchange", "currency", "marketCap", "enterpriseValue",
            "fullTimeEmployees", "longBusinessSummary", "website",
            "dividendYield", "beta", "52WeekChange", "sharesOutstanding",
        ]

        result: dict[str, Any] = {
            "ticker": ticker,
        }
        for field in keep_fields:
            val = info.get(field)
            if isinstance(val, float):
                result[field] = _safe_float(val)
            else:
                result[field] = val

        cache_set(cache_key, result, ttl=TTL_SHORT)
        return result

    except Exception as exc:
        logger.error("Error fetching company info for %s: %s", ticker, exc)
        return {"ticker": ticker, "error": str(exc)}


def compute_financial_ratios(ticker: str) -> dict[str, Any]:
    """
    Compute all financial ratios by calling analysis.financial_ratios.compute_ratios.
    Returns the full ratios dict.
    """
    return compute_ratios(ticker)
