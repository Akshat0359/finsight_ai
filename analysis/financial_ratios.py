"""
FinSight AI — Financial Ratios Computation
Pure Python/pandas — no LLM calls. Uses yfinance for data.
All functions handle None/NaN gracefully.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from cache.disk_cache import TTL_SHORT, build_cache_key, cache_get, cache_set

logger = logging.getLogger(__name__)


def _safe(value: Any, round_digits: int = 4) -> float | None:
    """Return rounded float or None for NaN/None/Inf values."""
    try:
        if value is None:
            return None
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, round_digits)
    except (TypeError, ValueError):
        return None


def _safe_divide(a: Any, b: Any) -> float | None:
    """Safe division returning None on error."""
    try:
        num = float(a)
        den = float(b)
        if den == 0 or np.isnan(num) or np.isnan(den):
            return None
        return round(num / den, 4)
    except (TypeError, ValueError):
        return None


def compute_price_returns(ticker: str) -> dict[str, float | None]:
    """Compute price return metrics using yfinance history."""
    returns: dict[str, float | None] = {
        "return_1m": None,
        "return_3m": None,
        "return_6m": None,
        "return_1y": None,
        "return_ytd": None,
        "volatility_30d": None,
    }
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="1y", auto_adjust=True)
        if hist.empty:
            return returns

        prices = hist["Close"].dropna()
        if len(prices) < 2:
            return returns

        now_price = float(prices.iloc[-1])

        def _return(days: int) -> float | None:
            if len(prices) < days:
                return None
            past = float(prices.iloc[-days])
            if past == 0:
                return None
            return round((now_price - past) / past * 100, 2)

        returns["return_1m"] = _return(21)
        returns["return_3m"] = _return(63)
        returns["return_6m"] = _return(126)
        returns["return_1y"] = _return(252)

        # YTD return
        try:
            import datetime
            ytd_start = prices.index[prices.index.year == pd.Timestamp.now().year][0]
            ytd_price = float(prices.loc[ytd_start])
            if ytd_price != 0:
                returns["return_ytd"] = round((now_price - ytd_price) / ytd_price * 100, 2)
        except (IndexError, KeyError):
            pass

        # 30-day annualised volatility
        if len(prices) >= 30:
            daily_returns = prices.pct_change().dropna()
            vol = float(daily_returns.tail(30).std() * np.sqrt(252) * 100)
            returns["volatility_30d"] = round(vol, 2)

    except Exception as exc:
        logger.warning("Error computing price returns for %s: %s", ticker, exc)

    return returns


def compute_ratios(ticker: str) -> dict[str, Any]:
    """
    Compute all financial ratios for a ticker using yfinance.
    Returns a dict matching the FinancialRatios schema plus price performance.
    Never raises — returns None for unavailable metrics.
    """
    cache_key = build_cache_key("ratios", ticker)
    cached = cache_get(cache_key)
    if cached:
        return cached

    result: dict[str, Any] = {
        # FinancialRatios fields
        "pe_ratio": None,
        "pb_ratio": None,
        "ev_ebitda": None,
        "debt_to_equity": None,
        "current_ratio": None,
        "roe": None,
        "roa": None,
        "gross_margin": None,
        "operating_margin": None,
        "free_cash_flow_ttm": None,
        "revenue_cagr_3y": None,
        # PricePerformance fields
        "return_1m": None,
        "return_3m": None,
        "return_6m": None,
        "return_1y": None,
        "return_ytd": None,
        "volatility_30d": None,
    }

    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}

        # ---- Simple info-based ratios ----
        result["pe_ratio"] = _safe(info.get("trailingPE") or info.get("forwardPE"))
        result["pb_ratio"] = _safe(info.get("priceToBook"))
        result["ev_ebitda"] = _safe(info.get("enterpriseToEbitda"))
        result["current_ratio"] = _safe(info.get("currentRatio"))
        result["gross_margin"] = _safe(
            (info.get("grossMargins") or 0) * 100
        )
        result["operating_margin"] = _safe(
            (info.get("operatingMargins") or 0) * 100
        )

        # ---- Balance sheet ratios ----
        try:
            bs = tk.balance_sheet
            if bs is not None and not bs.empty:
                bs_cols = list(bs.columns)
                latest_col = bs_cols[0]

                total_debt = bs.get("Total Debt", pd.Series()).get(latest_col, None)
                stockholders_equity = bs.get("Stockholders Equity", pd.Series()).get(
                    latest_col, None
                )
                total_assets = bs.get("Total Assets", pd.Series()).get(latest_col, None)

                result["debt_to_equity"] = _safe_divide(total_debt, stockholders_equity)

                # ROE = Net Income / Shareholders Equity
                try:
                    inc = tk.financials
                    if inc is not None and not inc.empty:
                        net_income = inc.get("Net Income", pd.Series()).iloc[0]
                        result["roe"] = _safe_divide(net_income, stockholders_equity)
                        if result["roe"] is not None:
                            result["roe"] = round(result["roe"] * 100, 2)
                        result["roa"] = _safe_divide(net_income, total_assets)
                        if result["roa"] is not None:
                            result["roa"] = round(result["roa"] * 100, 2)
                except Exception:
                    pass
        except Exception as exc:
            logger.debug("Balance sheet error for %s: %s", ticker, exc)

        # ---- Free Cash Flow TTM ----
        try:
            cf = tk.cashflow
            if cf is not None and not cf.empty:
                op_cf = cf.get("Operating Cash Flow", pd.Series())
                capex = cf.get("Capital Expenditure", pd.Series())
                if not op_cf.empty and not capex.empty:
                    ttm_op = float(op_cf.iloc[0])
                    ttm_capex = float(capex.iloc[0])
                    result["free_cash_flow_ttm"] = _safe(ttm_op + ttm_capex)  # capex is negative
        except Exception as exc:
            logger.debug("Cash flow error for %s: %s", ticker, exc)

        # ---- Revenue CAGR 3Y ----
        try:
            fin = tk.financials
            if fin is not None and not fin.empty:
                rev_row = fin.get("Total Revenue", fin.get("Revenue", pd.Series()))
                if hasattr(rev_row, "dropna"):
                    rev_vals = rev_row.dropna()
                    if len(rev_vals) >= 4:
                        r_now = float(rev_vals.iloc[0])
                        r_3y = float(rev_vals.iloc[3])
                        if r_3y > 0:
                            cagr = ((r_now / r_3y) ** (1 / 3) - 1) * 100
                            result["revenue_cagr_3y"] = _safe(cagr)
        except Exception as exc:
            logger.debug("Revenue CAGR error for %s: %s", ticker, exc)

        # ---- Price performance ----
        price_perf = compute_price_returns(ticker)
        result.update(price_perf)

    except Exception as exc:
        logger.error("Error computing ratios for %s: %s", ticker, exc)

    cache_set(cache_key, result, ttl=TTL_SHORT)
    return result
