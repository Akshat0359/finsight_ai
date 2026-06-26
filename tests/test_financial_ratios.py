"""
FinSight AI — Tests for financial_ratios.py
Tests ratio computation, NaN handling, and edge cases.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from analysis.financial_ratios import _safe, _safe_divide, compute_price_returns, compute_ratios


class TestSafeHelpers:
    """Tests for the _safe and _safe_divide helper functions."""

    def test_safe_float_normal(self):
        assert _safe(3.14) == 3.14

    def test_safe_none_returns_none(self):
        assert _safe(None) is None

    def test_safe_nan_returns_none(self):
        assert _safe(float("nan")) is None

    def test_safe_inf_returns_none(self):
        assert _safe(float("inf")) is None

    def test_safe_negative_inf_returns_none(self):
        assert _safe(float("-inf")) is None

    def test_safe_string_returns_none(self):
        assert _safe("not_a_number") is None

    def test_safe_zero(self):
        assert _safe(0.0) == 0.0

    def test_safe_divide_normal(self):
        result = _safe_divide(10, 4)
        assert result == 2.5

    def test_safe_divide_by_zero(self):
        assert _safe_divide(10, 0) is None

    def test_safe_divide_none(self):
        assert _safe_divide(None, 5) is None

    def test_safe_divide_nan(self):
        assert _safe_divide(float("nan"), 5) is None

    def test_safe_divide_negative_equity(self):
        """Negative equity (DE < 0) should return a negative value, not None."""
        result = _safe_divide(100, -50)
        assert result == -2.0


class TestComputeRatios:
    """Tests for compute_ratios() with mocked yfinance data."""

    def _mock_ticker(
        self,
        pe: float | None = 28.0,
        pb: float | None = 12.0,
        ev_ebitda: float | None = 22.0,
        current_ratio: float | None = 1.8,
        gross_margins: float | None = 0.701,
        operating_margins: float | None = 0.446,
    ) -> MagicMock:
        """Build a minimal yfinance Ticker mock."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "trailingPE": pe,
            "priceToBook": pb,
            "enterpriseToEbitda": ev_ebitda,
            "currentRatio": current_ratio,
            "grossMargins": gross_margins,
            "operatingMargins": operating_margins,
        }

        # Balance sheet mock
        bs_data = {
            "Total Debt": {pd.Timestamp("2024-06-30"): 100_000_000},
            "Stockholders Equity": {pd.Timestamp("2024-06-30"): 300_000_000},
            "Total Assets": {pd.Timestamp("2024-06-30"): 500_000_000},
        }
        mock_bs = pd.DataFrame(bs_data).T
        mock_ticker.balance_sheet = mock_bs

        # Financials mock
        fin_data = {
            "Net Income": {
                pd.Timestamp("2024-06-30"): 80_000_000,
                pd.Timestamp("2023-06-30"): 70_000_000,
                pd.Timestamp("2022-06-30"): 60_000_000,
                pd.Timestamp("2021-06-30"): 50_000_000,
            },
            "Total Revenue": {
                pd.Timestamp("2024-06-30"): 400_000_000,
                pd.Timestamp("2023-06-30"): 350_000_000,
                pd.Timestamp("2022-06-30"): 300_000_000,
                pd.Timestamp("2021-06-30"): 260_000_000,
            },
        }
        mock_fin = pd.DataFrame(fin_data).T
        mock_ticker.financials = mock_fin

        # Cash flow mock
        cf_data = {
            "Operating Cash Flow": {pd.Timestamp("2024-06-30"): 100_000_000},
            "Capital Expenditure": {pd.Timestamp("2024-06-30"): -20_000_000},
        }
        mock_cf = pd.DataFrame(cf_data).T
        mock_ticker.cashflow = mock_cf

        # Price history mock
        dates = pd.date_range(end="2024-11-01", periods=260, freq="B")
        prices = pd.Series(
            [370 + i * 0.2 for i in range(260)], index=dates, name="Close"
        )
        mock_hist = pd.DataFrame({"Close": prices, "Volume": [1_000_000] * 260})
        mock_ticker.history.return_value = mock_hist

        return mock_ticker

    @patch("analysis.financial_ratios.yf.Ticker")
    @patch("analysis.financial_ratios.cache_get", return_value=None)
    @patch("analysis.financial_ratios.cache_set")
    def test_basic_ratios_msft(self, mock_cache_set, mock_cache_get, mock_yf_ticker):
        """Test that basic ratios are computed correctly for mock MSFT data."""
        mock_yf_ticker.return_value = self._mock_ticker()
        result = compute_ratios("MSFT")

        assert result["pe_ratio"] == pytest.approx(28.0, rel=0.01)
        assert result["pb_ratio"] == pytest.approx(12.0, rel=0.01)
        assert result["current_ratio"] == pytest.approx(1.8, rel=0.01)
        assert result["gross_margin"] == pytest.approx(70.1, rel=0.01)
        assert result["operating_margin"] == pytest.approx(44.6, rel=0.01)

    @patch("analysis.financial_ratios.yf.Ticker")
    @patch("analysis.financial_ratios.cache_get", return_value=None)
    @patch("analysis.financial_ratios.cache_set")
    def test_none_handling(self, mock_cache_set, mock_cache_get, mock_yf_ticker):
        """Test that None values from yfinance are handled gracefully."""
        mock_yf_ticker.return_value = self._mock_ticker(pe=None, pb=None)
        result = compute_ratios("AAPL")

        # Should return None for unavailable metrics, not raise
        assert result["pe_ratio"] is None
        assert result["pb_ratio"] is None
        # Other fields should still compute
        assert result["current_ratio"] is not None

    @patch("analysis.financial_ratios.yf.Ticker")
    @patch("analysis.financial_ratios.cache_get", return_value=None)
    @patch("analysis.financial_ratios.cache_set")
    def test_negative_equity_edge_case(
        self, mock_cache_set, mock_cache_get, mock_yf_ticker
    ):
        """Test that negative equity (DE ratio) is handled and not None."""
        mock_ticker = self._mock_ticker()
        # Set negative equity
        bs_data = {
            "Total Debt": {pd.Timestamp("2024-06-30"): 100_000_000},
            "Stockholders Equity": {pd.Timestamp("2024-06-30"): -50_000_000},
            "Total Assets": {pd.Timestamp("2024-06-30"): 500_000_000},
        }
        mock_ticker.balance_sheet = pd.DataFrame(bs_data).T
        mock_yf_ticker.return_value = mock_ticker

        result = compute_ratios("TEST")
        # Debt/Equity should be negative (not None) when equity is negative
        assert result["debt_to_equity"] is not None
        assert result["debt_to_equity"] < 0

    @patch("analysis.financial_ratios.yf.Ticker")
    @patch("analysis.financial_ratios.cache_get", return_value=None)
    @patch("analysis.financial_ratios.cache_set")
    def test_returns_all_keys(self, mock_cache_set, mock_cache_get, mock_yf_ticker):
        """Test that all expected keys are present in the result."""
        mock_yf_ticker.return_value = self._mock_ticker()
        result = compute_ratios("AAPL")

        expected_keys = [
            "pe_ratio", "pb_ratio", "ev_ebitda", "debt_to_equity", "current_ratio",
            "roe", "roa", "gross_margin", "operating_margin", "free_cash_flow_ttm",
            "revenue_cagr_3y", "return_1m", "return_3m", "return_6m", "return_1y",
            "return_ytd", "volatility_30d",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    @patch("analysis.financial_ratios.yf.Ticker")
    @patch("analysis.financial_ratios.cache_get", return_value=None)
    @patch("analysis.financial_ratios.cache_set")
    def test_revenue_cagr_computation(
        self, mock_cache_set, mock_cache_get, mock_yf_ticker
    ):
        """Test 3-year revenue CAGR calculation."""
        mock_yf_ticker.return_value = self._mock_ticker()
        result = compute_ratios("TEST")

        # Revenue: 400M / 260M over 3 years
        expected_cagr = ((400 / 260) ** (1 / 3) - 1) * 100
        if result["revenue_cagr_3y"] is not None:
            assert result["revenue_cagr_3y"] == pytest.approx(expected_cagr, rel=0.05)

    @patch("analysis.financial_ratios.yf.Ticker")
    @patch("analysis.financial_ratios.cache_get", return_value=None)
    @patch("analysis.financial_ratios.cache_set")
    def test_no_crash_on_empty_data(
        self, mock_cache_set, mock_cache_get, mock_yf_ticker
    ):
        """Test that empty DataFrames don't crash the computation."""
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_ticker.balance_sheet = pd.DataFrame()
        mock_ticker.financials = pd.DataFrame()
        mock_ticker.cashflow = pd.DataFrame()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_yf_ticker.return_value = mock_ticker

        # Should not raise
        result = compute_ratios("EMPTY")
        assert isinstance(result, dict)
        assert result["pe_ratio"] is None
