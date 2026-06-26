"""
FinSight AI — Anomaly Detection
Statistical outlier flagging for financial metrics.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def zscore_outliers(
    values: list[float],
    threshold: float = 2.0,
) -> list[int]:
    """
    Return indices of values that are statistical outliers (|z-score| > threshold).
    """
    if len(values) < 3:
        return []
    arr = np.array(values, dtype=float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if std == 0:
        return []
    z_scores = np.abs((arr - mean) / std)
    return [int(i) for i in np.where(z_scores > threshold)[0]]


def detect_ratio_anomalies(ratios: dict[str, Any]) -> list[dict[str, str]]:
    """
    Flag financial ratios that are outside normal ranges.
    Returns a list of {metric, value, severity, reason} dicts.
    """
    anomalies: list[dict[str, str]] = []

    checks = [
        ("pe_ratio", 0, 100, "P/E ratio unusually high (>100) or negative"),
        ("debt_to_equity", 0, 5, "Debt/Equity ratio dangerously high (>5)"),
        ("current_ratio", 0.5, 10, "Current ratio below 0.5 (liquidity risk)"),
        ("roe", -50, 100, "ROE outside normal range"),
        ("gross_margin", -20, 100, "Gross margin negative or extremely low"),
        ("operating_margin", -50, 60, "Operating margin severely negative"),
    ]

    for metric, low, high, reason in checks:
        val = ratios.get(metric)
        if val is None:
            continue
        try:
            fval = float(val)
            if fval < low or fval > high:
                severity = "HIGH" if (fval < low * 0.5 or fval > high * 1.5) else "MEDIUM"
                anomalies.append({
                    "metric": metric,
                    "value": str(round(fval, 2)),
                    "severity": severity,
                    "reason": reason,
                })
        except (TypeError, ValueError):
            continue

    return anomalies


def detect_return_anomalies(returns: dict[str, Any]) -> list[dict[str, str]]:
    """Flag abnormal price returns."""
    anomalies: list[dict[str, str]] = []
    return_fields = ["return_1m", "return_3m", "return_6m", "return_1y"]
    thresholds = {
        "return_1m": (-30, 50),
        "return_3m": (-40, 100),
        "return_6m": (-50, 150),
        "return_1y": (-70, 300),
    }
    for field, (low, high) in thresholds.items():
        val = returns.get(field)
        if val is None:
            continue
        try:
            fval = float(val)
            if fval < low:
                anomalies.append({
                    "metric": field,
                    "value": str(round(fval, 2)),
                    "severity": "HIGH",
                    "reason": f"Extreme negative {field.replace('_', ' ')} ({fval:.1f}%)",
                })
            elif fval > high:
                anomalies.append({
                    "metric": field,
                    "value": str(round(fval, 2)),
                    "severity": "MEDIUM",
                    "reason": f"Unusually high {field.replace('_', ' ')} ({fval:.1f}%)",
                })
        except (TypeError, ValueError):
            continue
    return anomalies
