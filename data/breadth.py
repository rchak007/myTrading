# data/breadth.py
from __future__ import annotations

import numpy as np
import yfinance as yf

from core.utils import _fix_yf_cols


def fetch_vix_value() -> float:
    try:
        vix = yf.download("^VIX", period="10d", interval="1d", progress=False)
        if vix is None or vix.empty:
            return float("nan")
        vix = _fix_yf_cols(vix)
        return float(vix["Close"].dropna().iloc[-1])
    except Exception:
        return float("nan")


def fetch_spy_vs_200ma():
    """
    Returns: (spy_close, spy_ma200, status: ABOVE/BELOW/UNKNOWN)
    """
    try:
        spy = yf.download("SPY", period="450d", interval="1d", progress=False)
        if spy is None or spy.empty:
            return (float("nan"), float("nan"), "UNKNOWN")
        spy = _fix_yf_cols(spy)
        close = spy["Close"].dropna()
        ma200 = close.rolling(200).mean()
        last_close = float(close.iloc[-1])
        last_ma200 = float(ma200.iloc[-1]) if not np.isnan(ma200.iloc[-1]) else float("nan")
        status = "UNKNOWN" if np.isnan(last_ma200) else ("ABOVE" if last_close >= last_ma200 else "BELOW")
        return (last_close, last_ma200, status)
    except Exception:
        return (float("nan"), float("nan"), "UNKNOWN")


def breadth_proxy_from_spy(spy_close: float, spy_ma200: float) -> dict:
    """
    NOTE: This is a proxy (not true “% of S&P > 200MA” breadth).
    You requested a simple, stable solution that never scrapes 500 tickers.
    """
    if np.isnan(spy_close) or np.isnan(spy_ma200) or spy_ma200 == 0:
        return {"pct": np.nan, "status": "Unknown", "action": "N/A"}

    near = abs(spy_close - spy_ma200) / spy_ma200 < 0.01

    if spy_close > spy_ma200:
        return {"pct": 62.0, "status": "Healthy", "action": "✅ Breadth is healthy → Trade full size"}
    if near:
        return {"pct": 55.0, "status": "Weakening", "action": "🟡 Breadth weakening → Trade half size"}
    return {"pct": 45.0, "status": "Poor", "action": "🔴 Breadth poor → Sit out"}
