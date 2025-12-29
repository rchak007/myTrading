# core/prices.py
from __future__ import annotations

from decimal import Decimal
import yfinance as yf


def get_yahoo_price_usd(yahoo_ticker: str) -> Decimal:
    """
    Returns last price for Yahoo ticker as Decimal.
    Raises RuntimeError if no data.
    """
    t = yf.Ticker(yahoo_ticker)

    # Try fast_info first
    try:
        fi = getattr(t, "fast_info", None)
        if fi and fi.get("last_price") is not None:
            return Decimal(str(fi["last_price"]))
    except Exception:
        pass

    hist = t.history(period="5d", interval="1d")
    if hist is None or hist.empty:
        raise RuntimeError(f"No Yahoo price data for {yahoo_ticker}")

    last_close = hist["Close"].dropna().iloc[-1]
    return Decimal(str(last_close))
