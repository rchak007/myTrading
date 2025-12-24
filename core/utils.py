# core/utils.py
from __future__ import annotations

import math
import pandas as pd


def _fix_yf_cols(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance sometimes returns MultiIndex columns; flatten safely."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


def safe_float(x, default=float("nan")) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return default


def fmt_usd_compact(x: float) -> str:
    """$2.88T style formatting."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "N/A"
    ax = abs(x)
    if ax >= 1e12:
        return f"${x/1e12:.2f}T"
    if ax >= 1e9:
        return f"${x/1e9:.2f}B"
    if ax >= 1e6:
        return f"${x/1e6:.2f}M"
    return f"${x:,.0f}"
