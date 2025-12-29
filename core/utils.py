# core/utils.py
from __future__ import annotations

import math
import pandas as pd



import os
import streamlit as st

from dotenv import load_dotenv

# Load .env from current working directory (project root)
load_dotenv()

def get_secret(key: str) -> str:
    """
    Priority:
    1) Streamlit secrets (if present and non-empty)
    2) Environment variables (including .env via load_dotenv)
    """
    # 1) Try Streamlit secrets
    try:
        import streamlit as st
        v = st.secrets.get(key, "")
        if v not in (None, ""):
            return str(v)
    except Exception:
        pass

    # 2) Fallback to env
    return os.environ.get(key, "")


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
