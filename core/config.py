# core/config.py
"""
Single source of truth for indicator parameters and macro regime settings.

Usage:
    from core.config import INDICATOR_PARAMS, MACRO_REGIME_OVERRIDE

All consumers (app.py, bot.py, jobStocksSignals.py, jobCryptoSignals.py)
should import from here instead of hardcoding values.

bot.py can override via .env (it reads os.getenv with these as defaults).
app.py Streamlit sidebar can override MACRO_REGIME_OVERRIDE at runtime.
"""

from __future__ import annotations
import os


# ═══════════════════════════════════════════════════════════════════
# Indicator Parameters (shared across stocks + crypto)
# ═══════════════════════════════════════════════════════════════════
INDICATOR_PARAMS = {
    "atr_period":         int(os.getenv("ATR_PERIOD",          "10")),
    "atr_multiplier":     float(os.getenv("ATR_MULT",          "3.0")),
    "rsi_period":         int(os.getenv("RSI_PERIOD",          "14")),
    "vol_lookback":       int(os.getenv("VOL_LOOKBACK",        "20")),
    "vol_multiplier":     float(os.getenv("VOL_MULTIPLIER",    "1.2")),
    "rsi_buy_threshold":  float(os.getenv("RSI_BUY_THRESHOLD", "50.0")),
    "adxr_len":           int(os.getenv("ADXR_LEN",            "14")),
    "adxr_lenx":          int(os.getenv("ADXR_LENX",           "14")),
    "adxr_low_threshold": float(os.getenv("ADXR_LOW",          "20.0")),
    "adxr_flat_eps":      float(os.getenv("ADXR_EPS",          "1e-6")),
}


# ═══════════════════════════════════════════════════════════════════
# Macro Regime Override
# ═══════════════════════════════════════════════════════════════════
# Set via .env:  MACRO_REGIME_OVERRIDE=BULL   (or NEUTRAL, BEAR)
# Set to AUTO (default) to let the system compute from VIX+SPY+breadth.
# app.py sidebar can also override this at runtime via Streamlit selectbox.
MACRO_REGIME_OVERRIDE = os.getenv("MACRO_REGIME_OVERRIDE", "AUTO").upper()


# ═══════════════════════════════════════════════════════════════════
# Macro Regime Thresholds (configurable via .env)
# ═══════════════════════════════════════════════════════════════════
MACRO_THRESHOLDS = {
    "vix_low":          float(os.getenv("MACRO_VIX_LOW",         "20")),
    "vix_high":         float(os.getenv("MACRO_VIX_HIGH",        "30")),
    "breadth_healthy":  float(os.getenv("MACRO_BREADTH_HEALTHY", "60")),
    "breadth_weak":     float(os.getenv("MACRO_BREADTH_WEAK",    "50")),
}