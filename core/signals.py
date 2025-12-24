# core/signals.py
from __future__ import annotations

import numpy as np
import pandas as pd


def signal_supertrend_plus_volume(
    supertrend_signal: str,
    vol: float,
    avg_vol: float,
    vol_multiplier: float = 1.2,
) -> str:
    if supertrend_signal == "BUY" and pd.notna(avg_vol) and vol > avg_vol * vol_multiplier:
        return "BUY"
    return "SELL"


def signal_combined(supertrend_signal: str, vol_signal: str, rsi: float, rsi_buy_threshold: float = 50.0) -> str:
    if supertrend_signal == "BUY" and vol_signal == "BUY" and pd.notna(rsi) and float(rsi) > rsi_buy_threshold:
        return "BUY"
    return "SELL"


def signal_full_combined(combined_signal: str, most_signal: str) -> str:
    return "BUY" if (combined_signal == "BUY" and most_signal == "BUY") else "SELL"


def signal_super_most_adxr(supertrend_signal: str, most_signal: str, adxr_state: str) -> str:
    """
    Your final rule set:
      EXIT: Supertrend=SELL
      STANDDOWN: ADXR=LOW_FLAT
      BUY: Supertrend=BUY & MOST=BUY & ADXR=RISING
      HOLD: Supertrend=BUY but not aligned (MOST=SELL OR ADXR in FLAT/FALLING)
    """
    if supertrend_signal == "SELL":
        return "EXIT"
    if adxr_state == "LOW_FLAT":
        return "STANDDOWN"
    if supertrend_signal == "BUY" and most_signal == "BUY" and adxr_state == "RISING":
        return "BUY"
    return "HOLD"


FINAL_COLUMN_ORDER = [
    "Ticker",
    "Timeframe",
    "Bar Time",
    "Last Close",
    "SIGNAL-Super-MOST-ADXR",  # MUST be right after Last Close
    "Supertrend",
    "Supertrend Signal",
    "RSI",
    "MOST MA",
    "MOST Line",
    "MOST Signal",
    "ADXR State",
    "ADXR Signal",
    # move volume columns AFTER MOST/ADXR area
    "Volume",
    "Supertrend+Vol Signal",
    "Combined Signal",
    "Full Combined",
]
