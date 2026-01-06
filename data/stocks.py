# data/stocks.py
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from core.utils import _fix_yf_cols
from core.indicators import apply_indicators
from core.signals import (
    signal_supertrend_plus_volume,
    signal_combined,
    signal_full_combined,
    signal_super_most_adxr,
    FINAL_COLUMN_ORDER,
)


def fetch_stock_1d_df(ticker: str, lookback_days: int = 450) -> pd.DataFrame | None:
    try:
        raw = yf.download(ticker, period=f"{lookback_days}d", interval="1d", progress=False)
       
        if raw is None or raw.empty:
            return None
        raw = _fix_yf_cols(raw)
        df = raw[["High", "Low", "Close", "Volume"]].dropna()
        if ticker == 'BMNR':
            print("len(df) = ", len(df))         
        if df.empty or len(df) < 120:
            return None
        return df
    except Exception:
        if ticker == 'BMNR':
            print("BMNR error - fetch_stock_1d_df")
        return None


def build_stocks_signals_table(
    tickers: list[str],
    *,
    atr_period: int = 10,
    atr_multiplier: float = 3.0,
    rsi_period: int = 14,
    vol_lookback: int = 20,
    vol_multiplier: float = 1.2,
    rsi_buy_threshold: float = 50.0,
    adxr_len: int = 14,
    adxr_lenx: int = 14,
    adxr_low_threshold: float = 20.0,
    adxr_flat_eps: float = 1e-6,
) -> pd.DataFrame:
    rows = []

    for t in tickers:
        if t == 'BMNR':
            print("BMNR reached") 
        base = fetch_stock_1d_df(t)
        if base is None:
            continue
        if t == 'BMNR':
            print("BASE = ", base)         
        df = apply_indicators(
            base,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier,
            rsi_period=rsi_period,
            vol_lookback=vol_lookback,
            adxr_len=adxr_len,
            adxr_lenx=adxr_lenx,
            adxr_low_threshold=adxr_low_threshold,
            adxr_flat_eps=adxr_flat_eps,
        )

        last = df.iloc[-1]

        st_sig = str(last.get("Supertrend_Signal", "SELL"))
        most_sig = str(last.get("MOST_Signal", "SELL"))
        adxr_state = str(last.get("ADXR_State", "FLAT"))

        vol_sig = signal_supertrend_plus_volume(
            st_sig,
            float(last.get("Volume", np.nan)),
            float(last.get("Avg_Volume", np.nan)),
            vol_multiplier=vol_multiplier,
        )
        comb_sig = signal_combined(st_sig, vol_sig, float(last.get("RSI", np.nan)), rsi_buy_threshold=rsi_buy_threshold)
        full_sig = signal_full_combined(comb_sig, most_sig)
        super_most_adxr = signal_super_most_adxr(st_sig, most_sig, adxr_state)

        rows.append(
            {
                "Ticker": t,
                "Timeframe": "1D",
                "Bar Time": last.name,
                "Last Close": round(float(last["Close"]), 2),
                "SIGNAL-Super-MOST-ADXR": super_most_adxr,
                "Supertrend": round(float(last["Supertrend"]), 2) if pd.notna(last["Supertrend"]) else np.nan,
                "Supertrend Signal": st_sig,
                "RSI": round(float(last["RSI"]), 2) if pd.notna(last["RSI"]) else np.nan,
                "MOST MA": round(float(last["MOST_MA"]), 2) if pd.notna(last["MOST_MA"]) else np.nan,
                "MOST Line": round(float(last["MOST_Line"]), 2) if pd.notna(last["MOST_Line"]) else np.nan,
                "MOST Signal": most_sig,
                "ADXR State": adxr_state,
                "ADXR Signal": str(last.get("ADXR_Signal", "WEAK")),
                "Volume": int(last["Volume"]) if pd.notna(last["Volume"]) else np.nan,
                "Supertrend+Vol Signal": vol_sig,
                "Combined Signal": comb_sig,
                "Full Combined": full_sig,
            }
        )

    out = pd.DataFrame(rows)
    return out.reindex(columns=FINAL_COLUMN_ORDER)
