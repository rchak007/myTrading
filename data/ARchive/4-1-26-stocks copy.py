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
from core.macro import compute_regime_action

# Import scoring functions
from data.stock_scoring import (
    calculate_all_scores,
    get_earnings_alert,
)


def fetch_market_cap(ticker: str) -> str:
    """
    Fetch market cap for a single ticker via yfinance.info.
    Returns a compact string like '$1.23T', '$456B', '$12.3M'.
    Falls back to 'N/A' gracefully — never raises.
    ETFs / missing data → 'N/A'.
    """
    try:
        info = yf.Ticker(ticker).info
        val = info.get("marketCap") or info.get("market_cap")
        if val and isinstance(val, (int, float)) and val > 0:
            return round(float(val) / 1_000_000, 2)  # convert to millions        
    except Exception:
        pass
    return "N/A"



def fetch_current_price(ticker: str) -> float:
    """
    Fetch the live/intraday price for a ticker.
    Strategy (fastest → most reliable):
      1. 1-minute bar download (period=1d, interval=1m) — most recent tick
      2. fast_info.last_price  — near-real-time, usually accurate during market hours
      3. info.regularMarketPrice — heavier call, last resort
    Returns np.nan on failure.
    """
    # 1. Latest 1m bar — gives the true current price during market hours
    try:
        df = yf.download(ticker, period="1d", interval="1m", progress=False)
        if df is not None and not df.empty:
            # price = float(df["Close"].iloc[-1])
            val = df["Close"].iloc[-1]
            price = float(val.iloc[0]) if hasattr(val, "iloc") else float(val)
            if price > 0:
                return round(price, 4)
    except Exception:
        pass

    # 2. fast_info — lightweight, near-real-time
    try:
        fi = yf.Ticker(ticker).fast_info
        price = getattr(fi, "last_price", None)
        if price and price > 0:
            return round(float(price), 4)
    except Exception:
        pass

    # 3. Full info — heaviest, last resort
    try:
        info = yf.Ticker(ticker).info
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        if price and price > 0:
            return round(float(price), 4)
    except Exception:
        pass

    return np.nan

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


def _fetch_spy_close(lookback_days: int = 450) -> pd.Series | None:
    """
    Fetch SPY close prices for relative strength calculations.
    Returns None if fetch fails.
    """
    try:
        spy_raw = yf.download("SPY", period=f"{lookback_days}d", interval="1d", progress=False)
        if spy_raw is None or spy_raw.empty:
            return None
        spy_raw = _fix_yf_cols(spy_raw)
        return spy_raw["Close"]
    except Exception:
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
    include_scoring: bool = True,
) -> pd.DataFrame:
    """
    Build stocks signals table with optional 45° trend scoring and earnings alerts.
    
    Parameters:
    -----------
    include_scoring : bool, default True
        If True, includes "Score" and "Earnings_Alert" columns
    """
    rows = []
    
    # Fetch SPY data once for all stocks if scoring is enabled
    spy_close = _fetch_spy_close() if include_scoring else None

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

        # Calculate scoring and earnings if enabled
        score_30 = score_60 = score_90 = score_120 = score_weighted = np.nan
        earnings_alert = ""
        if include_scoring:
            try:
                score_data = calculate_all_scores(base, spy_close)
                score_30  = score_data["Score_30"]
                score_60  = score_data["Score_60"]
                score_90  = score_data["Score_90"]
                score_120 = score_data["Score_120"]
                score_weighted = score_data["Score_Weighted"]
                earnings_alert = get_earnings_alert(t)
            except Exception as e:
                print(f"Warning: Could not calculate score for {t}: {e}")

        # Fetch market cap — always, graceful N/A on failure (no crypto, stocks only)
        market_cap = fetch_market_cap(t)

        # Fetch current price for display
        current_price = fetch_current_price(t)

        row = {
            "Ticker": t,
            "Timeframe": "1D",
            "Bar Time": last.name,
            "Last Close": round(float(last["Close"]), 2),
            "Current Price": current_price,
            "SIGNAL-Super-MOST-ADXR": super_most_adxr,
        }
        
        # Add scoring columns after SIGNAL-Super-MOST-ADXR
        if include_scoring:
            row["Score_30"]  = int(score_30)  if pd.notna(score_30)  else 0
            row["Score_60"]  = int(score_60)  if pd.notna(score_60)  else 0
            row["Score_90"]  = int(score_90)  if pd.notna(score_90)  else 0
            row["Score_120"] = int(score_120) if pd.notna(score_120) else 0
            row["Score_Weighted"] = int(score_weighted) if pd.notna(score_weighted) else 0
            row["Earnings_Alert"] = earnings_alert

        # Market_Cap always after signal block
        # row["Market_Cap"] = market_cap
        row["Market_Cap_M"] = market_cap
        
        # Continue with existing columns
        row.update({
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
        })
        
        rows.append(row)

    out = pd.DataFrame(rows)
    
    # Define column order with new columns
    if include_scoring:
        columns_order = [
            "Ticker", "Timeframe", "Bar Time", "Last Close", "Current Price",
            "SIGNAL-Super-MOST-ADXR", "Supertrend", "Score_30", "Score_60", "Score_90", "Score_120", "Score_Weighted",
            "Earnings_Alert", "Market_Cap_M",
             "Supertrend Signal", "RSI",
            "MOST MA", "MOST Line", "MOST Signal",
            "ADXR State", "ADXR Signal", "Volume",
            "Supertrend+Vol Signal", "Combined Signal", "Full Combined"
        ]
    else:
        # Non-scoring path: insert Market_Cap right after SIGNAL-Super-MOST-ADXR
        base_order = list(FINAL_COLUMN_ORDER)
        sig_col = "SIGNAL-Super-MOST-ADXR"
        if sig_col in base_order:
            base_order.insert(base_order.index(sig_col) + 1, "Market_Cap")
        columns_order = base_order
    
    # Reindex with available columns only
    available_cols = [col for col in columns_order if col in out.columns]
    return out.reindex(columns=available_cols)


# ═══════════════════════════════════════════════════════════════════
# Macro regime enrichment (called AFTER build_stocks_signals_table)
# ═══════════════════════════════════════════════════════════════════

def add_macro_regime_columns(
    df: pd.DataFrame,
    macro_regime: str,
) -> pd.DataFrame:
    """
    Add 'Macro_Regime' and 'Regime_Action' columns to a stock signals DataFrame.

    Call this AFTER build_stocks_signals_table() with the regime string
    from compute_macro_regime().  Keeps build_stocks_signals_table() unchanged
    so crypto / other callers are unaffected.

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_stocks_signals_table()
    macro_regime : str
        "BULL", "NEUTRAL", or "BEAR" — from compute_macro_regime() or override.

    Returns
    -------
    pd.DataFrame with two new columns inserted right after SIGNAL-Super-MOST-ADXR:
        Macro_Regime  : str — same for every row (market-wide)
        Regime_Action : str — per-stock action combining signal + regime + score
    """
    if df.empty:
        return df

    df = df.copy()

    # Add Macro_Regime (same for all rows — it's a market-level field)
    df["Macro_Regime"] = macro_regime

    # Add Regime_Action (per-stock, combines signal + regime + score)
    sig_col = "SIGNAL-Super-MOST-ADXR"
    score_col = "Score_Weighted"

    def _action(row):
        sig = str(row.get(sig_col, ""))
        score = float(row.get(score_col, 0)) if pd.notna(row.get(score_col)) else 0.0
        return compute_regime_action(sig, macro_regime, score)

    df["Regime_Action"] = df.apply(_action, axis=1)

    # Reorder: insert Macro_Regime and Regime_Action right after SIGNAL-Super-MOST-ADXR
    cols = list(df.columns)
    for new_col in ["Regime_Action", "Macro_Regime"]:
        if new_col in cols:
            cols.remove(new_col)
    if sig_col in cols:
        idx = cols.index(sig_col) + 1
        cols.insert(idx, "Macro_Regime")
        cols.insert(idx + 1, "Regime_Action")
    else:
        # Fallback: append
        cols.extend(["Macro_Regime", "Regime_Action"])

    return df.reindex(columns=cols)