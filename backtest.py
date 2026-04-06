#!/usr/bin/env python3
"""
backtest.py — Historical backtest using the same Supertrend + MOST RSI + ADXR
signal logic from the live trading system.

Usage examples:
  # Stock, 1D chart, last 2 years
  python3 backtest.py AAPL -c 1d -t 2y

  # Crypto (Yahoo symbol), 4H chart with 1D filter, last 1 year
  python3 backtest.py SOL-USD -c 4h -f 1d -t 1y

  # Crypto not on Yahoo — use GeckoTerminal pool URL
  python3 backtest.py LQL -c 4h -t 6m \
    --gecko "https://www.geckoterminal.com/solana/pools/GiRyo4r3kREH8oRCe9GoJJARZuGo4ksto6xXvUok4wdd"

  # With custom indicator params
  python3 backtest.py ETH-USD -c 4h -f 1d -t 2y --atr-period 7 --atr-mult 2.5

Reuses: core.indicators.apply_indicators, core.signals.signal_super_most_adxr
"""

from __future__ import annotations

import argparse
import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ── Project imports (same as bot.py / crypto.py / stocks.py) ──
from core.utils import _fix_yf_cols
from core.indicators import apply_indicators
from core.signals import signal_super_most_adxr
from core.ohlcv_cache import cached_yahoo_download, cached_gecko_download


# ═══════════════════════════════════════════════════════════════════
# Constants — indicator defaults from single source of truth
# ═══════════════════════════════════════════════════════════════════
from core.config import INDICATOR_PARAMS

DEFAULT_ATR_PERIOD   = INDICATOR_PARAMS["atr_period"]
DEFAULT_ATR_MULT     = INDICATOR_PARAMS["atr_multiplier"]
DEFAULT_RSI_PERIOD   = INDICATOR_PARAMS["rsi_period"]
DEFAULT_VOL_LOOKBACK = INDICATOR_PARAMS["vol_lookback"]
DEFAULT_ADXR_LEN     = INDICATOR_PARAMS["adxr_len"]
DEFAULT_ADXR_LENX    = INDICATOR_PARAMS["adxr_lenx"]
DEFAULT_ADXR_LOW     = INDICATOR_PARAMS["adxr_low_threshold"]
DEFAULT_ADXR_EPS     = INDICATOR_PARAMS["adxr_flat_eps"]

# How to map user-friendly intervals to yfinance parameters
# For sub-daily intervals, yfinance needs "60m" and we resample to 4h etc.
YF_INTERVAL_MAP = {
    "1h":  "60m",
    "2h":  "60m",     # download 1h, resample to 2h
    "4h":  "60m",     # download 1h, resample to 4h
    "1d":  "1d",
    "1wk": "1wk",
}

# Lookback days to request from Yahoo for each chart interval
# For sub-daily: yfinance max is ~730 days for 1h data
LOOKBACK_MAP = {
    "1h":   60,
    "2h":  120,
    "4h":  720,       # Max ~730 days for 1h data on Yahoo
    "1d":  3650,      # ~10 years
    "1wk": 7300,      # ~20 years
}

# User-friendly timeframe string → days
TIMEFRAME_MAP = {
    "1m":   30,
    "2m":   60,
    "3m":   90,
    "6m":   180,
    "1y":   365,
    "2y":   730,
    "3y":   1095,
    "5y":   1825,
    "10y":  3650,
    "max":  9999,
}


# ═══════════════════════════════════════════════════════════════════
# Data fetching (reuses existing patterns from crypto.py / stocks.py)
# ═══════════════════════════════════════════════════════════════════

def fetch_yahoo_ohlcv(
    ticker: str,
    interval: str,
    lookback_days: int,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch OHLCV from Yahoo Finance with Parquet disk caching.
    For sub-daily intervals (1h/2h/4h), downloads 1h bars and resamples.
    For 1d/1wk, downloads directly.

    Always fetches extra data beyond lookback_days for indicator warmup
    (Supertrend, ADXR, MOST need ~100+ bars). The backtest engine trims
    to the correct window after indicators are applied.

    Returns DataFrame with columns: High, Low, Close, Volume
    Index: DatetimeIndex (tz-naive)
    """
    yf_interval = YF_INTERVAL_MAP.get(interval, "1d")
    max_lookback = LOOKBACK_MAP.get(interval, 365)

    # Add warmup buffer: need ~150 extra bars for indicators to stabilize
    warmup_days = {"1h": 10, "2h": 15, "4h": 30, "1d": 250, "1wk": 750}
    warmup = warmup_days.get(interval, 250)
    fetch_days = min(lookback_days + warmup, max_lookback)

    # Use the central cache layer
    raw = cached_yahoo_download(ticker, yf_interval, fetch_days, force_refresh=force_refresh)
    if raw is None or raw.empty:
        raise RuntimeError(f"No yfinance data for {ticker} (interval={yf_interval}, lookback={fetch_days}d)")

    df = raw[["High", "Low", "Close", "Volume"]].dropna()

    # Strip timezone (cache should already do this, but be safe)
    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        pass

    # Resample if sub-daily
    if interval in ("2h", "4h"):
        resample_rule = {"2h": "2h", "4h": "4h"}[interval]
        resampled = df.resample(resample_rule)
        df = pd.DataFrame({
            "High":   resampled["High"].max(),
            "Low":    resampled["Low"].min(),
            "Close":  resampled["Close"].last(),
            "Volume": resampled["Volume"].sum(),
        }).dropna()

    if len(df) < 100:
        raise RuntimeError(f"Insufficient data for {ticker}: {len(df)} bars (need >= 100)")

    print(f"✅ {ticker}: {len(df)} bars ({df.index[0]} → {df.index[-1]})")
    return df


def fetch_gecko_ohlcv(
    pool_url: str,
    interval: str,
    lookback_days: int,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch OHLCV from GeckoTerminal API with Parquet disk caching.
    Supports 1h, 4h, 1d intervals.
    """
    # Use the central cache layer
    df = cached_gecko_download(pool_url, interval, lookback_days, force_refresh=force_refresh)
    if df is None or df.empty:
        raise RuntimeError(f"No GeckoTerminal data for {pool_url} (interval={interval})")

    # Strip timezone for consistency
    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        try:
            df.index = df.index.tz_convert(None)
        except Exception:
            pass

    if len(df) < 100:
        raise RuntimeError(f"Insufficient GeckoTerminal data: {len(df)} bars (need >= 100)")

    return df


# ═══════════════════════════════════════════════════════════════════
# Historical Macro Regime (VIX + SPY vs 200MA + breadth proxy)
# ═══════════════════════════════════════════════════════════════════

def fetch_historical_macro_data(lookback_days: int = 3650) -> pd.DataFrame:
    """
    Fetch historical VIX + SPY daily data for regime computation.
    Returns a DataFrame indexed by date with columns:
      VIX_Close, SPY_Close, SPY_200MA, SPY_Above_200MA, Breadth_Pct, Macro_Regime
    """
    from core.macro import compute_macro_regime

    period = f"{lookback_days}d"

    # Fetch VIX
    vix_df = yf.download("^VIX", period=period, interval="1d", progress=False)
    if vix_df is not None and not vix_df.empty:
        vix_df = _fix_yf_cols(vix_df)
        vix_close = vix_df["Close"].rename("VIX_Close")
    else:
        print("⚠️  VIX data unavailable — regime will assume VIX=25 (cautious)")
        vix_close = pd.Series(dtype=float, name="VIX_Close")

    # Fetch SPY
    spy_df = yf.download("SPY", period=period, interval="1d", progress=False)
    if spy_df is not None and not spy_df.empty:
        spy_df = _fix_yf_cols(spy_df)
        spy_close = spy_df["Close"].rename("SPY_Close")
        spy_200ma = spy_close.rolling(200, min_periods=200).mean().rename("SPY_200MA")
    else:
        print("⚠️  SPY data unavailable — regime will default to NEUTRAL")
        spy_close = pd.Series(dtype=float, name="SPY_Close")
        spy_200ma = pd.Series(dtype=float, name="SPY_200MA")

    # Combine into daily DataFrame
    macro_df = pd.concat([vix_close, spy_close, spy_200ma], axis=1).sort_index()

    # Compute derived columns
    macro_df["SPY_Above_200MA"] = macro_df["SPY_Close"] > macro_df["SPY_200MA"]

    # Breadth proxy: same logic as data/breadth.py breadth_proxy_from_spy
    # (% distance of SPY from 200MA, mapped to estimated breadth %)
    dist_pct = ((macro_df["SPY_Close"] / macro_df["SPY_200MA"]) - 1) * 100
    macro_df["Breadth_Pct"] = (50 + dist_pct * 3).clip(10, 90)

    # Compute regime for each day
    regimes = []
    for idx, row in macro_df.iterrows():
        vix_val = float(row["VIX_Close"]) if pd.notna(row["VIX_Close"]) else np.nan
        spy_above = bool(row["SPY_Above_200MA"]) if pd.notna(row["SPY_Above_200MA"]) else False
        breadth = float(row["Breadth_Pct"]) if pd.notna(row["Breadth_Pct"]) else np.nan

        info = compute_macro_regime(vix_val, spy_above, breadth)
        regimes.append(info["regime"])

    macro_df["Macro_Regime"] = regimes

    # Strip timezone for clean joining
    try:
        macro_df.index = macro_df.index.tz_localize(None)
    except Exception:
        pass

    return macro_df


def lookup_regime_for_bar(
    bar_time: pd.Timestamp,
    macro_df: pd.DataFrame,
    override: str = "AUTO",
) -> str:
    """
    Look up the macro regime for a given bar timestamp.
    Uses the most recent daily regime at or before the bar time.

    Parameters
    ----------
    bar_time : pd.Timestamp
        The bar's timestamp (can be intraday for 4H etc.)
    macro_df : pd.DataFrame
        Output of fetch_historical_macro_data(), indexed by date
    override : str
        "AUTO" to use historical data, or "BULL"/"NEUTRAL"/"BEAR" to force

    Returns
    -------
    str : "BULL", "NEUTRAL", or "BEAR"
    """
    if override and override.upper() in ("BULL", "NEUTRAL", "BEAR"):
        return override.upper()

    if macro_df is None or macro_df.empty:
        return "NEUTRAL"

    # Normalize bar_time to date for comparison with daily index
    bar_date = pd.Timestamp(bar_time).normalize()
    if hasattr(bar_date, 'tzinfo') and bar_date.tzinfo is not None:
        bar_date = bar_date.tz_localize(None)

    mask = macro_df.index <= bar_date
    if mask.any():
        return str(macro_df.loc[mask, "Macro_Regime"].iloc[-1])
    return "NEUTRAL"


def apply_regime_to_signal(signal: str, regime: str) -> str:
    """
    Modify a backtest signal based on macro regime.
    For stocks (not crypto):
      - BEAR:    BUY → WAIT, HOLD → EXIT  (don't enter in bear markets)
      - NEUTRAL: BUY → BUY, HOLD → HOLD   (no change, standard conservative)
      - BULL:    BUY → BUY, HOLD → BUY    (HOLD upgraded, more aggressive entries)
      - EXIT and STANDDOWN always pass through unchanged

    Returns the modified signal string.
    """
    sig = signal.upper().strip()

    # EXIT and STANDDOWN always pass through
    if sig in ("EXIT", "STANDDOWN", "WAIT"):
        return sig

    if regime == "BEAR":
        if sig == "BUY":
            return "WAIT"       # Don't enter in bear
        if sig == "HOLD":
            return "EXIT"       # Don't hold in bear either
        return sig

    if regime == "NEUTRAL":
        return sig              # No modification — standard rules

    if regime == "BULL":
        if sig == "HOLD":
            return "BUY"        # Upgrade HOLD → BUY in bull
        return sig

    return sig


# ═══════════════════════════════════════════════════════════════════
# Core backtest engine
# ═══════════════════════════════════════════════════════════════════

def run_backtest(
    df_chart: pd.DataFrame,
    df_filter: pd.DataFrame | None,
    *,
    ticker: str,
    chart_interval: str,
    filter_interval: str | None,
    backtest_days: int | None = None,
    backtest_start_date: str | None = None,
    backtest_end_date: str | None = None,
    in_token_pct: float = 0.80,
    out_token_pct: float = 0.20,
    atr_period: int   = DEFAULT_ATR_PERIOD,
    atr_mult: float   = DEFAULT_ATR_MULT,
    rsi_period: int   = DEFAULT_RSI_PERIOD,
    vol_lookback: int = DEFAULT_VOL_LOOKBACK,
    adxr_len: int     = DEFAULT_ADXR_LEN,
    adxr_lenx: int    = DEFAULT_ADXR_LENX,
    adxr_low: float   = DEFAULT_ADXR_LOW,
    adxr_eps: float   = DEFAULT_ADXR_EPS,
    initial_capital: float = 10000.0,
    macro_regime_mode: str = "OFF",
    macro_df: pd.DataFrame | None = None,
    macro_override: str = "AUTO",
) -> dict:
    """
    Walk-forward backtest using the exact same signal logic as the live bot.

    Position sizing mirrors the live bot:
      - BUY (OUT→IN):  move in_token_pct of total value into token, rest stays cash
      - SELL (IN→OUT):  move to out_token_pct in token, rest goes to cash
      Default: in_token_pct=0.80, out_token_pct=0.20

    For each bar in df_chart:
      1. Compute signal_super_most_adxr (Supertrend + MOST + ADXR)
      2. If filter provided, compute filter signal on df_filter and apply
         combine_4h_1d_signals logic
      3. Track regime transitions: OUT→IN = BUY, IN→OUT = SELL
      4. Record trades, equity curve, and performance stats

    backtest_days: if set, trims the simulation to only the last N days
                   (indicators are computed on full data for warmup, then trimmed)

    Returns dict with: trades, equity_curve, stats, signals_df
    """
    indicator_params = dict(
        atr_period=atr_period,
        atr_multiplier=atr_mult,
        rsi_period=rsi_period,
        vol_lookback=vol_lookback,
        adxr_len=adxr_len,
        adxr_lenx=adxr_lenx,
        adxr_low_threshold=adxr_low,
        adxr_flat_eps=adxr_eps,
    )

    # ── Apply indicators to FULL data (warmup period included) ──
    print(f"📊 Applying indicators to {chart_interval} data ({len(df_chart)} bars) ...")
    ind_chart = apply_indicators(df_chart, **indicator_params)

    # ── Apply indicators to filter timeframe (if provided) ──
    ind_filter = None
    if df_filter is not None and not df_filter.empty:
        print(f"📊 Applying indicators to {filter_interval} filter ({len(df_filter)} bars) ...")
        ind_filter = apply_indicators(df_filter, **indicator_params)

    # ── Trim to backtest window (indicators already warmed up) ──
    if backtest_start_date is not None or backtest_end_date is not None:
        # Explicit date range
        pre_trim = len(ind_chart)
        if backtest_start_date is not None:
            start_ts = pd.Timestamp(backtest_start_date)
            ind_chart = ind_chart[ind_chart.index >= start_ts]
        if backtest_end_date is not None:
            end_ts = pd.Timestamp(backtest_end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            ind_chart = ind_chart[ind_chart.index <= end_ts]
        print(f"✂️  Trimmed chart: {pre_trim} → {len(ind_chart)} bars (date range: {backtest_start_date} → {backtest_end_date})")
    elif backtest_days is not None:
        from datetime import timedelta
        last_bar = ind_chart.index[-1]
        # Handle both tz-aware and naive timestamps
        if hasattr(last_bar, 'tzinfo') and last_bar.tzinfo is not None:
            cutoff = last_bar - timedelta(days=backtest_days)
        else:
            cutoff = pd.Timestamp(last_bar) - timedelta(days=backtest_days)
        pre_trim = len(ind_chart)
        ind_chart = ind_chart[ind_chart.index >= cutoff]
        print(f"✂️  Trimmed chart: {pre_trim} → {len(ind_chart)} bars (backtest window = {backtest_days}d, cutoff = {cutoff})")

    # ── Build per-bar signal series ──
    signals = []
    for i in range(len(ind_chart)):
        row = ind_chart.iloc[i]
        bar_time = ind_chart.index[i]

        st_sig     = str(row.get("Supertrend_Signal", "SELL"))
        most_sig   = str(row.get("MOST_Signal", "SELL"))
        adxr_state = str(row.get("ADXR_State", "FLAT"))

        chart_signal = signal_super_most_adxr(st_sig, most_sig, adxr_state)

        # Filter (HTF) logic — mirrors combine_4h_1d_signals
        filter_signal = "UNKNOWN"
        final_signal  = chart_signal  # default: no filter

        if ind_filter is not None:
            # Find the most recent filter bar at or before this chart bar
            filter_mask = ind_filter.index <= bar_time
            if filter_mask.any():
                f_row = ind_filter.loc[filter_mask].iloc[-1]
                f_st   = str(f_row.get("Supertrend_Signal", "SELL"))
                f_most = str(f_row.get("MOST_Signal", "SELL"))
                f_adxr = str(f_row.get("ADXR_State", "FLAT"))
                filter_signal = signal_super_most_adxr(f_st, f_most, f_adxr)

                # combine_4h_1d_signals logic
                if chart_signal == "EXIT":
                    final_signal = "EXIT"
                elif chart_signal == "STANDDOWN":
                    final_signal = "STANDDOWN"
                elif filter_signal == "UNKNOWN":
                    final_signal = chart_signal
                elif filter_signal in ("BUY", "HOLD"):
                    final_signal = chart_signal
                else:
                    final_signal = "WAIT"

        close = float(row["Close"])
        supertrend_val = float(row["Supertrend"]) if pd.notna(row.get("Supertrend")) else np.nan
        rsi_val = float(row["RSI"]) if pd.notna(row.get("RSI")) else np.nan

        # ── Macro regime modification (if enabled) ──
        bar_regime = "N/A"
        pre_regime_signal = final_signal
        if macro_regime_mode == "ON":
            bar_regime = lookup_regime_for_bar(bar_time, macro_df, override=macro_override)
            final_signal = apply_regime_to_signal(final_signal, bar_regime)

        signals.append({
            "Bar Time":       bar_time,
            "Close":          close,
            "Supertrend":     supertrend_val,
            "ST_Signal":      st_sig,
            "MOST_Signal":    most_sig,
            "ADXR_State":     adxr_state,
            "RSI":            rsi_val,
            "Chart_Signal":   chart_signal,
            "Filter_Signal":  filter_signal,
            "Pre_Regime_Signal": pre_regime_signal,
            "Macro_Regime":   bar_regime,
            "Final_Signal":   final_signal,
        })

    sig_df = pd.DataFrame(signals)

    # ── Buy & Hold baseline: buy at first bar, hold entire period ──
    first_close = sig_df.iloc[0]["Close"]
    bh_qty = initial_capital / first_close   # shares/tokens bought at bar 0

    # ── Simulate trades (Strategy) with 80/20 position sizing ──
    # State: track cash and token_qty separately at all times
    # Start fully in cash (regime = OUT)
    regime = "OUT"
    cash = initial_capital
    token_qty = 0.0
    trades = []
    equity_curve = []
    trade_entry_value = 0.0   # total value when we entered (for PnL calc)

    for _, s in sig_df.iterrows():
        final = s["Final_Signal"]
        close = s["Close"]
        bar_time = s["Bar Time"]

        # Desired regime from signal (same as bot.py)
        desired = "IN" if final in ("BUY", "HOLD") else "OUT"

        # ── Regime transition ──
        if regime == "OUT" and desired == "IN":
            # BUY: move to in_token_pct in token, rest stays cash
            total_value = cash + token_qty * close
            trade_entry_value = total_value
            target_token_value = total_value * in_token_pct
            target_cash = total_value * (1 - in_token_pct)

            buy_qty = (target_token_value - token_qty * close) / close
            if buy_qty > 0:
                token_qty += buy_qty
                cash = target_cash

            trades.append({
                "Bar Time": bar_time,
                "Action":   "BUY",
                "Price":    close,
                "Token Qty": round(token_qty, 6),
                "Cash":     round(cash, 2),
                "Total":    round(cash + token_qty * close, 2),
                "Alloc":    f"{in_token_pct*100:.0f}% token / {(1-in_token_pct)*100:.0f}% cash",
                "Signal":   final,
                "Macro_Regime": s.get("Macro_Regime", "N/A"),
            })
            regime = "IN"

        elif regime == "IN" and desired == "OUT":
            # SELL: move to out_token_pct in token, rest goes to cash
            total_value = cash + token_qty * close
            pnl = total_value - trade_entry_value
            pnl_pct = (pnl / trade_entry_value) * 100 if trade_entry_value > 0 else 0

            target_token_value = total_value * out_token_pct
            target_cash = total_value * (1 - out_token_pct)

            token_qty = target_token_value / close
            cash = target_cash

            trades.append({
                "Bar Time": bar_time,
                "Action":   "SELL",
                "Price":    close,
                "Token Qty": round(token_qty, 6),
                "Cash":     round(cash, 2),
                "Total":    round(total_value, 2),
                "PnL":      round(pnl, 2),
                "PnL%":     round(pnl_pct, 2),
                "Alloc":    f"{out_token_pct*100:.0f}% token / {(1-out_token_pct)*100:.0f}% cash",
                "Signal":   final,
                "Macro_Regime": s.get("Macro_Regime", "N/A"),
            })
            trade_entry_value = 0.0
            regime = "OUT"

        # ── Equity snapshot (strategy + buy & hold side by side) ──
        strat_equity = cash + token_qty * close
        bh_equity = bh_qty * close

        equity_curve.append({
            "Bar Time":     bar_time,
            "Equity":       round(strat_equity, 2),
            "BH_Equity":    round(bh_equity, 2),
            "Regime":       regime,
            "Close":        close,
            "Token Qty":    round(token_qty, 6),
            "Cash":         round(cash, 2),
        })

    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    # ── Final value = cash + token mark-to-market ──
    last_close = sig_df.iloc[-1]["Close"]
    final_value = cash + token_qty * last_close

    # ── Open position PnL (if still IN) ──
    if regime == "IN" and trade_entry_value > 0:
        open_pnl = final_value - trade_entry_value
        open_pnl_pct = (open_pnl / trade_entry_value) * 100
    else:
        open_pnl = 0.0
        open_pnl_pct = 0.0

    # ── Buy & Hold final numbers ──
    bh_final_value = bh_qty * last_close
    bh_pnl         = bh_final_value - initial_capital
    bh_return_pct  = (bh_pnl / initial_capital) * 100

    # ── Strategy return ──
    strategy_return = ((final_value - initial_capital) / initial_capital) * 100
    strategy_pnl    = final_value - initial_capital

    # ── Trade stats ──
    if not trades_df.empty:
        sells = trades_df[trades_df["Action"] == "SELL"]
        n_trades = len(sells)
        wins = sells[sells["PnL"] > 0] if "PnL" in sells.columns else pd.DataFrame()
        losses = sells[sells["PnL"] <= 0] if "PnL" in sells.columns else pd.DataFrame()
        win_rate = (len(wins) / n_trades * 100) if n_trades > 0 else 0
        avg_win = wins["PnL%"].mean() if not wins.empty else 0
        avg_loss = losses["PnL%"].mean() if not losses.empty else 0
        best_trade = sells["PnL%"].max() if not sells.empty and "PnL%" in sells.columns else 0
        worst_trade = sells["PnL%"].min() if not sells.empty and "PnL%" in sells.columns else 0
    else:
        n_trades = 0
        win_rate = avg_win = avg_loss = best_trade = worst_trade = 0

    # ── Max drawdown: strategy ──
    if not equity_df.empty:
        peak = equity_df["Equity"].cummax()
        drawdown = (equity_df["Equity"] - peak) / peak * 100
        max_drawdown = drawdown.min()
    else:
        max_drawdown = 0

    # ── Max drawdown: buy & hold ──
    if not equity_df.empty:
        bh_peak = equity_df["BH_Equity"].cummax()
        bh_drawdown = (equity_df["BH_Equity"] - bh_peak) / bh_peak * 100
        bh_max_drawdown = bh_drawdown.min()
    else:
        bh_max_drawdown = 0

    # ── Time in market ──
    if not equity_df.empty:
        bars_in = len(equity_df[equity_df["Regime"] == "IN"])
        time_in_market = (bars_in / len(equity_df)) * 100
    else:
        time_in_market = 0

    # ── Alpha = strategy outperformance vs buy & hold ──
    alpha = strategy_return - bh_return_pct

    stats = {
        "Ticker":             ticker,
        "Chart":              chart_interval.upper(),
        "Filter":             (filter_interval.upper() if filter_interval else "None"),
        "Position Sizing":    f"BUY={in_token_pct*100:.0f}% token / SELL={out_token_pct*100:.0f}% token",
        "Period":             f"{sig_df.iloc[0]['Bar Time']} → {sig_df.iloc[-1]['Bar Time']}",
        "Bars":               len(sig_df),
        "Initial Capital":    f"${initial_capital:,.2f}",
        # ── Strategy ──
        "Strat Final Value":  f"${final_value:,.2f}",
        "Strat P&L":          f"${strategy_pnl:+,.2f}",
        "Strat Return":       f"{strategy_return:+.2f}%",
        "Strat Max Drawdown": f"{max_drawdown:.2f}%",
        # ── Buy & Hold ──
        "B&H Final Value":    f"${bh_final_value:,.2f}",
        "B&H P&L":            f"${bh_pnl:+,.2f}",
        "B&H Return":         f"{bh_return_pct:+.2f}%",
        "B&H Max Drawdown":   f"{bh_max_drawdown:.2f}%",
        # ── Comparison ──
        "Alpha (Strat - B&H)": f"{alpha:+.2f}%",
        "P&L Advantage":      f"${(strategy_pnl - bh_pnl):+,.2f}",
        # ── Trade stats ──
        "Completed Trades":   n_trades,
        "Win Rate":           f"{win_rate:.1f}%",
        "Avg Win":            f"{avg_win:+.2f}%",
        "Avg Loss":           f"{avg_loss:+.2f}%",
        "Best Trade":         f"{best_trade:+.2f}%",
        "Worst Trade":        f"{worst_trade:+.2f}%",
        "Time in Market":     f"{time_in_market:.1f}%",
        "Open Position":      "YES" if regime == "IN" else "NO",
        "Open PnL":           f"${open_pnl:+,.2f}" if regime == "IN" else "N/A",
        "Open PnL%":          f"{open_pnl_pct:+.2f}%" if regime == "IN" else "N/A",
    }

    return {
        "stats":        stats,
        "trades":       trades_df,
        "equity_curve": equity_df,
        "signals":      sig_df,
    }


# ═══════════════════════════════════════════════════════════════════
# Trailing-Stop-Only Backtest (Stocks)
# ═══════════════════════════════════════════════════════════════════

def run_backtest_trailing_stop(
    df_chart: pd.DataFrame,
    *,
    ticker: str,
    chart_interval: str,
    backtest_days: int | None = None,
    backtest_start_date: str | None = None,
    backtest_end_date: str | None = None,
    atr_period: int   = DEFAULT_ATR_PERIOD,
    atr_mult: float   = DEFAULT_ATR_MULT,
    initial_capital: float = 10000.0,
    macro_regime_mode: str = "OFF",
    macro_df: pd.DataFrame | None = None,
    macro_override: str = "AUTO",
) -> dict:
    """
    Trailing-stop backtest using ONLY Supertrend — no MOST, no ADXR, no HTF filter.

    Designed for stocks where the goal is to beat buy-and-hold by:
      1. Riding the full uptrend (entered at bar 1, fully invested)
      2. Exiting on Supertrend SELL (trailing stop hit)
      3. Re-entering on the VERY NEXT Supertrend BUY (no multi-bar delay)

    With macro kill switch (optional):
      - When macro = BEAR, re-entry is BLOCKED even if Supertrend flips BUY
      - Prevents buying false rallies during real bear markets
      - Re-entry allowed again when macro goes back to NEUTRAL or BULL

    Position sizing: 100% in on BUY, 100% cash on SELL (stocks, not crypto 80/20).

    Returns same dict structure as run_backtest() for UI compatibility.
    """
    # Only need ATR for Supertrend — compute minimal indicators
    # We still use apply_indicators but only read Supertrend_Signal from the output
    indicator_params = dict(
        atr_period=atr_period,
        atr_multiplier=atr_mult,
        rsi_period=14,           # needed by apply_indicators but we ignore the output
        vol_lookback=20,
        adxr_len=14,
        adxr_lenx=14,
        adxr_low_threshold=20.0,
        adxr_flat_eps=1e-6,
    )

    print(f"📊 Applying Supertrend (ATR {atr_period}, mult {atr_mult}) to {len(df_chart)} bars ...")
    ind_chart = apply_indicators(df_chart, **indicator_params)

    # ── Trim to backtest window ──
    if backtest_start_date is not None or backtest_end_date is not None:
        pre_trim = len(ind_chart)
        if backtest_start_date is not None:
            ind_chart = ind_chart[ind_chart.index >= pd.Timestamp(backtest_start_date)]
        if backtest_end_date is not None:
            end_ts = pd.Timestamp(backtest_end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            ind_chart = ind_chart[ind_chart.index <= end_ts]
        print(f"✂️  Trimmed: {pre_trim} → {len(ind_chart)} bars")
    elif backtest_days is not None:
        from datetime import timedelta
        last_bar = ind_chart.index[-1]
        cutoff = pd.Timestamp(last_bar) - timedelta(days=backtest_days)
        pre_trim = len(ind_chart)
        ind_chart = ind_chart[ind_chart.index >= cutoff]
        print(f"✂️  Trimmed: {pre_trim} → {len(ind_chart)} bars ({backtest_days}d window)")

    # ── Build per-bar signals: Supertrend only ──
    signals = []
    for i in range(len(ind_chart)):
        row = ind_chart.iloc[i]
        bar_time = ind_chart.index[i]
        close = float(row["Close"])
        supertrend_val = float(row["Supertrend"]) if pd.notna(row.get("Supertrend")) else np.nan

        st_sig = str(row.get("Supertrend_Signal", "SELL"))

        # Supertrend-only signal: BUY or EXIT, that's it
        raw_signal = "BUY" if st_sig == "BUY" else "EXIT"

        # Macro kill switch: block re-entry in BEAR
        bar_regime = "N/A"
        final_signal = raw_signal
        if macro_regime_mode == "ON":
            bar_regime = lookup_regime_for_bar(bar_time, macro_df, override=macro_override)
            # In BEAR: block BUY (keep EXIT as-is)
            if bar_regime == "BEAR" and raw_signal == "BUY":
                final_signal = "WAIT"

        signals.append({
            "Bar Time":       bar_time,
            "Close":          close,
            "Supertrend":     supertrend_val,
            "ST_Signal":      st_sig,
            "Raw_Signal":     raw_signal,
            "Macro_Regime":   bar_regime,
            "Final_Signal":   final_signal,
        })

    sig_df = pd.DataFrame(signals)

    # ── Buy & Hold baseline ──
    first_close = sig_df.iloc[0]["Close"]
    bh_qty = initial_capital / first_close

    # ── Simulate: 100% in / 100% out (stocks) ──
    # Start INVESTED at bar 0 (we assume you already decided to buy this stock)
    regime = "IN"
    cash = 0.0
    token_qty = initial_capital / first_close
    trades = []
    equity_curve = []
    trade_entry_value = initial_capital

    # Record the initial BUY
    trades.append({
        "Bar Time": sig_df.iloc[0]["Bar Time"],
        "Action": "BUY",
        "Price": first_close,
        "Shares": round(token_qty, 4),
        "Cash": 0.0,
        "Total": round(initial_capital, 2),
        "PnL": 0.0,
        "PnL%": 0.0,
        "Signal": "INITIAL",
        "Macro_Regime": sig_df.iloc[0].get("Macro_Regime", "N/A"),
    })

    for idx in range(1, len(sig_df)):
        s = sig_df.iloc[idx]
        final = s["Final_Signal"]
        close = s["Close"]
        bar_time = s["Bar Time"]

        desired = "IN" if final == "BUY" else "OUT"

        if regime == "IN" and desired == "OUT":
            # SELL — go to 100% cash
            total_value = token_qty * close
            pnl = total_value - trade_entry_value
            pnl_pct = (pnl / trade_entry_value) * 100 if trade_entry_value > 0 else 0

            cash = total_value
            token_qty = 0.0

            trades.append({
                "Bar Time": bar_time,
                "Action": "SELL",
                "Price": close,
                "Shares": 0.0,
                "Cash": round(cash, 2),
                "Total": round(cash, 2),
                "PnL": round(pnl, 2),
                "PnL%": round(pnl_pct, 2),
                "Signal": final,
                "Macro_Regime": s.get("Macro_Regime", "N/A"),
            })
            regime = "OUT"

        elif regime == "OUT" and desired == "IN":
            # BUY — go to 100% invested
            total_value = cash
            trade_entry_value = total_value
            token_qty = total_value / close
            cash = 0.0

            trades.append({
                "Bar Time": bar_time,
                "Action": "BUY",
                "Price": close,
                "Shares": round(token_qty, 4),
                "Cash": 0.0,
                "Total": round(token_qty * close, 2),
                "PnL": 0.0,
                "PnL%": 0.0,
                "Signal": final,
                "Macro_Regime": s.get("Macro_Regime", "N/A"),
            })
            regime = "IN"

        # Equity snapshot
        strat_equity = cash + token_qty * close
        bh_equity = bh_qty * close

        equity_curve.append({
            "Bar Time": bar_time,
            "Equity": round(strat_equity, 2),
            "BH_Equity": round(bh_equity, 2),
            "Regime": regime,
            "Macro_Regime": s.get("Macro_Regime", "N/A"),
            "Close": close,
        })

    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    # ── Final values ──
    last_close = sig_df.iloc[-1]["Close"]
    final_value = cash + token_qty * last_close
    bh_final = bh_qty * last_close

    strategy_pnl = final_value - initial_capital
    strategy_return = (strategy_pnl / initial_capital) * 100
    bh_pnl = bh_final - initial_capital
    bh_return = (bh_pnl / initial_capital) * 100
    alpha = strategy_return - bh_return

    # ── Trade stats ──
    if not trades_df.empty:
        sells = trades_df[(trades_df["Action"] == "SELL") & (trades_df["Signal"] != "INITIAL")]
        n_trades = len(sells)
        wins = sells[sells["PnL"] > 0]
        losses = sells[sells["PnL"] <= 0]
        win_rate = (len(wins) / n_trades * 100) if n_trades > 0 else 0
        avg_win = wins["PnL%"].mean() if not wins.empty else 0
        avg_loss = losses["PnL%"].mean() if not losses.empty else 0
        best_trade = sells["PnL%"].max() if not sells.empty else 0
        worst_trade = sells["PnL%"].min() if not sells.empty else 0
    else:
        n_trades = win_rate = avg_win = avg_loss = best_trade = worst_trade = 0

    # ── Max drawdowns ──
    if not equity_df.empty:
        peak = equity_df["Equity"].cummax()
        dd = (equity_df["Equity"] - peak) / peak * 100
        max_dd = dd.min()
        bh_peak = equity_df["BH_Equity"].cummax()
        bh_dd = (equity_df["BH_Equity"] - bh_peak) / bh_peak * 100
        bh_max_dd = bh_dd.min()
        bars_in = len(equity_df[equity_df["Regime"] == "IN"])
        time_in = (bars_in / len(equity_df)) * 100
    else:
        max_dd = bh_max_dd = time_in = 0

    # ── Open position PnL ──
    if regime == "IN" and trade_entry_value > 0:
        open_pnl = final_value - trade_entry_value
        open_pnl_pct = (open_pnl / trade_entry_value) * 100
    else:
        open_pnl = open_pnl_pct = 0.0

    stats = {
        "Ticker": ticker,
        "Mode": "Trailing Stop (Supertrend only)"
                + (" + Macro Kill Switch" if macro_regime_mode == "ON" else ""),
        "Chart": chart_interval.upper(),
        "Filter": "None (trailing stop mode)",
        "Position Sizing": "100% in / 100% out",
        "Period": f"{sig_df.iloc[0]['Bar Time']} → {sig_df.iloc[-1]['Bar Time']}",
        "Bars": len(sig_df),
        "Initial Capital": f"${initial_capital:,.2f}",
        "Strat Final Value": f"${final_value:,.2f}",
        "Strat P&L": f"${strategy_pnl:+,.2f}",
        "Strat Return": f"{strategy_return:+.2f}%",
        "Strat Max Drawdown": f"{max_dd:.2f}%",
        "B&H Final Value": f"${bh_final:,.2f}",
        "B&H P&L": f"${bh_pnl:+,.2f}",
        "B&H Return": f"{bh_return:+.2f}%",
        "B&H Max Drawdown": f"{bh_max_dd:.2f}%",
        "Alpha (Strat - B&H)": f"{alpha:+.2f}%",
        "P&L Advantage": f"${(strategy_pnl - bh_pnl):+,.2f}",
        "Completed Trades": n_trades,
        "Win Rate": f"{win_rate:.1f}%",
        "Avg Win": f"{avg_win:+.2f}%",
        "Avg Loss": f"{avg_loss:+.2f}%",
        "Best Trade": f"{best_trade:+.2f}%",
        "Worst Trade": f"{worst_trade:+.2f}%",
        "Time in Market": f"{time_in:.1f}%",
        "Open Position": "YES" if regime == "IN" else "NO",
        "Open PnL": f"${open_pnl:+,.2f}" if regime == "IN" else "N/A",
        "Open PnL%": f"{open_pnl_pct:+.2f}%" if regime == "IN" else "N/A",
    }

    return {
        "stats": stats,
        "trades": trades_df,
        "equity_curve": equity_df,
        "signals": sig_df,
    }


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Backtest Supertrend + MOST RSI + ADXR strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 backtest.py AAPL -c 1d -t 2y
  python3 backtest.py SOL-USD -c 4h -f 1d -t 1y
  python3 backtest.py ETH-USD -c 4h -f 1d -t 2y
  python3 backtest.py LQL -c 4h -t 6m --gecko "https://www.geckoterminal.com/solana/pools/GiRyo..."
  python3 backtest.py AAPL -c 1d -t 3y --capital 50000
        """,
    )

    p.add_argument("symbol", help="Ticker symbol (e.g. AAPL, SOL-USD, BTC-USD, LQL)")
    p.add_argument("-m", "--mode", default="full",
                   choices=["full", "trailing-stop"],
                   help="Backtest mode: 'full' = Supertrend+MOST+ADXR (default), "
                        "'trailing-stop' = Supertrend only, 100%% in/out, instant re-entry")
    p.add_argument("-c", "--chart", default="1d",
                   choices=["1h", "2h", "4h", "1d", "1wk"],
                   help="Chart timeframe (default: 1d)")
    p.add_argument("-f", "--filter", default=None,
                   choices=["1h", "2h", "4h", "1d", "1wk"],
                   help="Higher-timeframe filter (e.g. -c 4h -f 1d). Optional.")
    p.add_argument("-t", "--timeframe", default="2y",
                   help="Lookback period: 1m,2m,3m,6m,1y,2y,3y,5y,10y,max or Nd for N days (default: 2y)")

    # Data source
    p.add_argument("--gecko", default=None,
                   help="GeckoTerminal pool URL (for tokens not on Yahoo)")

    # Capital & position sizing
    p.add_argument("--capital", type=float, default=10000.0,
                   help="Initial capital (default: 10000)")
    p.add_argument("--in-pct", type=float, default=None,
                   help="Token allocation %% on BUY (default: auto — crypto=0.80, stock=1.00)")
    p.add_argument("--out-pct", type=float, default=None,
                   help="Token allocation %% on SELL (default: auto — crypto=0.20, stock=0.00)")

    # Indicator params (override defaults)
    p.add_argument("--atr-period",  type=int,   default=DEFAULT_ATR_PERIOD)
    p.add_argument("--atr-mult",    type=float, default=DEFAULT_ATR_MULT)
    p.add_argument("--rsi-period",  type=int,   default=DEFAULT_RSI_PERIOD)
    p.add_argument("--vol-lookback", type=int,  default=DEFAULT_VOL_LOOKBACK)
    p.add_argument("--adxr-len",    type=int,   default=DEFAULT_ADXR_LEN)
    p.add_argument("--adxr-lenx",   type=int,   default=DEFAULT_ADXR_LENX)
    p.add_argument("--adxr-low",    type=float, default=DEFAULT_ADXR_LOW)
    p.add_argument("--adxr-eps",    type=float, default=DEFAULT_ADXR_EPS)

    # Output
    p.add_argument("--csv", default=None,
                   help="Save trade log + equity curve CSVs to this directory")
    p.add_argument("--no-signals", action="store_true",
                   help="Don't print the full signals table (just summary + trades)")

    # Macro regime
    p.add_argument("--regime", action="store_true",
                   help="Enable regime-aware backtesting (fetches historical VIX+SPY, adjusts signals)")
    p.add_argument("--regime-override", default="AUTO",
                   choices=["AUTO", "BULL", "NEUTRAL", "BEAR"],
                   help="Force a macro regime for all bars (default: AUTO = use historical data)")

    return p.parse_args()


def parse_timeframe(tf_str: str) -> int:
    """Convert timeframe string to days."""
    tf_str = tf_str.lower().strip()

    # Direct map lookup
    if tf_str in TIMEFRAME_MAP:
        return TIMEFRAME_MAP[tf_str]

    # Nd format (e.g. "500d")
    if tf_str.endswith("d") and tf_str[:-1].isdigit():
        return int(tf_str[:-1])

    # Ny format (e.g. "2y" already handled above, but just in case)
    if tf_str.endswith("y") and tf_str[:-1].isdigit():
        return int(tf_str[:-1]) * 365

    raise ValueError(f"Unknown timeframe: '{tf_str}'. Use 1m,2m,3m,6m,1y,2y,3y,5y,10y,max or Nd (e.g. 500d)")


def main():
    args = parse_args()

    lookback_days = parse_timeframe(args.timeframe)

    # Auto-detect crypto vs stock for position sizing defaults
    is_crypto = (
        "-USD" in args.symbol.upper()
        or "-EUR" in args.symbol.upper()
        or "-BTC" in args.symbol.upper()
        or args.gecko is not None
    )
    if args.in_pct is None:
        args.in_pct = 0.80 if is_crypto else 1.00
    if args.out_pct is None:
        args.out_pct = 0.20 if is_crypto else 0.00

    asset_label = "Crypto" if is_crypto else "Stock"

    print("=" * 70)
    print(f"  BACKTEST: {args.symbol} ({asset_label})")
    print(f"  Chart: {args.chart.upper()}"
          + (f" | Filter: {args.filter.upper()}" if args.filter else "")
          + f" | Lookback: {args.timeframe} ({lookback_days}d)")
    print(f"  Capital: ${args.capital:,.2f}  |  BUY={args.in_pct*100:.0f}% token  |  SELL={args.out_pct*100:.0f}% token")
    if args.gecko:
        print(f"  Data source: GeckoTerminal")
    print(f"  Params: ATR({args.atr_period}, {args.atr_mult}) RSI({args.rsi_period})"
          f" ADXR({args.adxr_len}, {args.adxr_lenx}, low={args.adxr_low})")
    if args.regime:
        print(f"  Macro Regime: ON (override={args.regime_override})")
    print("=" * 70)

    # ── Fetch macro regime data (if enabled) ──
    macro_df_hist = None
    if args.regime:
        print("\n📊 Fetching historical VIX + SPY for macro regime...")
        try:
            macro_df_hist = fetch_historical_macro_data(lookback_days + 500)
            print(f"✅ Macro data: {len(macro_df_hist)} daily bars")
            # Show regime distribution
            if "Macro_Regime" in macro_df_hist.columns:
                counts = macro_df_hist["Macro_Regime"].value_counts()
                for r, c in counts.items():
                    print(f"   {r}: {c} days ({c/len(macro_df_hist)*100:.0f}%)")
        except Exception as e:
            print(f"⚠️  Macro data fetch failed: {e} — proceeding without regime filter")
            args.regime = False

    # ── Fetch chart data ──
    try:
        if args.gecko:
            df_chart = fetch_gecko_ohlcv(args.gecko, args.chart, lookback_days)
        else:
            df_chart = fetch_yahoo_ohlcv(args.symbol, args.chart, lookback_days)
    except Exception as e:
        print(f"\n❌ Failed to fetch chart data: {e}")
        sys.exit(1)

    # ── Fetch filter data (if specified) ──
    df_filter = None
    if args.filter:
        try:
            if args.gecko:
                # For gecko tokens, filter is also from gecko (same pool, different interval)
                df_filter = fetch_gecko_ohlcv(args.gecko, args.filter, lookback_days)
            else:
                df_filter = fetch_yahoo_ohlcv(args.symbol, args.filter, lookback_days)
        except Exception as e:
            print(f"\n⚠️  Failed to fetch filter data: {e} — proceeding without filter")
            df_filter = None
            args.filter = None

    # ── Run backtest ──
    if args.mode == "trailing-stop":
        print("\n🎯 MODE: Trailing Stop (Supertrend only, instant re-entry)")
        result = run_backtest_trailing_stop(
            df_chart,
            ticker=args.symbol,
            chart_interval=args.chart,
            backtest_days=lookback_days,
            atr_period=args.atr_period,
            atr_mult=args.atr_mult,
            initial_capital=args.capital,
            macro_regime_mode="ON" if args.regime else "OFF",
            macro_df=macro_df_hist,
            macro_override=args.regime_override,
        )
    else:
        result = run_backtest(
            df_chart,
            df_filter,
            ticker=args.symbol,
            chart_interval=args.chart,
            filter_interval=args.filter,
            backtest_days=lookback_days,
            in_token_pct=args.in_pct,
            out_token_pct=args.out_pct,
            atr_period=args.atr_period,
            atr_mult=args.atr_mult,
            rsi_period=args.rsi_period,
            vol_lookback=args.vol_lookback,
            adxr_len=args.adxr_len,
            adxr_lenx=args.adxr_lenx,
            adxr_low=args.adxr_low,
            adxr_eps=args.adxr_eps,
            initial_capital=args.capital,
            macro_regime_mode="ON" if args.regime else "OFF",
            macro_df=macro_df_hist,
            macro_override=args.regime_override,
        )

    # ── Print results ──
    stats = result["stats"]
    trades_df = result["trades"]
    sig_df = result["signals"]

    print("\n" + "=" * 70)
    print("  BACKTEST RESULTS")
    print("=" * 70)

    # ── Header info ──
    for k in ("Ticker", "Chart", "Filter", "Period", "Bars", "Initial Capital"):
        print(f"  {k:<22s}: {stats[k]}")

    # ── Side-by-side comparison ──
    print("\n  " + "-" * 50)
    print(f"  {'':22s}  {'STRATEGY':>14s}  {'BUY & HOLD':>14s}")
    print("  " + "-" * 50)
    print(f"  {'Final Value':<22s}  {stats['Strat Final Value']:>14s}  {stats['B&H Final Value']:>14s}")
    print(f"  {'P&L':<22s}  {stats['Strat P&L']:>14s}  {stats['B&H P&L']:>14s}")
    print(f"  {'Return':<22s}  {stats['Strat Return']:>14s}  {stats['B&H Return']:>14s}")
    print(f"  {'Max Drawdown':<22s}  {stats['Strat Max Drawdown']:>14s}  {stats['B&H Max Drawdown']:>14s}")
    print("  " + "-" * 50)
    print(f"  {'Alpha (Strat - B&H)':<22s}: {stats['Alpha (Strat - B&H)']}")
    print(f"  {'P&L Advantage':<22s}: {stats['P&L Advantage']}")

    # ── Trade stats ──
    print(f"\n  {'Completed Trades':<22s}: {stats['Completed Trades']}")
    print(f"  {'Win Rate':<22s}: {stats['Win Rate']}")
    print(f"  {'Avg Win':<22s}: {stats['Avg Win']}")
    print(f"  {'Avg Loss':<22s}: {stats['Avg Loss']}")
    print(f"  {'Best Trade':<22s}: {stats['Best Trade']}")
    print(f"  {'Worst Trade':<22s}: {stats['Worst Trade']}")
    print(f"  {'Time in Market':<22s}: {stats['Time in Market']}")
    if stats["Open Position"] == "YES":
        print(f"  {'Open Position':<22s}: YES  (PnL: {stats['Open PnL']}  {stats['Open PnL%']})")
    print("=" * 70)

    # ── Print trades ──
    if not trades_df.empty:
        print(f"\n📋 Trade Log ({len(trades_df)} entries):")
        print("-" * 100)
        # Format for display
        display_cols = ["Bar Time", "Action", "Price", "Token Qty", "Cash", "Total", "Alloc"]
        if "PnL" in trades_df.columns:
            display_cols += ["PnL", "PnL%"]
        display_cols.append("Signal")
        available = [c for c in display_cols if c in trades_df.columns]
        print(trades_df[available].to_string(index=False))
    else:
        print("\n📋 No trades executed in this period.")

    # ── Print signal summary ──
    if not args.no_signals:
        print(f"\n📊 Signal Distribution:")
        if "Final_Signal" in sig_df.columns:
            counts = sig_df["Final_Signal"].value_counts()
            for sig, cnt in counts.items():
                pct = cnt / len(sig_df) * 100
                print(f"  {sig:<12s}: {cnt:>5d} bars ({pct:5.1f}%)")

    # ── Save CSVs ──
    if args.csv:
        os.makedirs(args.csv, exist_ok=True)
        base = f"{args.symbol}_{args.chart}"
        if args.filter:
            base += f"_f{args.filter}"

        # Trades
        if not trades_df.empty:
            trades_path = os.path.join(args.csv, f"bt_trades_{base}.csv")
            trades_df.to_csv(trades_path, index=False)
            print(f"\n💾 Trades saved: {trades_path}")

        # Equity curve
        eq_df = result["equity_curve"]
        if not eq_df.empty:
            eq_path = os.path.join(args.csv, f"bt_equity_{base}.csv")
            eq_df.to_csv(eq_path, index=False)
            print(f"💾 Equity curve saved: {eq_path}")

        # Full signals
        sig_path = os.path.join(args.csv, f"bt_signals_{base}.csv")
        sig_df.to_csv(sig_path, index=False)
        print(f"💾 Signals saved: {sig_path}")

    print("\n✅ Backtest complete.")


if __name__ == "__main__":
    main()