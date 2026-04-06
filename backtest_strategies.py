# backtest_strategies.py
"""
Stock backtest strategies — each function takes OHLCV data and returns
the same dict format as run_backtest() for UI compatibility.

Strategies:
  1. ma_crossover        — 50/200 MA golden/death cross
  2. price_vs_200ma      — Above 200MA = hold, below = sell
  3. chandelier_exit     — ATR trailing stop from highest high
  4. donchian_breakout   — Buy on N-day high, sell on M-day low
  5. mean_reversion_rsi  — Buy RSI<30, sell RSI>70
  6. support_resistance  — Auto-detect levels, buy breakout, stop at support

All share the same interface:
  run_backtest_<strategy>(df_chart, *, ticker, chart_interval, backtest_days, ...)
  → dict with keys: stats, trades, equity_curve, signals
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import timedelta


# ═══════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════

def _trim_df(df: pd.DataFrame, backtest_days=None, start_date=None, end_date=None) -> pd.DataFrame:
    """Trim DataFrame to backtest window."""
    if start_date is not None or end_date is not None:
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            df = df[df.index <= end_ts]
    elif backtest_days is not None:
        last = df.index[-1]
        cutoff = pd.Timestamp(last) - timedelta(days=backtest_days)
        df = df[df.index >= cutoff]
    return df


def _compute_stats(
    sig_df, equity_df, trades_df, trades,
    initial_capital, regime_col="Regime",
    ticker="", chart_interval="1d", mode_name="",
) -> dict:
    """Compute standardized stats dict from backtest results."""
    last_close = sig_df.iloc[-1]["Close"]

    # Final values from equity curve
    final_value = equity_df.iloc[-1]["Equity"] if not equity_df.empty else initial_capital
    bh_final = equity_df.iloc[-1]["BH_Equity"] if not equity_df.empty else initial_capital

    strategy_pnl = final_value - initial_capital
    strategy_return = (strategy_pnl / initial_capital) * 100
    bh_pnl = bh_final - initial_capital
    bh_return = (bh_pnl / initial_capital) * 100
    alpha = strategy_return - bh_return

    # Trade stats
    if not trades_df.empty:
        sells = trades_df[(trades_df["Action"] == "SELL") & (trades_df.get("Signal", pd.Series(dtype=str)) != "INITIAL")]
        if sells.empty:
            sells = trades_df[trades_df["Action"] == "SELL"]
        n_trades = len(sells)
        if "PnL" in sells.columns:
            wins = sells[sells["PnL"] > 0]
            losses = sells[sells["PnL"] <= 0]
            win_rate = (len(wins) / n_trades * 100) if n_trades > 0 else 0
            avg_win = wins["PnL%"].mean() if not wins.empty else 0
            avg_loss = losses["PnL%"].mean() if not losses.empty else 0
            best_trade = sells["PnL%"].max() if not sells.empty else 0
            worst_trade = sells["PnL%"].min() if not sells.empty else 0
        else:
            n_trades = win_rate = avg_win = avg_loss = best_trade = worst_trade = 0
    else:
        n_trades = win_rate = avg_win = avg_loss = best_trade = worst_trade = 0

    # Drawdowns
    if not equity_df.empty:
        peak = equity_df["Equity"].cummax()
        dd = (equity_df["Equity"] - peak) / peak * 100
        max_dd = dd.min()
        bh_peak = equity_df["BH_Equity"].cummax()
        bh_dd = (equity_df["BH_Equity"] - bh_peak) / bh_peak * 100
        bh_max_dd = bh_dd.min()
        if regime_col in equity_df.columns:
            bars_in = len(equity_df[equity_df[regime_col] == "IN"])
        else:
            bars_in = len(equity_df)
        time_in = (bars_in / len(equity_df)) * 100
    else:
        max_dd = bh_max_dd = time_in = 0

    return {
        "Ticker": ticker,
        "Mode": mode_name,
        "Chart": chart_interval.upper(),
        "Filter": "None",
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
        "Open Position": "NO",
        "Open PnL": "N/A",
        "Open PnL%": "N/A",
    }


def _simulate_trades(sig_df, initial_capital, signal_col="Final_Signal"):
    """
    Generic trade simulator: 100% in on BUY, 100% out on SELL.
    Returns trades list, equity_curve list, final cash, final token_qty, regime.
    """
    first_close = sig_df.iloc[0]["Close"]
    bh_qty = initial_capital / first_close

    regime = "OUT"
    cash = initial_capital
    token_qty = 0.0
    trades = []
    equity_curve = []
    trade_entry_value = 0.0

    for idx in range(len(sig_df)):
        s = sig_df.iloc[idx]
        final = s[signal_col]
        close = s["Close"]
        bar_time = s["Bar Time"]

        desired = "IN" if final == "BUY" else "OUT"

        if regime == "OUT" and desired == "IN":
            total_value = cash
            trade_entry_value = total_value
            token_qty = total_value / close
            cash = 0.0
            trades.append({
                "Bar Time": bar_time, "Action": "BUY", "Price": close,
                "Shares": round(token_qty, 4), "Cash": 0.0,
                "Total": round(token_qty * close, 2),
                "PnL": 0.0, "PnL%": 0.0,
                "Signal": final,
            })
            regime = "IN"

        elif regime == "IN" and desired == "OUT":
            total_value = token_qty * close
            pnl = total_value - trade_entry_value
            pnl_pct = (pnl / trade_entry_value) * 100 if trade_entry_value > 0 else 0
            cash = total_value
            token_qty = 0.0
            trades.append({
                "Bar Time": bar_time, "Action": "SELL", "Price": close,
                "Shares": 0.0, "Cash": round(cash, 2),
                "Total": round(cash, 2),
                "PnL": round(pnl, 2), "PnL%": round(pnl_pct, 2),
                "Signal": final,
            })
            trade_entry_value = 0.0
            regime = "OUT"

        strat_equity = cash + token_qty * close
        bh_equity = bh_qty * close
        equity_curve.append({
            "Bar Time": bar_time,
            "Equity": round(strat_equity, 2),
            "BH_Equity": round(bh_equity, 2),
            "Regime": regime,
            "Close": close,
        })

    return trades, equity_curve, cash, token_qty, regime


# ═══════════════════════════════════════════════════════════════════
# 1. Moving Average Crossover (50/200)
# ═══════════════════════════════════════════════════════════════════

def run_backtest_ma_crossover(
    df_chart: pd.DataFrame,
    *,
    ticker: str,
    chart_interval: str = "1d",
    backtest_days: int | None = None,
    backtest_start_date: str | None = None,
    backtest_end_date: str | None = None,
    fast_period: int = 50,
    slow_period: int = 200,
    initial_capital: float = 10000.0,
) -> dict:
    """
    Golden cross (fast > slow) = BUY, death cross (fast < slow) = SELL.
    """
    df = df_chart.copy()
    df["MA_Fast"] = df["Close"].rolling(fast_period, min_periods=fast_period).mean()
    df["MA_Slow"] = df["Close"].rolling(slow_period, min_periods=slow_period).mean()
    df = df.dropna(subset=["MA_Fast", "MA_Slow"])
    df = _trim_df(df, backtest_days, backtest_start_date, backtest_end_date)

    signals = []
    for i in range(len(df)):
        row = df.iloc[i]
        signal = "BUY" if row["MA_Fast"] > row["MA_Slow"] else "EXIT"
        signals.append({
            "Bar Time": df.index[i],
            "Close": float(row["Close"]),
            "MA_Fast": round(float(row["MA_Fast"]), 2),
            "MA_Slow": round(float(row["MA_Slow"]), 2),
            "Final_Signal": signal,
        })

    sig_df = pd.DataFrame(signals)
    trades, eq_curve, cash, token_qty, regime = _simulate_trades(sig_df, initial_capital)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_df = pd.DataFrame(eq_curve)

    stats = _compute_stats(sig_df, equity_df, trades_df, trades, initial_capital,
                           ticker=ticker, chart_interval=chart_interval,
                           mode_name=f"MA Crossover ({fast_period}/{slow_period})")
    return {"stats": stats, "trades": trades_df, "equity_curve": equity_df, "signals": sig_df}


# ═══════════════════════════════════════════════════════════════════
# 2. Price vs 200-day MA
# ═══════════════════════════════════════════════════════════════════

def run_backtest_price_vs_200ma(
    df_chart: pd.DataFrame,
    *,
    ticker: str,
    chart_interval: str = "1d",
    backtest_days: int | None = None,
    backtest_start_date: str | None = None,
    backtest_end_date: str | None = None,
    ma_period: int = 200,
    initial_capital: float = 10000.0,
) -> dict:
    """
    Price above 200MA = hold. Price below 200MA = sell. Simplest possible trend filter.
    """
    df = df_chart.copy()
    df["MA200"] = df["Close"].rolling(ma_period, min_periods=ma_period).mean()
    df = df.dropna(subset=["MA200"])
    df = _trim_df(df, backtest_days, backtest_start_date, backtest_end_date)

    signals = []
    for i in range(len(df)):
        row = df.iloc[i]
        signal = "BUY" if float(row["Close"]) > float(row["MA200"]) else "EXIT"
        signals.append({
            "Bar Time": df.index[i],
            "Close": float(row["Close"]),
            "MA200": round(float(row["MA200"]), 2),
            "Final_Signal": signal,
        })

    sig_df = pd.DataFrame(signals)
    trades, eq_curve, cash, token_qty, regime = _simulate_trades(sig_df, initial_capital)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_df = pd.DataFrame(eq_curve)

    stats = _compute_stats(sig_df, equity_df, trades_df, trades, initial_capital,
                           ticker=ticker, chart_interval=chart_interval,
                           mode_name=f"Price vs {ma_period}MA")
    return {"stats": stats, "trades": trades_df, "equity_curve": equity_df, "signals": sig_df}


# ═══════════════════════════════════════════════════════════════════
# 3. Chandelier Exit
# ═══════════════════════════════════════════════════════════════════

def run_backtest_chandelier(
    df_chart: pd.DataFrame,
    *,
    ticker: str,
    chart_interval: str = "1d",
    backtest_days: int | None = None,
    backtest_start_date: str | None = None,
    backtest_end_date: str | None = None,
    atr_period: int = 22,
    atr_mult: float = 3.0,
    initial_capital: float = 10000.0,
) -> dict:
    """
    Chandelier Exit: trailing stop at highest_high - ATR * multiplier.
    Only moves up, never down. Exit when close drops below the chandelier line.
    Re-enter when close moves back above.
    """
    df = df_chart.copy()

    # ATR
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(atr_period, min_periods=atr_period).mean()

    # Chandelier: highest high over ATR period, minus ATR * mult
    df["Highest_High"] = high.rolling(atr_period, min_periods=atr_period).max()
    df["Chandelier_Stop"] = df["Highest_High"] - df["ATR"] * atr_mult

    # Make chandelier only move up (ratchet)
    chandelier_ratchet = []
    prev_stop = 0.0
    for i in range(len(df)):
        raw = df["Chandelier_Stop"].iloc[i]
        if pd.isna(raw):
            chandelier_ratchet.append(np.nan)
            continue
        new_stop = max(float(raw), prev_stop) if float(df["Close"].iloc[i]) > prev_stop else float(raw)
        chandelier_ratchet.append(new_stop)
        prev_stop = new_stop
    df["Chandelier"] = chandelier_ratchet

    df = df.dropna(subset=["Chandelier"])
    df = _trim_df(df, backtest_days, backtest_start_date, backtest_end_date)

    signals = []
    for i in range(len(df)):
        row = df.iloc[i]
        signal = "BUY" if float(row["Close"]) > float(row["Chandelier"]) else "EXIT"
        signals.append({
            "Bar Time": df.index[i],
            "Close": float(row["Close"]),
            "Chandelier": round(float(row["Chandelier"]), 2),
            "ATR": round(float(row["ATR"]), 2),
            "Final_Signal": signal,
        })

    sig_df = pd.DataFrame(signals)
    trades, eq_curve, cash, token_qty, regime = _simulate_trades(sig_df, initial_capital)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_df = pd.DataFrame(eq_curve)

    stats = _compute_stats(sig_df, equity_df, trades_df, trades, initial_capital,
                           ticker=ticker, chart_interval=chart_interval,
                           mode_name=f"Chandelier Exit (ATR {atr_period}, {atr_mult}x)")
    return {"stats": stats, "trades": trades_df, "equity_curve": equity_df, "signals": sig_df}


# ═══════════════════════════════════════════════════════════════════
# 4. Donchian Channel Breakout (Turtle Traders)
# ═══════════════════════════════════════════════════════════════════

def run_backtest_donchian(
    df_chart: pd.DataFrame,
    *,
    ticker: str,
    chart_interval: str = "1d",
    backtest_days: int | None = None,
    backtest_start_date: str | None = None,
    backtest_end_date: str | None = None,
    entry_period: int = 20,
    exit_period: int = 10,
    initial_capital: float = 10000.0,
) -> dict:
    """
    Donchian breakout: BUY on new entry_period-day high, SELL on new exit_period-day low.
    Asymmetric channels — wider entry, tighter exit — lets winners run.
    """
    df = df_chart.copy()
    df["Entry_High"] = df["High"].rolling(entry_period, min_periods=entry_period).max().shift(1)
    df["Exit_Low"] = df["Low"].rolling(exit_period, min_periods=exit_period).min().shift(1)
    df = df.dropna(subset=["Entry_High", "Exit_Low"])
    df = _trim_df(df, backtest_days, backtest_start_date, backtest_end_date)

    # Donchian needs stateful logic — you enter on breakout, exit on breakdown
    signals = []
    in_trade = False
    for i in range(len(df)):
        row = df.iloc[i]
        close = float(row["Close"])
        entry_level = float(row["Entry_High"])
        exit_level = float(row["Exit_Low"])

        if not in_trade and close > entry_level:
            signal = "BUY"
            in_trade = True
        elif in_trade and close < exit_level:
            signal = "EXIT"
            in_trade = False
        elif in_trade:
            signal = "BUY"   # stay in
        else:
            signal = "EXIT"  # stay out

        signals.append({
            "Bar Time": df.index[i],
            "Close": close,
            "Entry_High": round(entry_level, 2),
            "Exit_Low": round(exit_level, 2),
            "Final_Signal": signal,
        })

    sig_df = pd.DataFrame(signals)
    trades, eq_curve, cash, token_qty, regime = _simulate_trades(sig_df, initial_capital)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_df = pd.DataFrame(eq_curve)

    stats = _compute_stats(sig_df, equity_df, trades_df, trades, initial_capital,
                           ticker=ticker, chart_interval=chart_interval,
                           mode_name=f"Donchian Breakout ({entry_period}/{exit_period})")
    return {"stats": stats, "trades": trades_df, "equity_curve": equity_df, "signals": sig_df}


# ═══════════════════════════════════════════════════════════════════
# 5. Mean Reversion RSI
# ═══════════════════════════════════════════════════════════════════

def run_backtest_mean_reversion_rsi(
    df_chart: pd.DataFrame,
    *,
    ticker: str,
    chart_interval: str = "1d",
    backtest_days: int | None = None,
    backtest_start_date: str | None = None,
    backtest_end_date: str | None = None,
    rsi_period: int = 14,
    buy_threshold: float = 30.0,
    sell_threshold: float = 70.0,
    initial_capital: float = 10000.0,
) -> dict:
    """
    Mean reversion: BUY when RSI drops below buy_threshold (oversold),
    SELL when RSI rises above sell_threshold (overbought).
    Best for large-cap stocks that revert to the mean.
    """
    df = df_chart.copy()

    # RSI calculation
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(rsi_period, min_periods=rsi_period).mean()
    # Use Wilder's smoothing after initial SMA
    for i in range(rsi_period, len(df)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (rsi_period - 1) + gain.iloc[i]) / rsi_period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (rsi_period - 1) + loss.iloc[i]) / rsi_period
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    df = df.dropna(subset=["RSI"])
    df = _trim_df(df, backtest_days, backtest_start_date, backtest_end_date)

    # Stateful: buy on oversold, sell on overbought
    signals = []
    in_trade = False
    for i in range(len(df)):
        row = df.iloc[i]
        rsi = float(row["RSI"])

        if not in_trade and rsi < buy_threshold:
            signal = "BUY"
            in_trade = True
        elif in_trade and rsi > sell_threshold:
            signal = "EXIT"
            in_trade = False
        elif in_trade:
            signal = "BUY"
        else:
            signal = "EXIT"

        signals.append({
            "Bar Time": df.index[i],
            "Close": float(row["Close"]),
            "RSI": round(rsi, 2),
            "Final_Signal": signal,
        })

    sig_df = pd.DataFrame(signals)
    trades, eq_curve, cash, token_qty, regime = _simulate_trades(sig_df, initial_capital)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_df = pd.DataFrame(eq_curve)

    stats = _compute_stats(sig_df, equity_df, trades_df, trades, initial_capital,
                           ticker=ticker, chart_interval=chart_interval,
                           mode_name=f"Mean Reversion RSI ({rsi_period}, {buy_threshold}/{sell_threshold})")
    return {"stats": stats, "trades": trades_df, "equity_curve": equity_df, "signals": sig_df}


# ═══════════════════════════════════════════════════════════════════
# 6. Support/Resistance Breakout (Auto-detect)
# ═══════════════════════════════════════════════════════════════════

def _detect_sr_levels(df: pd.DataFrame, lookback: int = 20, tolerance_pct: float = 1.5) -> tuple[float, float]:
    """
    Auto-detect nearest support and resistance from recent swing highs/lows.
    Returns (support, resistance) price levels.
    """
    recent = df.tail(lookback)
    if len(recent) < lookback:
        return (np.nan, np.nan)

    highs = recent["High"]
    lows = recent["Low"]
    current = float(df["Close"].iloc[-1])

    # Find swing highs (local maxima)
    resistance_candidates = []
    for i in range(2, len(recent) - 2):
        if (highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and
            highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]):
            resistance_candidates.append(float(highs.iloc[i]))

    # Find swing lows (local minima)
    support_candidates = []
    for i in range(2, len(recent) - 2):
        if (lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i-2] and
            lows.iloc[i] < lows.iloc[i+1] and lows.iloc[i] < lows.iloc[i+2]):
            support_candidates.append(float(lows.iloc[i]))

    # Nearest resistance above current price
    res_above = [r for r in resistance_candidates if r > current]
    resistance = min(res_above) if res_above else current * 1.05  # default 5% above

    # Nearest support below current price
    sup_below = [s for s in support_candidates if s < current]
    support = max(sup_below) if sup_below else current * 0.95  # default 5% below

    return (support, resistance)


def run_backtest_support_resistance(
    df_chart: pd.DataFrame,
    *,
    ticker: str,
    chart_interval: str = "1d",
    backtest_days: int | None = None,
    backtest_start_date: str | None = None,
    backtest_end_date: str | None = None,
    sr_lookback: int = 20,
    sr_tolerance_pct: float = 1.0,
    initial_capital: float = 10000.0,
) -> dict:
    """
    Support/Resistance breakout:
    - BUY when price breaks above resistance (new breakout)
    - SELL when price breaks below support (breakdown / stop hit)
    - S/R levels recalculated on each bar from recent swing highs/lows
    """
    df = df_chart.copy()

    # Need enough history for swing detection
    if len(df) < sr_lookback + 10:
        raise RuntimeError(f"Need at least {sr_lookback + 10} bars, got {len(df)}")

    # Compute S/R at each bar using rolling window
    supports = []
    resistances = []
    for i in range(len(df)):
        if i < sr_lookback + 4:
            supports.append(np.nan)
            resistances.append(np.nan)
            continue
        window = df.iloc[i - sr_lookback:i + 1]
        sup, res = _detect_sr_levels(window, sr_lookback, sr_tolerance_pct)
        supports.append(sup)
        resistances.append(res)

    df["Support"] = supports
    df["Resistance"] = resistances
    df = df.dropna(subset=["Support", "Resistance"])
    df = _trim_df(df, backtest_days, backtest_start_date, backtest_end_date)

    # Stateful breakout logic
    signals = []
    in_trade = False
    for i in range(len(df)):
        row = df.iloc[i]
        close = float(row["Close"])
        support = float(row["Support"])
        resistance = float(row["Resistance"])

        tol = close * sr_tolerance_pct / 100

        if not in_trade and close > resistance + tol:
            signal = "BUY"
            in_trade = True
        elif in_trade and close < support - tol:
            signal = "EXIT"
            in_trade = False
        elif in_trade:
            signal = "BUY"
        else:
            signal = "EXIT"

        signals.append({
            "Bar Time": df.index[i],
            "Close": close,
            "Support": round(support, 2),
            "Resistance": round(resistance, 2),
            "Final_Signal": signal,
        })

    sig_df = pd.DataFrame(signals)
    trades, eq_curve, cash, token_qty, regime = _simulate_trades(sig_df, initial_capital)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_df = pd.DataFrame(eq_curve)

    stats = _compute_stats(sig_df, equity_df, trades_df, trades, initial_capital,
                           ticker=ticker, chart_interval=chart_interval,
                           mode_name=f"Support/Resistance Breakout (lookback={sr_lookback})")
    return {"stats": stats, "trades": trades_df, "equity_curve": equity_df, "signals": sig_df}


# ═══════════════════════════════════════════════════════════════════
# 7. Supertrend Only (pure, no MOST, no ADXR, no macro)
# ═══════════════════════════════════════════════════════════════════

def run_backtest_supertrend_only(
    df_chart: pd.DataFrame,
    *,
    ticker: str,
    chart_interval: str = "1d",
    backtest_days: int | None = None,
    backtest_start_date: str | None = None,
    backtest_end_date: str | None = None,
    atr_period: int = 10,
    atr_multiplier: float = 3.0,
    initial_capital: float = 10000.0,
) -> dict:
    """
    Pure Supertrend — BUY when Supertrend flips green, EXIT when it flips red.
    No MOST, no ADXR, no macro, no HTF filter. Just the raw indicator.
    Starts OUT and waits for first Supertrend BUY (unlike trailing-stop which enters at bar 1).
    """
    from core.indicators import apply_indicators

    indicator_params = dict(
        atr_period=atr_period,
        atr_multiplier=atr_multiplier,
        rsi_period=14,
        vol_lookback=20,
        adxr_len=14,
        adxr_lenx=14,
        adxr_low_threshold=20.0,
        adxr_flat_eps=1e-6,
    )

    ind = apply_indicators(df_chart, **indicator_params)
    ind = _trim_df(ind, backtest_days, backtest_start_date, backtest_end_date)

    signals = []
    for i in range(len(ind)):
        row = ind.iloc[i]
        st_sig = str(row.get("Supertrend_Signal", "SELL"))
        signal = "BUY" if st_sig == "BUY" else "EXIT"
        signals.append({
            "Bar Time": ind.index[i],
            "Close": float(row["Close"]),
            "Supertrend": round(float(row["Supertrend"]), 2) if pd.notna(row.get("Supertrend")) else np.nan,
            "ST_Signal": st_sig,
            "Final_Signal": signal,
        })

    sig_df = pd.DataFrame(signals)
    trades, eq_curve, cash, token_qty, regime = _simulate_trades(sig_df, initial_capital)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_df = pd.DataFrame(eq_curve)

    stats = _compute_stats(sig_df, equity_df, trades_df, trades, initial_capital,
                           ticker=ticker, chart_interval=chart_interval,
                           mode_name=f"Supertrend Only (ATR {atr_period}, {atr_multiplier}x)")
    return {"stats": stats, "trades": trades_df, "equity_curve": equity_df, "signals": sig_df}


# ═══════════════════════════════════════════════════════════════════
# 8. SRMTF — Support/Resistance Channels (LonesomeTheBlue method)
#    Pivot-based S/R with strength scoring, channel clustering
# ═══════════════════════════════════════════════════════════════════

def _find_pivots(highs: np.ndarray, lows: np.ndarray, prd: int) -> list[float]:
    """
    Find pivot points — a bar is a pivot high if it's the highest of
    (prd bars left + itself + prd bars right). Same logic for pivot lows.
    Returns list of pivot price values.
    """
    pivots = []
    for i in range(prd, len(highs) - prd):
        # Pivot high: bar's high >= max of surrounding window
        window_highs = highs[i - prd: i + prd + 1]
        if highs[i] >= np.max(window_highs):
            pivots.append(float(highs[i]))

        # Pivot low: bar's low <= min of surrounding window
        window_lows = lows[i - prd: i + prd + 1]
        if lows[i] <= np.min(window_lows):
            pivots.append(float(lows[i]))

    return pivots


def _find_sr_channels(
    pivots: list[float],
    ohlc_df: pd.DataFrame,
    channel_width_pct: float = 6.0,
    min_strength: int = 2,
    max_channels: int = 6,
    strength_lookback: int = 500,
) -> list[dict]:
    """
    Cluster pivot points into S/R channels, score by strength.

    Logic (mirrors LonesomeTheBlue Pine Script):
    1. Calculate max channel width from price range
    2. For each pivot, group nearby pivots within channel width
    3. Score each channel: 20 points per pivot + 1 point per OHLC bar that touches it
    4. Sort by strength, return top N non-overlapping channels
    """
    if not pivots:
        return []

    prdhighest = max(pivots)
    prdlowest = min(pivots)
    cwidth = (prdhighest - prdlowest) * channel_width_pct / 100

    # Build raw channels: for each pivot, expand to include nearby pivots
    raw_channels = []
    for i, base_pv in enumerate(pivots):
        lo = base_pv
        hi = base_pv
        num_pivots = 0
        for pv in pivots:
            width = (hi - pv) if pv <= hi else (pv - lo)
            if width <= cwidth:
                lo = min(lo, pv)
                hi = max(hi, pv)
                num_pivots += 20  # 20 points per pivot (matches Pine)
        raw_channels.append({"hi": hi, "lo": lo, "pivot_strength": num_pivots, "idx": i})

    # Add OHLC touch strength (last N bars on the current timeframe)
    close_arr = ohlc_df["Close"].values
    high_arr = ohlc_df["High"].values
    low_arr = ohlc_df["Low"].values
    open_arr = ohlc_df["Open"].values if "Open" in ohlc_df.columns else close_arr
    bars_to_check = min(strength_lookback, len(close_arr))

    for ch in raw_channels:
        h, l = ch["hi"], ch["lo"]
        touch_count = 0
        for j in range(bars_to_check):
            idx = len(close_arr) - 1 - j
            if idx < 0:
                break
            if ((high_arr[idx] <= h and high_arr[idx] >= l) or
                (low_arr[idx] <= h and low_arr[idx] >= l) or
                (open_arr[idx] <= h and open_arr[idx] >= l) or
                (close_arr[idx] <= h and close_arr[idx] >= l)):
                touch_count += 1
        ch["total_strength"] = ch["pivot_strength"] + touch_count

    # Select top non-overlapping channels by strength
    # Sort by total_strength descending
    raw_channels.sort(key=lambda c: c["total_strength"], reverse=True)

    selected = []
    used_pivots = set()
    for ch in raw_channels:
        if ch["total_strength"] < min_strength * 20:
            continue
        # Check overlap with already-selected channels
        overlaps = False
        for sel in selected:
            if (ch["hi"] >= sel["lo"] and ch["lo"] <= sel["hi"]):
                overlaps = True
                break
        if overlaps:
            continue

        selected.append(ch)
        if len(selected) >= max_channels:
            break

    # Sort selected by strength (strongest first)
    selected.sort(key=lambda c: c["total_strength"], reverse=True)

    # Label as support or resistance relative to last close
    last_close = float(close_arr[-1]) if len(close_arr) > 0 else 0
    for ch in selected:
        if ch["lo"] > last_close:
            ch["type"] = "Resistance"
        elif ch["hi"] < last_close:
            ch["type"] = "Support"
        else:
            ch["type"] = "In-Channel"

    return selected


def run_backtest_srmtf(
    df_chart: pd.DataFrame,
    *,
    ticker: str,
    chart_interval: str = "1d",
    backtest_days: int | None = None,
    backtest_start_date: str | None = None,
    backtest_end_date: str | None = None,
    pivot_period: int = 5,
    loopback: int = 250,
    channel_width_pct: float = 6.0,
    min_strength: int = 2,
    max_channels: int = 6,
    initial_capital: float = 10000.0,
) -> dict:
    """
    SRMTF strategy (LonesomeTheBlue method):
    - Detect S/R channels using pivot points with strength scoring
    - BUY on resistance breakout (close > strongest resistance channel top)
    - SELL on support breakdown (close < strongest support channel bottom)
    - S/R levels recalculated on each bar using rolling window

    Parameters match the TradingView indicator:
      pivot_period=5, loopback=250, channel_width_pct=6, min_strength=2, max_channels=6
    """
    df = df_chart.copy()

    # Ensure we have enough data
    warmup = max(loopback + pivot_period * 2, 300)
    if len(df) < warmup:
        raise RuntimeError(f"Need at least {warmup} bars for SRMTF, got {len(df)}")

    # Precompute arrays
    highs_full = df["High"].values.astype(float)
    lows_full = df["Low"].values.astype(float)

    # Trim AFTER computing (we need the full data for S/R detection)
    trim_start_idx = warmup  # start simulation after warmup
    df_trimmed = df.iloc[trim_start_idx:]
    df_trimmed = _trim_df(df_trimmed, backtest_days, backtest_start_date, backtest_end_date)

    signals = []
    in_trade = False
    current_resistance_top = np.nan
    current_support_bottom = np.nan

    for i in range(len(df_trimmed)):
        abs_idx = df.index.get_loc(df_trimmed.index[i])
        if isinstance(abs_idx, slice):
            abs_idx = abs_idx.start

        # Get the lookback window for pivot detection
        start_idx = max(0, abs_idx - loopback)
        window_highs = highs_full[start_idx:abs_idx + 1]
        window_lows = lows_full[start_idx:abs_idx + 1]

        close = float(df_trimmed.iloc[i]["Close"])

        # Find pivots in the window
        pivots = _find_pivots(window_highs, window_lows, pivot_period)

        if pivots:
            # Find S/R channels
            ohlc_window = df.iloc[start_idx:abs_idx + 1]
            channels = _find_sr_channels(
                pivots, ohlc_window,
                channel_width_pct=channel_width_pct,
                min_strength=min_strength,
                max_channels=max_channels,
            )

            # Find strongest resistance above price and strongest support below
            resistances = [c for c in channels if c["hi"] > close]
            supports = [c for c in channels if c["lo"] < close]

            if resistances:
                # Nearest resistance
                nearest_res = min(resistances, key=lambda c: c["lo"])
                current_resistance_top = nearest_res["hi"]
            else:
                current_resistance_top = close * 1.10  # fallback: 10% above

            if supports:
                # Nearest support
                nearest_sup = max(supports, key=lambda c: c["hi"])
                current_support_bottom = nearest_sup["lo"]
            else:
                current_support_bottom = close * 0.90  # fallback: 10% below

        # Trading logic
        if not in_trade and close > current_resistance_top:
            signal = "BUY"
            in_trade = True
        elif in_trade and close < current_support_bottom:
            signal = "EXIT"
            in_trade = False
        elif in_trade:
            signal = "BUY"
        else:
            signal = "EXIT"

        signals.append({
            "Bar Time": df_trimmed.index[i],
            "Close": close,
            "Resistance": round(current_resistance_top, 2),
            "Support": round(current_support_bottom, 2),
            "Num_Channels": len(channels) if pivots else 0,
            "Final_Signal": signal,
        })

    sig_df = pd.DataFrame(signals)
    trades, eq_curve, cash, token_qty, regime = _simulate_trades(sig_df, initial_capital)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_df = pd.DataFrame(eq_curve)

    stats = _compute_stats(sig_df, equity_df, trades_df, trades, initial_capital,
                           ticker=ticker, chart_interval=chart_interval,
                           mode_name=f"SRMTF (prd={pivot_period}, lb={loopback}, w={channel_width_pct}%)")
    return {"stats": stats, "trades": trades_df, "equity_curve": equity_df, "signals": sig_df}


# ═══════════════════════════════════════════════════════════════════
# Wrappers for backtest.py strategies (common interface)
# ═══════════════════════════════════════════════════════════════════

def _wrap_trailing_stop(
    df_chart: pd.DataFrame,
    *,
    ticker: str,
    chart_interval: str = "1d",
    backtest_days: int | None = None,
    backtest_start_date: str | None = None,
    backtest_end_date: str | None = None,
    initial_capital: float = 10000.0,
    atr_period: int = 10,
    atr_mult: float = 3.0,
    **kwargs,
) -> dict:
    """Wrapper to give run_backtest_trailing_stop the common interface."""
    from backtest import run_backtest_trailing_stop
    return run_backtest_trailing_stop(
        df_chart,
        ticker=ticker,
        chart_interval=chart_interval,
        backtest_days=backtest_days,
        backtest_start_date=backtest_start_date,
        backtest_end_date=backtest_end_date,
        atr_period=atr_period,
        atr_mult=atr_mult,
        initial_capital=initial_capital,
        macro_regime_mode=kwargs.get("macro_regime_mode", "OFF"),
        macro_df=kwargs.get("macro_df"),
        macro_override=kwargs.get("macro_override", "AUTO"),
    )


def _wrap_full(
    df_chart: pd.DataFrame,
    *,
    ticker: str,
    chart_interval: str = "1d",
    backtest_days: int | None = None,
    backtest_start_date: str | None = None,
    backtest_end_date: str | None = None,
    initial_capital: float = 10000.0,
    **kwargs,
) -> dict:
    """Wrapper to give run_backtest (full) the common interface."""
    from backtest import run_backtest, DEFAULT_ATR_PERIOD, DEFAULT_ATR_MULT, \
        DEFAULT_RSI_PERIOD, DEFAULT_VOL_LOOKBACK, DEFAULT_ADXR_LEN, \
        DEFAULT_ADXR_LENX, DEFAULT_ADXR_LOW, DEFAULT_ADXR_EPS
    return run_backtest(
        df_chart,
        kwargs.get("df_filter"),
        ticker=ticker,
        chart_interval=chart_interval,
        filter_interval=kwargs.get("filter_interval"),
        backtest_days=backtest_days,
        backtest_start_date=backtest_start_date,
        backtest_end_date=backtest_end_date,
        in_token_pct=kwargs.get("in_token_pct", 1.0),
        out_token_pct=kwargs.get("out_token_pct", 0.0),
        atr_period=kwargs.get("atr_period", DEFAULT_ATR_PERIOD),
        atr_mult=kwargs.get("atr_mult", DEFAULT_ATR_MULT),
        rsi_period=kwargs.get("rsi_period", DEFAULT_RSI_PERIOD),
        vol_lookback=kwargs.get("vol_lookback", DEFAULT_VOL_LOOKBACK),
        adxr_len=kwargs.get("adxr_len", DEFAULT_ADXR_LEN),
        adxr_lenx=kwargs.get("adxr_lenx", DEFAULT_ADXR_LENX),
        adxr_low=kwargs.get("adxr_low", DEFAULT_ADXR_LOW),
        adxr_eps=kwargs.get("adxr_eps", DEFAULT_ADXR_EPS),
        initial_capital=initial_capital,
        macro_regime_mode=kwargs.get("macro_regime_mode", "OFF"),
        macro_df=kwargs.get("macro_df"),
        macro_override=kwargs.get("macro_override", "AUTO"),
    )


def _wrap_my_final_strategy1(
    df_chart: pd.DataFrame,
    *,
    ticker: str,
    chart_interval: str = "1d",
    backtest_days: int | None = None,
    backtest_start_date: str | None = None,
    backtest_end_date: str | None = None,
    initial_capital: float = 10000.0,
    score_threshold: int = 50,
    score_recompute_interval: int = 5,
    **kwargs,
) -> dict:
    """Wrapper for myFinalStrategy1 with the common strategy interface."""
    from backtest_my_final_strategy import run_backtest_my_final_strategy1
    return run_backtest_my_final_strategy1(
        df_chart,
        ticker=ticker,
        chart_interval=chart_interval,
        backtest_days=backtest_days,
        backtest_start_date=backtest_start_date,
        backtest_end_date=backtest_end_date,
        initial_capital=initial_capital,
        score_threshold=score_threshold,
        score_recompute_interval=score_recompute_interval,
        macro_df=kwargs.get("macro_df"),
        macro_override=kwargs.get("macro_override", "AUTO"),
        spy_close=kwargs.get("spy_close"),
        **{k: v for k, v in kwargs.items()
           if k not in ("macro_df", "macro_override", "spy_close")},
    )


# ═══════════════════════════════════════════════════════════════════
# UNIFIED Strategy Registry — single source of truth for ALL strategies
#
# Every strategy has:
#   key         : unique string id
#   name        : display name for UI
#   description : one-liner for tooltips
#   func        : callable with common interface:
#                  func(df_chart, *, ticker, chart_interval, backtest_days,
#                       backtest_start_date, backtest_end_date, initial_capital, **params)
#   params      : default kwargs passed to func (strategy-specific)
#   icon        : emoji for UI display
#
# Usage:
#   from backtest_strategies import STRATEGY_REGISTRY, run_strategy
#   result = run_strategy("chandelier", df, ticker="AAPL", backtest_days=730)
# ═══════════════════════════════════════════════════════════════════

STRATEGY_REGISTRY = {
    "trailing-stop": {
        "name": "Trailing Stop (Supertrend)",
        "icon": "🎯",
        "func": _wrap_trailing_stop,
        "description": "Supertrend-only exit/re-entry. Starts invested at bar 1. "
                       "Fastest re-entry. Optional macro kill switch.",
        "params": {"atr_period": 10, "atr_mult": 3.0},
    },
    "full": {
        "name": "Full (ST+MOST+ADXR)",
        "icon": "🔧",
        "func": _wrap_full,
        "description": "Triple-confirmation: Supertrend + MOST RSI + ADXR. "
                       "Best for crypto, conservative for stocks.",
        "params": {},
    },
    "supertrend-only": {
        "name": "Supertrend Only",
        "icon": "📊",
        "func": run_backtest_supertrend_only,
        "description": "Pure Supertrend — BUY on green, EXIT on red. Nothing else.",
        "params": {"atr_period": 10, "atr_multiplier": 3.0},
    },
    "ma-crossover": {
        "name": "MA Crossover (50/200)",
        "icon": "📊",
        "func": run_backtest_ma_crossover,
        "description": "Golden cross = BUY, death cross = SELL.",
        "params": {"fast_period": 50, "slow_period": 200},
    },
    "price-vs-200ma": {
        "name": "Price vs 200MA",
        "icon": "📊",
        "func": run_backtest_price_vs_200ma,
        "description": "Above 200MA = hold, below = sell. Simplest trend filter.",
        "params": {"ma_period": 200},
    },
    "chandelier": {
        "name": "Chandelier Exit",
        "icon": "📊",
        "func": run_backtest_chandelier,
        "description": "ATR trailing stop from highest high. Only moves up.",
        "params": {"atr_period": 22, "atr_mult": 3.0},
    },
    "donchian": {
        "name": "Donchian Breakout",
        "icon": "📊",
        "func": run_backtest_donchian,
        "description": "Buy on 20-day high, sell on 10-day low. Turtle Traders method.",
        "params": {"entry_period": 20, "exit_period": 10},
    },
    "mean-reversion-rsi": {
        "name": "Mean Reversion RSI",
        "icon": "📊",
        "func": run_backtest_mean_reversion_rsi,
        "description": "Buy oversold (RSI<30), sell overbought (RSI>70).",
        "params": {"rsi_period": 14, "buy_threshold": 30.0, "sell_threshold": 70.0},
    },
    "support-resistance": {
        "name": "S/R Breakout (Simple)",
        "icon": "📊",
        "func": run_backtest_support_resistance,
        "description": "Simple swing high/low S/R detection, buy breakout, stop at support.",
        "params": {"sr_lookback": 20, "sr_tolerance_pct": 1.0},
    },
    "srmtf": {
        "name": "SRMTF (Pivot S/R Channels)",
        "icon": "📊",
        "func": run_backtest_srmtf,
        "description": "LonesomeTheBlue method: pivot-based S/R channels with strength scoring.",
        "params": {"pivot_period": 5, "loopback": 250, "channel_width_pct": 6.0,
                   "min_strength": 2, "max_channels": 6},
    },
    "my-final-strategy1": {
        "name": "myFinalStrategy1 (Signal+Macro+Score)",
        "icon": "🏆",
        "func": _wrap_my_final_strategy1,
        "description": "True stock pipeline: SIGNAL-Super-MOST-ADXR + Macro Regime "
                       "+ 45° Score gate. Starts OUT, waits for all conditions.",
        "params": {"score_threshold": 50, "score_recompute_interval": 5},
    },
}


def get_strategy_display_name(key: str) -> str:
    """Get display name with icon for a strategy key."""
    info = STRATEGY_REGISTRY.get(key, {})
    icon = info.get("icon", "📊")
    name = info.get("name", key)
    return f"{icon} {name}"


def get_strategy_keys() -> list[str]:
    """Get all strategy keys in registry order."""
    return list(STRATEGY_REGISTRY.keys())


def get_strategy_choices() -> dict[str, str]:
    """Get {key: display_name} for all strategies. Used by UI dropdowns."""
    return {k: get_strategy_display_name(k) for k in STRATEGY_REGISTRY}


def run_strategy(
    key: str,
    df_chart: pd.DataFrame,
    *,
    ticker: str,
    chart_interval: str = "1d",
    backtest_days: int | None = None,
    backtest_start_date: str | None = None,
    backtest_end_date: str | None = None,
    initial_capital: float = 10000.0,
    **extra_kwargs,
) -> dict:
    """
    Run any strategy by key. Single entry point for all backtests.

    Usage:
        result = run_strategy("chandelier", df, ticker="AAPL", backtest_days=730)
        result = run_strategy("trailing-stop", df, ticker="AAPL", backtest_days=730,
                              macro_regime_mode="ON", macro_df=macro_df)

    Returns dict with: stats, trades, equity_curve, signals
    """
    if key not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{key}'. Available: {list(STRATEGY_REGISTRY.keys())}")

    info = STRATEGY_REGISTRY[key]
    func = info["func"]
    params = info["params"].copy()

    # Merge default params with any overrides from extra_kwargs
    params.update(extra_kwargs)

    return func(
        df_chart,
        ticker=ticker,
        chart_interval=chart_interval,
        backtest_days=backtest_days,
        backtest_start_date=backtest_start_date,
        backtest_end_date=backtest_end_date,
        initial_capital=initial_capital,
        **params,
    )