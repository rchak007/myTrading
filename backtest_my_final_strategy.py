# backtest_my_final_strategy.py
"""
myFinalStrategy1 — True backtest of the complete stock trading system:
  SIGNAL-Super-MOST-ADXR on 1D → Macro_Regime → Regime-modified signal → Score_Weighted gate

This replicates exactly how you trade stocks in real life:
  1. Compute SIGNAL-Super-MOST-ADXR per bar (Supertrend + MOST + ADXR)
  2. Look up historical macro regime (VIX + SPY vs 200MA + breadth) per bar
  3. Apply regime modification (BEAR blocks BUY, BULL upgrades HOLD→BUY)
  4. Compute rolling 45° Score_Weighted using data available up to that bar
  5. BUY only when: regime-modified signal = BUY AND Score_Weighted >= threshold
  6. SELL when: regime-modified signal = EXIT or STANDDOWN
  7. Starts OUT (cash) — waits for all conditions to align before first entry

Score is computed every `score_recompute_interval` bars (default 5) and cached
to avoid the O(N * 120-bar regression) cost of doing it every single bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import timedelta


def _lookup_macro_detail(bar_time, macro_df):
    """
    Look up VIX, SPY Close, SPY 200MA, Breadth % for a given bar time.
    Returns dict with the raw macro values for display in trade log.
    """
    result = {"VIX": np.nan, "SPY_Close": np.nan, "SPY_200MA": np.nan,
              "SPY_Above_200MA": "N/A", "Breadth_Pct": np.nan}

    if macro_df is None or macro_df.empty:
        return result

    bar_date = pd.Timestamp(bar_time).normalize()
    if hasattr(bar_date, 'tzinfo') and bar_date.tzinfo is not None:
        bar_date = bar_date.tz_localize(None)

    mask = macro_df.index <= bar_date
    if not mask.any():
        return result

    row = macro_df.loc[mask].iloc[-1]
    result["VIX"] = round(float(row["VIX_Close"]), 1) if pd.notna(row.get("VIX_Close")) else np.nan
    result["SPY_Close"] = round(float(row["SPY_Close"]), 2) if pd.notna(row.get("SPY_Close")) else np.nan
    result["SPY_200MA"] = round(float(row["SPY_200MA"]), 2) if pd.notna(row.get("SPY_200MA")) else np.nan
    result["SPY_Above_200MA"] = "YES" if row.get("SPY_Above_200MA") else "NO"
    result["Breadth_Pct"] = round(float(row["Breadth_Pct"]), 0) if pd.notna(row.get("Breadth_Pct")) else np.nan

    return result


def run_backtest_my_final_strategy1(
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
    atr_period: int = 10,
    atr_mult: float = 3.0,
    rsi_period: int = 14,
    vol_lookback: int = 20,
    adxr_len: int = 14,
    adxr_lenx: int = 14,
    adxr_low_threshold: float = 20.0,
    adxr_flat_eps: float = 1e-6,
    **kwargs,
) -> dict:
    """
    True backtest of the full stock pipeline:
      SIGNAL-Super-MOST-ADXR + Macro Regime + Score_Weighted gate.

    Parameters
    ----------
    score_threshold : int
        Minimum Score_Weighted required to enter a position (default 50).
        Your live system uses 70 for top picks, but 50 is a reasonable
        backtest threshold since score fluctuates bar-to-bar.
    score_recompute_interval : int
        Recompute the 45° score every N bars (default 5 = weekly on 1D).
        Saves compute time. Score doesn't change much day-to-day.
    """
    from core.indicators import apply_indicators
    from core.signals import signal_super_most_adxr
    from backtest import (
        fetch_historical_macro_data,
        lookup_regime_for_bar,
        apply_regime_to_signal,
    )
    from data.stock_scoring import calculate_all_scores

    # ── 1. Apply indicators to full data (warmup included) ──
    indicator_params = dict(
        atr_period=atr_period,
        atr_multiplier=atr_mult,
        rsi_period=rsi_period,
        vol_lookback=vol_lookback,
        adxr_len=adxr_len,
        adxr_lenx=adxr_lenx,
        adxr_low_threshold=adxr_low_threshold,
        adxr_flat_eps=adxr_flat_eps,
    )

    print(f"📊 [myFinalStrategy1] Applying indicators to {len(df_chart)} bars ...")
    # Save full OHLCV data BEFORE indicators/trimming — scoring needs this
    full_ohlcv = df_chart.copy()
    ind_chart = apply_indicators(df_chart, **indicator_params)

    # ── 2. Fetch historical macro data (VIX + SPY + breadth → regime per day) ──
    macro_df = kwargs.get("macro_df")
    if macro_df is None:
        print("🌐 [myFinalStrategy1] Fetching historical macro data (VIX + SPY) ...")
        macro_df = fetch_historical_macro_data(lookback_days=3650)
        print(f"   Macro data: {len(macro_df)} days loaded")

    macro_override = kwargs.get("macro_override", "AUTO")

    # ── 3. Fetch SPY close for relative strength in scoring ──
    spy_close = kwargs.get("spy_close")
    if spy_close is None:
        try:
            import yfinance as yf
            from core.utils import _fix_yf_cols
            print("📈 [myFinalStrategy1] Fetching SPY data for scoring ...")
            spy_raw = yf.download("SPY", period="3650d", interval="1d", progress=False)
            if spy_raw is not None and not spy_raw.empty:
                spy_raw = _fix_yf_cols(spy_raw)
                spy_close = spy_raw["Close"]
                print(f"   SPY data: {len(spy_close)} bars loaded")
        except Exception as e:
            print(f"⚠️  SPY fetch failed: {e} — scoring will skip relative strength")
            spy_close = None

    # ── 4. Trim to backtest window ──
    if backtest_start_date is not None or backtest_end_date is not None:
        pre_trim = len(ind_chart)
        if backtest_start_date is not None:
            ind_chart = ind_chart[ind_chart.index >= pd.Timestamp(backtest_start_date)]
        if backtest_end_date is not None:
            end_ts = pd.Timestamp(backtest_end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            ind_chart = ind_chart[ind_chart.index <= end_ts]
        print(f"✂️  Trimmed: {pre_trim} → {len(ind_chart)} bars")
    elif backtest_days is not None:
        last_bar = ind_chart.index[-1]
        cutoff = pd.Timestamp(last_bar) - timedelta(days=backtest_days)
        pre_trim = len(ind_chart)
        ind_chart = ind_chart[ind_chart.index >= cutoff]
        print(f"✂️  Trimmed: {pre_trim} → {len(ind_chart)} bars ({backtest_days}d window)")

    # ── 5. Build per-bar signals with macro + score ──
    signals = []
    cached_score = 0
    cached_score_30 = 0
    cached_score_60 = 0
    cached_score_90 = 0
    cached_score_120 = 0
    last_score_bar = -999  # force recompute on first bar

    for i in range(len(ind_chart)):
        row = ind_chart.iloc[i]
        bar_time = ind_chart.index[i]
        close = float(row["Close"])

        # ── 5a. Compute SIGNAL-Super-MOST-ADXR ──
        st_sig = str(row.get("Supertrend_Signal", "SELL"))
        most_sig = str(row.get("MOST_Signal", "SELL"))
        adxr_state = str(row.get("ADXR_State", "FLAT"))
        raw_signal = signal_super_most_adxr(st_sig, most_sig, adxr_state)

        # ── 5b. Look up macro regime + raw macro values for this bar ──
        bar_regime = lookup_regime_for_bar(bar_time, macro_df, override=macro_override)
        macro_detail = _lookup_macro_detail(bar_time, macro_df)

        # ── 5c. Apply regime modification to signal ──
        regime_signal = apply_regime_to_signal(raw_signal, bar_regime)

        # ── 5d. Compute rolling 45° score (every N bars) ──
        if (i - last_score_bar) >= score_recompute_interval or i == 0:
            # Get all data up to this bar (no look-ahead)
            # Use full_ohlcv (the raw df_chart before indicators) for scoring
            data_up_to_bar = full_ohlcv.loc[:bar_time]

            if i == 0:
                print(f"\n{'='*80}")
                print(f"🔍 BAR 0 FULL DIAGNOSTIC — {ticker} @ {bar_time}")
                print(f"{'='*80}")
                print(f"full_ohlcv: {len(full_ohlcv)} bars, "
                      f"{full_ohlcv.index[0]} → {full_ohlcv.index[-1]}")
                print(f"data_up_to_bar: {len(data_up_to_bar)} bars, "
                      f"{data_up_to_bar.index[0]} → {data_up_to_bar.index[-1]}")
                print(f"columns: {list(data_up_to_bar.columns)}")
                print(f"Close dtype: {data_up_to_bar['Close'].dtype}")
                print(f"Last close value: {data_up_to_bar['Close'].iloc[-1]} "
                      f"(type={type(data_up_to_bar['Close'].iloc[-1])})")
                print(f"Last 10 closes:")
                for dt, val in data_up_to_bar['Close'].tail(10).items():
                    print(f"  {dt} → {val}")

                # Run scoring with VERBOSE output
                from data.stock_scoring import calculate_45_degree_score, regression_stats
                from data.stock_scoring import ema, sma, returns_over_period, percent_above, max_dd_from_high

                c = data_up_to_bar["Close"]
                print(f"\n--- Raw scoring components (60d window) ---")
                slope_main, r2_main = regression_stats(c, 60, log=True)
                slope60, r2_60 = regression_stats(c, 60, log=True)
                print(f"slope_60d: {slope_main}")
                print(f"r2_60d:    {r2_main}")

                slope30, r2_30 = regression_stats(c, 30, log=True)
                print(f"slope_30d: {slope30}")
                print(f"r2_30d:    {r2_30}")

                slope90, r2_90 = regression_stats(c, 90, log=True)
                print(f"slope_90d: {slope90}")
                print(f"r2_90d:    {r2_90}")

                slope120, r2_120 = regression_stats(c, 120, log=True)
                print(f"slope_120d: {slope120}")
                print(f"r2_120d:    {r2_120}")

                ideal_slope = 0.00274
                print(f"\nideal slope = {ideal_slope}")
                for label, sl in [("30d", slope30), ("60d", slope_main), ("90d", slope90), ("120d", slope120)]:
                    if pd.notna(sl):
                        diff = abs(sl - ideal_slope)
                        pts = 30 if diff < 0.00137 else (15 if diff < 0.00274 else 0)
                        print(f"  {label}: slope={sl:.6f}, diff={diff:.6f}, "
                              f"angle={np.degrees(np.arctan(sl)):.2f}°, pts={pts}")
                    else:
                        print(f"  {label}: slope=NaN")

                ema21 = ema(c, 21)
                sma50 = sma(c, 50)
                ret3m = returns_over_period(c, 63)
                perc21 = percent_above(c, ema21, 60)
                perc50 = percent_above(c, sma50, 60)
                mdd = max_dd_from_high(c, 120)

                print(f"\nret_63d: {ret3m}")
                print(f"perc_above_21 (60d): {perc21}")
                print(f"perc_above_50 (60d): {perc50}")
                print(f"max_dd_120d: {mdd}")

                # SPY relative strength
                if spy_close is not None:
                    spy_slice_dbg = spy_close.loc[:bar_time]
                    spy_ret = returns_over_period(spy_slice_dbg, 63)
                    rs = ret3m - spy_ret if pd.notna(ret3m) and pd.notna(spy_ret) else np.nan
                    print(f"spy_ret_63d: {spy_ret}")
                    print(f"rs_vs_spy: {rs}")

                # Now run actual calculate_all_scores
                print(f"\n--- Calling calculate_all_scores ---")

            if len(data_up_to_bar) >= 120:
                # Trim SPY close to same date range for relative strength
                spy_slice = None
                if spy_close is not None:
                    spy_slice = spy_close.loc[:bar_time]
                    if len(spy_slice) < 63:
                        spy_slice = None

                try:
                    score_data = calculate_all_scores(data_up_to_bar, spy_slice)
                    cached_score = int(score_data.get("Score_Weighted", 0))
                    cached_score_30 = int(score_data.get("Score_30", 0))
                    cached_score_60 = int(score_data.get("Score_60", 0))
                    cached_score_90 = int(score_data.get("Score_90", 0))
                    cached_score_120 = int(score_data.get("Score_120", 0))
                    if i == 0:
                        print(f"Score_30={cached_score_30}, Score_60={cached_score_60}, "
                              f"Score_90={cached_score_90}, Score_120={cached_score_120}, "
                              f"Score_W={cached_score}")
                        print(f"Full score_data: {score_data}")
                        print(f"{'='*80}\n")
                except Exception as e:
                    if i == 0:
                        print(f"EXCEPTION: {e}")
                        import traceback
                        traceback.print_exc()
                    cached_score = 0
                    cached_score_30 = cached_score_60 = cached_score_90 = cached_score_120 = 0
            else:
                if i == 0:
                    print(f"🔍 [Score Debug] Bar 0: NOT ENOUGH DATA — "
                          f"data_up_to_bar has {len(data_up_to_bar)} bars (need 120)")
                cached_score = 0
                cached_score_30 = cached_score_60 = cached_score_90 = cached_score_120 = 0
            last_score_bar = i

        # ── 5e. Final decision: combine regime signal + score gate ──
        if regime_signal == "BUY":
            if cached_score >= score_threshold:
                final_signal = "BUY"
            else:
                final_signal = "WAIT"  # signal says buy but score too low
        elif regime_signal in ("EXIT", "STANDDOWN"):
            final_signal = "EXIT"
        elif regime_signal == "HOLD":
            final_signal = "HOLD"
        elif regime_signal == "WAIT":
            final_signal = "WAIT"
        else:
            final_signal = "EXIT"

        signals.append({
            "Bar Time": bar_time,
            "Close": close,
            "Supertrend": round(float(row["Supertrend"]), 2) if pd.notna(row.get("Supertrend")) else np.nan,
            "ST_Signal": st_sig,
            "MOST_Signal": most_sig,
            "ADXR_State": adxr_state,
            "Raw_Signal": raw_signal,
            "Macro_Regime": bar_regime,
            "VIX": macro_detail["VIX"],
            "SPY_Close": macro_detail["SPY_Close"],
            "SPY_200MA": macro_detail["SPY_200MA"],
            "SPY_Above_200MA": macro_detail["SPY_Above_200MA"],
            "Breadth_Pct": macro_detail["Breadth_Pct"],
            "Regime_Signal": regime_signal,
            "Score_30": cached_score_30,
            "Score_60": cached_score_60,
            "Score_90": cached_score_90,
            "Score_120": cached_score_120,
            "Score_Weighted": cached_score,
            "Final_Signal": final_signal,
        })

    sig_df = pd.DataFrame(signals)

    # ── 6. Simulate trades ──
    first_close = sig_df.iloc[0]["Close"]
    bh_qty = initial_capital / first_close  # B&H buys on bar 0

    regime = "OUT"
    cash = initial_capital
    token_qty = 0.0
    trades = []
    equity_curve = []
    trade_entry_value = 0.0

    # ── Record initial state (bar 0) — no trade, just snapshot of conditions ──
    s0 = sig_df.iloc[0]
    trades.append({
        "Bar Time": s0["Bar Time"],
        "Action": "--- START ---",
        "Price": s0["Close"],
        "Shares": 0.0,
        "Cash": round(initial_capital, 2),
        "Total": round(initial_capital, 2),
        "PnL": 0.0,
        "PnL%": 0.0,
        "Signal": s0.get("Final_Signal", ""),
        "Macro_Regime": s0.get("Macro_Regime", "N/A"),
        "VIX": s0.get("VIX", np.nan),
        "SPY_vs_200MA": s0.get("SPY_Above_200MA", "N/A"),
        "Breadth": s0.get("Breadth_Pct", np.nan),
        "Score_30": s0.get("Score_30", 0),
        "Score_60": s0.get("Score_60", 0),
        "Score_90": s0.get("Score_90", 0),
        "Score_120": s0.get("Score_120", 0),
        "Score_W": s0.get("Score_Weighted", 0),
    })

    for idx in range(len(sig_df)):
        s = sig_df.iloc[idx]
        final = s["Final_Signal"]
        close = s["Close"]
        bar_time = s["Bar Time"]

        # Determine desired state
        if final == "BUY":
            desired = "IN"
        elif final == "HOLD":
            desired = regime  # no change — stay in current state
        else:
            desired = "OUT"

        # ── Transitions ──
        if regime == "OUT" and desired == "IN":
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
                "VIX": s.get("VIX", np.nan),
                "SPY_vs_200MA": s.get("SPY_Above_200MA", "N/A"),
                "Breadth": s.get("Breadth_Pct", np.nan),
                "Score_30": s.get("Score_30", 0),
                "Score_60": s.get("Score_60", 0),
                "Score_90": s.get("Score_90", 0),
                "Score_120": s.get("Score_120", 0),
                "Score_W": s.get("Score_Weighted", 0),
            })
            regime = "IN"

        elif regime == "IN" and desired == "OUT":
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
                "VIX": s.get("VIX", np.nan),
                "SPY_vs_200MA": s.get("SPY_Above_200MA", "N/A"),
                "Breadth": s.get("Breadth_Pct", np.nan),
                "Score_30": s.get("Score_30", 0),
                "Score_60": s.get("Score_60", 0),
                "Score_90": s.get("Score_90", 0),
                "Score_120": s.get("Score_120", 0),
                "Score_W": s.get("Score_Weighted", 0),
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

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_df = pd.DataFrame(equity_curve)

    # ── 7. Compute stats ──
    last_close = sig_df.iloc[-1]["Close"]
    final_value = cash + token_qty * last_close
    bh_final = bh_qty * last_close

    # ── Record final state (last bar) — snapshot of where we ended ──
    sN = sig_df.iloc[-1]
    end_row = {
        "Bar Time": sN["Bar Time"],
        "Action": "--- END ---",
        "Price": last_close,
        "Shares": round(token_qty, 4),
        "Cash": round(cash, 2),
        "Total": round(final_value, 2),
        "PnL": round(final_value - initial_capital, 2),
        "PnL%": round((final_value - initial_capital) / initial_capital * 100, 2),
        "Signal": sN.get("Final_Signal", ""),
        "Macro_Regime": sN.get("Macro_Regime", "N/A"),
        "VIX": sN.get("VIX", np.nan),
        "SPY_vs_200MA": sN.get("SPY_Above_200MA", "N/A"),
        "Breadth": sN.get("Breadth_Pct", np.nan),
        "Score_30": sN.get("Score_30", 0),
        "Score_60": sN.get("Score_60", 0),
        "Score_90": sN.get("Score_90", 0),
        "Score_120": sN.get("Score_120", 0),
        "Score_W": sN.get("Score_Weighted", 0),
    }
    if not trades_df.empty:
        trades_df = pd.concat([trades_df, pd.DataFrame([end_row])], ignore_index=True)
    else:
        trades_df = pd.DataFrame([end_row])

    strategy_pnl = final_value - initial_capital
    strategy_return = (strategy_pnl / initial_capital) * 100
    bh_pnl = bh_final - initial_capital
    bh_return = (bh_pnl / initial_capital) * 100
    alpha = strategy_return - bh_return

    # Trade stats
    if not trades_df.empty:
        sells = trades_df[trades_df["Action"] == "SELL"]
        n_trades = len(sells)
        if "PnL" in sells.columns and not sells.empty:
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
        bars_in = len(equity_df[equity_df["Regime"] == "IN"])
        time_in = (bars_in / len(equity_df)) * 100
    else:
        max_dd = bh_max_dd = time_in = 0

    # Open position PnL
    if regime == "IN" and trade_entry_value > 0:
        open_pnl = final_value - trade_entry_value
        open_pnl_pct = (open_pnl / trade_entry_value) * 100
    else:
        open_pnl = open_pnl_pct = 0.0

    stats = {
        "Ticker": ticker,
        "Mode": f"myFinalStrategy1 (Score≥{score_threshold}, Macro={macro_override})",
        "Chart": chart_interval.upper(),
        "Filter": "None (1D with macro + score)",
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