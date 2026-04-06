#!/usr/bin/env python3
"""
backtest_app.py — Streamlit UI for backtesting the Supertrend + MOST RSI + ADXR strategy.

Run locally:
  streamlit run backtest_app.py --server.port 8503

Reuses the same backtest engine from backtest.py.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Ensure project root is on path (same pattern as app.py)
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import numpy as np
import pandas as pd
import streamlit as st

from backtest import (
    fetch_yahoo_ohlcv,
    fetch_gecko_ohlcv,
    parse_timeframe,
    fetch_historical_macro_data,
    DEFAULT_ATR_PERIOD,
    DEFAULT_ATR_MULT,
    DEFAULT_RSI_PERIOD,
    DEFAULT_VOL_LOOKBACK,
    DEFAULT_ADXR_LEN,
    DEFAULT_ADXR_LENX,
    DEFAULT_ADXR_LOW,
    DEFAULT_ADXR_EPS,
)
from backtest_strategies import (
    STRATEGY_REGISTRY,
    run_strategy,
    get_strategy_choices,
    get_strategy_keys,
    get_strategy_display_name,
)
from core.config import MACRO_REGIME_OVERRIDE


# ═══════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="📈 Strategy Backtester",
    page_icon="📈",
    layout="wide",
)


# ═══════════════════════════════════════════════════════════════════
# Sidebar — all parameters
# ═══════════════════════════════════════════════════════════════════
st.sidebar.title("📈 Backtester")
st.sidebar.markdown("---")

# ── Backtest Mode ──
st.sidebar.subheader("🎯 Strategy Mode")

strategy_choices = get_strategy_choices()
strategy_keys_list = get_strategy_keys()
strategy_labels_list = [strategy_choices[k] for k in strategy_keys_list]

selected_mode_label = st.sidebar.selectbox(
    "Strategy",
    strategy_labels_list,
    index=0,  # default to trailing stop
    help="Choose which strategy to backtest against buy-and-hold.",
)
selected_mode = strategy_keys_list[strategy_labels_list.index(selected_mode_label)]
is_trailing_stop_mode = selected_mode == "trailing-stop"
is_full_mode = selected_mode == "full"
is_my_final_mode = selected_mode == "my-final-strategy1"
is_strategy_mode = selected_mode in STRATEGY_REGISTRY and selected_mode not in ("trailing-stop", "full", "my-final-strategy1")

# Show strategy description
st.sidebar.caption(STRATEGY_REGISTRY[selected_mode]["description"])

st.sidebar.markdown("---")

# ── Symbol & Data Source ──
st.sidebar.subheader("Symbol & Data")
symbol = st.sidebar.text_input("Symbol", value="AAPL", help="Stock: AAPL, MSFT  |  Crypto: SOL-USD, ETH-USD, BTC-USD")

data_source = st.sidebar.radio("Data source", ["Yahoo Finance", "GeckoTerminal"], horizontal=True)
gecko_url = ""
if data_source == "GeckoTerminal":
    gecko_url = st.sidebar.text_input(
        "GeckoTerminal Pool URL",
        placeholder="https://www.geckoterminal.com/solana/pools/...",
        help="Full pool URL for tokens not on Yahoo Finance",
    )

# ── Chart & Filter ──
st.sidebar.subheader("Timeframes")

col1, col2 = st.sidebar.columns(2)
with col1:
    chart_interval = st.selectbox("Chart", ["4h", "1d", "1h", "2h", "1wk"], index=1)
with col2:
    filter_options = ["None", "1d", "1wk", "4h", "2h", "1h"]
    filter_choice = st.selectbox("Filter (HTF)", filter_options, index=0)

filter_interval = None if filter_choice == "None" else filter_choice

st.sidebar.subheader("Period")

period_mode = st.sidebar.radio("Period mode", ["Preset", "Date Range"], horizontal=True)

lookback_str = None
lookback_days = None
date_start = None
date_end = None

if period_mode == "Preset":
    lookback_options = ["1m", "2m", "3m", "6m", "1y", "2y", "3y", "5y", "10y", "max", "Custom days"]
    lookback_choice = st.sidebar.selectbox(
        "Lookback period",
        lookback_options,
        index=5,  # default 2y
    )
    if lookback_choice == "Custom days":
        custom_days = st.sidebar.number_input(
            "Custom lookback (days)",
            min_value=30,
            max_value=10000,
            value=365,
            step=30,
        )
        lookback_str = f"{custom_days}d"
    else:
        lookback_str = lookback_choice
else:
    # Date range picker
    from datetime import date, timedelta
    default_end = date.today()
    default_start = default_end - timedelta(days=730)

    date_start = st.sidebar.date_input("Start date", value=default_start)
    date_end = st.sidebar.date_input("End date", value=default_end)

    if date_start >= date_end:
        st.sidebar.error("Start date must be before end date")
        st.stop()

initial_capital = st.sidebar.number_input(
    "Initial Capital ($)",
    min_value=100.0,
    max_value=10_000_000.0,
    value=10000.0,
    step=1000.0,
    format="%.0f",
)

# ── Position Sizing (auto-detect crypto vs stock) ──
is_crypto = (
    "-USD" in symbol.upper()
    or "-EUR" in symbol.upper()
    or "-BTC" in symbol.upper()
    or data_source == "GeckoTerminal"
)

if is_crypto:
    default_in_pct = 0.80
    default_out_pct = 0.20
    asset_type_label = "🪙 Crypto detected → 80/20 default"
else:
    default_in_pct = 1.00
    default_out_pct = 0.00
    asset_type_label = "📈 Stock detected → 100/0 default"

with st.sidebar.expander("💰 Position Sizing", expanded=False):
    st.caption(asset_type_label)
    in_token_pct = st.slider("BUY → % in token", min_value=0.0, max_value=1.0, value=default_in_pct, step=0.05, format="%.0f%%",
                             help="On BUY signal: this % of total value goes into the token/stock")
    out_token_pct = st.slider("SELL → % in token", min_value=0.0, max_value=1.0, value=default_out_pct, step=0.05, format="%.0f%%",
                              help="On SELL signal: this % of total value stays in the token/stock")

# ── Indicator Parameters ──
with st.sidebar.expander("⚙️ Indicator Parameters", expanded=False):
    atr_period = st.number_input("ATR Period", value=DEFAULT_ATR_PERIOD, min_value=1, max_value=50)
    atr_mult = st.number_input("ATR Multiplier", value=DEFAULT_ATR_MULT, min_value=0.5, max_value=10.0, step=0.1, format="%.1f")
    rsi_period = st.number_input("RSI Period", value=DEFAULT_RSI_PERIOD, min_value=2, max_value=50)
    vol_lookback = st.number_input("Volume Lookback", value=DEFAULT_VOL_LOOKBACK, min_value=5, max_value=100)
    adxr_len = st.number_input("ADXR Length", value=DEFAULT_ADXR_LEN, min_value=5, max_value=50)
    adxr_lenx = st.number_input("ADXR LenX", value=DEFAULT_ADXR_LENX, min_value=5, max_value=50)
    adxr_low = st.number_input("ADXR Low Threshold", value=DEFAULT_ADXR_LOW, min_value=5.0, max_value=50.0, step=1.0, format="%.1f")
    adxr_eps = st.number_input("ADXR Flat Epsilon", value=DEFAULT_ADXR_EPS, format="%.1e", step=1e-7)

# ── Macro Regime ──
with st.sidebar.expander("🌐 Macro Regime (Stocks)", expanded=False):
    st.caption("Uses historical VIX + SPY vs 200MA to adjust signals per bar")
    regime_enabled = st.checkbox("Enable regime-aware backtesting", value=False,
                                  help="Fetches historical VIX+SPY data and adjusts signals: "
                                       "BEAR blocks BUY, BULL upgrades HOLD→BUY")
    regime_override_options = ["AUTO", "BULL", "NEUTRAL", "BEAR"]
    default_regime_idx = regime_override_options.index(MACRO_REGIME_OVERRIDE) \
        if MACRO_REGIME_OVERRIDE in regime_override_options else 0
    regime_override = st.selectbox(
        "Regime override",
        regime_override_options,
        index=default_regime_idx,
        help="AUTO = compute from historical VIX+SPY per bar. "
             "Force BULL/NEUTRAL/BEAR to test all bars under one regime.",
        disabled=not regime_enabled,
    )
    if regime_enabled:
        st.info("📊 Signal modifications:\n"
                "- **BEAR**: BUY→WAIT, HOLD→EXIT\n"
                "- **NEUTRAL**: no change\n"
                "- **BULL**: HOLD→BUY")
        if regime_override != "AUTO":
            st.warning(f"⚠️ All bars forced to {regime_override}")

st.sidebar.markdown("---")

# ── Run button ──
run_clicked = st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True)

# Quick presets
st.sidebar.markdown("---")
st.sidebar.subheader("Quick Presets")
preset_cols = st.sidebar.columns(2)
with preset_cols[0]:
    if st.button("Crypto 4H+1D", use_container_width=True):
        st.session_state["preset"] = {"chart": "4h", "filter": "1d", "lookback": "1y"}
        st.rerun()
with preset_cols[1]:
    if st.button("Stock 1D", use_container_width=True):
        st.session_state["preset"] = {"chart": "1d", "filter": "None", "lookback": "2y"}
        st.rerun()


# ═══════════════════════════════════════════════════════════════════
# Main content area
# ═══════════════════════════════════════════════════════════════════
st.title("📈 Strategy Backtester")
st.caption("Supertrend + MOST RSI + ADXR  |  Strategy vs Buy & Hold")

if not run_clicked:
    # Show instructions when no backtest has been run
    st.info("Configure parameters in the sidebar and click **🚀 Run Backtest** to start.")

    with st.expander("📖 How it works", expanded=True):
        st.markdown("""
**Signal logic** (same as the live bot):
- **Supertrend** (ATR 10, mult 3.0): BUY when price > Supertrend line, SELL when below
- **MOST RSI**: BUY when MOST MA > MOST Line, SELL otherwise
- **ADXR**: RISING = trend confirmed, LOW_FLAT = standdown (choppy market)

**Combined signal** `signal_super_most_adxr`:
- **BUY** → Supertrend=BUY AND MOST=BUY AND ADXR=RISING
- **HOLD** → Supertrend=BUY but not fully aligned
- **EXIT** → Supertrend=SELL
- **STANDDOWN** → ADXR=LOW_FLAT

**Filter (optional)**: Higher timeframe confirmation. E.g. 4H chart with 1D filter — EXIT always passes through, BUY/HOLD only if 1D confirms.

**Comparison**: Strategy P&L vs Buy & Hold P&L (same capital, same period).
        """)

    st.stop()


# ═══════════════════════════════════════════════════════════════════
# Run the backtest
# ═══════════════════════════════════════════════════════════════════

# Determine fetch lookback and display label
if period_mode == "Preset":
    lookback_days = parse_timeframe(lookback_str)
    period_label = f"last {lookback_str}"
else:
    # Date range: calculate days from start to today (fetch enough data including warmup)
    from datetime import date
    total_span = (date.today() - date_start).days
    lookback_days = total_span + 300  # extra for warmup
    period_label = f"{date_start} → {date_end}"

# Header
if is_strategy_mode:
    mode_label = f"📊 {STRATEGY_REGISTRY[selected_mode]['name']}"
elif is_trailing_stop_mode:
    mode_label = "🎯 Trailing Stop"
elif is_my_final_mode:
    mode_label = "🏆 myFinalStrategy1"
else:
    mode_label = "🔧 Full"
st.markdown(f"### Backtesting `{symbol}` — {chart_interval.upper()}"
            + (f" with {filter_interval.upper()} filter" if filter_interval and is_full_mode else "")
            + f" — {period_label}"
            + f" — **{mode_label}**")

# ── Fetch data ──
# For myFinalStrategy1, fetch extra history so 45° scoring has enough data
# before the backtest window starts (needs 120+ bars for regression)
fetch_lookback = lookback_days
if is_my_final_mode:
    fetch_lookback = lookback_days + 400  # extra bars for score warmup

with st.spinner(f"📥 Fetching {chart_interval.upper()} data for {symbol}..."):
    try:
        if data_source == "GeckoTerminal" and gecko_url:
            df_chart = fetch_gecko_ohlcv(gecko_url, chart_interval, fetch_lookback)
        else:
            df_chart = fetch_yahoo_ohlcv(symbol, chart_interval, fetch_lookback)
    except Exception as e:
        st.error(f"❌ Failed to fetch chart data: {e}")
        st.stop()

df_filter = None
if filter_interval:
    with st.spinner(f"📥 Fetching {filter_interval.upper()} filter data..."):
        try:
            if data_source == "GeckoTerminal" and gecko_url:
                df_filter = fetch_gecko_ohlcv(gecko_url, filter_interval, lookback_days)
            else:
                df_filter = fetch_yahoo_ohlcv(symbol, filter_interval, lookback_days)
        except Exception as e:
            st.warning(f"⚠️ Filter data failed: {e} — proceeding without filter")
            filter_interval = None

# ── Fetch macro regime data (if enabled or needed by strategy) ──
macro_df_hist = None
if regime_enabled or is_my_final_mode:
    with st.spinner("🌐 Fetching historical VIX + SPY for macro regime..."):
        try:
            macro_df_hist = fetch_historical_macro_data(lookback_days + 500)
            st.success(f"✅ Macro regime data: {len(macro_df_hist)} daily bars")
        except Exception as e:
            st.warning(f"⚠️ Macro data fetch failed: {e} — running without regime filter")
            if not is_my_final_mode:
                regime_enabled = False

# ── Run backtest engine ──
with st.spinner("📊 Running backtest..."):
    # Build trimming args based on mode
    if period_mode == "Date Range":
        bt_days = None
        bt_start = str(date_start)
        bt_end = str(date_end)
    else:
        bt_days = parse_timeframe(lookback_str)
        bt_start = None
        bt_end = None

    if is_trailing_stop_mode:
        result = run_strategy(
            selected_mode, df_chart,
            ticker=symbol,
            chart_interval=chart_interval,
            backtest_days=bt_days,
            backtest_start_date=bt_start,
            backtest_end_date=bt_end,
            initial_capital=initial_capital,
            atr_period=atr_period,
            atr_mult=atr_mult,
            macro_regime_mode="ON" if regime_enabled else "OFF",
            macro_df=macro_df_hist,
            macro_override=regime_override if regime_enabled else "AUTO",
        )
    elif is_full_mode:
        result = run_strategy(
            selected_mode, df_chart,
            ticker=symbol,
            chart_interval=chart_interval,
            backtest_days=bt_days,
            backtest_start_date=bt_start,
            backtest_end_date=bt_end,
            initial_capital=initial_capital,
            df_filter=df_filter,
            filter_interval=filter_interval,
            in_token_pct=in_token_pct,
            out_token_pct=out_token_pct,
            atr_period=atr_period,
            atr_mult=atr_mult,
            rsi_period=rsi_period,
            vol_lookback=vol_lookback,
            adxr_len=adxr_len,
            adxr_lenx=adxr_lenx,
            adxr_low=adxr_low,
            adxr_eps=adxr_eps,
            macro_regime_mode="ON" if regime_enabled else "OFF",
            macro_df=macro_df_hist,
            macro_override=regime_override if regime_enabled else "AUTO",
        )
    elif is_my_final_mode:
        result = run_strategy(
            selected_mode, df_chart,
            ticker=symbol,
            chart_interval=chart_interval,
            backtest_days=bt_days,
            backtest_start_date=bt_start,
            backtest_end_date=bt_end,
            initial_capital=initial_capital,
            macro_df=macro_df_hist,
            macro_override=regime_override if regime_enabled else "AUTO",
        )
    else:
        result = run_strategy(
            selected_mode, df_chart,
            ticker=symbol,
            chart_interval=chart_interval,
            backtest_days=bt_days,
            backtest_start_date=bt_start,
            backtest_end_date=bt_end,
            initial_capital=initial_capital,
        )

stats = result["stats"]
trades_df = result["trades"]
equity_df = result["equity_curve"]
sig_df = result["signals"]


# ═══════════════════════════════════════════════════════════════════
# Results display
# ═══════════════════════════════════════════════════════════════════

# ── Top metrics row ──
m1, m2, m3, m4 = st.columns(4)
m1.metric("Strategy Return", stats["Strat Return"])
m2.metric("Buy & Hold Return", stats["B&H Return"])
m3.metric("Alpha", stats["Alpha (Strat - B&H)"])
m4.metric("Completed Trades", stats["Completed Trades"])

st.markdown("---")

# ── Side-by-side P&L comparison ──
col_strat, col_bh = st.columns(2)

with col_strat:
    st.markdown("#### 🤖 Strategy")
    s1, s2 = st.columns(2)
    s1.metric("Final Value", stats["Strat Final Value"])
    s2.metric("P&L", stats["Strat P&L"])
    s3, s4 = st.columns(2)
    s3.metric("Return", stats["Strat Return"])
    s4.metric("Max Drawdown", stats["Strat Max Drawdown"])

with col_bh:
    st.markdown("#### 📦 Buy & Hold")
    b1, b2 = st.columns(2)
    b1.metric("Final Value", stats["B&H Final Value"])
    b2.metric("P&L", stats["B&H P&L"])
    b3, b4 = st.columns(2)
    b3.metric("Return", stats["B&H Return"])
    b4.metric("Max Drawdown", stats["B&H Max Drawdown"])

# ── Advantage callout ──
st.markdown("---")
adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
adv_col1.metric("P&L Advantage ($ over B&H)", stats["P&L Advantage"])
adv_col2.metric("Time in Market", stats["Time in Market"])
adv_col3.metric("Position Sizing", stats["Position Sizing"])
if stats["Open Position"] == "YES":
    adv_col4.metric("Open Position P&L", f"{stats['Open PnL']}  ({stats['Open PnL%']})")
else:
    adv_col4.metric("Open Position", "No")

st.markdown("---")

# ── Equity curve chart ──
st.subheader("📈 Equity Curve — Strategy vs Buy & Hold")

if not equity_df.empty:
    chart_data = equity_df[["Bar Time", "Equity", "BH_Equity"]].copy()
    chart_data = chart_data.set_index("Bar Time")
    chart_data.columns = ["Strategy", "Buy & Hold"]
    st.line_chart(chart_data, use_container_width=True)

# ── Macro regime timeline (if enabled or myFinalStrategy1) ──
if (regime_enabled or is_my_final_mode) and "Macro_Regime" in sig_df.columns:
    st.subheader("🌐 Macro Regime Timeline")

    # Build regime transitions table
    regime_changes = []
    prev_regime = None
    for _, row in sig_df.iterrows():
        r = row.get("Macro_Regime", "N/A")
        if r != prev_regime:
            regime_changes.append({
                "Date": row["Bar Time"],
                "Regime": r,
                "Close": row["Close"],
            })
            prev_regime = r

    if regime_changes:
        rc_df = pd.DataFrame(regime_changes)
        regime_icons = {"BULL": "🟢", "NEUTRAL": "🟡", "BEAR": "🔴"}
        rc_df["Regime"] = rc_df["Regime"].apply(lambda x: f"{regime_icons.get(x, '')} {x}")

        st.caption(f"{len(regime_changes)} regime transitions over the backtest period:")
        st.dataframe(rc_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ── Trade stats ──
st.subheader("📊 Trade Statistics")
ts1, ts2, ts3, ts4 = st.columns(4)
ts1.metric("Win Rate", stats["Win Rate"])
ts2.metric("Avg Win", stats["Avg Win"])
ts3.metric("Avg Loss", stats["Avg Loss"])
ts4.metric("Best / Worst", f"{stats['Best Trade']} / {stats['Worst Trade']}")

# ── Trade log ──
st.subheader("📋 Trade Log")
if not trades_df.empty:
    # For myFinalStrategy1, show all columns (it has VIX, SPY, Breadth, Scores)
    if is_my_final_mode:
        display_cols = [
            "Bar Time", "Action", "Price", "Cash", "Total",
            "PnL", "PnL%", "Signal", "Macro_Regime",
            "VIX", "SPY_vs_200MA", "Breadth",
            "Score_30", "Score_60", "Score_90", "Score_120", "Score_W",
        ]
    else:
        display_cols = ["Bar Time", "Action", "Price", "Token Qty", "Cash", "Total", "Alloc"]
        if "PnL" in trades_df.columns:
            display_cols.insert(6, "PnL")
            display_cols.insert(7, "PnL%")
        display_cols.append("Signal")
        if "Macro_Regime" in trades_df.columns:
            display_cols.append("Macro_Regime")
    available = [c for c in display_cols if c in trades_df.columns]

    # Color the Action column — readable on both light and dark themes
    def highlight_action(row):
        if row.get("Action") == "BUY":
            return ["background-color: #2d6a2d; color: white"] * len(row)
        elif row.get("Action") == "SELL":
            return ["background-color: #8b2020; color: white"] * len(row)
        elif row.get("Action") in ("--- START ---", "--- END ---"):
            return ["background-color: #1a3a5c; color: white"] * len(row)
        return [""] * len(row)

    styled = trades_df[available].style.apply(highlight_action, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)
else:
    st.info("No trades executed in this period.")

# ── Signal distribution ──
st.subheader("📊 Signal Distribution")
if "Final_Signal" in sig_df.columns:
    signal_counts = sig_df["Final_Signal"].value_counts()
    dist_cols = st.columns(len(signal_counts))
    for i, (sig_name, cnt) in enumerate(signal_counts.items()):
        pct = cnt / len(sig_df) * 100
        dist_cols[i].metric(sig_name, f"{cnt} bars", f"{pct:.1f}%")

# ── Macro regime distribution (if enabled) ──
if regime_enabled and "Macro_Regime" in sig_df.columns:
    st.subheader("🌐 Macro Regime Distribution")
    regime_counts = sig_df["Macro_Regime"].value_counts()
    regime_icons = {"BULL": "🟢", "NEUTRAL": "🟡", "BEAR": "🔴"}
    rc_cols = st.columns(len(regime_counts))
    for i, (r_name, cnt) in enumerate(regime_counts.items()):
        pct = cnt / len(sig_df) * 100
        icon = regime_icons.get(r_name, "")
        rc_cols[i].metric(f"{icon} {r_name}", f"{cnt} bars", f"{pct:.1f}%")

    # Show how many signals were modified by regime
    if "Pre_Regime_Signal" in sig_df.columns:
        modified = sig_df[sig_df["Pre_Regime_Signal"] != sig_df["Final_Signal"]]
        if len(modified) > 0:
            st.caption(f"📝 Regime modified {len(modified)} of {len(sig_df)} signals "
                       f"({len(modified)/len(sig_df)*100:.1f}%)")
            with st.expander(f"🔍 Modified Signals ({len(modified)} bars)", expanded=False):
                mod_cols = ["Bar Time", "Close", "Macro_Regime", "Pre_Regime_Signal", "Final_Signal",
                            "Chart_Signal", "Filter_Signal"]
                available_mod = [c for c in mod_cols if c in modified.columns]
                st.dataframe(modified[available_mod], use_container_width=True, hide_index=True)
        else:
            st.caption("✅ No signals were modified by the regime filter")

# ── Full signals table (expandable) ──
with st.expander("🔍 Full Signals Table", expanded=False):
    st.dataframe(sig_df, use_container_width=True, hide_index=True)

# ── Config summary ──
with st.expander("⚙️ Backtest Configuration", expanded=False):
    config_data = {
        "Symbol": symbol,
        "Data Source": data_source,
        "Chart Interval": chart_interval.upper(),
        "Filter Interval": filter_interval.upper() if filter_interval else "None",
        "Lookback": period_label,
        "Initial Capital": f"${initial_capital:,.2f}",
        "BUY → Token %": f"{in_token_pct*100:.0f}%",
        "SELL → Token %": f"{out_token_pct*100:.0f}%",
        "Macro Regime": "ON" if regime_enabled else "OFF",
        "Regime Override": regime_override if regime_enabled else "N/A",
        "ATR Period": atr_period,
        "ATR Multiplier": atr_mult,
        "RSI Period": rsi_period,
        "Vol Lookback": vol_lookback,
        "ADXR Length": adxr_len,
        "ADXR LenX": adxr_lenx,
        "ADXR Low Threshold": adxr_low,
        "ADXR Flat Epsilon": adxr_eps,
        "Period": stats["Period"],
        "Total Bars": stats["Bars"],
    }
    st.json(config_data)

# ── Download buttons ──
st.markdown("---")
st.subheader("💾 Download Results")
dl1, dl2, dl3 = st.columns(3)

with dl1:
    if not trades_df.empty:
        csv_trades = trades_df.to_csv(index=False)
        st.download_button("📥 Trade Log CSV", csv_trades, f"bt_trades_{symbol}_{chart_interval}.csv", "text/csv")

with dl2:
    if not equity_df.empty:
        csv_equity = equity_df.to_csv(index=False)
        st.download_button("📥 Equity Curve CSV", csv_equity, f"bt_equity_{symbol}_{chart_interval}.csv", "text/csv")

with dl3:
    csv_signals = sig_df.to_csv(index=False)
    st.download_button("📥 Signals CSV", csv_signals, f"bt_signals_{symbol}_{chart_interval}.csv", "text/csv")