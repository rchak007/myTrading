#!/usr/bin/env python3
"""
backtest_compare_app.py — Batch strategy comparison across all stock tickers.

Run:  streamlit run backtest_compare_app.py --server.port 8504

Shows a single table: every ticker × every strategy, with return vs buy-and-hold.
Click any row to see trade details below.
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import date, timedelta

from core.utils import _fix_yf_cols
from backtest import fetch_yahoo_ohlcv, parse_timeframe
from backtest_strategies import (
    STRATEGY_REGISTRY,
    run_strategy,
    get_strategy_choices,
    get_strategy_keys,
    get_strategy_display_name,
)

# ═══════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="📊 Strategy Comparison",
    page_icon="📊",
    layout="wide",
)

# ═══════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════
st.sidebar.title("📊 Batch Strategy Comparison")
st.sidebar.markdown("---")

# ── Ticker source ──
st.sidebar.subheader("Tickers")
ticker_source = st.sidebar.radio("Source", ["From app.py (STOCK_TICKERS)", "Custom list"], horizontal=True)

if ticker_source == "From app.py (STOCK_TICKERS)":
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("myTrading_app", str(APP_DIR / "app.py"))
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        ALL_TICKERS = getattr(mod, "STOCK_TICKERS", [])
    except Exception as e:
        st.sidebar.error(f"Could not load STOCK_TICKERS from app.py: {e}")
        ALL_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOG", "AMZN", "META", "PLTR"]
    st.sidebar.caption(f"Loaded {len(ALL_TICKERS)} tickers from app.py")
else:
    custom_input = st.sidebar.text_area(
        "Enter tickers (one per line or comma-separated)",
        value="AAPL\nMSFT\nNVDA\nTSLA\nGOOG\nPLTR",
        height=150,
    )
    ALL_TICKERS = [t.strip().upper() for t in custom_input.replace(",", "\n").split("\n") if t.strip()]

# Allow selecting a subset
selected_tickers = st.sidebar.multiselect(
    "Select tickers to test",
    ALL_TICKERS,
    default=ALL_TICKERS,
)

st.sidebar.markdown("---")

# ── Period ──
st.sidebar.subheader("Period")
period_mode = st.sidebar.radio("Period mode", ["Preset", "Date Range"], horizontal=True)

if period_mode == "Preset":
    lookback_options = ["6m", "1y", "2y", "3y", "5y"]
    lookback_str = st.sidebar.selectbox("Lookback", lookback_options, index=2)
    lookback_days = parse_timeframe(lookback_str)
    bt_start = None
    bt_end = None
    period_label = f"Last {lookback_str}"
else:
    default_end = date.today()
    default_start = default_end - timedelta(days=730)
    bt_start_date = st.sidebar.date_input("Start date", value=default_start)
    bt_end_date = st.sidebar.date_input("End date", value=default_end)
    if bt_start_date >= bt_end_date:
        st.sidebar.error("Start must be before end")
        st.stop()
    lookback_days = (date.today() - bt_start_date).days + 300
    bt_start = str(bt_start_date)
    bt_end = str(bt_end_date)
    period_label = f"{bt_start_date} → {bt_end_date}"

st.sidebar.markdown("---")

# ── Strategies ──
st.sidebar.subheader("Strategies")

strategy_choices = get_strategy_choices()  # {key: "icon name"} from unified registry
strategy_keys_all = get_strategy_keys()
strategy_labels_all = [strategy_choices[k] for k in strategy_keys_all]

selected_strategies = st.sidebar.multiselect(
    "Select strategies",
    strategy_labels_all,
    default=strategy_labels_all,
)
# Map back to keys
selected_strategy_keys = [strategy_keys_all[strategy_labels_all.index(s)] for s in selected_strategies]

st.sidebar.markdown("---")
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=10000.0, step=1000.0, format="%.0f")

st.sidebar.markdown("---")
run_clicked = st.sidebar.button("🚀 Run All Backtests", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# Main content
# ═══════════════════════════════════════════════════════════════════
st.title("📊 Batch Strategy Comparison")
st.caption(f"Compare {len(selected_strategies)} strategies × {len(selected_tickers)} tickers — {period_label}")

if not run_clicked:
    st.info("Configure settings in the sidebar and click **🚀 Run All Backtests**.")

    st.markdown(f"""
**Will test {len(selected_tickers)} tickers × {len(selected_strategies)} strategies = {len(selected_tickers) * len(selected_strategies)} backtests**

Estimated time: ~{len(selected_tickers) * len(selected_strategies) // 3} seconds
(each backtest takes ~0.3s, data download is cached per ticker)
    """)
    st.stop()


# ═══════════════════════════════════════════════════════════════════
# Run all backtests
# ═══════════════════════════════════════════════════════════════════

def run_single_backtest(df_chart, ticker, strategy_key, lookback_days, bt_start, bt_end, capital):
    """Run one backtest using the unified run_strategy() dispatcher."""
    try:
        result = run_strategy(
            strategy_key,
            df_chart,
            ticker=ticker,
            chart_interval="1d",
            backtest_days=lookback_days if bt_start is None else None,
            backtest_start_date=bt_start,
            backtest_end_date=bt_end,
            initial_capital=capital,
        )

        stats = result["stats"]
        display_name = get_strategy_display_name(strategy_key)

        # Parse numeric values from formatted strings
        def parse_pct(s):
            try:
                return float(str(s).replace("%", "").replace("+", "").replace(",", "").strip())
            except:
                return np.nan

        def parse_dollar(s):
            try:
                return float(str(s).replace("$", "").replace("+", "").replace(",", "").strip())
            except:
                return np.nan

        return {
            "Ticker": ticker,
            "Strategy": display_name,
            "Strat Return %": parse_pct(stats.get("Strat Return", "0")),
            "B&H Return %": parse_pct(stats.get("B&H Return", "0")),
            "Alpha %": parse_pct(stats.get("Alpha (Strat - B&H)", "0")),
            "Strat MaxDD %": parse_pct(stats.get("Strat Max Drawdown", "0")),
            "B&H MaxDD %": parse_pct(stats.get("B&H Max Drawdown", "0")),
            "Trades": stats.get("Completed Trades", 0),
            "Win Rate": stats.get("Win Rate", "0%"),
            "Time in Mkt": stats.get("Time in Market", "0%"),
            "P&L Adv $": parse_dollar(stats.get("P&L Advantage", "0")),
            "_result": result,
            "_error": None,
        }
    except Exception as e:
        return {
            "Ticker": ticker,
            "Strategy": get_strategy_display_name(strategy_key),
            "Strat Return %": np.nan,
            "B&H Return %": np.nan,
            "Alpha %": np.nan,
            "Strat MaxDD %": np.nan,
            "B&H MaxDD %": np.nan,
            "Trades": 0,
            "Win Rate": "-",
            "Time in Mkt": "-",
            "P&L Adv $": np.nan,
            "_result": None,
            "_error": str(e),
        }


# ── Download data once per ticker, then run all strategies ──
all_rows = []
progress_bar = st.progress(0)
status_text = st.empty()
total_tasks = len(selected_tickers) * len(selected_strategy_keys)
completed = 0

# Cache downloaded data per ticker
ticker_data_cache = {}

for ticker in selected_tickers:
    status_text.text(f"📥 Downloading {ticker}...")

    if ticker not in ticker_data_cache:
        try:
            df_chart = fetch_yahoo_ohlcv(ticker, "1d", lookback_days)
            ticker_data_cache[ticker] = df_chart
        except Exception as e:
            # Mark all strategies as failed for this ticker
            for sk in selected_strategy_keys:
                all_rows.append({
                    "Ticker": ticker,
                    "Strategy": get_strategy_display_name(sk),
                    "Strat Return %": np.nan,
                    "B&H Return %": np.nan,
                    "Alpha %": np.nan,
                    "Strat MaxDD %": np.nan,
                    "B&H MaxDD %": np.nan,
                    "Trades": 0,
                    "Win Rate": "-",
                    "Time in Mkt": "-",
                    "P&L Adv $": np.nan,
                    "_result": None,
                    "_error": f"Data download failed: {e}",
                })
                completed += 1
                progress_bar.progress(completed / total_tasks)
            continue

    df_chart = ticker_data_cache[ticker]

    for sk in selected_strategy_keys:
        status_text.text(f"📊 {ticker} — {get_strategy_display_name(sk)[:30]}...")
        row = run_single_backtest(df_chart, ticker, sk, lookback_days, bt_start, bt_end, initial_capital)
        all_rows.append(row)
        completed += 1
        progress_bar.progress(completed / total_tasks)

progress_bar.empty()
status_text.empty()

# ═══════════════════════════════════════════════════════════════════
# Build results DataFrame
# ═══════════════════════════════════════════════════════════════════

results_df = pd.DataFrame(all_rows)

# Separate internal columns
full_results = {(r["Ticker"], r["Strategy"]): r.get("_result") for r in all_rows}
errors = {(r["Ticker"], r["Strategy"]): r.get("_error") for r in all_rows if r.get("_error")}

# Drop internal columns for display
display_df = results_df.drop(columns=["_result", "_error"], errors="ignore")

st.success(f"✅ Completed {completed} backtests ({len(errors)} errors)")

# ═══════════════════════════════════════════════════════════════════
# Summary stats
# ═══════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🏆 Strategy Leaderboard")

# Average alpha by strategy
if not display_df.empty and "Alpha %" in display_df.columns:
    leaderboard = (
        display_df.groupby("Strategy")["Alpha %"]
        .agg(["mean", "median", "count", lambda x: (x > 0).sum()])
        .rename(columns={"mean": "Avg Alpha %", "median": "Median Alpha %",
                         "count": "Tickers Tested", "<lambda_0>": "Beat B&H Count"})
        .sort_values("Avg Alpha %", ascending=False)
    )
    leaderboard["Beat B&H %"] = (leaderboard["Beat B&H Count"] / leaderboard["Tickers Tested"] * 100).round(1)

    # Color the leaderboard
    def color_alpha(val):
        if pd.isna(val):
            return ""
        if val > 0:
            return "color: #2d6a2d; font-weight: bold"
        elif val < 0:
            return "color: #8b2020; font-weight: bold"
        return ""

    styled_lb = leaderboard[["Avg Alpha %", "Median Alpha %", "Beat B&H %", "Tickers Tested"]].style \
        .map(color_alpha, subset=["Avg Alpha %", "Median Alpha %"]) \
        .format({"Avg Alpha %": "{:+.2f}%", "Median Alpha %": "{:+.2f}%", "Beat B&H %": "{:.1f}%"})

    st.dataframe(styled_lb, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# Full results table
# ═══════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📋 Full Results")

# Filters
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    sort_by = st.selectbox("Sort by", ["Alpha %", "Strat Return %", "Ticker", "Strategy"], index=0)
with col_f2:
    sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)
with col_f3:
    filter_alpha = st.radio("Filter", ["All", "Alpha > 0 only", "Alpha < 0 only"], horizontal=True)

filtered_df = display_df.copy()
if filter_alpha == "Alpha > 0 only":
    filtered_df = filtered_df[filtered_df["Alpha %"] > 0]
elif filter_alpha == "Alpha < 0 only":
    filtered_df = filtered_df[filtered_df["Alpha %"] < 0]

ascending = sort_order == "Ascending"
if sort_by in filtered_df.columns:
    filtered_df = filtered_df.sort_values(sort_by, ascending=ascending, na_position="last")

st.caption(f"Showing {len(filtered_df)} of {len(display_df)} results")

# Color the alpha column
def highlight_row(row):
    alpha = row.get("Alpha %", 0)
    if pd.isna(alpha):
        return [""] * len(row)
    if alpha > 5:
        return ["background-color: rgba(45, 106, 45, 0.2)"] * len(row)
    elif alpha > 0:
        return ["background-color: rgba(45, 106, 45, 0.1)"] * len(row)
    elif alpha < -5:
        return ["background-color: rgba(139, 32, 32, 0.2)"] * len(row)
    elif alpha < 0:
        return ["background-color: rgba(139, 32, 32, 0.1)"] * len(row)
    return [""] * len(row)

format_dict = {
    "Strat Return %": "{:+.2f}%",
    "B&H Return %": "{:+.2f}%",
    "Alpha %": "{:+.2f}%",
    "Strat MaxDD %": "{:.2f}%",
    "B&H MaxDD %": "{:.2f}%",
    "P&L Adv $": "${:+,.2f}",
}

styled_full = filtered_df.style \
    .apply(highlight_row, axis=1) \
    .format(format_dict, na_rep="-")

st.dataframe(styled_full, use_container_width=True, height=600)

# ═══════════════════════════════════════════════════════════════════
# Pivot table: Tickers as rows, Strategies as columns (Alpha %)
# ═══════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📊 Alpha Heatmap (Ticker × Strategy)")

if not display_df.empty:
    pivot = display_df.pivot_table(
        index="Ticker",
        columns="Strategy",
        values="Alpha %",
        aggfunc="first",
    )

    # Color the heatmap
    def color_cell(val):
        if pd.isna(val):
            return "background-color: #333"
        if val > 10:
            return "background-color: #1a5c1a; color: white"
        elif val > 0:
            return "background-color: #2d6a2d; color: white"
        elif val > -10:
            return "background-color: #6a2d2d; color: white"
        else:
            return "background-color: #8b2020; color: white"

    styled_pivot = pivot.style \
        .map(color_cell) \
        .format("{:+.1f}%", na_rep="-")

    st.dataframe(styled_pivot, use_container_width=True, height=600)


# ═══════════════════════════════════════════════════════════════════
# Drill-down: click to see trade details
# ═══════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🔍 Trade Detail Drill-Down")

dd_col1, dd_col2 = st.columns(2)
with dd_col1:
    dd_ticker = st.selectbox("Ticker", selected_tickers, key="dd_ticker")
with dd_col2:
    dd_strategy = st.selectbox("Strategy", selected_strategies, key="dd_strategy")

dd_key = (dd_ticker, dd_strategy)
dd_result = full_results.get(dd_key)
dd_error = errors.get(dd_key)

if dd_error:
    st.error(f"Error: {dd_error}")
elif dd_result is None:
    st.info("No result available for this combination.")
else:
    stats = dd_result["stats"]
    trades_df = dd_result["trades"]
    equity_df = dd_result["equity_curve"]
    sig_df = dd_result["signals"]

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Strategy Return", stats.get("Strat Return", "N/A"))
    m2.metric("Buy & Hold Return", stats.get("B&H Return", "N/A"))
    m3.metric("Alpha", stats.get("Alpha (Strat - B&H)", "N/A"))
    m4.metric("Trades", stats.get("Completed Trades", 0))

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Win Rate", stats.get("Win Rate", "N/A"))
    m6.metric("Strat Max DD", stats.get("Strat Max Drawdown", "N/A"))
    m7.metric("B&H Max DD", stats.get("B&H Max Drawdown", "N/A"))
    m8.metric("Time in Market", stats.get("Time in Market", "N/A"))

    # Equity curve
    if not equity_df.empty and "Bar Time" in equity_df.columns:
        st.markdown("#### 📈 Equity Curve")
        chart_data = equity_df[["Bar Time", "Equity", "BH_Equity"]].copy()
        chart_data = chart_data.set_index("Bar Time")
        chart_data.columns = ["Strategy", "Buy & Hold"]
        st.line_chart(chart_data, use_container_width=True)

    # Trade log
    if not trades_df.empty:
        st.markdown("#### 📋 Trade Log")

        trade_display_cols = ["Bar Time", "Action", "Price", "Shares", "Cash", "Total"]
        if "PnL" in trades_df.columns:
            trade_display_cols.extend(["PnL", "PnL%"])
        trade_display_cols.append("Signal")
        if "Macro_Regime" in trades_df.columns:
            trade_display_cols.append("Macro_Regime")
        available = [c for c in trade_display_cols if c in trades_df.columns]

        # Try Token Qty as fallback for Shares
        if "Shares" not in trades_df.columns and "Token Qty" in trades_df.columns:
            available = [("Token Qty" if c == "Shares" else c) for c in available]

        def highlight_trade(row):
            action = row.get("Action", "")
            if action == "BUY":
                return ["background-color: #2d6a2d; color: white"] * len(row)
            elif action == "SELL":
                return ["background-color: #8b2020; color: white"] * len(row)
            return [""] * len(row)

        final_cols = [c for c in available if c in trades_df.columns]
        if final_cols:
            styled_trades = trades_df[final_cols].style.apply(highlight_trade, axis=1)
            st.dataframe(styled_trades, use_container_width=True, hide_index=True)
    else:
        st.info("No trades executed.")

    # Signal table (collapsible)
    with st.expander("🔍 Full Signals Table", expanded=False):
        if not sig_df.empty:
            st.dataframe(sig_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# Download
# ═══════════════════════════════════════════════════════════════════
st.markdown("---")
if not display_df.empty:
    csv = display_df.to_csv(index=False)
    st.download_button(
        "💾 Download Full Results CSV",
        csv,
        f"strategy_comparison_{period_label.replace(' ', '_')}.csv",
        "text/csv",
    )