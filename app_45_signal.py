#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app_45_signal.py - Streamlit UI for 45° Trend Analysis with Trading Signals

Usage:
    streamlit run app_45_signal.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Setup paths
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Import shared scanner functions
from data.scanner_45 import (
    scan_universe,
    build_universe,
    DEFAULT_CONFIG,
)

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="45° Signal Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# CONSTANTS
# -----------------------
OUTPUTS_DIR = APP_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

CSV_FULL = OUTPUTS_DIR / "45_signal_full.csv"
CSV_TOP = OUTPUTS_DIR / "45_signal_top.csv"
CSV_MYLIST = OUTPUTS_DIR / "45_signal_mylist.csv"


# -----------------------
# HELPER FUNCTIONS FOR UI
# -----------------------
def style_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply color coding to dataframe."""
    if df.empty:
        return df
    
    def color_signal(val):
        if val == "BUY":
            return "background-color: #90EE90"
        elif val == "EXIT":
            return "background-color: #FFB6C6"
        elif val == "HOLD":
            return "background-color: #FFE5A0"
        return ""
    
    def color_score(val):
        if pd.isna(val):
            return ""
        if val >= 80:
            return "background-color: #90EE90"
        elif val >= 70:
            return "background-color: #B0E0B0"
        elif val >= 60:
            return "background-color: #FFE5A0"
        return ""

    styled = df.style.map(color_signal, subset=["SIGNAL-Super-MOST-ADXR"])

    # Color all score columns that exist in df
    score_cols = [c for c in ["Score_30", "Score_60", "Score_90", "Score_120", "Score_Weighted"] if c in df.columns]
    if score_cols:
        styled = styled.map(color_score, subset=score_cols)
    
    # Format numbers
    format_dict = {
        "Price": "${:.2f}",
        "Score_30": "{:.0f}",
        "Score_60": "{:.0f}",
        "Score_90": "{:.0f}",
        "Score_120": "{:.0f}",
        "Score_Weighted": "{:.0f}",
        "Ret_63d_%": "{:.1f}",
        "RS_63d_vs_SPY_%": "{:.1f}",
        "MaxDD_120d_%": "{:.1f}",
        "Dist_to_21EMA_%": "{:.1f}",
        "Dist_to_50SMA_%": "{:.1f}",
    }
    
    return styled.format(format_dict, na_rep="-")


def safe_cols(df: pd.DataFrame, wanted: list) -> list:
    """Return only columns that exist in df — prevents KeyError on old CSVs."""
    return [c for c in wanted if c in df.columns]


# -----------------------
# SIDEBAR
# -----------------------
st.sidebar.title("📈 45° Signal Scanner")
st.sidebar.markdown("---")

st.sidebar.header("⚙️ Configuration")

scan_period = st.sidebar.selectbox(
    "Scan Period",
    ["360d", "420d", "540d"],
    index=1
)

min_dollar_vol = st.sidebar.number_input(
    "Min Dollar Volume",
    min_value=1_000_000,
    max_value=100_000_000,
    value=20_000_000,
    step=5_000_000
)

earnings_window = st.sidebar.number_input(
    "Earnings Window (days)",
    min_value=1,
    max_value=30,
    value=7
)

pass_score = st.sidebar.number_input(
    "Pass Score",
    min_value=50,
    max_value=100,
    value=70
)

st.sidebar.markdown("---")
st.sidebar.header("🎯 Universe Selection")

# Universe checkboxes - DEFAULT TO CHECKED
scan_spx = st.sidebar.checkbox("📊 S&P 500", value=True, help="Scan all ~500 S&P 500 stocks")
scan_ndx = st.sidebar.checkbox("💻 Nasdaq-100", value=True, help="Scan all ~100 Nasdaq stocks")

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Custom Tickers")

custom_tickers_input = st.sidebar.text_area(
    "Add custom tickers (one per line)",
    value="",  # Empty by default since we have SPX/NDX
    height=100
)

st.sidebar.subheader("📋 Watchlist")
watchlist_input = st.sidebar.text_area(
    "Watchlist (track separately)",
    value="TSLA\nNVDA\nAAPL",
    height=100
)

# -----------------------
# MAIN CONTENT
# -----------------------
st.title("📈 45° Trend Scanner with Trading Signals")
st.markdown("**Combines 45° trend quality with Supertrend-MOST-ADXR signals**")

tabs = st.tabs(["🔍 Scan", "📊 Full", "⭐ Top", "📋 Watch", "📖 Guide"])

# -----------------------
# TAB: Scan
# -----------------------
with tabs[0]:
    st.header("Run New Scan")
    
    # Show what will be scanned (preview)
    preview_count = 0
    
    if scan_spx:
        st.write("- 📊 S&P 500 (~500 tickers)")
        preview_count += 500
    if scan_ndx:
        st.write("- 💻 Nasdaq-100 (~100 tickers)")
        preview_count += 100
    
    custom_count = 0
    if custom_tickers_input.strip():
        custom_count = len([t.strip() for t in custom_tickers_input.split("\n") if t.strip()])
        st.write(f"- 📝 {custom_count} custom tickers")
        preview_count += custom_count
    
    if preview_count == 0:
        st.warning("⚠️ No tickers selected! Check a universe or add custom tickers.")
    else:
        st.write(f"**Estimated total: ~{preview_count} tickers**")
        
        # Estimate scan time
        if preview_count > 100:
            est_minutes = preview_count // 50
            st.info(f"⏱️ Estimated time: {est_minutes}-{est_minutes+5} minutes")
    
    if st.button("🚀 Run Scan", type="primary", disabled=preview_count == 0):
        # Build ticker list when button is clicked
        ticker_list = []
        
        # Add universe tickers
        if scan_spx or scan_ndx:
            universe_sources = []
            if scan_spx:
                universe_sources.append("SPX")
            if scan_ndx:
                universe_sources.append("NDX")
            
            with st.spinner(f"Fetching tickers from {'S&P 500' if scan_spx else ''}{' & ' if scan_spx and scan_ndx else ''}{'Nasdaq-100' if scan_ndx else ''}..."):
                try:
                    universe_tickers = build_universe(tuple(universe_sources), verbose=True)
                    ticker_list.extend(universe_tickers)
                    if universe_tickers:
                        st.success(f"✅ Loaded {len(universe_tickers)} tickers from indices")
                    else:
                        st.error("❌ Failed to fetch universe - check internet connection")
                except Exception as e:
                    st.error(f"❌ Error fetching universe: {e}")
        
        # Add custom tickers
        if custom_tickers_input.strip():
            raw = custom_tickers_input.replace(",", "\n")
            custom_tickers = [t.strip().upper() for t in raw.split("\n") if t.strip()]
            ticker_list.extend(custom_tickers)
        
        ticker_list = sorted(set(ticker_list))
        
        st.write(f"**Scanning {len(ticker_list)} unique tickers...**")
        
        if len(ticker_list) == 0:
            st.error("No tickers to scan!")
        else:
            # Build config
            config = DEFAULT_CONFIG.copy()
            config.update({
                "scan_period": scan_period,
                "min_dollar_volume": min_dollar_vol,
                "earnings_window_days": earnings_window,
            })
            
            # Progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total):
                status_text.text(f"Scanning {ticker_list[current-1]} ({current}/{total})...")
                progress_bar.progress(current / total)
            
            # Run scan
            with st.spinner("Scanning..."):
                df_results = scan_universe(
                    ticker_list,
                    config=config,
                    verbose=False,
                    progress_callback=update_progress
                )
            
            progress_bar.empty()
            status_text.empty()
            
            if df_results.empty:
                st.error("No tickers passed filters!")
            else:
                # Save results
                df_results.to_csv(CSV_FULL, index=False)

                # Sort by Score_Weighted for top list
                df_results_sorted = df_results.sort_values("Score_Weighted", ascending=False)
                top_df = df_results_sorted[df_results_sorted["Score_Weighted"] >= pass_score]
                top_df.to_csv(CSV_TOP, index=False)

                watchlist = [t.strip().upper() for t in watchlist_input.split("\n") if t.strip()]
                watch_df = df_results[df_results["Ticker"].isin(watchlist)]
                watch_df.to_csv(CSV_MYLIST, index=False)

                st.success(f"✅ Scanned {len(df_results)} tickers, {len(top_df)} scored >= {pass_score} (weighted)")

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Passed Filters", len(df_results))
                col2.metric("BUY Signals", len(df_results[df_results["SIGNAL-Super-MOST-ADXR"] == "BUY"]))
                col3.metric("Avg Score (Weighted)", f"{df_results['Score_Weighted'].mean():.0f}")

                # Preview top 10 by weighted score
                st.subheader("Top 10 Results (by Score_Weighted)")
                wanted = ["Ticker", "Score_Weighted", "Score_30", "Score_60", "Score_90", "Score_120",
                          "SIGNAL-Super-MOST-ADXR", "Earnings_Alert",
                          "Market_Cap", "Price", "RS_63d_vs_SPY_%", "Dist_to_21EMA_%"]
                st.dataframe(style_dataframe(df_results_sorted[safe_cols(df_results_sorted, wanted)].head(10)), width="stretch")

# -----------------------
# TAB: Full Results
# -----------------------
with tabs[1]:
    st.header("Full Results")
    
    if CSV_FULL.exists():
        df = pd.read_csv(CSV_FULL)
        
        col1, col2 = st.columns(2)
        min_score = col1.slider("Min Score (Weighted)", 0, 100, 0)
        signals = col2.multiselect("Signals", ["BUY", "HOLD", "EXIT"], ["BUY", "HOLD", "EXIT"])

        filtered = df[df["Score_Weighted"] >= min_score] if "Score_Weighted" in df.columns else df
        if signals:
            filtered = filtered[filtered["SIGNAL-Super-MOST-ADXR"].isin(signals)]
        
        # Sort by Score_Weighted descending
        filtered = filtered.sort_values("Score_Weighted", ascending=False) if "Score_Weighted" in filtered.columns else filtered

        st.write(f"**Showing {len(filtered)} of {len(df)} tickers**")

        wanted = ["Ticker", "Score_Weighted", "Score_30", "Score_60", "Score_90", "Score_120",
                  "SIGNAL-Super-MOST-ADXR", "Earnings_Alert",
                  "Market_Cap", "Price", "RS_63d_vs_SPY_%", "Dist_to_21EMA_%", "Buy_Hint"]
        st.dataframe(style_dataframe(filtered[safe_cols(filtered, wanted)]), width="stretch", height=600)
    else:
        st.info("No results yet. Run a scan first!")

# -----------------------
# TAB: Top Scorers
# -----------------------
with tabs[2]:
    st.header(f"Top Scorers (Score >= {pass_score})")
    
    if CSV_TOP.exists():
        df = pd.read_csv(CSV_TOP)
        
        if df.empty:
            st.warning(f"No tickers scored >= {pass_score}")
        else:
            col1, col2 = st.columns(2)
            col1.metric("Count", len(df))
            col2.metric("BUY Signals", len(df[df["SIGNAL-Super-MOST-ADXR"] == "BUY"]))

            wanted = ["Ticker", "Score_Weighted", "Score_30", "Score_60", "Score_90", "Score_120",
                      "SIGNAL-Super-MOST-ADXR", "Earnings_Alert",
                      "Market_Cap", "Price", "RS_63d_vs_SPY_%", "Dist_to_21EMA_%", "Buy_Hint"]
            st.dataframe(style_dataframe(df[safe_cols(df, wanted)]), width="stretch", height=600)
    else:
        st.info("No results yet. Run a scan first!")

# -----------------------
# TAB: Watchlist
# -----------------------
with tabs[3]:
    st.header("Watchlist")
    
    if CSV_MYLIST.exists():
        df = pd.read_csv(CSV_MYLIST)
        
        if df.empty:
            st.warning("No watchlist tickers in results")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("BUY", len(df[df["SIGNAL-Super-MOST-ADXR"] == "BUY"]))
            col2.metric("HOLD", len(df[df["SIGNAL-Super-MOST-ADXR"] == "HOLD"]))
            col3.metric("EXIT", len(df[df["SIGNAL-Super-MOST-ADXR"] == "EXIT"]))
            
            st.dataframe(style_dataframe(df), width="stretch", height=600)
    else:
        st.info("No watchlist data. Run a scan first!")

# -----------------------
# TAB: Guide
# -----------------------
with tabs[4]:
    st.header("📖 Quick Guide")
    
    st.markdown("""
    ## Score (0-100)
    - **80+**: Excellent trend
    - **70-79**: Very good
    - **60-69**: Good
    - **<60**: Weak
    
    ## SIGNAL-Super-MOST-ADXR
    - 🟢 **BUY**: Enter position
    - 🟡 **HOLD**: Wait or maintain
    - 🔴 **EXIT**: Sell or avoid
    
    ## Ideal Setup
    ```
    Score: 70+
    Signal: BUY
    Earnings: (blank)
    Dist to 21EMA: -1% to +3%
    RS vs SPY: Positive
    ```
    
    ## Universe Options
    - **S&P 500**: ~500 large-cap stocks
    - **Nasdaq-100**: ~100 tech/growth stocks
    - **Custom**: Add any tickers manually
    
    ## Tips
    - High score + BUY signal = best entries
    - Avoid earnings alerts (high volatility)
    - Don't chase extended stocks (>5% from 21-EMA)
    - Check watchlist tab for exit signals on holdings
    - Full market scan takes ~12-15 minutes
    """)

st.sidebar.markdown("---")
st.sidebar.caption("v3.0 - Zero code duplication")
st.sidebar.caption("Uses shared scanner_45 module")