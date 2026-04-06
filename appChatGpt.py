import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(page_title="SRMTF Backtest", layout="wide")

# =========================================
# DATA
# =========================================
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # yfinance sometimes returns MultiIndex columns like ('Close', 'NVDA')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).lower() for c in df.columns]

    required = ["open", "high", "low", "close", "volume"]
    df = df[required].dropna().copy()
    return df

# =========================================
# PIVOTS
# =========================================
def get_pivots(df, prd):
    pivots = []
    highs = df["high"].values
    lows = df["low"].values

    for i in range(prd, len(df) - prd):
        if highs[i] == max(highs[i - prd:i + prd + 1]):
            pivots.append(highs[i])
        if lows[i] == min(lows[i - prd:i + prd + 1]):
            pivots.append(lows[i])

    return pivots

# =========================================
# BUILD ZONES
# =========================================
def build_zones(df, prd=5, loopback=250, width_pct=6, min_strength=2, max_zones=6):
    df = df.tail(loopback).copy()
    pivots = get_pivots(df, prd)

    if len(pivots) == 0:
        return []

    highest_pivot = max(pivots)
    lowest_pivot = min(pivots)
    max_width = (highest_pivot - lowest_pivot) * width_pct / 100.0

    candidate_zones = []

    for pivot in pivots:
        z_low = pivot
        z_high = pivot
        strength = 0

        for pivot2 in pivots:
            width = max(abs(pivot2 - z_low), abs(pivot2 - z_high))
            if width <= max_width:
                z_low = min(z_low, pivot2)
                z_high = max(z_high, pivot2)
                strength += 1

        if strength >= min_strength:
            candidate_zones.append((z_low, z_high, strength))

    # remove near-duplicate zones
    deduped = []
    for low, high, strength in sorted(candidate_zones, key=lambda x: x[2], reverse=True):
        duplicate = False
        for e_low, e_high, _ in deduped:
            if abs(low - e_low) < 1e-9 and abs(high - e_high) < 1e-9:
                duplicate = True
                break
        if not duplicate:
            deduped.append((low, high, strength))

    deduped = sorted(deduped, key=lambda x: x[2], reverse=True)
    return deduped[:max_zones]

# =========================================
# CLASSIFY ZONES RELATIVE TO PRICE
# =========================================
def classify_zones(zones, current_price):
    out = []
    for low, high, strength in zones:
        if low > current_price and high > current_price:
            zone_type = "resistance"
        elif low < current_price and high < current_price:
            zone_type = "support"
        else:
            zone_type = "in_channel"
        out.append(
            {
                "low": float(low),
                "high": float(high),
                "strength": int(strength),
                "zone_type": zone_type,
            }
        )
    return out

# =========================================
# BACKTEST
# Buy = break above any zone high while price not in channel
# Sell = break below any zone low while price not in channel
# Starts with initial capital invested on first backtest date
# =========================================
def backtest(
    df,
    initial_capital,
    prd=5,
    loopback=250,
    width_pct=6,
    min_strength=2,
    max_zones=6,
):
    start_index = max(loopback + prd * 2, 260)
    if len(df) <= start_index:
        raise ValueError("Not enough data for the chosen settings.")

    price0 = float(df["close"].iloc[start_index])

    # User asked: take $10,000 (or chosen capital) as already invested to begin on 1st date
    shares = initial_capital / price0
    cash = 0.0
    position = 1

    trades = [
        {
            "type": "BUY_INITIAL",
            "date": df.index[start_index],
            "price": price0,
            "shares": shares,
        }
    ]

    equity_rows = []
    signal_rows = []

    for i in range(start_index, len(df)):
        slice_df = df.iloc[: i + 1].copy()
        current_price = float(df["close"].iloc[i])
        prev_price = float(df["close"].iloc[i - 1]) if i > 0 else current_price

        zones_raw = build_zones(
            slice_df,
            prd=prd,
            loopback=loopback,
            width_pct=width_pct,
            min_strength=min_strength,
            max_zones=max_zones,
        )
        zones = classify_zones(zones_raw, current_price)

        in_channel = False
        for z in zones:
            if z["low"] <= current_price <= z["high"]:
                in_channel = True
                break

        buy_signal = False
        sell_signal = False
        signal_zone_low = np.nan
        signal_zone_high = np.nan

        if not in_channel:
            for z in zones:
                if prev_price <= z["high"] and current_price > z["high"]:
                    buy_signal = True
                    signal_zone_low = z["low"]
                    signal_zone_high = z["high"]
                    break

            for z in zones:
                if prev_price >= z["low"] and current_price < z["low"]:
                    sell_signal = True
                    signal_zone_low = z["low"]
                    signal_zone_high = z["high"]
                    break

        # Execute signals
        if position == 0 and buy_signal:
            shares = cash / current_price
            cash = 0.0
            position = 1
            trades.append(
                {
                    "type": "BUY",
                    "date": df.index[i],
                    "price": current_price,
                    "shares": shares,
                }
            )

        elif position == 1 and sell_signal:
            cash = shares * current_price
            trades.append(
                {
                    "type": "SELL",
                    "date": df.index[i],
                    "price": current_price,
                    "shares": shares,
                }
            )
            shares = 0.0
            position = 0

        equity = cash if position == 0 else shares * current_price

        equity_rows.append(
            {
                "date": df.index[i],
                "equity": equity,
                "close": current_price,
                "position": position,
            }
        )

        signal_rows.append(
            {
                "date": df.index[i],
                "close": current_price,
                "buy_signal": buy_signal,
                "sell_signal": sell_signal,
                "in_channel": in_channel,
                "signal_zone_low": signal_zone_low,
                "signal_zone_high": signal_zone_high,
            }
        )

    equity_df = pd.DataFrame(equity_rows).set_index("date")
    signals_df = pd.DataFrame(signal_rows).set_index("date")
    trades_df = pd.DataFrame(trades)

    return trades_df, equity_df, signals_df, start_index

# =========================================
# PERFORMANCE STATS
# =========================================
def compute_stats(equity_df, initial_capital):
    final_equity = float(equity_df["equity"].iloc[-1])
    strategy_return_pct = (final_equity / initial_capital - 1.0) * 100.0

    running_max = equity_df["equity"].cummax()
    drawdown = equity_df["equity"] / running_max - 1.0
    max_drawdown_pct = float(drawdown.min()) * 100.0

    return {
        "final_equity": final_equity,
        "strategy_return_pct": strategy_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
    }

def compute_buy_hold(df, start_index, initial_capital):
    start_price = float(df["close"].iloc[start_index])
    end_price = float(df["close"].iloc[-1])

    bh_shares = initial_capital / start_price
    bh_final_value = bh_shares * end_price
    bh_return_pct = (bh_final_value / initial_capital - 1.0) * 100.0

    return {
        "start_price": start_price,
        "end_price": end_price,
        "bh_final_value": bh_final_value,
        "bh_return_pct": bh_return_pct,
    }

# =========================================
# PLOTS
# =========================================
def plot_price_and_signals(df, signals_df, zones, lookback=220):
    plot_df = df.tail(lookback).copy()
    plot_signals = signals_df.loc[signals_df.index.intersection(plot_df.index)].copy()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(plot_df.index, plot_df["close"], label="Close")

    # draw latest zones
    for z in zones:
        if z["zone_type"] == "resistance":
            color = "red"
        elif z["zone_type"] == "support":
            color = "green"
        else:
            color = "gray"

        ax.axhspan(z["low"], z["high"], alpha=0.18, color=color)

    buys = plot_signals[plot_signals["buy_signal"]]
    sells = plot_signals[plot_signals["sell_signal"]]

    if not buys.empty:
        ax.scatter(buys.index, buys["close"], marker="^", s=80, label="Buy Signal", zorder=5)
    if not sells.empty:
        ax.scatter(sells.index, sells["close"], marker="v", s=80, label="Sell Signal", zorder=5)

    ax.set_title("Price with Latest SR Zones and Buy/Sell Signals")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return fig

def plot_equity_vs_buyhold(equity_df, buy_hold_series):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(equity_df.index, equity_df["equity"], label="Strategy Equity")
    ax.plot(buy_hold_series.index, buy_hold_series.values, label="Buy & Hold Equity", linestyle="--")
    ax.set_title("Strategy vs Buy & Hold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return fig

# =========================================
# STREAMLIT UI
# =========================================
st.title("SRMTF Strategy Backtest")

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker", "NVDA")
    start_date = st.text_input("Start Date", "2020-01-01")
    end_date = st.text_input("End Date (optional)", "")
    initial_capital = st.number_input("Initial Capital ($)", min_value=1000, value=10000, step=1000)

    st.subheader("SRMTF Settings")
    prd = st.number_input("Pivot Period", min_value=1, max_value=30, value=5)
    loopback = st.number_input("Loopback Period", min_value=50, max_value=400, value=250)
    width_pct = st.number_input("Maximum Channel Width %", min_value=1.0, max_value=15.0, value=6.0, step=0.5)
    min_strength = st.number_input("Minimum Strength", min_value=1, max_value=20, value=2)
    max_zones = st.number_input("Maximum Number of S/R", min_value=1, max_value=10, value=6)

    run_button = st.button("Run Backtest", type="primary")

if run_button:
    try:
        df = load_data(ticker, start_date, end_date if end_date.strip() else None)

        trades_df, equity_df, signals_df, start_index = backtest(
            df=df,
            initial_capital=float(initial_capital),
            prd=int(prd),
            loopback=int(loopback),
            width_pct=float(width_pct),
            min_strength=int(min_strength),
            max_zones=int(max_zones),
        )

        latest_price = float(df["close"].iloc[-1])
        latest_zones_raw = build_zones(
            df=df,
            prd=int(prd),
            loopback=int(loopback),
            width_pct=float(width_pct),
            min_strength=int(min_strength),
            max_zones=int(max_zones),
        )
        latest_zones = classify_zones(latest_zones_raw, latest_price)

        stats = compute_stats(equity_df, float(initial_capital))
        bh = compute_buy_hold(df, start_index, float(initial_capital))

        # buy & hold equity line aligned with backtest window
        bh_start_price = float(df["close"].iloc[start_index])
        bh_shares = float(initial_capital) / bh_start_price
        bh_prices = df["close"].iloc[start_index:].copy()
        buy_hold_series = bh_prices * bh_shares

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Initial Capital ($)", f"{float(initial_capital):,.2f}")
        c2.metric("Strategy Return %", f"{stats['strategy_return_pct']:.2f}%")
        c3.metric("Buy & Hold Return %", f"{bh['bh_return_pct']:.2f}%")
        c4.metric("Final Equity ($)", f"{stats['final_equity']:,.2f}")
        c5.metric("Buy & Hold Final ($)", f"{bh['bh_final_value']:,.2f}")

        d1, d2 = st.columns([2, 1])

        with d1:
            st.pyplot(plot_price_and_signals(df, signals_df, latest_zones))
            st.pyplot(plot_equity_vs_buyhold(equity_df, buy_hold_series))

        with d2:
            st.subheader("Latest SR Zones")
            if latest_zones:
                zones_df = pd.DataFrame(latest_zones)
                zones_df = zones_df[["zone_type", "low", "high", "strength"]]
                st.dataframe(zones_df, use_container_width=True)
            else:
                st.info("No zones found.")

            st.subheader("Recent Signals")
            st.dataframe(
                signals_df[["close", "buy_signal", "sell_signal", "in_channel", "signal_zone_low", "signal_zone_high"]].tail(20),
                use_container_width=True,
            )

        st.subheader("Trades")
        if trades_df.empty:
            st.info("No trades.")
        else:
            st.dataframe(trades_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Set inputs in the sidebar and click Run Backtest.")