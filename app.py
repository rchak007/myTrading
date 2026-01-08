# app.py
from __future__ import annotations

import schwabdev

import os
import numpy as np
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
load_dotenv()


from data.stocks import build_stocks_signals_table
from data.crypto import (
    build_crypto_signals_table,
    fetch_coingecko_global,
    fetch_total_mcap_history_coingecko,
    fetch_total_mcap_history_coinmarketcap,
    compute_total_vs_200ma,
    fetch_altcoin_season_index,
)
from data.breadth import fetch_vix_value, fetch_spy_vs_200ma, breadth_proxy_from_spy
from core.utils import fmt_usd_compact

# First: prove where Streamlit is running + where it thinks outputs is
import os
from pathlib import Path
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = APP_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


from datetime import datetime
from data.schwab.token_store import load_tokens_db
from data.schwab.token_sync import sync_db_to_local_multi, sync_local_to_db_multi

# TOKEN_PATH = OUTPUTS_DIR / "schwab_tokens.json"
# TOKEN_PATH = APP_DIR / "tokens.json"

TOKEN_PATHS = [
    APP_DIR / "tokens.json",
    APP_DIR / "data" / "schwab" / "tokens.json",
]
USER_ID = "main"

with st.sidebar:
    st.subheader("🔐 Schwab Token (DB)")

    t = load_tokens_db(USER_ID)
    if t:
        st.write("DB token expires:",
                 datetime.fromtimestamp(t["expires_at"]).strftime("%Y-%m-%d %H:%M:%S"))
        if st.button("⬇️ Sync DB → local (for schwabdev)"):
            ok = sync_db_to_local_multi(USER_ID, TOKEN_PATHS)
            st.success("Synced DB → local ✅" if ok else "No DB token found ❌")
        if st.button("⬆️ Sync local → DB (after login/refresh)"):
            ok = sync_local_to_db_multi(USER_ID, TOKEN_PATHS)
            st.success("Synced local → DB ✅" if ok else "No valid local token ❌")
    else:
        st.warning("No token in DB yet.")
        st.caption("After you login once, click 'Sync local → DB'.")


STOCKS_CSV = OUTPUTS_DIR / "supertrend_stocks_1d.csv"
CRYPTO_CSV = OUTPUTS_DIR / "supertrend_crypto_4h.csv"

# st.caption(f"cwd: {os.getcwd()}")
# st.caption(f"app.py dir: {APP_DIR}")
# st.caption(f"outputs dir: {OUTPUTS_DIR}")
# st.caption(f"stocks csv path: {STOCKS_CSV}")
# st.caption(f"crypto csv path: {CRYPTO_CSV}")


try:
    test_path = OUTPUTS_DIR / "_write_test.txt"
    test_path.write_text("ok\n", encoding="utf-8")
    # st.success(f"Write test OK: {test_path}")
except Exception as e:
    st.error("Write test FAILED — outputs folder is not writable:")
    st.exception(e)



# -----------------------
# SETTINGS / PARAMETERS
# -----------------------
# USED
ATR_PERIOD = 10
ATR_MULTIPLIER = 3.0
RSI_PERIOD = 14
VOL_LOOKBACK = 20
VOL_MULTIPLIER = 1.2
RSI_BUY_THRESHOLD = 50.0

ADXR_LEN = 14
ADXR_LENX = 14
ADXR_LOW_THRESHOLD = 20.0
ADXR_FLAT_EPS = 1e-6

# NOT USED YET (kept for future expansion)
# - Schwab integration
# - wallet balances
# - geo / time session filters
PLACEHOLDER_FUTURE = True

# OUTPUTS_DIR = "outputs"
# os.makedirs(OUTPUTS_DIR, exist_ok=True)


# Your real lists (keep yours here)
# STOCK_TICKERS = [
#     "AAPL","AAOI","ALAB","AMD","AMZN","APH","APLD","APP","ARKB","ARM","ASML","AVGO",
#     "BE","BMNR","CEG","CFG","COIN","COHR","CORZ","CRDO","CRVW","CRWD","ETHA","GEV","GOOG",
#     "HODL","HOOD","IBIT","IDR","INOD","IONQ","IREN","LEU","LITE","LRCX","LTBR",
#     "META","MP","MSFT","MSTR","MSTX","MU","NET","NPPTF","NVDA","OKLO","ORCL","PLTR",
#     "QBTS","QUBT","RGTI","RDDT","SOFI","SSK","STKE","VRT","TER","TSLA","TSM","UPXI",
# ]
STOCK_TICKERS = [
    "AAPL","AAOI","ALAB","AMD","AMZN","APH","APLD","APP","ARKB", "ARM" , "ASML", "AVGO",
    "BE", "BMNR", "BWXT", "CEG",  "CFG","COIN","COHR","CORZ","CRDO","CRVW", "CRWD", "ETHA","GEV","GOOG",
    "HODL","HOOD","IBIT","IDR","INOD","IONQ","IREN","LEU","LITE","LRCX","LTBR",
    "META","MNST", "MP","MSFT","MSTR","MSTX","MU","NET","NPPTF","NVDA","OKLO", "ORCL",  "PLTR",
    "QBTS","QUBT","RGTI","RDDT","SOFI", "SMR", "SSK","STKE", "STRC",
    "TER","TSLA","TSM","UPXI","VRT", "WDC"  
]

CRYPTO_TICKERS = [
    "BTC-USD","ETH-USD","SOL-USD","HYPE32196-USD", "SUI20947-USD", "LINK-USD","DOGE-USD", "ONDO-USD","BNB-USD",

    "AAVE-USD" , "ADA-USD" , "AIXBT-USD", "AKT-USD", "ASTER36341-USD", "AUKI-USD", "AURORA14803-USD", "blue-usd" , "cetus-usd" ,"cookie31838-usd" ,"CRV-USD",
    "DRIFT31278-USD", "ELIZAOS-USD",  "elon-usd" ,"ENA-USD","ENS-USD",
    "fluid-usd", "FAI34330-USD", "griffain-USD",
    "HNT-USD","JTO-USD", "JUP29210-USD", "KMNO-USD", "LFNTY-USD", 
    "MON30495-USD", "navx-USD" , "NEAR-USD", "ORCA-USD" , "ore32782-USD",
    "pippin-usd" , "PNK-USD", "PYTH-USD","RAY-USD","RENDER-USD",
     "SUAI-USD", "suins-usd",  "TAI20605-USD",
    "VIRTUAL-USD", "W-USD" , "WAL36119-USD", "WLD-USD", "wlfi33251-usd",  "XBG-USD" , "XRP-USD", "ZEREBRO-USD" , "ZEUS30391-USD", "zk24091-USD"
]



CRYPTO_NOT_FOUND_YAHOO = {
    "LVL": "https://www.geckoterminal.com/solana/pools/GiRyo4r3kREH8oRCe9GoJJARZuGo4ksto6xXvUok4wdd",
    "A0X": "https://www.geckoterminal.com/base/pools/0xfd100e192d0ff7a284f31a93b367d582666e406b",
    "GAME": "https://www.geckoterminal.com/base/pools/0xd418dfe7670c21f682e041f34250c114db5d7789",
    "AoT": "https://www.geckoterminal.com/base/pools/0x6e7a1875810afb6074953094c35a101e1cc7aee010fb1372d8f41b0fbe92d83c",   
}

# AoT 0xcc4adb618253ed0d4d8a188fb901d70c54735e03


@st.cache_data(ttl=120)
def fetch_schwab_stock_holdings() -> pd.DataFrame:
    """
    Returns a DataFrame: Ticker, QTY, VALUE
    Aggregated across all accounts, equities only.
    """
    app_key = os.getenv("app_key")
    app_secret = os.getenv("app_secret")
    callback_url = os.getenv("callback_url")

    if not app_key or not app_secret or not callback_url:
        # If env missing, return empty so UI still works
        return pd.DataFrame(columns=["Ticker", "QTY", "VALUE"])

    # Ensure schwabdev can find tokens locally
    sync_db_to_local_multi(USER_ID, TOKEN_PATHS)

    client = schwabdev.Client(app_key, app_secret, callback_url)

    try:
        resp = client.account_details_all(fields="positions")
    except TypeError:
        resp = client.account_details_all(fields=["positions"])

    data = resp.json()
    rows = []

    for acct in data:
        sa = acct.get("securitiesAccount", {})
        for pos in sa.get("positions", []) or []:
            inst = pos.get("instrument", {}) or {}
            symbol = inst.get("symbol")
            asset_type = inst.get("assetType")

            # keep only stock-like holdings (equities)
            if asset_type not in ("EQUITY",):
                continue

            long_qty = float(pos.get("longQuantity") or 0.0)
            short_qty = float(pos.get("shortQuantity") or 0.0)
            qty = long_qty - short_qty

            mkt_value = float(pos.get("marketValue") or 0.0)

            if symbol:
                rows.append({"Ticker": symbol, "QTY": qty, "VALUE": mkt_value})

    if not rows:
        return pd.DataFrame(columns=["Ticker", "QTY", "VALUE"])

    out = pd.DataFrame(rows).groupby("Ticker", as_index=False).agg({"QTY": "sum", "VALUE": "sum"})

    # If schwabdev refreshed tokens, save local -> DB
    sync_local_to_db_multi(USER_ID, TOKEN_PATHS)    

    return out

st.set_page_config(page_title="Supertrend + MOST RSI + ADXR", layout="wide")
st.title("🟢 Exact KivancOzbilgic Supertrend + MOST RSI + ADXR — Stocks 1D + Crypto 4H")




# -----------------------
# TOP: Left/Right blocks
# -----------------------
left, right = st.columns([1, 1])

with left:
    st.markdown("### Signals Explained (compact)")

    st.markdown("- **Supertrend**: BUY if price > line (uptrend), SELL if price < line (downtrend)")
    st.markdown("- **MOST RSI**: BUY if yellow MA > brown MOST line; SELL otherwise")
    st.markdown("- **ADXR State**: RISING / FALLING / FLAT / LOW_FLAT *(LOW_FLAT = stand down / chop)*")

    st.markdown("**SIGNAL-Super-MOST-ADXR** logic:")
    st.markdown("• **EXIT** → Supertrend=SELL")
    st.markdown("• **BUY** → Supertrend=BUY AND MOST=BUY AND ADXR=RISING")
    st.markdown("• **HOLD** → Supertrend=BUY but **not aligned**:")
    st.markdown("   - MOST=SELL  **OR**  ADXR=FLAT/FALLING")
    st.markdown("• **STANDDOWN** → ADXR=LOW_FLAT (avoid new trades)")

with right:
    st.markdown("### 📊 Stock Market Context")

    vix = fetch_vix_value()
    spy_close, spy_ma200, spy_status = fetch_spy_vs_200ma()

    c1, c2 = st.columns(2)
    with c1:
        st.metric("VIX level", "N/A" if np.isnan(vix) else f"{vix:.2f}")
    with c2:
        st.metric("SPY vs 200MA", spy_status)

    if not np.isnan(spy_close) and not np.isnan(spy_ma200):
        st.caption(f"SPY Close: {spy_close:.2f}  |  200MA: {spy_ma200:.2f}")

    breadth = breadth_proxy_from_spy(spy_close, spy_ma200)
    st.markdown("**Breadth (proxy): % of S&P 500 stocks above 200MA**")
    if np.isnan(breadth["pct"]):
        st.write("Current: N/A")
    else:
        st.write(f"Current: **{breadth['pct']:.0f}%** → {breadth['action']}")

    st.code(
        "% of S&P stocks > 200-MA    Your Action\n"
        ">60%   ✅ Breadth is healthy → Trade full size\n"
        "50–60% 🟡 Breadth weakening → Trade half size\n"
        "<50%   🔴 Breadth poor → Sit out",
        language="text",
    )

    st.markdown("**Position Sizing Framework (3 filters):**")
    st.markdown("- VIX < 20 ✅  + Breadth healthy ✅  + SPY > 200MA ✅  → **Risk 2% per trade**")
    st.markdown("- VIX < 20 ✅  + Breadth diverging ⚠️  + SPY > 200MA ✅  → **Risk 1% per trade**")
    st.markdown("- VIX 20–30 ⚠️ + Either breadth OR 200MA warning → **Risk 0.5% per trade**")
    st.markdown("- VIX > 30 ❌ OR SPY < 200MA ❌ → **Sit out**")


st.divider()


# -----------------------
# Crypto Context block
# -----------------------
st.markdown("### 🪙 Crypto Market Context")

cg_err = None
total_err = None
alt_err = None

try:
    cg = fetch_coingecko_global()
except Exception as e:
    cg = {"total_mcap_usd": np.nan, "btc_dom": np.nan, "eth_dom": np.nan}
    cg_err = str(e)

total_hist = None
try:
    total_hist = fetch_total_mcap_history_coingecko(days=900)
    if not isinstance(total_hist, pd.DataFrame):
        raise ValueError(f"Expected DataFrame from CoinGecko history, got: {type(total_hist)}")
except Exception as e:
    try:
        total_hist = fetch_total_mcap_history_coinmarketcap(days=900)
    except Exception as e2:
        total_err = f"CoinGecko history failed: {e} | CMC fallback failed: {e2}"

total_ctx = compute_total_vs_200ma(total_hist) if total_hist is not None else {
    "mcap": np.nan, "ma200": np.nan, "status": "N/A", "days_below": None, "phase": "N/A"
}

try:
    alt = fetch_altcoin_season_index()
except Exception as e:
    alt = {"score": np.nan, "label": "N/A"}
    alt_err = str(e)

k1, k2, k3 = st.columns(3)
with k1:
    st.metric("TOTAL (mcap)", fmt_usd_compact(cg.get("total_mcap_usd", np.nan)))
with k2:
    bd = cg.get("btc_dom", np.nan)
    st.metric("BTC.D", "N/A" if np.isnan(bd) else f"{bd:.2f}%")
with k3:
    sc = alt.get("score", np.nan)
    st.metric("Altcoin Season", "N/A" if np.isnan(sc) else f"{int(sc)}/100 ({alt.get('label','N/A')})")

st.markdown("**Filter #2: Total Crypto Market Cap Trend (TOTAL vs 200MA)**")
if not np.isnan(total_ctx.get("mcap", np.nan)) and not np.isnan(total_ctx.get("ma200", np.nan)):
    st.write(f"- TOTAL: **{fmt_usd_compact(total_ctx['mcap'])}**")
    st.write(f"- 200MA: **{fmt_usd_compact(total_ctx['ma200'])}**")
    st.write(f"- Status: **{total_ctx['status']}**")
    if total_ctx["status"] == "BELOW":
        st.write(f"- Days below 200MA (consecutive): **{total_ctx['days_below']}**")
    st.write(f"- Phase: **{total_ctx['phase']}**")
else:
    st.write("TOTAL vs 200MA: N/A (history fetch failed or not enough data)")

st.code(
    "Rule: TOTAL crypto mcap vs 200-day MA\n"
    "🟢 Bull:  TOTAL > 200MA             → Safe to trade crypto\n"
    "🟡 Trans: TOTAL < 200MA (<30 days)  → 50% size, BTC/ETH only\n"
    "🔴 Bear:  TOTAL < 200MA (>30 days)  → Sit out crypto",
    language="text",
)

if cg_err:
    st.caption(f"CoinGecko global fetch issue: {cg_err}")
if total_err:
    st.caption(f"TOTAL history issue: {total_err}")
if alt_err:
    st.caption(f"Altcoin index issue: {alt_err}")


st.divider()


# -----------------------
# Refresh button + CSV outputs
# -----------------------
if st.button("🔄 Refresh signals (recompute + rewrite CSVs)"):
    try: 
        st.info("Refresh clicked — starting compute...")
        with st.spinner("Computing stock signals..."):
            df_stocks = build_stocks_signals_table(
                STOCK_TICKERS,
                atr_period=ATR_PERIOD,
                atr_multiplier=ATR_MULTIPLIER,
                rsi_period=RSI_PERIOD,
                vol_lookback=VOL_LOOKBACK,
                vol_multiplier=VOL_MULTIPLIER,
                rsi_buy_threshold=RSI_BUY_THRESHOLD,
                adxr_len=ADXR_LEN,
                adxr_lenx=ADXR_LENX,
                adxr_low_threshold=ADXR_LOW_THRESHOLD,
                adxr_flat_eps=ADXR_FLAT_EPS,
            )
            df_stocks.to_csv(os.path.join(OUTPUTS_DIR, "supertrend_stocks_1d.csv"), index=False)

        with st.spinner("Computing crypto signals..."):
            df_crypto = build_crypto_signals_table(
                CRYPTO_TICKERS,
                gecko_pools=CRYPTO_NOT_FOUND_YAHOO,   # <-- ADD THIS
                atr_period=ATR_PERIOD,
                atr_multiplier=ATR_MULTIPLIER,
                rsi_period=RSI_PERIOD,
                vol_lookback=VOL_LOOKBACK,
                vol_multiplier=VOL_MULTIPLIER,
                rsi_buy_threshold=RSI_BUY_THRESHOLD,
                adxr_len=ADXR_LEN,
                adxr_lenx=ADXR_LENX,
                adxr_low_threshold=ADXR_LOW_THRESHOLD,
                adxr_flat_eps=ADXR_FLAT_EPS,
            )
            df_crypto.to_csv(os.path.join(OUTPUTS_DIR, "supertrend_crypto_4h.csv"), index=False)

        st.success("Done. CSVs updated in outputs/.")
    except Exception as e:
        st.error("Refresh failed — here is the exact error:")
        st.exception(e)

st.subheader("📈 Stocks 1D Signals")
try:
    df_stocks = pd.read_csv(os.path.join(OUTPUTS_DIR, "supertrend_stocks_1d.csv"))

    # --- Add Schwab portfolio columns (QTY, VALUE) ---
    try:
        holdings = fetch_schwab_stock_holdings()
        df_stocks = df_stocks.merge(holdings, on="Ticker", how="left")
    except Exception:
        # If Schwab call fails, still show signals
        df_stocks["QTY"] = np.nan
        df_stocks["VALUE"] = np.nan

    # Fill blanks for non-held tickers
    if "QTY" in df_stocks.columns:
        df_stocks["QTY"] = df_stocks["QTY"].fillna(0)
    if "VALUE" in df_stocks.columns:
        df_stocks["VALUE"] = df_stocks["VALUE"].fillna(0.0)

    # --- Reorder columns ---
    # Put QTY and VALUE right after SIGNAL-Super-MOST-ADXR
    cols = list(df_stocks.columns)
    for c in ["QTY", "VALUE"]:
        if c in cols:
            cols.remove(c)

    insert_after = "SIGNAL-Super-MOST-ADXR"
    if insert_after in cols:
        idx = cols.index(insert_after) + 1
        cols[idx:idx] = ["QTY", "VALUE"]
    else:
        # fallback if column name changes
        cols = ["Ticker", "QTY", "VALUE"] + [c for c in cols if c not in ("Ticker",)]

    # Move Timeframe and Bar Time to the end
    for move_col in ["Timeframe", "Bar Time"]:
        if move_col in cols:
            cols.remove(move_col)
            cols.append(move_col)

    df_stocks = df_stocks[cols]                

    st.dataframe(df_stocks, use_container_width=True)
except Exception as e:
    st.info(f"No Stocks 1D CSV yet. Click Refresh. ({e})")

# st.subheader("₿ Crypto 4H Signals")
# try:
#     df_crypto = pd.read_csv(os.path.join(OUTPUTS_DIR, "supertrend_crypto_4h.csv"))
#     st.dataframe(df_crypto, use_container_width=True)
# except Exception as e:
#     st.info(f"No Crypto 4H CSV yet. Click Refresh. ({e})")

st.subheader("₿ Crypto 4H Signals")
try:
    df_crypto = pd.read_csv(os.path.join(OUTPUTS_DIR, "supertrend_crypto_4h.csv"))

    # Enrich with wallet qty/price/usd fields from ASSET_REGISTRY
    from data.crypto import enrich_crypto_portfolio_fields

    @st.cache_data(ttl=120)
    def _enrich_cached(df_in: pd.DataFrame) -> pd.DataFrame:
        return enrich_crypto_portfolio_fields(df_in)

    df_crypto = _enrich_cached(df_crypto)

    st.dataframe(df_crypto, use_container_width=True)

except Exception as e:
    st.info(f"No Crypto 4H CSV yet. Click Refresh. ({e})")

