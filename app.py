# app.py
from __future__ import annotations

from data.schwab.schwab_helper import create_schwab_client, SchwabAuthError


import schwabdev

import os
import numpy as np
import pandas as pd
import streamlit as st


from datetime import datetime, timezone

from data.schwab.token_store import load_tokens_db, delete_tokens_db
from data.schwab.token_sync import sync_db_to_local_multi, sync_local_to_db_multi



import schwabdev


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

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# APP_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = APP_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# TOKEN_PATH = OUTPUTS_DIR / "schwab_tokens.json"
# TOKEN_PATH = APP_DIR / "tokens.json"
TOKEN_PATHS = [
    APP_DIR / "tokens.json",
    APP_DIR / "data" / "schwab" / "tokens.json",
]
USER_ID = "main"

STOCKS_CSV = OUTPUTS_DIR / "supertrend_stocks_1d.csv"
CRYPTO_CSV = OUTPUTS_DIR / "supertrend_crypto_4h.csv"





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
    "AAPL","AAOI","ABTC", "ALAB","AMD","AMZN","APH","APLD","APP","ARKB", "ARM" , "ASML", "AVGO",
    "BE", "BMNR", "BWXT", "CEG",  "CFG","COIN","COHR","COPX", "CORZ","CRDO","CRVW", "CRWD", "ETHA","GEV", "GLD", "GOOG",
    "HODL","HOOD","IBIT","IDR","INOD","IONQ","IREN","LEU","LITE","LRCX","LTBR",
    "META","MNST", "MP","MSFT","MSTR","MSTX","MU","NET","NPPTF","NVDA","OKLO", "ORCL",  "PLTR",
    "QBTS","QUBT","RGTI","RDDT","SLV", "SOFI", "SMR", "SSK","STKE", "STRC",
    "TER","TSLA","TSM","UPXI","VRT", "WDC"  
]

CRYPTO_TICKERS = [
    "BTC-USD","ETH-USD","SOL-USD","HYPE32196-USD", "SUI20947-USD", "LINK-USD","DOGE-USD", "ONDO-USD","BNB-USD",

    "AAVE-USD" , "ADA-USD" , "AIXBT-USD", "AKT-USD", "ANON35092-USD", "ASTER36341-USD", "AUKI-USD", "AURORA14803-USD", "blue-usd" , "cetus-usd" ,"cookie31838-usd" ,"CRV-USD",
    "DOGE-USD", "DRIFT31278-USD", "ELIZAOS-USD",  "elon-usd" ,"ENA-USD","ENS-USD",
    "fluid-usd", "fluxb-usd","FAI34330-USD", "griffain-USD",
    "HNT-USD","JTO-USD", "JUP29210-USD", "KMNO-USD", "LFNTY-USD", 
    "MOBILE-USD",  "MON30495-USD", "navx-USD" , "NEAR-USD", "NOS-USD",  "ORCA-USD" , "ore32782-USD",
    "pippin-usd" , "PNK-USD", "PROVE-USD", "PYTH-USD","RAY-USD","RENDER-USD",
     "SUAI-USD", "suins-usd",  "TAI20605-USD",
    "VIRTUAL-USD", "W-USD" , "WAL36119-USD", "WLD-USD", "wlfi33251-usd",  "XBG-USD" , "XRP-USD", "ZEREBRO-USD" , "ZEUS30391-USD", "zk24091-USD"
]



CRYPTO_NOT_FOUND_YAHOO = {
    "LQL": "https://www.geckoterminal.com/solana/pools/GiRyo4r3kREH8oRCe9GoJJARZuGo4ksto6xXvUok4wdd",
    "A0X": "https://www.geckoterminal.com/base/pools/0xfd100e192d0ff7a284f31a93b367d582666e406b",
    "GAME": "https://www.geckoterminal.com/base/pools/0xd418dfe7670c21f682e041f34250c114db5d7789",
    "AoT": "https://www.geckoterminal.com/base/pools/0x6e7a1875810afb6074953094c35a101e1cc7aee010fb1372d8f41b0fbe92d83c",   
}

# AoT 0xcc4adb618253ed0d4d8a188fb901d70c54735e03



def _parse_iso_dt(s: str) -> datetime | None:
    if not s:
        return None
    try:
        # supabase stores ISO; ensure tz-aware
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def schwab_token_status(user_id: str, token_paths: list[Path]) -> dict:
    db = load_tokens_db(user_id=user_id)
    now = datetime.now(timezone.utc)

    status = {
        "profile_user_id": user_id,
        "callback_url": st.secrets.get("callback_url", None) if hasattr(st, "secrets") else None,
        "db": None,
        "local_files": [],
    }

    # DB info
    if db:
        exp = _parse_iso_dt(db.get("expires_at"))
        seconds_left = int((exp - now).total_seconds()) if exp else None
        status["db"] = {
            "has_access_token": bool(db.get("access_token")),
            "has_refresh_token": bool(db.get("refresh_token")),
            "expires_at": db.get("expires_at"),
            "seconds_left": seconds_left,
            "expired": (seconds_left is not None and seconds_left <= 0),
            "updated_at": db.get("updated_at"),
        }
    else:
        status["db"] = {"present": False}

    # Local files info
    for p in token_paths:
        exists = p.exists()
        status["local_files"].append({
            "path": str(p),
            "exists": exists,
            "size": p.stat().st_size if exists else 0,
            "mtime": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat() if exists else None
        })

    return status


# Replace your fetch_schwab_stock_holdings() function in app.py with this:

from data.schwab.schwab_helper import create_schwab_client, SchwabAuthError

def fetch_schwab_stock_holdings() -> pd.DataFrame:
    """
    Returns a DataFrame: Ticker, QTY, VALUE
    Aggregated across all accounts, equities only.
    """
    try:
        # For local development: skip DB, use local tokens only
        # Set local_only=True to bypass database token checks
        LOCAL_ONLY = True  # Change to False when deploying to Streamlit Cloud
        
        client_wrapper = create_schwab_client(USER_ID, TOKEN_PATHS, local_only=LOCAL_ONLY)
        
        # Fetch positions (handles token refresh automatically)
        data = client_wrapper.fetch_positions()
        
        # DEBUG: Show raw response structure
        st.write("🔍 DEBUG: Number of accounts:", len(data))
        
        # Parse positions
        rows = []
        for i, acct in enumerate(data):
            st.write(f"📊 Account {i+1}:")
            
            sa = acct.get("securitiesAccount", {})
            acct_type = sa.get("type", "UNKNOWN")
            st.write(f"  - Type: {acct_type}")
            
            positions = sa.get("positions", []) or []
            st.write(f"  - Total positions: {len(positions)}")
            
            equity_count = 0
            for pos in positions:
                inst = pos.get("instrument", {}) or {}
                symbol = inst.get("symbol")
                asset_type = inst.get("assetType")
                
                st.write(f"    - {symbol}: {asset_type}")
                
                if asset_type != "EQUITY":
                    continue
                
                equity_count += 1
                long_qty = float(pos.get("longQuantity") or 0.0)
                short_qty = float(pos.get("shortQuantity") or 0.0)
                qty = long_qty - short_qty
                mkt_value = float(pos.get("marketValue") or 0.0)
                
                if symbol:
                    rows.append({"Ticker": symbol, "QTY": qty, "VALUE": mkt_value})
                    st.write(f"      ✅ Added: {symbol} | Qty: {qty} | Value: ${mkt_value:,.2f}")
            
            st.write(f"  - EQUITY positions: {equity_count}")
        
        st.write(f"🎯 Total equity rows collected: {len(rows)}")
        
        if not rows:
            st.warning("⚠️ No EQUITY positions found in any Schwab accounts.")
            st.info("This could mean: 1) All positions are cash/options/futures, or 2) Account structure is different than expected")
            return pd.DataFrame(columns=["Ticker", "QTY", "VALUE"])
        
        out = (
            pd.DataFrame(rows)
            .groupby("Ticker", as_index=False)
            .agg({"QTY": "sum", "VALUE": "sum"})
        )
        
        st.write(f"📈 Final holdings DataFrame ({len(out)} unique tickers):")
        st.dataframe(out)
        
        return out
        
    except SchwabAuthError as e:
        # User-friendly error message
        st.error(f"🔐 Schwab Authentication Error:\n\n{str(e)}")
        return pd.DataFrame(columns=["Ticker", "QTY", "VALUE"])
    
    except Exception as e:
        st.error(f"Unexpected error fetching Schwab holdings:\n{str(e)}")
        st.exception(e)
        return pd.DataFrame(columns=["Ticker", "QTY", "VALUE"])


# Also update your schwab_auth_panel() test button:

def schwab_auth_panel():
    st.sidebar.header("🔐 Schwab Auth")

    # Profiles so LOCAL + HOSTED tokens are separate
    profile = st.sidebar.radio("Token profile", ["LOCAL", "HOSTED"], horizontal=True)
    user_id = "main_local" if profile == "LOCAL" else "main_hosted"

    token_paths = [Path(p) for p in TOKEN_PATHS]
    st.sidebar.caption(f"DB user_id: `{user_id}`")

    with st.sidebar.expander("Status", expanded=True):
        st.json(schwab_token_status(user_id, token_paths))

    col1, col2 = st.sidebar.columns(2)

    if col1.button("📥 DB → Local"):
        ok = sync_db_to_local_multi(user_id, token_paths)
        st.sidebar.success(f"DB → Local sync: {ok}")

    if col2.button("📤 Local → DB"):
        sync_local_to_db_multi(user_id, token_paths)
        st.sidebar.success("Local → DB sync done")

    st.sidebar.divider()

    # Clear buttons
    st.sidebar.subheader("Reset / Re-auth")
    st.sidebar.caption("Clears tokens so next Schwab call forces a fresh login.")

    c1, c2 = st.sidebar.columns(2)

    if c1.button("🗑️ Clear LOCAL files"):
        for p in token_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception as e:
                st.sidebar.warning(f"Could not delete {p}: {e}")
        st.sidebar.success("Local token files cleared.")

    if c2.button("🗑️ Clear DB tokens"):
        try:
            delete_tokens_db(user_id=user_id)
            st.sidebar.success("DB tokens cleared.")
        except Exception as e:
            st.sidebar.error(f"DB delete failed: {e}")

    if st.sidebar.button("🔥 Clear BOTH (DB + Local)", type="primary"):
        for p in token_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception as e:
                st.sidebar.warning(f"Could not delete {p}: {e}")
        try:
            delete_tokens_db(user_id=user_id)
        except Exception as e:
            st.sidebar.error(f"DB delete failed: {e}")
        st.sidebar.success("✅ Cleared DB + Local. Next Schwab call will require login.")

    st.sidebar.divider()

    # Test connection with better error handling
    st.sidebar.subheader("Test")
    
    col_a, col_b = st.sidebar.columns(2)
    
    # Normal test (uses existing tokens)
    if col_a.button("✅ Test", help="Test with existing tokens"):
        try:
            client_wrapper = create_schwab_client(user_id, token_paths)
            data = client_wrapper.fetch_positions()
            st.sidebar.success(f"✅ Schwab OK! Found {len(data)} accounts.")
            
        except SchwabAuthError as e:
            st.sidebar.error(str(e))
            
        except Exception as e:
            st.sidebar.error(f"Unexpected error: {str(e)}")
            st.sidebar.exception(e)
    
    # Force new login (clears tokens and triggers OAuth)
    if col_b.button("🔐 Login", help="Force new OAuth login", type="primary"):
        st.sidebar.info("🌐 Starting OAuth flow...")
        st.sidebar.warning(
            "⚠️ IMPORTANT:\n"
            "1. A browser window will open\n"
            "2. Login to Schwab\n"
            "3. After login, you'll see 'Site can't be reached'\n"
            "4. COPY the full URL from browser\n"
            "5. Check your VSCode TERMINAL below\n"
            "6. Look for: 'Paste the callback URL here:'\n"
            "7. PASTE the URL and hit Enter"
        )
        
        try:
            # Clear tokens first
            for p in token_paths:
                p.unlink(missing_ok=True)
            delete_tokens_db(user_id=user_id)
            
            # Force OAuth flow
            client_wrapper = create_schwab_client(user_id, token_paths)
            client = client_wrapper.get_client(force_new_auth=True)
            
            # If we get here, auth succeeded
            st.sidebar.success("✅ Login successful!")
            st.sidebar.info("Now click 'Local → DB' to save tokens to database.")
            
        except Exception as e:
            st.sidebar.error(f"Login failed: {str(e)}")
            st.sidebar.exception(e)





def main():

    st.set_page_config(page_title="Supertrend + MOST RSI + ADXR", layout="wide")
    st.title("🟢 Exact KivancOzbilgic Supertrend + MOST RSI + ADXR — Stocks 1D + Crypto 4H")

    try:
        test_path = OUTPUTS_DIR / "_write_test.txt"
        test_path.write_text("ok\n", encoding="utf-8")
        # st.success(f"Write test OK: {test_path}")
    except Exception as e:
        print("Write test FAILED:", e)
        st.error("Write test FAILED — outputs folder is not writable:")
        st.exception(e)


    schwab_auth_panel()

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

    # @st.cache_data(ttl=120)






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




    st.subheader("📈 Stocks 1D Signals")
    try:
        df_stocks = pd.read_csv(os.path.join(OUTPUTS_DIR, "supertrend_stocks_1d.csv"))

        # --- Add Schwab portfolio columns (QTY, VALUE) ---
        st.write("🔄 Fetching Schwab holdings...")
        try:
            holdings = fetch_schwab_stock_holdings()
            st.write(f"✅ Got {len(holdings)} holdings from Schwab")
            
            # Show what we got
            if not holdings.empty:
                st.write("Holdings preview:")
                st.dataframe(holdings)
            
            df_stocks = df_stocks.merge(holdings, on="Ticker", how="left")
            st.write("✅ Merged holdings with signals")
            
        except SchwabAuthError as e:
            st.error(f"Schwab auth failed: {e}")
            df_stocks["QTY"] = np.nan
            df_stocks["VALUE"] = np.nan
            
        except Exception as e:
            # Don't silence errors - show them!
            st.error(f"Error fetching Schwab holdings: {str(e)}")
            st.exception(e)
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

        st.dataframe(df_stocks, width="stretch")

    except Exception as e:
        st.info(f"No Stocks 1D CSV yet. Click Refresh. ({e})")
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

        # ---------------------------
        # Add Total Val + ALT% columns
        # ---------------------------
        # 1) Rename "USD Value" -> "ALT USD Val"
        if "USD Value" in df_crypto.columns:
            df_crypto = df_crypto.rename(columns={"USD Value": "ALT USD Val"})

        # 2) Compute Total Val = ALT USD Val + USDC Value
        # Ensure numeric (handles None / strings)
        for c in ["ALT USD Val", "USDC Value"]:
            if c in df_crypto.columns:
                df_crypto[c] = pd.to_numeric(df_crypto[c], errors="coerce").fillna(0.0)

        if "ALT USD Val" in df_crypto.columns and "USDC Value" in df_crypto.columns:
            df_crypto["Total Val"] = df_crypto["ALT USD Val"] + df_crypto["USDC Value"]
            df_crypto["ALT%"] = np.where(
                df_crypto["Total Val"] > 0,
                (df_crypto["ALT USD Val"] / df_crypto["Total Val"]) * 100.0,
                0.0,
            ).round(2)
        else:
            # If columns aren't present, still create them so UI is consistent
            df_crypto["Total Val"] = 0.0
            df_crypto["ALT%"] = 0.0

        # ---------------------------
        # ACTION column (rebalance alert)
        # ---------------------------
        SIGNAL_COL = "SIGNAL-Super-MOST-ADXR"

        # Ensure numeric ALT% (it already is, but keep safe)
        if "ALT%" in df_crypto.columns:
            df_crypto["ALT%"] = pd.to_numeric(df_crypto["ALT%"], errors="coerce").fillna(0.0)

        def _action_row(r):
            sig = str(r.get(SIGNAL_COL, "")).upper()
            alt_pct = float(r.get("ALT%", 0.0))

            # BUY signal but ALT exposure low -> buy ALT
            if sig == "BUY" and alt_pct < 50.0:
                return "🔴 BUY ALT"

            # EXIT signal but ALT exposure high -> sell ALT
            if sig == "EXIT" and alt_pct > 50.0:
                return "🔴 SELL ALT"

            return ""  # no action

        df_crypto["ACTION"] = df_crypto.apply(_action_row, axis=1)



        # ---------------------------
        # Reorder columns for clarity
        # ALT USD Val -> ALT% -> USDC Value -> Total Val
        # ---------------------------
        desired_order = []
        cols = list(df_crypto.columns)

        def _move_after(col_to_move, after_col):
            if col_to_move in cols and after_col in cols:
                cols.remove(col_to_move)
                idx = cols.index(after_col) + 1
                cols.insert(idx, col_to_move)


        # Put ACTION right after SIGNAL
        _move_after("ACTION", "SIGNAL-Super-MOST-ADXR")
        # Move ALT% right after ALT USD Val
        _move_after("ALT%", "ALT USD Val")

        # Move Total Val right after USDC Value
        _move_after("Total Val", "USDC Value")

        df_crypto = df_crypto[cols]



        # st.dataframe(df_crypto, use_container_width=True)
        def _style_action(val):
            if isinstance(val, str) and val.startswith("🔴"):
                return "color: red; font-weight: 700;"
            return ""

        styled = df_crypto.style.applymap(_style_action, subset=["ACTION"])
        st.dataframe(styled, use_container_width=True)
        

        # # ---------------------------
        # # Grand Total (sum of Total Val)
        # # ---------------------------
        # if "Total Val" in df_crypto.columns:
        #     grand_total = float(pd.to_numeric(df_crypto["Total Val"], errors="coerce").fillna(0.0).sum())
        #     # st.markdown(f"**Grand Total (ALT + USDC): ${grand_total:,.2f}**")
        #     st.metric("Grand Total (ALT + USDC)", f"${grand_total:,.2f}")
        # else:
        #     st.markdown("**Grand Total (ALT + USDC): $0.00**")

        # ---------------------------
        # Portfolio Totals
        # ---------------------------
        alt_total = 0.0
        usdc_total = 0.0
        grand_total = 0.0
        alt_pct_total = 0.0

        if "ALT USD Val" in df_crypto.columns:
            alt_total = float(
                pd.to_numeric(df_crypto["ALT USD Val"], errors="coerce")
                .fillna(0.0)
                .sum()
            )

        if "USDC Value" in df_crypto.columns:
            usdc_total = float(
                pd.to_numeric(df_crypto["USDC Value"], errors="coerce")
                .fillna(0.0)
                .sum()
            )

        if "Total Val" in df_crypto.columns:
            grand_total = float(
                pd.to_numeric(df_crypto["Total Val"], errors="coerce")
                .fillna(0.0)
                .sum()
            )
        if grand_total > 0:
            alt_pct_total = (alt_total / grand_total) * 100.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total ALT USD Val", f"${alt_total:,.2f}")
        c2.metric("Total USDC Value", f"${usdc_total:,.2f}")
        c3.metric("Grand Total (ALT + USDC)", f"${grand_total:,.2f}")
        c4.metric("Total ALT %", f"{alt_pct_total:.2f}%")




    except Exception as e:
        st.info(f"No Crypto 4H CSV yet. Click Refresh. ({e})")




if __name__ == "__main__":
    main()