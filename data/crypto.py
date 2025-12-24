# data/crypto.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timezone

# from dotenv import load_dotenv
# load_dotenv()


from core.utils import _fix_yf_cols
from core.indicators import apply_indicators
from core.signals import (
    signal_supertrend_plus_volume,
    signal_combined,
    signal_full_combined,
    signal_super_most_adxr,
    FINAL_COLUMN_ORDER,
)


from core.utils import get_secret

def _coingecko_headers() -> dict:
    headers = {}
    api_key = get_secret("COINGECKO_API_KEY").strip()
    if api_key:
        headers["x-cg-pro-api-key"] = api_key
    return headers


def fetch_crypto_4h_df(ticker: str, lookback_days: int = 70) -> pd.DataFrame | None:
    """
    Yahoo often supports 1h; we resample to 4H.
    """
    try:
        raw = yf.download(ticker, period=f"{lookback_days}d", interval="60m", progress=False)
        if raw is None or raw.empty:
            return None
        raw = _fix_yf_cols(raw)
        df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

        # yfinance timestamps can include tz; remove so resample works cleanly
        try:
            df.index = df.index.tz_localize(None)
        except Exception:
            pass

        resampled = df.resample("4H")
        df_4h = pd.DataFrame(
            {
                "High": resampled["High"].max(),
                "Low": resampled["Low"].min(),
                "Close": resampled["Close"].last(),
                "Volume": resampled["Volume"].sum(),
            }
        ).dropna()

        if df_4h.empty or len(df_4h) < 250:
            return None
        return df_4h
    except Exception:
        return None


def build_crypto_signals_table(
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
) -> pd.DataFrame:
    rows = []

    for t in tickers:
        base = fetch_crypto_4h_df(t)
        if base is None:
            continue

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

        rows.append(
            {
                "Ticker": t,
                "Timeframe": "4H",
                "Bar Time": last.name,
                "Last Close": round(float(last["Close"]), 6),
                "SIGNAL-Super-MOST-ADXR": super_most_adxr,
                "Supertrend": round(float(last["Supertrend"]), 6) if pd.notna(last["Supertrend"]) else np.nan,
                "Supertrend Signal": st_sig,
                "RSI": round(float(last["RSI"]), 2) if pd.notna(last["RSI"]) else np.nan,
                "MOST MA": round(float(last["MOST_MA"]), 2) if pd.notna(last["MOST_MA"]) else np.nan,
                "MOST Line": round(float(last["MOST_Line"]), 2) if pd.notna(last["MOST_Line"]) else np.nan,
                "MOST Signal": most_sig,
                "ADXR State": adxr_state,
                "ADXR Signal": str(last.get("ADXR_Signal", "WEAK")),
                "Volume": float(last["Volume"]) if pd.notna(last["Volume"]) else np.nan,
                "Supertrend+Vol Signal": vol_sig,
                "Combined Signal": comb_sig,
                "Full Combined": full_sig,
            }
        )

    out = pd.DataFrame(rows)
    return out.reindex(columns=FINAL_COLUMN_ORDER)


# # -----------------------------
# # Crypto Context (TOTAL vs 200MA, BTC.D, Altcoin Index)
# # -----------------------------
# def _coingecko_headers() -> dict:
#     headers = {}
#     api_key = os.environ.get("COINGECKO_API_KEY", "").strip()
#     if api_key:
#         headers["x-cg-pro-api-key"] = api_key
#     return headers


def fetch_coingecko_global() -> dict:
    url = "https://api.coingecko.com/api/v3/global"
    r = requests.get(url, headers=_coingecko_headers(), timeout=12)
    r.raise_for_status()
    j = r.json()
    d = j.get("data", {})
    total_mcap_usd = float(d.get("total_market_cap", {}).get("usd", np.nan))
    btc_dom = float(d.get("market_cap_percentage", {}).get("btc", np.nan))
    eth_dom = float(d.get("market_cap_percentage", {}).get("eth", np.nan))
    return {"total_mcap_usd": total_mcap_usd, "btc_dom": btc_dom, "eth_dom": eth_dom}


def fetch_total_mcap_history_coingecko(days: int = 900) -> pd.DataFrame:
    url = f"https://api.coingecko.com/api/v3/global/market_cap_chart?vs_currency=usd&days={days}"
    r = requests.get(url, headers=_coingecko_headers(), timeout=15)
    r.raise_for_status()
    j = r.json()

    pairs = None
    if isinstance(j, dict):
        pairs = j.get("market_cap_chart") or j.get("market_cap") or j.get("market_caps")
    if pairs is None:
        raise ValueError("Unexpected CoinGecko market cap history response.")

    df = pd.DataFrame(pairs, columns=["ts_ms", "mcap"])
    df["date"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.date
    df = df.groupby("date", as_index=False)["mcap"].last()
    df["mcap"] = df["mcap"].astype(float)
    return df.sort_values("date").reset_index(drop=True)


def fetch_total_mcap_history_coinmarketcap(days: int = 900) -> pd.DataFrame:
    # api_key = os.environ.get("CMC_API_KEY", "").strip()
    # if not api_key:
    #     raise ValueError("CMC_API_KEY not set.")
    
    api_key = get_secret("CMC_API_KEY").strip()
    if not api_key:
        raise ValueError("CMC_API_KEY not set")
    return {
        "X-CMC_PRO_API_KEY": api_key,
        "Accepts": "application/json"
    }

    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(days=days)

    url = "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/historical"
    params = {"time_start": start.isoformat(), "time_end": end.isoformat(), "interval": "daily"}
    headers = {"X-CMC_PRO_API_KEY": api_key, "Accepts": "application/json"}
    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    j = r.json()

    data = j.get("data", [])
    if not data:
        raise ValueError("CMC historical returned no data.")

    rows = []
    for item in data:
        ts = item.get("timestamp")
        quote = (item.get("quote") or {}).get("USD") or {}
        mcap = quote.get("total_market_cap")
        if ts is None or mcap is None:
            continue
        d = pd.to_datetime(ts, utc=True).date()
        rows.append((d, float(mcap)))

    return pd.DataFrame(rows, columns=["date", "mcap"]).sort_values("date").reset_index(drop=True)


# def compute_total_vs_200ma(total_df: pd.DataFrame) -> dict:
#     """
#     Your rule:
#       Bull: TOTAL > 200MA
#       Transition: below for <=30 days
#       Bear: below for >30 days
#     """
#     if total_df is None or total_df.empty or len(total_df) < 220:
#         return {"mcap": np.nan, "ma200": np.nan, "status": "N/A", "days_below": None, "phase": "N/A"}

#     s = total_df.copy()
#     s["ma200"] = s["mcap"].rolling(200).mean()

#     last = s.iloc[-1]
#     mcap = float(last["mcap"])
#     ma200 = float(last["ma200"]) if pd.notna(last["ma200"]) else np.nan
#     if np.isnan(ma200):
#         return {"mcap": mcap, "ma200": np.nan, "status": "N/A", "days_below": None, "phase": "N/A"}

#     above = mcap >= ma200
#     status = "ABOVE" if above else "BELOW"

#     days_below = 0
#     if not above:
#         i = len(s) - 1
#         while i >= 0:
#             row = s.iloc[i]
#             if pd.isna(row["ma200"]):
#                 break
#             if float(row["mcap"]) < float(row["ma200"]):
#                 days_below += 1
#                 i -= 1
#             else:
#                 break

#     if above:
#         phase = "BULL (Safe to trade crypto)"
#         days_below = 0
#     else:
#         phase = (
#             "TRANSITION (<30d below 200MA) — half size, BTC/ETH only"
#             if days_below <= 30
#             else "BEAR (>30d below 200MA) — sit out crypto"
#         )

#     return {"mcap": mcap, "ma200": ma200, "status": status, "days_below": days_below, "phase": phase}

def compute_total_vs_200ma(total_df) -> dict:
    # Guard: sometimes fetchers may return dicts on failure
    if total_df is None:
        return {"mcap": np.nan, "ma200": np.nan, "status": "N/A", "days_below": None, "phase": "N/A"}

    # If a dict slipped through, treat it as an error payload and fail gracefully
    if isinstance(total_df, dict):
        return {
            "mcap": np.nan,
            "ma200": np.nan,
            "status": "N/A",
            "days_below": None,
            "phase": "N/A",
            "error": str(total_df),
        }

    # Now safe to assume DataFrame-like
    if total_df.empty or len(total_df) < 220:
        return {"mcap": np.nan, "ma200": np.nan, "status": "N/A", "days_below": None, "phase": "N/A"}

    s = total_df.copy()
    s["ma200"] = s["mcap"].rolling(200).mean()

    last = s.iloc[-1]
    mcap = float(last["mcap"])
    ma200 = float(last["ma200"]) if pd.notna(last["ma200"]) else np.nan
    if np.isnan(ma200):
        return {"mcap": mcap, "ma200": np.nan, "status": "N/A", "days_below": None, "phase": "N/A"}

    above = mcap >= ma200
    status = "ABOVE" if above else "BELOW"

    days_below = 0
    if not above:
        i = len(s) - 1
        while i >= 0:
            row = s.iloc[i]
            if pd.isna(row["ma200"]):
                break
            if float(row["mcap"]) < float(row["ma200"]):
                days_below += 1
                i -= 1
            else:
                break

    if above:
        phase = "BULL (Safe to trade crypto)"
    else:
        phase = "TRANSITION (<30d below 200MA) — half size, BTC/ETH only" if days_below <= 30 else "BEAR (>30d below 200MA) — sit out crypto"

    return {"mcap": mcap, "ma200": ma200, "status": status, "days_below": (days_below if not above else 0), "phase": phase}

def fetch_altcoin_season_index() -> dict:
    url = "https://www.blockchaincenter.net/api/altcoin-season-index/"
    r = requests.get(url, timeout=12)
    r.raise_for_status()
    j = r.json()

    score = None
    if isinstance(j, dict):
        for k in ["altcoinSeasonIndex", "altcoin_season_index", "value", "index"]:
            if k in j:
                score = j.get(k)
                break
        if score is None and "data" in j and isinstance(j["data"], dict):
            score = j["data"].get("altcoinSeasonIndex", None)
    elif isinstance(j, list) and len(j) > 0 and isinstance(j[-1], dict):
        score = j[-1].get("value", j[-1].get("altcoinSeasonIndex", None))

    try:
        score = int(round(float(score))) if score is not None else None
    except Exception:
        score = None

    if score is None:
        return {"score": np.nan, "label": "N/A"}

    if score >= 75:
        label = "ALTCOIN_SEASON"
    elif score <= 25:
        label = "BTC_SEASON"
    else:
        label = "NEUTRAL"

    return {"score": score, "label": label}
