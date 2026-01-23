# data/crypto.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timezone
from decimal import Decimal

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

# NEW imports
from core.registry import load_asset_registry, find_registry_entry_by_yahoo_ticker
from core.prices import get_yahoo_price_usd
# from core.balances.solana import get_spl_token_balance
from core.balances.evm import get_erc20_balance
from core.balances.sui import get_coin_balance

import re
from functools import lru_cache

from core.balances.solana import get_spl_token_balance_from_cache


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

        try:
            df.index = df.index.tz_localize(None)
        except Exception:
            pass

        resampled = df.resample("4h")
        df_4h = pd.DataFrame(
            {
                "High": resampled["High"].max(),
                "Low": resampled["Low"].min(),
                "Close": resampled["Close"].last(),
                "Volume": resampled["Volume"].sum(),
            }
        ).dropna()

        print("Ticker ", ticker, " length= ", len(df_4h) )
        if df_4h.empty or len(df_4h) < 240:
            return None
        return df_4h
    except Exception:
        return None





def build_crypto_signals_table(
    tickers: list[str],
    *,
    gecko_pools: dict[str, str] | None = None,
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
    gecko_pools = gecko_pools or {}

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


# 2) extra geckoterminal pools (LVL/A0X/GAME etc)
    for sym, pool_url in gecko_pools.items():
        base = fetch_geckoterminal_4h_df_from_pool_url(pool_url)
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
                "Ticker": sym,              # <-- IMPORTANT: keep it as "LVL" etc
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

    out = pd.DataFrame(rows).reindex(columns=FINAL_COLUMN_ORDER)
    return out




GECKO_BASE = "https://api.geckoterminal.com/api/v2"
_GECKO_UA = "myTrading-geckoterminal/1.0"


def _parse_geckoterminal_pool_url(url: str) -> tuple[str, str]:
    """
    Example:
      https://www.geckoterminal.com/solana/pools/<POOL>
      https://www.geckoterminal.com/base/pools/<POOL>

    Returns: (network, pool_address)
    """
    m = re.search(r"geckoterminal\.com/([^/]+)/pools/([^/?#]+)", url)
    if not m:
        raise ValueError(f"Invalid GeckoTerminal pool URL: {url}")
    return m.group(1).strip().lower(), m.group(2).strip()


def _gecko_get(path: str, params: dict | None = None) -> dict:
    url = f"{GECKO_BASE}{path}"
    r = requests.get(url, params=params, headers={"User-Agent": _GECKO_UA}, timeout=25)
    r.raise_for_status()
    return r.json()


@lru_cache(maxsize=256)
def fetch_geckoterminal_ohlcv_4h(network: str, pool_address: str, limit: int = 500) -> pd.DataFrame | None:
    """
    Fetches 4H OHLCV directly from GeckoTerminal pool candles.
    Uses timeframe=hour + aggregate=4 => 4H.
    Returns df indexed by datetime (naive), with Open/High/Low/Close/Volume.
    """
    try:
        j = _gecko_get(
            f"/networks/{network}/pools/{pool_address}/ohlcv/hour",
            params={"aggregate": 4, "limit": int(limit)},
        )
        attrs = (j.get("data") or {}).get("attributes") or {}
        rows = attrs.get("ohlcv_list") or []
        if not rows:
            return None

        df = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
        df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(None)
        df = df.drop(columns=["ts"]).set_index("dt").sort_index()

        # numeric
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna()

        # your Yahoo fetch requires ~240 bars; keep consistent
        if df.empty or len(df) < 240:
            return None

        return df
    except Exception:
        return None


def fetch_geckoterminal_4h_df_from_pool_url(pool_url: str) -> pd.DataFrame | None:
    network, pool = _parse_geckoterminal_pool_url(pool_url)
    return fetch_geckoterminal_ohlcv_4h(network, pool)


@lru_cache(maxsize=256)
def get_geckoterminal_price_usd_from_pool_url(pool_url: str) -> Decimal | None:
    """
    Price proxy: use the last close from 4H candles.
    This avoids needing any extra endpoints.
    """
    df = fetch_geckoterminal_4h_df_from_pool_url(pool_url)
    if df is None or df.empty:
        return None
    last_close = df["Close"].iloc[-1]
    try:
        return Decimal(str(float(last_close)))
    except Exception:
        return None

# -----------------------------
# Crypto Context (TOTAL vs 200MA, BTC.D, Altcoin Index)
# -----------------------------
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
    api_key = get_secret("CMC_API_KEY").strip()
    if not api_key:
        raise ValueError("CMC_API_KEY not set")

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


def compute_total_vs_200ma(total_df) -> dict:
    if total_df is None:
        return {"mcap": np.nan, "ma200": np.nan, "status": "N/A", "days_below": None, "phase": "N/A"}

    if isinstance(total_df, dict):
        return {"mcap": np.nan, "ma200": np.nan, "status": "N/A", "days_below": None, "phase": "N/A", "error": str(total_df)}

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

    phase = "BULL (Safe to trade crypto)" if above else ("TRANSITION (<30d below 200MA) — half size, BTC/ETH only" if days_below <= 30 else "BEAR (>30d below 200MA) — sit out crypto")
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


# ============================================================
# NEW: Portfolio enrichment (Qty, Price, USD Value, USDC Value)
# ============================================================

ETH_RPC_URLS = ["https://eth.llamarpc.com", "https://ethereum.publicnode.com"]
BASE_RPC_URLS = ["https://mainnet.base.org", "https://base.publicnode.com"]
BSC_RPC_URLS = [
    "https://bsc-dataseed.binance.org",
    "https://bsc.publicnode.com",
]


def get_solana_native_balance(wallet: str) -> Decimal:
    """
    Solana native SOL balance via JSON-RPC getBalance.
    Returns SOL as Decimal.
    """
    url = "https://api.mainnet-beta.solana.com"
    payload = {"jsonrpc": "2.0", "id": 1, "method": "getBalance", "params": [wallet]}
    r = requests.post(url, json=payload, timeout=12)
    r.raise_for_status()
    lamports = int(r.json()["result"]["value"])
    return Decimal(lamports) / Decimal(1_000_000_000)


def get_evm_native_balance(wallet: str, rpc_urls: list[str]) -> Decimal:
    """
    EVM native balance (ETH/BNB/etc) via eth_getBalance.
    Returns native coin amount as Decimal.
    """
    payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_getBalance", "params": [wallet, "latest"]}

    last_err = None
    for url in rpc_urls:
        try:
            r = requests.post(url, json=payload, timeout=12)
            r.raise_for_status()
            wei_hex = r.json().get("result")
            if not wei_hex:
                continue
            wei = int(wei_hex, 16)
            return Decimal(wei) / Decimal(10**18)
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"All RPCs failed for native balance: {last_err}")



def enrich_crypto_portfolio_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - Qty
      - Price
      - USD Value
      - USDC Value

    Uses ASSET_REGISTRY in secrets/env.
    If blockchain blank/missing => leaves fields blank.
    """
    registry = load_asset_registry()

    # Ensure columns exist
    for col in ["Qty", "Price", "USD Value", "USDC Value"]:
        if col not in df.columns:
            df[col] = np.nan

    # If registry missing, just return with blank cols
    if not registry:
        return _reorder_crypto_columns(df)

    qty_list = []
    price_list = []
    usd_list = []
    usdc_list = []
    sol_wallet_cache = {}

    for _, row in df.iterrows():
        yahoo_ticker = str(row.get("Ticker", "")).strip()
        entry = find_registry_entry_by_yahoo_ticker(registry, yahoo_ticker)

        # fallback: match by symbol key like "LVL"
        if not entry:
            sym = str(row.get("Ticker", "")).strip()
            entry = registry.get(sym) or registry.get(sym.upper()) or registry.get(sym.lower())


        if not entry:
            qty_list.append(np.nan)
            price_list.append(np.nan)
            usd_list.append(np.nan)
            usdc_list.append(np.nan)
            continue

        chain = str(entry.get("blockchain", "")).strip().lower()
        wallet = str(entry.get("wallet_address", "")).strip()
        token_contract = str(entry.get("token_contract", "")).strip()
        stable_contract = str(entry.get("stablecoin_contract", "")).strip()

        # If chain not provided, leave blank
        if not chain:
            qty_list.append(np.nan)
            price_list.append(np.nan)
            usd_list.append(np.nan)
            usdc_list.append(np.nan)
            continue

        # Price (Yahoo) — if yahoo_ticker exists, we try it; else blank
        price_dec = None

        # 1) If row ticker is a Gecko-only symbol (like LVL/A0X/GAME), use GeckoTerminal close
        try:
            # we let app.py store the mapping in env/registry OR we detect via registry entry
            # easiest: if entry has "geckoterminal_pool" field use it
            gecko_pool = str(entry.get("geckoterminal_pool", "")).strip()
            if gecko_pool:
                price_dec = get_geckoterminal_price_usd_from_pool_url(gecko_pool)
        except Exception:
            price_dec = None

        try:
            if yahoo_ticker:
                price_dec = get_yahoo_price_usd(yahoo_ticker)
        except Exception:
            price_dec = None

        # Qty + USDC balances
        qty_dec = None
        usdc_dec = None

        try:
            
            if chain == "solana":
                # Native SOL if token_contract is blank
                if wallet and not token_contract:
                    qty_dec = get_solana_native_balance(wallet)
                elif wallet and token_contract:
                    # qty_dec = get_spl_token_balance(wallet, token_contract)
                    qty_dec = get_spl_token_balance_from_cache(wallet, token_contract, sol_wallet_cache)
                    debug = True
                    if debug and qty_dec != 0:
                        print(f"[SOL] {yahoo_ticker} | Wallet={wallet} | Qty={qty_dec}")


                if wallet and stable_contract:
                    # usdc_dec = get_spl_token_balance(wallet, stable_contract)
                    usdc_dec = get_spl_token_balance_from_cache(wallet, stable_contract, sol_wallet_cache)
                    debug = True
                    if debug and usdc_dec != 0:
                        print(f"[SOL] USDC | Wallet={wallet} | Qty={usdc_dec}")




            elif chain == "ethereum":
                if wallet and not token_contract:
                    qty_dec = get_evm_native_balance(wallet, ETH_RPC_URLS)
                elif wallet and token_contract:
                    qty_dec = get_erc20_balance(wallet, token_contract, ETH_RPC_URLS)

                if wallet and stable_contract:
                    usdc_dec = get_erc20_balance(wallet, stable_contract, ETH_RPC_URLS)

            # elif chain == "base":
            #     if wallet and token_contract:
            #         qty_dec = get_erc20_balance(wallet, token_contract, BASE_RPC_URLS)
            #     if wallet and stable_contract:
            #         usdc_dec = get_erc20_balance(wallet, stable_contract, BASE_RPC_URLS)

            elif chain == "base":
                if wallet and not token_contract:
                    qty_dec = get_evm_native_balance(wallet, BASE_RPC_URLS)
                elif wallet and token_contract:
                    qty_dec = get_erc20_balance(wallet, token_contract, BASE_RPC_URLS)

                if wallet and stable_contract:
                    usdc_dec = get_erc20_balance(wallet, stable_contract, BASE_RPC_URLS)


            elif chain == "sui":
                if wallet and token_contract:
                    qty_dec = get_coin_balance(wallet, token_contract)
                if wallet and stable_contract:
                    usdc_dec = get_coin_balance(wallet, stable_contract)

            elif chain in ("bsc", "bnb"):
                if wallet and not token_contract:
                    qty_dec = get_evm_native_balance(wallet, BSC_RPC_URLS)
                elif wallet and token_contract:
                    qty_dec = get_erc20_balance(wallet, token_contract, BSC_RPC_URLS)

                if wallet and stable_contract:
                    usdc_dec = get_erc20_balance(wallet, stable_contract, BSC_RPC_URLS)



            else:
                qty_dec = None
                usdc_dec = None

        except Exception:
            # if balance fetch fails, keep blank for that row
            qty_dec = None
            usdc_dec = None

        # Compute USD value
        usd_val = None
        if qty_dec is not None and price_dec is not None:
            try:
                usd_val = qty_dec * price_dec
            except Exception:
                usd_val = None

        # Convert to floats for dataframe display
        qty_list.append(float(qty_dec) if qty_dec is not None else np.nan)
        price_list.append(float(price_dec) if price_dec is not None else np.nan)
        usd_list.append(float(usd_val) if usd_val is not None else np.nan)

        # USDC value = USDC qty (assume $1)
        usdc_list.append(float(usdc_dec) if usdc_dec is not None else np.nan)

    out = df.copy()
    out["Qty"] = pd.Series(qty_list).round(4)
    out["Price"] = pd.Series(price_list).round(4)
    out["USD Value"] = pd.Series(usd_list).round(4)
    out["USDC Value"] = pd.Series(usdc_list).round(4)

    return _reorder_crypto_columns(out)


def _reorder_crypto_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Move 'Bar Time' to end
    2) Insert Qty/Price/USD Value/USDC Value right after 'SIGNAL-Super-MOST-ADXR'
    """
    cols = list(df.columns)

    insert_after = "SIGNAL-Super-MOST-ADXR"
    new_cols = ["Qty", "Price", "USD Value", "USDC Value"]

    # remove new cols to reinsert
    cols_wo_new = [c for c in cols if c not in new_cols]

    # move bar time to end later
    bar_time_present = "Bar Time" in cols_wo_new
    if bar_time_present:
        cols_wo_new.remove("Bar Time")

    # insert after signal col
    if insert_after in cols_wo_new:
        idx = cols_wo_new.index(insert_after) + 1
        cols_wo_new = cols_wo_new[:idx] + new_cols + cols_wo_new[idx:]
    else:
        # fallback: append them near start
        cols_wo_new = cols_wo_new[:4] + new_cols + cols_wo_new[4:]

    # put Bar Time at end
    if bar_time_present:
        cols_wo_new.append("Bar Time")

    # keep only existing columns
    cols_final = [c for c in cols_wo_new if c in df.columns]
    return df.reindex(columns=cols_final)
