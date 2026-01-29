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
    import threading
    
    result = {"data": None, "error": None}
    
    def download_worker():
        try:
            result["data"] = yf.download(ticker, period=f"{lookback_days}d", interval="60m", progress=False)
        except Exception as e:
            result["error"] = str(e)
    
    try:
        # Run download in a thread with 30-second timeout
        thread = threading.Thread(target=download_worker)
        thread.daemon = True
        thread.start()
        thread.join(timeout=30)
        
        if thread.is_alive():
            print(f"⏱️  Ticker {ticker}: Timeout after 30 seconds - skipping")
            return None
        
        if result["error"]:
            print(f"❌ Ticker {ticker}: Error - {result['error']}")
            return None
        
        raw = result["data"]
        if raw is None or raw.empty:
            print(f"⚠️  Ticker {ticker}: No data from Yahoo Finance")
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

        print(f"✅ Ticker {ticker} length= {len(df_4h)}")
        if df_4h.empty or len(df_4h) < 240:
            print(f"⚠️  Ticker {ticker}: Insufficient data (need 240 bars, got {len(df_4h)})")
            return None
        return df_4h
        
    except Exception as e:
        print(f"❌ Ticker {ticker}: Error fetching data - {str(e)}")
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

    print("In build_crypto_signals_table ")

    for t in tickers:
        try:
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
        except Exception as e:
            print(f"❌ Error processing ticker {t}: {str(e)}")
            continue


# 2) extra geckoterminal pools (LVL/A0X/GAME etc)
    for sym, pool_url in gecko_pools.items():
        try:
            base = fetch_geckoterminal_4h_df_from_pool_url(pool_url)
            if base is None:
                print(f"⚠️  Gecko pool {sym}: No data from GeckoTerminal")
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
            print(f"✅ Gecko pool {sym}: Successfully processed")
        except Exception as e:
            print(f"❌ Error processing gecko pool {sym}: {str(e)}")
            continue


    df_out = pd.DataFrame(rows)

    # Reorder final columns if you want a standard layout
    if not df_out.empty:
        all_cols = list(df_out.columns)
        # your preferred order (adjust as needed)
        final_order = FINAL_COLUMN_ORDER.copy()
        final_order = [c for c in final_order if c in all_cols]
        leftover = [c for c in all_cols if c not in final_order]
        df_out = df_out[final_order + leftover]

    return df_out


def fetch_geckoterminal_4h_df_from_pool_url(pool_url: str, lookback_days: int = 70) -> pd.DataFrame | None:
    """
    Given a GeckoTerminal pool URL like:
      https://www.geckoterminal.com/solana/pools/GiRyo4r3kREH8oRCe9GoJJARZuGo4ksto6xXvUok4wdd
    parse out the network + pool address, call the GeckoTerminal API, and return 4H OHLCV.
    """
    try:
        match = re.search(r"geckoterminal\.com/(\w+)/pools/([A-Za-z0-9]+)", pool_url)
        if not match:
            print(f"⚠️  Invalid GeckoTerminal URL format: {pool_url}")
            return None
        network = match.group(1)
        pool_address = match.group(2)

        # GeckoTerminal timeframe mapping
        timeframe = "hour"  # or "minute", "day"
        # aggregate = 4 means 4-hour candles
        aggregate = 4
        # currency = 'usd'

        limit = lookback_days * 6 + 50  # 6 four-hour bars per day
        url = (
            f"https://api.geckoterminal.com/api/v2/networks/{network}/pools/{pool_address}/ohlcv/{timeframe}"
            f"?aggregate={aggregate}&limit={limit}&currency=usd"
        )

        headers = _coingecko_headers()
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            print(f"⚠️  GeckoTerminal API returned status {resp.status_code} for {pool_url}")
            return None

        data = resp.json()
        ohlcv_list = data.get("data", {}).get("attributes", {}).get("ohlcv_list", [])
        if not ohlcv_list:
            print(f"⚠️  No OHLCV data from GeckoTerminal for {pool_url}")
            return None

        rows = []
        for candle in ohlcv_list:
            # candle format: [timestamp, open, high, low, close, volume]
            if len(candle) < 6:
                continue
            ts = candle[0]
            o, h, l, c, v = float(candle[1]), float(candle[2]), float(candle[3]), float(candle[4]), float(candle[5])
            dt = pd.to_datetime(ts, unit="s", utc=True)
            rows.append({
                "timestamp": dt,
                "Open": o,
                "High": h,
                "Low": l,
                "Close": c,
                "Volume": v,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            print(f"⚠️  Empty DataFrame from GeckoTerminal for {pool_url}")
            return None

        df = df.set_index("timestamp").sort_index()
        df = df[["High", "Low", "Close", "Volume"]].dropna()

        if len(df) < 240:
            print(f"⚠️  Insufficient data from GeckoTerminal (need 240 bars, got {len(df)}) for {pool_url}")
            return None

        print(f"✅ GeckoTerminal {network}/{pool_address}: {len(df)} bars fetched")
        return df

    except Exception as e:
        print(f"❌ Error fetching GeckoTerminal data from {pool_url}: {str(e)}")
        return None


def fetch_coingecko_global() -> dict:
    url = "https://api.coingecko.com/api/v3/global"
    headers = _coingecko_headers()
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json().get("data", {})


def fetch_total_mcap_history_coingecko(days: int = 900) -> pd.DataFrame:
    url = f"https://api.coingecko.com/api/v3/global/market_cap_chart?days={days}"
    headers = _coingecko_headers()
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "market_cap_chart" not in data or "market_cap" not in data["market_cap_chart"]:
        raise ValueError("No market_cap_chart data in CoinGecko response")
    points = data["market_cap_chart"]["market_cap"]
    rows = []
    for ts_ms, mcap in points:
        dt = pd.to_datetime(ts_ms, unit="ms", utc=True)
        rows.append({"date": dt, "total_mcap": mcap})
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df


def fetch_total_mcap_history_coinmarketcap(days: int = 900) -> pd.DataFrame:
    end_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = end_ts - (days * 86400)
    url = "https://api.coinmarketcap.com/data-api/v3/global-metrics/quotes/historical"
    params = {"format": "chart_crypto_details", "interval": "1d", "timeStart": start_ts, "timeEnd": end_ts}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    points = data.get("data", {}).get("points", {})
    if not points:
        raise ValueError("No points data from CMC")
    rows = []
    for ts_str, val in points.items():
        dt = pd.to_datetime(int(ts_str), unit="s", utc=True)
        mcap = val.get("v", [0])[0]
        rows.append({"date": dt, "total_mcap": mcap})
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df


def compute_total_vs_200ma(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    df = df.sort_index()
    df["MA200"] = df["total_mcap"].rolling(200, min_periods=1).mean()
    last = df.iloc[-1]
    total_mcap = last["total_mcap"]
    ma200 = last["MA200"]
    above = total_mcap > ma200
    df["Below200"] = df["total_mcap"] < df["MA200"]
    consec_below = 0
    if not above:
        for i in range(len(df) - 1, -1, -1):
            if df.iloc[i]["Below200"]:
                consec_below += 1
            else:
                break
    return {
        "total_mcap": total_mcap,
        "ma200": ma200,
        "above_200ma": above,
        "consecutive_days_below": consec_below,
    }


def fetch_altcoin_season_index() -> dict:
    url = "https://www.blockchaincenter.net/api/altcoin_season/"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data


# ====================================================================================
# HELPER: enrich_crypto_portfolio_fields
# ====================================================================================
def enrich_crypto_portfolio_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with 'Ticker', add columns:
      - Qty
      - Price
      - USD Value (Qty * Price)
      - USDC Value (stablecoin balance in that wallet)
    Returns a new DataFrame with these columns inserted after 'SIGNAL-Super-MOST-ADXR'.
    """

    print("In enrich_crypto_portfolio_fields function")
    # Import balance functions - these may not exist in all setups
    try:
        from core.balances.solana import get_wallet_token_balances_by_mint
    except ImportError:
        get_wallet_token_balances_by_mint = None
    
    try:
        from core.balances.evm import get_erc20_balance
    except ImportError:
        pass  # get_erc20_balance already imported at top
    
    try:
        from core.balances.hyperliquid import get_spot_balances, get_mid_prices
    except ImportError:
        get_spot_balances = None
        get_mid_prices = None

    # RPC lists
    ETH_RPC_URLS = [
        "https://eth.llamarpc.com",
        "https://ethereum.publicnode.com",
    ]
    BASE_RPC_URLS = [
        "https://mainnet.base.org",
        "https://base.llamarpc.com",
    ]
    BSC_RPC_URLS = [
        "https://bsc-dataseed.binance.org",
        "https://bsc-dataseed1.defibit.io",
    ]
    OPTIMISM_RPC_URLS = [
        "https://mainnet.optimism.io",
        "https://optimism.llamarpc.com",
    ]
    ARBITRUM_RPC_URLS = [
        "https://arb1.arbitrum.io/rpc",
        "https://arbitrum.llamarpc.com",
    ]
    ZKSYNC_RPC_URLS = [
        "https://mainnet.era.zksync.io",
    ]
    print("just before load asset registry")
    registry = load_asset_registry()
    
    # Debug: Check if registry loaded
    print(f"📋 Registry loaded: {len(registry) if registry else 0} entries")
    if not registry:
        print("⚠️  WARNING: Asset registry is empty! Wallet balances will be zero.")
        print("   Make sure ASSET_REGISTRY environment variable or secrets are set.")
    
    # Build sol_wallet_cache - it's just an empty dict that will be populated on-demand
    # by get_spl_token_balance_from_cache
    sol_wallet_cache = {}

    qty_list = []
    price_list = []
    usd_list = []
    usdc_list = []

    for _, row in df.iterrows():
        try:
            yahoo_ticker = row.get("Ticker", "")

            entry = find_registry_entry_by_yahoo_ticker(registry, yahoo_ticker)
            if not entry:
                print(f"⚠️  No registry entry found for {yahoo_ticker}")
                qty_list.append(np.nan)
                price_list.append(np.nan)
                usd_list.append(np.nan)
                usdc_list.append(np.nan)
                continue

            chain = entry.get("blockchain", "").lower()
            wallet = entry.get("wallet_address")
            token_contract = entry.get("token_contract")
            stable_contract = entry.get("stablecoin_contract")
            print("chain wallet token_contract stable_contract = ", chain, wallet, token_contract, stable_contract)



            # Price - wrap in try-except to handle failures gracefully
            price_dec = None
            try:
                price_dec = get_yahoo_price_usd(yahoo_ticker)
            except Exception as e:
                print(f"⚠️  Could not fetch price for {yahoo_ticker}: {e}")
                price_dec = None

            # Qty + USDC balances
            qty_dec = None
            usdc_dec = None

            try:
                
                if chain == "solana":
                    # SPL token balance
                    if wallet and token_contract and sol_wallet_cache is not None:
                        qty_dec = get_spl_token_balance_from_cache(wallet, token_contract, sol_wallet_cache)
                        debug = True
                        if debug and qty_dec != 0:
                            print(f"[SOL] {yahoo_ticker} | Wallet={wallet} | Qty={qty_dec}")

                    if wallet and stable_contract and sol_wallet_cache is not None:
                        usdc_dec = get_spl_token_balance_from_cache(wallet, stable_contract, sol_wallet_cache)
                        debug = True
                        if debug and usdc_dec != 0:
                            print(f"[SOL] USDC | Wallet={wallet} | Qty={usdc_dec}")




                elif chain == "ethereum":
                    if wallet and token_contract:
                        qty_dec = get_erc20_balance(wallet, token_contract, ETH_RPC_URLS)
                        debug = True
                        if debug and qty_dec != 0:
                            print(f"[ETH] {yahoo_ticker} | Wallet={wallet} | Qty={qty_dec}")
                    if wallet and stable_contract:
                        usdc_dec = get_erc20_balance(wallet, stable_contract, ETH_RPC_URLS)
                        debug = True
                        if debug and usdc_dec != 0:
                            print(f"[ETH] USDC | Wallet={wallet} | Qty={usdc_dec}")


                elif chain == "base":
                    if wallet and token_contract:
                        qty_dec = get_erc20_balance(wallet, token_contract, BASE_RPC_URLS)
                        debug = True
                        if debug and qty_dec != 0:
                            print(f"[BASE] {yahoo_ticker} | Wallet={wallet} | Qty={qty_dec}")
                    if wallet and stable_contract:
                        usdc_dec = get_erc20_balance(wallet, stable_contract, BASE_RPC_URLS)
                        debug = True
                        if debug and usdc_dec != 0:
                            print(f"[BASE] USDC | Wallet={wallet} | Qty={usdc_dec}")

                elif chain == "sui":
                    if wallet and token_contract:
                        qty_dec = get_coin_balance(wallet, token_contract)
                        debug = True
                        if debug and qty_dec != 0:
                            print(f"[SUI] {yahoo_ticker} | Wallet={wallet} | Qty={qty_dec}")                    
                    if wallet and stable_contract:
                        usdc_dec = get_coin_balance(wallet, stable_contract)
                        debug = True
                        if debug and usdc_dec != 0:
                            print(f"[SUI] USDC | Wallet={wallet} | Qty={usdc_dec}")
                elif chain in ("bsc", "bnb"):
                    if wallet and token_contract:
                        qty_dec = get_erc20_balance(wallet, token_contract, BSC_RPC_URLS)
                        debug = True
                        if debug and qty_dec != 0:
                            print(f"[BNB] {yahoo_ticker} | Wallet={wallet} | Qty={qty_dec}")    
                    if wallet and stable_contract:
                        usdc_dec = get_erc20_balance(wallet, stable_contract, BSC_RPC_URLS)
                        debug = True
                        if debug and usdc_dec != 0:
                            print(f"[BNB] USDT | Wallet={wallet} | Qty={usdc_dec}")                    
                elif chain == "hyperliquid":
                    if get_spot_balances and get_mid_prices:
                        state = get_spot_balances(wallet)
                        mids = get_mid_prices()

                        # state is a dict: {"HYPE": Decimal("123"), ...}
                        qty_dec = state.get("HYPE", Decimal("0"))
                        price_dec = mids.get("HYPE", Decimal("0"))

                elif chain == "optimism":
                    rpc_urls = OPTIMISM_RPC_URLS
                    if wallet and token_contract:
                        qty_dec = get_erc20_balance(wallet, token_contract, rpc_urls)
                        debug = True
                        if debug and qty_dec != 0:
                            print(f"[Optimism] {yahoo_ticker} | Wallet={wallet} | Qty={qty_dec}")                        
                    if wallet and stable_contract:
                        usdc_dec = get_erc20_balance(wallet, stable_contract, rpc_urls)
                        debug = True
                        if debug and usdc_dec != 0:
                            print(f"[OPTIMISM] USDC | Wallet={wallet} | Qty={usdc_dec}")
                elif chain == "arbitrum":
                    rpc_urls = ARBITRUM_RPC_URLS
                    if wallet and token_contract:
                        qty_dec = get_erc20_balance(wallet, token_contract, rpc_urls)
                        debug = True
                        if debug and qty_dec != 0:
                            print(f"[Arbitrum] {yahoo_ticker} | Wallet={wallet} | Qty={qty_dec}")    
                    if wallet and stable_contract:
                        usdc_dec = get_erc20_balance(wallet, stable_contract, rpc_urls)
                        debug = True
                        if debug and usdc_dec != 0:
                            print(f"[Arbitrum] USDC | Wallet={wallet} | Qty={usdc_dec}")

                elif chain == "zksync":
                    rpc_urls = ZKSYNC_RPC_URLS

                    if wallet and token_contract:
                        qty_dec = get_erc20_balance(wallet, token_contract, rpc_urls)
                        debug = True
                        if debug and qty_dec != 0:
                            print(f"[ZK] {yahoo_ticker} | Wallet={wallet} | Qty={qty_dec}")    
                    if wallet and stable_contract:
                        usdc_dec = get_erc20_balance(wallet, stable_contract, rpc_urls)
                        debug = True
                        if debug and usdc_dec != 0:
                            print(f"[ZK] USDC | Wallet={wallet} | Qty={usdc_dec}")

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
        
        except Exception as e:
            # If anything fails for this row, append NaN values and continue
            print(f"❌ Error enriching portfolio fields for {yahoo_ticker}: {e}")
            qty_list.append(np.nan)
            price_list.append(np.nan)
            usd_list.append(np.nan)
            usdc_list.append(np.nan)

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


# ====================================================================================
# NEW CONSOLIDATED FUNCTION: build_complete_crypto_table
# ====================================================================================
def build_complete_crypto_table(
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
    """
    ALL-IN-ONE function that:
    1. Builds crypto signals table
    2. Enriches with wallet balances (Qty, Price, USD Value, USDC Value)
    3. Adds Total Val, ALT%, ACTION columns
    4. Reorders columns
    5. Sorts by Total Val descending
    
    Returns the complete DataFrame ready for display or CSV export.
    """
    print("Called build_complete_crypto_table in crypto.py ")
    # Step 1: Build signals table
    df_crypto = build_crypto_signals_table(
        tickers,
        gecko_pools=gecko_pools,
        atr_period=atr_period,
        atr_multiplier=atr_multiplier,
        rsi_period=rsi_period,
        vol_lookback=vol_lookback,
        vol_multiplier=vol_multiplier,
        rsi_buy_threshold=rsi_buy_threshold,
        adxr_len=adxr_len,
        adxr_lenx=adxr_lenx,
        adxr_low_threshold=adxr_low_threshold,
        adxr_flat_eps=adxr_flat_eps,
    )
    
    if df_crypto.empty:
        print("df_crypto is empty ")
        return df_crypto
    
    # Step 2: Enrich with wallet balances
    df_crypto = enrich_crypto_portfolio_fields(df_crypto)
    
    # Step 3: Rename "USD Value" -> "ALT USD Val"
    if "USD Value" in df_crypto.columns:
        df_crypto = df_crypto.rename(columns={"USD Value": "ALT USD Val"})
    
    # Step 4: Ensure numeric columns
    for c in ["ALT USD Val", "USDC Value"]:
        if c in df_crypto.columns:
            df_crypto[c] = pd.to_numeric(df_crypto[c], errors="coerce").fillna(0.0)
    
    # Add missing columns if not present
    if "ALT USD Val" not in df_crypto.columns:
        df_crypto["ALT USD Val"] = 0.0
    if "USDC Value" not in df_crypto.columns:
        df_crypto["USDC Value"] = 0.0
    
    # Step 5: Compute Total Val and ALT%
    df_crypto["Total Val"] = df_crypto["ALT USD Val"] + df_crypto["USDC Value"]
    df_crypto["ALT%"] = np.where(
        df_crypto["Total Val"] > 0,
        (df_crypto["ALT USD Val"] / df_crypto["Total Val"]) * 100.0,
        0.0,
    ).round(2)
    
    # Step 6: Add ACTION column
    SIGNAL_COL = "SIGNAL-Super-MOST-ADXR"
    
    def _action_row(r):
        sig = str(r.get(SIGNAL_COL, "")).upper()
        alt_pct = float(pd.to_numeric(r.get("ALT%", 0.0), errors="coerce") or 0.0)
        
        # BUY signal but ALT exposure low -> buy ALT
        if sig == "BUY" and alt_pct < 50.0:
            return "🔴 BUY ALT"
        
        # EXIT signal but ALT exposure high -> sell ALT
        if sig == "EXIT" and alt_pct > 50.0:
            return "🔴 SELL ALT"
        
        return ""  # no action
    
    df_crypto["ACTION"] = df_crypto.apply(_action_row, axis=1)
    
    # Step 7: Reorder columns
    cols = list(df_crypto.columns)
    
    def _move_after(col_to_move, after_col):
        nonlocal cols
        if col_to_move in cols and after_col in cols:
            cols.remove(col_to_move)
            idx = cols.index(after_col) + 1
            cols.insert(idx, col_to_move)
    
    # Put ACTION right after SIGNAL
    _move_after("ACTION", SIGNAL_COL)
    # Move ALT% right after ALT USD Val
    _move_after("ALT%", "ALT USD Val")
    # Move Total Val right after USDC Value
    _move_after("Total Val", "USDC Value")
    
    df_crypto = df_crypto[cols]
    
    # Step 8: Sort by Total Val descending
    if "Total Val" in df_crypto.columns:
        df_crypto = df_crypto.sort_values("Total Val", ascending=False)
    
    return df_crypto