#!/usr/bin/env python3
"""
taostats.py — Bittensor Subnet Alpha Token Scanner & Signal Generator
NOTE: If running and seeing no output, try: python3 -u taostats.py --validate


Connects to the Taostats API to fetch subnet pool data, builds OHLCV-like
price history from the 7-day price snapshots, and runs our standard
Supertrend + MOST RSI + ADXR signal pipeline.

Also includes a persistent OHLCV collector that appends price snapshots to
a local CSV so we can build up enough history for proper backtesting over time.

Usage:
  # Scan all top subnets (one-shot signal scan)
  python3 taostats.py --scan

  # Scan specific subnets by netuid
  python3 taostats.py --scan --netuids 4 64 120 51

  # Collect a price snapshot (run via cron every 4 hours to build OHLCV)
  python3 taostats.py --collect

  # Run signals from collected OHLCV history (once you have 100+ bars)
  python3 taostats.py --signals --netuids 4

  # Show pool details for a single subnet
  python3 taostats.py --pool 4

Environment:
  TAOSTATS_API_KEY    — your taostats.io API key (e.g. "tao-xxxxx:yyyyy")

Reuses: core.indicators.apply_indicators, core.signals.signal_super_most_adxr
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Load .env (same pattern as bot.py)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed — rely on shell environment variables

# Force unbuffered output so prints appear immediately
import functools
print = functools.partial(print, flush=True)

# ── Check required packages ──
_missing = []
try:
    import numpy as np
except ImportError:
    _missing.append("numpy")
try:
    import pandas as pd
except ImportError:
    _missing.append("pandas")
try:
    import requests
except ImportError:
    _missing.append("requests")

if _missing:
    print(f"❌ Missing packages: {', '.join(_missing)}")
    print(f"   Install with: pip install {' '.join(_missing)}")
    sys.exit(1)

# ── Project imports (same indicator/signal stack as crypto.py / stocks.py) ──
try:
    from core.indicators import apply_indicators
    from core.signals import signal_super_most_adxr
    from core.config import INDICATOR_PARAMS
    HAS_CORE = True
except Exception as _import_err:
    HAS_CORE = False
    print(f"⚠️  core imports not available ({_import_err}) — running in API-only mode")
    print("   Place this file alongside your project or adjust PYTHONPATH.\n")


# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════
API_BASE = "https://api.taostats.io"
API_KEY = os.getenv("TAOSTATS_API_KEY", "")

# Where to store collected OHLCV snapshots
DATA_DIR = os.getenv("TAOSTATS_DATA_DIR", "./taostats_data")
OHLCV_FILE_TEMPLATE = os.path.join(DATA_DIR, "ohlcv_sn{netuid}.csv")

# Rate limit: free tier = 5 calls/min
RATE_LIMIT_DELAY = 1.5  # seconds between API calls


# ═══════════════════════════════════════════════════════════════════
# API Client
# ═══════════════════════════════════════════════════════════════════
class TaostatsAPI:
    """Lightweight client for the Taostats dtao API."""

    def __init__(self, api_key: str = ""):
        self.base_url = API_BASE
        self.api_key = api_key or API_KEY
        if not self.api_key:
            print("❌ TAOSTATS_API_KEY not set. Export it:")
            print('   export TAOSTATS_API_KEY="tao-xxxxx:yyyyy"')
            sys.exit(1)

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        headers = {
            "Authorization": self.api_key,
            "accept": "application/json",
        }
        url = f"{self.base_url}{endpoint}"
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Taostats API error {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def pool_latest(self, netuid: int) -> dict:
        """
        GET /api/dtao/pool/latest/v1?netuid={N}
        Returns: price, root_prop, fear_and_greed_index, seven_day_prices,
                 volume, price_change_1_hour/1_day/1_week/1_month, mcap, etc.
        """
        return self._get("/api/dtao/pool/latest/v1", {"netuid": netuid})

    def pool_latest_all(self) -> dict:
        """Get pool data for ALL subnets at once."""
        return self._get("/api/dtao/pool/latest/v1")

    def tao_flow(self, netuid: int, timestamp_start: int = None, timestamp_end: int = None) -> dict:
        """
        GET /api/dtao/tao_flow/v1
        Historical tao flow into/out of subnet pool.
        """
        params = {"netuid": netuid}
        if timestamp_start:
            params["timestamp_start"] = timestamp_start
        if timestamp_end:
            params["timestamp_end"] = timestamp_end
        return self._get("/api/dtao/tao_flow/v1", params)

    def subnet_info(self, netuid: int = None) -> dict:
        """GET /api/subnet/latest/v1 — subnet parameters, emissions, net flows."""
        params = {}
        if netuid is not None:
            params["netuid"] = netuid
        return self._get("/api/subnet/latest/v1", params)

    def stake_balance(self, coldkey: str) -> dict:
        """GET /api/dtao/stake_balance/latest/v1 — your positions across subnets."""
        return self._get("/api/dtao/stake_balance/latest/v1", {"coldkey": coldkey})

    def validate_key(self) -> dict:
        """Check API key validity and usage."""
        headers = {
            "Authorization": self.api_key,
            "accept": "application/json",
        }
        resp = requests.get(
            "https://management-api.taostats.io/api/v1/key/validate",
            headers=headers, timeout=15,
        )
        return resp.json()

    def get_tao_price_usd(self) -> float:
        """Fetch current TAO/USD price from the stats endpoint."""
        try:
            # Try Yahoo first (fast, no API call needed)
            import yfinance as yf
            data = yf.download("TAO22974-USD", period="1d", interval="1d", progress=False)
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)
                return float(data["Close"].iloc[-1])
        except Exception:
            pass
        # Fallback: use CoinGecko
        try:
            resp = requests.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": "bittensor", "vs_currencies": "usd"},
                timeout=10,
            )
            if resp.status_code == 200:
                return float(resp.json().get("bittensor", {}).get("usd", 0))
        except Exception:
            pass
        return 0.0


# ═══════════════════════════════════════════════════════════════════
# Parse pool data into structured format
# ═══════════════════════════════════════════════════════════════════
def parse_pool_data(pool_item: dict, tao_price_usd: float = 0.0) -> dict:
    """Extract key trading fields from a pool/latest response item."""
    price_tao = float(pool_item.get("price", 0) or 0)
    # The pool endpoint doesn't include tao_price_usd — caller must provide it
    price_usd = price_tao * tao_price_usd

    # Volume fields: actual API uses tao_volume_24_hr, not volume_24hr
    vol_tao = float(pool_item.get("tao_volume_24_hr", 0) or 0)
    vol_usd = vol_tao * tao_price_usd

    # Market cap: API returns in rao (raw units) — divide by 1e9 for TAO
    mcap_raw = float(pool_item.get("market_cap", 0) or 0)
    # If mcap_raw is very large (rao), convert; if already in TAO, use as-is
    mcap_tao = mcap_raw / 1e9 if mcap_raw > 1e12 else mcap_raw
    mcap_usd = mcap_tao * tao_price_usd

    # Liquidity
    liq_raw = float(pool_item.get("liquidity", 0) or 0)
    liq_tao = liq_raw / 1e9 if liq_raw > 1e12 else liq_raw
    liq_usd = liq_tao * tao_price_usd

    # Pool contents: total_tao / total_alpha (not tao_in / alpha_in)
    alpha_in = float(pool_item.get("alpha_in_pool", 0) or 0)
    alpha_in = alpha_in / 1e9 if alpha_in > 1e12 else alpha_in
    total_tao = float(pool_item.get("total_tao", 0) or 0)
    total_tao = total_tao / 1e9 if total_tao > 1e12 else total_tao

    return {
        "netuid":              int(pool_item.get("netuid", 0)),
        "name":                pool_item.get("name", pool_item.get("subnet_name", "")),
        "symbol":              pool_item.get("symbol", ""),
        "price_tao":           price_tao,
        "price_usd":           price_usd,
        "tao_price_usd":       tao_price_usd,
        "emission_pct":        float(pool_item.get("emission", 0) or 0) * 100,
        "root_prop":           float(pool_item.get("root_prop", 0) or 0),
        "mcap_tao":            mcap_tao,
        "mcap_usd":            mcap_usd,
        "volume_24h_tao":      vol_tao,
        "volume_24h_usd":      vol_usd,
        "liquidity_tao":       liq_tao,
        "liquidity_usd":       liq_usd,
        "alpha_in_pool":       alpha_in,
        "tao_in_pool":         total_tao,
        "price_change_1h":     float(pool_item.get("price_change_1_hour", 0) or 0),
        "price_change_1d":     float(pool_item.get("price_change_1_day", 0) or 0),
        "price_change_1w":     float(pool_item.get("price_change_1_week", 0) or 0),
        "price_change_1m":     float(pool_item.get("price_change_1_month", 0) or 0),
        "fear_greed_index":    float(pool_item.get("fear_and_greed_index", 50) or 50),
        "fear_greed_sentiment": pool_item.get("fear_and_greed_sentiment", ""),
        "seven_day_prices":    pool_item.get("seven_day_prices", []),
        # Extra trading fields
        "highest_24h":         float(pool_item.get("highest_price_24_hr", 0) or 0),
        "lowest_24h":          float(pool_item.get("lowest_price_24_hr", 0) or 0),
        "buys_24h":            int(pool_item.get("buys_24_hr", 0) or 0),
        "sells_24h":           int(pool_item.get("sells_24_hr", 0) or 0),
        "buyers_24h":          int(pool_item.get("buyers_24_hr", 0) or 0),
        "sellers_24h":         int(pool_item.get("sellers_24_hr", 0) or 0),
    }


# ═══════════════════════════════════════════════════════════════════
# Build pseudo-OHLCV from 7-day price array
# ═══════════════════════════════════════════════════════════════════
def build_ohlcv_from_7day(seven_day_prices: list, current_price: float) -> pd.DataFrame:
    """
    The pool/latest endpoint returns `seven_day_prices` — a list of ~42
    data points spanning 7 days (roughly 4-hour intervals).

    We group these into 4H-like bars to build pseudo-OHLCV for indicator
    computation. Volume is not available per-bar so we fill with NaN.

    NOTE: This gives us only ~42 bars which is NOT enough for proper
    Supertrend/ADXR (need ~100+). This is a quick-look preview only.
    For real backtesting, use the --collect mode to build OHLCV over time.
    """
    if not seven_day_prices or len(seven_day_prices) < 5:
        return pd.DataFrame()

    # seven_day_prices format: list of [timestamp, price] or just prices
    rows = []
    for item in seven_day_prices:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            ts = item[0]
            price = float(item[1])
            if isinstance(ts, str):
                dt = pd.to_datetime(ts)
            else:
                dt = pd.to_datetime(ts, unit="s", utc=True)
            rows.append({"timestamp": dt, "price": price})
        elif isinstance(item, dict):
            ts = item.get("timestamp", item.get("time", ""))
            price = float(item.get("price", item.get("value", 0)))
            dt = pd.to_datetime(ts)
            rows.append({"timestamp": dt, "price": price})
        elif isinstance(item, (int, float)):
            # Just a list of prices without timestamps — estimate timestamps
            rows.append({"price": float(item)})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # If no timestamps, generate synthetic ones (4h apart going backwards)
    if "timestamp" not in df.columns:
        now = pd.Timestamp.now(tz="UTC")
        n = len(df)
        df["timestamp"] = [now - timedelta(hours=4 * (n - 1 - i)) for i in range(n)]

    df = df.set_index("timestamp").sort_index()

    # Each row is already roughly a 4H snapshot — treat as a "candle"
    # where O=H=L=C=price (no intra-bar data available)
    df["Open"] = df["price"]
    df["High"] = df["price"]
    df["Low"] = df["price"]
    df["Close"] = df["price"]
    df["Volume"] = 0.0  # no per-bar volume from this endpoint

    return df[["Open", "High", "Low", "Close", "Volume"]]


# ═══════════════════════════════════════════════════════════════════
# OHLCV Collector — append snapshots to CSV for building history
# ═══════════════════════════════════════════════════════════════════
def collect_price_snapshot(api: TaostatsAPI, netuids: list[int] | None = None):
    """
    Fetch current pool data and append a row to per-subnet OHLCV CSVs.

    Run this via cron every 4 hours to build up proper OHLCV history:
      0 */4 * * * cd /path/to/project && python3 taostats.py --collect

    Over time you'll accumulate enough bars (100+) to run proper
    Supertrend / MOST RSI / ADXR signals.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    tao_usd = api.get_tao_price_usd()

    if netuids:
        # Fetch individual pools
        for netuid in netuids:
            try:
                result = api.pool_latest(netuid)
                data_list = result.get("data", [])
                if not data_list:
                    print(f"⚠️  SN{netuid}: No pool data")
                    continue
                pool = parse_pool_data(data_list[0], tao_price_usd=tao_usd)
                _append_snapshot(pool)
                time.sleep(RATE_LIMIT_DELAY)
            except Exception as e:
                print(f"❌ SN{netuid}: {e}")
    else:
        # Fetch all at once
        result = api.pool_latest_all()
        data_list = result.get("data", [])
        for item in data_list:
            try:
                pool = parse_pool_data(item, tao_price_usd=tao_usd)
                _append_snapshot(pool)
            except Exception as e:
                print(f"❌ SN{item.get('netuid','?')}: {e}")

    print(f"\n✅ Snapshots collected at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")


def _append_snapshot(pool: dict):
    """Append a single price snapshot to the subnet's CSV file."""
    netuid = pool["netuid"]
    path = OHLCV_FILE_TEMPLATE.format(netuid=netuid)

    now = datetime.now(timezone.utc)
    row = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "Open": pool["price_tao"],
        "High": pool["price_tao"],
        "Low": pool["price_tao"],
        "Close": pool["price_tao"],
        "Volume": pool["volume_24h_tao"],
        "price_usd": pool["price_usd"],
        "mcap_usd": pool["mcap_usd"],
        "emission_pct": pool["emission_pct"],
        "root_prop": pool["root_prop"],
        "fear_greed": pool["fear_greed_index"],
    }

    write_header = not os.path.exists(path)
    with open(path, "a") as f:
        if write_header:
            f.write(",".join(row.keys()) + "\n")
        f.write(",".join(str(v) for v in row.values()) + "\n")

    print(f"  📸 SN{netuid:>3d} ({pool['name']:<20s}) | τ{pool['price_tao']:.6f} | ${pool['price_usd']:.4f}")


# ═══════════════════════════════════════════════════════════════════
# Load collected OHLCV and resample to 4H bars
# ═══════════════════════════════════════════════════════════════════
def load_collected_ohlcv(netuid: int, interval: str = "4h") -> pd.DataFrame | None:
    """
    Load OHLCV from collected snapshots CSV and resample into proper candles.

    If snapshots were collected more frequently than 4h, this resamples them.
    If collected at exactly 4h intervals, each row becomes one bar.
    """
    path = OHLCV_FILE_TEMPLATE.format(netuid=netuid)
    if not os.path.exists(path):
        print(f"⚠️  No collected data for SN{netuid}. Run --collect first.")
        return None

    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    if len(df) < 10:
        print(f"⚠️  SN{netuid}: Only {len(df)} snapshots — need more data. Keep running --collect.")
        return None

    # Resample to desired interval
    resample_map = {"1h": "1h", "4h": "4h", "1d": "1D"}
    rule = resample_map.get(interval, "4h")

    resampled = df[["Open", "High", "Low", "Close", "Volume"]].resample(rule)
    ohlcv = pd.DataFrame({
        "High":   resampled["High"].max(),
        "Low":    resampled["Low"].min(),
        "Close":  resampled["Close"].last(),
        "Volume": resampled["Volume"].mean(),  # average since we get 24h vol
    }).dropna()

    # For proper OHLCV, set High/Low from multiple snapshots within the bar
    # If only 1 snapshot per bar, H=L=C (same as 7-day preview)
    print(f"📊 SN{netuid}: {len(ohlcv)} bars loaded from {len(df)} snapshots ({interval})")
    return ohlcv


# ═══════════════════════════════════════════════════════════════════
# Signal Scanner — quick-look from 7-day data + pool metadata
# ═══════════════════════════════════════════════════════════════════
def scan_subnets(api: TaostatsAPI, netuids: list[int] | None = None):
    """
    Scan subnets and display trading-relevant data.

    Uses 7-day price snapshots for a quick-look signal preview.
    For real signals, use --signals with collected OHLCV data.
    """
    print("=" * 110)
    print("🧠 BITTENSOR SUBNET ALPHA SCANNER")
    print(f"   {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 110)

    # Fetch TAO/USD price once for all subnets
    print("📡 Fetching TAO/USD price...")
    tao_usd = api.get_tao_price_usd()
    if tao_usd > 0:
        print(f"   TAO = ${tao_usd:.2f}")
    else:
        print("   ⚠️  Could not fetch TAO/USD — USD values will show as $0")

    if netuids:
        pools = []
        for netuid in netuids:
            try:
                result = api.pool_latest(netuid)
                data_list = result.get("data", [])
                if data_list:
                    pools.append(parse_pool_data(data_list[0], tao_price_usd=tao_usd))
                time.sleep(RATE_LIMIT_DELAY)
            except Exception as e:
                print(f"❌ SN{netuid}: {e}")
    else:
        try:
            result = api.pool_latest_all()
            data_list = result.get("data", [])
            pools = [parse_pool_data(item, tao_price_usd=tao_usd) for item in data_list]
        except Exception as e:
            print(f"❌ Failed to fetch pool data: {e}")
            return

    # Sort by market cap descending
    pools.sort(key=lambda p: p["mcap_usd"], reverse=True)

    # ── Header ──
    print(f"\n{'#':>3s}  {'Subnet':<22s}  {'SN':>4s}  {'Price(τ)':>12s}  "
          f"{'Price($)':>10s}  {'MCap($)':>12s}  {'Vol24h($)':>12s}  "
          f"{'1H%':>7s}  {'1D%':>7s}  {'1W%':>7s}  {'1M%':>7s}  "
          f"{'Emis%':>6s}  {'RootP':>6s}  {'F&G':>5s}  {'Sentiment':>10s}")
    print("-" * 170)

    rows_for_df = []

    for i, pool in enumerate(pools):
        # Skip root (netuid 0) and very small subnets
        if pool["netuid"] == 0:
            continue

        # Color-code momentum
        def _fmt_pct(v):
            s = f"{v:+.2f}%"
            return s

        print(f"{i+1:>3d}  {pool['name']:<22s}  {pool['netuid']:>4d}  "
              f"τ{pool['price_tao']:>10.6f}  "
              f"${pool['price_usd']:>9.4f}  "
              f"${pool['mcap_usd']:>10,.0f}  "
              f"${pool['volume_24h_usd']:>10,.0f}  "
              f"{_fmt_pct(pool['price_change_1h']):>7s}  "
              f"{_fmt_pct(pool['price_change_1d']):>7s}  "
              f"{_fmt_pct(pool['price_change_1w']):>7s}  "
              f"{_fmt_pct(pool['price_change_1m']):>7s}  "
              f"{pool['emission_pct']:>5.2f}%  "
              f"{pool['root_prop']:>5.1f}%  "
              f"{pool['fear_greed_index']:>5.1f}  "
              f"{pool['fear_greed_sentiment']:>10s}")

        # ── Try running indicators on 7-day data (preview only) ──
        signal_preview = "—"
        if HAS_CORE and pool["seven_day_prices"]:
            try:
                ohlcv_7d = build_ohlcv_from_7day(pool["seven_day_prices"], pool["price_tao"])
                if len(ohlcv_7d) >= 20:
                    # With only ~42 bars, indicators won't be fully warmed up
                    # but we can get a directional read
                    ind = apply_indicators(ohlcv_7d, **_get_indicator_params())
                    last = ind.iloc[-1]
                    st_sig = str(last.get("Supertrend_Signal", "?"))
                    most_sig = str(last.get("MOST_Signal", "?"))
                    adxr_state = str(last.get("ADXR_State", "?"))
                    signal_preview = signal_super_most_adxr(st_sig, most_sig, adxr_state)
            except Exception as e:
                signal_preview = f"ERR:{str(e)[:15]}"

        rows_for_df.append({
            "netuid": pool["netuid"],
            "name": pool["name"],
            "price_tao": pool["price_tao"],
            "price_usd": pool["price_usd"],
            "mcap_usd": pool["mcap_usd"],
            "1H%": pool["price_change_1h"],
            "1D%": pool["price_change_1d"],
            "1W%": pool["price_change_1w"],
            "1M%": pool["price_change_1m"],
            "emission%": pool["emission_pct"],
            "root_prop": pool["root_prop"],
            "fear_greed": pool["fear_greed_index"],
            "sentiment": pool["fear_greed_sentiment"],
            "signal_preview": signal_preview,
        })

    # ── Summary ──
    print("-" * 170)
    print(f"\n📊 Scanned {len(pools)} subnets")

    if HAS_CORE:
        # Print signal preview summary
        signals = [r["signal_preview"] for r in rows_for_df if r["signal_preview"] not in ("—", "")]
        if signals:
            print(f"\n🔮 7-Day Signal Preview (limited data — use --collect for proper signals):")
            for r in rows_for_df:
                if r["signal_preview"] not in ("—", ""):
                    emoji = {"BUY": "🟢", "HOLD": "🟡", "EXIT": "🔴", "STANDDOWN": "⚪"}.get(
                        r["signal_preview"], "❓"
                    )
                    print(f"  {emoji} SN{r['netuid']:>3d} {r['name']:<20s} → {r['signal_preview']}")

    return pd.DataFrame(rows_for_df)


# ═══════════════════════════════════════════════════════════════════
# Full Signal Run from Collected OHLCV
# ═══════════════════════════════════════════════════════════════════
def run_signals_from_collected(netuids: list[int], interval: str = "4h"):
    """
    Run proper Supertrend + MOST RSI + ADXR signals from collected OHLCV data.
    Requires ~100+ bars collected via --collect over time.
    """
    if not HAS_CORE:
        print("❌ core.indicators / core.signals required for signal computation.")
        return

    print("=" * 90)
    print("🧠 BITTENSOR SUBNET SIGNALS (from collected OHLCV)")
    print(f"   Interval: {interval} | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 90)

    params = _get_indicator_params()
    results = []

    for netuid in netuids:
        ohlcv = load_collected_ohlcv(netuid, interval)
        if ohlcv is None or len(ohlcv) < 50:
            print(f"⚠️  SN{netuid}: Need at least 50 bars (have {len(ohlcv) if ohlcv is not None else 0})")
            continue

        try:
            df = apply_indicators(ohlcv, **params)
            last = df.iloc[-1]

            st_sig = str(last.get("Supertrend_Signal", "SELL"))
            most_sig = str(last.get("MOST_Signal", "SELL"))
            adxr_state = str(last.get("ADXR_State", "FLAT"))
            final = signal_super_most_adxr(st_sig, most_sig, adxr_state)

            emoji = {"BUY": "🟢", "HOLD": "🟡", "EXIT": "🔴", "STANDDOWN": "⚪"}.get(final, "❓")

            print(f"\n{emoji} SN{netuid}")
            print(f"  {'Close':.<20s} {float(last['Close']):.6f}")
            print(f"  {'Supertrend':.<20s} {float(last['Supertrend']):.6f}  ({st_sig})")
            print(f"  {'RSI':.<20s} {float(last['RSI']):.2f}")
            print(f"  {'MOST MA':.<20s} {float(last['MOST_MA']):.2f}")
            print(f"  {'MOST Line':.<20s} {float(last['MOST_Line']):.2f}  ({most_sig})")
            print(f"  {'ADXR State':.<20s} {adxr_state}")
            print(f"  {'SIGNAL':.<20s} {final}")
            print(f"  {'Bars':.<20s} {len(df)}")
            print(f"  {'Period':.<20s} {df.index[0]} → {df.index[-1]}")

            results.append({
                "netuid": netuid,
                "close": float(last["Close"]),
                "supertrend_signal": st_sig,
                "most_signal": most_sig,
                "adxr_state": adxr_state,
                "final_signal": final,
                "rsi": float(last["RSI"]),
                "bars": len(df),
            })
        except Exception as e:
            print(f"❌ SN{netuid}: {e}")

    if results:
        print(f"\n{'=' * 90}")
        df_out = pd.DataFrame(results)
        print(df_out.to_string(index=False))
    else:
        print("\n⚠️  No signals generated. Collect more OHLCV data with --collect.")

    return results


# ═══════════════════════════════════════════════════════════════════
# Show single pool details
# ═══════════════════════════════════════════════════════════════════
def show_pool(api: TaostatsAPI, netuid: int):
    """Display detailed pool info for a single subnet."""
    print("📡 Fetching TAO/USD price...")
    tao_usd = api.get_tao_price_usd()
    if tao_usd == 0:
        print("⚠️  Could not fetch TAO/USD — USD values will show as $0")

    result = api.pool_latest(netuid)
    data_list = result.get("data", [])
    if not data_list:
        print(f"❌ No data for SN{netuid}")
        return

    pool = parse_pool_data(data_list[0], tao_price_usd=tao_usd)

    print(f"\n{'=' * 60}")
    print(f"🧠 SUBNET {netuid}: {pool['name']} ({pool['symbol']})")
    print(f"{'=' * 60}")
    print(f"  {'Price (TAO)':.<25s} τ{pool['price_tao']:.8f}")
    print(f"  {'Price (USD)':.<25s} ${pool['price_usd']:.6f}")
    print(f"  {'TAO Price':.<25s} ${pool['tao_price_usd']:.2f}")
    print(f"  {'Market Cap':.<25s} ${pool['mcap_usd']:,.0f}  (τ{pool['mcap_tao']:,.0f})")
    print(f"  {'Volume 24h':.<25s} ${pool['volume_24h_usd']:,.0f}  (τ{pool['volume_24h_tao']:,.2f})")
    print(f"  {'Liquidity':.<25s} ${pool['liquidity_usd']:,.0f}  (τ{pool['liquidity_tao']:,.2f})")
    print(f"  {'Emission':.<25s} {pool['emission_pct']:.2f}%")
    print(f"  {'Root Proportion':.<25s} {pool['root_prop']:.1f}%")
    print(f"  {'Alpha in Pool':.<25s} {pool['alpha_in_pool']:,.0f}")
    print(f"  {'TAO in Pool':.<25s} {pool['tao_in_pool']:,.2f}")
    print(f"\n  Price Changes:")
    print(f"    {'1 Hour':.<20s} {pool['price_change_1h']:+.2f}%")
    print(f"    {'1 Day':.<20s} {pool['price_change_1d']:+.2f}%")
    print(f"    {'1 Week':.<20s} {pool['price_change_1w']:+.2f}%")
    print(f"    {'1 Month':.<20s} {pool['price_change_1m']:+.2f}%")
    print(f"\n  24h Range: τ{pool['lowest_24h']:.8f} — τ{pool['highest_24h']:.8f}")
    print(f"  24h Trades: {pool['buys_24h']} buys / {pool['sells_24h']} sells  "
          f"({pool['buyers_24h']} buyers / {pool['sellers_24h']} sellers)")
    print(f"\n  Fear & Greed: {pool['fear_greed_index']:.1f} ({pool['fear_greed_sentiment']})")

    # 7-day price array info
    n_prices = len(pool["seven_day_prices"])
    print(f"\n  7-Day Price History: {n_prices} data points")
    if n_prices > 0:
        prices_raw = pool["seven_day_prices"]
        # Try to extract just the price values
        prices = []
        for item in prices_raw:
            if isinstance(item, (list, tuple)):
                prices.append(float(item[1]) if len(item) >= 2 else float(item[0]))
            elif isinstance(item, dict):
                prices.append(float(item.get("price", item.get("value", 0))))
            elif isinstance(item, (int, float)):
                prices.append(float(item))

        if prices:
            print(f"    7D High: τ{max(prices):.8f}")
            print(f"    7D Low:  τ{min(prices):.8f}")
            print(f"    7D Δ:    {((prices[-1] / prices[0]) - 1) * 100:+.2f}%")

    # Show raw JSON for debugging
    print(f"\n  📦 Raw API fields available: {list(data_list[0].keys())}")

    return pool


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════
def _get_indicator_params() -> dict:
    """Get indicator params from core.config or use defaults."""
    if HAS_CORE:
        return {
            "atr_period":         INDICATOR_PARAMS["atr_period"],
            "atr_multiplier":     INDICATOR_PARAMS["atr_multiplier"],
            "rsi_period":         INDICATOR_PARAMS["rsi_period"],
            "vol_lookback":       INDICATOR_PARAMS["vol_lookback"],
            "adxr_len":           INDICATOR_PARAMS["adxr_len"],
            "adxr_lenx":          INDICATOR_PARAMS["adxr_lenx"],
            "adxr_low_threshold": INDICATOR_PARAMS["adxr_low_threshold"],
            "adxr_flat_eps":      INDICATOR_PARAMS["adxr_flat_eps"],
        }
    return {
        "atr_period": 10, "atr_multiplier": 3.0, "rsi_period": 14,
        "vol_lookback": 20, "adxr_len": 14, "adxr_lenx": 14,
        "adxr_low_threshold": 20.0, "adxr_flat_eps": 1e-6,
    }


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Bittensor Subnet Alpha Scanner & Signal Generator (Taostats API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate your API key
  python3 taostats.py --validate

  # Scan all subnets (quick-look)
  python3 taostats.py --scan

  # Scan specific subnets
  python3 taostats.py --scan --netuids 4 64 120 51

  # Pool details for one subnet
  python3 taostats.py --pool 4

  # Collect price snapshot (run via cron every 4h)
  python3 taostats.py --collect

  # Run signals from collected data
  python3 taostats.py --signals --netuids 4 64

Cron example (every 4 hours):
  0 */4 * * * cd /path/to/project && python3 taostats.py --collect >> taostats_collect.log 2>&1
""",
    )

    parser.add_argument("--validate", action="store_true", help="Validate API key")
    parser.add_argument("--scan", action="store_true", help="Scan subnets (quick-look signals)")
    parser.add_argument("--pool", type=int, metavar="NETUID", help="Show detailed pool info for one subnet")
    parser.add_argument("--collect", action="store_true", help="Collect price snapshot to build OHLCV history")
    parser.add_argument("--signals", action="store_true", help="Run signals from collected OHLCV data")
    parser.add_argument("--netuids", nargs="+", type=int, help="Specific subnet netuids to target")
    parser.add_argument("--interval", default="4h", choices=["1h", "4h", "1d"], help="Signal interval (default: 4h)")
    parser.add_argument("--balance", metavar="COLDKEY", help="Show your stake balances (provide coldkey SS58)")

    args = parser.parse_args()

    if not any([args.validate, args.scan, args.pool, args.collect, args.signals, args.balance]):
        parser.print_help()
        sys.exit(0)

    api = TaostatsAPI()

    if args.validate:
        try:
            info = api.validate_key()
            print("✅ API key is valid!")
            print(json.dumps(info, indent=2))
        except Exception as e:
            print(f"❌ API key validation failed: {e}")
        return

    if args.balance:
        try:
            result = api.stake_balance(args.balance)
            data = result.get("data", [])
            print(f"\n💰 Stake Balances for {args.balance[:12]}...")
            print(f"{'SN':>4s}  {'Balance(τ)':>14s}  {'Hotkey':>15s}")
            print("-" * 40)
            total_tao = 0
            for pos in data:
                balance_tao = float(pos.get("balance_as_tao", 0)) / 1_000_000_000
                total_tao += balance_tao
                hotkey_name = pos.get("hotkey_name", pos.get("hotkey", {}).get("ss58", "")[:12])
                print(f"{pos.get('netuid', '?'):>4}  τ{balance_tao:>12.4f}  {hotkey_name:>15s}")
            print("-" * 40)
            print(f"{'TOTAL':>4s}  τ{total_tao:>12.4f}")
        except Exception as e:
            print(f"❌ {e}")
        return

    if args.pool is not None:
        show_pool(api, args.pool)
        return

    if args.scan:
        scan_subnets(api, args.netuids)
        return

    if args.collect:
        collect_price_snapshot(api, args.netuids)
        return

    if args.signals:
        if not args.netuids:
            # Default: scan data dir for collected CSVs
            if os.path.exists(DATA_DIR):
                import re
                files = os.listdir(DATA_DIR)
                netuids_found = []
                for f in files:
                    m = re.match(r"ohlcv_sn(\d+)\.csv", f)
                    if m:
                        netuids_found.append(int(m.group(1)))
                if netuids_found:
                    args.netuids = sorted(netuids_found)
                    print(f"📂 Found collected data for subnets: {args.netuids}")
                else:
                    print("❌ No collected OHLCV data found. Run --collect first.")
                    return
            else:
                print("❌ No data directory. Run --collect first.")
                return
        run_signals_from_collected(args.netuids, args.interval)
        return


if __name__ == "__main__":
    main()