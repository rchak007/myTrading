# core/ohlcv_cache.py
"""
Central Parquet-based OHLCV cache for Yahoo Finance and GeckoTerminal data.

Stores downloaded price data on disk so repeated runs (backtests, signals,
comparisons) don't re-fetch from the network.

Cache directory: ~/.myTrading/cache/ohlcv/
File naming:     {ticker}__{interval}.parquet    (e.g. AAPL__1d.parquet)
GeckoTerminal:   gecko__{network}__{pool_addr}__{interval}.parquet

Strategy:
  1. Check if cache file exists and has data covering the requested range.
  2. If fully covered and "fresh enough" (last bar within staleness window),
     return cached data — zero network calls.
  3. If cache exists but is stale (missing recent days), do an incremental
     fetch for only the missing delta and append.
  4. If no cache at all, do a full fetch and save.

Thread-safe via file locks (one writer at a time per ticker).
"""

from __future__ import annotations

import os
import re
import fcntl
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# ── Cache directory ─────────────────────────────────────────────────
CACHE_DIR = Path(os.getenv(
    "OHLCV_CACHE_DIR",
    str(Path.home() / ".myTrading" / "cache" / "ohlcv"),
))

# How many hours old the latest bar can be before we consider the cache stale.
# For daily data (stocks): market closes ~4 PM ET, so 8h covers most cases.
# For crypto (24/7): 6h is reasonable for 4h bars.
STALENESS_HOURS = {
    "1h":  2,
    "2h":  3,
    "4h":  6,
    "1d":  20,   # daily bars: refetch if last bar is >20h old
    "1wk": 168,  # weekly: 7 days
}


def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(ticker: str, interval: str) -> Path:
    """Return the parquet file path for a Yahoo Finance ticker."""
    safe_name = ticker.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return CACHE_DIR / f"{safe_name}__{interval}.parquet"


def _gecko_cache_path(network: str, pool_address: str, interval: str) -> Path:
    """Return the parquet file path for a GeckoTerminal pool."""
    short_addr = pool_address[:16]  # keep it readable
    return CACHE_DIR / f"gecko__{network}__{short_addr}__{interval}.parquet"


def _lock_path(cache_file: Path) -> Path:
    """Lock file companion for a cache file."""
    return cache_file.with_suffix(".lock")


# ── Read / Write with file locking ──────────────────────────────────

def _read_cache(cache_file: Path) -> pd.DataFrame | None:
    """Read a parquet cache file. Returns None if missing or corrupt."""
    if not cache_file.exists():
        return None
    try:
        df = pd.read_parquet(cache_file)
        if df.empty:
            return None
        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        # Strip timezone if present
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        print(f"⚠️  Cache read failed for {cache_file.name}: {e}")
        return None


def _write_cache(cache_file: Path, df: pd.DataFrame):
    """Write DataFrame to parquet with file locking."""
    _ensure_cache_dir()
    lock_file = _lock_path(cache_file)
    try:
        with open(lock_file, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                df.to_parquet(cache_file, engine="pyarrow")
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
    except Exception as e:
        print(f"⚠️  Cache write failed for {cache_file.name}: {e}")


def _is_fresh(df: pd.DataFrame, interval: str) -> bool:
    """Check if the cached data is fresh enough (latest bar within staleness window)."""
    if df is None or df.empty:
        return False
    last_bar = df.index[-1]
    if not isinstance(last_bar, pd.Timestamp):
        last_bar = pd.Timestamp(last_bar)
    staleness_h = STALENESS_HOURS.get(interval, 20)
    cutoff = datetime.now() - timedelta(hours=staleness_h)
    return last_bar >= pd.Timestamp(cutoff)


def _covers_range(df: pd.DataFrame, lookback_days: int) -> bool:
    """Check if cached data covers the requested lookback period."""
    if df is None or df.empty:
        return False
    earliest_needed = datetime.now() - timedelta(days=lookback_days)
    return df.index[0] <= pd.Timestamp(earliest_needed)


# ═══════════════════════════════════════════════════════════════════
# Public API — Yahoo Finance
# ═══════════════════════════════════════════════════════════════════

def cached_yahoo_download(
    ticker: str,
    interval: str,
    lookback_days: int,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame | None:
    """
    Download OHLCV from Yahoo Finance with Parquet caching.

    Returns raw DataFrame with columns like Open, High, Low, Close, Volume.
    The caller is responsible for resampling (e.g. 1h → 4h), indicator
    warmup trimming, etc. — this function only handles the fetch + cache.

    Parameters
    ----------
    ticker        : Yahoo Finance symbol (e.g. "AAPL", "BTC-USD")
    interval      : "1d", "1wk", or sub-daily mapped interval (e.g. "60m")
                    NOTE: pass the yfinance interval, not the user-friendly one.
    lookback_days : Total calendar days to fetch (including warmup).
    force_refresh : If True, skip cache and re-download everything.

    Returns
    -------
    DataFrame or None if download fails.
    """
    import yfinance as yf
    from core.utils import _fix_yf_cols

    _ensure_cache_dir()

    # Normalize interval for cache key (so "60m" and "1h" don't create dupes)
    cache_interval = interval  # use the yfinance interval as-is for the key
    cache_file = _cache_path(ticker, cache_interval)

    # ── 1. Try cache ───────────────────────────────────────────────
    if not force_refresh:
        cached = _read_cache(cache_file)
        if cached is not None:
            is_fresh = _is_fresh(cached, _map_yf_interval_to_user(interval))
            covers = _covers_range(cached, lookback_days)

            if is_fresh and covers:
                print(f"💾 Cache hit: {ticker} ({cache_interval}) — {len(cached)} bars, "
                      f"range {cached.index[0].date()} → {cached.index[-1].date()}")
                return cached

            # ── 2. Incremental fetch (cache exists but stale) ──────
            if covers and not is_fresh:
                delta_df = _incremental_fetch(ticker, interval, cached)
                if delta_df is not None:
                    merged = _merge_and_save(cached, delta_df, cache_file)
                    print(f"📥 Cache updated: {ticker} ({cache_interval}) — "
                          f"+{len(delta_df)} bars → {len(merged)} total")
                    return merged

            # Cache exists but doesn't cover range — need wider fetch
            # Fall through to full fetch below, but we'll try to extend
            if not covers:
                print(f"📥 Cache exists but too short for {lookback_days}d — re-fetching {ticker}...")

    # ── 3. Full fetch ──────────────────────────────────────────────
    print(f"📥 Fetching {ticker} | interval={interval} | lookback={lookback_days}d ...")
    try:
        raw = yf.download(ticker, period=f"{lookback_days}d", interval=interval, progress=False)
        if raw is None or raw.empty:
            print(f"❌ No yfinance data for {ticker}")
            return None

        raw = _fix_yf_cols(raw)

        # Strip timezone
        if raw.index.tz is not None:
            try:
                raw.index = raw.index.tz_localize(None)
            except TypeError:
                raw.index = raw.index.tz_convert(None)

        # If we had older cached data that didn't cover the range, merge it
        # (the older data might have bars further back than yfinance returns)
        cached = _read_cache(cache_file)
        if cached is not None and not cached.empty:
            merged = _merge_and_save(cached, raw, cache_file)
            print(f"💾 Cache saved (merged): {ticker} — {len(merged)} bars")
            return merged
        else:
            _write_cache(cache_file, raw)
            print(f"💾 Cache saved: {ticker} — {len(raw)} bars "
                  f"({raw.index[0].date()} → {raw.index[-1].date()})")
            return raw

    except Exception as e:
        print(f"❌ Yahoo download failed for {ticker}: {e}")
        # Last resort: return stale cache if available
        cached = _read_cache(cache_file)
        if cached is not None:
            print(f"⚠️  Returning stale cache for {ticker} ({len(cached)} bars)")
            return cached
        return None


def _incremental_fetch(
    ticker: str, interval: str, cached: pd.DataFrame
) -> pd.DataFrame | None:
    """Fetch only the missing bars since the last cached bar."""
    import yfinance as yf
    from core.utils import _fix_yf_cols

    last_bar = cached.index[-1]
    days_missing = (datetime.now() - last_bar).days + 2  # +2 for safety overlap

    if days_missing < 1:
        return None  # already fresh

    # For sub-daily, minimum fetch period
    min_days = {"60m": 3, "1h": 3, "1d": 2, "1wk": 8}.get(interval, 2)
    fetch_days = max(days_missing, min_days)

    try:
        raw = yf.download(ticker, period=f"{fetch_days}d", interval=interval, progress=False)
        if raw is None or raw.empty:
            return None
        raw = _fix_yf_cols(raw)
        if raw.index.tz is not None:
            try:
                raw.index = raw.index.tz_localize(None)
            except TypeError:
                raw.index = raw.index.tz_convert(None)
        return raw
    except Exception as e:
        print(f"⚠️  Incremental fetch failed for {ticker}: {e}")
        return None


def _merge_and_save(
    cached: pd.DataFrame, new_data: pd.DataFrame, cache_file: Path
) -> pd.DataFrame:
    """Merge cached + new data, deduplicate by index, save to disk."""
    # Combine, keeping new data where timestamps overlap (newer is more accurate)
    combined = pd.concat([cached, new_data])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    _write_cache(cache_file, combined)
    return combined


def _map_yf_interval_to_user(yf_interval: str) -> str:
    """Map yfinance interval back to user-friendly interval for staleness check."""
    mapping = {"60m": "1h", "1h": "1h", "1d": "1d", "1wk": "1wk"}
    return mapping.get(yf_interval, "1d")


# ═══════════════════════════════════════════════════════════════════
# Public API — GeckoTerminal
# ═══════════════════════════════════════════════════════════════════

def cached_gecko_download(
    pool_url: str,
    interval: str,
    lookback_days: int,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame | None:
    """
    Download OHLCV from GeckoTerminal with Parquet caching.

    Parameters
    ----------
    pool_url      : Full GeckoTerminal URL
    interval      : "1h", "4h", or "1d"
    lookback_days : Calendar days to fetch

    Returns
    -------
    DataFrame with columns: High, Low, Close, Volume (index = DatetimeIndex)
    """
    import requests

    match = re.search(r"geckoterminal\.com/(\w+)/pools/([A-Za-z0-9]+)", pool_url)
    if not match:
        print(f"⚠️  Invalid GeckoTerminal URL: {pool_url}")
        return None

    network = match.group(1)
    pool_address = match.group(2)

    _ensure_cache_dir()
    cache_file = _gecko_cache_path(network, pool_address, interval)

    # ── Try cache ──────────────────────────────────────────────────
    if not force_refresh:
        cached = _read_cache(cache_file)
        if cached is not None:
            is_fresh = _is_fresh(cached, interval)
            covers = _covers_range(cached, lookback_days)
            if is_fresh and covers:
                print(f"💾 Cache hit: gecko/{network}/{pool_address[:12]} ({interval}) — {len(cached)} bars")
                return cached

    # ── Full fetch ─────────────────────────────────────────────────
    gecko_tf_map = {
        "1h": ("hour", 1),
        "4h": ("hour", 4),
        "1d": ("day",  1),
    }
    if interval not in gecko_tf_map:
        print(f"⚠️  Unsupported GeckoTerminal interval: {interval}")
        return None

    timeframe, aggregate = gecko_tf_map[interval]
    bars_per_day = {"1h": 24, "4h": 6, "1d": 1}[interval]
    limit = lookback_days * bars_per_day + 200  # extra for warmup

    # CoinGecko API key
    headers = {}
    try:
        from core.utils import get_secret
        api_key = get_secret("COINGECKO_API_KEY").strip()
        if api_key:
            headers["x-cg-pro-api-key"] = api_key
    except Exception:
        pass

    url = (
        f"https://api.geckoterminal.com/api/v2/networks/{network}/pools/{pool_address}"
        f"/ohlcv/{timeframe}?aggregate={aggregate}&limit={limit}&currency=usd"
    )

    print(f"📥 Fetching GeckoTerminal {network}/{pool_address[:12]}... | {interval} | lookback={lookback_days}d")

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            print(f"⚠️  GeckoTerminal API returned {resp.status_code}")
            # Return stale cache if available
            cached = _read_cache(cache_file)
            return cached

        data = resp.json()
        ohlcv_list = data.get("data", {}).get("attributes", {}).get("ohlcv_list", [])
        if not ohlcv_list:
            print(f"⚠️  No OHLCV data from GeckoTerminal")
            cached = _read_cache(cache_file)
            return cached

        rows = []
        for candle in ohlcv_list:
            if len(candle) < 6:
                continue
            ts = candle[0]
            o, h, l, c, v = (float(candle[1]), float(candle[2]),
                             float(candle[3]), float(candle[4]), float(candle[5]))
            dt = pd.to_datetime(ts, unit="s")
            rows.append({"timestamp": dt, "Open": o, "High": h,
                         "Low": l, "Close": c, "Volume": v})

        df = pd.DataFrame(rows)
        if df.empty:
            return None

        df = df.set_index("timestamp").sort_index()
        df = df[["High", "Low", "Close", "Volume"]].dropna()

        # Merge with existing cache
        cached = _read_cache(cache_file)
        if cached is not None and not cached.empty:
            merged = _merge_and_save(cached, df, cache_file)
            print(f"💾 Cache saved (merged): gecko/{network}/{pool_address[:12]} — {len(merged)} bars")
            return merged
        else:
            _write_cache(cache_file, df)
            print(f"💾 Cache saved: gecko/{network}/{pool_address[:12]} — {len(df)} bars")
            return df

    except Exception as e:
        print(f"❌ GeckoTerminal fetch failed: {e}")
        cached = _read_cache(cache_file)
        if cached is not None:
            print(f"⚠️  Returning stale cache ({len(cached)} bars)")
            return cached
        return None


# ═══════════════════════════════════════════════════════════════════
# Cache management utilities
# ═══════════════════════════════════════════════════════════════════

def cache_stats() -> dict:
    """Return stats about the cache directory."""
    _ensure_cache_dir()
    files = list(CACHE_DIR.glob("*.parquet"))
    total_size = sum(f.stat().st_size for f in files)
    return {
        "cache_dir": str(CACHE_DIR),
        "num_files": len(files),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "files": [f.name for f in sorted(files)],
    }


def clear_cache(ticker: str | None = None, interval: str | None = None):
    """
    Clear cache files.
    - No args: clear everything.
    - ticker only: clear all intervals for that ticker.
    - ticker + interval: clear one specific file.
    """
    _ensure_cache_dir()
    if ticker is None:
        # Clear all
        for f in CACHE_DIR.glob("*.parquet"):
            f.unlink()
        for f in CACHE_DIR.glob("*.lock"):
            f.unlink()
        print("🗑️  All cache cleared")
        return

    if interval:
        target = _cache_path(ticker, interval)
        if target.exists():
            target.unlink()
            print(f"🗑️  Cleared cache: {target.name}")
    else:
        safe_name = ticker.replace("/", "_").replace("\\", "_")
        for f in CACHE_DIR.glob(f"{safe_name}__*.parquet"):
            f.unlink()
            print(f"🗑️  Cleared cache: {f.name}")


def force_refresh_ticker(ticker: str, interval: str, lookback_days: int) -> pd.DataFrame | None:
    """Convenience: clear cache for a ticker and re-download."""
    return cached_yahoo_download(ticker, interval, lookback_days, force_refresh=True)