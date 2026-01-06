#!/usr/bin/env python3
"""
Test GeckoTerminal 4H OHLCV for a Solana token (pool-based candles).

- Finds the top pool for the token mint
- Fetches 4H candles (timeframe=hour, aggregate=4)
- Prints a preview

Requirements:
  pip install requests pandas
"""

import sys
import time
from typing import Dict, Any, Optional, Tuple

import requests
import pandas as pd


ASSET = {
    "ticker": "LQL",
    "blockchain": "solana",
    "wallet_address": "Ce2RXLKEnpWuJm4uDu25T6vAY7Y3bPY9MebK1NGfeH9B",
    "token_contract": "cqBsZzsbfMKJMtV4shiTZXpEK4MUVurBacA5F6opump",  # Solana mint
    "yahoo_ticker": "LQL-USD",
    "stablecoin_contract": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC mint
}

BASE = "https://api.geckoterminal.com/api/v2"
UA = "myTrading-geckoterminal-test/1.1"


def _get(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {r.url}\nBody: {r.text[:500]}")
    return r.json()


def pick_top_pool(network: str, token_address: str) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (pool_address, pool_object) for the 'best' pool.
    Strategy: pick the pool with highest reserve_in_usd.
    """
    url = f"{BASE}/networks/{network}/tokens/{token_address}/pools"
    data = _get(url)

    pools = data.get("data") or []
    if not pools:
        raise RuntimeError(
            f"No pools found for token {token_address} on network={network}. "
            "Check the mint/contract and the network name."
        )

    def reserve_usd(pool: Dict[str, Any]) -> float:
        attrs = pool.get("attributes") or {}
        v = attrs.get("reserve_in_usd")
        try:
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0

    pools_sorted = sorted(pools, key=reserve_usd, reverse=True)
    top = pools_sorted[0]

    # For OHLCV endpoint, GeckoTerminal expects the pool *address*
    pool_addr = (top.get("attributes") or {}).get("address")
    if not pool_addr:
        raise RuntimeError("Could not extract pool address from response.")

    return pool_addr, top


def fetch_ohlcv_4h(network: str, pool_address: str, limit: int = 200) -> pd.DataFrame:
    """
    GeckoTerminal public endpoint (NO /onchain):
      /networks/{network}/pools/{pool_address}/ohlcv/{timeframe}

    4H candles:
      timeframe=hour, aggregate=4
    """
    url = f"{BASE}/networks/{network}/pools/{pool_address}/ohlcv/hour"
    params = {"aggregate": 4, "limit": limit}

    data = _get(url, params=params)

    attrs = (data.get("data") or {}).get("attributes") or {}
    rows = attrs.get("ohlcv_list") or []
    if not rows:
        raise RuntimeError(f"No OHLCV returned for pool={pool_address} on {network}.")

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df[["dt", "open", "high", "low", "close", "volume", "ts"]].sort_values("ts").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def main() -> int:
    network = ASSET["blockchain"]      # 'solana'
    token = ASSET["token_contract"]    # mint on Solana

    print(f"Token: {ASSET['ticker']}  network={network}  mint={token}")
    print("Finding top pool...")

    pool_addr, pool_obj = pick_top_pool(network, token)
    attrs = pool_obj.get("attributes") or {}

    print(f"Top pool address: {pool_addr}")
    print(f"reserve_in_usd={attrs.get('reserve_in_usd', 'n/a')}  name={attrs.get('name', 'n/a')}")

    time.sleep(0.2)

    print("\nFetching 4H OHLCV...")
    df = fetch_ohlcv_4h(network, pool_addr, limit=200)

    print(f"Returned candles: {len(df)}")
    print("\nLast 10 candles (UTC):")
    print(df.tail(10).to_string(index=False))

    print("\nLatest close:", df.iloc[-1]["close"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
