"""
core/execution/sui_swap.py

SUI blockchain execution module — balance queries and DEX swaps.

SWAP ROUTING:
  Aftermath Finance aggregator (aftermath.finance) — the dominant
  DEX aggregator on SUI, aggregates across Cetus, DeepBook, Turbos,
  FlowX, Kriya, and more.

  API docs: https://aftermath.finance/docs/router

Architecture:
  - SUI uses the Move VM — transactions are built differently from EVM.
  - Wallets are Ed25519 keypairs (like Solana), not secp256k1 (like EVM).
  - The SUI private key from .env is Fernet-encrypted (same scheme as all bots).
  - We use the Aftermath Router REST API to get a pre-built transaction block,
    then sign it locally and submit via SUI RPC.

Setup:
  pip install pysui --break-system-packages

.env:
  SUI_RPC_URL=https://fullnode.mainnet.sui.io:443  (default)

Gas reserve:
  SUI_FEE_RESERVE=0.05  (~$0.05 — SUI gas is very cheap)

asset_registry.json entry example:
  "SUI20947": {
    "ticker": "SUI20947",
    "blockchain": "sui",
    "wallet_address": "0xYOUR_SUI_ADDRESS",
    "token_contract": "0x2::sui::SUI",
    "yahoo_ticker": "SUI20947-USD",
    "stablecoin_contract": "0xdba34672...::usdc::USDC"
  }
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import requests
from decimal import Decimal
from typing import Optional

log = logging.getLogger("bot.sui")

# ═══════════════════════════════════════════════════════════════════
# SUI chain configuration
# ═══════════════════════════════════════════════════════════════════
SUI_RPC_URL = os.getenv("SUI_RPC_URL", "https://fullnode.mainnet.sui.io:443")

# Aftermath Finance Router API
AFTERMATH_API_BASE = "https://aftermath.finance/api"

# Common coin types on SUI
SUI_COIN_TYPE  = "0x2::sui::SUI"
USDC_COIN_TYPE = "0xdba34672e30cb065b1f93e3ab55318768fd6fef66c15942c9f7cb846e2f900e7::usdc::USDC"

# Gas reserve (SUI is very cheap ~$0.001 per tx)
SUI_FEE_RESERVE = float(os.getenv("SUI_FEE_RESERVE", "0.05"))

# Slippage for SUI swaps (fraction, not bps)
SUI_SLIPPAGE = float(os.getenv("SUI_SLIPPAGE", "0.03"))  # 3%


# ═══════════════════════════════════════════════════════════════════
# SUI RPC helpers
# ═══════════════════════════════════════════════════════════════════
def _rpc_call(method: str, params: list, rpc_url: str = None) -> dict:
    """Raw JSON-RPC call to SUI fullnode."""
    url = rpc_url or SUI_RPC_URL
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(f"SUI RPC error: {data['error']}")
    return data["result"]


# ═══════════════════════════════════════════════════════════════════
# Balance queries
# ═══════════════════════════════════════════════════════════════════
def get_coin_metadata(coin_type: str) -> dict:
    """Fetch coin metadata (name, symbol, decimals) from SUI RPC."""
    return _rpc_call("suix_getCoinMetadata", [coin_type])


def get_coin_balance(owner: str, coin_type: str) -> float:
    """Get balance of a coin type for an owner address. Returns UI amount."""
    res = _rpc_call("suix_getBalance", [owner, coin_type])
    raw = Decimal(res.get("totalBalance", "0"))
    meta = get_coin_metadata(coin_type)
    decimals = int(meta.get("decimals", 9))
    return float(raw / (Decimal(10) ** decimals))


def get_coin_decimals(coin_type: str) -> int:
    """Get decimals for a coin type."""
    meta = get_coin_metadata(coin_type)
    return int(meta.get("decimals", 9))


def get_all_coins(owner: str, coin_type: str) -> list[dict]:
    """
    Get all coin objects of a given type for an owner.
    Needed to build swap transactions (SUI uses UTXO-like coin objects).
    """
    coins = []
    cursor = None
    while True:
        params = [owner, coin_type, cursor, 50]  # limit 50 per page
        res = _rpc_call("suix_getCoins", params)
        coins.extend(res.get("data", []))
        if not res.get("hasNextPage", False):
            break
        cursor = res.get("nextCursor")
    return coins


# ═══════════════════════════════════════════════════════════════════
# Aftermath Finance Router — quote + swap
# ═══════════════════════════════════════════════════════════════════
def aftermath_get_route(
    coin_in_type: str,
    coin_out_type: str,
    amount_in_raw: int,
) -> dict:
    """
    Get optimal swap route from Aftermath Finance aggregator.
    Returns route data including expected output and the list of venues.
    """
    url = f"{AFTERMATH_API_BASE}/router/trade-route"
    payload = {
        "coinInType":  coin_in_type,
        "coinOutType": coin_out_type,
        "coinInAmount": str(amount_in_raw),
    }

    log.info("Aftermath route: %s→%s amount_raw=%d",
             coin_in_type.split("::")[-1], coin_out_type.split("::")[-1], amount_in_raw)

    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    route = r.json()

    if not route or "coinOut" not in route:
        raise RuntimeError(f"Aftermath: no route found for {coin_in_type}→{coin_out_type}")

    expected_out = route.get("coinOut", {}).get("amount", 0)
    log.info("Aftermath route found: expectedOut=%s venues=%s",
             expected_out, [r.get("protocol") for r in route.get("routes", [])])
    return route


def aftermath_build_tx(
    route: dict,
    sender: str,
    slippage: float = SUI_SLIPPAGE,
) -> dict:
    """
    Build a swap transaction via Aftermath Finance API.
    Returns the transaction block bytes (base64) ready for signing.
    """
    url = f"{AFTERMATH_API_BASE}/router/transactions/trade"
    payload = {
        "route":    route,
        "sender":   sender,
        "slippage": slippage,
    }

    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    tx_data = r.json()

    if "tx" not in tx_data and "txBytes" not in tx_data:
        raise RuntimeError(f"Aftermath: no transaction in response: {tx_data}")

    return tx_data


def sign_and_execute_sui_tx(
    tx_bytes_b64: str,
    private_key_hex: str,
) -> str:
    """
    Sign a SUI transaction with Ed25519 key and execute via RPC.

    Parameters
    ----------
    tx_bytes_b64   : base64 encoded transaction bytes from Aftermath
    private_key_hex: hex-encoded Ed25519 private key (32 bytes)

    Returns
    -------
    Transaction digest string on success.
    """
    import nacl.signing

    # Parse private key
    pk_bytes = bytes.fromhex(private_key_hex.replace("0x", ""))
    signing_key = nacl.signing.SigningKey(pk_bytes[:32])

    # Decode tx bytes
    tx_bytes = base64.b64decode(tx_bytes_b64)

    # SUI signing: intent = [0, 0, 0] + tx_bytes → hash → sign
    import hashlib
    intent_msg = bytes([0, 0, 0]) + tx_bytes
    digest     = hashlib.blake2b(intent_msg, digest_size=32).digest()

    signed = signing_key.sign(digest)
    signature = signed.signature  # 64 bytes Ed25519 signature

    # SUI signature format: flag_byte (0x00 = Ed25519) + sig(64) + pubkey(32)
    pub_key = signing_key.verify_key.encode()
    sui_sig = bytes([0x00]) + signature + pub_key
    sig_b64 = base64.b64encode(sui_sig).decode("utf-8")

    # Execute via RPC
    result = _rpc_call(
        "sui_executeTransactionBlock",
        [
            tx_bytes_b64,
            [sig_b64],
            {
                "showEffects": True,
                "showEvents":  True,
            },
            "WaitForLocalExecution",
        ],
    )

    # Parse result
    effects = result.get("effects", {})
    status  = effects.get("status", {})

    if status.get("status") == "success":
        digest_str = result.get("digest", "")
        log.info("✅ SUI tx SUCCESS | digest=%s", digest_str)
        return digest_str

    error = status.get("error", "unknown error")
    raise RuntimeError(f"SUI tx FAILED: {error}")


# ═══════════════════════════════════════════════════════════════════
# sui_swap() — SINGLE ENTRY POINT for all SUI swaps in bot.py
# ═══════════════════════════════════════════════════════════════════
def sui_swap(
    *,
    private_key: str,
    wallet_address: str,
    coin_in_type: str,
    coin_out_type: str,
    amount_in_human: float,
    coin_in_decimals: int,
    coin_out_decimals: int,
    slippage: float = SUI_SLIPPAGE,
) -> str:
    """
    Unified SUI swap entry point for bot.py.

    Parameters
    ----------
    private_key      : hex Ed25519 private key (decrypted from Fernet)
    wallet_address   : 0x... SUI address
    coin_in_type     : full coin type e.g. "0x2::sui::SUI"
    coin_out_type    : full coin type e.g. "0xdba3...::usdc::USDC"
    amount_in_human  : human-readable amount to sell
    coin_in_decimals : decimals for input coin
    coin_out_decimals: decimals for output coin
    slippage         : fraction (0.03 = 3%)

    Returns
    -------
    Transaction digest string.
    """
    amount_in_raw = int(amount_in_human * (10 ** coin_in_decimals))

    log.info("[sui] swap: %s → %s | amount=%.6f (raw=%d) slippage=%.1f%%",
             coin_in_type.split("::")[-1], coin_out_type.split("::")[-1],
             amount_in_human, amount_in_raw, slippage * 100)

    # Step 1: Get route from Aftermath
    route = aftermath_get_route(coin_in_type, coin_out_type, amount_in_raw)

    # Step 2: Build transaction
    tx_data = aftermath_build_tx(route, wallet_address, slippage)

    # Extract tx bytes (Aftermath may return as "tx" or "txBytes")
    tx_bytes_b64 = tx_data.get("txBytes") or tx_data.get("tx", "")
    if not tx_bytes_b64:
        raise RuntimeError("[sui] Aftermath returned no transaction bytes")

    # Step 3: Sign and execute
    digest = sign_and_execute_sui_tx(tx_bytes_b64, private_key)

    log.info("[sui] ✅ swap confirmed: digest=%s", digest)
    return digest