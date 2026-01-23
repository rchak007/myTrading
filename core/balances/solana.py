# core/balances/solana.py
from __future__ import annotations

import random
import time
import requests
from decimal import Decimal
from typing import Any, Dict, List, Tuple


TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM_ID = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"

# Add a few public RPCs to rotate (mainnet-beta will 429 quickly)
DEFAULT_SOL_RPCS = [
    "https://api.mainnet-beta.solana.com",
    "https://solana.publicnode.com",
    "https://rpc.ankr.com/solana",          # sometimes needs key; if it fails it will be skipped
    "https://solana-mainnet.g.alchemy.com/v2/demo",  # public demo key (rate-limited but ok)
]


def _rpc_call(
    rpc_urls: List[str],
    method: str,
    params: list,
    *,
    max_retries: int = 6,
    timeout: int = 30,
) -> Dict[str, Any]:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    last_err = None

    urls = [u for u in rpc_urls if u] or DEFAULT_SOL_RPCS
    # rotate order a bit
    urls = urls[:] 
    random.shuffle(urls)

    for url in urls:
        for attempt in range(max_retries):
            try:
                r = requests.post(url, json=payload, timeout=timeout)
                if r.status_code == 429:
                    # exponential backoff with jitter
                    sleep_s = min(8.0, (0.6 * (2 ** attempt)) + random.random() * 0.3)
                    time.sleep(sleep_s)
                    last_err = RuntimeError(f"[Solana RPC] 429 Too Many Requests @ {url}")
                    continue

                if r.status_code != 200:
                    last_err = RuntimeError(f"[Solana RPC] HTTP {r.status_code} @ {url}: {r.text[:300]}")
                    # try next url
                    break

                j = r.json()
                if "error" in j:
                    last_err = RuntimeError(f"[Solana RPC] error @ {url}: {j['error']}")
                    # try next url
                    break

                return j["result"]

            except Exception as e:
                last_err = RuntimeError(f"[Solana RPC] failure @ {url}: {e}")
                # small backoff
                time.sleep(0.25 + random.random() * 0.25)

    raise last_err if last_err else RuntimeError("[Solana RPC] Unknown failure")


def _extract_mint_and_ui_amount(acct: dict) -> Tuple[str, Decimal] | None:
    """
    Returns (mint, ui_amount) from a token account in jsonParsed encoding.
    """
    try:
        info = acct["account"]["data"]["parsed"]["info"]
        mint = str(info.get("mint", "")).strip()
        token_amount = info["tokenAmount"]
        ui_str = token_amount.get("uiAmountString")
        if ui_str is None:
            raw = Decimal(token_amount["amount"])
            decimals = int(token_amount["decimals"])
            ui = raw / (Decimal(10) ** decimals)
        else:
            ui = Decimal(ui_str)
        return mint, ui
    except Exception:
        return None


def get_wallet_token_balances_by_mint(
    wallet: str,
    *,
    rpc_urls: List[str] | None = None,
    debug: bool = False,
) -> Dict[str, Decimal]:
    """
    Fetches ALL token accounts for wallet (Tokenkeg + Token2022) ONCE each.
    Returns dict: mint -> total ui amount.
    """
    debug = True
    urls = rpc_urls or DEFAULT_SOL_RPCS
    balances: Dict[str, Decimal] = {}

    if debug:
        print("Wallet =", wallet)
        print("RPC urls =", urls)

    # Tokenkeg
    res1 = _rpc_call(
        urls,
        "getTokenAccountsByOwner",
        [wallet, {"programId": TOKEN_PROGRAM_ID}, {"encoding": "jsonParsed"}],
    )
    accts1 = res1.get("value", []) or []
    if debug:
        print("Accounts(Tokenkeg) =", len(accts1))

    for a in accts1:
        item = _extract_mint_and_ui_amount(a)
        if not item:
            continue
        mint, ui = item
        balances[mint] = balances.get(mint, Decimal("0")) + ui

    # Token-2022
    res2 = _rpc_call(
        urls,
        "getTokenAccountsByOwner",
        [wallet, {"programId": TOKEN_2022_PROGRAM_ID}, {"encoding": "jsonParsed"}],
    )
    accts2 = res2.get("value", []) or []
    if debug:
        print("Accounts(Token2022) =", len(accts2))

    for a in accts2:
        item = _extract_mint_and_ui_amount(a)
        if not item:
            continue
        mint, ui = item
        balances[mint] = balances.get(mint, Decimal("0")) + ui

    if debug:
        print("---- Token balances (by mint) ----")
        for mint, amt in balances.items():
            if amt != 0:
                print(f"Mint: {mint} | Balance: {amt}")
        print("----------------------------------")

    return balances


def get_spl_token_balance_from_cache(
    wallet: str,
    mint: str,
    wallet_cache: Dict[str, Dict[str, Decimal]],
    *,
    rpc_urls: List[str] | None = None,
    debug: bool = False,
) -> Decimal:
    """
    Uses per-wallet cache to avoid repeated RPC calls.
    """
    if wallet not in wallet_cache:
        wallet_cache[wallet] = get_wallet_token_balances_by_mint(wallet, rpc_urls=rpc_urls, debug=debug)

    return wallet_cache[wallet].get(mint, Decimal("0"))
