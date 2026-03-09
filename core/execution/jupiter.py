# core/execution/jupiter.py
#
# Uses Jupiter Ultra Swap API — the same engine as jup.ag mobile/web.
# Ultra handles: slippage, priority fees, transaction landing, retries, MEV protection.
# No manual slippage tuning, no polling, no skipPreflight hacks needed.
#
# Flow:
#   1. GET  /ultra/v1/order  → get quote + unsigned base64 tx + requestId
#   2. Sign the transaction locally with our Keypair
#   3. POST /ultra/v1/execute → Jupiter lands it, returns status + txid
#
from __future__ import annotations

import base64
import os
import requests
import logging

from solders.keypair import Keypair
from solders.transaction import VersionedTransaction

log = logging.getLogger("bot.jupiter")

WSOL_MINT     = "So11111111111111111111111111111111111111112"
USDC_MINT     = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
WSOL_DECIMALS = 9
USDC_DECIMALS = 6

ULTRA_ORDER_URL   = "https://api.jup.ag/ultra/v1/order"
ULTRA_EXECUTE_URL = "https://api.jup.ag/ultra/v1/execute"


def _auth_headers() -> dict:
    """Return Authorization header if JUPITER_API_KEY is set in env."""
    key = os.getenv("JUPITER_API_KEY", "").strip()
    if key:
        return {"Authorization": f"Bearer {key}"}
    return {}


# ─────────────────────────────────────────────
# RPC helpers (balance checks — unchanged)
# ─────────────────────────────────────────────

def rpc_call(rpc_url: str, method: str, params: list) -> dict:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    r = requests.post(rpc_url, json=payload, timeout=20)
    r.raise_for_status()
    j = r.json()
    if "error" in j:
        raise RuntimeError(f"RPC error: {j['error']}")
    return j["result"]


def get_sol_balance(rpc_url: str, pubkey: str) -> float:
    res = rpc_call(rpc_url, "getBalance", [pubkey, {"commitment": "confirmed"}])
    return int(res["value"]) / 1e9


def get_spl_token_balance_ui(rpc_url: str, pubkey: str, mint: str) -> float:
    res = rpc_call(
        rpc_url,
        "getTokenAccountsByOwner",
        [pubkey, {"mint": mint}, {"encoding": "jsonParsed", "commitment": "confirmed"}],
    )
    total = 0.0
    for acc in res.get("value", []):
        info = acc["account"]["data"]["parsed"]["info"]
        amt = info["tokenAmount"].get("uiAmount")
        if amt is not None:
            total += float(amt)
    return total


def to_smallest(amount: float, decimals: int) -> int:
    return int(amount * (10 ** decimals))


# ─────────────────────────────────────────────
# Ultra Swap API
# ─────────────────────────────────────────────

def ultra_get_order(
    input_mint: str,
    output_mint: str,
    amount_smallest: int,
    taker_pubkey: str,
) -> dict:
    """
    GET /ultra/v1/order
    Returns the order dict containing:
      - transaction : base64 unsigned tx → sign this
      - requestId   : pass to /execute
      - inAmount, outAmount, slippageBps, swapType, etc.
    """
    params = {
        "inputMint":  input_mint,
        "outputMint": output_mint,
        "amount":     str(amount_smallest),
        "taker":      taker_pubkey,
    }
    r = requests.get(ULTRA_ORDER_URL, params=params, headers=_auth_headers(), timeout=20)
    r.raise_for_status()
    data = r.json()

    if "error" in data:
        raise RuntimeError(f"Jupiter Ultra /order error: {data['error']}")
    if "transaction" not in data:
        raise RuntimeError(f"Jupiter Ultra /order missing transaction: {data}")

    log.info(
        "Ultra order: %s→%s | in=%s out=%s slippage=%sbps type=%s",
        input_mint[:6], output_mint[:6],
        data.get("inAmount"), data.get("outAmount"),
        data.get("slippageBps"), data.get("swapType"),
    )
    return data


def ultra_sign_and_execute(
    order_response: dict,
    keypair: Keypair,
) -> str:
    """
    Signs the transaction from ultra_get_order() and POSTs to /execute.
    Jupiter lands it via Jupiter Beam (their own infrastructure).

    Returns the transaction signature string on success.
    Raises RuntimeError on any failure — caller flips regime only on success.
    """
    # 1. Decode + sign
    raw       = base64.b64decode(order_response["transaction"])
    vt        = VersionedTransaction.from_bytes(raw)
    vt_signed = VersionedTransaction(vt.message, [keypair])
    signed_b64 = base64.b64encode(bytes(vt_signed)).decode("utf-8")

    # 2. Submit to Jupiter
    payload = {
        "signedTransaction": signed_b64,
        "requestId":         order_response["requestId"],
    }
    r = requests.post(ULTRA_EXECUTE_URL, json=payload, headers=_auth_headers(), timeout=60)
    r.raise_for_status()
    result = r.json()

    log.info("Ultra execute response: %s", result)

    # 3. Parse result
    status = result.get("status", "")
    tx_sig  = (
        result.get("signature")
        or result.get("txid")
        or result.get("txSig")
        or ""
    )

    if status == "Success":
        log.info("✅ Ultra swap SUCCESS | sig=%s", tx_sig)
        return str(tx_sig)

    # Non-success → raise so bot.py never flips regime
    error_code = result.get("error") or result.get("errorCode") or ""
    error_msg  = result.get("message") or result.get("errorMessage") or ""
    input_amt  = result.get("inputAmountResult",  "?")
    output_amt = result.get("outputAmountResult", "?")

    raise RuntimeError(
        f"Ultra swap FAILED | status={status} error={error_code} "
        f"msg={error_msg} inAmt={input_amt} outAmt={output_amt} sig={tx_sig}"
    )


# ─────────────────────────────────────────────
# Legacy stubs — raise immediately so nothing
# silently falls back to the old broken path
# ─────────────────────────────────────────────

def get_quote(*a, **kw):
    raise NotImplementedError("Replaced by ultra_get_order()")

def get_swap_tx(*a, **kw):
    raise NotImplementedError("Replaced by ultra_get_order()")

def sign_and_send_swap(*a, **kw):
    raise NotImplementedError("Replaced by ultra_sign_and_execute()")