# core/execution/jupiter.py
from __future__ import annotations

import base64
import time
import requests
from typing import Optional

from solders.keypair import Keypair
from solders.transaction import VersionedTransaction


WSOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

WSOL_DECIMALS = 9
USDC_DECIMALS = 6

JUP_QUOTE_URL = "https://public.jupiterapi.com/quote"
JUP_SWAP_URL  = "https://public.jupiterapi.com/swap"


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
    lamports = int(res["value"])
    return lamports / 1e9


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


def get_quote(input_mint: str, output_mint: str, amount_smallest: int, slippage_bps: int) -> dict:
    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": str(amount_smallest),
        "slippageBps": str(slippage_bps),
        "restrictIntermediateTokens": "true",  # route only through high-liquidity intermediates
    }
    r = requests.get(JUP_QUOTE_URL, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def get_swap_tx(quote_response: dict, user_pubkey: str) -> dict:
    payload = {
        "quoteResponse": quote_response,
        "userPublicKey": user_pubkey,
        "wrapAndUnwrapSol": True,
        "dynamicComputeUnitLimit": True,
        "dynamicSlippage": True,   # Jupiter estimates real slippage, overrides slippageBps
        "prioritizationFeeLamports": {
            "priorityLevelWithMaxLamports": {
                "priorityLevel": "veryHigh",
                "maxLamports": 1000000,  # ~$0.15 max priority fee
            }
        },
    }
    r = requests.post(JUP_SWAP_URL, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()


def sign_and_send_swap(
    *,
    rpc_url: str,
    swap_tx_b64: str,
    keypair: Keypair,
) -> str:
    raw = base64.b64decode(swap_tx_b64)
    vt = VersionedTransaction.from_bytes(raw)
    vt_signed = VersionedTransaction(vt.message, [keypair])

    wire = base64.b64encode(bytes(vt_signed)).decode("utf-8")
    sig = str(rpc_call(rpc_url, "sendTransaction", [wire, {
        "encoding": "base64",
        "skipPreflight": True,
        "preflightCommitment": "confirmed",
        "maxRetries": 3,
    }]))

    # Wait for on-chain confirmation and verify it didn't fail
    for _ in range(30):  # up to ~30s
        time.sleep(1)
        try:
            result = rpc_call(rpc_url, "getSignatureStatuses",
                              [[sig], {"searchTransactionHistory": True}])
            status = (result.get("value") or [None])[0]
            if status is None:
                continue
            if status.get("err"):
                raise RuntimeError(
                    f"Transaction {sig[:20]}... failed on-chain: {status['err']}"
                )
            if status.get("confirmationStatus") in ("confirmed", "finalized"):
                return sig  # ✅ confirmed
        except RuntimeError:
            raise
        except Exception:
            pass  # RPC hiccup — keep polling

    raise RuntimeError(f"Transaction {sig[:20]}... not confirmed after 30s")