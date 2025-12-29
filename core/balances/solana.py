# core/balances/solana.py
from __future__ import annotations

import requests
from decimal import Decimal


DEFAULT_SOL_RPC = "https://api.mainnet-beta.solana.com"


def _rpc_call(rpc_url: str, method: str, params: list):
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    r = requests.post(rpc_url, json=payload, timeout=30)
    r.raise_for_status()
    j = r.json()
    if "error" in j:
        raise RuntimeError(j["error"])
    return j["result"]


def get_spl_token_balance(wallet: str, mint: str, rpc_url: str = DEFAULT_SOL_RPC) -> Decimal:
    """
    Sums all token accounts for mint owned by wallet.
    Returns ui amount as Decimal.
    """
    result = _rpc_call(
        rpc_url,
        "getTokenAccountsByOwner",
        [wallet, {"mint": mint}, {"encoding": "jsonParsed"}],
    )

    accounts = result.get("value", [])
    if not accounts:
        return Decimal("0")

    total = Decimal("0")
    for acct in accounts:
        info = acct["account"]["data"]["parsed"]["info"]
        token_amount = info["tokenAmount"]
        ui_str = token_amount.get("uiAmountString")
        if ui_str is None:
            # fallback
            raw = Decimal(token_amount["amount"])
            decimals = int(token_amount["decimals"])
            ui_str = str(raw / (Decimal(10) ** decimals))
        total += Decimal(ui_str)

    return total
