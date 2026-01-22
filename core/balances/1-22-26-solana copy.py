# # core/balances/solana.py
# from __future__ import annotations

# import requests
# from decimal import Decimal


# DEFAULT_SOL_RPC = "https://api.mainnet-beta.solana.com"


# def _rpc_call(rpc_url: str, method: str, params: list):
#     payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
#     r = requests.post(rpc_url, json=payload, timeout=30)
#     r.raise_for_status()
#     j = r.json()
#     if "error" in j:
#         raise RuntimeError(j["error"])
#     return j["result"]


# def get_spl_token_balance(wallet: str, mint: str, rpc_url: str = DEFAULT_SOL_RPC) -> Decimal:
#     """
#     Sums all token accounts for mint owned by wallet.
#     Returns ui amount as Decimal.
#     """
#     print("Wallet = ", wallet)
#     print("Mint = ", mint)
#     print("RPC = ", str)
#     result = _rpc_call(
#         rpc_url,
#         "getTokenAccountsByOwner",
#         [wallet, {"mint": mint}, {"encoding": "jsonParsed"}],
#     )

#     accounts = result.get("value", [])
#     if not accounts:
#         print("Failed to get value")
#         return Decimal("0")

#     total = Decimal("0")
#     for acct in accounts:
#         info = acct["account"]["data"]["parsed"]["info"]
#         token_amount = info["tokenAmount"]
#         ui_str = token_amount.get("uiAmountString")
#         if ui_str is None:
#             # fallback
#             raw = Decimal(token_amount["amount"])
#             decimals = int(token_amount["decimals"])
#             ui_str = str(raw / (Decimal(10) ** decimals))
#         total += Decimal(ui_str)
#     print("Total = ", total)
#     return total



# core/balances/solana.py
from __future__ import annotations

import requests
from decimal import Decimal
from typing import Any, Dict, List, Optional


DEFAULT_SOL_RPC = "https://api.mainnet-beta.solana.com"

TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM_ID = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"


def _rpc_call(rpc_url: str, method: str, params: list) -> Dict[str, Any]:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    try:
        r = requests.post(rpc_url, json=payload, timeout=30)
    except Exception as e:
        raise RuntimeError(f"[Solana RPC] request failed: {e}")

    if r.status_code != 200:
        raise RuntimeError(f"[Solana RPC] HTTP {r.status_code}: {r.text[:500]}")

    j = r.json()
    if "error" in j:
        raise RuntimeError(f"[Solana RPC] error: {j['error']}")
    return j["result"]


def _sum_accounts_ui_amount(accounts: List[dict], mint: str) -> Decimal:
    total = Decimal("0")
    for acct in accounts:
        try:
            info = acct["account"]["data"]["parsed"]["info"]
            if str(info.get("mint", "")).strip() != mint:
                continue
            token_amount = info["tokenAmount"]
            ui_str = token_amount.get("uiAmountString")
            if ui_str is None:
                raw = Decimal(token_amount["amount"])
                decimals = int(token_amount["decimals"])
                ui_str = str(raw / (Decimal(10) ** decimals))
            total += Decimal(ui_str)
        except Exception:
            # ignore weird accounts rather than crashing
            continue
    return total


def _get_accounts_by_program(wallet: str, program_id: str, rpc_url: str) -> List[dict]:
    # We use programId filter then filter by mint ourselves.
    # This catches both Tokenkeg and Token-2022 mints reliably.
    res = _rpc_call(
        rpc_url,
        "getTokenAccountsByOwner",
        [wallet, {"programId": program_id}, {"encoding": "jsonParsed"}],
    )
    return res.get("value", []) or []


def get_spl_token_balance(
    wallet: str,
    mint: str,
    rpc_url: str = DEFAULT_SOL_RPC,
    debug: bool = False,
) -> Decimal:
    """
    Returns total ui amount of `mint` held by `wallet`.
    Checks BOTH token programs: SPL Token and Token-2022.
    """
    debug = True
    if debug:
        print("Wallet =", wallet)
        print("Mint   =", mint)
        print("RPC    =", rpc_url)

    try:
        accts_tokenkeg = _get_accounts_by_program(wallet, TOKEN_PROGRAM_ID, rpc_url)
        accts_2022 = _get_accounts_by_program(wallet, TOKEN_2022_PROGRAM_ID, rpc_url)

        if debug:
            print("Accounts(Tokenkeg) =", len(accts_tokenkeg))
            print("Accounts(Token2022) =", len(accts_2022))

        total = _sum_accounts_ui_amount(accts_tokenkeg, mint) + _sum_accounts_ui_amount(accts_2022, mint)

        if debug:
            print("Total =", total)

        return total

    except Exception as e:
        if debug:
            print("ERROR getting balance:", e)
        # Return 0 rather than blowing up your whole table
        return Decimal("0")
