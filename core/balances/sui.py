# core/balances/sui.py
from __future__ import annotations

import requests
from decimal import Decimal


SUI_RPC_MAINNET = "https://fullnode.mainnet.sui.io:443"


def _rpc_call(method: str, params: list, rpc_url: str = SUI_RPC_MAINNET):
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    r = requests.post(rpc_url, json=payload, timeout=30)
    r.raise_for_status()
    j = r.json()
    if "error" in j:
        raise RuntimeError(j["error"])
    return j["result"]


def get_coin_metadata(coin_type: str) -> dict:
    return _rpc_call("suix_getCoinMetadata", [coin_type])


def get_coin_balance(owner: str, coin_type: str) -> Decimal:
    res = _rpc_call("suix_getBalance", [owner, coin_type])
    raw = Decimal(res.get("totalBalance", "0"))
    meta = get_coin_metadata(coin_type)
    decimals = int(meta.get("decimals", 0))
    return raw / (Decimal(10) ** decimals)
