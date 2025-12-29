# core/balances/evm.py
from __future__ import annotations

import time
import requests
from decimal import Decimal

BALANCE_OF_SIG = "0x70a08231"  # balanceOf(address)
DECIMALS_SIG = "0x313ce567"    # decimals()


def _pad_address(addr: str) -> str:
    return addr.lower().replace("0x", "").rjust(64, "0")


def _rpc_call(rpc_urls: list[str], method: str, params: list, tries_per_rpc: int = 2, timeout: int = 30):
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    last_err = None

    for url in rpc_urls:
        for attempt in range(tries_per_rpc):
            try:
                r = requests.post(url, json=payload, timeout=timeout)
                r.raise_for_status()
                data = r.json()
                if "error" in data:
                    last_err = RuntimeError(f"RPC error from {url}: {data['error']}")
                    time.sleep(0.25 * (attempt + 1))
                    continue
                return data["result"]
            except Exception as e:
                last_err = RuntimeError(f"RPC failure from {url}: {e}")
                time.sleep(0.25 * (attempt + 1))

    raise last_err if last_err else RuntimeError("Unknown RPC failure")


def get_erc20_decimals(contract: str, rpc_urls: list[str]) -> int:
    result = _rpc_call(rpc_urls, "eth_call", [{"to": contract, "data": DECIMALS_SIG}, "latest"])
    return int(result, 16)


def get_erc20_balance(wallet: str, contract: str, rpc_urls: list[str]) -> Decimal:
    data = BALANCE_OF_SIG + _pad_address(wallet)
    raw = _rpc_call(rpc_urls, "eth_call", [{"to": contract, "data": data}, "latest"])
    bal_int = Decimal(int(raw, 16))
    decimals = get_erc20_decimals(contract, rpc_urls)
    return bal_int / (Decimal(10) ** decimals)
