# core/balances/hyperliquid.py

from __future__ import annotations
from decimal import Decimal
from typing import Dict, Tuple
import requests

HL_INFO_URL = "https://api.hyperliquid.xyz/info"


def _post(payload: dict, timeout: int = 20) -> dict:
    r = requests.post(HL_INFO_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_spot_balances(user_evm_address: str) -> Dict[str, Decimal]:
    """
    Returns spot balances from Hyperliquid clearinghouse.
    Example return:
        {
            "HYPE": Decimal("123.45"),
            "USDC": Decimal("6789.01")
        }
    """
    payload = {
        "type": "spotClearinghouseState",
        "user": user_evm_address,
    }
    data = _post(payload)

    balances: Dict[str, Decimal] = {}
    for b in data.get("balances", []):
        coin = str(b.get("coin", "")).upper()
        total = Decimal(str(b.get("total", "0")))
        balances[coin] = total

    return balances


def get_mid_prices() -> Dict[str, Decimal]:
    """
    Returns mid prices from Hyperliquid.
    Example:
        { "HYPE": Decimal("21.34"), "BTC": Decimal("43120.5"), ... }
    """
    payload = {"type": "allMids"}
    data = _post(payload)

    prices: Dict[str, Decimal] = {}
    for k, v in data.items():
        try:
            prices[k.upper()] = Decimal(str(v))
        except Exception:
            continue

    return prices


def get_hype_position(
    user_evm_address: str,
) -> Tuple[Decimal, Decimal, Decimal]:
    """
    Convenience helper:
    Returns (hype_qty, hype_price, hype_usd_value)
    """
    balances = get_spot_balances(user_evm_address)
    prices = get_mid_prices()

    qty = balances.get("HYPE", Decimal("0"))
    price = prices.get("HYPE", Decimal("0"))
    usd = qty * price

    return qty, price, usd
