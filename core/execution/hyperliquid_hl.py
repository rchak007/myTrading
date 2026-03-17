# core/execution/hyperliquid_hl.py
"""
Hyperliquid spot trading module for HYPE/USDC (and any other HL spot pair).

Architecture notes:
- Hyperliquid is its own L1 — no gas, no DEX router, no token contracts.
- Your USDC must already be in the HL *spot* wallet (deposited via Arbitrum bridge).
- Trading is done via HL's order-book API using IOC limit orders as market equivalents.
- Authentication uses a dedicated API wallet key generated at app.hyperliquid.xyz/API,
  authorized to trade on behalf of your main Coinbase wallet address.

Setup:
  pip install hyperliquid-python-sdk --break-system-packages

Key generation (on the Pi, same Fernet key as all other bots):
  python3 -c "
  from cryptography.fernet import Fernet
  key = open('/etc/myTrading/bot.key', 'rb').read()
  f = Fernet(key)
  pk = input('paste HL API private key: ').strip()
  print(f.encrypt(pk.encode()).decode())
  "

asset_registry.json entry example:
  "HYPE": {
    "ticker": "HYPE",
    "blockchain": "hyperliquid",
    "yahoo_ticker": "HYPE32196-USD",
    "wallet_address": "0xYOUR_MAIN_COINBASE_WALLET",
    "hl_api_key_env": "HL_API_KEY_ENCRYPTED",
    "token_contract": "",
    "stablecoin_contract": "",
    "decimals": 4
  }

.env entry:
  HL_API_KEY_ENCRYPTED=gAAAAA...  (output of encryption above)
"""

from __future__ import annotations

import logging
import os

import eth_account

log = logging.getLogger("bot.hl")

# ── Lazy import guard ──────────────────────────────────────────────
# hyperliquid-python-sdk is only required if HL bots are configured.
# This avoids import errors on systems that haven't installed the SDK.
def _require_hl_sdk():
    try:
        from hyperliquid.info import Info          # noqa: F401
        from hyperliquid.exchange import Exchange  # noqa: F401
        from hyperliquid.utils import constants    # noqa: F401
    except ImportError:
        raise ImportError(
            "hyperliquid-python-sdk is not installed. "
            "Run: pip install hyperliquid-python-sdk --break-system-packages"
        )


# ── HL slippage (separate from SLIPPAGE_BPS used by Jupiter/Uniswap) ──
# HL uses fraction (0.03 = 3%) not basis points.
HL_SLIPPAGE = float(os.getenv("HL_SLIPPAGE", "0.03"))


# ─────────────────────────────────────────────────────────────────────
# Client initialisation
# ─────────────────────────────────────────────────────────────────────

def get_hl_clients(api_private_key: str, main_wallet_address: str):
    """
    Build and return (info, exchange) for Hyperliquid mainnet.

    Parameters
    ----------
    api_private_key    : hex private key of the *API wallet* (not your main wallet).
                         Generated at app.hyperliquid.xyz/API.
    main_wallet_address: 0x address of your main Coinbase wallet — this is the account
                         the API wallet is authorised to trade on behalf of.

    Returns
    -------
    (Info, Exchange) — both connected to mainnet.
    """
    _require_hl_sdk()
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange
    from hyperliquid.utils import constants

    wallet = eth_account.Account.from_key(api_private_key)
    info   = Info(constants.MAINNET_API_URL, skip_ws=True)
    exchange = Exchange(
        wallet,
        constants.MAINNET_API_URL,
        account_address=main_wallet_address,  # CRITICAL: main wallet, not API wallet
    )
    return info, exchange


# ─────────────────────────────────────────────────────────────────────
# Balance helpers
# ─────────────────────────────────────────────────────────────────────

def get_hl_spot_balances(info, address: str) -> dict[str, float]:
    """
    Returns token balances in the HL *spot* wallet.
    Example: {"USDC": 250.0, "HYPE": 12.345}

    Note: this is spot_user_state(), NOT user_state() which is perps.
    """
    state = info.spot_user_state(address)
    balances: dict[str, float] = {}
    for b in state.get("balances", []):
        coin  = b.get("coin", "")
        total = float(b.get("total", 0))
        balances[coin] = total
    return balances


def get_hl_mid_price(info, coin: str = "HYPE") -> float:
    """
    Fetch current mid price for a coin directly from HL's own feed.
    Useful as a more accurate execution price vs yfinance.
    Returns 0.0 on failure (caller should fall back to yfinance close).
    """
    try:
        mids = info.all_mids()
        return float(mids.get(coin, 0))
    except Exception as e:
        log.warning("HL mid price fetch failed for %s: %s", coin, e)
        return 0.0


# ─────────────────────────────────────────────────────────────────────
# Order execution
# ─────────────────────────────────────────────────────────────────────

def hl_market_buy(
    exchange,
    coin: str,
    usdc_amount: float,
    current_price: float,
    slippage: float = HL_SLIPPAGE,
) -> str:
    """
    Buy `coin` with USDC via an IOC limit order (market equivalent).

    HL spot does not support "spend exactly X USDC" — you specify size in
    the token being bought. We convert: qty = usdc_amount / current_price.
    The IOC limit price is set above mid by `slippage` to guarantee fill.

    Returns the raw SDK response as a string (logged as tx_sig).
    """
    # qty       = round(usdc_amount / current_price, 4)
    qty = round(usdc_amount / current_price, 2)
    # limit_px  = round(current_price * (1 + slippage), 6)
    limit_px = float(f"{current_price * (1 + slippage):.5g}")

    log.info("HL BUY %s: qty=%.4f limit_px=%.4f (usdc_equiv=%.2f slippage=%.1f%%)",
             coin, qty, limit_px, usdc_amount, slippage * 100)

    result = exchange.order(
        coin,
        is_buy=True,
        sz=qty,
        limit_px=limit_px,
        order_type={"limit": {"tif": "Ioc"}},
        reduce_only=False,
    )
    return str(result)


def hl_market_sell(
    exchange,
    coin: str,
    token_amount: float,
    current_price: float,
    slippage: float = HL_SLIPPAGE,
) -> str:
    """
    Sell `token_amount` of `coin` for USDC via an IOC limit order.
    The IOC limit price is set below mid by `slippage` to guarantee fill.

    Returns the raw SDK response as a string (logged as tx_sig).
    """
    # qty      = round(token_amount, 4)
    qty = round(token_amount, 2)
    # limit_px = round(current_price * (1 - slippage), 6)
    limit_px = float(f"{current_price * (1 - slippage):.5g}")

    log.info("HL SELL %s: qty=%.4f limit_px=%.4f slippage=%.1f%%",
             coin, qty, limit_px, slippage * 100)

    result = exchange.order(
        coin,
        is_buy=False,
        sz=qty,
        limit_px=limit_px,
        order_type={"limit": {"tif": "Ioc"}},
        reduce_only=False,
    )
    return str(result)