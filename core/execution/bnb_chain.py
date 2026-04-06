"""
core/execution/bnb_chain.py

BNB Smart Chain (BSC) execution module — balance queries and DEX swaps.

SWAP ROUTING:
  bnb_swap() tries PancakeSwap V3 first (fee tiers: 0.01%, 0.05%, 0.25%, 1%).
  If QuoterV2 finds no V3 pool → automatically falls back to 0x Swap API v2.

  Manual override for testing 0x directly:
    FORCE_0X=true python3 -m bots.bot SELL DOGE 25

REQUIRES:
  BSC_RPC_URL in .env  — e.g. https://bsc-dataseed.binance.org or an Ankr/QuickNode endpoint
  ZEROX_API_KEY in .env — same key used by the Ethereum/Base 0x fallback

Stablecoin: USDT (BSC)  → 0x55d398326f99059fF775485246999027B3197955
  Most BSC pairs route through USDT, not USDC.

Architecture note:
  This module deliberately mirrors uniswap.py's structure and reuses
  the same shared helpers (ERC20_ABI, ensure_approval, _checksum, etc.)
  from uniswap.py to avoid code duplication.  Balance queries also reuse
  get_evm_native_balance / get_evm_token_balance from uniswap.py once
  BSC is added to CHAIN_CONFIG.
"""

from __future__ import annotations

import logging
import os
import time
import requests
from typing import Optional

from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from eth_account import Account
from eth_account.signers.local import LocalAccount

log = logging.getLogger("bot.bnb")

# ═══════════════════════════════════════════════════════════════════
# BSC chain configuration
# ═══════════════════════════════════════════════════════════════════
BSC_RPC_URL  = os.getenv("BSC_RPC_URL", "https://bsc-dataseed.binance.org")
BSC_CHAIN_ID = 56

# Wrapped BNB
WBNB_ADDRESS = "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c"

# USDT on BSC (the dominant stablecoin — most pairs route through USDT)
BSC_USDT_ADDRESS = "0x55d398326f99059fF775485246999027B3197955"

# PancakeSwap V3 contracts on BSC
PANCAKE_V3_ROUTER   = "0x13f4EA83D0bd40E75C8222255bc855a974568Dd4"  # SmartRouter V3
PANCAKE_V3_QUOTER   = "0xB048Bbc1Ee6b733FFfCFb9e9CeF7375518e25997"  # QuoterV2

# PancakeSwap V3 fee tiers (0.01%, 0.05%, 0.25%, 1%)
PANCAKE_FEE_TIERS = [100, 500, 2500, 10000]

# BNB gas fee reserve
BNB_FEE_RESERVE = float(os.getenv("BNB_FEE_RESERVE", "0.005"))  # ~$3 buffer for gas

# ═══════════════════════════════════════════════════════════════════
# Reuse shared ABIs and helpers from uniswap.py
# ═══════════════════════════════════════════════════════════════════
from core.execution.uniswap import (
    ERC20_ABI,
    ensure_approval,
    UNISWAP_V3_ROUTER_ABI,   # PancakeSwap V3 uses same interface
    QUOTER_V2_ABI,            # PancakeSwap QuoterV2 uses same interface
)


# ═══════════════════════════════════════════════════════════════════
# Web3 helpers
# ═══════════════════════════════════════════════════════════════════
_w3_cache: Web3 | None = None


def _get_w3() -> Web3:
    global _w3_cache
    if _w3_cache is None:
        _w3_cache = Web3(Web3.HTTPProvider(BSC_RPC_URL, request_kwargs={"timeout": 30}))
        # BSC uses Clique PoA — needs ExtraDataToPOA middleware
        _w3_cache.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        if not _w3_cache.is_connected():
            _w3_cache = None
            raise RuntimeError(f"Cannot connect to BSC RPC: {BSC_RPC_URL}")
    return _w3_cache


def _checksum(address: str) -> str:
    return Web3.to_checksum_address(address)


# ═══════════════════════════════════════════════════════════════════
# Balance queries
# ═══════════════════════════════════════════════════════════════════
def get_bnb_native_balance(wallet_address: str) -> float:
    """Get native BNB balance."""
    w3  = _get_w3()
    wei = w3.eth.get_balance(_checksum(wallet_address))
    return float(w3.from_wei(wei, "ether"))


def get_bnb_token_balance(
    wallet_address: str,
    token_contract: str,
    decimals: Optional[int] = None,
) -> float:
    """Get BEP-20 token balance on BSC."""
    w3    = _get_w3()
    token = w3.eth.contract(address=_checksum(token_contract), abi=ERC20_ABI)
    raw   = token.functions.balanceOf(_checksum(wallet_address)).call()
    if decimals is None:
        decimals = token.functions.decimals().call()
    return float(raw) / (10 ** decimals)


# ═══════════════════════════════════════════════════════════════════
# PancakeSwap V3 — QuoterV2 (read-only, zero gas)
# ═══════════════════════════════════════════════════════════════════
def quote_best_fee_tier(
    token_in: str,
    token_out: str,
    amount_in_raw: int,
    fee_tiers: list | None = None,
) -> tuple:
    """
    Use PancakeSwap QuoterV2 to find the best V3 fee tier on BSC.
    Returns (best_fee_tier, best_amount_out_raw).
    If best_amount_out_raw == 0 → no V3 pool → bnb_swap() will use 0x fallback.
    """
    if fee_tiers is None:
        fee_tiers = PANCAKE_FEE_TIERS

    w3       = _get_w3()
    quoter   = w3.eth.contract(address=_checksum(PANCAKE_V3_QUOTER), abi=QUOTER_V2_ABI)
    best_fee = fee_tiers[0]
    best_out = 0

    for fee in fee_tiers:
        try:
            result = quoter.functions.quoteExactInputSingle({
                "tokenIn":           _checksum(token_in),
                "tokenOut":          _checksum(token_out),
                "amountIn":          amount_in_raw,
                "fee":               fee,
                "sqrtPriceLimitX96": 0,
            }).call()
            amount_out = result[0]
            log.info("[bnb] PancakeV3 QuoterV2 fee=%d → amountOut=%d", fee, amount_out)
            if amount_out > best_out:
                best_out = amount_out
                best_fee = fee
        except Exception as e:
            log.debug("[bnb] PancakeV3 QuoterV2 fee=%d failed: %s", fee, e)

    if best_out == 0:
        log.warning("[bnb] PancakeV3: no pool for %s→%s — will fall back to 0x",
                    token_in[:10], token_out[:10])
    else:
        log.info("[bnb] PancakeV3 best: fee=%d amountOut=%d", best_fee, best_out)

    return (best_fee, best_out)


# ═══════════════════════════════════════════════════════════════════
# PancakeSwap V3 — swap (exactInputSingle via SmartRouter multicall)
# ═══════════════════════════════════════════════════════════════════
def pancake_swap(
    *,
    private_key: str,
    token_in: str,
    token_out: str,
    amount_in_human: float,
    token_in_decimals: int,
    token_out_decimals: int,
    expected_out_per_in: float = 0.0,
    slippage_bps: int = 50,
    fee_tier: int = 2500,
    deadline_seconds: int = 300,
) -> str:
    """Execute a PancakeSwap V3 exactInputSingle swap on BSC."""
    w3       = _get_w3()
    account  = Account.from_key(private_key)
    wallet   = _checksum(account.address)

    token_in_cs  = _checksum(token_in)
    token_out_cs = _checksum(token_out)
    router_cs    = _checksum(PANCAKE_V3_ROUTER)
    amount_raw   = int(amount_in_human * (10 ** token_in_decimals))

    # Compute amountOutMinimum
    if expected_out_per_in > 0:
        expected_out_human = amount_in_human * expected_out_per_in
        slippage_factor    = 1.0 - (slippage_bps / 10_000.0)
        min_out_human      = expected_out_human * slippage_factor
        amount_out_minimum = int(min_out_human * (10 ** token_out_decimals))
        log.info("[bnb] PancakeV3 amountOutMin: expected=%.6f slippage=%.2f%% min=%.6f",
                 expected_out_human, slippage_bps / 100, min_out_human)
    else:
        amount_out_minimum = 0
        log.warning("[bnb] PancakeV3: no expected_out_per_in — amountOutMinimum=0 (risky)")

    # Approve router
    approval_hash = ensure_approval(
        w3, account,
        token_contract=token_in_cs,
        spender=router_cs,
        amount_raw=amount_raw,
        chain_id=BSC_CHAIN_ID,
    )
    if approval_hash:
        log.debug("[bnb] PancakeV3 approval mined — waiting 3s")
        time.sleep(3)

    # Build swap calldata
    router = w3.eth.contract(address=router_cs, abi=UNISWAP_V3_ROUTER_ABI)

    swap_data = router.encode_abi(
        "exactInputSingle",
        [{
            "tokenIn":           token_in_cs,
            "tokenOut":          token_out_cs,
            "fee":               fee_tier,
            "recipient":         wallet,
            "amountIn":          amount_raw,
            "amountOutMinimum":  amount_out_minimum,
            "sqrtPriceLimitX96": 0,
        }],
    )

    deadline = int(time.time()) + deadline_seconds
    tx_data  = router.encode_abi("multicall", [deadline, [swap_data]])

    nonce     = w3.eth.get_transaction_count(wallet)
    gas_price = w3.eth.gas_price
    MIN_GAS_PRICE = 3 * 10 ** 9  # 3 gwei — BSC minimum
    if gas_price < MIN_GAS_PRICE:
        log.warning("[bnb] gas_price=%d — using 3 gwei floor", gas_price)
        gas_price = MIN_GAS_PRICE

    tx = {
        "chainId":  BSC_CHAIN_ID,
        "to":       router_cs,
        "data":     tx_data,
        "value":    0,
        "gas":      350_000,
        "gasPrice": gas_price,
        "nonce":    nonce,
        "from":     wallet,
    }

    signed  = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    log.info("[bnb] PancakeV3 tx sent: %s", tx_hash.hex())

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    if receipt.status != 1:
        raise RuntimeError(f"[bnb] PancakeV3 swap REVERTED: {tx_hash.hex()}")

    log.info("[bnb] ✅ PancakeV3 swap confirmed: %s | block=%d gas_used=%d",
             tx_hash.hex(), receipt.blockNumber, receipt.gasUsed)
    return tx_hash.hex()


# ═══════════════════════════════════════════════════════════════════
# 0x Swap API — BSC fallback
# ═══════════════════════════════════════════════════════════════════
def swap_via_0x(
    *,
    private_key: str,
    token_in: str,
    token_out: str,
    amount_in_human: float,
    token_in_decimals: int,
    token_out_decimals: int,
    slippage_bps: int = 50,
    deadline_seconds: int = 300,
) -> str:
    """Execute a swap on BSC via 0x Swap API v2 (aggregator fallback)."""
    api_key = os.getenv("ZEROX_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ZEROX_API_KEY not set — required for 0x BSC swaps")

    w3      = _get_w3()
    account = Account.from_key(private_key)
    wallet  = _checksum(account.address)

    amount_raw = int(amount_in_human * (10 ** token_in_decimals))
    slippage   = slippage_bps / 10_000.0

    # 0x API v2 uses chain-specific subdomains
    base_url = "https://bsc.api.0x.org"
    headers  = {"0x-api-key": api_key, "0x-version": "2"}

    # Step 1: Get quote
    params = {
        "chainId":     BSC_CHAIN_ID,
        "sellToken":   _checksum(token_in),
        "buyToken":    _checksum(token_out),
        "sellAmount":  str(amount_raw),
        "taker":       wallet,
        "slippageBps": slippage_bps,
    }

    log.info("[bnb] 0x quote: %s→%s amount=%d slippage=%d bps",
             token_in[:10], token_out[:10], amount_raw, slippage_bps)

    resp = requests.get(f"{base_url}/swap/permit2/quote", params=params,
                        headers=headers, timeout=30)
    resp.raise_for_status()
    quote = resp.json()

    if "transaction" not in quote:
        raise RuntimeError(f"[bnb] 0x: no 'transaction' in quote: {quote}")

    # Step 2: Approve
    spender = None
    issues  = quote.get("issues", {})
    if isinstance(issues.get("allowance"), dict):
        spender = issues["allowance"].get("spender")
    if not spender:
        spender = quote.get("transaction", {}).get("to")
    if not spender:
        raise RuntimeError("[bnb] 0x: could not determine spender")

    approval_hash = ensure_approval(
        w3, account,
        token_contract=_checksum(token_in),
        spender=_checksum(spender),
        amount_raw=amount_raw,
        chain_id=BSC_CHAIN_ID,
    )
    if approval_hash:
        time.sleep(3)

    # Step 3: Build and send tx
    tx_data   = quote["transaction"]
    nonce     = w3.eth.get_transaction_count(wallet)
    gas_price = w3.eth.gas_price
    MIN_GAS_PRICE = 3 * 10 ** 9
    if gas_price < MIN_GAS_PRICE:
        gas_price = MIN_GAS_PRICE

    tx = {
        "chainId":  BSC_CHAIN_ID,
        "to":       _checksum(tx_data["to"]),
        "data":     tx_data["data"],
        "value":    int(tx_data.get("value", 0)),
        "gas":      int(int(tx_data.get("gas", 350000)) * 1.25),
        "gasPrice": gas_price,
        "nonce":    nonce,
        "from":     wallet,
    }

    signed  = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    log.info("[bnb] 0x tx sent: %s", tx_hash.hex())

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
    if receipt.status != 1:
        raise RuntimeError(f"[bnb] 0x swap REVERTED: {tx_hash.hex()}")

    log.info("[bnb] ✅ 0x swap confirmed: %s | block=%d gas_used=%d",
             tx_hash.hex(), receipt.blockNumber, receipt.gasUsed)
    return tx_hash.hex()


# ═══════════════════════════════════════════════════════════════════
# bnb_swap() — SINGLE ENTRY POINT for all BSC swaps in bot.py
# ═══════════════════════════════════════════════════════════════════
def bnb_swap(
    *,
    private_key: str,
    token_in: str,
    token_out: str,
    amount_in_human: float,
    token_in_decimals: int,
    token_out_decimals: int,
    expected_out_per_in: float = 0.0,
    slippage_bps: int = 50,
    deadline_seconds: int = 300,
) -> str:
    """
    Unified BSC swap entry point for bot.py.
    Same signature as evm_swap() for consistency.

    Routing:
      1. FORCE_0X=true → skip to 0x
      2. Try PancakeSwap V3 via QuoterV2
      3. No V3 pool → fall back to 0x
    """
    force_0x = os.getenv("FORCE_0X", "false").lower() == "true"

    if force_0x:
        log.info("[bnb] FORCE_0X=true — skipping PancakeV3, going to 0x")
        return swap_via_0x(
            private_key=private_key,
            token_in=token_in,
            token_out=token_out,
            amount_in_human=amount_in_human,
            token_in_decimals=token_in_decimals,
            token_out_decimals=token_out_decimals,
            slippage_bps=slippage_bps,
            deadline_seconds=deadline_seconds,
        )

    # Try PancakeSwap V3 first
    amount_in_raw          = int(amount_in_human * (10 ** token_in_decimals))
    best_fee, best_out_raw = quote_best_fee_tier(token_in, token_out, amount_in_raw)

    if best_out_raw > 0:
        quoted_out_human = best_out_raw / (10 ** token_out_decimals)
        quoted_rate      = quoted_out_human / amount_in_human
        log.info("[bnb] PancakeV3 pool found: rate=%.6f fee=%d", quoted_rate, best_fee)
        return pancake_swap(
            private_key=private_key,
            token_in=token_in,
            token_out=token_out,
            amount_in_human=amount_in_human,
            token_in_decimals=token_in_decimals,
            token_out_decimals=token_out_decimals,
            expected_out_per_in=quoted_rate,
            slippage_bps=slippage_bps,
            fee_tier=best_fee,
            deadline_seconds=deadline_seconds,
        )

    # No V3 pool → 0x fallback
    log.warning("[bnb] No PancakeV3 pool — falling back to 0x")
    return swap_via_0x(
        private_key=private_key,
        token_in=token_in,
        token_out=token_out,
        amount_in_human=amount_in_human,
        token_in_decimals=token_in_decimals,
        token_out_decimals=token_out_decimals,
        slippage_bps=slippage_bps,
        deadline_seconds=deadline_seconds,
    )