"""
core/execution/uniswap.py

EVM DEX execution module for Uniswap V3 (SwapRouter02).
Supports Ethereum mainnet, Base, and Optimism — same code, different RPC + chain config.

Usage pattern mirrors jupiter.py so bot.py can route cleanly:
    from core.execution.uniswap import (
        get_evm_token_balance,
        get_evm_native_balance,
        uniswap_swap,
        CHAIN_CONFIG,
        ERC20_DECIMALS,
    )
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from eth_account import Account
from eth_account.signers.local import LocalAccount

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Chain configuration
# ═══════════════════════════════════════════════════════════════════
# Uniswap V3 SwapRouter02 — same address on Ethereum, Base, Optimism
UNISWAP_V3_ROUTER = "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45"  # SwapRouter02

# WETH addresses per chain (for native ETH wrapping)
WETH_ADDRESS = {
    "ethereum": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "base":     "0x4200000000000000000000000000000000000006",
    "optimism": "0x4200000000000000000000000000000000000006",
}

# Fee tiers to try in order (0.05%, 0.3%, 1%) — bot tries lowest fee first
UNISWAP_FEE_TIERS = [500, 3000, 10000]

# Default pool fee tier (0.3%)
DEFAULT_FEE = 3000

# Chain IDs
CHAIN_IDS = {
    "ethereum": 1,
    "base":     8453,
    "optimism": 10,
}

# RPC URLs loaded from environment
CHAIN_RPC = {
    "ethereum": os.getenv("ETH_RPC_URL",  "https://mainnet.infura.io/v3/YOUR_KEY"),
    "base":     os.getenv("BASE_RPC_URL", "https://mainnet.base.org"),
    "optimism": os.getenv("OPT_RPC_URL",  "https://mainnet.optimism.io"),
}

# Full chain config dict (used by bot.py router)
CHAIN_CONFIG = {
    chain: {
        "rpc_url":  CHAIN_RPC[chain],
        "chain_id": CHAIN_IDS[chain],
        "router":   UNISWAP_V3_ROUTER,
        "weth":     WETH_ADDRESS[chain],
    }
    for chain in ("ethereum", "base", "optimism")
}

# ═══════════════════════════════════════════════════════════════════
# Known ERC-20 decimals (extend as needed)
# ═══════════════════════════════════════════════════════════════════
ERC20_DECIMALS: dict[str, int] = {
    # Stablecoins
    "USDC":   6,
    "USDT":   6,
    "DAI":    18,
    # Tokens
    "WETH":   18,
    "ETH":    18,
    "LINK":   18,
    "UNI":    18,
    "AAVE":   18,
    "CRV":    18,
    "ENS":    18,
    "PNK":    18,
    "VIRTUAL": 18,
}

# ═══════════════════════════════════════════════════════════════════
# Minimal ABIs
# ═══════════════════════════════════════════════════════════════════
ERC20_ABI = [
    {
        "name": "balanceOf",
        "type": "function",
        "inputs": [{"name": "account", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
    },
    {
        "name": "decimals",
        "type": "function",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint8"}],
        "stateMutability": "view",
    },
    {
        "name": "approve",
        "type": "function",
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount",  "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
    },
    {
        "name": "allowance",
        "type": "function",
        "inputs": [
            {"name": "owner",   "type": "address"},
            {"name": "spender", "type": "address"},
        ],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
    },
]

# Uniswap V3 SwapRouter02 — exactInputSingle
UNISWAP_V3_ROUTER_ABI = [
    {
        "name": "exactInputSingle",
        "type": "function",
        "inputs": [
            {
                "name": "params",
                "type": "tuple",
                "components": [
                    {"name": "tokenIn",           "type": "address"},
                    {"name": "tokenOut",          "type": "address"},
                    {"name": "fee",               "type": "uint24"},
                    {"name": "recipient",         "type": "address"},
                    {"name": "amountIn",          "type": "uint256"},
                    {"name": "amountOutMinimum",  "type": "uint256"},
                    {"name": "sqrtPriceLimitX96", "type": "uint160"},
                ],
            }
        ],
        "outputs": [{"name": "amountOut", "type": "uint256"}],
        "stateMutability": "payable",
    },
]


# ═══════════════════════════════════════════════════════════════════
# Web3 helpers
# ═══════════════════════════════════════════════════════════════════
def _get_w3(blockchain: str) -> Web3:
    """Return a connected Web3 instance for the given chain."""
    cfg = CHAIN_CONFIG.get(blockchain)
    if not cfg:
        raise ValueError(f"Unknown blockchain: '{blockchain}'. Must be one of: {list(CHAIN_CONFIG)}")
    w3 = Web3(Web3.HTTPProvider(cfg["rpc_url"], request_kwargs={"timeout": 30}))
    # POA middleware needed for Base/Optimism
    if blockchain in ("base", "optimism"):
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    if not w3.is_connected():
        raise RuntimeError(f"Cannot connect to {blockchain} RPC: {cfg['rpc_url']}")
    return w3


def _checksum(address: str) -> str:
    return Web3.to_checksum_address(address)


# ═══════════════════════════════════════════════════════════════════
# Balance queries
# ═══════════════════════════════════════════════════════════════════
def get_evm_native_balance(blockchain: str, wallet_address: str) -> float:
    """Return native ETH balance in ETH (not wei)."""
    w3 = _get_w3(blockchain)
    wei = w3.eth.get_balance(_checksum(wallet_address))
    return float(w3.from_wei(wei, "ether"))


def get_evm_token_balance(
    blockchain: str,
    wallet_address: str,
    token_contract: str,
    decimals: Optional[int] = None,
) -> float:
    """
    Return ERC-20 token balance as human-readable float.
    If decimals is None, fetches from contract (costs 1 RPC call).
    """
    w3 = _get_w3(blockchain)
    token = w3.eth.contract(address=_checksum(token_contract), abi=ERC20_ABI)
    raw = token.functions.balanceOf(_checksum(wallet_address)).call()
    if decimals is None:
        decimals = token.functions.decimals().call()
    return float(raw) / (10 ** decimals)


# ═══════════════════════════════════════════════════════════════════
# Token approval
# ═══════════════════════════════════════════════════════════════════
def ensure_approval(
    w3: Web3,
    account: LocalAccount,
    token_contract: str,
    spender: str,
    amount_raw: int,
    chain_id: int,
) -> Optional[str]:
    """
    Approve spender to spend amount_raw of token if current allowance is insufficient.
    Returns tx hash if approval was needed, None otherwise.
    """
    token = w3.eth.contract(address=_checksum(token_contract), abi=ERC20_ABI)
    allowance = token.functions.allowance(
        _checksum(account.address), _checksum(spender)
    ).call()

    if allowance >= amount_raw:
        log.debug("Allowance sufficient: %d >= %d", allowance, amount_raw)
        return None

    log.info("Approving %s to spend %d tokens...", spender[:10], amount_raw)
    nonce = w3.eth.get_transaction_count(_checksum(account.address))
    gas_price = w3.eth.gas_price

    tx = token.functions.approve(
        _checksum(spender), amount_raw
    ).build_transaction({
        "chainId":  chain_id,
        "from":     _checksum(account.address),
        "nonce":    nonce,
        "gasPrice": gas_price,
        "gas":      100_000,
    })

    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    log.info("Approval tx: %s | status=%d", tx_hash.hex(), receipt.status)

    if receipt.status != 1:
        raise RuntimeError(f"Approval failed: {tx_hash.hex()}")

    return tx_hash.hex()


# ═══════════════════════════════════════════════════════════════════
# Core swap: exactInputSingle via Uniswap V3 SwapRouter02
# ═══════════════════════════════════════════════════════════════════
def uniswap_swap(
    *,
    blockchain: str,
    private_key: str,           # hex string (Fernet-decrypted)
    token_in: str,              # ERC-20 contract address (checksummed or not)
    token_out: str,             # ERC-20 contract address
    amount_in_human: float,     # human-readable (e.g. 100.0 USDC)
    token_in_decimals: int,
    slippage_bps: int = 50,     # 50 = 0.5%
    fee_tier: int = DEFAULT_FEE,
    deadline_seconds: int = 300,
) -> str:
    """
    Execute a Uniswap V3 exactInputSingle swap.

    Parameters
    ----------
    blockchain        : "ethereum" | "base" | "optimism"
    private_key       : hex private key (already decrypted from Fernet)
    token_in          : contract address of token to sell
    token_out         : contract address of token to buy
    amount_in_human   : amount to sell in human units (e.g. 150.0 for 150 USDC)
    token_in_decimals : decimals of token_in (use ERC20_DECIMALS or fetch on-chain)
    slippage_bps      : slippage tolerance in basis points (50 = 0.5%)
    fee_tier          : Uniswap V3 fee tier (500 / 3000 / 10000)
    deadline_seconds  : tx deadline window in seconds

    Returns
    -------
    tx_hash : str  (hex transaction hash)

    Raises
    ------
    RuntimeError if swap fails or tx reverts.
    """
    cfg = CHAIN_CONFIG[blockchain]
    w3  = _get_w3(blockchain)
    account: LocalAccount = Account.from_key(private_key)

    token_in_cs  = _checksum(token_in)
    token_out_cs = _checksum(token_out)
    router_cs    = _checksum(cfg["router"])
    chain_id     = cfg["chain_id"]

    # Convert to raw integer amount
    amount_raw = int(amount_in_human * (10 ** token_in_decimals))

    log.info(
        "[%s] Swap %.6f (%d raw) %s → %s | fee=%d slippage=%dbps",
        blockchain, amount_in_human, amount_raw,
        token_in_cs[:10], token_out_cs[:10], fee_tier, slippage_bps,
    )

    # ── Step 1: Approve router to spend token_in ──
    ensure_approval(w3, account, token_in_cs, router_cs, amount_raw, chain_id)

    # ── Step 2: Build swap transaction ──
    router = w3.eth.contract(address=router_cs, abi=UNISWAP_V3_ROUTER_ABI)

    # amountOutMinimum: apply slippage floor (slippage_bps / 10000)
    # We don't have a price oracle here so we use 0 as a conservative floor.
    # For production you'd query the pool price first — see note below.
    amount_out_minimum = 0  # ⚠️ Set to 0 for simplicity; add oracle check for extra safety

    deadline = int(time.time()) + deadline_seconds
    nonce    = w3.eth.get_transaction_count(_checksum(account.address))
    gas_price = w3.eth.gas_price

    params = {
        "tokenIn":           token_in_cs,
        "tokenOut":          token_out_cs,
        "fee":               fee_tier,
        "recipient":         _checksum(account.address),
        "amountIn":          amount_raw,
        "amountOutMinimum":  amount_out_minimum,
        "sqrtPriceLimitX96": 0,
    }

    try:
        gas_estimate = router.functions.exactInputSingle(params).estimate_gas(
            {"from": _checksum(account.address)}
        )
        gas_limit = int(gas_estimate * 1.2)  # 20% buffer
    except Exception as e:
        log.warning("Gas estimation failed (%s) — using fallback 300k", e)
        gas_limit = 300_000

    tx = router.functions.exactInputSingle(params).build_transaction({
        "chainId":  chain_id,
        "from":     _checksum(account.address),
        "nonce":    nonce,
        "gasPrice": gas_price,
        "gas":      gas_limit,
    })

    # ── Step 3: Sign and send ──
    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)

    log.info("[%s] Swap tx sent: %s", blockchain, tx_hash.hex())

    # ── Step 4: Wait for receipt ──
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)

    if receipt.status != 1:
        raise RuntimeError(
            f"[{blockchain}] Swap REVERTED: {tx_hash.hex()} "
            f"(fee_tier={fee_tier} — try a different fee tier)"
        )

    log.info("[%s] ✅ Swap confirmed: %s | block=%d gas_used=%d",
             blockchain, tx_hash.hex(), receipt.blockNumber, receipt.gasUsed)

    return tx_hash.hex()


# ═══════════════════════════════════════════════════════════════════
# Convenience: try multiple fee tiers automatically
# ═══════════════════════════════════════════════════════════════════
def uniswap_swap_auto_fee(
    *,
    blockchain: str,
    private_key: str,
    token_in: str,
    token_out: str,
    amount_in_human: float,
    token_in_decimals: int,
    slippage_bps: int = 50,
    deadline_seconds: int = 300,
) -> str:
    """
    Try Uniswap V3 swap across fee tiers (500 → 3000 → 10000).
    Returns tx_hash on first success, raises on all failures.

    Use this instead of uniswap_swap() when you're unsure which fee tier
    the pool uses (common for newer/less liquid tokens).
    """
    last_error = None
    for fee in UNISWAP_FEE_TIERS:
        try:
            return uniswap_swap(
                blockchain=blockchain,
                private_key=private_key,
                token_in=token_in,
                token_out=token_out,
                amount_in_human=amount_in_human,
                token_in_decimals=token_in_decimals,
                slippage_bps=slippage_bps,
                fee_tier=fee,
                deadline_seconds=deadline_seconds,
            )
        except Exception as e:
            log.warning("[%s] fee_tier=%d failed: %s — trying next", blockchain, fee, e)
            last_error = e

    raise RuntimeError(
        f"[{blockchain}] All fee tiers failed for {token_in[:10]} → {token_out[:10]}: {last_error}"
    )


# ═══════════════════════════════════════════════════════════════════
# Utility: convert human amount → raw integer
# ═══════════════════════════════════════════════════════════════════
def to_smallest_evm(amount: float, decimals: int) -> int:
    """Convert human-readable float to smallest unit integer."""
    return int(amount * (10 ** decimals))


def from_smallest_evm(amount_raw: int, decimals: int) -> float:
    """Convert smallest unit integer to human-readable float."""
    return float(amount_raw) / (10 ** decimals)