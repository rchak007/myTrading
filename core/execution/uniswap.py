"""
core/execution/uniswap.py

EVM DEX execution module for Uniswap V3 (SwapRouter02).
Supports Ethereum mainnet, Base, and Optimism — same code, different RPC + chain config.

KEY FIX (2026-03-07):
  - amountOutMinimum was 0 — swaps could silently return 0 tokens and still show "Success"
  - Now calculated from expected_price_usd * amount_in * (1 - slippage) with proper decimals
  - uniswap_swap() now verifies amountOut > 0 from receipt logs, raises if swap gave nothing
  - token_out_decimals added as explicit parameter
  - get_token_decimals_onchain() helper added for unknown tokens
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

# WETH addresses per chain
WETH_ADDRESS = {
    "ethereum": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "base":     "0x4200000000000000000000000000000000000006",
    "optimism": "0x4200000000000000000000000000000000000006",
}

# Fee tiers to try in order (0.05%, 0.3%, 1%)
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
# Known ERC-20 decimals
# Add any new token here — if missing, bot fetches on-chain (1 RPC call)
# ═══════════════════════════════════════════════════════════════════
ERC20_DECIMALS: dict[str, int] = {
    # Stablecoins
    "USDC":    6,
    "USDT":    6,
    "DAI":     18,
    # Native / wrapped
    "WETH":    18,
    "ETH":     18,
    # DeFi blue chips
    "LINK":    18,
    "UNI":     18,
    "AAVE":    18,
    "CRV":     18,
    "ENS":     18,
    "ONDO":    18,
    # Base / Optimism ecosystem
    "VIRTUAL": 18,
    "AERODROME": 18,
    "BALD":    18,
    # Other EVM tokens tracked in your registry
    "PNK":     18,
    "RENDER":  18,
    "WLD":     18,
    "OP":      18,
    "ARB":     18,
    "GMX":     18,
    "GNO":     18,
    "LDO":     18,
    "RPL":     18,
    "PENDLE":  18,
    "ENA":     18,
    "FLUID":   18,
    "ZK":      18,
}

# USDC contract addresses per chain — used as default stablecoin
USDC_ADDRESS = {
    "ethereum": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "base":     "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913",
    "optimism": "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",
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


def get_token_decimals_onchain(blockchain: str, token_contract: str) -> int:
    """
    Fetch decimals from the token contract on-chain.
    Falls back to 18 if the call fails.
    """
    try:
        w3 = _get_w3(blockchain)
        token = w3.eth.contract(address=_checksum(token_contract), abi=ERC20_ABI)
        return int(token.functions.decimals().call())
    except Exception as e:
        log.warning("Could not fetch decimals for %s on %s: %s — defaulting to 18",
                    token_contract, blockchain, e)
        return 18


def get_decimals(blockchain: str, token_contract: str, symbol: str = "") -> int:
    """
    Get decimals for a token:
    1. Check ERC20_DECIMALS by symbol (fast, no RPC)
    2. Fetch on-chain if not found
    """
    if symbol and symbol.upper() in ERC20_DECIMALS:
        return ERC20_DECIMALS[symbol.upper()]
    return get_token_decimals_onchain(blockchain, token_contract)


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
    nonce     = w3.eth.get_transaction_count(_checksum(account.address))
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

    signed   = account.sign_transaction(tx)
    tx_hash  = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt  = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
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
    private_key: str,
    token_in: str,
    token_out: str,
    amount_in_human: float,
    token_in_decimals: int,
    token_out_decimals: int,
    # FIX: pass expected price so we can compute a real amountOutMinimum
    # For SELL token→USDC: expected_price_usd = token price in USD
    # For BUY  USDC→token: expected_price_usd = 1 / token_price (i.e. USD per token inverted)
    # Set to 0.0 to disable the floor (NOT recommended for production)
    expected_out_per_in: float = 0.0,
    slippage_bps: int = 50,
    fee_tier: int = DEFAULT_FEE,
    deadline_seconds: int = 300,
) -> str:
    """
    Execute a Uniswap V3 exactInputSingle swap with a proper amountOutMinimum.

    Parameters
    ----------
    blockchain           : "ethereum" | "base" | "optimism"
    private_key          : hex private key (already decrypted from Fernet)
    token_in             : contract address of token to sell
    token_out            : contract address of token to buy
    amount_in_human      : amount to sell in human units (e.g. 1866.0 VIRTUAL)
    token_in_decimals    : decimals of token_in
    token_out_decimals   : decimals of token_out  ← NEW required param
    expected_out_per_in  : expected output tokens per input token, used to
                           compute amountOutMinimum with slippage floor.
                           e.g. selling VIRTUAL at $0.68 → USDC:
                             expected_out_per_in = 0.68  (each VIRTUAL → 0.68 USDC)
                           e.g. buying VIRTUAL at $0.68 with USDC:
                             expected_out_per_in = 1/0.68 ≈ 1.47  (each USDC → 1.47 VIRTUAL)
                           Pass 0.0 to skip the floor (unsafe).
    slippage_bps         : slippage tolerance in basis points (50 = 0.5%, 200 = 2%)
    fee_tier             : Uniswap V3 fee tier (500 / 3000 / 10000)
    deadline_seconds     : tx deadline window in seconds

    Returns
    -------
    tx_hash : str  (hex transaction hash)

    Raises
    ------
    RuntimeError if swap fails, tx reverts, or output is zero.
    """
    cfg      = CHAIN_CONFIG[blockchain]
    w3       = _get_w3(blockchain)
    account: LocalAccount = Account.from_key(private_key)

    token_in_cs  = _checksum(token_in)
    token_out_cs = _checksum(token_out)
    router_cs    = _checksum(cfg["router"])
    chain_id     = cfg["chain_id"]

    # Convert input to raw integer
    amount_raw = int(amount_in_human * (10 ** token_in_decimals))

    # ── Compute amountOutMinimum ──
    # This is the critical fix: never send 0 here for real swaps.
    # If expected_out_per_in is provided, compute a floor with slippage.
    if expected_out_per_in > 0:
        expected_out_human    = amount_in_human * expected_out_per_in
        slippage_factor       = 1.0 - (slippage_bps / 10_000.0)
        min_out_human         = expected_out_human * slippage_factor
        amount_out_minimum    = int(min_out_human * (10 ** token_out_decimals))
        log.info(
            "[%s] amountOutMinimum: expected=%.6f slippage=%.2f%% min=%.6f (raw=%d)",
            blockchain, expected_out_human, slippage_bps / 100,
            min_out_human, amount_out_minimum,
        )
    else:
        # ⚠️  No price given — floor is 0. Swap can silently give nothing.
        # Only acceptable in dry-run / testing scenarios.
        amount_out_minimum = 0
        log.warning(
            "[%s] ⚠️  amountOutMinimum=0 (no expected_out_per_in provided). "
            "Swap may silently return 0 tokens!", blockchain
        )

    log.info(
        "[%s] Swap %.6f (%d raw) %s → %s | fee=%d slippage=%dbps amountOutMin=%d",
        blockchain, amount_in_human, amount_raw,
        token_in_cs[:10], token_out_cs[:10], fee_tier, slippage_bps, amount_out_minimum,
    )

    # ── Step 1: Approve router to spend token_in ──
    ensure_approval(w3, account, token_in_cs, router_cs, amount_raw, chain_id)

    # ── Step 2: Build swap transaction ──
    router  = w3.eth.contract(address=router_cs, abi=UNISWAP_V3_ROUTER_ABI)
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
        gas_limit = int(gas_estimate * 1.2)
    except Exception as e:
        # Gas estimation often fails on Base/Optimism with "Invalid params" even
        # for valid pools — do NOT skip, just use a safe fallback and attempt anyway.
        log.warning("[%s] Gas estimation failed for fee=%d (%s) — using fallback 300k, attempting anyway",
                    blockchain, fee_tier, e)
        gas_limit = 300_000

    tx = router.functions.exactInputSingle(params).build_transaction({
        "chainId":  chain_id,
        "from":     _checksum(account.address),
        "nonce":    nonce,
        "gasPrice": gas_price,
        "gas":      gas_limit,
    })

    # ── Step 3: Sign and send ──
    signed   = account.sign_transaction(tx)
    tx_hash  = w3.eth.send_raw_transaction(signed.raw_transaction)

    log.info("[%s] Swap tx sent: %s", blockchain, tx_hash.hex())

    # ── Step 4: Wait for receipt ──
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)

    if receipt.status != 1:
        raise RuntimeError(
            f"[{blockchain}] Swap REVERTED: {tx_hash.hex()} "
            f"(fee_tier={fee_tier})"
        )

    # ── Step 5: Verify non-zero output via balance diff ──
    # Most reliable method: compare token_out balance before vs after the swap block.
    # Log-parsing is fragile (proxy contracts, wrapped transfers, etc.)
    try:
        erc20_out = w3.eth.contract(address=token_out_cs, abi=ERC20_ABI)
        bal_after  = erc20_out.functions.balanceOf(_checksum(account.address)).call(
            block_identifier=receipt.blockNumber
        )
        bal_before = erc20_out.functions.balanceOf(_checksum(account.address)).call(
            block_identifier=receipt.blockNumber - 1
        )
        amount_out_raw = bal_after - bal_before

        log.info(
            "[%s] Balance check: before=%d after=%d diff=%d (block %d)",
            blockchain, bal_before, bal_after, amount_out_raw, receipt.blockNumber,
        )

        if amount_out_raw <= 0:
            raise RuntimeError(
                f"[{blockchain}] Swap tx {tx_hash.hex()} succeeded on-chain "
                f"but token_out ({token_out_cs[:10]}) balance did not increase "
                f"(before={bal_before} after={bal_after}). Check pool liquidity."
            )

        amount_out_human = amount_out_raw / (10 ** token_out_decimals)
        log.info(
            "[%s] ✅ Swap confirmed: %s | block=%d gas_used=%d | received=%.6f token_out",
            blockchain, tx_hash.hex(), receipt.blockNumber,
            receipt.gasUsed, amount_out_human,
        )

    except RuntimeError:
        raise
    except Exception as e:
        # Balance check failed — don't block, trust receipt.status=1
        log.warning("[%s] Could not verify output balance: %s — trusting receipt.status=1", blockchain, e)
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
    token_out_decimals: int,
    expected_out_per_in: float = 0.0,
    slippage_bps: int = 50,
    deadline_seconds: int = 300,
) -> str:
    """
    Try Uniswap V3 swap across fee tiers (500 → 3000 → 10000).
    Returns tx_hash on first success, raises on all failures.

    expected_out_per_in: see uniswap_swap() docstring.
    For SELL (token → USDC): pass token_price_usd
    For BUY  (USDC → token): pass 1.0 / token_price_usd
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
                token_out_decimals=token_out_decimals,
                expected_out_per_in=expected_out_per_in,
                slippage_bps=slippage_bps,
                fee_tier=fee,
                deadline_seconds=deadline_seconds,
            )
        except Exception as e:
            log.warning("[%s] fee_tier=%d failed: %s — trying next", blockchain, fee, e)
            last_error = e

    raise RuntimeError(
        f"[{blockchain}] All fee tiers failed for "
        f"{token_in[:10]} → {token_out[:10]}: {last_error}"
    )


# ═══════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════
def to_smallest_evm(amount: float, decimals: int) -> int:
    """Convert human-readable float to smallest unit integer."""
    return int(amount * (10 ** decimals))


def from_smallest_evm(amount_raw: int, decimals: int) -> float:
    """Convert smallest unit integer to human-readable float."""
    return float(amount_raw) / (10 ** decimals)