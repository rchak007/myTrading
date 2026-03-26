"""
core/execution/uniswap.py

EVM execution module — balance queries, token helpers, and DEX swaps.

SWAP ROUTING (updated 2026-03-25):
  ALL EVM swaps (Ethereum, Base, Optimism) now route through 0x Swap API v2.
  0x handles V2/V3/V4/multi-hop routing automatically — no more per-chain
  Uniswap version fragmentation, no more QuoterV2 fee-tier guessing.

  Previously used Uniswap V3 SwapRouter02 directly, which broke when:
    - Ethereum tokens (PNK, ONDO) migrated liquidity to Uniswap V4
    - Base RPC nonce timing caused "Invalid params" after approval tx

  evm_swap() is the single entry point for bot.py — replaces uniswap_swap_auto_fee().

REQUIRES:
  ZEROX_API_KEY in .env  — get free key at https://dashboard.0x.org/apps
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

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Chain configuration
# ═══════════════════════════════════════════════════════════════════
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

# WETH addresses per chain
WETH_ADDRESS = {
    "ethereum": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "base":     "0x4200000000000000000000000000000000000006",
    "optimism": "0x4200000000000000000000000000000000000006",
}

# Full chain config dict (used by bot.py router)
CHAIN_CONFIG = {
    chain: {
        "rpc_url":  CHAIN_RPC[chain],
        "chain_id": CHAIN_IDS[chain],
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
    "USDC":      6,
    "USDT":      6,
    "DAI":       18,
    # Native / wrapped
    "WETH":      18,
    "ETH":       18,
    # DeFi blue chips
    "LINK":      18,
    "UNI":       18,
    "AAVE":      18,
    "CRV":       18,
    "ENS":       18,
    "ONDO":      18,
    # Base / Optimism ecosystem
    "VIRTUAL":   18,
    "AERODROME": 18,
    "BALD":      18,
    # Other EVM tokens tracked in registry
    "PNK":       18,
    "RENDER":    18,
    "WLD":       18,
    "OP":        18,
    "ARB":       18,
    "GMX":       18,
    "GNO":       18,
    "LDO":       18,
    "RPL":       18,
    "PENDLE":    18,
    "ENA":       18,
    "FLUID":     18,
    "ZK":        18,
}

# USDC contract addresses per chain
USDC_ADDRESS = {
    "ethereum": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "base":     "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913",
    "optimism": "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",
}

# ═══════════════════════════════════════════════════════════════════
# Minimal ABIs (still needed for balance checks and approvals)
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

# ═══════════════════════════════════════════════════════════════════
# 0x Swap API v2 configuration
# ═══════════════════════════════════════════════════════════════════
ZEROX_API_BASE = "https://api.0x.org"


# ═══════════════════════════════════════════════════════════════════
# Web3 helpers
# ═══════════════════════════════════════════════════════════════════
def _get_w3(blockchain: str) -> Web3:
    """Return a connected Web3 instance for the given chain."""
    cfg = CHAIN_CONFIG.get(blockchain)
    if not cfg:
        raise ValueError(f"Unknown blockchain: '{blockchain}'. Must be one of: {list(CHAIN_CONFIG)}")
    w3 = Web3(Web3.HTTPProvider(cfg["rpc_url"], request_kwargs={"timeout": 30}))
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
        w3    = _get_w3(blockchain)
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
    w3  = _get_w3(blockchain)
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
    w3    = _get_w3(blockchain)
    token = w3.eth.contract(address=_checksum(token_contract), abi=ERC20_ABI)
    raw   = token.functions.balanceOf(_checksum(wallet_address)).call()
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
    Returns tx hash if approval was needed, None if allowance already sufficient.
    """
    token     = w3.eth.contract(address=_checksum(token_contract), abi=ERC20_ABI)
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

    signed  = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    log.info("Approval tx: %s | status=%d", tx_hash.hex(), receipt.status)

    if receipt.status != 1:
        raise RuntimeError(f"Approval failed: {tx_hash.hex()}")

    return tx_hash.hex()


# ═══════════════════════════════════════════════════════════════════
# 0x Swap API v2  — PRIMARY swap implementation for ALL EVM chains
# ═══════════════════════════════════════════════════════════════════
def swap_via_0x(
    *,
    blockchain: str,
    private_key: str,
    token_in: str,
    token_out: str,
    amount_in_human: float,
    token_in_decimals: int,
    token_out_decimals: int,
    slippage_bps: int = 100,
    deadline_seconds: int = 300,
) -> str:
    """
    Execute a swap via the 0x Swap API v2 (AllowanceHolder endpoint).

    Why 0x instead of Uniswap directly:
      - Handles V2/V3/V4/multi-hop routing automatically
      - No need to know which Uniswap version has the pool
      - Works for tokens like PNK/ONDO whose liquidity migrated to V4
      - Same routing engine used by Coinbase Wallet, MetaMask, Phantom
      - Eliminates nonce race condition (approval + swap handled cleanly)

    Flow:
      1. GET /swap/allowance-holder/quote
         → 0x finds best route and returns allowanceTarget + signed tx data
      2. Approve allowanceTarget to spend token_in (if needed)
      3. Sign + broadcast the transaction
      4. Wait for receipt
      5. Verify non-zero output via balance diff

    Parameters
    ----------
    blockchain         : "ethereum" | "base" | "optimism"
    private_key        : decrypted hex private key
    token_in           : ERC-20 contract address of token to sell
    token_out          : ERC-20 contract address of token to buy
    amount_in_human    : amount in human units (e.g. 94.25 for 94.25 ONDO)
    token_in_decimals  : decimals of token_in
    token_out_decimals : decimals of token_out
    slippage_bps       : slippage tolerance in basis points (100 = 1%, 300 = 3%)
    deadline_seconds   : kept for API consistency, not used by 0x

    Returns
    -------
    tx_hash : str (hex)

    Raises
    ------
    RuntimeError if API fails, tx reverts, or output balance is zero.
    """
    chain_id = CHAIN_IDS.get(blockchain)
    if chain_id is None:
        raise ValueError(f"0x: unsupported blockchain '{blockchain}'")

    api_key = os.getenv("ZEROX_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "ZEROX_API_KEY not set in .env — get a free key at dashboard.0x.org/apps"
        )

    w3      = _get_w3(blockchain)
    account = Account.from_key(private_key)
    wallet  = _checksum(account.address)

    amount_raw = int(amount_in_human * (10 ** token_in_decimals))

    log.info(
        "[%s] 0x swap: %.6f (%d raw) %s → %s | slippage=%dbps",
        blockchain, amount_in_human, amount_raw,
        token_in[:10], token_out[:10], slippage_bps,
    )

    headers = {
        "0x-api-key":   api_key,
        "0x-version":   "v2",
        "Content-Type": "application/json",
    }

    # ── Step 1: Get quote ──
    quote_url = f"{ZEROX_API_BASE}/swap/allowance-holder/quote"
    params = {
        "chainId":     str(chain_id),
        "sellToken":   _checksum(token_in),
        "buyToken":    _checksum(token_out),
        "sellAmount":  str(amount_raw),
        "taker":       wallet,
        "slippageBps": str(slippage_bps),
    }

    log.info("[%s] 0x: fetching quote...", blockchain)
    try:
        resp = requests.get(quote_url, headers=headers, params=params, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(
                f"[{blockchain}] 0x API error {resp.status_code}: {resp.text[:300]}"
            )
        quote = resp.json()
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"[{blockchain}] 0x API request failed: {e}")

    # Log the route 0x chose
    try:
        fills = quote.get("route", {}).get("fills", [])
        if fills:
            route_str = " → ".join(
                f"{f.get('source','?')}({int(f.get('proportionBps', 0)) / 100:.0f}%)"
                for f in fills
            )
            log.info("[%s] 0x route: %s", blockchain, route_str)
    except Exception:
        pass

    # Check liquidity available
    if not quote.get("liquidityAvailable", True):
        raise RuntimeError(
            f"[{blockchain}] 0x: no liquidity available for "
            f"{token_in[:10]} → {token_out[:10]}"
        )

    # Log any 0x protocol fee on this swap
    try:
        zero_ex_fee = quote.get("fees", {}).get("zeroExFee")
        if zero_ex_fee and int(zero_ex_fee.get("amount", 0)) > 0:
            fee_human = int(zero_ex_fee["amount"]) / (10 ** token_out_decimals)
            log.info("[%s] 0x protocol fee: %.4f token_out", blockchain, fee_human)
    except Exception:
        pass

    # ── Step 2: Approve allowanceTarget ──
    # 0x returns the spender address in issues.allowance.spender
    spender = None
    issues  = quote.get("issues", {})
    if isinstance(issues.get("allowance"), dict):
        spender = issues["allowance"].get("spender")

    # Fallback: get from transaction.to (the 0x AllowanceHolder contract)
    if not spender:
        spender = quote.get("transaction", {}).get("to")

    if not spender:
        raise RuntimeError(
            f"[{blockchain}] 0x: could not determine spender/allowanceTarget from quote"
        )

    log.info("[%s] 0x allowanceTarget: %s", blockchain, spender)

    approval_hash = ensure_approval(
        w3, account,
        token_contract=_checksum(token_in),
        spender=_checksum(spender),
        amount_raw=amount_raw,
        chain_id=chain_id,
    )

    # Wait for nonce to settle after approval
    # (Base/Ethereum RPCs can return stale nonce immediately after approval mines)
    if approval_hash is not None:
        log.debug("[%s] Approval mined — waiting 2s for nonce to settle", blockchain)
        time.sleep(2)

    # ── Step 3: Build, sign and broadcast ──
    tx_data = quote.get("transaction")
    if not tx_data:
        raise RuntimeError(f"[{blockchain}] 0x: no 'transaction' in quote response")

    nonce     = w3.eth.get_transaction_count(wallet)
    gas_price = w3.eth.gas_price

    tx = {
        "chainId":  chain_id,
        "to":       _checksum(tx_data["to"]),
        "data":     tx_data["data"],
        "value":    int(tx_data.get("value", 0)),
        "gas":      int(int(tx_data.get("gas", 300000)) * 1.25),  # 25% buffer
        "gasPrice": gas_price,
        "nonce":    nonce,
        "from":     wallet,
    }

    log.info(
        "[%s] 0x: broadcasting tx | gas=%d gasPrice=%d gwei",
        blockchain, tx["gas"], gas_price // 10**9,
    )

    signed  = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    log.info("[%s] 0x: tx sent: %s", blockchain, tx_hash.hex())

    # ── Step 4: Wait for receipt ──
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)

    if receipt.status != 1:
        raise RuntimeError(f"[{blockchain}] 0x swap REVERTED: {tx_hash.hex()}")

    # ── Step 5: Verify non-zero output via balance diff ──
    try:
        time.sleep(2)  # brief pause before balance check (Base block availability)
        erc20_out  = w3.eth.contract(address=_checksum(token_out), abi=ERC20_ABI)
        bal_after  = erc20_out.functions.balanceOf(wallet).call(
            block_identifier=receipt.blockNumber
        )
        bal_before = erc20_out.functions.balanceOf(wallet).call(
            block_identifier=receipt.blockNumber - 1
        )
        amount_out_raw = bal_after - bal_before

        log.info(
            "[%s] 0x balance check: before=%d after=%d diff=%d (block %d)",
            blockchain, bal_before, bal_after, amount_out_raw, receipt.blockNumber,
        )

        if amount_out_raw <= 0:
            raise RuntimeError(
                f"[{blockchain}] 0x tx {tx_hash.hex()} succeeded on-chain "
                f"but token_out ({token_out[:10]}) balance did not increase "
                f"(before={bal_before} after={bal_after})"
            )

        amount_out_human = amount_out_raw / (10 ** token_out_decimals)
        log.info(
            "[%s] ✅ 0x swap confirmed: %s | block=%d gas_used=%d | received=%.6f",
            blockchain, tx_hash.hex(), receipt.blockNumber,
            receipt.gasUsed, amount_out_human,
        )

    except RuntimeError:
        raise
    except Exception as e:
        # Balance check failed — trust receipt.status=1
        log.warning(
            "[%s] 0x: could not verify output balance: %s — trusting receipt.status=1",
            blockchain, e,
        )
        log.info(
            "[%s] ✅ 0x swap confirmed: %s | block=%d gas_used=%d",
            blockchain, tx_hash.hex(), receipt.blockNumber, receipt.gasUsed,
        )

    return tx_hash.hex()


# ═══════════════════════════════════════════════════════════════════
# evm_swap() — SINGLE ENTRY POINT for all EVM swaps
# ═══════════════════════════════════════════════════════════════════
def evm_swap(
    *,
    blockchain: str,
    private_key: str,
    token_in: str,
    token_out: str,
    amount_in_human: float,
    token_in_decimals: int,
    token_out_decimals: int,
    expected_out_per_in: float = 0.0,  # kept for API compatibility, not used by 0x
    slippage_bps: int = 50,
    deadline_seconds: int = 300,
) -> str:
    """
    Single entry point for all EVM swaps in bot.py.
    Routes ALL chains through 0x Swap API v2.

    Drop-in replacement for uniswap_swap_auto_fee() — identical signature.
    In bot.py just change the function name, all parameters stay the same.

    Ethereum  → 0x (V4 pools, multi-hop, PNK/ONDO all work)
    Base      → 0x (eliminates nonce race condition, better routing)
    Optimism  → 0x
    """
    log.info("[%s] evm_swap → 0x API (slippage=%dbps)", blockchain, slippage_bps)
    return swap_via_0x(
        blockchain=blockchain,
        private_key=private_key,
        token_in=token_in,
        token_out=token_out,
        amount_in_human=amount_in_human,
        token_in_decimals=token_in_decimals,
        token_out_decimals=token_out_decimals,
        slippage_bps=slippage_bps,
        deadline_seconds=deadline_seconds,
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