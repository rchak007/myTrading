"""
core/execution/uniswap.py

EVM execution module — balance queries, token helpers, and DEX swaps.

SWAP ROUTING (updated 2026-03-26):
  evm_swap() tries Uniswap V3 first (free, no protocol fee).
  If QuoterV2 finds no V3 pool → automatically falls back to 0x Swap API v2.
  0x handles V2/V3/V4/multi-hop routing — solves tokens like PNK/ONDO
  whose liquidity migrated to Uniswap V4.

  Manual override for testing 0x directly:
    FORCE_0X=true python3 -m bots.bot SELL ONDO 25
    FORCE_0X=true python3 -m bots.bot SELL VIRTUAL 25

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
# Uniswap V3 SwapRouter02 — DIFFERENT address per chain!
UNISWAP_V3_ROUTER = {
    "ethereum": "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",  # SwapRouter02
    "base":     "0x2626664c2603336E57B271c5C0b26F421741e481",  # SwapRouter02 (Base-specific!)
    "optimism": "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",  # SwapRouter02
}

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
        "router":   UNISWAP_V3_ROUTER[chain],
        "weth":     WETH_ADDRESS[chain],
    }
    for chain in ("ethereum", "base", "optimism")
}

# ═══════════════════════════════════════════════════════════════════
# Known ERC-20 decimals
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
    # Other EVM tokens
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

# Uniswap V3 SwapRouter02 ABI
# CRITICAL: SwapRouter02 does NOT have `deadline` in the params struct.
# Deadline is passed via multicall(uint256 deadline, bytes[] data).
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
    {
        "name": "multicall",
        "type": "function",
        "inputs": [
            {"name": "deadline", "type": "uint256"},
            {"name": "data",     "type": "bytes[]"},
        ],
        "outputs": [{"name": "results", "type": "bytes[]"}],
        "stateMutability": "payable",
    },
]

# QuoterV2 contract addresses
QUOTER_V2_ADDRESS = {
    "ethereum": "0x61fFE014bA17989E743c5F6cB21bF9697530B21e",
    "base":     "0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a",
    "optimism": "0x61fFE014bA17989E743c5F6cB21bF9697530B21e",
}

QUOTER_V2_ABI = [
    {
        "name": "quoteExactInputSingle",
        "type": "function",
        "inputs": [
            {
                "name": "params",
                "type": "tuple",
                "components": [
                    {"name": "tokenIn",             "type": "address"},
                    {"name": "tokenOut",            "type": "address"},
                    {"name": "amountIn",            "type": "uint256"},
                    {"name": "fee",                 "type": "uint24"},
                    {"name": "sqrtPriceLimitX96",   "type": "uint160"},
                ],
            }
        ],
        "outputs": [
            {"name": "amountOut",               "type": "uint256"},
            {"name": "sqrtPriceX96After",       "type": "uint160"},
            {"name": "initializedTicksCrossed", "type": "uint32"},
            {"name": "gasEstimate",             "type": "uint256"},
        ],
        "stateMutability": "nonpayable",
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
    try:
        w3    = _get_w3(blockchain)
        token = w3.eth.contract(address=_checksum(token_contract), abi=ERC20_ABI)
        return int(token.functions.decimals().call())
    except Exception as e:
        log.warning("Could not fetch decimals for %s on %s: %s — defaulting to 18",
                    token_contract, blockchain, e)
        return 18


def get_decimals(blockchain: str, token_contract: str, symbol: str = "") -> int:
    if symbol and symbol.upper() in ERC20_DECIMALS:
        return ERC20_DECIMALS[symbol.upper()]
    return get_token_decimals_onchain(blockchain, token_contract)


# ═══════════════════════════════════════════════════════════════════
# Balance queries
# ═══════════════════════════════════════════════════════════════════
def get_evm_native_balance(blockchain: str, wallet_address: str) -> float:
    w3  = _get_w3(blockchain)
    wei = w3.eth.get_balance(_checksum(wallet_address))
    return float(w3.from_wei(wei, "ether"))


def get_evm_token_balance(
    blockchain: str,
    wallet_address: str,
    token_contract: str,
    decimals: Optional[int] = None,
) -> float:
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
    Approve spender to spend amount_raw of token if allowance is insufficient.
    Returns tx hash if approval was sent, None if already sufficient.
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
# Uniswap V3 — QuoterV2 (read-only, zero gas)
# ═══════════════════════════════════════════════════════════════════
def quote_best_fee_tier(
    blockchain: str,
    token_in: str,
    token_out: str,
    amount_in_raw: int,
    fee_tiers: list | None = None,
) -> tuple:
    """
    Use QuoterV2.quoteExactInputSingle to find the best Uniswap V3 fee tier.
    Returns (best_fee_tier, best_amount_out_raw).
    If best_amount_out_raw == 0 → no liquid V3 pool exists → evm_swap() will use 0x.
    """
    if fee_tiers is None:
        fee_tiers = UNISWAP_FEE_TIERS

    quoter_addr = QUOTER_V2_ADDRESS.get(blockchain)
    if not quoter_addr:
        log.warning("[%s] No QuoterV2 address — skipping V3 quote", blockchain)
        return (fee_tiers[0], 0)

    w3       = _get_w3(blockchain)
    quoter   = w3.eth.contract(address=_checksum(quoter_addr), abi=QUOTER_V2_ABI)
    best_fee = fee_tiers[0]
    best_out = 0

    for fee in fee_tiers:
        try:
            result     = quoter.functions.quoteExactInputSingle({
                "tokenIn":           _checksum(token_in),
                "tokenOut":          _checksum(token_out),
                "amountIn":          amount_in_raw,
                "fee":               fee,
                "sqrtPriceLimitX96": 0,
            }).call()
            amount_out = result[0]
            log.info("[%s] QuoterV2 fee=%d → amountOut=%d", blockchain, fee, amount_out)
            if amount_out > best_out:
                best_out = amount_out
                best_fee = fee
        except Exception as e:
            log.debug("[%s] QuoterV2 fee=%d quote failed: %s", blockchain, fee, e)

    if best_out == 0:
        log.warning(
            "[%s] QuoterV2: no liquid V3 pool for %s→%s — will use 0x fallback",
            blockchain, token_in[:10], token_out[:10],
        )
    else:
        log.info("[%s] QuoterV2 best: fee=%d amountOut=%d", blockchain, best_fee, best_out)

    return (best_fee, best_out)


# ═══════════════════════════════════════════════════════════════════
# Uniswap V3 — core swap (exactInputSingle via SwapRouter02 multicall)
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
    expected_out_per_in: float = 0.0,
    slippage_bps: int = 50,
    fee_tier: int = DEFAULT_FEE,
    deadline_seconds: int = 300,
) -> str:
    """Execute a Uniswap V3 exactInputSingle swap via SwapRouter02 multicall."""
    cfg      = CHAIN_CONFIG[blockchain]
    w3       = _get_w3(blockchain)
    account  = Account.from_key(private_key)

    token_in_cs  = _checksum(token_in)
    token_out_cs = _checksum(token_out)
    router_cs    = _checksum(cfg["router"])
    chain_id     = cfg["chain_id"]
    amount_raw   = int(amount_in_human * (10 ** token_in_decimals))

    # Compute amountOutMinimum
    if expected_out_per_in > 0:
        expected_out_human = amount_in_human * expected_out_per_in
        slippage_factor    = 1.0 - (slippage_bps / 10_000.0)
        min_out_human      = expected_out_human * slippage_factor
        amount_out_minimum = int(min_out_human * (10 ** token_out_decimals))
        log.info(
            "[%s] amountOutMinimum: expected=%.6f slippage=%.2f%% min=%.6f (raw=%d)",
            blockchain, expected_out_human, slippage_bps / 100,
            min_out_human, amount_out_minimum,
        )
    else:
        amount_out_minimum = 0
        log.warning(
            "[%s] ⚠️  amountOutMinimum=0 (no expected_out_per_in). "
            "Swap may silently return 0 tokens!", blockchain
        )

    log.info(
        "[%s] Uniswap V3 swap %.6f (%d raw) %s → %s | fee=%d slippage=%dbps amountOutMin=%d",
        blockchain, amount_in_human, amount_raw,
        token_in_cs[:10], token_out_cs[:10], fee_tier, slippage_bps, amount_out_minimum,
    )

    # Step 1: Approve router
    approval_hash = ensure_approval(w3, account, token_in_cs, router_cs, amount_raw, chain_id)

    # FIX: Base/Ethereum RPCs don't immediately reflect new nonce after approval.
    # Sleep 2s prevents "Invalid params" nonce collision on next tx.
    if approval_hash is not None:
        log.debug("[%s] Approval mined — waiting 2s for nonce to settle", blockchain)
        time.sleep(2)

    # Step 2: Build swap via multicall(deadline, [calldata])
    router    = w3.eth.contract(address=router_cs, abi=UNISWAP_V3_ROUTER_ABI)
    deadline  = int(time.time()) + deadline_seconds
    nonce     = w3.eth.get_transaction_count(_checksum(account.address))
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

    swap_calldata = router.encode_abi("exactInputSingle", args=[params])

    try:
        gas_estimate = router.functions.multicall(deadline, [swap_calldata]).estimate_gas(
            {"from": _checksum(account.address)}
        )
        gas_limit = int(gas_estimate * 1.2)
    except Exception as e:
        log.warning("[%s] Gas estimation failed for fee=%d (%s) — using fallback 300k",
                    blockchain, fee_tier, e)
        gas_limit = 300_000

    tx = router.functions.multicall(deadline, [swap_calldata]).build_transaction({
        "chainId":  chain_id,
        "from":     _checksum(account.address),
        "nonce":    nonce,
        "gasPrice": gas_price,
        "gas":      gas_limit,
    })

    # Step 3: Sign and send
    signed  = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    log.info("[%s] Uniswap V3 tx sent: %s", blockchain, tx_hash.hex())

    # Step 4: Wait for receipt
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)

    if receipt.status != 1:
        raise RuntimeError(
            f"[{blockchain}] Swap REVERTED: {tx_hash.hex()} (fee_tier={fee_tier})"
        )

    # Step 5: Verify non-zero output via balance diff
    try:
        erc20_out  = w3.eth.contract(address=token_out_cs, abi=ERC20_ABI)
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
                f"but token_out ({token_out_cs[:10]}) balance did not increase. "
                f"(before={bal_before} after={bal_after})"
            )

        log.info(
            "[%s] ✅ Uniswap V3 confirmed: %s | block=%d gas_used=%d | received=%.6f",
            blockchain, tx_hash.hex(), receipt.blockNumber,
            receipt.gasUsed, amount_out_raw / (10 ** token_out_decimals),
        )

    except RuntimeError:
        raise
    except Exception as e:
        log.warning("[%s] Could not verify output balance: %s — trusting receipt.status=1", blockchain, e)
        log.info("[%s] ✅ Uniswap V3 confirmed: %s | block=%d gas_used=%d",
                 blockchain, tx_hash.hex(), receipt.blockNumber, receipt.gasUsed)

    return tx_hash.hex()


# ═══════════════════════════════════════════════════════════════════
# 0x Swap API v2 — fallback when no Uniswap V3 pool exists
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
    Execute a swap via 0x Swap API v2 (AllowanceHolder endpoint).

    Used when:
      - Uniswap V3 QuoterV2 finds no liquid pool (V4 tokens like PNK, ONDO)
      - FORCE_0X=true env var is set (manual testing)

    0x automatically finds best route across V2/V3/V4/multi-hop/cross-protocol.
    Same routing engine used by Coinbase Wallet, MetaMask, Phantom.
    """
    chain_id = CHAIN_IDS.get(blockchain)
    if chain_id is None:
        raise ValueError(f"0x: unsupported blockchain '{blockchain}'")

    api_key = os.getenv("ZEROX_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "ZEROX_API_KEY not set in .env — get free key at dashboard.0x.org/apps"
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

    # Step 1: Get quote from 0x
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

    # Log route 0x chose
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

    if not quote.get("liquidityAvailable", True):
        raise RuntimeError(
            f"[{blockchain}] 0x: no liquidity available for "
            f"{token_in[:10]} → {token_out[:10]}"
        )

    # Log 0x protocol fee if applicable
    try:
        zero_ex_fee = quote.get("fees", {}).get("zeroExFee")
        if zero_ex_fee and int(zero_ex_fee.get("amount", 0)) > 0:
            fee_human = int(zero_ex_fee["amount"]) / (10 ** token_out_decimals)
            log.info("[%s] 0x protocol fee: %.4f token_out", blockchain, fee_human)
    except Exception:
        pass

    # Step 2: Approve 0x allowanceTarget
    spender = None
    issues  = quote.get("issues", {})
    if isinstance(issues.get("allowance"), dict):
        spender = issues["allowance"].get("spender")
    if not spender:
        spender = quote.get("transaction", {}).get("to")
    if not spender:
        raise RuntimeError(
            f"[{blockchain}] 0x: could not determine spender from quote response"
        )

    log.info("[%s] 0x allowanceTarget: %s", blockchain, spender)

    approval_hash = ensure_approval(
        w3, account,
        token_contract=_checksum(token_in),
        spender=_checksum(spender),
        amount_raw=amount_raw,
        chain_id=chain_id,
    )

    if approval_hash is not None:
        log.debug("[%s] 0x approval mined — waiting 2s for nonce to settle", blockchain)
        time.sleep(2)

    # Step 3: Sign and broadcast
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

    # Step 4: Wait for receipt
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)

    if receipt.status != 1:
        raise RuntimeError(f"[{blockchain}] 0x swap REVERTED: {tx_hash.hex()}")

    # Step 5: Verify non-zero output via balance diff
    try:
        time.sleep(2)  # wait for block availability (Base RPC timing)
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

        log.info(
            "[%s] ✅ 0x swap confirmed: %s | block=%d gas_used=%d | received=%.6f",
            blockchain, tx_hash.hex(), receipt.blockNumber,
            receipt.gasUsed, amount_out_raw / (10 ** token_out_decimals),
        )

    except RuntimeError:
        raise
    except Exception as e:
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
# evm_swap() — SINGLE ENTRY POINT for all EVM swaps in bot.py
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
    expected_out_per_in: float = 0.0,  # kept for API compatibility, not used by 0x path
    slippage_bps: int = 50,
    deadline_seconds: int = 300,
) -> str:
    """
    Unified EVM swap entry point for bot.py.
    Drop-in replacement for uniswap_swap_auto_fee() — identical signature.

    Routing logic:
      1. FORCE_0X=true → skip straight to 0x (for manual testing)
      2. Try Uniswap V3 via QuoterV2 (free, no 0x protocol fee)
      3. QuoterV2 returns 0 (no V3 pool) → fall back to 0x automatically

    ── Testing ──────────────────────────────────────────────────────
    Force 0x for any token (bypass Uniswap):
      FORCE_0X=true python3 -m bots.bot SELL ONDO 25
      FORCE_0X=true python3 -m bots.bot SELL VIRTUAL 25
      FORCE_0X=true python3 -m bots.bot BUY LINK 50

    Normal automatic routing (no flag needed in production):
      python3 -m bots.bot SELL VIRTUAL 25   → V3 pool found → Uniswap (free)
      python3 -m bots.bot SELL ONDO 25      → no V3 pool → auto 0x fallback
    ─────────────────────────────────────────────────────────────────
    """
    force_0x = os.getenv("FORCE_0X", "false").lower() == "true"

    if force_0x:
        log.info(
            "[%s] FORCE_0X=true — skipping Uniswap V3, going directly to 0x",
            blockchain,
        )
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

    # ── Try Uniswap V3 first (no protocol fee) ──
    amount_in_raw          = int(amount_in_human * (10 ** token_in_decimals))
    best_fee, best_out_raw = quote_best_fee_tier(
        blockchain, token_in, token_out, amount_in_raw
    )

    if best_out_raw > 0:
        quoted_out_human = best_out_raw / (10 ** token_out_decimals)
        quoted_rate      = quoted_out_human / amount_in_human
        log.info(
            "[%s] V3 pool found: rate=%.6f fee=%d → Uniswap V3 (no 0x fee)",
            blockchain, quoted_rate, best_fee,
        )
        return uniswap_swap(
            blockchain=blockchain,
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

    # ── No V3 pool → fall back to 0x ──
    log.warning(
        "[%s] No Uniswap V3 pool found for %s→%s — falling back to 0x API",
        blockchain, token_in[:10], token_out[:10],
    )
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


# Backwards compatibility alias — nothing breaks if bot.py still imports this name
uniswap_swap_auto_fee = evm_swap


# ═══════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════
def to_smallest_evm(amount: float, decimals: int) -> int:
    """Convert human-readable float to smallest unit integer."""
    return int(amount * (10 ** decimals))


def from_smallest_evm(amount_raw: int, decimals: int) -> float:
    """Convert smallest unit integer to human-readable float."""
    return float(amount_raw) / (10 ** decimals)