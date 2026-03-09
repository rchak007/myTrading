#!/usr/bin/env python3
# bots/bot.py
#
# Multi-chain trading bot: Solana (Jupiter) + Ethereum/Base/Optimism (Uniswap V3)
# Reads bot_registry.json for wallet/asset pairs and asset_registry.json
# for token details (contract addresses, blockchain, yahoo tickers, etc.).
# Runs all bots sequentially in a single process loop.
#
# Blockchain routing:
#   asset.blockchain == "solana"   → Jupiter DEX
#   asset.blockchain == "ethereum" → Uniswap V3 (Ethereum mainnet)
#   asset.blockchain == "base"     → Uniswap V3 (Base)
#   asset.blockchain == "optimism" → Uniswap V3 (Optimism)
#
# KEY FIX (2026-03-07):
#   - EVM swaps now pass expected_out_per_in (derived from current price) so
#     uniswap.py computes a real amountOutMinimum instead of 0.
#   - token_out_decimals now explicitly passed for both BUY and SELL directions.
#   - State is only updated to desired_regime when execute_plan confirms a tx_sig.
#     Previously regime flipped even when the swap silently returned 0 tokens.
#   - USDC decimals per chain now correctly resolved (Base USDC = 6).
#   - TOKEN_DECIMALS expanded to cover all tokens in common registries.

from __future__ import annotations

import json
import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
load_dotenv()

from cryptography.fernet import Fernet

# Solana
from solders.keypair import Keypair
from core.execution.jupiter import (
    WSOL_MINT,
    USDC_MINT,
    WSOL_DECIMALS,
    USDC_DECIMALS,
    get_sol_balance,
    get_spl_token_balance_ui,
    get_quote,
    get_swap_tx,
    sign_and_send_swap,
    to_smallest,
)

# EVM (Ethereum / Base / Optimism)
from core.execution.uniswap import (
    CHAIN_CONFIG,
    ERC20_DECIMALS,
    USDC_ADDRESS,
    get_evm_native_balance,
    get_evm_token_balance,
    get_decimals,
    uniswap_swap_auto_fee,
    to_smallest_evm,
)

from core.indicators import apply_indicators
from core.signals import signal_super_most_adxr

import shutil
from datetime import datetime
from zoneinfo import ZoneInfo


# ═══════════════════════════════════════════════════════════════════
# Global Config (from .env)
# ═══════════════════════════════════════════════════════════════════
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("bot")

# Solana RPC
SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")

BOT_TZ = os.getenv("BOT_TIMEZONE", "UTC")

# Paths
BOT_REGISTRY_PATH    = os.getenv("BOT_REGISTRY_PATH",   "./bot_registry.json")
ASSET_REGISTRY_PATH  = os.getenv("ASSET_REGISTRY_PATH", "./asset_registry.json")
STATE_DIR            = os.getenv("BOT_STATE_DIR",        "./outputs")
TRADE_LOG_DIR        = os.getenv("BOT_TRADE_LOG_DIR",    "./outputs")
STATE_MIRROR_DIR     = os.getenv("BOT_STATE_MIRROR_DIR")        # optional
TRADE_LOG_MIRROR_DIR = os.getenv("BOT_TRADE_LOG_MIRROR_DIR")    # optional
HEARTBEAT_LOG_DIR    = os.getenv("BOT_HEARTBEAT_LOG_DIR",  "./outputs")
HEARTBEAT_MIRROR_DIR = os.getenv("BOT_HEARTBEAT_MIRROR_DIR")    # optional

# Trading behavior
DRY_RUN       = os.getenv("DRY_RUN", "false").lower() == "true"
SLIPPAGE_BPS  = int(os.getenv("SLIPPAGE_BPS", "50"))
SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", "60"))

# Allocation targets
IN_TOKEN_PCT  = float(os.getenv("IN_TOKEN_PCT",  "0.80"))  # BUY/HOLD  → 80% token
OUT_TOKEN_PCT = float(os.getenv("OUT_TOKEN_PCT", "0.20"))  # EXIT      → 20% token

# Safety / dust
USD_TOLERANCE   = float(os.getenv("USD_TOLERANCE",   "5"))
MIN_SWAP_USD    = float(os.getenv("MIN_SWAP_USD",     "10"))
SOL_FEE_RESERVE = float(os.getenv("SOL_FEE_RESERVE", "0.01"))
ETH_FEE_RESERVE = float(os.getenv("ETH_FEE_RESERVE", "0.005"))  # ~$10-15 buffer for gas

# Indicator params (shared across all bots)
ATR_PERIOD   = int(os.getenv("ATR_PERIOD",   "10"))
ATR_MULT     = float(os.getenv("ATR_MULT",   "3.0"))
RSI_PERIOD   = int(os.getenv("RSI_PERIOD",   "14"))
VOL_LOOKBACK = int(os.getenv("VOL_LOOKBACK", "20"))
ADXR_LEN     = int(os.getenv("ADXR_LEN",    "14"))
ADXR_LENX    = int(os.getenv("ADXR_LENX",   "14"))
ADXR_LOW     = float(os.getenv("ADXR_LOW",  "20.0"))
ADXR_EPS     = float(os.getenv("ADXR_EPS",  "1e-6"))

# Lookback per interval
LOOKBACK_MAP = {
    "1h":  30,
    "2h":  60,
    "4h":  90,
    "1d":  365,
    "1wk": 730,
}

# yfinance interval mapping
YF_INTERVAL_MAP = {
    "1h":  "60m",
    "2h":  "2h",
    "4h":  "4h",
    "1d":  "1d",
    "1wk": "1wk",
}

# EVM chains supported (Uniswap V3)
EVM_CHAINS    = {"ethereum", "base", "optimism"}
SOLANA_CHAINS = {"solana"}


# ═══════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════
@dataclass
class AssetInfo:
    """Resolved from asset_registry.json."""
    ticker:              str
    blockchain:          str
    wallet_address:      str    # public address (EVM) — informational
    token_contract:      str    # "" = native SOL; EVM = ERC-20 address
    yahoo_ticker:        str
    stablecoin_contract: str
    decimals:            int = 9

    @property
    def is_native_sol(self) -> bool:
        return self.blockchain == "solana" and self.token_contract == ""

    @property
    def is_evm(self) -> bool:
        return self.blockchain in EVM_CHAINS

    @property
    def is_solana(self) -> bool:
        return self.blockchain in SOLANA_CHAINS

    # ── Solana helpers ──
    @property
    def token_mint(self) -> str:
        """Mint address for Jupiter swaps (Solana only)."""
        return WSOL_MINT if self.is_native_sol else self.token_contract

    @property
    def stable_mint(self) -> str:
        return self.stablecoin_contract


@dataclass
class BotEntry:
    """One row from bot_registry.json, enriched with asset details."""
    name:             str
    wallet_env:       str
    asset_key:        str
    interval:         str
    asset:            AssetInfo = field(default=None)
    confirm_interval: Optional[str] = None

    @property
    def bot_id(self) -> str:
        return f"{self.asset_key}_{self.interval}"


@dataclass
class BotState:
    last_bar_ts: Optional[str] = None
    regime:      str = "OUT"


# ═══════════════════════════════════════════════════════════════════
# Registry loaders
# ═══════════════════════════════════════════════════════════════════
TOKEN_DECIMALS = {
    # ── Solana tokens ──
    "SOL":    9,
    "USDC":   6,
    "PYTH":   6,
    "JUP":    6,
    "BONK":   5,
    "RAY":    6,
    "JTO":    9,
    "WIF":    6,
    "HNT":    8,
    "MOBILE": 6,
    "ORCA":   6,
    "KMNO":   6,
    "JTO":    9,
    "DRIFT":  6,
    "W":      6,
    "ZEUS":   6,
    "NOS":    6,
    "NAVX":   6,
    "ORE":    11,
    "FLUXB":  9,
    "SUAI":   6,
    "LFNTY":  6,
    "AUKI":   6,
    "CETUS":  8,
    "BLUE":   6,
    "ELON":   9,
    # ── EVM tokens (Ethereum / Base / Optimism) ──
    "ETH":      18,
    "WETH":     18,
    "LINK":     18,
    "UNI":      18,
    "AAVE":     18,
    "CRV":      18,
    "ENS":      18,
    "PNK":      18,
    "VIRTUAL":  18,
    "RENDER":   18,
    "WLD":      18,
    "OP":       18,
    "ARB":      18,
    "GMX":      18,
    "GNO":      18,
    "LDO":      18,
    "RPL":      18,
    "PENDLE":   18,
    "ENA":      18,
    "FLUID":    18,
    "ZK":       18,
    "ONDO":     18,
    "AERODROME":18,
    "AOT":      18,
    "HYPE":     8,   # Hyperliquid
}


def load_asset_registry(path: str) -> dict[str, AssetInfo]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    registry = {}
    for key, val in raw.items():
        decimals = TOKEN_DECIMALS.get(
            key.upper(),
            18 if val.get("blockchain") in EVM_CHAINS else 9
        )
        registry[key.upper()] = AssetInfo(
            ticker=val["ticker"],
            blockchain=val["blockchain"],
            wallet_address=val.get("wallet_address", ""),
            token_contract=val.get("token_contract", ""),
            yahoo_ticker=val["yahoo_ticker"],
            stablecoin_contract=val.get(
                "stablecoin_contract",
                # Defaults: USDC on each chain
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"   # Solana USDC
                if val.get("blockchain") == "solana"
                else USDC_ADDRESS.get(val.get("blockchain", ""), 
                     "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")  # ETH USDC default
            ),
            decimals=decimals,
        )
    return registry


def load_bot_registry(
    path: str,
    asset_reg: dict[str, AssetInfo],
) -> list[BotEntry]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    bots = []
    for entry in raw:
        asset_key = entry["asset"].upper()
        if asset_key not in asset_reg:
            log.warning("Asset '%s' not in asset_registry — skipping bot '%s'",
                        asset_key, entry.get("name"))
            continue

        bot = BotEntry(
            name=entry["name"],
            wallet_env=entry["wallet_env"],
            asset_key=asset_key,
            interval=entry.get("interval", "1h"),
            asset=asset_reg[asset_key],
            confirm_interval=entry.get("confirm_interval"),
        )
        bots.append(bot)

    return bots


# ═══════════════════════════════════════════════════════════════════
# Fernet key + keypair loading (Solana + EVM)
# ═══════════════════════════════════════════════════════════════════
_fernet_cache: Fernet | None = None


def _get_fernet() -> Fernet:
    global _fernet_cache
    if _fernet_cache is None:
        key_path = os.getenv("BOT_FERNET_KEY_PATH", "/etc/myTrading/bot.key")
        if not os.path.exists(key_path):
            key_path = os.getenv("JUPBOT_FERNET_KEY_PATH", "/etc/myTrading/jupbot.key")
        with open(key_path, "rb") as f:
            _fernet_cache = Fernet(f.read().strip())
    return _fernet_cache


def load_keypair_solana(wallet_env: str) -> Keypair:
    """Decrypt Solana private key (base58) from env var."""
    enc = (os.getenv(wallet_env) or "").strip()
    if not enc:
        raise RuntimeError(f"Missing {wallet_env} in .env")
    pk_b58 = _get_fernet().decrypt(enc.encode("utf-8")).decode("utf-8").strip()
    return Keypair.from_base58_string(pk_b58)


def load_private_key_evm(wallet_env: str) -> str:
    """
    Decrypt EVM private key (hex string, with or without 0x prefix).
    Returns the raw hex string — eth_account.Account.from_key() accepts either format.
    """
    enc = (os.getenv(wallet_env) or "").strip()
    if not enc:
        raise RuntimeError(f"Missing {wallet_env} in .env")
    pk_hex = _get_fernet().decrypt(enc.encode("utf-8")).decode("utf-8").strip()
    return pk_hex


def load_wallet(wallet_env: str, blockchain: str):
    """
    Load the correct wallet type based on blockchain.
    Returns:
        Keypair           for Solana
        str (hex privkey) for EVM chains
    """
    if blockchain in SOLANA_CHAINS:
        return load_keypair_solana(wallet_env)
    elif blockchain in EVM_CHAINS:
        return load_private_key_evm(wallet_env)
    else:
        raise ValueError(f"Unknown blockchain '{blockchain}' for wallet loading")


# ═══════════════════════════════════════════════════════════════════
# State management
# ═══════════════════════════════════════════════════════════════════
def _state_path(bot_id: str) -> str:
    return os.path.join(STATE_DIR, f"bot_state_{bot_id}.json")


def _load_state(bot_id: str) -> BotState:
    # Also try legacy jupbot_state_ prefix for backwards compatibility
    legacy = os.path.join(STATE_DIR, f"jupbot_state_{bot_id}.json")
    for path in [_state_path(bot_id), legacy]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return BotState(
                last_bar_ts=data.get("last_bar_ts"),
                regime=data.get("regime", "OUT"),
            )
        except Exception:
            pass
    return BotState()


def _save_state(bot_id: str, st: BotState) -> None:
    path = _state_path(bot_id)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload = {"last_bar_ts": st.last_bar_ts, "regime": st.regime}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if STATE_MIRROR_DIR:
        try:
            mirror = os.path.join(STATE_MIRROR_DIR, f"bot_state_{bot_id}.json")
            os.makedirs(os.path.dirname(mirror), exist_ok=True)
            shutil.copyfile(path, mirror)
        except Exception as e:
            log.warning("[%s] State mirror failed: %s", bot_id, e)


# ═══════════════════════════════════════════════════════════════════
# Market data
# ═══════════════════════════════════════════════════════════════════
def fetch_df(yahoo_ticker: str, interval: str) -> pd.DataFrame:
    lookback    = LOOKBACK_MAP.get(interval, 30)
    yf_interval = YF_INTERVAL_MAP.get(interval, "60m")

    df = yf.download(
        yahoo_ticker,
        period=f"{lookback}d",
        interval=yf_interval,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No yfinance data for {yahoo_ticker} {yf_interval}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df[["High", "Low", "Close", "Volume"]].dropna()
    if len(df) < 100:
        raise RuntimeError(
            f"Not enough candles for {yahoo_ticker} {yf_interval}: {len(df)}"
        )

    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        pass

    return df


# ═══════════════════════════════════════════════════════════════════
# Signal logic
# ═══════════════════════════════════════════════════════════════════
def desired_regime_from_final_signal(final_sig: str) -> str:
    return "IN" if final_sig in ("BUY", "HOLD") else "OUT"


# ═══════════════════════════════════════════════════════════════════
# Portfolio — routes to Solana or EVM balance functions
# ═══════════════════════════════════════════════════════════════════
def get_portfolio_solana(
    pubkey: str,
    asset: AssetInfo,
    token_price: float,
) -> dict:
    """Get portfolio snapshot for a Solana wallet."""
    if asset.is_native_sol:
        token_bal      = get_sol_balance(SOLANA_RPC_URL, pubkey)
        tradable_token = max(0.0, token_bal - SOL_FEE_RESERVE)
    else:
        token_bal      = get_spl_token_balance_ui(SOLANA_RPC_URL, pubkey, asset.token_contract)
        tradable_token = token_bal
        sol_bal        = get_sol_balance(SOLANA_RPC_URL, pubkey)
        if sol_bal < SOL_FEE_RESERVE:
            log.warning("Low SOL for fees: %.4f SOL (need %.4f)", sol_bal, SOL_FEE_RESERVE)

    stable_bal = get_spl_token_balance_ui(SOLANA_RPC_URL, pubkey, asset.stable_mint)
    return _build_portfolio_dict(token_bal, tradable_token, stable_bal, token_price)


def get_portfolio_evm(
    wallet_address: str,
    asset: AssetInfo,
    token_price: float,
) -> dict:
    """Get portfolio snapshot for an EVM wallet."""
    blockchain = asset.blockchain

    # Token balance
    if asset.token_contract:
        token_bal = get_evm_token_balance(
            blockchain, wallet_address, asset.token_contract, asset.decimals
        )
    else:
        token_bal = get_evm_native_balance(blockchain, wallet_address)

    # Gas reserve check
    eth_bal = get_evm_native_balance(blockchain, wallet_address)
    if eth_bal < ETH_FEE_RESERVE:
        log.warning(
            "[%s] Low ETH for gas: %.6f ETH (need %.6f)",
            blockchain, eth_bal, ETH_FEE_RESERVE,
        )
    tradable_token = token_bal

    # Stablecoin balance (USDC on EVM — always 6 decimals)
    stable_bal = get_evm_token_balance(
        blockchain, wallet_address, asset.stablecoin_contract,
        decimals=ERC20_DECIMALS.get("USDC", 6)
    )

    return _build_portfolio_dict(token_bal, tradable_token, stable_bal, token_price)


def get_portfolio(
    wallet,
    asset: AssetInfo,
    token_price: float,
) -> dict:
    """Dispatch to Solana or EVM portfolio function."""
    if asset.is_solana:
        pubkey = str(wallet.pubkey())
        return get_portfolio_solana(pubkey, asset, token_price)
    elif asset.is_evm:
        return get_portfolio_evm(asset.wallet_address, asset, token_price)
    else:
        raise ValueError(f"Unknown blockchain: {asset.blockchain}")


def _build_portfolio_dict(
    token_bal: float,
    tradable_token: float,
    stable_bal: float,
    token_price: float,
) -> dict:
    token_val  = token_bal * token_price
    stable_val = stable_bal
    total      = token_val + stable_val
    token_pct  = (token_val / total) if total > 0 else 0.0
    return {
        "token_bal":      token_bal,
        "tradable_token": tradable_token,
        "stable_bal":     stable_bal,
        "token_val":      token_val,
        "stable_val":     stable_val,
        "total":          total,
        "token_pct":      token_pct,
    }


# ═══════════════════════════════════════════════════════════════════
# Rebalance planning
# ═══════════════════════════════════════════════════════════════════
def rebalance_plan(
    port: dict,
    token_price: float,
    target_token_pct: float,
) -> dict:
    total = port["total"]
    if total <= 0:
        return {"action": "NONE"}

    desired_token_val = total * target_token_pct
    diff = desired_token_val - port["token_val"]

    if abs(diff) < max(USD_TOLERANCE, MIN_SWAP_USD):
        return {"action": "NONE", "usd_diff": diff}

    if diff > 0:
        spend = min(port["stable_bal"], diff)
        if spend < MIN_SWAP_USD:
            return {"action": "NONE", "usd_diff": diff}
        return {"action": "BUY_TOKEN", "stable_amount": spend, "usd_diff": diff}

    need_sell_usd = min(port["token_val"] - desired_token_val, port["token_val"])
    sell_token    = min(port["tradable_token"], need_sell_usd / token_price)
    if sell_token * token_price < MIN_SWAP_USD:
        return {"action": "NONE", "usd_diff": diff}
    return {"action": "SELL_TOKEN", "token_amount": sell_token, "usd_diff": diff}


# ═══════════════════════════════════════════════════════════════════
# Execution — routes to Jupiter (Solana) or Uniswap V3 (EVM)
# ═══════════════════════════════════════════════════════════════════
def execute_plan(
    *,
    wallet,
    plan: dict,
    asset: AssetInfo,
    token_price: float,     # FIX: needed for amountOutMinimum calculation
) -> dict | None:

    if plan["action"] == "NONE":
        log.info("Rebalance: NONE (within tolerance)")
        return None

    if DRY_RUN:
        log.warning("DRY_RUN: would execute plan=%s", plan)
        return None

    if asset.is_solana:
        return _execute_plan_solana(wallet=wallet, plan=plan, asset=asset)
    elif asset.is_evm:
        return _execute_plan_evm(
            private_key=wallet, plan=plan, asset=asset, token_price=token_price
        )
    else:
        raise ValueError(f"Unknown blockchain: {asset.blockchain}")


def _execute_plan_solana(*, wallet: Keypair, plan: dict, asset: AssetInfo) -> dict | None:
    """Execute a swap on Solana via Jupiter."""
    pubkey      = str(wallet.pubkey())
    token_mint  = asset.token_mint
    stable_mint = asset.stable_mint
    ticker      = asset.ticker

    if plan["action"] == "BUY_TOKEN":
        stable_amt = float(plan["stable_amount"])
        amt_small  = to_smallest(stable_amt, USDC_DECIMALS)
        quote      = get_quote(stable_mint, token_mint, amt_small, SLIPPAGE_BPS)
        swap       = get_swap_tx(quote, pubkey)
        tx         = swap.get("swapTransaction")
        if not tx:
            raise RuntimeError(f"Jupiter swap missing swapTransaction: {swap}")
        sig = sign_and_send_swap(rpc_url=SOLANA_RPC_URL, swap_tx_b64=tx, keypair=wallet)
        log.info("✅ SOL BUY %s: spent USDC=%.2f sig=%s", ticker, stable_amt, sig)
        return {"action": f"BUY_{ticker}", "amount": stable_amt, "amount_ccy": "USDC", "tx_sig": sig}

    if plan["action"] == "SELL_TOKEN":
        token_amt = float(plan["token_amount"])
        amt_small = to_smallest(token_amt, asset.decimals)
        quote     = get_quote(token_mint, stable_mint, amt_small, SLIPPAGE_BPS)
        swap      = get_swap_tx(quote, pubkey)
        tx        = swap.get("swapTransaction")
        if not tx:
            raise RuntimeError(f"Jupiter swap missing swapTransaction: {swap}")
        sig = sign_and_send_swap(rpc_url=SOLANA_RPC_URL, swap_tx_b64=tx, keypair=wallet)
        log.info("✅ SOL SELL %s: sold %.6f sig=%s", ticker, token_amt, sig)
        return {"action": f"SELL_{ticker}", "amount": token_amt, "amount_ccy": ticker, "tx_sig": sig}

    raise RuntimeError(f"Unknown plan action: {plan}")


def _execute_plan_evm(
    *,
    private_key: str,
    plan: dict,
    asset: AssetInfo,
    token_price: float,     # FIX: used to compute amountOutMinimum
) -> dict | None:
    """Execute a swap on an EVM chain via Uniswap V3."""
    blockchain      = asset.blockchain
    token_contract  = asset.token_contract
    stable_contract = asset.stablecoin_contract
    ticker          = asset.ticker
    usdc_decimals   = ERC20_DECIMALS.get("USDC", 6)
    token_decimals  = asset.decimals

    if plan["action"] == "BUY_TOKEN":
        stable_amt = float(plan["stable_amount"])

        # FIX: expected output = stable_amt / token_price tokens
        # e.g. spending $301 USDC to buy VIRTUAL at $0.68 → expect 301/0.68 ≈ 443 VIRTUAL
        expected_out_per_in = (1.0 / token_price) if token_price > 0 else 0.0

        tx_hash = uniswap_swap_auto_fee(
            blockchain=blockchain,
            private_key=private_key,
            token_in=stable_contract,
            token_out=token_contract,
            amount_in_human=stable_amt,
            token_in_decimals=usdc_decimals,
            token_out_decimals=token_decimals,      # FIX: explicit
            expected_out_per_in=expected_out_per_in, # FIX: real floor
            slippage_bps=SLIPPAGE_BPS,
        )
        log.info("✅ EVM BUY %s [%s]: spent USDC=%.2f tx=%s",
                 ticker, blockchain, stable_amt, tx_hash)
        return {
            "action": f"BUY_{ticker}",
            "amount": stable_amt,
            "amount_ccy": "USDC",
            "tx_sig": tx_hash,
        }

    if plan["action"] == "SELL_TOKEN":
        token_amt = float(plan["token_amount"])

        # FIX: expected output = token_amt * token_price USDC
        # e.g. selling 1866 VIRTUAL at $0.685 → expect 1866 * 0.685 ≈ $1,278 USDC
        expected_out_per_in = token_price  # each token → token_price USD

        tx_hash = uniswap_swap_auto_fee(
            blockchain=blockchain,
            private_key=private_key,
            token_in=token_contract,
            token_out=stable_contract,
            amount_in_human=token_amt,
            token_in_decimals=token_decimals,        # FIX: explicit
            token_out_decimals=usdc_decimals,        # FIX: USDC out
            expected_out_per_in=expected_out_per_in, # FIX: real floor
            slippage_bps=SLIPPAGE_BPS,
        )
        log.info("✅ EVM SELL %s [%s]: sold %.6f tx=%s",
                 ticker, blockchain, token_amt, tx_hash)
        return {
            "action": f"SELL_{ticker}",
            "amount": token_amt,
            "amount_ccy": ticker,
            "tx_sig": tx_hash,
        }

    raise RuntimeError(f"Unknown plan action: {plan}")


# ═══════════════════════════════════════════════════════════════════
# Trade logging
# ═══════════════════════════════════════════════════════════════════
def _trade_log_path(bot_id: str) -> str:
    return os.path.join(TRADE_LOG_DIR, f"bot_trades_{bot_id}.csv")


def log_trade(
    *,
    bot_id: str,
    bot_name: str,
    action: str,
    regime_from: str,
    regime_to: str,
    price: float,
    amount: float,
    amount_ccy: str,
    tx_sig: str | None,
    dry_run: bool,
    blockchain: str = "solana",
):
    path = _trade_log_path(bot_id)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    file_exists = os.path.exists(path)

    with open(path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write(
                "timestamp,bot_name,blockchain,action,regime_from,regime_to,"
                "price,amount,amount_ccy,tx_sig,dry_run\n"
            )

        pst = ZoneInfo("America/Los_Angeles")
        ts  = datetime.now(pst).strftime("%Y-%m-%d %H:%M:%S")

        f.write(
            f"{ts},{bot_name},{blockchain},{action},"
            f"{regime_from},{regime_to},"
            f"{price:.4f},{amount:.6f},{amount_ccy},"
            f"{tx_sig or ''},{dry_run}\n"
        )

    if TRADE_LOG_MIRROR_DIR:
        try:
            mirror = os.path.join(TRADE_LOG_MIRROR_DIR, f"bot_trades_{bot_id}.csv")
            os.makedirs(os.path.dirname(mirror), exist_ok=True)
            shutil.copyfile(path, mirror)
        except Exception as e:
            log.warning("[%s] Trade log mirror failed: %s", bot_id, e)


# ═══════════════════════════════════════════════════════════════════
# Higher-timeframe confirmation filter
# ═══════════════════════════════════════════════════════════════════
def get_htf_confirmation(yahoo_ticker: str, confirm_interval: str) -> dict:
    df  = fetch_df(yahoo_ticker, confirm_interval)
    ind = apply_indicators(
        df,
        atr_period=ATR_PERIOD,
        atr_multiplier=ATR_MULT,
        rsi_period=RSI_PERIOD,
        vol_lookback=VOL_LOOKBACK,
        adxr_len=ADXR_LEN,
        adxr_lenx=ADXR_LENX,
        adxr_low_threshold=ADXR_LOW,
        adxr_flat_eps=ADXR_EPS,
    )
    last      = ind.iloc[-1]
    htf_st    = str(last.get("Supertrend_Signal", "SELL"))
    htf_most  = str(last.get("MOST_Signal",       "SELL"))
    htf_adxr  = str(last.get("ADXR_State",        "FLAT"))
    htf_final = signal_super_most_adxr(htf_st, htf_most, htf_adxr)

    return {
        "supertrend":   htf_st,
        "most":         htf_most,
        "adxr_state":   htf_adxr,
        "final_signal": htf_final,
        "allows_buy":   htf_final in ("BUY", "HOLD"),
    }


# ═══════════════════════════════════════════════════════════════════
# Error log — written to file and mirrored to jobMyTrading repo
# ═══════════════════════════════════════════════════════════════════
ERROR_LOG_DIR    = os.getenv("BOT_ERROR_LOG_DIR",    HEARTBEAT_LOG_DIR)
ERROR_MIRROR_DIR = os.getenv("BOT_ERROR_MIRROR_DIR", HEARTBEAT_MIRROR_DIR)


def _error_log_path(bot_id: str) -> str:
    return os.path.join(ERROR_LOG_DIR, f"bot_errors_{bot_id}.log")


def write_error_log(
    *,
    bot_id: str,
    bot_name: str,
    blockchain: str,
    context: str,       # short description of what was being attempted
    error: str,         # str(exception)
    bar_ts: str = "",
    price: float = 0.0,
):
    """
    Append one error line to bot_errors_<bot_id>.log and mirror to jobMyTrading.
    Format matches heartbeat log so it can be read alongside it.
    """
    pst = ZoneInfo("America/Los_Angeles")
    now = datetime.now(pst)
    today_str = now.strftime("%Y-%m-%d")
    ts        = now.strftime("%Y-%m-%d %H:%M:%S")

    path = _error_log_path(bot_id)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("date,timestamp,bot_name,blockchain,bar_ts,price,context,error\n")
        # Sanitise error string — remove newlines/commas so CSV stays valid
        safe_error = error.replace("\n", " | ").replace(",", ";")
        f.write(f"{today_str},{ts},{bot_name},{blockchain},{bar_ts},{price:.4f},"
                f"{context},{safe_error}\n")

    if ERROR_MIRROR_DIR:
        try:
            mirror = os.path.join(ERROR_MIRROR_DIR, f"bot_errors_{bot_id}.log")
            os.makedirs(os.path.dirname(mirror), exist_ok=True)
            shutil.copyfile(path, mirror)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# Heartbeat log
# ═══════════════════════════════════════════════════════════════════
def _heartbeat_log_path(bot_id: str) -> str:
    return os.path.join(HEARTBEAT_LOG_DIR, f"bot_heartbeat_{bot_id}.log")


def write_heartbeat(
    *,
    bot_id: str,
    bot_name: str,
    blockchain: str,
    bar_ts: str,
    price: float,
    st_sig: str,
    most_sig: str,
    adxr_state: str,
    final_sig: str,
    regime: str,
    desired_regime: str,
    action: str,
    htf_info: dict | None = None,
    confirm_interval: str | None = None,
):
    pst       = ZoneInfo("America/Los_Angeles")
    now_pst   = datetime.now(pst)
    today_str = now_pst.strftime("%Y-%m-%d")
    ts        = now_pst.strftime("%Y-%m-%d %H:%M:%S")

    path = _heartbeat_log_path(bot_id)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    should_overwrite = True
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                f.readline()  # header
                second_line = f.readline().strip()
                if second_line and second_line.startswith(today_str):
                    should_overwrite = False
        except Exception:
            should_overwrite = True

    mode = "w" if should_overwrite else "a"

    htf_str = ""
    if htf_info and confirm_interval:
        htf_str = (
            f"{confirm_interval}:ST={htf_info['supertrend']}|"
            f"MOST={htf_info['most']}|ADXR={htf_info['adxr_state']}|"
            f"FINAL={htf_info['final_signal']}|allows_buy={htf_info['allows_buy']}"
        )

    with open(path, mode, encoding="utf-8") as f:
        if should_overwrite:
            f.write(
                "date,time,bot_name,blockchain,bar_ts,price,ST_sig,MOST_sig,"
                "ADXR_state,final_signal,regime,desired_regime,action,htf_filter\n"
            )
        f.write(
            f"{today_str},{ts},{bot_name},{blockchain},{bar_ts},{price:.4f},"
            f"{st_sig},{most_sig},{adxr_state},{final_sig},"
            f"{regime},{desired_regime},{action},{htf_str}\n"
        )

    if HEARTBEAT_MIRROR_DIR:
        try:
            mirror = os.path.join(HEARTBEAT_MIRROR_DIR, f"bot_heartbeat_{bot_id}.log")
            os.makedirs(os.path.dirname(mirror), exist_ok=True)
            shutil.copyfile(path, mirror)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# Per-bot tick
# ═══════════════════════════════════════════════════════════════════
def tick_bot(bot: BotEntry, wallet, st: BotState) -> BotState:
    """
    Process one tick for a single bot entry.
    wallet: Keypair for Solana, hex str for EVM.
    Returns updated BotState.
    """
    asset = bot.asset
    blog  = logging.getLogger(f"bot.{bot.bot_id}")

    # ── Fetch market data & compute signals ──
    df  = fetch_df(asset.yahoo_ticker, bot.interval)
    ind = apply_indicators(
        df,
        atr_period=ATR_PERIOD,
        atr_multiplier=ATR_MULT,
        rsi_period=RSI_PERIOD,
        vol_lookback=VOL_LOOKBACK,
        adxr_len=ADXR_LEN,
        adxr_lenx=ADXR_LENX,
        adxr_low_threshold=ADXR_LOW,
        adxr_flat_eps=ADXR_EPS,
    )

    last = ind.iloc[-1]

    # ── Bar timestamp ──
    tz = (
        ZoneInfo("America/Los_Angeles")
        if BOT_TZ.upper() == "PST"
        else ZoneInfo("UTC")
    )
    bar_ts = (
        pd.Timestamp(last.name)
        .tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
        .tz_convert(tz)
        .strftime("%Y-%m-%d %H:%M:%S")
    )

    # ── Signals ──
    st_sig     = str(last.get("Supertrend_Signal", "SELL"))
    most_sig   = str(last.get("MOST_Signal",       "SELL"))
    adxr_state = str(last.get("ADXR_State",        "FLAT"))

    final_sig      = signal_super_most_adxr(st_sig, most_sig, adxr_state)
    desired_regime = desired_regime_from_final_signal(final_sig)
    price          = float(last["Close"])

    # ── Higher-timeframe confirmation (if configured) ──
    htf_info = None
    if bot.confirm_interval and desired_regime == "IN":
        try:
            htf_info = get_htf_confirmation(asset.yahoo_ticker, bot.confirm_interval)
            blog.info(
                "HTF %s: ST=%s MOST=%s ADXR=%s => FINAL=%s | allows_buy=%s",
                bot.confirm_interval,
                htf_info["supertrend"], htf_info["most"],
                htf_info["adxr_state"], htf_info["final_signal"],
                htf_info["allows_buy"],
            )
            if not htf_info["allows_buy"]:
                blog.info("HTF %s says %s — BLOCKING BUY, staying OUT",
                          bot.confirm_interval, htf_info["final_signal"])
                desired_regime = "OUT"
        except Exception as e:
            blog.warning("HTF %s fetch failed: %s — proceeding with primary signal only",
                         bot.confirm_interval, e)

    # ── Logging ──
    rsi_val  = last.get("RSI",        None)
    adxr_val = last.get("ADXR",       None)
    st_val   = last.get("Supertrend", None)
    most_val = last.get("MOST",       None)
    atr_val  = last.get("ATR",        None)

    rsi_str  = f"{float(rsi_val):.2f}"  if rsi_val  is not None and pd.notna(rsi_val)  else "N/A"
    adxr_str = f"{float(adxr_val):.2f}" if adxr_val is not None and pd.notna(adxr_val) else "N/A"
    st_str   = f"{float(st_val):.4f}"   if st_val   is not None and pd.notna(st_val)   else "N/A"
    most_str = f"{float(most_val):.4f}" if most_val is not None and pd.notna(most_val) else "N/A"
    atr_str  = f"{float(atr_val):.4f}"  if atr_val  is not None and pd.notna(atr_val)  else "N/A"

    blog.info("───── %s [%s] (%s) ─────", asset.ticker, asset.blockchain, bot.interval)
    blog.info("Bar=%s | Close=%.4f | RSI=%s | ATR=%s | ADXR=%s",
              bar_ts, price, rsi_str, atr_str, adxr_str)
    blog.info("Supertrend=%s (sig=%s) | MOST=%s (sig=%s) | ADXR_State=%s",
              st_str, st_sig, most_str, most_sig, adxr_state)
    blog.info("FINAL_SIGNAL=%s | regime: current=%s desired=%s | new_bar=%s",
              final_sig, st.regime, desired_regime,
              "YES" if st.last_bar_ts != bar_ts else "NO")

    # ── Heartbeat ──
    action_str = "NONE"
    if st.last_bar_ts != bar_ts and desired_regime != st.regime:
        action_str = f"REBALANCE_{desired_regime}"

    write_heartbeat(
        bot_id=bot.bot_id,
        bot_name=bot.name,
        blockchain=asset.blockchain,
        bar_ts=bar_ts,
        price=price,
        st_sig=st_sig,
        most_sig=most_sig,
        adxr_state=adxr_state,
        final_sig=final_sig,
        regime=st.regime,
        desired_regime=desired_regime,
        action=action_str,
        htf_info=htf_info,
        confirm_interval=bot.confirm_interval,
    )

    # ── Helper: execute a rebalance to target_pct and update regime if successful ──
    def _do_rebalance(target_pct: float, reason: str) -> None:
        port = get_portfolio(wallet, asset, price)
        blog.info(
            "Portfolio: total=$%.2f %s=%.4f (tradable=%.4f) stable=%.2f token_pct=%.1f%%",
            port["total"], asset.ticker, port["token_bal"],
            port["tradable_token"], port["stable_bal"],
            port["token_pct"] * 100.0,
        )

        plan = rebalance_plan(port, price, target_pct)
        blog.info("Plan: %s", plan)

        exec_result = None
        swap_error  = None
        try:
            exec_result = execute_plan(
                wallet=wallet,
                plan=plan,
                asset=asset,
                token_price=price,
            )
        except Exception as exc:
            swap_error = exc
            blog.error("❌ Swap failed for %s [%s]: %s", asset.ticker, asset.blockchain, exc)
            write_error_log(
                bot_id=bot.bot_id,
                bot_name=bot.name,
                blockchain=asset.blockchain,
                context=f"execute_plan:{plan.get('action','?')}:{reason}",
                error=str(exc),
                bar_ts=bar_ts,
                price=price,
            )

        if exec_result:
            log_trade(
                bot_id=bot.bot_id,
                bot_name=bot.name,
                action=exec_result["action"],
                regime_from=st.regime,
                regime_to=desired_regime,
                price=price,
                amount=exec_result["amount"],
                amount_ccy=exec_result["amount_ccy"],
                tx_sig=exec_result["tx_sig"],
                dry_run=DRY_RUN,
                blockchain=asset.blockchain,
            )

        if plan.get("action") != "NONE":
            if exec_result and exec_result.get("tx_sig"):
                st.regime = desired_regime
                blog.info("✅ Regime updated to %s", desired_regime)
            elif swap_error:
                blog.warning(
                    "⚠️  Swap ERRORED (%s) — regime stays %s. "
                    "Will NOT retry this bar (last_bar_ts updated).",
                    swap_error, st.regime,
                )
            else:
                blog.warning(
                    "⚠️  Plan was %s but no tx_sig returned — regime stays %s.",
                    plan["action"], st.regime,
                )

    # ── Skip same bar if regime aligned AND wallet confirms it ──
    if st.last_bar_ts == bar_ts and desired_regime == st.regime:
        try:
            port = get_portfolio(wallet, asset, price)
            token_pct = port["token_pct"]
            if st.regime == "OUT" and token_pct > 0.5:
                blog.warning(
                    "⚠️  Drift: regime=OUT but token_pct=%.1f%% — forcing corrective SELL.",
                    token_pct * 100,
                )
                _do_rebalance(OUT_TOKEN_PCT, reason="drift_correction")
            elif st.regime == "IN" and token_pct < 0.5:
                blog.warning(
                    "⚠️  Drift: regime=IN but token_pct=%.1f%% — forcing corrective BUY.",
                    token_pct * 100,
                )
                _do_rebalance(IN_TOKEN_PCT, reason="drift_correction")
            else:
                blog.info("Same bar + regime aligned — skipping trade logic.")
        except Exception as e:
            blog.warning("Drift check failed: %s — skipping.", e)
        st.last_bar_ts = bar_ts
        _save_state(bot.bot_id, st)
        return st

    # ── Trade on regime flip ──
    if desired_regime != st.regime:
        target_pct = IN_TOKEN_PCT if desired_regime == "IN" else OUT_TOKEN_PCT
        blog.info("🔄 REGIME FLIP %s → %s. Rebalancing to %s%%=%.0f%%",
                  st.regime, desired_regime, asset.ticker, target_pct * 100)
        _do_rebalance(target_pct, reason="regime_flip")

    else:
        blog.info("No regime change — holding %s (%s).", st.regime, asset.ticker)

    st.last_bar_ts = bar_ts
    _save_state(bot.bot_id, st)
    return st


# ═══════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 60)
    log.info("bot.py multi-chain starting  |  dry_run=%s", DRY_RUN)
    log.info("=" * 60)

    asset_reg = load_asset_registry(ASSET_REGISTRY_PATH)
    log.info("Loaded %d assets from %s", len(asset_reg), ASSET_REGISTRY_PATH)

    bots = load_bot_registry(BOT_REGISTRY_PATH, asset_reg)
    if not bots:
        log.error("No valid bot entries in %s — exiting.", BOT_REGISTRY_PATH)
        return
    log.info("Loaded %d bot(s) from %s", len(bots), BOT_REGISTRY_PATH)

    # Log chain breakdown
    chains: dict[str, int] = {}
    for b in bots:
        chains[b.asset.blockchain] = chains.get(b.asset.blockchain, 0) + 1
    for chain, count in chains.items():
        log.info("  Chain: %-12s → %d bot(s)", chain, count)

    # ── Load wallets once at startup ──
    wallets: dict[tuple[str, str], object] = {}
    for bot in bots:
        key = (bot.wallet_env, bot.asset.blockchain)
        if key not in wallets:
            w = load_wallet(bot.wallet_env, bot.asset.blockchain)
            wallets[key] = w
            if bot.asset.is_solana:
                log.info("  [%s | solana] wallet=%s pubkey=%s",
                         bot.name, bot.wallet_env, str(w.pubkey()))
            else:
                log.info("  [%s | %s] wallet=%s (EVM key loaded)",
                         bot.name, bot.asset.blockchain, bot.wallet_env)

    # ── Load per-bot state ──
    states: dict[str, BotState] = {}
    for bot in bots:
        states[bot.bot_id] = _load_state(bot.bot_id)
        log.info("  [%s] state: regime=%s last_bar=%s",
                 bot.name, states[bot.bot_id].regime, states[bot.bot_id].last_bar_ts)

    log.info("Entering main loop (sleep=%ds)...", SLEEP_SECONDS)
    log.info("=" * 60)

    while True:
        for bot in bots:
            try:
                key    = (bot.wallet_env, bot.asset.blockchain)
                wallet = wallets[key]
                st     = states[bot.bot_id]
                states[bot.bot_id] = tick_bot(bot, wallet, st)
            except Exception as e:
                log.exception("[%s] Unhandled error in tick_bot: %s", bot.name, e)
                try:
                    write_error_log(
                        bot_id=bot.bot_id,
                        bot_name=bot.name,
                        blockchain=bot.asset.blockchain,
                        context="tick_bot:unhandled_exception",
                        error=str(e),
                    )
                except Exception:
                    pass  # never let logging crash the loop
            time.sleep(10)

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()