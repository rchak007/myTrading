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
    ultra_get_order,
    ultra_sign_and_execute,
    to_smallest,
)

# EVM (Ethereum / Base / Optimism)
from core.execution.uniswap import (
    CHAIN_CONFIG,
    ERC20_DECIMALS,
    get_evm_native_balance,
    get_evm_token_balance,
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
BOT_REGISTRY_PATH   = os.getenv("BOT_REGISTRY_PATH",   "./bot_registry.json")
ASSET_REGISTRY_PATH = os.getenv("ASSET_REGISTRY_PATH", "./asset_registry.json")
STATE_DIR            = os.getenv("BOT_STATE_DIR",       "./outputs")
TRADE_LOG_DIR        = os.getenv("BOT_TRADE_LOG_DIR",   "./outputs")
STATE_MIRROR_DIR     = os.getenv("BOT_STATE_MIRROR_DIR")       # optional
TRADE_LOG_MIRROR_DIR = os.getenv("BOT_TRADE_LOG_MIRROR_DIR")   # optional
HEARTBEAT_LOG_DIR    = os.getenv("BOT_HEARTBEAT_LOG_DIR",  "./outputs")
HEARTBEAT_MIRROR_DIR = os.getenv("BOT_HEARTBEAT_MIRROR_DIR")   # optional

# Trading behavior
DRY_RUN      = os.getenv("DRY_RUN", "false").lower() == "true"
SLIPPAGE_BPS = int(os.getenv("SLIPPAGE_BPS", "50"))
SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", "60"))

# Allocation targets
IN_TOKEN_PCT  = float(os.getenv("IN_TOKEN_PCT",  "0.80"))  # BUY/HOLD
OUT_TOKEN_PCT = float(os.getenv("OUT_TOKEN_PCT", "0.20"))  # EXIT/STANDDOWN

# Safety / dust
USD_TOLERANCE   = float(os.getenv("USD_TOLERANCE",   "5"))
MIN_SWAP_USD    = float(os.getenv("MIN_SWAP_USD",     "10"))
SOL_FEE_RESERVE = float(os.getenv("SOL_FEE_RESERVE", "0.01"))
ETH_FEE_RESERVE = float(os.getenv("ETH_FEE_RESERVE", "0.005"))  # ~$10-15 buffer for gas

# Indicator params (shared across all bots)
ATR_PERIOD  = int(os.getenv("ATR_PERIOD",  "10"))
ATR_MULT    = float(os.getenv("ATR_MULT",  "3.0"))
RSI_PERIOD  = int(os.getenv("RSI_PERIOD",  "14"))
VOL_LOOKBACK = int(os.getenv("VOL_LOOKBACK", "20"))
ADXR_LEN    = int(os.getenv("ADXR_LEN",   "14"))
ADXR_LENX   = int(os.getenv("ADXR_LENX",  "14"))
ADXR_LOW    = float(os.getenv("ADXR_LOW",  "20.0"))
ADXR_EPS    = float(os.getenv("ADXR_EPS",  "1e-6"))

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
EVM_CHAINS = {"ethereum", "base", "optimism"}
SOLANA_CHAINS = {"solana"}


# ═══════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════
@dataclass
class AssetInfo:
    """Resolved from asset_registry.json."""
    ticker:             str
    blockchain:         str
    wallet_address:     str         # public address (EVM) — informational
    token_contract:     str         # "" = native SOL; EVM = ERC-20 address
    yahoo_ticker:       str
    stablecoin_contract: str
    decimals:           int = 9

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
# Path to token decimals cache file (in repo root alongside asset_registry.json)
TOKEN_DECIMALS_PATH = os.getenv("TOKEN_DECIMALS_PATH", "./token_decimals.json")


def _load_token_decimals() -> dict[str, int]:
    """Load token decimals from JSON file. Returns empty dict if file missing."""
    try:
        with open(TOKEN_DECIMALS_PATH, "r", encoding="utf-8") as f:
            return {k.upper(): int(v) for k, v in json.load(f).items()}
    except FileNotFoundError:
        return {}
    except Exception as e:
        log.warning("Could not load token_decimals.json: %s", e)
        return {}


def _save_token_decimals(decimals: dict[str, int]) -> None:
    """Save token decimals dict to JSON file."""
    try:
        with open(TOKEN_DECIMALS_PATH, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in sorted(decimals.items())}, f, indent=2)
    except Exception as e:
        log.warning("Could not save token_decimals.json: %s", e)


def _lookup_decimals_on_chain(
    key: str,
    token_contract: str,
    wallet_address: str,
) -> int | None:
    """
    Look up token decimals from Solana RPC.
    Returns None if lookup fails.
    """
    try:
        from core.execution.jupiter import rpc_call
        rpc_url = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
        res = rpc_call(
            rpc_url,
            "getTokenAccountsByOwner",
            [wallet_address, {"mint": token_contract}, {"encoding": "jsonParsed", "commitment": "confirmed"}],
        )
        for acc in res.get("value", []):
            info = acc["account"]["data"]["parsed"]["info"]
            dec = int(info["tokenAmount"]["decimals"])
            log.info("🔍 On-chain decimals for %s: %d", key, dec)
            return dec
    except Exception as e:
        log.warning("Could not look up decimals for %s on-chain: %s", key, e)
    return None


# Load decimals cache at module level
TOKEN_DECIMALS: dict[str, int] = _load_token_decimals()


def load_asset_registry(path: str) -> dict[str, AssetInfo]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    decimals_updated = False
    registry = {}

    for key, val in raw.items():
        key_upper    = key.upper()
        blockchain   = val.get("blockchain", "")
        token_contract  = val.get("token_contract", "")
        wallet_address  = val.get("wallet_address", "")

        # Decimals priority:
        # 1) Explicit "decimals" field in asset_registry.json  ← always wins
        # 2) token_decimals.json cache (loaded into TOKEN_DECIMALS at startup)
        # 3) On-chain RPC lookup for Solana tokens (auto-saved to JSON for next time)
        # 4) Chain default: EVM=18, Solana=9
        if "decimals" in val:
            decimals = int(val["decimals"])

        elif key_upper in TOKEN_DECIMALS:
            decimals = TOKEN_DECIMALS[key_upper]

        elif blockchain == "solana" and token_contract and wallet_address:
            looked_up = _lookup_decimals_on_chain(key_upper, token_contract, wallet_address)
            if looked_up is not None:
                decimals = looked_up
                TOKEN_DECIMALS[key_upper] = decimals
                decimals_updated = True
            else:
                decimals = 9
                log.warning("⚠️  Could not determine decimals for %s — defaulting to 9. "
                            "Add manually to token_decimals.json if wrong.", key_upper)

        elif blockchain in EVM_CHAINS:
            decimals = 18   # EVM tokens are virtually always 18

        else:
            decimals = 9    # Solana fallback

        registry[key_upper] = AssetInfo(
            ticker=val["ticker"],
            blockchain=blockchain,
            wallet_address=wallet_address,
            token_contract=token_contract,
            yahoo_ticker=val["yahoo_ticker"],
            stablecoin_contract=val.get(
                "stablecoin_contract",
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"   # Solana USDC
                if blockchain == "solana"
                else "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"  # Ethereum/Base/Optimism USDC
            ),
            decimals=decimals,
        )

    # Persist any newly discovered decimals back to token_decimals.json
    if decimals_updated:
        _save_token_decimals(TOKEN_DECIMALS)
        log.info("💾 token_decimals.json updated with newly discovered entries")

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
# Fernet key + keypair loading (Solana + EVM, same encryption)
# ═══════════════════════════════════════════════════════════════════
_fernet_cache: Fernet | None = None


def _get_fernet() -> Fernet:
    global _fernet_cache
    if _fernet_cache is None:
        key_path = os.getenv("BOT_FERNET_KEY_PATH", "/etc/myTrading/bot.key")
        # Fallback to legacy path
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
# State management (per bot — unchanged from jupBot)
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
# Market data (unchanged)
# ═══════════════════════════════════════════════════════════════════
def fetch_df(yahoo_ticker: str, interval: str) -> pd.DataFrame:
    lookback = LOOKBACK_MAP.get(interval, 30)
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
        token_bal = get_sol_balance(SOLANA_RPC_URL, pubkey)
        tradable_token = max(0.0, token_bal - SOL_FEE_RESERVE)
    else:
        token_bal = get_spl_token_balance_ui(SOLANA_RPC_URL, pubkey, asset.token_contract)
        tradable_token = token_bal
        sol_bal = get_sol_balance(SOLANA_RPC_URL, pubkey)
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
        # Native ETH (unlikely to trade but supported)
        token_bal = get_evm_native_balance(blockchain, wallet_address)

    # Gas reserve check
    eth_bal = get_evm_native_balance(blockchain, wallet_address)
    if eth_bal < ETH_FEE_RESERVE:
        log.warning(
            "[%s] Low ETH for gas: %.6f ETH (need %.6f)",
            blockchain, eth_bal, ETH_FEE_RESERVE,
        )
    tradable_token = token_bal

    # Stablecoin balance (USDC on EVM)
    stable_bal = get_evm_token_balance(
        blockchain, wallet_address, asset.stablecoin_contract,
        decimals=ERC20_DECIMALS.get("USDC", 6)
    )

    return _build_portfolio_dict(token_bal, tradable_token, stable_bal, token_price)


def get_portfolio(
    wallet,           # Keypair (Solana) or str pubkey (EVM)
    asset: AssetInfo,
    token_price: float,
) -> dict:
    """Dispatch to Solana or EVM portfolio function."""
    if asset.is_solana:
        pubkey = str(wallet.pubkey())
        return get_portfolio_solana(pubkey, asset, token_price)
    elif asset.is_evm:
        # For EVM we need the public address — stored in asset_registry.json
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
# Rebalance planning — unchanged, generic for any token pair
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
    sell_token = min(port["tradable_token"], need_sell_usd / token_price)
    if sell_token * token_price < MIN_SWAP_USD:
        return {"action": "NONE", "usd_diff": diff}
    return {"action": "SELL_TOKEN", "token_amount": sell_token, "usd_diff": diff}


# ═══════════════════════════════════════════════════════════════════
# Execution — routes to Jupiter (Solana) or Uniswap V3 (EVM)
# ═══════════════════════════════════════════════════════════════════
def execute_plan(
    *,
    wallet,           # Keypair (Solana) or hex str (EVM)
    plan: dict,
    asset: AssetInfo,
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
        return _execute_plan_evm(private_key=wallet, plan=plan, asset=asset)
    else:
        raise ValueError(f"Unknown blockchain: {asset.blockchain}")


def _execute_plan_solana(*, wallet: Keypair, plan: dict, asset: AssetInfo) -> dict | None:
    """Execute a swap on Solana via Jupiter Ultra API."""
    pubkey      = str(wallet.pubkey())
    token_mint  = asset.token_mint
    stable_mint = asset.stable_mint
    ticker      = asset.ticker

    if plan["action"] == "BUY_TOKEN":
        stable_amt = float(plan["stable_amount"])
        amt_small  = to_smallest(stable_amt, USDC_DECIMALS)
        order      = ultra_get_order(stable_mint, token_mint, amt_small, pubkey)
        sig        = ultra_sign_and_execute(order, wallet)
        log.info("✅ SOL BUY %s: spent USDC=%.2f sig=%s", ticker, stable_amt, sig)
        return {"action": f"BUY_{ticker}", "amount": stable_amt, "amount_ccy": "USDC", "tx_sig": sig}

    if plan["action"] == "SELL_TOKEN":
        token_amt = float(plan["token_amount"])
        amt_small = to_smallest(token_amt, asset.decimals)
        order     = ultra_get_order(token_mint, stable_mint, amt_small, pubkey)
        sig       = ultra_sign_and_execute(order, wallet)
        log.info("✅ SOL SELL %s: sold %.6f sig=%s", ticker, token_amt, sig)
        return {"action": f"SELL_{ticker}", "amount": token_amt, "amount_ccy": ticker, "tx_sig": sig}

    raise RuntimeError(f"Unknown plan action: {plan}")


def _execute_plan_evm(*, private_key: str, plan: dict, asset: AssetInfo) -> dict | None:
    """Execute a swap on an EVM chain via Uniswap V3."""
    blockchain      = asset.blockchain
    token_contract  = asset.token_contract
    stable_contract = asset.stablecoin_contract
    ticker          = asset.ticker
    usdc_decimals   = ERC20_DECIMALS.get("USDC", 6)

    if plan["action"] == "BUY_TOKEN":
        stable_amt = float(plan["stable_amount"])
        tx_hash = uniswap_swap_auto_fee(
            blockchain=blockchain,
            private_key=private_key,
            token_in=stable_contract,
            token_out=token_contract,
            amount_in_human=stable_amt,
            token_in_decimals=usdc_decimals,
            token_out_decimals=asset.decimals,
            slippage_bps=SLIPPAGE_BPS,
        )
        log.info("✅ EVM BUY %s [%s]: spent USDC=%.2f tx=%s", ticker, blockchain, stable_amt, tx_hash)
        return {"action": f"BUY_{ticker}", "amount": stable_amt, "amount_ccy": "USDC", "tx_sig": tx_hash}

    if plan["action"] == "SELL_TOKEN":
        token_amt = float(plan["token_amount"])
        tx_hash = uniswap_swap_auto_fee(
            blockchain=blockchain,
            private_key=private_key,
            token_in=token_contract,
            token_out=stable_contract,
            amount_in_human=token_amt,
            token_in_decimals=asset.decimals,
            token_out_decimals=usdc_decimals,
            slippage_bps=SLIPPAGE_BPS,
        )
        log.info("✅ EVM SELL %s [%s]: sold %.6f tx=%s", ticker, blockchain, token_amt, tx_hash)
        return {"action": f"SELL_{ticker}", "amount": token_amt, "amount_ccy": ticker, "tx_sig": tx_hash}

    raise RuntimeError(f"Unknown plan action: {plan}")


# ═══════════════════════════════════════════════════════════════════
# Trade logging (per bot — unchanged structure)
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
# Higher-timeframe confirmation filter (unchanged)
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
    htf_most  = str(last.get("MOST_Signal", "SELL"))
    htf_adxr  = str(last.get("ADXR_State", "FLAT"))
    htf_final = signal_super_most_adxr(htf_st, htf_most, htf_adxr)

    return {
        "supertrend":  htf_st,
        "most":        htf_most,
        "adxr_state":  htf_adxr,
        "final_signal": htf_final,
        "allows_buy":  htf_final in ("BUY", "HOLD"),
    }


# ═══════════════════════════════════════════════════════════════════
# Heartbeat log (per bot — unchanged, adds blockchain field)
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
    st_sig    = str(last.get("Supertrend_Signal", "SELL"))
    most_sig  = str(last.get("MOST_Signal", "SELL"))
    adxr_state = str(last.get("ADXR_State", "FLAT"))

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

    # ── Always fetch + log balances so we can detect drift ──
    port = None
    try:
        port = get_portfolio(wallet, asset, price)
        blog.info(
            "💼 Balance: total=$%.2f | %s=%.4f ($%.2f, %.1f%%) | stable=%.2f",
            port["total"], asset.ticker,
            port["token_bal"], port["token_val"], port["token_pct"] * 100.0,
            port["stable_bal"],
        )
    except Exception as e:
        blog.warning("⚠️  Balance fetch failed: %s — will skip drift check", e)

    # ── Drift detection: regime=OUT but still holding token (or vice versa) ──
    drift_action = None
    if port is not None:
        token_pct = port["token_pct"]
        if st.regime == "OUT" and token_pct > 0.50:
            blog.warning(
                "🚨 DRIFT DETECTED: regime=OUT but token_pct=%.1f%% > 50%% — forcing SELL",
                token_pct * 100.0,
            )
            drift_action = "SELL"
        elif st.regime == "IN" and token_pct < 0.30:
            blog.warning(
                "🚨 DRIFT DETECTED: regime=IN but token_pct=%.1f%% < 30%% — forcing BUY",
                token_pct * 100.0,
            )
            drift_action = "BUY"
        else:
            blog.info(
                "✅ Balance aligned: regime=%s token_pct=%.1f%% — no drift",
                st.regime, token_pct * 100.0,
            )

    # ── Skip same bar UNLESS drift detected ──
    if st.last_bar_ts == bar_ts:
        if drift_action is None:
            blog.info("Same bar + no drift — skipping trade logic.")
            return st
        else:
            blog.info("Same bar but DRIFT detected — proceeding with corrective rebalance.")

    # ── Helper: execute rebalance and log trade ──
    def _do_rebalance(target_pct: float, reason: str):
        nonlocal port
        if port is None:
            blog.warning("No portfolio data — cannot rebalance (%s)", reason)
            return
        blog.info("🔄 %s → target %.0f%% %s", reason, target_pct * 100, asset.ticker)
        plan = rebalance_plan(port, price, target_pct)
        blog.info("Plan: %s", plan)
        exec_result = execute_plan(wallet=wallet, plan=plan, asset=asset)
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
            st.regime = desired_regime

    # ── Corrective rebalance for drift (same-bar override) ──
    if drift_action == "SELL":
        _do_rebalance(OUT_TOKEN_PCT, "DRIFT CORRECTION SELL")
        st.last_bar_ts = bar_ts
        _save_state(bot.bot_id, st)
        return st

    if drift_action == "BUY":
        _do_rebalance(IN_TOKEN_PCT, "DRIFT CORRECTION BUY")
        st.last_bar_ts = bar_ts
        _save_state(bot.bot_id, st)
        return st

    # ── Normal regime flip (new bar) ──
    if desired_regime != st.regime:
        target_pct = IN_TOKEN_PCT if desired_regime == "IN" else OUT_TOKEN_PCT
        _do_rebalance(target_pct, f"REGIME FLIP {st.regime}→{desired_regime}")
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
    chains = {}
    for b in bots:
        chains[b.asset.blockchain] = chains.get(b.asset.blockchain, 0) + 1
    for chain, count in chains.items():
        log.info("  Chain: %-12s → %d bot(s)", chain, count)

    # ── Load wallets once at startup ──
    # Key: (wallet_env, blockchain) — same wallet_env can be Solana on one bot, EVM on another
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
                log.exception("[%s] Error: %s", bot.name, e)
            time.sleep(10)

        time.sleep(SLEEP_SECONDS)



# ═══════════════════════════════════════════════════════════════════
# Manual trade CLI
# ═══════════════════════════════════════════════════════════════════
def manual_trade(action: str, ticker: str, usd_amount: float) -> None:
    """
    Execute a single manual trade from the command line.

    Examples:
        python3 -m bots.bot SELL WETH 100   -> sell $100 worth of WETH -> USDC
        python3 -m bots.bot BUY  WETH 100   -> buy  $100 worth of WETH with USDC
    """
    import sys
    import traceback

    mlog = logging.getLogger("bot.manual")
    action = action.upper()
    ticker = ticker.upper()

    # ── Validate inputs ──
    if action not in ("BUY", "SELL"):
        mlog.error("Unknown action '%s'. Must be BUY or SELL.", action)
        sys.exit(1)
    if usd_amount <= 0:
        mlog.error("USD amount must be > 0 (got %s)", usd_amount)
        sys.exit(1)

    mlog.info("=" * 60)
    mlog.info("MANUAL TRADE: %s %s  USD=%.2f  dry_run=%s", action, ticker, usd_amount, DRY_RUN)
    mlog.info("=" * 60)

    # ── Load asset registry ──
    try:
        asset_reg = load_asset_registry(ASSET_REGISTRY_PATH)
        mlog.info("Loaded %d assets from %s", len(asset_reg), ASSET_REGISTRY_PATH)
    except Exception as e:
        mlog.error("Failed to load asset_registry: %s", e)
        mlog.debug(traceback.format_exc())
        sys.exit(1)

    if ticker not in asset_reg:
        mlog.error("'%s' not found in asset_registry.", ticker)
        mlog.error("Available tickers: %s", sorted(asset_reg))
        sys.exit(1)
    asset = asset_reg[ticker]

    # ── Find wallet_env from bot_registry ──
    try:
        with open(BOT_REGISTRY_PATH, "r", encoding="utf-8") as f:
            bot_entries = json.load(f)
    except Exception as e:
        mlog.error("Failed to load bot_registry at %s: %s", BOT_REGISTRY_PATH, e)
        sys.exit(1)

    wallet_env = None
    for entry in bot_entries:
        if entry.get("asset", "").upper() == ticker:
            wallet_env = entry["wallet_env"]
            break
    if wallet_env is None:
        mlog.error("No bot_registry entry found for '%s'.", ticker)
        mlog.error("Cannot determine wallet_env — add an entry to bot_registry.json")
        sys.exit(1)

    mlog.info("Asset     : %s (%s)", asset.ticker, asset.blockchain)
    mlog.info("Wallet env: %s", wallet_env)
    mlog.info("Contract  : %s", asset.token_contract or "native")
    mlog.info("Stable    : %s", asset.stablecoin_contract)
    mlog.info("Decimals  : %d", asset.decimals)

    # ── Fetch current price ──
    mlog.info("Fetching price for %s ...", asset.yahoo_ticker)
    try:
        df = fetch_df(asset.yahoo_ticker, "4h")
        price = float(df.iloc[-1]["Close"])
        mlog.info("Price     : $%.6f  (last 4H close)", price)
    except Exception as e:
        mlog.error("Failed to fetch price for %s: %s", asset.yahoo_ticker, e)
        mlog.debug(traceback.format_exc())
        sys.exit(1)

    # ── Fetch live balances ──
    mlog.info("Fetching wallet balances...")
    try:
        wallet_tmp = load_wallet(wallet_env, asset.blockchain)
        port = get_portfolio(wallet_tmp, asset, price)
        mlog.info(
            "Balance   : total=$%.2f | %s=%.6f ($%.2f, %.1f%%) | stable=%.2f",
            port["total"], asset.ticker,
            port["token_bal"], port["token_val"], port["token_pct"] * 100,
            port["stable_bal"],
        )
    except Exception as e:
        mlog.warning("Could not fetch balances (non-fatal): %s", e)
        mlog.debug(traceback.format_exc())
        port = None

    # ── Build plan ──
    if action == "SELL":
        token_qty = usd_amount / price
        plan = {"action": "SELL_TOKEN", "token_amount": token_qty, "usd_diff": -usd_amount}
        mlog.info("Plan      : SELL %.6f %s (~$%.2f)", token_qty, ticker, usd_amount)
        # Sanity check
        if port and token_qty > port["token_bal"]:
            mlog.warning(
                "Requested sell %.6f %s but wallet only has %.6f — will sell all available",
                token_qty, ticker, port["token_bal"],
            )
            plan["token_amount"] = port["tradable_token"]
    else:
        plan = {"action": "BUY_TOKEN", "stable_amount": usd_amount, "usd_diff": usd_amount}
        mlog.info("Plan      : BUY $%.2f of %s", usd_amount, ticker)
        # Sanity check
        if port and usd_amount > port["stable_bal"]:
            mlog.warning(
                "Requested spend $%.2f USDC but wallet only has $%.2f — will spend available",
                usd_amount, port["stable_bal"],
            )
            plan["stable_amount"] = port["stable_bal"]

    mlog.info("Full plan : %s", plan)

    if DRY_RUN:
        mlog.warning("DRY_RUN=true — plan built but NOT executed. Set DRY_RUN=false to trade.")
        return

    # ── Load wallet ──
    try:
        wallet = load_wallet(wallet_env, asset.blockchain)
        if asset.is_solana:
            mlog.info("Wallet    : %s (Solana)", str(wallet.pubkey()))
        else:
            mlog.info("Wallet    : %s (%s)", asset.wallet_address, asset.blockchain)
    except Exception as e:
        mlog.error("Failed to load wallet from env var '%s': %s", wallet_env, e)
        mlog.error("Make sure the encrypted key is set in .env and Fernet key is at %s",
                   os.getenv("BOT_FERNET_KEY_PATH", "/etc/myTrading/bot.key"))
        mlog.debug(traceback.format_exc())
        sys.exit(1)

    # ── Execute ──
    mlog.info("Executing trade...")
    try:
        result = execute_plan(wallet=wallet, plan=plan, asset=asset)
    except Exception as e:
        mlog.error("Trade execution FAILED: %s", e)
        mlog.error("Full traceback:")
        mlog.error(traceback.format_exc())
        sys.exit(1)

    if result:
        mlog.info("=" * 60)
        mlog.info("SUCCESS: action=%s  amount=%s %s  tx=%s",
                  result.get("action"), result.get("amount"),
                  result.get("amount_ccy"), result.get("tx_sig"))
        mlog.info("=" * 60)
    else:
        mlog.warning("execute_plan returned None (plan action was NONE — within tolerance?)")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        prog="bot.py",
        description="Multi-chain trading bot (Solana + EVM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 bot.py                       run normal loop (all bots)
  python3 bot.py SELL WETH 100         manually sell $100 of WETH -> USDC
  python3 bot.py BUY  WETH 100         manually buy  $100 of WETH with USDC
  python3 bot.py SELL JUP29210 50      manually sell $50 of JUP -> USDC
  DRY_RUN=true python3 bot.py SELL WETH 100   dry run (no real tx)
        """,
    )
    parser.add_argument("action",     nargs="?", choices=["BUY","SELL","buy","sell"], help="BUY or SELL")
    parser.add_argument("ticker",     nargs="?", help="Token ticker e.g. WETH, GRIFFAIN, JUP29210")
    parser.add_argument("usd_amount", nargs="?", type=float, help="USD amount to trade")

    args = parser.parse_args()

    if args.action and args.ticker and args.usd_amount:
        manual_trade(args.action, args.ticker, args.usd_amount)
    elif args.action or args.ticker or args.usd_amount:
        print("Manual trade requires all three args: ACTION TICKER USD_AMOUNT")
        print("Example: python3 bot.py SELL WETH 100")
        parser.print_help()
        sys.exit(1)
    else:
        main()