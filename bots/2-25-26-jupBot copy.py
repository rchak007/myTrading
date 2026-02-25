#!/usr/bin/env python3
# bots/jupBot.py
#
# Multi-wallet, multi-token Jupiter trading bot.
# Reads bot_registry.json for wallet/asset pairs and asset_registry.json
# for token details (mint addresses, yahoo tickers, etc.).
# Runs all bots sequentially in a single process loop.

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

from solders.keypair import Keypair
from cryptography.fernet import Fernet

from core.indicators import apply_indicators
from core.signals import signal_super_most_adxr

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
log = logging.getLogger("jupBot")

RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
BOT_TZ = os.getenv("BOT_TIMEZONE", "UTC")

# Paths
BOT_REGISTRY_PATH = os.getenv("BOT_REGISTRY_PATH", "./bot_registry.json")
ASSET_REGISTRY_PATH = os.getenv("ASSET_REGISTRY_PATH", "./asset_registry.json")
STATE_DIR = os.getenv("JUPBOT_STATE_DIR", "./outputs")
TRADE_LOG_DIR = os.getenv("JUPBOT_TRADE_LOG_DIR", "./outputs")
STATE_MIRROR_DIR = os.getenv("JUPBOT_STATE_MIRROR_DIR")  # optional
TRADE_LOG_MIRROR_DIR = os.getenv("JUPBOT_TRADE_LOG_MIRROR_DIR")  # optional

# Trading behavior
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"
SLIPPAGE_BPS = int(os.getenv("SLIPPAGE_BPS", "30"))
SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", "60"))

# Allocation targets
IN_TOKEN_PCT = float(os.getenv("IN_TOKEN_PCT", "0.80"))   # BUY/HOLD
OUT_TOKEN_PCT = float(os.getenv("OUT_TOKEN_PCT", "0.20"))  # EXIT/STANDDOWN

# Safety / dust
USD_TOLERANCE = float(os.getenv("USD_TOLERANCE", "5"))
MIN_SWAP_USD = float(os.getenv("MIN_SWAP_USD", "10"))
SOL_FEE_RESERVE = float(os.getenv("SOL_FEE_RESERVE", "0.01"))

# Indicator params (shared across all bots)
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "10"))
ATR_MULT = float(os.getenv("ATR_MULT", "3.0"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
VOL_LOOKBACK = int(os.getenv("VOL_LOOKBACK", "20"))
ADXR_LEN = int(os.getenv("ADXR_LEN", "14"))
ADXR_LENX = int(os.getenv("ADXR_LENX", "14"))
ADXR_LOW = float(os.getenv("ADXR_LOW", "20.0"))
ADXR_EPS = float(os.getenv("ADXR_EPS", "1e-6"))

# Lookback per interval (days of history to fetch)
LOOKBACK_MAP = {
    "1h":  30,
    "2h":  60,
    "4h":  90,
    "1d":  365,
    "1wk": 730,
}

# yfinance interval mapping (yfinance uses "60m" not "1h", etc.)
YF_INTERVAL_MAP = {
    "1h":  "60m",
    "2h":  "2h",
    "4h":  "4h",       # Note: yfinance 4h support may be limited
    "1d":  "1d",
    "1wk": "1wk",
}


# ═══════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════
@dataclass
class AssetInfo:
    """Resolved from asset_registry.json."""
    ticker: str
    blockchain: str
    token_contract: str        # empty string = native SOL
    yahoo_ticker: str
    stablecoin_contract: str
    decimals: int = 9          # default; overridden for known tokens

    @property
    def is_native_sol(self) -> bool:
        return self.token_contract == ""

    @property
    def token_mint(self) -> str:
        """Mint address for Jupiter swaps."""
        return WSOL_MINT if self.is_native_sol else self.token_contract

    @property
    def stable_mint(self) -> str:
        return self.stablecoin_contract


@dataclass
class BotEntry:
    """One row from bot_registry.json, enriched with asset details."""
    name: str
    wallet_env: str
    asset_key: str
    interval: str
    asset: AssetInfo = field(default=None)

    @property
    def bot_id(self) -> str:
        """Unique ID for state/log files, e.g. 'SOL_1h' or 'PYTH_1h'."""
        return f"{self.asset_key}_{self.interval}"


@dataclass
class BotState:
    last_bar_ts: Optional[str] = None
    regime: str = "OUT"


# ═══════════════════════════════════════════════════════════════════
# Registry loaders
# ═══════════════════════════════════════════════════════════════════
# Known SPL token decimals (add more as you onboard tokens)
TOKEN_DECIMALS = {
    "SOL":  9,   # WSOL / native
    "USDC": 6,
    "PYTH": 6,
    "JUP":  6,
    "BONK": 5,
    "RAY":  6,
    "JTO":  9,
    "WIF":  6,
    "RNDR": 8,
    "HNT":  8,
}


def load_asset_registry(path: str) -> dict[str, AssetInfo]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    registry = {}
    for key, val in raw.items():
        decimals = TOKEN_DECIMALS.get(key.upper(), 9)
        registry[key.upper()] = AssetInfo(
            ticker=val["ticker"],
            blockchain=val["blockchain"],
            token_contract=val.get("token_contract", ""),
            yahoo_ticker=val["yahoo_ticker"],
            stablecoin_contract=val.get(
                "stablecoin_contract",
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC default
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
        )
        bots.append(bot)

    return bots


# ═══════════════════════════════════════════════════════════════════
# Fernet key + keypair loading
# ═══════════════════════════════════════════════════════════════════
_fernet_cache: Fernet | None = None

def _get_fernet() -> Fernet:
    global _fernet_cache
    if _fernet_cache is None:
        key_path = os.getenv("JUPBOT_FERNET_KEY_PATH", "/etc/myTrading/jupbot.key")
        with open(key_path, "rb") as f:
            _fernet_cache = Fernet(f.read().strip())
    return _fernet_cache


def load_keypair(wallet_env: str) -> Keypair:
    """Decrypt the private key from the given .env variable name."""
    enc = (os.getenv(wallet_env) or "").strip()
    if not enc:
        raise RuntimeError(f"Missing {wallet_env} in .env")

    pk_b58 = _get_fernet().decrypt(enc.encode("utf-8")).decode("utf-8").strip()
    return Keypair.from_base58_string(pk_b58)


# ═══════════════════════════════════════════════════════════════════
# State management (per bot)
# ═══════════════════════════════════════════════════════════════════
def _state_path(bot_id: str) -> str:
    return os.path.join(STATE_DIR, f"jupbot_state_{bot_id}.json")


def _load_state(bot_id: str) -> BotState:
    try:
        with open(_state_path(bot_id), "r", encoding="utf-8") as f:
            data = json.load(f)
        return BotState(
            last_bar_ts=data.get("last_bar_ts"),
            regime=data.get("regime", "OUT"),
        )
    except Exception:
        return BotState()


def _save_state(bot_id: str, st: BotState) -> None:
    path = _state_path(bot_id)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload = {"last_bar_ts": st.last_bar_ts, "regime": st.regime}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Mirror state (if configured)
    if STATE_MIRROR_DIR:
        try:
            mirror = os.path.join(STATE_MIRROR_DIR, f"jupbot_state_{bot_id}.json")
            os.makedirs(os.path.dirname(mirror), exist_ok=True)
            shutil.copyfile(path, mirror)
        except Exception as e:
            log.warning("[%s] State mirror failed: %s", bot_id, e)


# ═══════════════════════════════════════════════════════════════════
# Market data
# ═══════════════════════════════════════════════════════════════════
def fetch_df(yahoo_ticker: str, interval: str) -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance for the given ticker and interval.
    Returns df with columns: High, Low, Close, Volume.
    """
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

    # Normalize multiindex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df[["High", "Low", "Close", "Volume"]].dropna()
    if len(df) < 100:
        raise RuntimeError(
            f"Not enough candles for {yahoo_ticker} {yf_interval}: {len(df)}"
        )

    # Make timezone-naive
    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        pass

    return df


# ═══════════════════════════════════════════════════════════════════
# Signal logic (unchanged)
# ═══════════════════════════════════════════════════════════════════
def desired_regime_from_final_signal(final_sig: str) -> str:
    return "IN" if final_sig in ("BUY", "HOLD") else "OUT"


# ═══════════════════════════════════════════════════════════════════
# Portfolio — handles native SOL vs SPL tokens
# ═══════════════════════════════════════════════════════════════════
def get_token_balance(
    rpc_url: str,
    pubkey: str,
    asset: AssetInfo,
) -> float:
    """Get the balance of the risk token (SOL native or any SPL token)."""
    if asset.is_native_sol:
        return get_sol_balance(rpc_url, pubkey)
    else:
        return get_spl_token_balance_ui(rpc_url, pubkey, asset.token_contract)


def get_portfolio(
    rpc_url: str,
    pubkey: str,
    asset: AssetInfo,
    token_price: float,
) -> dict:
    """
    Build portfolio snapshot.
    Returns dict with token/stable balances and USD values.
    """
    token_bal = get_token_balance(rpc_url, pubkey, asset)
    stable_bal = get_spl_token_balance_ui(rpc_url, pubkey, asset.stable_mint)

    # For native SOL, reserve some for tx fees
    if asset.is_native_sol:
        tradable_token = max(0.0, token_bal - SOL_FEE_RESERVE)
    else:
        tradable_token = token_bal
        # Still need native SOL for fees — check we have enough
        sol_bal = get_sol_balance(rpc_url, pubkey)
        if sol_bal < SOL_FEE_RESERVE:
            log.warning(
                "Low SOL for fees: %.4f SOL (need %.4f)",
                sol_bal, SOL_FEE_RESERVE,
            )

    token_val = token_bal * token_price
    stable_val = stable_bal  # stablecoin ≈ 1 USD
    total = token_val + stable_val
    token_pct = (token_val / total) if total > 0 else 0.0

    return {
        "token_bal": token_bal,
        "tradable_token": tradable_token,
        "stable_bal": stable_bal,
        "token_val": token_val,
        "stable_val": stable_val,
        "total": total,
        "token_pct": token_pct,
    }


# ═══════════════════════════════════════════════════════════════════
# Rebalance planning — generic for any token pair
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

    # diff > 0 => buy token with stablecoin
    if diff > 0:
        spend = min(port["stable_bal"], diff)
        if spend < MIN_SWAP_USD:
            return {"action": "NONE", "usd_diff": diff}
        return {"action": "BUY_TOKEN", "stable_amount": spend, "usd_diff": diff}

    # diff < 0 => sell token for stablecoin
    need_sell_usd = min(
        port["token_val"] - desired_token_val,
        port["token_val"],
    )
    sell_token = min(port["tradable_token"], need_sell_usd / token_price)
    if sell_token * token_price < MIN_SWAP_USD:
        return {"action": "NONE", "usd_diff": diff}
    return {"action": "SELL_TOKEN", "token_amount": sell_token, "usd_diff": diff}


# ═══════════════════════════════════════════════════════════════════
# Execution — generic for any Solana token pair via Jupiter
# ═══════════════════════════════════════════════════════════════════
def execute_plan(
    *,
    rpc_url: str,
    kp: Keypair,
    pubkey: str,
    plan: dict,
    asset: AssetInfo,
) -> dict | None:

    if plan["action"] == "NONE":
        log.info("Rebalance: NONE (within tolerance)")
        return None

    if DRY_RUN:
        log.warning("DRY_RUN: would execute plan=%s", plan)
        return None

    token_mint = asset.token_mint
    stable_mint = asset.stable_mint
    token_decimals = asset.decimals
    ticker = asset.ticker

    if plan["action"] == "BUY_TOKEN":
        stable_amt = float(plan["stable_amount"])
        amt_small = to_smallest(stable_amt, USDC_DECIMALS)
        quote = get_quote(stable_mint, token_mint, amt_small, SLIPPAGE_BPS)
        swap = get_swap_tx(quote, pubkey)
        tx = swap.get("swapTransaction")
        if not tx:
            raise RuntimeError(f"Jupiter swap missing swapTransaction: {swap}")

        sig = sign_and_send_swap(rpc_url=rpc_url, swap_tx_b64=tx, keypair=kp)
        log.info("✅ BUY %s: spent USDC=%.2f sig=%s", ticker, stable_amt, sig)
        return {
            "action": f"BUY_{ticker}",
            "amount": stable_amt,
            "amount_ccy": "USDC",
            "tx_sig": sig,
        }

    if plan["action"] == "SELL_TOKEN":
        token_amt = float(plan["token_amount"])
        amt_small = to_smallest(token_amt, token_decimals)
        quote = get_quote(token_mint, stable_mint, amt_small, SLIPPAGE_BPS)
        swap = get_swap_tx(quote, pubkey)
        tx = swap.get("swapTransaction")
        if not tx:
            raise RuntimeError(f"Jupiter swap missing swapTransaction: {swap}")

        sig = sign_and_send_swap(rpc_url=rpc_url, swap_tx_b64=tx, keypair=kp)
        log.info("✅ SELL %s: sold %.6f sig=%s", ticker, token_amt, sig)
        return {
            "action": f"SELL_{ticker}",
            "amount": token_amt,
            "amount_ccy": ticker,
            "tx_sig": sig,
        }

    raise RuntimeError(f"Unknown plan action: {plan}")


# ═══════════════════════════════════════════════════════════════════
# Trade logging (per bot)
# ═══════════════════════════════════════════════════════════════════
def _trade_log_path(bot_id: str) -> str:
    return os.path.join(TRADE_LOG_DIR, f"jupbot_trades_{bot_id}.csv")


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
):
    path = _trade_log_path(bot_id)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    file_exists = os.path.exists(path)

    with open(path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write(
                "timestamp,bot_name,action,regime_from,regime_to,"
                "price,amount,amount_ccy,tx_sig,dry_run\n"
            )

        pst = ZoneInfo("America/Los_Angeles")
        ts = datetime.now(pst).strftime("%Y-%m-%d %H:%M:%S")

        f.write(
            f"{ts},"
            f"{bot_name},"
            f"{action},"
            f"{regime_from},"
            f"{regime_to},"
            f"{price:.4f},"
            f"{amount:.6f},"
            f"{amount_ccy},"
            f"{tx_sig or ''},"
            f"{dry_run}\n"
        )

    # Mirror trade log (if configured)
    if TRADE_LOG_MIRROR_DIR:
        try:
            mirror = os.path.join(TRADE_LOG_MIRROR_DIR, f"jupbot_trades_{bot_id}.csv")
            os.makedirs(os.path.dirname(mirror), exist_ok=True)
            shutil.copyfile(path, mirror)
        except Exception as e:
            log.warning("[%s] Trade log mirror failed: %s", bot_id, e)


# ═══════════════════════════════════════════════════════════════════
# Per-bot tick — runs once per new bar
# ═══════════════════════════════════════════════════════════════════
def tick_bot(bot: BotEntry, kp: Keypair, st: BotState) -> BotState:
    """
    Process one tick for a single bot entry.
    Returns updated BotState.
    """
    asset = bot.asset
    blog = logging.getLogger(f"jupBot.{bot.bot_id}")

    # ── Fetch market data & compute signals ──
    df = fetch_df(asset.yahoo_ticker, bot.interval)

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

    # ── Skip if same bar already processed ──
    if st.last_bar_ts == bar_ts:
        return st

    # ── Compute signal ──
    st_sig = str(last.get("Supertrend_Signal", "SELL"))
    most_sig = str(last.get("MOST_Signal", "SELL"))
    adxr_state = str(last.get("ADXR_State", "FLAT"))

    final_sig = signal_super_most_adxr(st_sig, most_sig, adxr_state)
    desired_regime = desired_regime_from_final_signal(final_sig)

    price = float(last["Close"])
    blog.info(
        "Bar=%s Close=%.4f | ST=%s MOST=%s ADXR=%s => FINAL=%s | "
        "prev_regime=%s desired=%s",
        bar_ts, price, st_sig, most_sig, adxr_state,
        final_sig, st.regime, desired_regime,
    )

    # ── Trade only on regime flip ──
    if desired_regime != st.regime:
        target_pct = IN_TOKEN_PCT if desired_regime == "IN" else OUT_TOKEN_PCT
        blog.info(
            "REGIME FLIP %s → %s. Rebalancing to %s%%=%.0f%%",
            st.regime, desired_regime, asset.ticker, target_pct * 100,
        )

        pubkey = str(kp.pubkey())
        port = get_portfolio(RPC_URL, pubkey, asset, price)
        blog.info(
            "Portfolio: total=$%.2f %s=%.4f (tradable %.4f) USDC=%.2f %s%%=%.1f%%",
            port["total"], asset.ticker, port["token_bal"],
            port["tradable_token"], port["stable_bal"],
            asset.ticker, port["token_pct"] * 100.0,
        )

        plan = rebalance_plan(port, price, target_pct)
        blog.info("Plan: %s", plan)

        exec_result = execute_plan(
            rpc_url=RPC_URL, kp=kp, pubkey=pubkey,
            plan=plan, asset=asset,
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
            )

        if plan.get("action") != "NONE":
            st.regime = desired_regime
    else:
        blog.info("No regime change (%s). Skipping.", st.regime)

    st.last_bar_ts = bar_ts
    _save_state(bot.bot_id, st)
    return st


# ═══════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 60)
    log.info("jupBot multi-wallet starting  |  dry_run=%s", DRY_RUN)
    log.info("=" * 60)

    # ── Load registries ──
    asset_reg = load_asset_registry(ASSET_REGISTRY_PATH)
    log.info("Loaded %d assets from %s", len(asset_reg), ASSET_REGISTRY_PATH)

    bots = load_bot_registry(BOT_REGISTRY_PATH, asset_reg)
    if not bots:
        log.error("No valid bot entries in %s — exiting.", BOT_REGISTRY_PATH)
        return
    log.info("Loaded %d bot(s) from %s", len(bots), BOT_REGISTRY_PATH)

    # ── Load keypairs once at startup ──
    keypairs: dict[str, Keypair] = {}
    for bot in bots:
        if bot.wallet_env not in keypairs:
            kp = load_keypair(bot.wallet_env)
            keypairs[bot.wallet_env] = kp
            log.info(
                "  [%s] wallet=%s pubkey=%s",
                bot.name, bot.wallet_env, str(kp.pubkey()),
            )

    # ── Load per-bot state ──
    states: dict[str, BotState] = {}
    for bot in bots:
        states[bot.bot_id] = _load_state(bot.bot_id)
        log.info(
            "  [%s] state: regime=%s last_bar=%s",
            bot.name,
            states[bot.bot_id].regime,
            states[bot.bot_id].last_bar_ts,
        )

    log.info("Entering main loop (sleep=%ds)...", SLEEP_SECONDS)
    log.info("=" * 60)

    # ── Main loop ──
    while True:
        for bot in bots:
            try:
                kp = keypairs[bot.wallet_env]
                st = states[bot.bot_id]
                states[bot.bot_id] = tick_bot(bot, kp, st)
            except Exception as e:
                log.exception("[%s] Error: %s", bot.name, e)

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()