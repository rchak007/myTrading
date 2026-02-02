#!/usr/bin/env python3
# bots/jupBot.py

from __future__ import annotations

import json
import os
import time
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
load_dotenv()

from solders.keypair import Keypair

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

from cryptography.fernet import Fernet

import shutil
from zoneinfo import ZoneInfo



# -----------------
# Config
# -----------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("jupBot")

RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
STATE_PATH = os.getenv("JUPBOT_STATE_PATH", "./outputs/jupbot_state.json")

STATE_MIRROR_PATH = os.getenv("JUPBOT_STATE_MIRROR_PATH")
BOT_TZ = os.getenv("BOT_TIMEZONE", "UTC")


# Trading behavior
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"
SLIPPAGE_BPS = int(os.getenv("SLIPPAGE_BPS", "30"))  # 0.30%
SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", "60"))

# Allocation targets
IN_SOL_PCT = float(os.getenv("IN_SOL_PCT", "0.80"))    # when BUY/HOLD
OUT_SOL_PCT = float(os.getenv("OUT_SOL_PCT", "0.20"))  # when EXIT/STANDDOWN

# Safety / dust
USD_TOLERANCE = float(os.getenv("USD_TOLERANCE", "5"))
MIN_SWAP_USD = float(os.getenv("MIN_SWAP_USD", "10"))
SOL_FEE_RESERVE = float(os.getenv("SOL_FEE_RESERVE", "0.01"))

# Market data
YF_TICKER = os.getenv("YF_TICKER", "SOL-USD")
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "30"))   # enough for indicators
INTERVAL = "60m"  # 1 hour timeframe

# Indicator params (match your Streamlit defaults if you want)
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "10"))
ATR_MULT = float(os.getenv("ATR_MULT", "3.0"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
VOL_LOOKBACK = int(os.getenv("VOL_LOOKBACK", "20"))
ADXR_LEN = int(os.getenv("ADXR_LEN", "14"))
ADXR_LENX = int(os.getenv("ADXR_LENX", "14"))
ADXR_LOW = float(os.getenv("ADXR_LOW", "20.0"))
ADXR_EPS = float(os.getenv("ADXR_EPS", "1e-6"))


TRADE_LOG_PATH = os.getenv(
    "TRADE_LOG_PATH",
    "./outputs/jupbot_trades.csv"
)

@dataclass
class BotState:
    last_bar_ts: Optional[str] = None
    regime: str = "OUT"   # "IN" or "OUT"

def _load_state() -> BotState:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return BotState(
            last_bar_ts=data.get("last_bar_ts"),
            regime=data.get("regime", "OUT"),
        )
    except Exception:
        return BotState()


# def _save_state(st: BotState) -> None:
#     os.makedirs(os.path.dirname(STATE_PATH) or ".", exist_ok=True)
#     with open(STATE_PATH, "w", encoding="utf-8") as f:
#         json.dump({"last_bar_ts": st.last_bar_ts, "regime": st.regime}, f, indent=2)


def _save_state(st: BotState) -> None:
    os.makedirs(os.path.dirname(STATE_PATH) or ".", exist_ok=True)

    payload = {
        "last_bar_ts": st.last_bar_ts,
        "regime": st.regime,
    }

    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # 🔁 Mirror state to jobMyTrading (if configured)
    if STATE_MIRROR_PATH:
        try:
            os.makedirs(os.path.dirname(STATE_MIRROR_PATH), exist_ok=True)
            shutil.copyfile(STATE_PATH, STATE_MIRROR_PATH)
        except Exception as e:
            log.warning("State mirror failed: %s", e)



def desired_regime_from_final_signal(final_sig: str) -> str:
    return "IN" if final_sig in ("BUY", "HOLD") else "OUT"


def _read_fernet_key() -> bytes:
    key_path = os.getenv("JUPBOT_FERNET_KEY_PATH", "/etc/myTrading/jupbot.key")
    with open(key_path, "rb") as f:
        return f.read().strip()

def load_keypair() -> Keypair:
    load_dotenv()

    enc = (os.getenv("SOLANA_PRIVATE_KEY_ENC") or "").strip()
    if not enc:
        raise RuntimeError("Missing SOLANA_PRIVATE_KEY_ENC in .env")

    f = Fernet(_read_fernet_key())
    pk_b58 = f.decrypt(enc.encode("utf-8")).decode("utf-8").strip()

    return Keypair.from_base58_string(pk_b58)


# def load_keypair() -> Keypair:
#     load_dotenv()
#     pk = (os.getenv("SOLANA_PRIVATE_KEY_B58") or "").strip()
#     if not pk:
#         raise RuntimeError("Missing SOLANA_PRIVATE_KEY_B58 in .env")
#     return Keypair.from_base58_string(pk)


def fetch_1h_df() -> pd.DataFrame:
    """
    Returns df with columns: High, Low, Close, Volume (and index = timestamps).
    """
    df = yf.download(YF_TICKER, period=f"{LOOKBACK_DAYS}d", interval=INTERVAL, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No yfinance data for {YF_TICKER} {INTERVAL}")

    # Normalize columns (yfinance sometimes returns multiindex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df[["High", "Low", "Close", "Volume"]].dropna()
    if len(df) < 100:
        raise RuntimeError(f"Not enough candles yet: {len(df)}")

    # Make sure timezone-naive like your crypto module does
    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        pass

    return df


def decide_target_sol_pct(final_signal: str) -> float:
    """
    Your rule-set returns: BUY / HOLD / EXIT / STANDDOWN  :contentReference[oaicite:3]{index=3}
    - BUY/HOLD => be in SOL (80/20)
    - EXIT/STANDDOWN => be in USDC (20/80)
    """
    if final_signal in ("BUY", "HOLD"):
        return IN_SOL_PCT
    return OUT_SOL_PCT


def portfolio(rpc_url: str, pubkey: str, sol_price: float) -> dict:
    sol = get_sol_balance(rpc_url, pubkey)
    usdc = get_spl_token_balance_ui(rpc_url, pubkey, USDC_MINT)
    tradable_sol = max(0.0, sol - SOL_FEE_RESERVE)

    sol_val = sol * sol_price
    usdc_val = usdc
    total = sol_val + usdc_val

    sol_pct = (sol_val / total) if total > 0 else 0.0

    return {
        "sol": sol,
        "tradable_sol": tradable_sol,
        "usdc": usdc,
        "sol_val": sol_val,
        "usdc_val": usdc_val,
        "total": total,
        "sol_pct": sol_pct,
    }


def rebalance_plan(port: dict, sol_price: float, target_sol_pct: float) -> dict:
    total = port["total"]
    if total <= 0:
        return {"action": "NONE"}

    desired_sol_val = total * target_sol_pct
    diff = desired_sol_val - port["sol_val"]  # + => buy SOL with USDC, - => sell SOL for USDC

    if abs(diff) < max(USD_TOLERANCE, MIN_SWAP_USD):
        return {"action": "NONE", "usd_diff": diff}

    if diff > 0:
        spend = min(port["usdc"], diff)
        if spend < MIN_SWAP_USD:
            return {"action": "NONE", "usd_diff": diff}
        return {"action": "BUY_SOL", "usdc": spend, "usd_diff": diff}

    # diff < 0 => need to sell SOL
    need_sell_usd = min(port["sol_val"] - desired_sol_val, port["sol_val"])
    sell_sol = min(port["tradable_sol"], need_sell_usd / sol_price)
    if sell_sol * sol_price < MIN_SWAP_USD:
        return {"action": "NONE", "usd_diff": diff}
    return {"action": "SELL_SOL", "sol": sell_sol, "usd_diff": diff}


def execute_plan(*, rpc_url: str, kp: Keypair, pubkey: str, plan: dict) -> dict | None:
    if plan["action"] == "NONE":
        log.info("Rebalance: NONE (within tolerance)")
        return

    if DRY_RUN:
        log.warning("DRY_RUN: would execute plan=%s", plan)
        return

    if plan["action"] == "BUY_SOL":
        usdc_amt = float(plan["usdc"])
        amt_small = to_smallest(usdc_amt, USDC_DECIMALS)
        quote = get_quote(USDC_MINT, WSOL_MINT, amt_small, SLIPPAGE_BPS)
        swap = get_swap_tx(quote, pubkey)
        tx = swap.get("swapTransaction")
        if not tx:
            raise RuntimeError(f"Jupiter swap response missing swapTransaction: {swap}")
        
        if DRY_RUN:
            return {
                "action": "BUY_SOL",
                "amount": usdc_amt,
                "amount_ccy": "USDC",
                "tx_sig": None,
            }
        sig = sign_and_send_swap(rpc_url=rpc_url, swap_tx_b64=tx, keypair=kp)
        log.info("✅ BUY_SOL: spent USDC=%.2f sig=%s", usdc_amt, sig)
        return {
            "action": "BUY_SOL",
            "amount": usdc_amt,
            "amount_ccy": "USDC",
            "tx_sig": sig,
        }

    if plan["action"] == "SELL_SOL":
        sol_amt = float(plan["sol"])
        amt_small = to_smallest(sol_amt, WSOL_DECIMALS)
        quote = get_quote(WSOL_MINT, USDC_MINT, amt_small, SLIPPAGE_BPS)
        swap = get_swap_tx(quote, pubkey)
        tx = swap.get("swapTransaction")
        if not tx:
            raise RuntimeError(f"Jupiter swap response missing swapTransaction: {swap}")
        
        if DRY_RUN:
            return {
                "action": "SELL_SOL",
                "amount": sol_amt,
                "amount_ccy": "SOL",
                "tx_sig": None,
            }
        sig = sign_and_send_swap(rpc_url=rpc_url, swap_tx_b64=tx, keypair=kp)
        log.info("✅ SELL_SOL: sold SOL=%.6f sig=%s", sol_amt, sig)
        return {
            "action": "SELL_SOL",
            "amount": sol_amt,
            "amount_ccy": "SOL",
            "tx_sig": sig,
        }

    raise RuntimeError(f"Unknown plan action: {plan}")


from datetime import datetime

def log_trade(
    *,
    action: str,
    regime_from: str,
    regime_to: str,
    price: float,
    amount: float,
    amount_ccy: str,
    tx_sig: str | None,
    dry_run: bool,
):
    os.makedirs(os.path.dirname(TRADE_LOG_PATH) or ".", exist_ok=True)

    file_exists = os.path.exists(TRADE_LOG_PATH)

    with open(TRADE_LOG_PATH, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write(
                "timestamp,action,regime_from,regime_to,price,amount,amount_ccy,tx_sig,dry_run\n"
            )

        f.write(
            f"{datetime.utcnow().isoformat()},"
            f"{action},"
            f"{regime_from},"
            f"{regime_to},"
            f"{price:.4f},"
            f"{amount:.6f},"
            f"{amount_ccy},"
            f"{tx_sig or ''},"
            f"{dry_run}\n"
        )


def main():
    kp = load_keypair()
    pubkey = str(kp.pubkey())
    log.info("Starting bot. pubkey=%s rpc=%s dry_run=%s", pubkey, RPC_URL, DRY_RUN)

    st = _load_state()

    while True:
        try:
            df = fetch_1h_df()

            # Apply YOUR indicator pipeline (Supertrend, MOST RSI, ADXR state, etc.) :contentReference[oaicite:4]{index=4}
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
            # bar_ts = str(last.name)
            tz = ZoneInfo("America/Los_Angeles") if BOT_TZ.upper() == "PST" else ZoneInfo("UTC")

            bar_ts = (
                pd.Timestamp(last.name)
                .tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
                .tz_convert(tz)
                .strftime("%Y-%m-%d %H:%M:%S")
            )


            # Only act once per new hour bar
            if st.last_bar_ts == bar_ts:
                time.sleep(SLEEP_SECONDS)
                continue

            st_sig = str(last.get("Supertrend_Signal", "SELL"))
            most_sig = str(last.get("MOST_Signal", "SELL"))
            adxr_state = str(last.get("ADXR_State", "FLAT"))

            # Your final rule set (BUY/HOLD/EXIT/STANDDOWN) :contentReference[oaicite:5]{index=5}
            final_sig = signal_super_most_adxr(st_sig, most_sig, adxr_state)

            desired_regime = desired_regime_from_final_signal(final_sig)

            log.info("State: prev_regime=%s desired_regime=%s (FINAL=%s)", st.regime, desired_regime, final_sig)


            price = float(last["Close"])
            # target_sol_pct = decide_target_sol_pct(final_sig)

            # log.info(
            #     "Bar=%s Close=%.2f | ST=%s MOST=%s ADXR=%s => FINAL=%s | Target SOL%%=%.0f%%",
            #     bar_ts, price, st_sig, most_sig, adxr_state, final_sig, target_sol_pct * 100.0
            # )

            # port = portfolio(RPC_URL, pubkey, price)
            # log.info(
            #     "Portfolio: total=$%.2f SOL=%.4f (tradable %.4f) USDC=%.2f SOL%%=%.1f%%",
            #     port["total"], port["sol"], port["tradable_sol"], port["usdc"], port["sol_pct"] * 100.0
            # )

            # plan = rebalance_plan(port, price, target_sol_pct)
            # log.info("Plan: %s", plan)

            # execute_plan(rpc_url=RPC_URL, kp=kp, pubkey=pubkey, plan=plan)
            # ✅ Trade ONLY when regime flips
            if desired_regime != st.regime:
                target_sol_pct = IN_SOL_PCT if desired_regime == "IN" else OUT_SOL_PCT
                log.info("REGIME FLIP %s -> %s. Doing ONE rebalance to SOL%%=%.0f%%", st.regime, desired_regime, target_sol_pct * 100)

                port = portfolio(RPC_URL, pubkey, price)
                plan = rebalance_plan(port, price, target_sol_pct)
                exec_result = execute_plan(rpc_url=RPC_URL, kp=kp, pubkey=pubkey, plan=plan)
                if exec_result:
                    log_trade(
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
                log.info("No regime change (%s). Skipping trading (no rebalancing).", st.regime)

            st.last_bar_ts = bar_ts
            _save_state(st)

        except Exception as e:
            log.exception("Loop error: %s", e)

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
