#!/usr/bin/env python3
"""
order_exec_config.py
====================
Every limit the order engine will not exceed.

Deliberately CODE, not config the sheet can reach. The sheet is the untrusted
side of this system; a cap that could be raised from the sheet is not a cap.
Changing anything here means an SSH to Pi 1 and a deliberate edit, which is the
friction we want in front of real money.

Start small. These are not the numbers you will end on — they are the numbers
that make the first live weeks survivable if something is wrong.
"""
from __future__ import annotations

import os
from pathlib import Path

# ─────────────────────────────────────────────────────── kill switch
# Presence of this file blocks every submission. Removing it resumes on the
# next cycle. Nothing else is needed for the panic case.
KILL_SWITCH = Path(os.getenv("TRADING_DISABLED_FILE",
                             "/etc/myTrading/TRADING_DISABLED"))

# Master arm. False means evaluate, log, write the sheet — and never call
# Schwab. Flip only after a full cycle of dry runs reads correctly.
LIVE_TRADING = os.getenv("ORDER_ENGINE_LIVE", "0") == "1"

# ─────────────────────────────────────────────────────── caps
# Per ORDER. $250 buys roughly one share of several holdings, which is enough
# to prove the pipeline end to end while capping a bug at the price of a
# cheap lesson.
MAX_NOTIONAL_PER_ORDER = float(os.getenv("MAX_NOTIONAL_PER_ORDER", "250"))

# Per DAY, across every row. A loop that fires repeatedly hits this and stops.
MAX_NOTIONAL_PER_DAY = float(os.getenv("MAX_NOTIONAL_PER_DAY", "500"))

# Count cap as well as dollar cap: many tiny wrong orders are also a failure.
MAX_SUBMISSIONS_PER_DAY = int(os.getenv("MAX_SUBMISSIONS_PER_DAY", "3"))

# A sheet with 200 armed rows is a sheet nobody is reading.
MAX_ARMED_ROWS = 25

# No row lives forever. An intent from eight months ago is about a different
# market than the one it will fire into.
MAX_EXPIRY_DAYS = 90

# A market order on a thin open after an overnight gap is exactly how
# "buy above 245" becomes a fill at 261.
ALLOW_MARKET_ORDERS = False

# If the price has already run this far past the limit in the adverse
# direction, the setup that was intended no longer exists. Block and let a
# human look rather than chase it.
GAP_THROUGH_PCT = 5.0

# Schwab's close and our own signals CSV must agree within this, or we do not
# trust either enough to trade on. Disagreement means one source is stale.
PRICE_DISAGREEMENT_PCT = 2.0

# ─────────────────────────────────────────────────────── allowlists
# Empty ACCOUNTS means "none" — fail closed. Fill it in deliberately with the
# accounts you have CONFIRMED accept order placement through the developer app.
# Several of the custodial/PCRA accounts may not.
ACCOUNT_ALLOWLIST: set[str] = set(
    a.strip() for a in os.getenv("ORDER_ACCOUNT_ALLOWLIST", "").split(",")
    if a.strip())

# Empty TICKERS means "any ticker you hold or watch" — see order_engine, which
# still requires the symbol to be in STOCK_TICKERS. Narrow this while testing.
TICKER_ALLOWLIST: set[str] = set(
    t.strip().upper() for t in os.getenv("ORDER_TICKER_ALLOWLIST", "").split(",")
    if t.strip())

# ─────────────────────────────────────────────────────── timing
# US equities close 13:00 PT. Evaluating before the bar is final reads a
# partial candle; 15 minutes lets the data source settle.
EVALUATE_AFTER_PT = (13, 15)

# ─────────────────────────────────────────────────────── state
STATE_DIR = Path(os.getenv("REMOTE_OPS_STATE",
                           Path.home() / ".local" / "state" / "myTrading"))
LEDGER = STATE_DIR / "order_ledger.csv"       # append-only, the real record
AUDIT = STATE_DIR / "order_audit.log"         # one JSON object per line

# States a row can hold. Terminal ones are never re-evaluated.
LIVE_STATES = {"ARMED", "TRIGGERED", "SUBMITTED", "BLOCKED"}
TERMINAL_STATES = {"FILLED", "CANCELLED", "EXPIRED", "VOID", "REJECTED"}


def kill_switch_on() -> bool:
    return KILL_SWITCH.exists()


def summary() -> str:
    """What the engine will and will not do, in one block. Printed on every
    run so a surprising cap is visible before it bites, not after."""
    mode = "🔴 LIVE — orders WILL be placed" if LIVE_TRADING else \
           "🟢 DRY RUN — nothing will be submitted"
    ks = " · ⛔ KILL SWITCH IS ON" if kill_switch_on() else ""
    return (
        f"{mode}{ks}\n"
        f"  per order   ${MAX_NOTIONAL_PER_ORDER:,.2f}\n"
        f"  per day     ${MAX_NOTIONAL_PER_DAY:,.2f}, max "
        f"{MAX_SUBMISSIONS_PER_DAY} submission(s)\n"
        f"  accounts    {sorted(ACCOUNT_ALLOWLIST) or 'NONE — every row will be blocked'}\n"
        f"  tickers     {sorted(TICKER_ALLOWLIST) or 'any in STOCK_TICKERS'}\n"
        f"  market ord  {'allowed' if ALLOW_MARKET_ORDERS else 'refused'}"
    )
