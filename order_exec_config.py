#!/usr/bin/env python3
"""
order_exec_config.py
====================
What the order engine will and will not do.

DOLLAR CAPS ARE OFF by default — Chakravarti's call, 2026-09-26. The limits
that matter here are the account's own and are enforced in
order_engine.check_guards: a sell cannot exceed the shares held, a buy cannot
exceed the free cash. Those scale with the portfolio and catch a mis-typed
quantity regardless of its dollar value, which a fixed ceiling does not.

Everything here is CODE, not anything the sheet can reach, so a cap can never
be raised by editing a spreadsheet. Several values read an env var, which means
a ceiling can be imposed for a while — during a change, say — without editing
code.
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
# OFF by default, at Chakravarti's request 2026-09-26. A fixed dollar ceiling
# would block legitimate trades in a portfolio this size, and it is not what
# actually catches a mis-typed cell.
#
# The real limits are the account's own, enforced in order_engine.check_guards
# and impossible to exceed:
#     a SELL cannot exceed the shares actually held in that account
#     a BUY cannot exceed that account's free cash
# Those scale with the portfolio and catch "1000 instead of 100" regardless of
# the dollar amount, which a fixed ceiling does not.
#
# The machinery stays wired so a ceiling can be imposed from .env at any time
# without a code change — e.g. while testing something new:
#     MAX_NOTIONAL_PER_ORDER=250
MAX_NOTIONAL_PER_ORDER = float(os.getenv("MAX_NOTIONAL_PER_ORDER", "inf"))
MAX_NOTIONAL_PER_DAY = float(os.getenv("MAX_NOTIONAL_PER_DAY", "inf"))

# Also uncapped. Note what this one was catching that nothing else does: a
# RUNAWAY LOOP — the engine re-firing the same intent because of a bug rather
# than a typo. The ledger guards that already (a terminal Row_ID is never
# re-evaluated), so this is belt to that brace, not the only brace.
MAX_SUBMISSIONS_PER_DAY = int(os.getenv("MAX_SUBMISSIONS_PER_DAY", "0")) or 10**9

# Not a money limit — a sanity stop. If the sheet ever holds this many intent
# rows something has gone wrong with it (a formula filled down, a bad paste),
# and the engine should stop rather than work through them. Generous on
# purpose: it should never fire during normal use.
MAX_ARMED_ROWS = int(os.getenv("MAX_ARMED_ROWS", "200"))

# A blank Expires_On already means "no expiry", so this only bounds a date you
# typed deliberately. Generous rather than absent: a 2099 expiry is a typo, and
# refusing it is worth more than honouring it.
MAX_EXPIRY_DAYS = int(os.getenv("MAX_EXPIRY_DAYS", "3650"))

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
    def _cap(v, money=True):
        if v == float("inf") or v >= 10**9:
            return "no limit"
        return f"${v:,.2f}" if money else f"{v:g}"

    return (
        f"{mode}{ks}\n"
        f"  per order   {_cap(MAX_NOTIONAL_PER_ORDER)}\n"
        f"  per day     {_cap(MAX_NOTIONAL_PER_DAY)}, submissions "
        f"{_cap(MAX_SUBMISSIONS_PER_DAY, money=False)}\n"
        f"  real limit  shares held (sells) · free cash (buys)\n"
        f"  accounts    {sorted(ACCOUNT_ALLOWLIST) or 'NONE — every row will be blocked'}\n"
        f"  tickers     {sorted(TICKER_ALLOWLIST) or 'any in STOCK_TICKERS'}\n"
        f"  market ord  {'allowed' if ALLOW_MARKET_ORDERS else 'refused'}"
    )
