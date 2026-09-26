#!/usr/bin/env python3
"""
order_intent.py
===============
Validate and normalise one order row from the sheet.

NO SIGNING. Decided 2026-09-26: Chakravarti types intents straight into the
Orders tab from a phone, and requiring a laptop to mint an HMAC defeats the
purpose of having a sheet at all.

    The risk that buys: someone with edit access to the spreadsheet can cause
    TRADES. They cannot move money out — a brokerage order can only ever buy or
    sell inside the account. Bounded, and a deliberate call.

    What replaces the token: guards that understand the actual account (see
    order_engine.check_guards) — hard caps on notional, a refusal to sell more
    than is held, a refusal to trade a ticker that is not held or watched, and
    a validation line written back into the sheet on every cycle so a bad row
    announces itself long before the close.

    The realistic failure here is not a break-in, it is a mis-typed cell: a
    formula autofilling down, 10 becoming 100, a paste landing one row off.
    Everything below exists to catch that.

`Action` encodes side AND direction (see ACTIONS), so the sheet needs no
separate Side / Trigger_Type / Order_Type / TIF columns. Fewer cells to mistype
is a safety property, not a shortcut.
"""
from __future__ import annotations

import re
from hashlib import sha256

# Action -> (side, trigger direction). This is the whole reason the sheet needs
# fewer columns than the design doc assumed.
ACTIONS = {
    "SELL-TRIM":     ("SELL", "CLOSE_ABOVE"),   # take profit into strength
    "SELL-STOPLOSS": ("SELL", "CLOSE_BELOW"),   # protection
    "BUY-DIP":       ("BUY",  "CLOSE_BELOW"),   # buy the pullback
    "BUY-BREAKOUT":  ("BUY",  "CLOSE_ABOVE"),   # buy strength
}

QTY_UNITS = {"SHARES", "PCT_POS", "USD"}

FIELDS = ["Row_ID", "Acct", "Ticker", "Action", "Trigger_Price",
          "Limit_Price", "Qty", "Qty_Unit", "After_Close", "Expires_On"]

# Row_ID is the replay key, so it must be a stable, typeable identity.
RE_ROW_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$")
RE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class IntentError(ValueError):
    """The intent is malformed. Never execute a row that raises this."""


def _f2(v) -> str:
    """Float to exactly 2 decimals, or an empty segment.

    Both sides must agree byte for byte, so this is the only place a number
    becomes text. 245 and 245.0 and "245.00" must all render identically or an
    armed row stops verifying for reasons nobody can see.
    """
    if v is None or str(v).strip() in ("", "-"):
        return ""
    try:
        return f"{float(str(v).replace(',', '').replace('$', '').strip()):.2f}"
    except (TypeError, ValueError) as e:
        raise IntentError(f"not a number: {v!r}") from e


def _s(v, upper: bool = True) -> str:
    s = "" if v is None else str(v).strip()
    return s.upper() if upper else s


def normalize(intent: dict) -> dict:
    """Validate and coerce an intent. Raises IntentError on anything unusable.

    Runs before both token minting and verification, so a row that cannot be
    armed also cannot be executed — the two paths cannot disagree about what
    counts as valid.
    """
    out = {}

    out["Row_ID"] = _s(intent.get("Row_ID"), upper=False)
    if not RE_ROW_ID.match(out["Row_ID"]):
        raise IntentError(
            f"Row_ID {out['Row_ID']!r} must be 3-64 chars of letters, digits, "
            f". _ or -, starting alphanumeric (e.g. 2026-10-14-AVGO-01)")

    acct = _s(intent.get("Acct"))
    if not re.fullmatch(r"\d{3}", acct):
        raise IntentError(f"Acct {acct!r} must be the LAST 3 DIGITS, e.g. 431")
    out["Acct"] = acct

    tkr = _s(intent.get("Ticker"))
    if not re.fullmatch(r"[A-Z][A-Z0-9.]{0,9}", tkr):
        raise IntentError(f"Ticker {tkr!r} is not a plausible symbol")
    out["Ticker"] = tkr

    action = _s(intent.get("Action"))
    if action not in ACTIONS:
        raise IntentError(f"Action {action!r} must be one of {sorted(ACTIONS)}")
    out["Action"] = action

    trig = _f2(intent.get("Trigger_Price"))
    if not trig or float(trig) <= 0:
        raise IntentError("Trigger_Price must be greater than zero")
    out["Trigger_Price"] = trig

    out["Limit_Price"] = _f2(intent.get("Limit_Price"))
    if out["Limit_Price"] and float(out["Limit_Price"]) <= 0:
        raise IntentError("Limit_Price must be greater than zero when given")

    qty = _f2(intent.get("Qty"))
    if not qty or float(qty) <= 0:
        raise IntentError("Qty must be greater than zero")
    out["Qty"] = qty

    unit = _s(intent.get("Qty_Unit")) or "SHARES"
    if unit not in QTY_UNITS:
        raise IntentError(f"Qty_Unit {unit!r} must be one of {sorted(QTY_UNITS)}")
    out["Qty_Unit"] = unit

    ac = _s(intent.get("After_Close")) or "N"
    if ac not in ("Y", "N"):
        raise IntentError("After_Close must be Y or N")
    out["After_Close"] = ac

    exp = _s(intent.get("Expires_On"), upper=False)
    if exp and not RE_DATE.match(exp):
        raise IntentError(f"Expires_On {exp!r} must be YYYY-MM-DD or blank")
    out["Expires_On"] = exp

    # A SELL-STOPLOSS that waits for Pi 1 to notice is a contradiction: the one
    # order whose job is protecting you must not depend on a Raspberry Pi being
    # alive. Sheet design §7 states this; refusing it here makes it true.
    if out["Action"] == "SELL-STOPLOSS" and out["After_Close"] == "Y":
        raise IntentError(
            "SELL-STOPLOSS must have After_Close = N so the order rests at "
            "Schwab. A watched stop stops protecting you the moment Pi 1 dies.")

    return out


def fingerprint(intent: dict) -> str:
    """A short stable id for this exact intent.

    Not a security control — there is no secret in it. It is how the engine
    notices that a row it has already acted on has since been EDITED, so an
    executed row cannot quietly become a second, different order.
    """
    n = normalize(intent)
    canon = "|".join(n[f] for f in FIELDS)
    return sha256(canon.encode("utf-8")).hexdigest()[:12]


def idempotency_key(row_id: str, fp: str) -> str:
    """Deterministic key recorded BEFORE a submit is attempted.

    Derived rather than random so a crash between write-ahead and submit can be
    reconciled: the same intent always produces the same key.
    """
    return sha256(f"{row_id}{fp}".encode("utf-8")).hexdigest()[:16]


def describe(intent: dict) -> str:
    """One line a human can check against what they meant."""
    n = normalize(intent)
    side, direction = ACTIONS[n["Action"]]
    unit = {"SHARES": "sh", "PCT_POS": "% of position", "USD": "USD"}[n["Qty_Unit"]]
    when = ("when the DAILY CLOSE is "
            + ("above " if direction == "CLOSE_ABOVE" else "below ")
            + f"${n['Trigger_Price']}"
            ) if n["After_Close"] == "Y" else (
        f"resting at Schwab, trigger ${n['Trigger_Price']}")
    lim = f", limit ${n['Limit_Price']}" if n["Limit_Price"] else ", NO LIMIT PRICE"
    exp = f", expires {n['Expires_On']}" if n["Expires_On"] else ", NO EXPIRY"
    return f"{side} {n['Qty']} {unit} of {n['Ticker']} in {n['Acct']} — {when}{lim}{exp}"
