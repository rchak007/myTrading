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

THE SHEET CARRIES ONLY CLOSE-TRIGGERED ORDERS. Anything Schwab can already
express — a limit to buy a dip, a limit to trim into strength — belongs at
Schwab, where it does not depend on a Raspberry Pi being awake. Chakravarti's
call 2026-09-26, and it matches the sheet design's own rule: use the watched
path only where the condition cannot be expressed as a resting order.

So every row here means the same shape of thing:

    <SIDE> <QTY> of <TICKER> in <ACCT> when the DAILY CLOSE is <ABOVE|BELOW> <PRICE>

which is why there is no After_Close column any more — it would be Y on every
row — and no Order_Type or TIF.
"""
from __future__ import annotations

import re
from hashlib import sha256

# Side and direction are separate columns now, replacing the old single Action
# with its four hyphenated names. Chakravarti's call 2026-09-26: BUY/SELL x
# ABOVE/BELOW is mechanical and unambiguous, where "SELL-TRIM" carried an
# implied motive that did not always match what the row was for.
SIDES = {"BUY", "SELL"}
DIRECTIONS = {"ABOVE", "BELOW"}

FIELDS = ["Row_ID", "Acct", "Ticker", "Side", "Close_Is", "Trigger_Price",
          "Limit_Price", "Qty", "Expires_On"]

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

    side = _s(intent.get("Side"))
    if side not in SIDES:
        raise IntentError(f"Side {side!r} must be BUY or SELL")
    out["Side"] = side

    dirn = _s(intent.get("Close_Is"))
    if dirn not in DIRECTIONS:
        raise IntentError(
            f"Close_Is {dirn!r} must be ABOVE or BELOW — the condition is "
            f"'act when the daily close is ABOVE/BELOW Trigger_Price'")
    out["Close_Is"] = dirn

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

    exp = _s(intent.get("Expires_On"), upper=False)
    if exp and not RE_DATE.match(exp):
        raise IntentError(f"Expires_On {exp!r} must be YYYY-MM-DD or blank")
    out["Expires_On"] = exp

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
    qty = f"{float(n['Qty']):g}"
    lim = (f", limit ${n['Limit_Price']}" if n["Limit_Price"]
           else ", MARKET (fills at the open, whatever it is)")
    exp = f", expires {n['Expires_On']}" if n["Expires_On"] else ", no expiry"
    return (f"{n['Side']} {qty} sh of {n['Ticker']} in {n['Acct']} when the "
            f"DAILY CLOSE is {n['Close_Is']} ${n['Trigger_Price']}{lim}{exp}")
