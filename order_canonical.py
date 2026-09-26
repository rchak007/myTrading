#!/usr/bin/env python3
"""
order_canonical.py
==================
The canonical string for an order intent, and the HMAC token over it.

Imported by BOTH `arm_order.py` (which mints tokens) and `order_engine.py`
(which verifies them). One implementation on purpose: two would drift, and
every row would fail verification with no obvious cause.

WHAT THE TOKEN IS FOR
    The sheet is untrusted input. Anyone — or anything — with edit access to the
    spreadsheet could otherwise type a row that spends money. The token proves
    an intent was authored by someone holding the secret, which lives only on
    Pi 1 and the Dell, never in git, never in the sheet, never in a cloud secret
    store.

    Change ANY intent cell after arming and the token no longer matches, so the
    row is refused rather than executed. That is the point: a typo, a stray
    paste, or a compromised sheet cannot become an order.

    See Documentation/orderExecutionDesign-9-7-26.md §6.2.

CANONICAL FORM
    Pipe-delimited, fixed field order, taken from the columns the Orders tab
    actually has:

        Row_ID|Acct|Ticker|Action|Trigger_Price|Limit_Price|Qty|Qty_Unit|After_Close|Expires_On

    The design doc lists separate Side / Trigger_Type / Bar / Order_Type / TIF
    columns. Our sheet does not need them: `Action` already encodes side AND
    direction (see ACTIONS below), `Limit_Price` is given outright rather than
    derived from an offset, `Bar` is DAILY in v1, and TIF follows from
    After_Close. Fewer cells to mistype is a safety property, not a shortcut.

    Formatting rules, which both sides must apply identically:
      * strings stripped and uppercased, EXCEPT Row_ID and Expires_On
      * floats to exactly 2 decimals
      * an empty optional field renders as an empty segment
"""
from __future__ import annotations

import hmac
import os
import re
from hashlib import sha256
from pathlib import Path

# Where the shared secret lives. 32 random bytes, mode 0400, root-owned.
#   sudo sh -c 'head -c 32 /dev/urandom | base64 > /etc/myTrading/order_hmac.key'
#   sudo chmod 0400 /etc/myTrading/order_hmac.key
HMAC_KEY_FILE = Path(os.getenv("ORDER_HMAC_KEY", "/etc/myTrading/order_hmac.key"))

TOKEN_LEN = 12          # hex chars kept; 48 bits is ample against typing attacks

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


def canonical(intent: dict) -> str:
    """The exact string the token is computed over."""
    n = normalize(intent)
    return "|".join(n[f] for f in FIELDS)


def load_secret(path: Path | None = None) -> bytes:
    """Read the shared secret, refusing a world-readable one.

    Fails closed and loudly: a missing or sloppy key must stop the engine, not
    silently downgrade it to trusting the sheet.
    """
    p = path or HMAC_KEY_FILE
    if not p.exists():
        raise IntentError(
            f"HMAC key not found at {p}. Create it once:\n"
            f"  sudo mkdir -p {p.parent}\n"
            f"  sudo sh -c 'head -c 32 /dev/urandom | base64 > {p}'\n"
            f"  sudo chmod 0400 {p}")
    mode = p.stat().st_mode & 0o077
    if mode:
        raise IntentError(
            f"{p} is readable by group or others (mode {oct(p.stat().st_mode)[-3:]}). "
            f"chmod 0400 it — a shared secret anyone can read is not a secret.")
    data = p.read_bytes().strip()
    if len(data) < 16:
        raise IntentError(f"{p} is too short to be a 32-byte key")
    return data


def token_for(intent: dict, secret: bytes | None = None) -> str:
    """The Confirm_Token for this intent."""
    s = secret if secret is not None else load_secret()
    mac = hmac.new(s, canonical(intent).encode("utf-8"), sha256)
    return mac.hexdigest()[:TOKEN_LEN]


def verify(intent: dict, supplied: str, secret: bytes | None = None) -> bool:
    """Constant-time check that `supplied` matches this intent.

    compare_digest rather than ==: a timing side channel here is far-fetched,
    but the cost of using it is nil and the habit is worth keeping.
    """
    want = token_for(intent, secret)
    got = _s(supplied, upper=False).lower()
    return hmac.compare_digest(want, got)


def idempotency_key(row_id: str, token: str) -> str:
    """Deterministic key recorded BEFORE a submit is attempted (§6.4).

    Derived rather than random so a crash between write-ahead and submit can be
    reconciled: the same intent always produces the same key.
    """
    return sha256(f"{row_id}{token}".encode("utf-8")).hexdigest()[:16]


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
