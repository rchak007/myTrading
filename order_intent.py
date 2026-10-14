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
PRICE_DIRECTIONS = {"ABOVE", "BELOW"}

# Structure triggers. The condition is not a price but an EVENT: the next
# pivot labelled this way, confirmed after the row was typed. hhll.py already
# computes these labels with Break-of-Structure confirmation on a close, and
# since 2026-10-14 also exports when each became knowable.
#
# All eight Side x label combinations are allowed. BUY/HL is the conventional
# one, but BUY/LL (buying capitulation) is a real trade, and Side x condition
# has been deliberately mechanical here since the Action column was split.
STRUCTURE_DIRECTIONS = {"HH", "HL", "LL", "LH"}
DIRECTIONS = PRICE_DIRECTIONS | STRUCTURE_DIRECTIONS

# SEVERAL LABELS ON ONE ROW. Chakravarti, 2026-10-14, reading an SPCX chart:
# *"i should be able to say BUY HL or HH ... so not just BUY HL"*. Right — in
# an uptrend the pullback low AND the breakout high are both entries, and
# which arrives first is not knowable when the row is typed.
#
#   Close_Is: HL HH      fires on whichever is confirmed first
#
# Separated by space, slash or comma. Stored SORTED so "HL HH" and "HH HL" are
# the same row — otherwise re-typing it in the other order changes the
# fingerprint and the engine VOIDs it as INTENT_CHANGED for no real change.
#
# NOT offered: H / L, hhll's labels for the first pivot of each kind. They
# occur exactly ONCE per ticker ever (measured across SPCX, JOBY, NVDA over a
# year: one each), because after the first there is always a previous
# same-kind pivot to compare against. A trigger on them would never fire
# again.
_SPLIT = re.compile(r"[\s/,]+")

FIELDS = ["Row_ID", "Acct", "Ticker", "Side", "Close_Is", "Trigger_Price",
          "Limit_Price", "Qty", "Expires_On"]

# CHAINING USES A HUMAN-OWNED ID SPACE, NOT Row_ID.
#
# The first design had the child name its parent's Row_ID. Chakravarti caught
# why that cannot work, 2026-10-14: *"i want ROW_ID to be left alone since
# only Pi will update."* Row_ID is BLANK when he types the row — Pi 1 stamps
# it on the next cycle — so the value he would have to reference does not
# exist yet. Not a preference; the design was simply impossible to use.
#
#   TRIGGER_ID         a name this row PUBLISHES          "I am 1"
#   AFTER_TRIGGER_ID   a name this row WAITS FOR          "after 1"
#
# Two columns, and here that is right. The earlier objection to two columns
# was that both ends encoding the LINK can disagree — parent claiming it
# triggers row 5 while row 5 waits on row 2. These are not two link
# directions: one DECLARES a name, the other REFERENCES it, like a label and
# a goto. A declaration cannot contradict a reference.
TRIGGER_FIELD = "Trigger_ID"
CHAIN_FIELD = "After_Trigger_ID"
CHAIN_FIELDS = (TRIGGER_FIELD, CHAIN_FIELD)

# Short and typeable: "1", "A", "joby-entry". His own namespace, reusable once
# the chain that used it is finished.
RE_TRIGGER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")

MAX_CHAIN_DEPTH = 3

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

    raw = _s(intent.get("Close_Is"))
    parts = [p for p in _SPLIT.split(raw) if p]
    unknown = [p for p in parts if p not in DIRECTIONS]
    if not parts or unknown:
        raise IntentError(
            f"Close_Is {raw!r} must be ABOVE or BELOW (act when the daily "
            f"close is through Trigger_Price), or one or more of "
            f"{' '.join(sorted(STRUCTURE_DIRECTIONS))} (act when the next "
            f"pivot with any of those labels is confirmed, e.g. 'HL HH')")
    if any(p in PRICE_DIRECTIONS for p in parts):
        if len(parts) > 1:
            raise IntentError(
                f"Close_Is {raw!r} mixes a price condition with something "
                f"else. ABOVE and BELOW stand alone; only the structure "
                f"labels can be combined.")
        out["Close_Is"] = parts[0]
    else:
        # Sorted and de-duplicated: order is not meaning, and leaving it free
        # would make "HL HH" and "HH HL" different fingerprints on the same
        # intent.
        out["Close_Is"] = "/".join(sorted(set(parts)))
    dirn = out["Close_Is"]

    # TRIGGER_PRICE IS REQUIRED FOR A PRICE CONDITION AND OPTIONAL FOR A
    # STRUCTURE ONE. On a structure row it is a BOUND, not the trigger:
    # "buy the next HL, but only if the close is under 6.40". Blank means the
    # structure event alone is enough.
    trig = _f2(intent.get("Trigger_Price"))
    if dirn in PRICE_DIRECTIONS:
        if not trig or float(trig) <= 0:
            raise IntentError("Trigger_Price must be greater than zero")
    elif trig and float(trig) <= 0:
        raise IntentError(
            f"Trigger_Price must be greater than zero when given. On a {dirn} "
            f"row it is OPTIONAL and acts as a price bound, not the trigger")
    out["Trigger_Price"] = trig

    out["Limit_Price"] = _f2(intent.get("Limit_Price"))
    if out["Limit_Price"] and float(out["Limit_Price"]) <= 0:
        raise IntentError("Limit_Price must be greater than zero when given")

    # QTY MAY BE BLANK ON A CHAINED ROW, meaning "whatever the parent actually
    # filled". Chakravarti, 2026-10-14: *"yes thats fine - but i could fill it
    # too. but if its blank sure use parent qty."*
    #
    # It earns its place: a parent that partially fills 31 of 50 would
    # otherwise need the child edited by hand to 31 — and editing a live row
    # is precisely what INTENT_CHANGED voids. Blank inherits, so the partial
    # case needs no intervention.
    mine = _s(intent.get(TRIGGER_FIELD), upper=False)
    if mine and not RE_TRIGGER_ID.match(mine):
        raise IntentError(
            f"{TRIGGER_FIELD} {mine!r} must be 1-32 chars of letters, digits, "
            f". _ or -, starting alphanumeric — your own label for this row, "
            f"e.g. 1")
    out[TRIGGER_FIELD] = mine

    after = _s(intent.get(CHAIN_FIELD), upper=False)
    if after and not RE_TRIGGER_ID.match(after):
        raise IntentError(
            f"{CHAIN_FIELD} {after!r} must be the {TRIGGER_FIELD} of the row "
            f"that has to fill first, or blank")
    out[CHAIN_FIELD] = after

    qty = _f2(intent.get("Qty"))
    if not qty:
        if not after:
            raise IntentError(
                "Qty must be greater than zero — it may only be left blank on "
                f"a chained row, where it inherits the parent's filled "
                f"quantity ({CHAIN_FIELD} is empty here)")
    elif float(qty) <= 0:
        raise IntentError("Qty must be greater than zero")
    out["Qty"] = qty

    if after and mine and after == mine:
        raise IntentError(
            f"{CHAIN_FIELD} {after!r} is this row's own {TRIGGER_FIELD} — a "
            f"row cannot wait for itself")

    exp = _s(intent.get("Expires_On"), upper=False)
    if exp and not RE_DATE.match(exp):
        raise IntentError(f"Expires_On {exp!r} must be YYYY-MM-DD or blank")
    out["Expires_On"] = exp

    return out


def wanted_labels(close_is: str) -> set:
    """The structure labels a Close_Is accepts. Empty for a price condition.

    One place to split, so the engine never has to know that several labels
    are stored slash-joined.
    """
    s = _s(close_is)
    if s in PRICE_DIRECTIONS:
        return set()
    return {p for p in _SPLIT.split(s) if p in STRUCTURE_DIRECTIONS}


def chain_problems(rows: list[dict], done_ids=frozenset(),
                   max_depth: int = MAX_CHAIN_DEPTH) -> dict:
    """{Row_ID: complaint} for every row whose chain is unusable.

    Checked across the WHOLE tab rather than per row, because a duplicate
    label, a loop and an over-long chain are all properties of the graph, not
    of any single row.

    `done_ids` are Row_IDs already in a terminal state. A Trigger_ID belonging
    to one of them is FREE AGAIN — his words: *"if i give trigger_id value
    used elsewhere in orders not completed yet you can give error"*. Reusing
    "1" next month is normal; reusing it while the first "1" is still live is
    the mistake.

    Catches, in this order:
      - a Trigger_ID used by two rows that are both still live
      - an After_Trigger_ID naming a label no live row publishes
      - a cycle (1 waits for 2 waits for 1) — otherwise rows that look ARMED
        forever and never fire
      - a chain deeper than max_depth

    DEPTH IS REPORTED, NOT SILENTLY DROPPED. Chakravarti asked for the row to
    be highlighted RED so he can see it and restructure, rather than having it
    quietly refuse to act.
    """
    out: dict = {}

    # Label -> Row_ID, live rows only.
    publishes: dict = {}
    for r in rows:
        rid = _s(r.get("Row_ID"), upper=False)
        lbl = _s(r.get(TRIGGER_FIELD), upper=False)
        if not lbl or (rid and rid in done_ids):
            continue
        if lbl in publishes:
            out[rid or lbl] = (
                f"TRIGGER_ID_IN_USE: {lbl!r} is already the {TRIGGER_FIELD} of "
                f"a row that has not finished. Pick another, or wait for that "
                f"one to complete.")
            continue
        publishes[lbl] = rid

    # Row_ID -> the LABEL it waits for; and Row_ID -> its own label.
    waits, label_of, by_id = {}, {}, {}
    for r in rows:
        rid = _s(r.get("Row_ID"), upper=False)
        if not rid:
            continue
        by_id[rid] = r
        label_of[rid] = _s(r.get(TRIGGER_FIELD), upper=False)
        waits[rid] = _s(r.get(CHAIN_FIELD), upper=False)

    for rid, want in waits.items():
        if not want or rid in out or rid in done_ids:
            continue
        parent = publishes.get(want)
        if not parent:
            out[rid] = (f"CHAIN_BROKEN: {CHAIN_FIELD} {want!r} — no live row "
                        f"on this tab has that {TRIGGER_FIELD}")
            continue
        # DEPTH IS COUNTED IN ROWS, NOT LINKS. A->B->C is three rows and is
        # allowed at a cap of 3; a fourth is not. Counting links let a 4-row
        # chain through a cap of 3, which is not what "depth 3" means to
        # anyone reading the tab.
        seen, cur, rows_in_chain = {rid}, parent, 1
        while cur:
            if cur in seen:
                out[rid] = (f"CHAIN_LOOP: following {CHAIN_FIELD} comes back "
                            f"to {label_of.get(cur) or cur!r} — this chain can "
                            f"never fire")
                break
            seen.add(cur)
            rows_in_chain += 1
            if rows_in_chain > max_depth:
                out[rid] = (f"CHAIN_TOO_DEEP: {rows_in_chain}+ rows linked by "
                            f"{CHAIN_FIELD}, limit is {max_depth}. Far more "
                            f"likely a paste error than an intention — split "
                            f"it or re-plan.")
                break
            cur = publishes.get(waits.get(cur, ""))
    return out


def parent_of(rows: list[dict], done_ids=frozenset()) -> dict:
    """{Row_ID: parent Row_ID} — After_Trigger_ID resolved to a real row.

    The human columns are labels; everything downstream (the ledger, fill
    lookup, cascade) works in Row_IDs. This is the one place the two
    namespaces meet.
    """
    publishes = {}
    for r in rows:
        rid = _s(r.get("Row_ID"), upper=False)
        lbl = _s(r.get(TRIGGER_FIELD), upper=False)
        if lbl and rid and rid not in done_ids and lbl not in publishes:
            publishes[lbl] = rid
    out = {}
    for r in rows:
        rid = _s(r.get("Row_ID"), upper=False)
        want = _s(r.get(CHAIN_FIELD), upper=False)
        if rid and want and want in publishes:
            out[rid] = publishes[want]
    return out


def fingerprint(intent: dict) -> str:
    """A short stable id for this exact intent.

    Not a security control — there is no secret in it. It is how the engine
    notices that a row it has already acted on has since been EDITED, so an
    executed row cannot quietly become a second, different order.
    """
    n = normalize(intent)
    canon = "|".join(n[f] for f in FIELDS)
    # APPENDED ONLY WHEN SET, so an unchained row hashes exactly as it did
    # before the chain columns existed. Adding them to FIELDS would have changed
    # every fingerprint on the tab, and the engine reads a changed
    # fingerprint as "this row was edited after I acted on it" — every live
    # row would have VOIDed itself with INTENT_CHANGED on the first run after
    # deploy.
    for f in CHAIN_FIELDS:
        if n.get(f):
            canon += f"|{f}={n[f]}"
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
    lim = (f", GTC limit ${n['Limit_Price']}" if n["Limit_Price"]
           else ", NO LIMIT PRICE — a GTC order needs one")
    exp = f", expires {n['Expires_On']}" if n["Expires_On"] else ", no expiry"
    return (f"{n['Side']} {qty} sh of {n['Ticker']} in {n['Acct']} when the "
            f"DAILY CLOSE is {n['Close_Is']} ${n['Trigger_Price']}{lim}{exp}")
