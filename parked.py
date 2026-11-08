#!/usr/bin/env python3
"""
parked.py
=========
Reserved money that is sitting in a stock instead of in cash.

Chakravarti, 2026-11-08: *"I'll trim some stocks or sell some & reserve it
but that money instead of cash I'll buy SPY QQQ or even NVDA in some cases.
So example I sell BE $1000 then if it's fenced comes into SeedReserved. Then
I'll buy SPY with that money."*

So BE's reserve still exists — it is just not cash any more. It is two
shares of SPY, held on BE's behalf until BE is worth buying again.

WHY A PARCEL LEDGER AND NOT COLUMNS PER VEHICLE. The obvious shape is
Seed_Reserved_SPY, Seed_Reserved_QQQ, Seed_Reserved_NVDA... and he saw the
problem himself while proposing it: *"sorry gets complicated"*. Two columns
per vehicle means every new vehicle is a code change AND a sheet change.
A parcel — (account, owner, vehicle, qty) — scales to any vehicle for free,
and the display renders them into one cell:

    BE  row    Parked        "2 SPY"          or  "2 SPY · 3 QQQ"
    SPY row    Reserved_For  "2 BE · 3 MU"    Owed_Qty 5

QUANTITY, NOT DOLLARS. His call, and the right one: *"i think
seedreservedSPY for BE will be SPY qty - so we know how BE we can buy later
- its whatever price SPY is at that point"*. $1,000 parked in SPY is not
$1,000 later. Recording shares makes the reserve mark-to-market, which is
honest; recording dollars would pretend the parking was free of risk. The
cash at the time IS kept, but only as a record of what went in.

THE PARCEL IS A FICTION THE BROKER DOES NOT SHARE. Schwab sells SHARES, not
parcels. If 431 holds 12 SPY of which 5 are parked, nothing at Schwab stops
a sale of all 12 — not a stop this system recommends, and not an order typed
straight into ThinkOrSwim. So `free_qty()` exists to be checked BEFORE a
sell, and `breaches()` exists to shout AFTER one, the same way the reserve
ledger already reports a buy that overdrew its fence. Prevention cannot be
complete here; detection has to cover the rest.

APPEND-ONLY, LIKE THE RESERVE LEDGER AND THE NOTES. PARK and UNPARK are
rows; the current state is a fold of the log. Nothing is ever overwritten,
so "how much did BE have parked in August" stays answerable.
"""
from __future__ import annotations

import csv
import fcntl
import os
import uuid
from datetime import datetime
from pathlib import Path

try:
    import pytz
    _PST = pytz.timezone("America/Los_Angeles")
except Exception:                                    # pragma: no cover
    _PST = None

STATE_DIR = Path(os.environ.get(
    "MYTRADING_STATE_DIR", Path.home() / ".local" / "state" / "myTrading"))
PARKED_PATH = STATE_DIR / "parked.csv"

COLS = ["Timestamp_PST", "Event_ID", "Account", "Owner", "Vehicle",
        "Qty", "Cash_At_Park", "Event", "Ref", "Reason"]

EVENTS = {"PARK", "UNPARK"}
_EPS = 1e-6


def _now() -> str:
    if _PST is not None:
        return datetime.now(_PST).strftime("%Y-%m-%d %H:%M:%S %Z")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def acct(a) -> str:
    return str(a or "").strip()[-3:]


def tkr(t) -> str:
    return str(t or "").strip().upper()


def read_log(path: Path = PARKED_PATH) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            return [r for r in csv.DictReader(fh) if r.get("Owner")]
    except Exception:                                    # noqa: BLE001
        return []


def _append(account, owner, vehicle, qty, event, *, cash=None, ref="",
            reason="", path: Path = PARKED_PATH, log=print) -> None:
    if event not in EVENTS:
        raise ValueError(f"unknown event {event!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            w = csv.DictWriter(fh, fieldnames=COLS)
            if new:
                w.writeheader()
            w.writerow({
                "Timestamp_PST": _now(),
                "Event_ID": uuid.uuid4().hex[:12],
                "Account": acct(account),
                "Owner": tkr(owner),
                "Vehicle": tkr(vehicle),
                "Qty": f"{float(qty):g}",
                "Cash_At_Park": "" if cash is None else f"{float(cash):.2f}",
                "Event": event,
                "Ref": ref or f"op:{uuid.uuid4().hex[:8]}",
                "Reason": reason,
            })
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    log(f"parked: {event} {acct(account)} {tkr(owner)} "
        f"{float(qty):g} {tkr(vehicle)}")


# ---------------------------------------------------------------------
# The fold
# ---------------------------------------------------------------------
def fold(path: Path = PARKED_PATH) -> dict[tuple[str, str, str], float]:
    """{(account, owner, vehicle): qty} for everything still parked.

    Replayed in file order, which is chronological because the file is only
    appended to. A parcel folded back to zero is dropped rather than kept as
    a zero row — "no longer parked" and "parked nothing" are the same state
    and should not read differently on the sheet.
    """
    out: dict[tuple[str, str, str], float] = {}
    for r in read_log(path):
        key = (acct(r.get("Account")), tkr(r.get("Owner")),
               tkr(r.get("Vehicle")))
        try:
            q = float(r.get("Qty") or 0)
        except (TypeError, ValueError):
            continue
        ev = str(r.get("Event", "")).strip().upper()
        if ev == "PARK":
            out[key] = out.get(key, 0.0) + q
        elif ev == "UNPARK":
            out[key] = out.get(key, 0.0) - q
        if abs(out.get(key, 0.0)) < _EPS:
            out.pop(key, None)
    return out


def outstanding(account, owner, vehicle, path: Path = PARKED_PATH) -> float:
    return fold(path).get((acct(account), tkr(owner), tkr(vehicle)), 0.0)


def for_owner(account, owner, path: Path = PARKED_PATH) -> str:
    """'2 SPY · 3 QQQ' — what this ticker's reserve is sitting in."""
    bits = [(v, q) for (a, o, v), q in fold(path).items()
            if a == acct(account) and o == tkr(owner) and q > _EPS]
    return " · ".join(f"{q:g} {v}" for v, q in sorted(bits))


def for_vehicle(account, vehicle, path: Path = PARKED_PATH) -> str:
    """'2 BE · 3 MU' — whose money this holding is actually carrying."""
    bits = [(o, q) for (a, o, v), q in fold(path).items()
            if a == acct(account) and v == tkr(vehicle) and q > _EPS]
    return " · ".join(f"{q:g} {o}" for o, q in sorted(bits))


def owed_qty(account, vehicle, path: Path = PARKED_PATH) -> float:
    """Total shares of this vehicle spoken for by other tickers' reserves."""
    return sum(q for (a, _o, v), q in fold(path).items()
               if a == acct(account) and v == tkr(vehicle) and q > _EPS)


def free_qty(account, vehicle, held_qty, path: Path = PARKED_PATH) -> float:
    """How many shares are genuinely yours to sell.

    CHECK THIS BEFORE PLACING A SELL. Chakravarti, 2026-11-08: *"what if i
    put sell order on SPY on its own and it encroaches into BE and MU?"* —
    this is the number that answers it. Can go negative, and a negative is
    a breach that already happened, not an error in the arithmetic.
    """
    return float(held_qty or 0) - owed_qty(account, vehicle, path)


def breaches(positions, path: Path = PARKED_PATH) -> list[str]:
    """Vehicles holding fewer shares than are parked in them.

    `positions` is {(account, ticker): qty}. Called after a run, because a
    sale placed outside this system — a manual order at Schwab, a stop the
    recommender suggested on the whole position — cannot be prevented here,
    only noticed. Same posture as the reserve ledger's BREACH on a buy that
    overdrew its fence: say it loudly, do not absorb it.
    """
    out = []
    for (a, v), owed in sorted(_by_vehicle(path).items()):
        held = float((positions or {}).get((a, v), 0) or 0)
        if held + _EPS < owed:
            out.append(
                f"{a}/{v}: {owed:g} share(s) parked but only {held:g} held — "
                f"short {owed - held:g}. A sale went through the parcels; "
                f"unpark the difference or buy it back.")
    return out


def _by_vehicle(path: Path = PARKED_PATH) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for (a, _o, v), q in fold(path).items():
        if q > _EPS:
            out[(a, v)] = out.get((a, v), 0.0) + q
    return out


# ---------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------
def park(account, owner, vehicle, qty, *, cash=None, reason="",
         path: Path = PARKED_PATH, log=print) -> float:
    """Record that `qty` shares of `vehicle` are held for `owner`'s reserve."""
    qty = float(qty)
    if qty <= 0:
        raise ValueError("park quantity must be a positive number")
    if tkr(owner) == tkr(vehicle):
        # Not pedantry: it would make free_qty subtract a holding from
        # itself, so the position would look fully spoken for forever.
        raise ValueError(f"{tkr(owner)} cannot be parked in itself")
    _append(account, owner, vehicle, qty, "PARK", cash=cash, reason=reason,
            path=path, log=log)
    return outstanding(account, owner, vehicle, path)


def unpark(account, owner, vehicle, qty, *, reason="",
           path: Path = PARKED_PATH, log=print) -> float:
    """Release some or all of a parcel — the vehicle was sold for the owner."""
    qty = float(qty)
    if qty <= 0:
        raise ValueError("unpark quantity must be a positive number")
    cur = outstanding(account, owner, vehicle, path)
    if qty > cur + _EPS:
        # No force flag. Unparking more than is parked is not a decision,
        # it is a typo, and it would leave a negative parcel that quietly
        # frees shares nobody owns.
        raise ValueError(
            f"{acct(account)}/{tkr(owner)} has only {cur:g} {tkr(vehicle)} "
            f"parked; cannot unpark {qty:g}")
    _append(account, owner, vehicle, qty, "UNPARK", reason=reason,
            path=path, log=log)
    return outstanding(account, owner, vehicle, path)


def render(path: Path = PARKED_PATH) -> list[str]:
    """Every live parcel, one line each, for the `parked` verb."""
    rows = sorted((a, o, v, q) for (a, o, v), q in fold(path).items()
                  if q > _EPS)
    if not rows:
        return ["nothing parked"]
    return [f"{a}/{o:<6} {q:>10g} {v}" for a, o, v, q in rows]
