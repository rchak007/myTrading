#!/usr/bin/env python3
"""
probe_schwab_api.py
-------------------
Read-only compatibility probe for whatever schwabdev / Schwab API version is
installed on THIS machine. Makes no writes, changes no state, touches no cache.

Schwab's transactions endpoint has two constraints that are version- and
account-specific, undocumented in schwabdev, and fatal if guessed wrong:

  1. a maximum start..end window
  2. an enum of valid `types` values, passed as a STRING (not a list)

Guessing either one produces a 400, or worse, a silent partial result. This
probe measures both, so txn_cache.py can be configured from fact.

    python probe_schwab_api.py
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import dotenv
import schwabdev

ROOT = Path(__file__).resolve().parent.parent
TOKENS_DB = Path.home() / ".schwabdev" / "tokens.db"

# What txn_cache.py currently believes is valid.
CANDIDATE_TYPES = [
    "TRADE", "DIVIDEND_OR_INTEREST", "CORPORATE_ACTION", "SECURITY_TRANSFER",
    "RECEIVE_AND_DELIVER", "ACH_RECEIPT", "ACH_DISBURSEMENT", "CASH_RECEIPT",
    "CASH_DISBURSEMENT", "ELECTRONIC_FUND", "WIRE_IN", "WIRE_OUT", "FEE",
    "TAX", "ADJUSTMENT", "JOURNAL", "MEMORANDUM",
]


def iso(d: date, end: bool = False) -> str:
    return d.strftime("%Y-%m-%dT23:59:59.999Z" if end else "%Y-%m-%dT00:00:00.000Z")


def msg(resp) -> str:
    """First line of a Schwab error, trimmed but not truncated to uselessness."""
    try:
        j = resp.json()
        m = j.get("message") or j.get("error") or str(j)
    except Exception:
        m = resp.text
    return " ".join(str(m).split())[:300]


def main() -> int:
    dotenv.load_dotenv(ROOT / ".env")
    client = schwabdev.Client(
        os.getenv("app_key"), os.getenv("app_secret"), os.getenv("callback_url"),
        tokens_db=str(TOKENS_DB),
    )

    accounts = client.linked_accounts().json()
    print(f"linked accounts: {len(accounts)}")
    h = accounts[0]["hashValue"]
    today = date.today()

    # ---------------------------------------------------------------- windows
    print("\n=== 1. maximum date window (types=TRADE) ===")
    max_ok = 0
    for days in (30, 60, 90, 180, 270, 364, 365, 366):
        start = today - timedelta(days=days)
        r = client.transactions(h, iso(start), iso(today, end=True), "TRADE")
        ok = r.status_code == 200
        if ok:
            max_ok = max(max_ok, days)
        print(f"  {days:>4}d  HTTP {r.status_code}  {'ok' if ok else msg(r)}")

    probe_days = min(max_ok, 30) or 30
    p_start = iso(today - timedelta(days=probe_days))
    p_end = iso(today, end=True)
    print(f"\n  -> largest window that worked: {max_ok}d")
    print(f"  -> using {probe_days}d for the type probe")

    # ---------------------------------------------------------------- types
    print("\n=== 2. which `types` values are valid, one at a time ===")
    valid, invalid = [], []
    for t in CANDIDATE_TYPES:
        r = client.transactions(h, p_start, p_end, t)
        if r.status_code == 200:
            valid.append(t)
            print(f"  {t:<22} OK    n={len(r.json())}")
        else:
            invalid.append(t)
            print(f"  {t:<22} {r.status_code}  {msg(r)[:120]}")

    # ---------------------------------------------------------------- combos
    print("\n=== 3. can several types share one request? ===")
    if len(valid) >= 2:
        for label, val in (
            ("two, comma",  ",".join(valid[:2])),
            ("all valid, comma", ",".join(valid)),
        ):
            r = client.transactions(h, p_start, p_end, val)
            print(f"  {label:<18} HTTP {r.status_code}  "
                  f"{('n=' + str(len(r.json()))) if r.status_code == 200 else msg(r)[:150]}")

    # ---------------------------------------------------------------- summary
    print("\n=== SUMMARY ===")
    print(f"  max window : {max_ok} days")
    print(f"  valid types ({len(valid)}): {', '.join(valid) if valid else '(none)'}")
    print(f"  INVALID     ({len(invalid)}): {', '.join(invalid) if invalid else '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
