#!/usr/bin/env python3
"""
test_coverage.py
================
Tests for the Dashboard coverage flags in orders_sheet.py. Runs anywhere — no
Schwab, no Sheets, no network.

    .venv/bin/python test_coverage.py

These flags decide whether a position reads as PROTECTED. Both bugs this file
exists to catch said "covered" about something that was not, which is the
expensive direction to be wrong in.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd                                     # noqa: E402
from orders_sheet import coverage_for, _classify        # noqa: E402

FAILED = []
PRICE = 357.68          # TSLA on 2026-10-02


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def intent(side, close_is, trig, status="VALID", qty=1.0, ticker="TSLA",
           acct="171"):
    return pd.DataFrame([{"Ticker": ticker, "Account": acct, "Side": side,
                          "Close_Is": close_is, "Trigger_Price": float(trig),
                          "Qty": qty, "Row_ID": "r", "Status": status}])


def slot_of(cov, mark):
    hit = [k for k, v in cov.items() if v == mark]
    return hit[0] if hit else None


print("\n── a sheet intent is classified by its DIRECTION, not by price ──")
# THE REPORTED BUG. TSLA at 357.68 with "BUY if it CLOSES BELOW 400" was shown
# as Has_Breakout, because 400 happens to sit above the current price. Buying
# weakness is the opposite of buying strength; Close_Is says which it is.
cov = coverage_for("TSLA", "171", PRICE, None, intent("BUY", "BELOW", 400),
                   held_qty=109)
check("BUY/BELOW above today's price is a DIP, not a breakout",
      slot_of(cov, "P") == "Has_Dip", slot_of(cov, "P"))

# All four, each with the trigger deliberately on the far side of price, so a
# price-based rule would get every one of them wrong.
for side, d, trig, want in (("BUY", "BELOW", 400, "Has_Dip"),
                            ("BUY", "ABOVE", 300, "Has_Breakout"),
                            ("SELL", "BELOW", 400, "Has_Stop"),
                            ("SELL", "ABOVE", 300, "Has_Trim")):
    got = slot_of(coverage_for("TSLA", "171", PRICE, None,
                               intent(side, d, trig), held_qty=109), "P")
    check(f"{side}/{d} @{trig} (price {PRICE}) → {want}", got == want, got)

print("\n── a resting Schwab order keeps the price rule ──")
# It carries no direction field, so which side of price it sits on is the only
# information there is.
for side, px, want in (("SELL", 300, "Has_Stop"), ("SELL", 400, "Has_Trim"),
                       ("BUY", 300, "Has_Dip"), ("BUY", 400, "Has_Breakout")):
    orders = pd.DataFrame([{"Ticker": "TSLA", "Account": "171", "Side": side,
                            "Stop_Price": float(px), "Limit_Price": None}])
    got = slot_of(coverage_for("TSLA", "171", PRICE, orders), "Y")
    check(f"resting {side} @{px} → {want}", got == want, got)
check("_classify falls back to price when direction is absent",
      _classify("SELL", 300, PRICE) == "Has_Stop"
      and _classify("SELL", 400, PRICE) == "Has_Trim")
check("_classify ignores a junk direction rather than trusting it",
      _classify("SELL", 300, PRICE, "SIDEWAYS") == "Has_Stop")

print("\n── a spent intent protects nothing ──")
# SUBMITTED means the intent handed off to Schwab. If that order still rests,
# orders_df reports Y — stronger and truer. If it filled, nothing remains.
# TSLA claimed coverage for days from a BUY that filled on 2026-09-29.
for status in ("SUBMITTED", "FILLED", "CANCELLED", "EXPIRED", "VOID",
               "REJECTED", "BLOCKED"):
    # read_intents drops these before coverage_for ever sees them; this asserts
    # the list itself, which is the thing that was wrong.
    from orders_sheet import read_intents                # noqa: E402
    import inspect
    src = inspect.getsource(read_intents)
    check(f"{status} is treated as history", f'"{status}"' in src)

print("\n── the rules that were already right ──")
orders = pd.DataFrame([{"Ticker": "TSLA", "Account": "171", "Side": "SELL",
                        "Stop_Price": 300.0, "Limit_Price": None}])
both = coverage_for("TSLA", "171", PRICE, orders, intent("SELL", "BELOW", 320))
check("a resting order outranks a sheet intent in the same slot",
      both["Has_Stop"] == "Y", both["Has_Stop"])

# A SELL intent for more shares than are held would be refused by the engine as
# OVERSELL. Counting it as protection would be the worst kind of wrong: the
# Dashboard saying covered about the one order that cannot run.
over = coverage_for("TSLA", "171", PRICE, None,
                    intent("SELL", "BELOW", 320, qty=5.0), held_qty=2)
check("an OVERSELL intent claims nothing", over["Has_Stop"] == "N",
      over["Has_Stop"])
ok = coverage_for("TSLA", "171", PRICE, None,
                  intent("SELL", "BELOW", 320, qty=2.0), held_qty=2)
check("...but selling exactly what is held is fine", ok["Has_Stop"] == "P",
      ok["Has_Stop"])

wrong_acct = coverage_for("TSLA", "431", PRICE, None,
                          intent("SELL", "BELOW", 320), held_qty=100)
check("an intent in another account protects nothing here",
      wrong_acct["Has_Stop"] == "N", wrong_acct["Has_Stop"])

check("no price → every flag blank, nothing asserted",
      set(coverage_for("TSLA", "171", None, None).values()) == {""})

print("\n── the ORDERS block shows what the flag points at ──")
# A block saying Has_Stop = P and then showing nothing underneath is the
# Dashboard asserting protection it never points at.
from orders_sheet import build_dashboard                 # noqa: E402
pos = pd.DataFrame([{"Ticker": "BE", "Acct": "171", "Qty": 23,
                     "Avg_Cost": 217.85, "Market_Value": 6518.43,
                     "Unrealized_PL": 1507.88, "Has_Stop": "P",
                     "Has_Trim": "N", "Has_Dip": "N", "Has_Breakout": "N",
                     "Seed_Reserved": ""}])
ints = intent("SELL", "BELOW", 214, qty=23.0, ticker="BE")
ints.loc[0, "Row_ID"] = "2026-09-29-BE-01"
rows, marks = build_dashboard(pos, None, {"BE": {"quote": {"lastPrice": 283.29}}},
                              set(), None, None, ints)
flat = [" ".join(str(c) for c in r) for r in rows]
check("the waiting intent appears under ORDERS",
      any("CLOSE BELOW" in r for r in flat))
check("its trigger price is shown", any("214" in r for r in flat))
check("its Row_ID is shown, so the sheet row is findable",
      any("2026-09-29-BE-01" in r for r in flat))
check("'sheet' marks where a Schwab timestamp would be",
      any("sheet" in r for r in flat))
check("it is marked for its own paint", len(marks["intent"]) == 1,
      str(marks["intent"]))
check("'no open orders' is NOT claimed when an intent is waiting",
      not any("no open orders" in r for r in flat))

bare, _ = build_dashboard(pos, None, {"BE": {"quote": {"lastPrice": 283.29}}},
                          set(), None, None, None)
check("...but it still says so when there is genuinely nothing",
      any("no open orders" in " ".join(str(c) for c in r) for r in bare))

print(f"\n{'ALL PASS' if not FAILED else str(len(FAILED)) + ' FAILED: ' + ', '.join(FAILED)}")
raise SystemExit(1 if FAILED else 0)
