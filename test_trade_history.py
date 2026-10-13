#!/usr/bin/env python3
"""
test_trade_history.py
=====================
Tests for trade_history.py. No network, no sheet, no Schwab.

    .venv/bin/python test_trade_history.py

Two things must never go wrong here: a trade must not be recorded twice (the
fills window is re-fetched every 35 minutes), and a hand-placed trade must not
be reported as one the machine made.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_tmp = Path(tempfile.mkdtemp()) / "th.csv"
os.environ["TRADE_HISTORY_FILE"] = str(_tmp)
import pandas as pd                                      # noqa: E402
import trade_history as TH                               # noqa: E402

TH.FILE = _tmp
FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def fill(ref, ticker, side="BUY", when="2026-10-01T10:00:00", oid="", qty=1,
         px=100.0, acct="171", asset="EQUITY"):
    return {"Ref": ref, "Account": acct, "Ticker": ticker, "Side": side,
            "Fill_QTY": qty, "Fill_Price": px, "Net_Amount": -qty * px,
            "Asset_Type": asset, "Fill_Time": when, "Order_ID": oid}


TH.ledger_order_ids = lambda log=print: {"1008110931971": "2026-09-28-TSLA-02"}

print("\n── both sources, told apart by the order id ──")
# Schwab makes no distinction: a fill is a fill whether the engine placed the
# order or Chakravarti tapped it in on his phone. The order id is the only link.
n = TH.record(pd.DataFrame([
    fill("A1", "TSLA", "BUY", "2026-09-29T14:05:17", oid="1008110931971"),
    fill("A2", "MSTR", "SELL", "2026-10-05T15:17:00", oid="999999"),
    fill("A3", "AMD", "SELL", "2026-09-30T12:00:00", oid=""),
]), log=lambda *a: None)
check("all three recorded", n == 3, str(n))
by = {r["Ticker"]: r for r in TH.read()}
check("an order id in the ledger reads PI", by["TSLA"]["Source"] == "PI",
      by["TSLA"]["Source"])
check("...and carries the Row_ID that caused it",
      by["TSLA"]["Row_ID"] == "2026-09-28-TSLA-02", by["TSLA"]["Row_ID"])
check("an order id NOT in the ledger reads MANUAL",
      by["MSTR"]["Source"] == "MANUAL", by["MSTR"]["Source"])
# Calling an unattributable fill MANUAL would be a claim, not an observation.
check("no order id reads '?', not MANUAL", by["AMD"]["Source"] == "?",
      by["AMD"]["Source"])
check("options are recorded like anything else",
      TH.record(pd.DataFrame([fill("A4", "TSLA", "SELL", asset="OPTION",
                                   oid="x")]), log=lambda *a: None) == 1)

print("\n── the replay guard ──")
# The fills window is re-fetched every cycle. Without dedupe the same trade
# would be recorded every 35 minutes, forever.
again = pd.DataFrame([fill("A1", "TSLA", "BUY", oid="1008110931971"),
                      fill("A2", "MSTR", "SELL", oid="999999")])
check("re-recording the same fills adds nothing",
      TH.record(again, log=lambda *a: None) == 0)
check("a genuinely new fill still gets through",
      TH.record(pd.DataFrame([fill("A9", "NVDA")]), log=lambda *a: None) == 1)
check("a fill with no Ref is skipped rather than duplicated endlessly",
      TH.record(pd.DataFrame([fill("", "NVDA")]), log=lambda *a: None) == 0)
check("an empty frame is a no-op",
      TH.record(pd.DataFrame(), log=lambda *a: None) == 0
      and TH.record(None, log=lambda *a: None) == 0)

print("\n── newest first ──")
rows = TH.newest_first(TH.read())
stamps = [r["Filled_At"] for r in rows]
check("sorted descending", stamps == sorted(stamps, reverse=True), str(stamps))
check("the most recent is at the top", rows[0]["Ticker"] == "MSTR",
      rows[0]["Ticker"])
# A missing timestamp is still a trade that happened.
TH.record(pd.DataFrame([fill("A10", "ZZZ", when="")]), log=lambda *a: None)
rows = TH.newest_first(TH.read())
check("a fill with no timestamp sinks but is NOT dropped",
      any(r["Ticker"] == "ZZZ" for r in rows)
      and rows[-1]["Ticker"] == "ZZZ", str([r["Ticker"] for r in rows]))

print("\n── the backfill walks months without gaps ──")
# A gap between windows loses every trade in it, silently — the tab would
# just have nothing from that month and nothing would say why.
import datetime as _d                                      # noqa: E402
import cash_reserve as _cr                                 # noqa: E402

_seen = []


def _fake_fetch(cw, *, days_back=7, start=None, end=None, log=print):
    _seen.append((start.date(), end.date()))
    return pd.DataFrame()


_cr.fetch_fills = _fake_fetch
TH.backfill(object(), "2026-01-01", "2026-10-12", log=lambda *a: None)
check("one window per month", len(_seen) == 10, str(len(_seen)))
check("starts on the date asked for", _seen[0][0] == _d.date(2026, 1, 1),
      str(_seen[0][0]))
check("stops on the end date, not the month end",
      _seen[-1][1] == _d.date(2026, 10, 12), str(_seen[-1][1]))
check("NO GAPS between windows",
      all(_seen[i][0] == _seen[i - 1][1] + _d.timedelta(days=1)
          for i in range(1, len(_seen))),
      str([(str(a), str(b)) for a, b in _seen]))
check("every day in the range is covered exactly once",
      sum((e - s).days + 1 for s, e in _seen)
      == (_seen[-1][1] - _seen[0][0]).days + 1)
_seen.clear()
TH.backfill(object(), "2025-11-01", "2026-02-15", log=lambda *a: None)
check("crosses a year boundary", len(_seen) == 4 and _seen[0][0].year == 2025
      and _seen[-1][0].year == 2026, str(len(_seen)))
_seen.clear()
check("start after end does nothing rather than looping",
      TH.backfill(object(), "2026-10-12", "2026-01-01",
                  log=lambda *a: None) == 0 and not _seen)

print("\n── a backfilled month is recorded, and re-running adds nothing ──")


def _one_fill(cw, *, days_back=7, start=None, end=None, log=print):
    return pd.DataFrame([fill(f"BF{start:%Y%m}", "MU",
                              when=f"{start:%Y-%m}-15T10:00:00")])


_cr.fetch_fills = _one_fill
n1 = TH.backfill(object(), "2026-01-01", "2026-03-31", log=lambda *a: None)
check("three months recorded three fills", n1 == 3, str(n1))
check("re-running the same window adds nothing",
      TH.backfill(object(), "2026-01-01", "2026-03-31",
                  log=lambda *a: None) == 0)
check("an overlapping window adds only what is new",
      TH.backfill(object(), "2026-03-01", "2026-04-30",
                  log=lambda *a: None) == 1)

print("\n── the sheet is no longer capped ──")
check("SHEET_MAX defaults to 0, meaning everything", TH.SHEET_MAX == 0,
      str(TH.SHEET_MAX))

print("\n── what it is not ──")
check("every recorded row has a Ref", all(r.get("Ref") for r in TH.read()))
check("the file keeps every column it promises",
      set(TH.read()[0]) == set(TH.COLS),
      str(set(TH.COLS) ^ set(TH.read()[0])))

print(f"\n{'ALL PASS' if not FAILED else str(len(FAILED)) + ' FAILED: ' + ', '.join(FAILED)}")
raise SystemExit(1 if FAILED else 0)
