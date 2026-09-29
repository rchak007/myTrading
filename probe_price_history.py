#!/usr/bin/env python3
"""
probe_price_history.py
======================
What does Schwab's daily price history actually return, and how should its
timestamps be read?

    .venv/bin/python probe_price_history.py TSLA

Read-only.

WHY
    order_engine refused to act on 2026-09-28 (a Monday) saying the last daily
    bar was 2026-09-24. Friday the 25th should exist, and so should Monday
    itself by 20:30 PT. Two candidates:

      * a timezone off-by-one — a daily bar stamped midnight ET renders as the
        PREVIOUS day when converted with the local clock, which is how a
        Friday bar becomes "Thursday";
      * genuinely stale data.

    Those need different fixes, so this prints the raw epoch alongside three
    interpretations of it and lets the difference speak.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def main() -> int:
    ticker = (sys.argv[1] if len(sys.argv) > 1 else "TSLA").upper()

    import jobStocksSignals as job
    client = job.get_schwab_client()
    inner = client
    getter = getattr(client, "get_client", None)
    if callable(getter):
        inner = getter() or client

    meth = getattr(inner, "price_history", None)
    if not callable(meth):
        print("no price_history() on this client")
        return 1

    resp = meth(ticker, periodType="month", period=1,
                frequencyType="daily", frequency=1)
    body = resp.json() if hasattr(resp, "json") else resp
    candles = (body or {}).get("candles") or []

    print(f"\n{ticker}: {len(candles)} daily candle(s), "
          f"empty={body.get('empty')}")
    print(f"now: local {datetime.now():%Y-%m-%d %H:%M %Z} · "
          f"UTC {datetime.now(timezone.utc):%Y-%m-%d %H:%M}\n")

    try:
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
    except Exception:
        ET = None

    print(f"  {'epoch ms':>14}  {'as LOCAL':<12} {'as UTC':<12} "
          f"{'as ET':<12} {'close':>9}")
    for c in candles[-8:]:
        ms = c.get("datetime", 0)
        secs = ms / 1000
        loc = datetime.fromtimestamp(secs).date()
        utc = datetime.fromtimestamp(secs, timezone.utc).date()
        et = datetime.fromtimestamp(secs, ET).date() if ET else "-"
        print(f"  {ms:>14}  {str(loc):<12} {str(utc):<12} {str(et):<12} "
              f"{c.get('close'):>9}")

    if candles:
        last = candles[-1]
        print(f"\nlast candle full payload:\n  {last}")
        print("\nIf 'as LOCAL' is one day behind 'as UTC'/'as ET', the bar is")
        print("stamped at midnight ET and the local conversion is the bug.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
