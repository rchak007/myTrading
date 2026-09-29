#!/usr/bin/env python3
"""
probe_price_today.py
====================
Which price_history arguments return TODAY's daily bar?

    .venv/bin/python probe_price_today.py TSLA

Read-only.

WHY
    Measured 2026-09-28 at 23:45 ET, nearly eight hours after the close:
    price_history(periodType="month", period=1, frequencyType="daily") returned
    23 candles ending FRIDAY 09-25. Today's bar was absent.

    That breaks the premise of the close-triggered engine, which is supposed to
    read today's close after 13:15 PT and act on it. If today's bar cannot be
    fetched the trigger fires a day late, and the design has to say so instead
    of quietly being wrong.

    Schwab's priceHistory takes EITHER a relative period OR an explicit
    startDate/endDate in epoch milliseconds. A relative period is the more
    likely culprit: "the last month" is plausibly interpreted as complete days
    only. This tries the alternatives and prints the newest bar each returns.
"""
from __future__ import annotations

import inspect
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:
    ET = timezone.utc


def newest(body) -> str:
    candles = (body or {}).get("candles") or []
    if not candles:
        return "no candles"
    c = candles[-1]
    d = datetime.fromtimestamp(c.get("datetime", 0) / 1000, ET).date()
    return f"{len(candles):>3} bars, newest {d} close {c.get('close')}"


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
        print("no price_history()")
        return 1

    try:
        print(f"\nsignature: price_history{inspect.signature(meth)}\n")
    except (TypeError, ValueError):
        print("\nsignature unavailable\n")

    now = datetime.now(timezone.utc)
    end_ms = int(now.timestamp() * 1000)
    start_ms = int((now - timedelta(days=20)).timestamp() * 1000)
    print(f"today (ET): {datetime.now(ET).date()}   "
          f"endDate epoch ms: {end_ms}\n")

    attempts = [
        ("month/1 daily (what we use)",
         dict(periodType="month", period=1, frequencyType="daily", frequency=1)),
        ("month/1 + explicit endDate",
         dict(periodType="month", period=1, frequencyType="daily", frequency=1,
              endDate=end_ms)),
        ("explicit start+end, no period",
         dict(frequencyType="daily", frequency=1,
              startDate=start_ms, endDate=end_ms)),
        ("day/10 daily",
         dict(periodType="day", period=10, frequencyType="daily", frequency=1)),
        ("year/1 daily",
         dict(periodType="year", period=1, frequencyType="daily", frequency=1)),
        ("month/1 + needExtendedHoursData",
         dict(periodType="month", period=1, frequencyType="daily", frequency=1,
              needExtendedHoursData=True)),
        ("month/1 + needPreviousClose",
         dict(periodType="month", period=1, frequencyType="daily", frequency=1,
              needPreviousClose=True)),
    ]

    for label, kw in attempts:
        try:
            resp = meth(ticker, **kw)
            code = getattr(resp, "status_code", 200)
            if code != 200:
                print(f"  {label:<34} HTTP {code} "
                      f"{str(getattr(resp,'text',''))[:90]}")
                continue
            body = resp.json() if hasattr(resp, "json") else resp
            print(f"  {label:<34} {newest(body)}")
        except TypeError as e:
            print(f"  {label:<34} rejected: {e}")
        except Exception as e:
            print(f"  {label:<34} {type(e).__name__}: {str(e)[:70]}")

    print("\nWhichever line shows today's date is the call the engine should make.")
    print("If NONE do, today's close is simply not available from this endpoint")
    print("and the engine must say a close trigger acts the NEXT session.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
