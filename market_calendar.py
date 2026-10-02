#!/usr/bin/env python3
"""
market_calendar.py
==================
Is the US stock market open right now?

    .venv/bin/python market_calendar.py            # today, and the next closures
    .venv/bin/python market_calendar.py --year 2027

PURE. No network, no dependencies beyond the standard library, no clock except
the one you hand it. Runs on either Pi.

RULES, NOT A TABLE
    Every NYSE holiday is derivable: fixed dates with an observance rule, Nth
    weekdays, and Good Friday from Easter. A hardcoded list would be correct
    until the year it silently was not — and a calendar that quietly expires
    is exactly the kind of failure this system keeps having. Nothing here
    needs maintaining.

    `pandas_market_calendars` would also do it, but is not installed on either
    Pi and is a heavy dependency for ten dates a year.

EARLY CLOSES ARE NOT A DETAIL
    On the day after Thanksgiving, and before Independence Day and Christmas,
    the market shuts at 13:00 ET — 10:00 PT. An options reminder scheduled for
    11:00 PT would otherwise fire into a closed market three times a year and
    look exactly like a working one.

HALF-DAY AND HOLIDAY ONLY. Unscheduled closures — a hurricane, a presidential
funeral, 9/11 — are not predictable and are not modelled. The cost is sending
one reminder into a market that shut unexpectedly, which is not worth a feed.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
    PT = ZoneInfo("America/Los_Angeles")
except Exception:                       # pragma: no cover - tzdata missing
    ET = PT = None

OPEN_ET = time(9, 30)
CLOSE_ET = time(16, 0)
EARLY_CLOSE_ET = time(13, 0)


def easter(year: int) -> date:
    """Gregorian Easter Sunday. Anonymous algorithm (Meeus/Jones/Butcher).

    Needed only for Good Friday, which is the one NYSE holiday with no simple
    calendar rule — it is the single reason a table is otherwise tempting.
    """
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (a + 11 * h) // 319
    r = (2 * e + 2 * i - h + m - k + 32) % 7
    n = (h - m + r + 90) // 25
    p = (h - m + r + n + 19) % 32
    return date(year, n, p)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The nth `weekday` of a month (Monday=0). n=-1 means the last one."""
    if n > 0:
        d = date(year, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    d = nxt - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date) -> date | None:
    """NYSE observance for a fixed-date holiday.

    Saturday moves BACK to Friday, Sunday moves FORWARD to Monday — except
    New Year's Day, which the NYSE does not observe on the preceding December
    31. That exception is why this returns None rather than a date.
    """
    if d.weekday() == 5:                                  # Saturday
        return None if (d.month, d.day) == (1, 1) else d - timedelta(days=1)
    if d.weekday() == 6:                                  # Sunday
        return d + timedelta(days=1)
    return d


def holidays(year: int) -> dict:
    """{date: name} for every full NYSE closure in `year`."""
    out = {}

    def put(d, name):
        if d is not None and d.year == year:
            out[d] = name

    put(_observed(date(year, 1, 1)), "New Year's Day")
    put(_nth_weekday(year, 1, 0, 3), "Martin Luther King Jr. Day")
    put(_nth_weekday(year, 2, 0, 3), "Washington's Birthday")
    put(easter(year) - timedelta(days=2), "Good Friday")
    put(_nth_weekday(year, 5, 0, -1), "Memorial Day")
    if year >= 2022:                       # first observed by the NYSE in 2022
        put(_observed(date(year, 6, 19)), "Juneteenth")
    put(_observed(date(year, 7, 4)), "Independence Day")
    put(_nth_weekday(year, 9, 0, 1), "Labor Day")
    put(_nth_weekday(year, 11, 3, 4), "Thanksgiving")
    put(_observed(date(year, 12, 25)), "Christmas")
    return out


def early_closes(year: int) -> dict:
    """{date: name} for 13:00 ET half-days."""
    hol = holidays(year)
    out = {}

    def half(d, name):
        if d.weekday() < 5 and d not in hol:
            out[d] = name

    half(_nth_weekday(year, 11, 3, 4) + timedelta(days=1), "day after Thanksgiving")
    # Only when the eve is itself a session: if July 4 lands on Saturday the
    # holiday moves to Friday the 3rd, and there is no half-day at all.
    for m, d_, name in ((7, 3, "July 3"), (12, 24, "Christmas Eve")):
        eve = date(year, m, d_)
        nxt = eve + timedelta(days=1)
        if nxt in hol or (nxt.weekday() < 5 and _observed(nxt) == nxt):
            half(eve, name)
    return out


def is_trading_day(d: date) -> bool:
    """A weekday that is not a full closure."""
    return d.weekday() < 5 and d not in holidays(d.year)


def close_time(d: date) -> time | None:
    """Closing time in ET, or None if the market is shut that day."""
    if not is_trading_day(d):
        return None
    return EARLY_CLOSE_ET if d in early_closes(d.year) else CLOSE_ET


def is_open(when: datetime | None = None) -> bool:
    """Is the market open at `when` (default: now)?

    Naive datetimes are read as EASTERN, not as the host's clock. Pi 1 and Pi 2
    have disagreed about their timezone before, and a market-hours check that
    silently used the wrong one would be wrong for exactly six hours a day.
    """
    when = when or (datetime.now(ET) if ET else datetime.now())
    if when.tzinfo is None and ET:
        when = when.replace(tzinfo=ET)
    et = when.astimezone(ET) if (ET and when.tzinfo) else when
    shut = close_time(et.date())
    return shut is not None and OPEN_ET <= et.time() < shut


def describe(when: datetime | None = None) -> str:
    """One line for a log: open, or shut and why."""
    when = when or (datetime.now(ET) if ET else datetime.now())
    et = when.astimezone(ET) if (ET and when.tzinfo) else when
    d = et.date()
    if d.weekday() >= 5:
        return f"closed — {d:%A}"
    if d in holidays(d.year):
        return f"closed — {holidays(d.year)[d]}"
    shut = close_time(d)
    half = early_closes(d.year).get(d)
    if et.time() < OPEN_ET:
        return f"not open yet — opens {OPEN_ET:%H:%M} ET"
    if et.time() >= shut:
        return (f"closed — {half}, shut {shut:%H:%M} ET" if half
                else f"closed — shut {shut:%H:%M} ET")
    return f"OPEN until {shut:%H:%M} ET" + (f" ({half})" if half else "")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Is the US market open?")
    ap.add_argument("--year", type=int, help="list that year's closures")
    ap.add_argument("--date", help="check a specific YYYY-MM-DD")
    args = ap.parse_args()

    if args.year:
        hol, half = holidays(args.year), early_closes(args.year)
        for d in sorted({**hol, **half}):
            tag = "half-day" if d in half else "CLOSED  "
            print(f"  {d:%Y-%m-%d %a}  {tag}  {hol.get(d) or half[d]}")
        return 0

    if args.date:
        d = date.fromisoformat(args.date)
        print(f"{d:%Y-%m-%d %a}  "
              f"{'trading day' if is_trading_day(d) else 'CLOSED'}"
              + (f", shuts {close_time(d):%H:%M} ET" if close_time(d) else ""))
        return 0

    print(describe())
    return 0 if is_open() else 1


if __name__ == "__main__":
    raise SystemExit(main())
