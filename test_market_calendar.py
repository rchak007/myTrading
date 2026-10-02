#!/usr/bin/env python3
"""
test_market_calendar.py
=======================
Tests for market_calendar.py. No network, no dependencies.

    .venv/bin/python test_market_calendar.py

The dates here are the PUBLISHED NYSE calendars for 2026 and 2027, typed in by
hand. That is the point: the module derives them from rules, and this asserts
the rules land on the real answers — including the years where the observance
rules do something surprising.
"""
from __future__ import annotations

import sys
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
import market_calendar as mc                               # noqa: E402

ET = ZoneInfo("America/New_York")
PT = ZoneInfo("America/Los_Angeles")
FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


print("\n── the published 2026 NYSE calendar ──")
want26 = {date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16),
          date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19),
          date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26),
          date(2026, 12, 25)}
got26 = set(mc.holidays(2026))
check("all ten closures, no more and no fewer", got26 == want26,
      f"missing {sorted(want26 - got26)} extra {sorted(got26 - want26)}")
check("half-days are Nov 27 and Dec 24",
      set(mc.early_closes(2026)) == {date(2026, 11, 27), date(2026, 12, 24)},
      str(sorted(mc.early_closes(2026))))

print("\n── the observance rules, where they get interesting ──")
# July 4 2026 is a SATURDAY, so the holiday moves BACK to Friday the 3rd — and
# there is then no July 3 half-day, because the 3rd is the holiday itself.
check("Jul 4 on a Saturday → closed Friday Jul 3",
      date(2026, 7, 3) in mc.holidays(2026))
check("...and no July 3 half-day that year",
      date(2026, 7, 3) not in mc.early_closes(2026))
# July 4 2027 is a SUNDAY, so it moves FORWARD to Monday.
check("Jul 4 on a Sunday → closed Monday Jul 5",
      date(2027, 7, 5) in mc.holidays(2027))
# Christmas 2027 is a SATURDAY → Friday Dec 24, and no Christmas Eve half-day.
check("Christmas on a Saturday → closed Friday Dec 24",
      date(2027, 12, 24) in mc.holidays(2027))
check("...and no Christmas Eve half-day that year",
      date(2027, 12, 24) not in mc.early_closes(2027))
# THE EXCEPTION: New Year's Day on a Saturday does NOT close Dec 31. 2022 is
# the real case — Jan 1 2022 was a Saturday and the NYSE traded Dec 31 2021.
check("New Year's on a Saturday does NOT close the previous Dec 31",
      date(2021, 12, 31) not in mc.holidays(2021)
      and date(2022, 1, 1) not in mc.holidays(2022))
check("Juneteenth is absent before 2022", date(2021, 6, 18) not in mc.holidays(2021)
      and date(2021, 6, 19) not in mc.holidays(2021))
check("Juneteenth is present from 2022", date(2022, 6, 20) in mc.holidays(2022))

print("\n── Good Friday, the one with no calendar rule ──")
for y, d in ((2026, date(2026, 4, 3)), (2027, date(2027, 3, 26)),
             (2024, date(2024, 3, 29)), (2025, date(2025, 4, 18))):
    check(f"Good Friday {y} is {d}", d in mc.holidays(y),
          str(sorted(x for x in mc.holidays(y) if x.month in (3, 4))))

print("\n── open and shut ──")
check("a normal Tuesday mid-session is open",
      mc.is_open(datetime(2026, 10, 6, 11, 0, tzinfo=ET)))
check("one minute before the bell is not",
      not mc.is_open(datetime(2026, 10, 6, 9, 29, tzinfo=ET)))
check("the opening bell itself is",
      mc.is_open(datetime(2026, 10, 6, 9, 30, tzinfo=ET)))
check("16:00 ET is already shut",
      not mc.is_open(datetime(2026, 10, 6, 16, 0, tzinfo=ET)))
check("Saturday is shut", not mc.is_open(datetime(2026, 10, 3, 11, 0, tzinfo=ET)))
check("Thanksgiving is shut",
      not mc.is_open(datetime(2026, 11, 26, 11, 0, tzinfo=ET)))

print("\n── the half-day trap, which is the whole reason this exists ──")
# The OPTIONS reminder fires at 09:30, 11:00 and 12:30 PT. On a half-day the
# market shuts at 13:00 ET = 10:00 PT, so two of those three are after the
# close — and would otherwise look exactly like a working reminder.
half = date(2026, 11, 27)
for hh, mm, want in ((9, 30, True), (11, 0, False), (12, 30, False)):
    when = datetime(half.year, half.month, half.day, hh, mm, tzinfo=PT)
    check(f"{hh:02d}:{mm:02d} PT on the day after Thanksgiving → "
          f"{'open' if want else 'SHUT'}", mc.is_open(when) is want,
          mc.describe(when))
# ...and all three land inside a normal session.
for hh, mm in ((9, 30), (11, 0), (12, 30)):
    when = datetime(2026, 10, 6, hh, mm, tzinfo=PT)
    check(f"{hh:02d}:{mm:02d} PT on a normal Tuesday → open", mc.is_open(when),
          mc.describe(when))

check("close_time is None on a closed day", mc.close_time(date(2026, 12, 25)) is None)
check("close_time is 13:00 ET on a half-day",
      mc.close_time(date(2026, 11, 27)) == time(13, 0))
check("close_time is 16:00 ET normally",
      mc.close_time(date(2026, 10, 6)) == time(16, 0))

print("\n── timezone handling ──")
# A naive datetime is read as EASTERN, not as whatever the host clock says.
# The two Pis have disagreed about their timezone before, and getting this
# wrong would be wrong for exactly six hours a day.
check("a naive datetime is read as Eastern",
      mc.is_open(datetime(2026, 10, 6, 11, 0))
      and not mc.is_open(datetime(2026, 10, 6, 7, 0)))
check("describe() says WHY it is shut",
      "Thanksgiving" in mc.describe(datetime(2026, 11, 26, 11, 0, tzinfo=ET)),
      mc.describe(datetime(2026, 11, 26, 11, 0, tzinfo=ET)))

print(f"\n{'ALL PASS' if not FAILED else str(len(FAILED)) + ' FAILED: ' + ', '.join(FAILED)}")
raise SystemExit(1 if FAILED else 0)
