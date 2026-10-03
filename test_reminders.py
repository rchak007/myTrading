#!/usr/bin/env python3
"""
test_reminders.py
=================
Tests for the reminder cadence logic. No network, no email.

    .venv/bin/python test_reminders.py

`due()` decides whether Chakravarti hears about something. Both ways it can be
wrong are bad in different directions: too eager turns a reminder into noise
he learns to ignore, too quiet loses the item entirely.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reminders as R                                    # noqa: E402

FAILED = []
TODAY = date.today()


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def row(**kw):
    base = {"id": "x", "added": "2026-01-01", "every_days": "2",
            "weekday": "", "last_sent": "", "done": "", "title": "t"}
    return base | kw


def ago(n):
    return (TODAY - timedelta(days=n)).isoformat()


print("\n── the day counter ──")
check("never sent is due immediately — the point of adding it",
      R.due(row(last_sent="")))
check("sent today is not due", not R.due(row(last_sent=TODAY.isoformat())))
check("sent 1 day ago, every 2, not due", not R.due(row(last_sent=ago(1))))
check("sent 2 days ago, every 2, due", R.due(row(last_sent=ago(2))))
check("struck out is never due",
      not R.due(row(last_sent="", done="2026-10-02")))
check("a junk every_days falls back to 2, it does not crash",
      R.due(row(every_days="soon", last_sent=ago(5))))
check("every_days 0 means EVERY send — how the options channel works",
      R.due(row(every_days="0", last_sent=TODAY.isoformat())))

print("\n── pinned to a weekday ──")
# A day counter CANNOT hold a weekday. every_days=7 drifts the first time a
# send is missed or late, and "every Monday" quietly becomes "every Thursday".
mon, other = "Mon", R.WEEKDAY_NAME[(TODAY.weekday() + 3) % 7][:3]
is_mon = TODAY.weekday() == 0
check(f"a Monday row is due only on Monday (today is {TODAY:%A})",
      R.due(row(weekday=mon, last_sent="")) is is_mon)
check("a row pinned to TODAY is due",
      R.due(row(weekday=R.WEEKDAY_NAME[TODAY.weekday()][:3], last_sent="")))
check("...but not twice on the same day",
      not R.due(row(weekday=R.WEEKDAY_NAME[TODAY.weekday()][:3],
                    last_sent=TODAY.isoformat())))
check("...and it IS due again if the last send was a week ago",
      R.due(row(weekday=R.WEEKDAY_NAME[TODAY.weekday()][:3], last_sent=ago(7))))
check("a row pinned to another day is not due",
      not R.due(row(weekday=other, last_sent="")))
check("weekday beats every_days rather than fighting it",
      not R.due(row(weekday=other, every_days="0", last_sent="")))
check("struck out still wins over a weekday",
      not R.due(row(weekday=R.WEEKDAY_NAME[TODAY.weekday()][:3],
                    done="x", last_sent="")))

print("\n── parsing what a human might type ──")
for raw, want in (("Mon", 0), ("mon", 0), ("MONDAY", 0), ("friday", 4),
                  ("Sun", 6), ("0", 0), ("3", 3), ("", None),
                  ("someday", None), ("9", None), ("-1", None)):
    check(f"weekday {raw!r} → {want}", R.weekday_of({"weekday": raw}) == want,
          str(R.weekday_of({"weekday": raw})))

print("\n── the real files still parse ──")
for ch in sorted(R.CHANNELS):
    R.use_channel(ch)
    rows = R.read_rows()
    check(f"{ch}: loads rows", len(rows) > 0, str(len(rows)))
    check(f"{ch}: every row has an id", all(r.get("id") for r in rows))
    subj, body = R.compose([r for r in rows if not r.get("done")] or rows[:1])
    check(f"{ch}: composes a subject", bool(subj), subj)
    check(f"{ch}: the body names how to stop one", "TO STOP ONE" in body)
R.use_channel("open-items")
ids = {r["id"] for r in R.read_rows()}
check("the Monday screenshot reminder exists", "chitra-screenshots" in ids,
      str(sorted(ids)))
screenshot = next(r for r in R.read_rows() if r["id"] == "chitra-screenshots")
check("...and it is pinned to Monday", R.weekday_of(screenshot) == 0)
check("...and the email will say 'every Monday', not 'every 2 day(s)'",
      "every Monday" in R.compose([screenshot])[1],
      R.compose([screenshot])[1].split("\n")[3])

print(f"\n{'ALL PASS' if not FAILED else str(len(FAILED)) + ' FAILED: ' + ', '.join(FAILED)}")
raise SystemExit(1 if FAILED else 0)
