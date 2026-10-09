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

print("\n── escalating: fire on the day, then daily until delivered ──")
# "send this every Sunday. and then till i confirm pasting here everyday. But
# once i send set this to be sent on Sunday again."
from datetime import date as _date                        # noqa: E402
SUN = R.last_occurrence(6)
esc = lambda **kw: row(weekday="Sun", escalate="Y", **kw)
check("never delivered → due, whatever day it is",
      R.due(esc(last_sent="", last_done="")))
check("...but only once a day",
      not R.due(esc(last_sent=TODAY.isoformat(), last_done="")))
check("delivered since the last Sunday → quiet",
      not R.due(esc(last_sent="", last_done=SUN.isoformat())))
check("delivered BEFORE the last Sunday → due again",
      R.due(esc(last_sent="", last_done=(SUN - timedelta(days=1)).isoformat())))
# last_sent must NOT satisfy it — that is the whole point.
check("being SENT does not satisfy it, only being DELIVERED",
      R.due(esc(last_sent=ago(1), last_done="")))
check("struck out still wins",
      not R.due(esc(last_sent="", last_done="", done="x")))
check("without escalate it is Sunday-only",
      R.due(row(weekday=R.WEEKDAY_NAME[TODAY.weekday()][:3], last_sent=""))
      and not R.due(row(weekday="Sun", last_sent="")) or TODAY.weekday() == 6)

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
check("...and it is pinned to Sunday", R.weekday_of(screenshot) == 6,
      str(R.weekday_of(screenshot)))
check("...and it escalates",
      str(screenshot.get("escalate", "")).upper().startswith("Y"),
      repr(screenshot.get("escalate")))
check("...and the email says so, not 'every 2 day(s)'",
      "every Sunday, then DAILY until you send it" in R.compose([screenshot])[1],
      R.compose([screenshot])[1].split("\n")[3])

print("\n── the principle cards ride on OPTIONS, once a day ──")
# Moved off the open-items email, which goes quiet the moment everything is
# struck out — principles you only see while you happen to owe a task are
# principles you stop seeing. But three a day with 1.3 MB attached is a
# mailbox problem, so only the first send of each trading day carries them.
check("open-items no longer carries cards",
      R.CHANNELS["open-items"]["images"] is False)
check("options carries them on the FIRST send only",
      R.CHANNELS["options"]["images"] == "first")
check("the options subject says OPTIONS / PRINCIPLES",
      "OPTIONS / PRINCIPLES" in R.CHANNELS["options"]["subject"]
      and "OPTIONS / PRINCIPLES" in R.CHANNELS["options"]["subject1"])
R.use_channel("options")
check("there are cards to send", len(R.notes_images()) == 4,
      str([p.name for p in R.notes_images()]))
check("...and they are ordered by their numeric prefix",
      [p.name[:2] for p in R.notes_images()] == ["01", "02", "03", "04"])
R.use_channel("open-items")

print("\n── Pi-1 poll staleness is measured against the SCHEDULE ──")
# THE FALSE ALARM, 2026-10-03. A flat "3 hours" threshold emailed
# "🔴 Pi 1 has not polled in 16h" at 08:30 on a SATURDAY. Friday's last slot
# is 16:50 and the next is Monday 01:15, so 16 hours was exactly right.
from datetime import datetime                              # noqa: E402
from zoneinfo import ZoneInfo                              # noqa: E402
import token_watch as T                                    # noqa: E402
PT = ZoneInfo("America/Los_Angeles")
D = lambda *a: datetime(*a, tzinfo=PT)
for name, now_, poll_, want in (
        ("Sat 08:30 after a normal Friday — the false alarm",
         D(2026, 10, 3, 8, 30), D(2026, 10, 2, 16, 58), False),
        ("Sun 20:00", D(2026, 10, 4, 20, 0), D(2026, 10, 2, 16, 58), False),
        ("Mon 01:10, before the day's first slot",
         D(2026, 10, 5, 1, 10), D(2026, 10, 2, 16, 58), False),
        ("Mon 02:10, 01:15 was skipped",
         D(2026, 10, 5, 2, 10), D(2026, 10, 2, 16, 58), True),
        ("Mon 09:00 after a genuinely dead weekend",
         D(2026, 10, 5, 9, 0), D(2026, 10, 2, 16, 58), True),
        ("Tue 11:00, polled 10:52 — healthy",
         D(2026, 10, 6, 11, 0), D(2026, 10, 6, 10, 52), False),
        ("Tue 11:05, a slot missed but still inside its grace",
         D(2026, 10, 6, 11, 5), D(2026, 10, 6, 10, 20), False),
        ("Tue 11:35, that slot now past grace",
         D(2026, 10, 6, 11, 35), D(2026, 10, 6, 10, 20), True),
        ("Fri 22:00, after the last slot of the week",
         D(2026, 10, 2, 22, 0), D(2026, 10, 2, 16, 58), False)):
    check(f"{name} → {'ALERT' if want else 'quiet'}",
          T.poll_is_stale(poll_, now_) is want)
check("an unparseable timestamp alerts on nothing",
      T.poll_is_stale(None) is False)

print(f"\n{'ALL PASS' if not FAILED else str(len(FAILED)) + ' FAILED: ' + ', '.join(FAILED)}")
raise SystemExit(1 if FAILED else 0)
