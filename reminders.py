#!/usr/bin/env python3
"""
reminders.py
============
Email the things still outstanding in reminders.csv, and keep doing it.

    .venv/bin/python reminders.py --list      # show state, send nothing
    .venv/bin/python reminders.py --dry-run   # print the email
    .venv/bin/python reminders.py             # send what is due
    .venv/bin/python reminders.py --done <id> # strike one out

RUNS ON PI 2, which holds the Gmail credentials. It needs nothing from Schwab
and nothing from Pi 1.

WHY A FILE IN THE REPO
    A reminder that lived in machine-local state would vanish with an SD card,
    and a reminder you cannot see is worse than none. In the repo it is
    versioned, visible in a diff, and editable from either machine.

ONE EMAIL
    When anything is due, EVERY outstanding item goes in that one email and all
    of them have their clock reset. Otherwise items added on different days
    drift onto separate schedules and arrive as separate mails — which is how a
    reminder turns into noise you learn to ignore.

STRIKING OUT
    Put anything in the `done` column. Nothing is deleted — a struck row stays
    as a record that it was finished, and re-reading an old one is occasionally
    the point.

INFO NOTES
    Images in ./reminder_notes/ ride along INLINE at the end of every email.
    They are not to-dos and are never struck out — they are the things worth
    glancing at repeatedly. The IA house rules card is the first one.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

COLS = ["id", "added", "every_days", "weekday", "escalate", "last_sent",
        "last_done", "done", "title", "url", "note"]
# Mon=0, matching date.weekday(). Accepts "Mon", "monday", or the number.
WEEKDAYS = {n: i for i, n in enumerate(
    ["mon", "tue", "wed", "thu", "fri", "sat", "sun"])}
WEEKDAY_NAME = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                "Saturday", "Sunday"]


def last_occurrence(weekday: int, today: date | None = None) -> date:
    """The most recent date on or before today falling on `weekday`."""
    today = today or date.today()
    return today - timedelta(days=(today.weekday() - weekday) % 7)


def weekday_of(row) -> int | None:
    """The weekday a row is pinned to, or None for a plain day-counter row."""
    raw = str(row.get("weekday", "")).strip().lower()
    if not raw:
        return None
    if raw[:3] in WEEKDAYS:
        return WEEKDAYS[raw[:3]]
    try:
        n = int(raw)
        return n if 0 <= n <= 6 else None
    except ValueError:
        return None

# ─────────────────────────────────────────────────────────── channels
# Two lists, two cadences, one nagger. A second copy of this file for the
# options list would be a second set of the Gmail gotchas to keep in sync, and
# those have cost real debugging once already.
#
#   open-items  daily at 09:15. `every_days` gates each row. Carries the
#               reminder_notes/ cards.
#   options     09:30 / 11:00 / 12:30 PT, and ONLY while the market is open.
#               No per-row cadence — cron already decides when, so every
#               outstanding row goes in every send. No images: three a day
#               with a megabyte of cards attached is not a reminder.
CHANNELS = {
    "open-items": {
        "file": "reminders.csv",
        "emoji": "\U0001F4CC",
        "what": "still to look at",
        "subject": "{emoji} {n} thing{s} still to look at",
        "subject1": "{emoji} Still to look at: {first}",
        # The cards moved to the OPTIONS channel: this one goes quiet the
        # moment everything is struck out, and principles you only see while
        # you happen to owe a task are principles you stop seeing.
        "intro": ["These are outstanding in reminders.csv. They will keep arriving",
                  "until struck out."],
        "images": False,
        "market_hours": False,
    },
    "options": {
        "file": "options_reminders.csv",
        "emoji": "\U0001F3AF",
        "what": "OPTIONS / PRINCIPLES",
        "subject": "{emoji} OPTIONS / PRINCIPLES \u2014 {n} to act on while the market is open",
        "subject1": "{emoji} OPTIONS / PRINCIPLES \u2014 {first}",
        "intro": ["Sent at 09:30, 11:00 and 12:30 PT on trading days only.",
                  "These are the moves that need the market open to make."],
        # "first" = the FIRST send of each trading day only. Three a day with
        # 1.3 MB of cards attached is a mailbox problem, and the principles
        # are for planning at the open, not for the 12:30 nudge.
        "images": "first",
        "market_hours": True,
    },
}
CHANNEL_NAME = "open-items"
CHANNEL = CHANNELS[CHANNEL_NAME]
FILE = Path(os.getenv("REMINDERS_FILE", HERE / CHANNEL["file"]))


def use_channel(name: str) -> dict:
    """Point the module at one channel's list. Returns its config."""
    global CHANNEL, CHANNEL_NAME, FILE
    CHANNEL_NAME = name
    CHANNEL = CHANNELS[name]
    FILE = Path(os.getenv("REMINDERS_FILE", HERE / CHANNEL["file"]))
    return CHANNEL

# INFO NOTES. Every image in here is shown inline at the END of every reminder
# email — not tied to any one row, and not struck out with them. These are the
# things worth re-reading periodically rather than doing once: the IA house
# rules went in first. Drop another picture in the folder and it joins them; no
# code change.
NOTES_DIR = Path(os.getenv("REMINDER_NOTES_DIR", HERE / "reminder_notes"))
NOTE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def notes_images() -> list[Path]:
    """Sorted by filename — hence the `01-`, `02-` prefixes.

    Filesystem order is arbitrary, and plain names would put the sequence at
    the mercy of the alphabet. The number makes the running order a decision.
    """
    if not NOTES_DIR.is_dir():
        return []
    return sorted(p for p in NOTES_DIR.iterdir()
                  if p.is_file() and p.suffix.lower() in NOTE_SUFFIXES)


def read_rows() -> list[dict]:
    if not FILE.exists():
        return []
    with FILE.open(newline="", encoding="utf-8") as fh:
        # Comment lines carry the usage notes; csv would treat them as data.
        body = [l for l in fh if not l.lstrip().startswith("#")]
    return [dict(r) for r in csv.DictReader(body)]


def write_rows(rows: list[dict]) -> None:
    """Rewrite the data, preserving the comment header.

    This file IS the state, unlike the append-only money ledgers — a reminder
    has no history worth keeping beyond whether it is still outstanding.
    """
    head = []
    if FILE.exists():
        with FILE.open(encoding="utf-8") as fh:
            for line in fh:
                if line.lstrip().startswith("#"):
                    head.append(line)
                else:
                    break
    tmp = FILE.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        fh.writelines(head)
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})
    tmp.replace(FILE)


def days_since(stamp: str) -> float | None:
    try:
        return (date.today() - datetime.fromisoformat(stamp[:10]).date()).days
    except Exception:
        return None


def due(row: dict) -> bool:
    """Outstanding, and enough days since the last nag.

    A row never sent is due immediately — the point of adding it is to be
    reminded, not to wait a cycle first.
    """
    if str(row.get("done", "")).strip():
        return False

    # PINNED TO A WEEKDAY. A day counter cannot hold a weekday: `every_days=7`
    # drifts the moment one send is missed or late, and "every Monday" quietly
    # becomes "every Thursday". So a weekday row is due on that day and only
    # that day, once.
    wd = weekday_of(row)
    if wd is not None:
        # ESCALATING: fire on the day, then keep firing EVERY day until the
        # thing is actually delivered. Chakravarti's ask, verbatim: "send this
        # every Sunday. and then till i confirm pasting here everyday. But
        # once i send set this to be sent on Sunday again."
        #
        # `last_done` is what stops it — NOT last_sent. A reminder that
        # silences itself by being sent is a reminder that never gets
        # anything done.
        if str(row.get("escalate", "")).strip().upper().startswith("Y"):
            if days_since(row.get("last_sent", "")) == 0:
                return False                     # already nagged today
            done_on = days_since(row.get("last_done", ""))
            if done_on is None:
                return True                      # never delivered
            delivered = date.today() - timedelta(days=int(done_on))
            return delivered < last_occurrence(wd)
        if date.today().weekday() != wd:
            return False
        return days_since(row.get("last_sent", "")) != 0

    try:
        every = int(row.get("every_days") or 2)
    except ValueError:
        every = 2
    # 0 means EVERY send. That is how the options channel works: cron already
    # decides the three times a day, so a per-row day counter would fight it.
    if every <= 0:
        return True
    since = days_since(row.get("last_sent", ""))
    return True if since is None else since >= every


def compose(rows: list[dict]) -> tuple[str, str]:
    n = len(rows)
    tpl = CHANNEL["subject"] if n != 1 else CHANNEL["subject1"]
    subject = tpl.format(emoji=CHANNEL["emoji"], n=n,
                         s="s" if n != 1 else "",
                         first=rows[0]["title"][:60])
    body = list(CHANNEL["intro"]) + [""]
    for r in rows:
        age = days_since(r.get("added", ""))
        body.append(f"• {r.get('title') or r['id']}")
        if r.get("url"):
            body.append(f"    {r['url']}")
        if r.get("note"):
            body.append(f"    {r['note']}")
        wd = weekday_of(r)
        esc = str(r.get("escalate", "")).strip().upper().startswith("Y")
        cadence = ("" if CHANNEL["market_hours"]
                   else (f" · every {WEEKDAY_NAME[wd]}"
                         + (", then DAILY until you send it" if esc else ""))
                   if wd is not None
                   else f" · every {r.get('every_days','2')} day(s)")
        body.append(f"    added {r.get('added','?')}"
                    + (f", {age} day(s) ago" if age is not None else "")
                    + cadence)
        body.append("")
    body += ["TO STOP ONE",
             f"  On Pi 2:  .venv/bin/python reminders.py"
             + ("" if CHANNEL_NAME == "open-items" else f" --channel {CHANNEL_NAME}")
             + "  --done <id>",
             f"  Or put anything in the `done` column of {FILE.name}.",
             "",
             "  ids: " + ", ".join(r["id"] for r in rows),
             ""]
    if CHANNEL["images"] and notes_images():
        body += ["PRINCIPLES below — not to-dos. They are here to be re-read.",
                 ""]
    body.append(f"— reminders.py ({CHANNEL_NAME}) on Pi 2")
    return subject, "\n".join(body)


def main() -> int:
    ap = argparse.ArgumentParser(description="Nag until struck out")
    ap.add_argument("--list", action="store_true", help="show state, send nothing")
    ap.add_argument("--dry-run", action="store_true", help="print the email")
    ap.add_argument("--done", metavar="ID", help="strike one out")
    ap.add_argument("--delivered", metavar="ID",
                    help="mark an escalating row satisfied — it drops back to "
                         "its weekday until the next one comes round")
    ap.add_argument("--force", action="store_true", help="send even if not due")
    ap.add_argument("--channel", default="open-items", choices=sorted(CHANNELS),
                    help="which list (default: open-items)")
    args = ap.parse_args()
    use_channel(args.channel)

    rows = read_rows()
    if not rows:
        print(f"no reminders in {FILE}")
        return 0

    if args.done:
        hit = [r for r in rows if r["id"] == args.done]
        if not hit:
            print(f"no reminder with id {args.done!r}. "
                  f"ids: {', '.join(r['id'] for r in rows)}")
            return 1
        hit[0]["done"] = date.today().isoformat()
        write_rows(rows)
        print(f"struck out {args.done} — {hit[0].get('title','')}")
        return 0

    if args.delivered:
        hit = [r for r in rows if r["id"] == args.delivered]
        if not hit:
            print(f"no reminder with id {args.delivered!r}")
            return 1
        hit[0]["last_done"] = date.today().isoformat()
        write_rows(rows)
        wd = weekday_of(hit[0])
        print(f"{args.delivered} marked delivered — quiet until "
              + (WEEKDAY_NAME[wd] if wd is not None else "its next turn"))
        return 0

    if args.list:
        for r in rows:
            mark = "✔" if str(r.get("done", "")).strip() else ("→" if due(r) else " ")
            print(f" {mark} {r['id']:<20} last_sent={r.get('last_sent') or 'never':<12} "
                  f"{r.get('title','')[:50]}")
        return 0

    # ONE EMAIL, ALL OUTSTANDING. If anything is due, everything still open
    # goes in it — otherwise items added on different days drift onto their own
    # schedules and arrive as separate mails, which is how a reminder becomes
    # noise. Chakravarti asked for one every two days, not one per item.
    outstanding = [r for r in rows if not str(r.get("done", "")).strip()]
    anything_due = args.force or any(due(r) for r in outstanding)
    pending = outstanding if anything_due else []
    if not pending:
        print(f"{CHANNEL_NAME}: nothing due ({len(outstanding)} outstanding, "
              f"none ready to re-send)")
        return 0

    # MARKET HOURS GATE. Checked here rather than in cron, because cron cannot
    # know about Good Friday or that the day after Thanksgiving shuts at 10:00
    # PT — so a 11:00 reminder would fire into a closed market three times a
    # year and look exactly like a working one.
    if CHANNEL["market_hours"] and not args.force:
        try:
            import market_calendar
            if not market_calendar.is_open():
                print(f"market {market_calendar.describe()} — nothing sent")
                return 0
        except Exception as e:
            # Fail OPEN: a broken calendar should cost an extra email, never a
            # missed one. Say so, so it does not pass for normal.
            print(f"⚠️  market calendar unavailable ({e}) — sending anyway")

    subject, body = compose(pending)
    mode = CHANNEL["images"]
    first_today = not any(days_since(r.get("last_sent", "")) == 0
                          for r in outstanding)
    pics = (notes_images()
            if mode is True or (mode == "first" and first_today) else [])
    if args.dry_run:
        print(f"Subject: {subject}\n\n{body}")
        for p in pics:
            print(f"[inline image: {p.name}, {p.stat().st_size/1024:.0f} KB]")
        if CHANNEL["images"] and not pics:
            print(f"[cards held back — not the first send today]"
                  if CHANNEL["images"] == "first"
                  else f"[no info-note images in {NOTES_DIR}]")
        return 0

    from token_watch import send            # one mailer, one set of gotchas
    if not send(subject, body, images=pics):
        # Do NOT stamp last_sent on failure, or a refused send silently costs
        # a whole cycle — the same reason token_watch only marks on success.
        print("send failed — last_sent not updated, will retry next run")
        return 1

    today = date.today().isoformat()
    for r in pending:
        r["last_sent"] = today
    write_rows(rows)
    print(f"reminded about {len(pending)}: {', '.join(r['id'] for r in pending)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
