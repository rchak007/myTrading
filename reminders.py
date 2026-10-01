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
from datetime import date, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

FILE = Path(os.getenv("REMINDERS_FILE", HERE / "reminders.csv"))
COLS = ["id", "added", "every_days", "last_sent", "done", "title", "url", "note"]

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
    since = days_since(row.get("last_sent", ""))
    if since is None:
        return True
    try:
        every = int(row.get("every_days") or 2)
    except ValueError:
        every = 2
    return since >= every


def compose(rows: list[dict]) -> tuple[str, str]:
    n = len(rows)
    subject = (f"📌 {n} thing{'s' if n != 1 else ''} still to look at"
               if n != 1 else f"📌 Still to look at: {rows[0]['title'][:60]}")
    body = ["These are outstanding in reminders.csv. They will keep arriving",
            "until struck out.", ""]
    for r in rows:
        age = days_since(r.get("added", ""))
        body.append(f"• {r.get('title') or r['id']}")
        if r.get("url"):
            body.append(f"    {r['url']}")
        if r.get("note"):
            body.append(f"    {r['note']}")
        body.append(f"    added {r.get('added','?')}"
                    + (f", {age} day(s) ago" if age is not None else "")
                    + f" · every {r.get('every_days','2')} day(s)")
        body.append("")
    body += ["TO STOP ONE",
             "  On Pi 2:  .venv/bin/python reminders.py --done <id>",
             "  Or put anything in the `done` column of reminders.csv.",
             "",
             "  ids: " + ", ".join(r["id"] for r in rows),
             ""]
    if notes_images():
        body += ["INFO NOTES below — not to-dos. They stay in every email.", ""]
    body.append("— reminders.py on Pi 2")
    return subject, "\n".join(body)


def main() -> int:
    ap = argparse.ArgumentParser(description="Nag until struck out")
    ap.add_argument("--list", action="store_true", help="show state, send nothing")
    ap.add_argument("--dry-run", action="store_true", help="print the email")
    ap.add_argument("--done", metavar="ID", help="strike one out")
    ap.add_argument("--force", action="store_true", help="send even if not due")
    args = ap.parse_args()

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
        print(f"nothing due ({len(outstanding)} outstanding, none ready "
              f"to re-send)")
        return 0

    subject, body = compose(pending)
    pics = notes_images()
    if args.dry_run:
        print(f"Subject: {subject}\n\n{body}")
        for p in pics:
            print(f"[inline image: {p.name}, {p.stat().st_size/1024:.0f} KB]")
        if not pics:
            print(f"[no info-note images in {NOTES_DIR}]")
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
