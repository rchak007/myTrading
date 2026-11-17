#!/usr/bin/env python3
"""
ticker_notes.py
===============
A sticky note per ticker. Chakravarti's, 2026-10-23: *"PBLS — 20 shares I
never sell and only trade the remaining."*

WHY THIS IS NOT A COLUMN IN A SPREADSHEET. Everything on the Dashboard is
regenerated wholesale every cycle, so anything typed into a cell is gone on
the next run. That is deliberate and worth keeping — it is why the tab can
never drift from the data. A note therefore has to live where the generator
can read it, which means a file.

APPEND-ONLY, LIKE THE RESERVE LEDGER. Every action is a row:

    ADD      a dated fragment on the end of whatever is there
    REPLACE  throw the fragments away, start again with this one
    DELETE   clear it

The current note is a FOLD of the log, never a field that gets overwritten.
That is what makes "if note is deleted its ok, maybe archive log can have it"
true for free: a delete writes a row, it does not erase one, so every word
ever written is still in the file and `history()` will read it back.

EVERY FRAGMENT CARRIES ITS OWN DATE. Not one date for the note — a note built
up over three months is three observations, and which was made when is most
of the value. Appending today never restamps what you wrote in October.
"""
from __future__ import annotations

import csv
import fcntl
import os
import uuid
from datetime import datetime
from pathlib import Path

try:
    import pytz
    _PST = pytz.timezone("America/Los_Angeles")
except Exception:                                    # pragma: no cover
    _PST = None

STATE_DIR = Path(os.environ.get(
    "MYTRADING_STATE_DIR", Path.home() / ".local" / "state" / "myTrading"))
NOTES_PATH = STATE_DIR / "ticker_notes.csv"

COLS = ["Timestamp_PST", "Event_ID", "Ticker", "Action", "Text", "Source"]

# ATTENDED is a note action, not a separate ledger. Chakravarti,
# 2026-11-13: *"since i am managing so many tickers - i also need a verb
# called lastAttendedTo ... so when i browse i know which i took care of."*
#
# It rides here because it is the same shape as a note — per-ticker,
# dated, append-only — and because the history then lives in one file. An
# ATTENDED row carries no text and never appears in the note itself; it is
# folded separately by attended().
# DUE rides here for the same reasons ATTENDED does. Chakravarti,
# 2026-11-16: a date by which a ticker needs a decision, so the DUE tab can
# sort the whole universe by what is overdue.
#
# ATTENDED answers "when did I last look at this"; DUE answers "when must I
# look next". Same shape, same file, one history — and the pair is far more
# useful together than either alone, which is the argument for not splitting
# them across two ledgers.
#
# The date is the row's Text. Clearing a due date writes a DUE row with an
# EMPTY Text rather than deleting anything: append-only, so "it used to be
# due on the 3rd and I cleared it" stays answerable.
ACTIONS = {"ADD", "REPLACE", "DELETE", "ATTENDED", "DUE"}

# One note should not be able to push a Dashboard row past what Sheets will
# render, or bury the numbers underneath it.
MAX_TEXT = 500


def _now() -> str:
    if _PST is not None:
        return datetime.now(_PST).strftime("%Y-%m-%d %H:%M:%S %Z")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def tkr(t) -> str:
    return str(t or "").strip().upper()


def read_log(path: Path = NOTES_PATH) -> list[dict]:
    """Every row, oldest first. Empty list if the file is absent or unreadable."""
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            return [r for r in csv.DictReader(fh) if r.get("Ticker")]
    except Exception:
        return []


def _append(ticker: str, action: str, text: str, source: str,
            path: Path = NOTES_PATH, log=print) -> None:
    """One row, under a lock. Same contract as the reserve ledger."""
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            w = csv.DictWriter(fh, fieldnames=COLS)
            if new:
                w.writeheader()
            w.writerow({
                "Timestamp_PST": _now(),
                "Event_ID": uuid.uuid4().hex[:12],
                "Ticker": tkr(ticker),
                "Action": action,
                "Text": text,
                "Source": source,
            })
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    log(f"ticker_notes: {action} {tkr(ticker)} -> {path}")


def attend(ticker, source="cli", path: Path = NOTES_PATH, log=print) -> str:
    """Stamp a ticker as looked-at today. Returns the date written."""
    _append(ticker, "ATTENDED", "", source, path=path, log=log)
    return _now()[:10]


def attended(path: Path = NOTES_PATH) -> dict[str, str]:
    """{TICKER: YYYY-MM-DD} — the LAST time each was attended to.

    Last, not first: re-attending overwrites, because the question is
    always "when did I last look at this", never "when did I start".
    """
    out: dict[str, str] = {}
    for r in read_log(path):
        if str(r.get("Action", "")).strip().upper() == "ATTENDED":
            out[tkr(r.get("Ticker"))] = str(r.get("Timestamp_PST", ""))[:10]
    return out


def parse_due(text) -> str:
    """A typed due date into YYYY-MM-DD, or "" to clear it.

    Accepts what a person actually types on a phone:

        2026-12-01   an ISO date
        12/01        this year, or next if that is already past
        30d  4w  3m  relative to today
        today  tomorrow  eom
        -  none  clear  x        clear the date

    Raises ValueError on anything else. A due date that silently parsed to
    the wrong day would be worse than a refusal — the whole tab sorts on it.
    """
    from datetime import date as _date, timedelta
    s = str(text or "").strip().lower()
    if s in ("", "-", "none", "clear", "x", "off"):
        return ""
    today = _date.today()
    if s == "today":
        return today.isoformat()
    if s == "tomorrow":
        return (today + timedelta(days=1)).isoformat()
    if s == "eom":
        nxt = (today.replace(day=28) + timedelta(days=4))
        return (nxt - timedelta(days=nxt.day)).isoformat()

    import re
    m = re.fullmatch(r"\+?(\d+)\s*([dwmy])", s)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        days = {"d": 1, "w": 7, "m": 30, "y": 365}[unit] * n
        return (today + timedelta(days=days)).isoformat()

    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        return _date(y, mo, d).isoformat()        # ValueError if not real

    # M/D or M/D/Y. Bare M/D means the NEXT occurrence, so typing 1/15 in
    # December means next January rather than a date eleven months past.
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?", s)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if y:
            y = int(y)
            y += 2000 if y < 100 else 0
            return _date(y, mo, d).isoformat()
        cand = _date(today.year, mo, d)
        if cand < today:
            cand = _date(today.year + 1, mo, d)
        return cand.isoformat()

    raise ValueError(
        f"{text!r} is not a date. Use 2026-12-01, 12/01, 30d, 4w, 3m, "
        f"today, tomorrow, eom — or '-' to clear it.")


def set_due(ticker, when, *, source="cli", path: Path = NOTES_PATH,
            log=print) -> str:
    """Set (or clear) a ticker's due date. Returns the date written, or ""."""
    iso = parse_due(when)
    _append(ticker, "DUE", iso, source, path=path, log=log)
    return iso


def due_dates(path: Path = NOTES_PATH) -> dict[str, str]:
    """{TICKER: YYYY-MM-DD} — the CURRENT due date for each ticker.

    Last row wins, and a row with an empty date removes the ticker from the
    mapping entirely rather than mapping it to "". Callers then only have to
    handle "present" and "absent", not a third empty-string case that reads
    as a date everywhere it is forgotten.
    """
    out: dict[str, str] = {}
    for r in read_log(path):
        if str(r.get("Action", "")).strip().upper() != "DUE":
            continue
        t, iso = tkr(r.get("Ticker")), str(r.get("Text", "")).strip()
        if iso:
            out[t] = iso
        else:
            out.pop(t, None)
    return out


def days_until(ticker, path: Path = NOTES_PATH):
    """Whole days until this ticker is due. Negative is overdue; None unset."""
    from datetime import date as _date
    d = due_dates(path).get(tkr(ticker))
    if not d:
        return None
    try:
        y, m, dd = (int(x) for x in d.split("-"))
        return (_date(y, m, dd) - _date.today()).days
    except Exception:                                    # noqa: BLE001
        return None


def days_since(ticker, path: Path = NOTES_PATH):
    """Whole days since this ticker was attended to, or None."""
    from datetime import date as _date
    d = attended(path).get(tkr(ticker))
    if not d:
        return None
    try:
        y, m, dd = (int(x) for x in d.split("-"))
        return (_date.today() - _date(y, m, dd)).days
    except Exception:                                    # noqa: BLE001
        return None


# ---------------------------------------------------------------------
# The fold
# ---------------------------------------------------------------------
def current(path: Path = NOTES_PATH) -> dict[str, list[tuple[str, str]]]:
    """{TICKER: [(YYYY-MM-DD, text), ...]} — the live note for every ticker.

    Rows are replayed in file order, which is chronological because the file
    is only ever appended to. A ticker whose last meaningful action was DELETE
    does not appear at all.
    """
    out: dict[str, list[tuple[str, str]]] = {}
    for r in read_log(path):
        t = tkr(r.get("Ticker"))
        action = str(r.get("Action", "")).strip().upper()
        date = str(r.get("Timestamp_PST", ""))[:10]
        text = str(r.get("Text", "")).strip()
        if action == "DELETE":
            out.pop(t, None)
        elif action == "REPLACE":
            out[t] = [(date, text)] if text else []
            if not text:
                out.pop(t, None)
        elif action == "ADD" and text:
            out.setdefault(t, []).append((date, text))
    return {t: frags for t, frags in out.items() if frags}


def render(ticker, path: Path = NOTES_PATH, sep: str = "  ·  ") -> str:
    """One line for the sheet: 'Oct 23: never sell 20 · Nov 01: trimmed 5'."""
    frags = current(path).get(tkr(ticker), [])
    return sep.join(f"{d}: {txt}" for d, txt in frags)


def history(ticker, path: Path = NOTES_PATH) -> list[dict]:
    """EVERY row for a ticker, deletes included. This is the archive."""
    return [r for r in read_log(path) if tkr(r.get("Ticker")) == tkr(ticker)]


# ---------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------
def _check(ticker, text) -> tuple[str, str]:
    t = tkr(ticker)
    if not t:
        raise ValueError("need a ticker")
    text = str(text or "").strip()
    if not text:
        raise ValueError("note text is empty — use note_delete to clear one")
    if len(text) > MAX_TEXT:
        raise ValueError(f"note is {len(text)} chars; keep it under {MAX_TEXT}")
    return t, text


def add(ticker, text, *, source="cli", path: Path = NOTES_PATH, log=print) -> str:
    """Append a dated fragment, creating the note if there is none."""
    t, text = _check(ticker, text)
    _append(t, "ADD", text, source, path, log=lambda *_: None)
    note = render(t, path)
    log(f"NOTE {t} — now reads:\n  {note}")
    return note


def replace(ticker, text, *, source="cli", path: Path = NOTES_PATH,
            log=print) -> str:
    """Throw away every existing fragment and start again with this one.

    The old text is NOT lost — it stays in the log above this row, which is
    what history() reads.
    """
    t, text = _check(ticker, text)
    was = render(t, path)
    _append(t, "REPLACE", text, source, path, log=lambda *_: None)
    note = render(t, path)
    log(f"NOTE {t} replaced.\n  was:  {was or '(nothing)'}\n  now:  {note}")
    return note


def delete(ticker, *, source="cli", path: Path = NOTES_PATH, log=print) -> str:
    """Clear the note. Refuses when there is nothing there, so a typo'd
    ticker says so instead of silently succeeding."""
    t = tkr(ticker)
    was = render(t, path)
    if not was:
        raise ValueError(f"{t} has no note to delete")
    _append(t, "DELETE", "", source, path, log=lambda *_: None)
    log(f"NOTE {t} deleted. It stays in the log — "
        f"{len(history(t, path))} row(s) of history kept.\n  was:  {was}")
    return was


def summary(path: Path = NOTES_PATH) -> str:
    """Every live note, for the `notes` verb."""
    live = current(path)
    if not live:
        return "no notes"
    lines = [f"{len(live)} ticker(s) with notes:", ""]
    for t in sorted(live):
        lines.append(f"{t}")
        for d, txt in live[t]:
            lines.append(f"   {d}  {txt}")
    return "\n".join(lines)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Per-ticker sticky notes")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--add", nargs="+", metavar=("TICKER", "TEXT"))
    ap.add_argument("--replace", nargs="+", metavar=("TICKER", "TEXT"))
    ap.add_argument("--delete", metavar="TICKER")
    ap.add_argument("--history", metavar="TICKER")
    a = ap.parse_args()

    if a.add:
        add(a.add[0], " ".join(a.add[1:]))
    elif a.replace:
        replace(a.replace[0], " ".join(a.replace[1:]))
    elif a.delete:
        delete(a.delete)
    elif a.history:
        for r in history(a.history):
            print(f"{r['Timestamp_PST']}  {r['Action']:<8} {r['Text']}")
    else:
        print(summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
