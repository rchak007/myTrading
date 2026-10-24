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
ACTIONS = {"ADD", "REPLACE", "DELETE"}

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
