#!/usr/bin/env python3
"""
orders_tidy.py
==============
Sort the Orders tab: live rows at the top, finished ones at the bottom.

Chakravarti, 2026-11-08: *"can you re-arrange the rows so the ones down and
older are at bottom and those current are at the top and still waiting
etc."*

WHY THIS IS A VERB AND NOT PART OF EVERY REBUILD. The Orders tab is the one
tab he TYPES into. Every other tab is regenerated wholesale each cycle and
nothing is lost, but here a rewrite lands on top of whatever he is in the
middle of entering. Re-sorting on a schedule would eventually move a row
out from under the cursor. So it happens when asked, and only then.

AND IT MUST HOLD THE ENGINE'S LOCK. order_engine reads the tab, remembers
each record's POSITION as `_sheet_row`, and later writes Status, Validation
and the fill fields back to those row numbers. Sorting between its read and
its write would post every result against the wrong order — a FILLED stamped
onto someone else's row. flock on the engine's own lock makes the two
mutually exclusive; if the engine is running we do nothing and say so.

WHAT COUNTS AS LIVE. Anything the engine may still act on:

    (blank)    not yet seen
    ARMED      valid, watching for its condition
    WAITING    chained, parent has not filled
    TRIGGERED  condition met, placing
    SUBMITTED  resting at Schwab

and anything it has finished with is done:

    FILLED  BLOCKED  EXPIRED  VOID

ORDER IS PRESERVED WITHIN EACH GROUP. A stable sort, newest Date first. Two
rows typed the same day keep the order he typed them in, which is usually
the order he thinks about them in.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import orders_columns as oc                           # noqa: E402

LIVE = ("", "ARMED", "WAITING", "TRIGGERED", "SUBMITTED")
DONE = ("FILLED", "BLOCKED", "EXPIRED", "VOID")

LOCK = Path(os.getenv("ORDER_ENGINE_LOCK", "/tmp/order_engine.lock"))


def is_live(status) -> bool:
    return str(status or "").strip().upper() in LIVE


def sort_rows(rows: list[dict]) -> list[dict]:
    """Live first, then done; newest Date first inside each group.

    Pure, so the ordering can be tested without a spreadsheet. Rows with no
    Row_ID and no Ticker are spacers and are dropped — they are what makes a
    half-sorted tab look ragged, and they carry nothing.
    """
    keep = [r for r in rows
            if str(r.get("Row_ID", "")).strip()
            or str(r.get("Ticker", "")).strip()]

    def key(item):
        i, r = item
        # Descending by date within the group. A missing or unparseable date
        # sorts OLDEST, not newest: an undated row is not fresh news, and
        # putting it on top would reward the mistake.
        d = str(r.get("Date", "")).strip()[:10]
        return (0 if is_live(r.get("Status")) else 1,
                _desc(d),
                i)                      # stable within a day

    return [r for _, r in sorted(enumerate(keep), key=key)]


def _desc(d: str) -> str:
    """Sort key that puts the LATEST date first, blanks last."""
    if not d:
        # Larger than any inverted date, so blanks land at the BOTTOM of
        # their group. "2026-…" inverts to "7973-…", so 9s beat every real
        # one; "0000" would have put undated rows on top, which is exactly
        # backwards.
        return "9999-99-99"
    # Invert each digit so a plain ascending sort runs newest-first.
    return "".join(str(9 - int(c)) if c.isdigit() else c for c in d)


def apply(book=None, log=print) -> int:
    """Rewrite the Orders tab in sorted order. Returns rows moved, or -1.

    -1 means we declined because the engine holds its lock. That is a
    normal outcome, not a failure: the engine is mid-run and its row
    numbers are in use.
    """
    import fcntl
    try:
        fh = LOCK.open("a")
    except Exception as e:                               # noqa: BLE001
        log(f"cannot open {LOCK}: {e}")
        return -1
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("order_engine is running — not sorting while it holds row "
                "numbers. Try again in a minute.")
            return -1
        return _apply_locked(book, log)
    finally:
        fh.close()


def _apply_locked(book, log) -> int:
    import orders_sheet
    book = book or orders_sheet._open_book()
    ws = book.worksheet("Orders")
    grid = ws.get_all_values()

    hdr_i = oc.HEADER_ROW - 1
    if len(grid) <= hdr_i:
        log("Orders tab has no header — nothing to sort")
        return 0
    header = grid[hdr_i]
    names = [h.strip() for h in header]
    body = grid[oc.DATA_START_ROW - 1:]

    rows = []
    for raw in body:
        cells = list(raw) + [""] * (len(names) - len(raw))
        rows.append({n: str(cells[j]).strip()
                     for j, n in enumerate(names) if n})
    before = [r.get("Row_ID", "") for r in rows
              if r.get("Row_ID") or r.get("Ticker")]

    ordered = sort_rows(rows)
    after = [r.get("Row_ID", "") for r in ordered]
    if before == after:
        log(f"Orders tab already in order ({len(after)} row(s))")
        return 0

    out = [[r.get(n, "") for n in names] for r in ordered]
    # Blank the tail so dropped spacers and the shortened block do not leave
    # a ghost of the previous ordering underneath.
    pad = len(body) - len(out)
    out.extend([[""] * len(names) for _ in range(max(0, pad))])

    last = chr(64 + len(names)) if len(names) <= 26 else "AZ"
    ws.update(values=out,
              range_name=f"A{oc.DATA_START_ROW}:{last}"
                         f"{oc.DATA_START_ROW + len(out) - 1}",
              value_input_option="RAW")

    live = sum(1 for r in ordered if is_live(r.get("Status")))
    log(f"Orders tab sorted: {live} live row(s) on top, "
        f"{len(ordered) - live} finished below")

    # HOW FRESH IS THE THING WE SORTED ON. Chakravarti, 2026-11-08: *"you
    # might have to check the status to know its done no?"* — exactly. The
    # order reflects the Status column, and Status is only as current as the
    # engine's last poll. A row that filled at Schwab ten minutes ago still
    # reads SUBMITTED and still sorts as live. That is not wrong, but it is
    # worth saying out loud rather than letting a tidy tab imply freshness
    # it does not have.
    seen = sorted(str(r.get("Last_Checked", "")).strip()
                  for r in ordered if str(r.get("Last_Checked", "")).strip())
    if seen:
        log(f"   statuses as of the engine's last poll: {seen[-1]}"
            + (f" (oldest row {seen[0]})" if seen[0] != seen[-1] else ""))
    else:
        log("   ⚠️  no Last_Checked on any row — the engine has not stamped "
            "these yet, so the order reflects what the sheet says, not "
            "what Schwab says.")
    return len(ordered)


if __name__ == "__main__":
    raise SystemExit(0 if apply() >= 0 else 1)
