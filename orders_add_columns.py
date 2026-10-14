#!/usr/bin/env python3
"""
orders_add_columns.py
=====================
Add any missing HUMAN columns to the Orders tab. Idempotent.

    .venv/bin/python orders_add_columns.py            # DRY RUN — says what it would do
    .venv/bin/python orders_add_columns.py --apply    # actually insert them

RUN ON PI 1. Pi 2 holds a read-only sheet identity on purpose, so it can see
the layout and tell you what is missing but cannot change it.

WHY A SCRIPT AND NOT "insert a column by hand"
    Inserting by hand is thirty seconds and one typo. The header text IS the
    contract now — orders_columns.py resolves every read and write by matching
    that string — so "Trigger ID" or a trailing space produces a column the
    engine cannot see, and a row that quietly never chains. This types it.

SAFE BY CONSTRUCTION
    - DRY RUN by default; --apply is required to touch anything.
    - Only ever INSERTS. Never deletes, renames or reorders an existing
      column, so no data can be lost or shifted onto a different field.
    - Does nothing at all when the columns already exist, so re-running after
      a half-finished attempt is fine.
    - New columns go at the END of the human block, immediately before Notes,
      which keeps the engine columns contiguous to the right of them.

    Everything downstream reads BY NAME, so the inserted position does not
    matter to correctness — it is chosen only to keep the tab readable.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import orders_columns as oco                               # noqa: E402


def _load_env() -> None:
    """Read .env ourselves. Cron sources it; an interactive shell does not,
    and this is a by-hand script — the same trap schwab_auth already fell
    into once."""
    try:
        for line in (HERE / ".env").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(),
                                      v.strip().strip('"').strip("'"))
    except Exception:
        pass


def open_tab(write: bool):
    import gspread
    from google.oauth2.service_account import Credentials
    scope = ("https://www.googleapis.com/auth/spreadsheets" if write else
             "https://www.googleapis.com/auth/spreadsheets.readonly")
    creds = (os.getenv("GSHEET_WRITER_CREDS") or os.getenv("REMOTE_OPS_CREDS")
             or os.getenv("GSHEET_READER_CREDS"))
    if not creds:
        raise RuntimeError("no sheet credentials in the environment")
    c = Credentials.from_service_account_file(creds, scopes=[scope])
    book = gspread.authorize(c).open_by_key(os.environ["GSHEET_ORDERS_ID"])
    return book.worksheet("Orders"), creds


def main() -> int:
    ap = argparse.ArgumentParser(description="Add missing Orders columns")
    ap.add_argument("--apply", action="store_true",
                    help="actually insert them (default is a dry run)")
    args = ap.parse_args()
    _load_env()

    try:
        ws, creds = open_tab(write=args.apply)
        cols = oco.read(ws)
    except Exception as e:
        print(f"⛔ cannot read the Orders tab: {type(e).__name__}: {e}")
        return 1

    print(f"credentials : {Path(creds).name}")
    print(f"header row  : {oco.HEADER_ROW}, {cols.width()} columns")
    missing = cols.missing(oco.HUMAN)
    if not missing:
        print("\nnothing to do — every HUMAN column is already present:")
        print("   " + "  ".join(f"{cols.col(n)}={n}" for n in oco.HUMAN))
        return 0

    # Immediately before Notes when it exists, else at the end of the human
    # block. Position is cosmetic — reads resolve by name — so this only keeps
    # the human columns together and the engine columns to their right.
    anchor = cols["Notes"] if "Notes" in cols else len(oco.HUMAN) - 1
    print(f"\nmissing: {', '.join(missing)}")
    print(f"would insert {len(missing)} column(s) at "
          f"{oco.letter(anchor)}, pushing Notes and the engine columns right")
    for i, name in enumerate(missing):
        print(f"   {oco.letter(anchor + i)}  {name}")

    if not args.apply:
        print("\nDRY RUN — nothing changed. Re-run with --apply to insert.")
        return 0

    try:
        # One call, one header write per column. insert_cols shifts existing
        # cells RIGHT; it never overwrites, so no data is at risk.
        ws.insert_cols([[] for _ in missing], col=anchor + 1)
        ws.batch_update([
            {"range": f"{oco.letter(anchor + i)}{oco.HEADER_ROW}",
             "values": [[name]]}
            for i, name in enumerate(missing)
        ], value_input_option="RAW")
    except Exception as e:
        print(f"⛔ insert failed: {type(e).__name__}: {e}")
        print("   Nothing partial is expected, but CHECK THE TAB before "
              "re-running.")
        return 1

    # Re-read rather than trust the write. The whole point of this file is
    # that the header string is the contract.
    after = oco.read(ws)
    still = after.missing(oco.HUMAN)
    if still:
        print(f"⛔ still missing after the insert: {', '.join(still)}")
        return 1
    print("\ninserted. Live layout now:")
    for n in after.names:
        if n:
            tag = ("human" if n in oco.HUMAN else
                   "engine" if n in oco.ENGINE else "EXTRA")
            print(f"   {after.col(n):>3}  {n:<18}{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
