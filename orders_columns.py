#!/usr/bin/env python3
"""
orders_columns.py
=================
Where each Orders-tab column actually IS, read from the header row.

    .venv/bin/python orders_columns.py          # print the live layout

WHY THIS EXISTS
    The column letters used to be constants — COL_STATUS = "L",
    COL_VALIDATION = "N", COL_ENGINE_NOTE = "T" — in one file, with the
    matching integer indices in two others and the column list written out in
    three. Insert a single column and every one of them points at the wrong
    cell, and the engine writes Status into what is now Notes. Silently: a
    sheet write does not fail for being in the wrong place.

    The comment above those constants already said "a write-back aimed at the
    wrong column silently overwrites a different field". It was right, and
    naming them was not enough — the only way to be sure is to ASK THE SHEET.

HOW
    Row 8 holds the column headers. Find a name, get its letter and index.
    Adding a column anywhere then costs nothing: the next run reads the new
    header and writes to the right place.

FAILS CLOSED
    A header that cannot be read, or that is missing a column the caller
    needs, raises. The alternative is guessing an offset and writing a
    verdict into somebody's Notes.
"""
from __future__ import annotations

HEADER_ROW = 8               # Orders tab: status block 1-6, banner 7, headers 8
DATA_START_ROW = 9

# What the tab is expected to contain. Used to CREATE it and to sanity-check
# what comes back — never to locate anything. Position is read, not assumed.
HUMAN = ["Row_ID", "Date", "Acct", "Ticker", "Side", "Close_Is",
         "Trigger_Price", "Limit_Price", "Qty", "Expires_On", "Notes"]
ENGINE = ["Status", "Status_Date", "Validation", "Current_Price",
          "Schwab_Order_ID", "Filled_Qty", "Fill_Price", "Seed_Left",
          "Engine_Note", "Last_Checked"]
ALL = HUMAN + ENGINE


def letter(idx0: int) -> str:
    """0-based column index to a spreadsheet letter. A..Z, then AA.."""
    n, out = idx0 + 1, ""
    while n:
        n, r = divmod(n - 1, 26)
        out = chr(ord("A") + r) + out
    return out


class Columns:
    """The live layout. `c["Status"]` is the 0-based index, `c.col("Status")`
    the letter."""

    def __init__(self, header: list[str]):
        self.names = [str(h).strip() for h in header]
        self._idx = {}
        for i, h in enumerate(self.names):
            if h and h not in self._idx:        # first wins; a duplicate
                self._idx[h] = i                # header is a typo, not a move

    def __contains__(self, name) -> bool:
        return name in self._idx

    def __getitem__(self, name: str) -> int:
        try:
            return self._idx[name]
        except KeyError:
            raise KeyError(
                f"the Orders tab has no {name!r} column. Found: "
                f"{', '.join(n for n in self.names if n)}") from None

    def col(self, name: str) -> str:
        return letter(self[name])

    def get(self, row: list, name: str, default="") -> str:
        """One cell from a raw row, tolerating a short row.

        gspread truncates trailing empties, so a row whose last cells are
        blank comes back shorter than the header. Reading past the end is the
        normal case, not an error.
        """
        i = self[name]
        return str(row[i]).strip() if i < len(row) else default

    def width(self) -> int:
        return len(self.names)

    def missing(self, needed) -> list:
        return [n for n in needed if n not in self._idx]

    def __repr__(self):
        return "Columns(" + ", ".join(
            f"{self.col(n)}={n}" for n in self.names if n) + ")"


def read(ws, log=print) -> Columns:
    """The layout from the worksheet's header row. Raises if unusable."""
    try:
        row = ws.row_values(HEADER_ROW)
    except Exception as e:
        raise RuntimeError(f"could not read the Orders header row "
                           f"{HEADER_ROW}: {e}") from e
    cols = Columns(row)
    gone = cols.missing(HUMAN)
    if gone:
        raise RuntimeError(
            f"the Orders tab is missing {', '.join(gone)} in row {HEADER_ROW}. "
            f"Refusing to read or write it — guessing a position is how a "
            f"verdict ends up in somebody's Notes.")
    return cols


def main() -> int:
    import os
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    for ln in (Path(__file__).resolve().parent / ".env").read_text().splitlines():
        if "=" in ln and not ln.lstrip().startswith("#"):
            k, _, v = ln.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    import gspread
    from google.oauth2.service_account import Credentials
    creds = os.getenv("GSHEET_READER_CREDS") or os.getenv("REMOTE_OPS_CREDS")
    c = Credentials.from_service_account_file(
        creds, scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"])
    ws = gspread.authorize(c).open_by_key(
        os.environ["GSHEET_ORDERS_ID"]).worksheet("Orders")
    cols = read(ws)
    print(f"Orders tab, header row {HEADER_ROW}, {cols.width()} columns\n")
    for n in cols.names:
        if n:
            tag = "human" if n in HUMAN else ("engine" if n in ENGINE else "EXTRA")
            print(f"  {cols.col(n):>3}  {n:<18}{tag}")
    extra = [n for n in cols.names if n and n not in ALL]
    if extra:
        print(f"\n  not in the expected set: {', '.join(extra)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
