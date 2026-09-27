#!/usr/bin/env python3
"""
orders_sheet_init.py
====================
Create the four tabs of myTrading-ORDERS-pi1 from scratch.

    python3 orders_sheet_init.py --dry-run     # show what it would do
    python3 orders_sheet_init.py --force       # actually write

DESTRUCTIVE. --force wipes every tab it manages and rewrites the skeleton.
Anything you had typed in Orders is gone. There is no undo here beyond Google
Sheets' own version history (File -> Version history), which does cover it.

Layout per Documentation/ordersSheetDesign-9-18-26.md:

    Orders      rows 1-6  status header, written by Pi 1 every cycle
                row  7    ownership banner
                row  8    column headers
                row  9+   DATA. You own B-K (A is stamped by Pi 1),
                          Pi 1 owns L-U, V+ is spare.
    Positions   Pi 1 only
    Cash        Pi 1 only
    History     Pi 1 only, append-only

Environment (same shell rules as remote_ops.py — nothing is read from .env
automatically, so source it first):

    GSHEET_ORDERS_ID     the spreadsheet key, the /d/<ID>/ part of the URL
    REMOTE_OPS_CREDS     service-account json (falls back to GSHEET_CREDS)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

DATA_START_ROW = 9

# --------------------------------------------------------------------- Orders
# A-K are yours. L-V are Pi 1's. Nothing straddles.
# Every row means: <SIDE> <QTY> of <TICKER> in <ACCT> when the DAILY CLOSE is
# <ABOVE|BELOW> <TRIGGER>. Anything Schwab can already express as a resting
# order belongs at Schwab, not here — so there is no After_Close column (it
# would be Y on every row) and no order type or TIF.
# Row_ID is column A but you leave it BLANK — Pi 1 stamps it on first sight and
# it never changes after that. It is the row's identity, which is what makes
# inserting new rows at the TOP safe: the engine matches on Row_ID, never on
# where the row currently sits.
ORDERS_HUMAN = [
    "Row_ID", "Date", "Acct", "Ticker", "Side", "Close_Is",
    "Trigger_Price", "Limit_Price", "Qty", "Expires_On", "Notes",
]
ORDERS_ENGINE = [
    "Status", "Status_Date", "Validation", "Current_Price",
    "Schwab_Order_ID", "Filled_Qty", "Fill_Price", "Seed_Left",
    "Engine_Note", "Last_Checked",
]
# No trailing Comments column — Notes (K) sits with the intent, where a
# note about a row is actually readable beside it.
ORDERS_COLS = ORDERS_HUMAN + ORDERS_ENGINE

POSITIONS_COLS = [
    "Ticker", "Acct", "Qty", "Avg_Cost", "Market_Value", "Unrealized_PL",
    "Has_Stop", "Has_Trim", "Has_Dip", "Has_Breakout", "Seed_Reserved", "Updated",
]
CASH_COLS = [
    "Acct", "Nickname", "Cash", "Cash_In_Open_Orders", "Cash_After_Open_Orders",
    "Seed_Reserved", "Free_To_Deploy", "Updated",
]
HISTORY_COLS = ORDERS_COLS + ["Completed_At", "Outcome"]

# Rows 1-6. Pi 1 rewrites column B every cycle; the LAST POLL staleness is the
# health check, so the placeholders say so rather than looking like real values.
HEADER_BLOCK = [
    ["SYSTEM STATUS", "not yet written by Pi 1"],
    ["SCHWAB TOKEN", "—"],
    ["LAST POLL", "never — if this stays blank, nothing is polling"],
    ["TRADING", "—"],
    ["FREE TO DEPLOY", "—"],
    ["ALERTS", "—"],
]

BANNER = ("(Pi 1 fills)", "▼ YOU FILL B-K ▼", "", "", "", "", "", "", "", "",
          "your notes",
          "▼ PI 1 FILLS L-U — DO NOT TYPE HERE ▼")

ACTIONS = ("Leave Row_ID (A) BLANK — Pi 1 stamps it. Add new rows at the TOP.   ·   "
           "Side: BUY | SELL   ·   Close_Is: ABOVE | BELOW   ·   "
           "every row fires on a completed DAILY CLOSE, and submits the NEXT "
           "MORNING.   Limit_Price blank = MARKET (usually right for an exit: "
           "a limit that does not fill leaves you holding). Set a limit on a "
           "BUY so a gap up cannot make you overpay.   "
           "Plain limit orders belong at Schwab, not here.")


def col_letter(n: int) -> str:
    """1 -> A, 27 -> AA."""
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def open_book():
    from google.oauth2.service_account import Credentials
    import gspread

    sid = os.getenv("GSHEET_ORDERS_ID", "")
    creds_path = os.getenv("REMOTE_OPS_CREDS",
                           os.getenv("GSHEET_CREDS", "/etc/myTrading/gsheets-ops.json"))
    if not sid:
        sys.exit("GSHEET_ORDERS_ID is not set — add it to .env and `set -a; . ./.env; set +a`")
    if not Path(creds_path).exists():
        sys.exit(f"service-account key not found at {creds_path}")

    creds = Credentials.from_service_account_file(creds_path, scopes=SCOPES)
    return gspread.authorize(creds).open_by_key(sid)


def ensure_tab(book, title: str, rows: int, cols: int):
    """Get the worksheet, creating it if absent. Existing content is untouched."""
    try:
        return book.worksheet(title)
    except Exception:
        return book.add_worksheet(title=title, rows=rows, cols=cols)


def build_orders(ws, dry: bool) -> None:
    payload = []
    for label, value in HEADER_BLOCK:
        payload.append([label, value] + [""] * (len(ORDERS_COLS) - 2))
    payload.append(list(BANNER) + [""] * (len(ORDERS_COLS) - len(BANNER)))
    payload.append(ORDERS_COLS)

    last = col_letter(len(ORDERS_COLS))
    print(f"  Orders     : header block rows 1-6, banner row 7, columns row 8, "
          f"data from row {DATA_START_ROW}  (A1:{last}8)")
    if dry:
        return
    ws.clear()
    ws.update(values=payload, range_name=f"A1:{last}8")
    ws.freeze(rows=8)
    ws.format(f"A8:{last}8", {"textFormat": {"bold": True}})
    ws.format("A1:B6", {"textFormat": {"bold": True}})
    # Tint the engine-owned block so typing there feels wrong.
    ws.format("L1:U1000", {"backgroundColor": {"red": 0.96, "green": 0.96, "blue": 0.96}})
    ws.update(values=[[f"Actions: {ACTIONS}"]], range_name=f"{col_letter(len(ORDERS_COLS))}1")


def build_simple(ws, title: str, cols: list[str], dry: bool) -> None:
    last = col_letter(len(cols))
    print(f"  {title:<11}: {len(cols)} columns, headers row 1  (A1:{last}1)")
    if dry:
        return
    ws.clear()
    ws.update(values=[cols], range_name=f"A1:{last}1")
    ws.freeze(rows=1)
    ws.format(f"A1:{last}1", {"textFormat": {"bold": True}})


def main() -> int:
    ap = argparse.ArgumentParser(description="Initialise the orders spreadsheet")
    ap.add_argument("--force", action="store_true",
                    help="actually write — WIPES every managed tab")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the plan, touch nothing")
    args = ap.parse_args()

    if not args.force and not args.dry_run:
        ap.error("pass --dry-run to preview, or --force to write. "
                 "--force wipes the tabs; Google Sheets version history is your undo.")

    dry = not args.force
    book = open_book()
    print(f"\nspreadsheet: {book.title}")
    print("mode: DRY RUN, nothing written\n" if dry else "mode: WRITING — tabs will be wiped\n")

    build_orders(ensure_tab(book, "Orders", 500, len(ORDERS_COLS) + 4), dry)
    build_simple(ensure_tab(book, "Positions", 200, len(POSITIONS_COLS)),
                 "Positions", POSITIONS_COLS, dry)
    build_simple(ensure_tab(book, "Cash", 50, len(CASH_COLS)), "Cash", CASH_COLS, dry)
    build_simple(ensure_tab(book, "History", 2000, len(HISTORY_COLS)),
                 "History", HISTORY_COLS, dry)

    if dry:
        print("\nnothing was written. re-run with --force to apply.")
    else:
        print("\ndone. Orders data starts at row 9.")
        print("Sheet1 (the default tab) is left alone — delete it by hand if unused.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
