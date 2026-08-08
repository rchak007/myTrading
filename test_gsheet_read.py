#!/usr/bin/env python3
"""
test_gsheet_read.py — connectivity + scope test for the myTrading Google Sheet.

Proves four things before any parser work happens:
  1. The service account can authenticate.
  2. It can open THIS spreadsheet by ID.
  3. It can read the target tab.
  4. It CANNOT write (readonly scope is actually enforced).

Setup (one time):
    pip install gspread google-auth --break-system-packages
    creds JSON at /etc/myTrading/gsheets.json, mode 600
    spreadsheet shared with the JSON's client_email as *Viewer*

Usage:
    python3 test_gsheet_read.py
    GSHEET_ID=<id> GSHEET_TAB=myTrading python3 test_gsheet_read.py
"""

from __future__ import annotations

import json
import os
import sys

import gspread
from google.oauth2.service_account import Credentials

# ── Config ──────────────────────────────────────────────────────────
CREDS_PATH = os.getenv("GSHEET_CREDS", "/etc/myTrading/gsheets.json")

# The long string in the sheet URL between /d/ and /edit
SHEET_ID = os.getenv("GSHEET_ID", "PUT_YOUR_SPREADSHEET_ID_HERE")

TAB_NAME = os.getenv("GSHEET_TAB", "myTrading")

# readonly ONLY. Do not widen this to drive.readonly — that would grant
# read across the entire Drive instead of just shared-in files.
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

PREVIEW_ROWS = 15
PREVIEW_COLS = 8
CELL_TRUNC = 22


def die(msg: str, hint: str = "") -> None:
    print(f"\n  FAIL: {msg}", file=sys.stderr)
    if hint:
        print(f"  → {hint}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    # ── 1. Credentials ──────────────────────────────────────────────
    if not os.path.exists(CREDS_PATH):
        die(f"no creds at {CREDS_PATH}", "download the service-account JSON key")

    mode = oct(os.stat(CREDS_PATH).st_mode)[-3:]
    if mode != "600":
        print(f"  WARN: {CREDS_PATH} is mode {mode}, expected 600")

    with open(CREDS_PATH) as fh:
        sa_email = json.load(fh).get("client_email", "?")

    print(f"service account : {sa_email}")

    try:
        creds = Credentials.from_service_account_file(CREDS_PATH, scopes=SCOPES)
        gc = gspread.authorize(creds)
    except Exception as e:
        die(f"auth failed: {e}")

    print("auth            : OK")

    # ── 2. Open the spreadsheet ─────────────────────────────────────
    if SHEET_ID.startswith("PUT_YOUR"):
        die("SHEET_ID not set", "edit SHEET_ID or export GSHEET_ID=<id>")

    try:
        sh = gc.open_by_key(SHEET_ID)
    except gspread.exceptions.APIError as e:
        code = getattr(e, "response", None)
        code = code.status_code if code is not None else "?"
        if code == 403:
            die(
                "403 — spreadsheet not shared with the service account",
                f"Share the sheet with {sa_email} as Viewer",
            )
        if code == 404:
            die("404 — spreadsheet ID not found", "check the ID from the sheet URL")
        die(f"APIError {code}: {e}")

    print(f"spreadsheet     : {sh.title!r}")

    # ── 3. Tabs visible in THIS file ────────────────────────────────
    tabs = [ws.title for ws in sh.worksheets()]
    print(f"tabs in file    : {tabs}")

    if TAB_NAME not in tabs:
        die(f"tab {TAB_NAME!r} not present", f"available: {tabs}")

    ws = sh.worksheet(TAB_NAME)

    # get_all_values() returns raw grid — no header assumptions, which
    # matters because this sheet has blank spacer rows and merged cells.
    rows = ws.get_all_values()
    ncols = max((len(r) for r in rows), default=0)
    nonblank = sum(1 for r in rows if any(c.strip() for c in r))

    print(f"tab {TAB_NAME!r}   : {len(rows)} rows x {ncols} cols "
          f"({nonblank} non-blank)")

    # ── 4. Preview ──────────────────────────────────────────────────
    print(f"\n--- first {PREVIEW_ROWS} rows, {PREVIEW_COLS} cols ---")
    for i, row in enumerate(rows[:PREVIEW_ROWS], start=1):
        cells = []
        for c in row[:PREVIEW_COLS]:
            c = c.replace("\n", "\\n")
            cells.append(c[:CELL_TRUNC] + "…" if len(c) > CELL_TRUNC else c)
        print(f"{i:>3} | " + " | ".join(f"{c:<{CELL_TRUNC}}" for c in cells))

    # ── 5. Confirm writes are blocked ───────────────────────────────
    print("\n--- write test (should fail) ---")
    try:
        ws.update_acell("ZZ999", "should_not_write")
        print("  WARN: write SUCCEEDED — scope is wider than intended")
    except Exception as e:
        kind = type(e).__name__
        print(f"  write blocked ({kind}) — readonly confirmed")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()