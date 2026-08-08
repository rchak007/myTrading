# =====================================================================
# gsheet_notes.py — per-ticker trade-journal popup for the CSV viewer
# ---------------------------------------------------------------------
# Reads the myTrading Google Sheet (readonly) and exposes, for any CSV
# rendered in dashboard.py that carries a Ticker column, a modal showing
# every sheet row for that ticker.
#
# Credentials resolution order:
#   1. st.secrets["gsheets"]["service_account"]   (Streamlit Cloud)
#   2. $GSHEET_CREDS or /etc/myTrading/gsheets.json  (Pi / WSL)
# The key is IDENTITY only. Read restriction comes from the Viewer share
# grant on the file plus the spreadsheets.readonly scope below.
#
# Degrades silently: if gspread is missing or no creds resolve, the
# viewer keeps working and just doesn't offer the popup.
# =====================================================================
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

# Column names in a CSV that mean "stock symbol".
TICKER_ALIASES = ("ticker", "symbol")

DEFAULT_CREDS_PATH = os.getenv("GSHEET_CREDS", "/etc/myTrading/gsheets.json")

# Columns surfaced in the popup's compact table, in order. Anything in the
# sheet not listed here still shows up in the per-row detail below it.
TABLE_COLS = ("Date", "Acct", "Action", "REASON", "Qty", "Price", "Value")


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
def _cfg() -> dict:
    try:
        return dict(st.secrets.get("gsheets", {}))
    except Exception:
        return {}


def sheet_id() -> str:
    return _cfg().get("sheet_id", os.getenv("GSHEET_ID", ""))


def tab_name() -> str:
    return _cfg().get("tab", os.getenv("GSHEET_TAB", "myTrading"))


def _credentials():
    """Service-account creds from secrets, else from a local key file."""
    try:
        from google.oauth2.service_account import Credentials
    except ImportError:
        return None

    sa = _cfg().get("service_account")
    if sa:
        return Credentials.from_service_account_info(dict(sa), scopes=SCOPES)

    if Path(DEFAULT_CREDS_PATH).exists():
        return Credentials.from_service_account_file(DEFAULT_CREDS_PATH, scopes=SCOPES)

    return None


def available() -> bool:
    """True when the sheet can plausibly be read (no network call made)."""
    try:
        import gspread  # noqa: F401
    except ImportError:
        return False
    return bool(sheet_id()) and _credentials() is not None


# ---------------------------------------------------------------------
# Sheet -> DataFrame
# ---------------------------------------------------------------------
def _to_frame(rows: list[list[str]]) -> pd.DataFrame:
    """
    Turn the raw value grid into a tidy frame.

    Handles the sheet's real shape:
      - header row is found by locating the cell that reads TICKER, so
        leading title/blank rows above it are irrelevant
      - blank spacer rows drop out (no ticker)
      - a row with exactly one non-blank cell is a section banner
        (e.g. "RIZZY  methods"); it is carried forward as Section
      - _row keeps the 1-based sheet row so you can jump back to it
    """
    hdr_idx = None
    for i, row in enumerate(rows):
        if any(c.strip().lower() == "ticker" for c in row):
            hdr_idx = i
            break
    if hdr_idx is None:
        raise ValueError("no TICKER header cell found in the tab")

    raw_hdr = rows[hdr_idx]
    header, seen = [], {}
    for j, name in enumerate(raw_hdr):
        name = name.strip() or f"col_{j}"
        seen[name] = seen.get(name, 0) + 1
        header.append(name if seen[name] == 1 else f"{name}_{seen[name]}")

    tcol = next(i for i, c in enumerate(raw_hdr) if c.strip().lower() == "ticker")
    width = len(header)

    records, section = [], ""
    for n, row in enumerate(rows[hdr_idx + 1:], start=hdr_idx + 2):
        cells = [c.strip() for c in row[:width]]
        cells += [""] * (width - len(cells))
        filled = [c for c in cells if c]
        if not filled:
            continue
        ticker = cells[tcol] if tcol < width else ""
        if not ticker:
            if len(filled) == 1:          # lone cell => section banner
                section = filled[0]
            continue
        rec = dict(zip(header, cells))
        rec["Section"] = section
        rec["_row"] = n
        records.append(rec)

    return pd.DataFrame.from_records(records)


@st.cache_data(ttl=300, show_spinner=False)
def load_sheet(sid: str, tab: str) -> pd.DataFrame:
    """One API call per 5 minutes, shared by every CSV block on the page."""
    import gspread

    gc = gspread.authorize(_credentials())
    ws = gc.open_by_key(sid).worksheet(tab)
    return _to_frame(ws.get_all_values())


# ---------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------
def find_ticker_col(df: pd.DataFrame) -> str | None:
    """Name of the CSV's ticker column, or None (macro.csv -> None)."""
    for col in df.columns:
        if str(col).strip().lower() in TICKER_ALIASES:
            return col
    return None


def rows_for(sheet: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if sheet.empty:
        return sheet
    key = str(ticker).strip().upper()
    return sheet[sheet["TICKER"].str.strip().str.upper() == key]


# ---------------------------------------------------------------------
# The popup
# ---------------------------------------------------------------------
@st.dialog("📓 Trade journal", width="large")
def show_ticker_notes(ticker: str) -> None:
    st.subheader(str(ticker).upper())

    try:
        sheet = load_sheet(sheet_id(), tab_name())
    except Exception as e:
        st.error(f"Could not read the sheet: {e}")
        return

    hits = rows_for(sheet, ticker)
    if hits.empty:
        st.info(f"No rows for {str(ticker).upper()} in the sheet.")
        return

    st.caption(f"{len(hits)} row(s) · tab `{tab_name()}`")

    cols = [c for c in TABLE_COLS if c in hits.columns]
    if cols:
        st.dataframe(hits[cols], use_container_width=True, hide_index=True)

    # Comments carry embedded newlines, which a dataframe cell truncates.
    # Render them as text so they are actually readable and scrollable.
    if "Comments" in hits.columns:
        st.markdown("**Comments**")
        for _, r in hits.iterrows():
            note = str(r.get("Comments", "")).strip()
            if not note:
                continue
            head = " · ".join(
                str(r[c]) for c in ("Date", "Action", "Qty", "Price")
                if c in hits.columns and str(r[c]).strip()
            )
            sect = f" · _{r['Section']}_" if str(r.get("Section", "")).strip() else ""
            st.markdown(f"**row {r['_row']}** — {head}{sect}")
            st.text(note)
            st.divider()