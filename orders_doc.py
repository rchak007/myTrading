#!/usr/bin/env python3
"""
orders_doc.py
=============
Fills the `howOrdersWork` tab in the ORDERS book — how to type a row on the
Orders tab, every column, every trigger kind, every status that comes back.

    .venv/bin/python orders_doc.py            # write the tab
    .venv/bin/python orders_doc.py --print    # preview, write nothing

GENERATED, NOT HAND-WRITTEN, for the same reason VERBS-howToUse is. The
Orders tab's own header row carries one cramped sentence of instructions, and
it went stale the moment structure triggers arrived — nothing made it update.
Here the constants come FROM the code: DIRECTIONS, QTY_RESERVE, the chain
field names and the status vocabulary are all imported, so a doc that
disagrees with the engine cannot survive a regeneration.

The prose is still prose and still needs editing by hand when behaviour
changes. What cannot drift is the part that matters most in a hurry — the
exact strings the parser will accept.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import order_intent as oi                                    # noqa: E402

TAB = os.getenv("GSHEET_ORDERS_DOC_TAB", "howOrdersWork")

# Pulled from the code so the tab cannot claim a trigger the parser rejects.
PRICE = " / ".join(sorted(oi.PRICE_DIRECTIONS))
STRUCT = " · ".join(sorted(oi.STRUCTURE_DIRECTIONS))


def rows() -> list[list[str]]:
    """Every row of the tab. Four columns throughout so nothing is ragged."""
    R: list[list[str]] = []
    add = R.append
    gap = lambda: add(["", "", "", ""])

    def head(t, sub=""):
        gap()
        add([t.upper(), sub, "", ""])

    add(["HOW THE ORDERS TAB WORKS", "",
         "Generated from the code — re-run orders_doc after any change.", ""])
    add(["", "You fill B–K. Pi 1 fills L–U. Never type in L–U.", "", ""])
    add(["", "Leave Row_ID (A) BLANK — Pi 1 stamps it. "
             "Add new rows at the TOP.", "", ""])

    # ── the one-line mental model ────────────────────────────────────────
    head("THE SHAPE OF A ROW",
         "every row says: WHEN this happens, DO this, THIS BIG")
    add(["WHEN", "Close_Is + Trigger_Price", "the condition", ""])
    add(["DO", "Side + Ticker + Acct", "BUY or SELL, where", ""])
    add(["THIS BIG", "Qty", "shares, dollars, or the fenced reserve", ""])
    add(["", "Everything else is optional.", "", ""])

    # ── columns ──────────────────────────────────────────────────────────
    head("THE COLUMNS YOU FILL", "B through K")
    add(["COLUMN", "WHAT TO TYPE", "EXAMPLE", "NOTES"])
    cols = [
        ("Date", "the date you typed it", "2026-10-27",
         "Used for expiry maths and the Row_ID Pi 1 builds."),
        ("Acct", "last 3 digits", "431",
         "Must be an account that actually holds / can fund it."),
        ("Ticker", "symbol", "PBLS", "Uppercase."),
        ("Side", "BUY or SELL", "BUY",
         "All eight Side x condition combinations are legal."),
        ("Close_Is", f"{PRICE} or {STRUCT}", "HL HH",
         "The trigger. See the two kinds below."),
        ("Trigger_Price", "a price, for ABOVE / BELOW only", "73",
         "LEAVE BLANK on a structure trigger — the pivot IS the level."),
        ("Limit_Price", "a price, or blank", "(blank)",
         "Blank = MARKET. Blank is usually right for an EXIT: a limit that "
         "does not fill leaves you still holding. Set one on a BUY so a gap "
         "up cannot make you overpay."),
        ("Qty", "shares, $dollars, RESERVE, or blank", "RESERVE",
         "Four forms — see the Qty section."),
        ("Expires_On", "a date, or blank", "2026-12-31",
         "Blank = never expires. The row waits forever until it fires."),
        ("Trigger_ID", "your own label, or blank", "1",
         "Only needed if another row waits on this one."),
        ("After_Trigger_ID", "another row's Trigger_ID", "1",
         "This row stays WAITING until that row FILLS."),
        ("Notes", "free text", "buy back what I sold",
         "Yours. Pi 1 never writes here."),
    ]
    for c in cols:
        add(list(c))

    # ── triggers ─────────────────────────────────────────────────────────
    head("TRIGGER KIND 1 — PRICE", "Close_Is = ABOVE or BELOW")
    add(["", "Fires on a COMPLETED DAILY CLOSE past Trigger_Price.", "", ""])
    add(["", "A wick through the level does NOT fire it. The close must be "
             "past.", "", ""])
    add(["SELL / BELOW", "73", "exit if it closes under 73",
         "a close-confirmed stop"])
    add(["BUY / BELOW", "312", "buy the dip at 312", "a dip bid"])
    add(["BUY / ABOVE", "420", "buy the breakout", "momentum entry"])
    add(["SELL / ABOVE", "412", "trim into strength", "a trim"])
    add(["", "", "", ""])
    add(["⚠", "A close-triggered stop CANNOT protect against a gap.",
         "It submits the NEXT session.",
         "For intraday protection put a real STOP at Schwab instead."])

    head("TRIGGER KIND 2 — STRUCTURE", f"Close_Is = {STRUCT}")
    add(["", "Fires when the next pivot of that kind is CONFIRMED — not on a "
             "price.", "", ""])
    add(["", "Leave Trigger_Price BLANK. The pivot is the level.", "", ""])
    add(["HH", "higher high", "uptrend extends", "breakout"])
    add(["HL", "higher low", "pullback held", "the classic entry"])
    add(["LL", "lower low", "downtrend extends", "breakdown / capitulation"])
    add(["LH", "lower high", "bounce failed", "the classic exit"])
    add(["", "", "", ""])
    add(["SEVERAL AT ONCE", "HL HH", "fires on whichever confirms FIRST",
         "Space, slash or comma. Order does not matter — stored sorted."])
    add(["", "Only pivots confirmed AFTER you typed the row count.",
         "An HL that formed last week will not fire it.", ""])
    add(["", "Confirmation needs a close, so these are close-triggered too.",
         "", ""])

    # ── qty ──────────────────────────────────────────────────────────────
    head("QTY — FOUR FORMS", "this is where most mistakes happen")
    add(["WHAT YOU TYPE", "MEANS", "RESOLVED", "USE IT WHEN"])
    add(["10", "10 shares", "when you type it",
         "You know the share count you want."])
    add(["$730", "as many WHOLE shares as $730 buys",
         "at FIRE time, from that day's price",
         "You know the money, not the count. MUST have the $ sign."])
    add([oi.QTY_RESERVE, "everything fenced for this ticker RIGHT NOW",
         "at FIRE time, from the reserve ledger",
         "Buying back what you sold. The ticker must be FENCED in that "
         "account, or the row is BLOCKED."])
    add(["(blank)", "the parent row's FILLED quantity",
         "when the parent fills",
         f"CHAINED ROWS ONLY — needs {oi.CHAIN_FIELD} set. Blank on an "
         f"unchained row is an error, not a default."])
    add(["", "", "", ""])
    add(["⚠", "$730 without the dollar sign reads as 730 SHARES.",
         "On a $75 stock that is a 10x order.",
         "The parser checks for $ first, deliberately."])
    add(["⚠", "A budget that buys 0 whole shares is BLOCKED, not rounded.",
         "$40 at $27 = 1 share. $20 at $27 = BLOCKED.", ""])

    # ── chaining ─────────────────────────────────────────────────────────
    head("CHAINING — ONE ROW WAITS FOR ANOTHER",
         f"{oi.TRIGGER_FIELD} and {oi.CHAIN_FIELD}")
    add(["", "Sell, then buy back only if the sell actually filled.", "", ""])
    add(["ROW 1", f"{oi.TRIGGER_FIELD} = 1", "SELL AAOI BELOW 73, Qty 10",
         "the parent"])
    add(["ROW 2", f"{oi.CHAIN_FIELD} = 1", "BUY AAOI HL HH, Qty blank",
         "waits; inherits 10 if the parent filled 10"])
    add(["", "", "", ""])
    add(["", "The child shows WAITING until the parent FILLS.",
         "If the parent is cancelled or expires, the child is cancelled too.",
         ""])
    add(["", f"{oi.TRIGGER_FIELD} must be unique among rows not yet done.",
         "Re-using a live one is an error on that row.", ""])
    add(["", f"Chains deeper than {oi.MAX_CHAIN_DEPTH} rows are painted RED.",
         "Not blocked — flagged, so you can pivot.", ""])

    # ── statuses ─────────────────────────────────────────────────────────
    head("WHAT COMES BACK IN COLUMN L", "Pi 1 writes these — do not type them")
    add(["STATUS", "MEANS", "", "WHAT TO DO"])
    st = [
        ("(blank)", "not seen yet, or waiting for the next poll", "",
         "Nothing. Pi 1 polls on a schedule."),
        ("ARMED", "valid and watching for the condition", "", "Nothing."),
        ("WAITING", "chained — the parent has not filled", "", "Nothing."),
        ("TRIGGERED", "the condition was met", "", "Order is being placed."),
        ("SUBMITTED", "the order is AT SCHWAB", "",
         "Done. It is a real resting order now."),
        ("BLOCKED", "a guard stopped it", "",
         "Read the Validation column — it says which guard and why."),
        ("EXPIRED", "Expires_On passed before it fired", "",
         "Re-type it if you still want it."),
        ("VOID", "the row changed after being armed", "",
         "Expected if you edited it. The engine re-reads the new version."),
    ]
    for s in st:
        add(list(s))

    # ── rules ────────────────────────────────────────────────────────────
    head("RULES THAT BITE", "")
    add(["", "Every row fires on a COMPLETED DAILY CLOSE and submits the "
             "NEXT session.", "Nothing here is intraday.", ""])
    add(["", "Orders go in as GTC LIMIT + extended hours.",
         "That is why a filled trigger can execute pre-market.", ""])
    add(["", "Plain limit orders you just want resting belong at SCHWAB, "
             "not here.", "This tab is for CONDITIONAL orders only.", ""])
    add(["", "Never clear a Status to re-use a row. Add a new row at the top.",
         "", ""])
    add(["", "Editing an armed row VOIDs it and re-arms the new version.",
         "That is a feature — it means the sheet is always the truth.", ""])

    # ── worked example ───────────────────────────────────────────────────
    head("WORKED EXAMPLE — buy back with the money you freed", "")
    add(["", "You hold PBLS in 431 and want to add on a turn, spending only "
             "what is fenced for it.", "", ""])
    add(["Acct", "431", "", ""])
    add(["Ticker", "PBLS", "", ""])
    add(["Side", "BUY", "", ""])
    add(["Close_Is", "HL HH", "fires on whichever confirms first", ""])
    add(["Trigger_Price", "(blank)", "structure trigger — no price", ""])
    add(["Limit_Price", "(blank)", "or set one to cap a gap up", ""])
    add(["Qty", oi.QTY_RESERVE, "spends exactly what is fenced for PBLS", ""])
    add(["", "", "", ""])
    add(["", "At fire time: $401.75 reserved ÷ the day's price = whole "
             "shares.", "At $27.00 that is 14 shares, $23.68 left over.", ""])

    gap()
    add(["", "Generated by orders_doc.py. Re-run the orders_doc verb after "
             "any change to the engine.", "", ""])
    return R


def write(log=print) -> str:
    import gspread
    from google.oauth2.service_account import Credentials

    creds = os.getenv("REMOTE_OPS_CREDS",
                      os.getenv("GSHEET_CREDS", "/etc/myTrading/gsheets.json"))
    sid = os.environ["GSHEET_ORDERS_ID"]
    book = gspread.authorize(Credentials.from_service_account_file(
        creds, scopes=["https://www.googleapis.com/auth/spreadsheets"])
    ).open_by_key(sid)

    data = rows()
    try:
        ws = book.worksheet(TAB)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = book.add_worksheet(title=TAB, rows=len(data) + 20, cols=4)

    ws.resize(rows=max(len(data) + 20, 80), cols=4)
    ws.update(values=data, range_name=f"A1:D{len(data)}")

    # Cosmetics are best-effort: a formatting failure must not lose the text.
    import contextlib
    with contextlib.suppress(Exception):
        dark = {"backgroundColor": {"red": 0.20, "green": 0.25, "blue": 0.35},
                "textFormat": {"bold": True, "fontSize": 12,
                               "foregroundColor": {"red": 1, "green": 1,
                                                   "blue": 1}}}
        ws.format("A1:D1", dark)
        # Section headers: the rows where column A is upper-case and B is a
        # lower-case subtitle. Found by scanning what we just built, so a
        # re-ordering cannot leave the paint on the wrong rows.
        heads = [i + 1 for i, r in enumerate(data)
                 if r[0] and r[0] == r[0].upper() and len(r[0]) > 3
                 and not r[1][:1].isdigit() and r[0] not in
                 {"COLUMN", "STATUS", "WHAT YOU TYPE", "ROW 1", "ROW 2"}]
        if heads:
            ws.format([f"A{r}:D{r}" for r in heads],
                      {"backgroundColor": {"red": 0.93, "green": 0.90,
                                           "blue": 0.98},
                       "textFormat": {"bold": True}})
        warn = [i + 1 for i, r in enumerate(data) if r[0] == "⚠"]
        if warn:
            ws.format([f"A{r}:D{r}" for r in warn],
                      {"backgroundColor": {"red": 1.0, "green": 0.93,
                                           "blue": 0.80}})
        ws.columns_auto_resize(0, 3)
        ws.freeze(rows=1)

    msg = f"wrote {len(data)} row(s) to '{TAB}'"
    log(msg)
    return msg


def main() -> int:
    if "--print" in sys.argv:
        for r in rows():
            print(" | ".join(x[:46] for x in r))
        return 0
    write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
