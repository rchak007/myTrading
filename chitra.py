#!/usr/bin/env python3
"""
chitra.py
=========
Chitra's account — the sister's portfolio Chakravarti manages.

Two outputs, both written by Pi 1 on the normal cycle:

    the Chitra tab       the whole account, priced live
    the Dashboard        a CHITRA row inside any block whose ticker she shares

WHERE THE TRUTH IS
    `chitra_holdings.csv` in the repo, NOT the sheet. Chakravarti sends an
    updated statement now and then; Pi 2 edits the file and pushes, Pi 1
    renders it. Anything typed into the tab is overwritten next run.

NOT AN API ACCOUNT
    These holdings come from a statement, not from Schwab. Only the PRICE is
    live — quantity and cost basis are as fresh as `As_Of` in the CSV header,
    and the tab says so in as many words.

DISPLAY ONLY, AND THAT IS LOAD-BEARING
    Her rows are injected at RENDER time, after the TOTAL row is computed.
    They never enter `positions`, so they cannot reach the per-ticker TOTAL,
    the market-value sort, coverage flags, `sell_guard`, the reserve ledger or
    the order engine. Folding someone else's shares into those would be the
    worst kind of wrong: his numbers, silently inflated by her account.

    That is also why the coverage columns are blank on her rows rather than N.
    There is no Schwab connection here, so "no stop" is not something this
    code is in a position to assert.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILE = Path(os.getenv("CHITRA_FILE", HERE / "chitra_holdings.csv"))

ACCT_LABEL = "CHITRA"           # what appears in the Acct column
TAB = "Chitra"
COLS = ["Ticker", "Qty", "Avg_Cost", "Live_Price", "Day_%",
        "Market_Value", "Cost_Basis", "Unrealized_PL", "Unrealized_%", "Note"]


def _num(v, default=None):
    try:
        if v is None or str(v).strip() == "":
            return default
        return float(str(v).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return default


def meta() -> dict:
    """`Key  Value` pairs from the comment header — Account, As_Of.

    Parsed from the comments rather than kept in a second file, so the
    statement date can never drift from the rows it describes.
    """
    out = {}
    if not FILE.exists():
        return out
    for line in FILE.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("#"):
            break
        parts = line.lstrip("# ").split(None, 1)
        if len(parts) == 2 and parts[0] in ("Account", "As_Of"):
            out[parts[0]] = parts[1].strip()
    return out


def load(log=print) -> list[dict]:
    """Rows from the CSV. Returns [] on anything unexpected — never raises.

    A missing or malformed file must not take the Dashboard down with it: her
    account is an addition to the picture, not a prerequisite for it.
    """
    if not FILE.exists():
        log(f"Chitra: no {FILE.name}, skipping")
        return []
    try:
        with FILE.open(newline="", encoding="utf-8") as fh:
            body = [l for l in fh if not l.lstrip().startswith("#")]
        rows = []
        for r in csv.DictReader(body):
            t = str(r.get("Ticker", "")).strip().upper()
            qty = _num(r.get("Qty"))
            if not t or qty is None:
                continue
            rows.append({"Ticker": t, "Qty": qty,
                         "Cost_Basis": _num(r.get("Cost_Basis"), 0.0) or 0.0,
                         "Asset": str(r.get("Asset", "EQUITY")).strip().upper(),
                         "Note": str(r.get("Note", "")).strip()})
        return rows
    except Exception as e:
        log(f"⚠️  Chitra: could not read {FILE.name}: {e}")
        return []


def _price(row, quotes, extract) -> tuple:
    """(price, day_pct). CASH is held at $1.00 and never quoted.

    A sweep fund has no market quote, and asking for one returns nothing —
    which would then read as a priceless row rather than as cash.
    """
    if row["Asset"] == "CASH":
        return 1.0, 0.0
    if not quotes:
        return None, None
    entry = quotes.get(row["Ticker"])
    return extract(entry) if entry else (None, None)


def build_table(rows, quotes, extract, log=print) -> list[list]:
    """The Chitra tab body, with a TOTAL line. Values live, basis from file."""
    out, tot_val, tot_cost = [], 0.0, 0.0
    for r in rows:
        px, pct = _price(r, quotes, extract)
        val = (px * r["Qty"]) if px is not None else None
        upl = (val - r["Cost_Basis"]) if val is not None else None
        pcnt = (100.0 * upl / r["Cost_Basis"]) if (upl is not None
                                                   and r["Cost_Basis"]) else None
        if val is not None:
            tot_val += val
            tot_cost += r["Cost_Basis"]
        out.append([
            r["Ticker"], round(r["Qty"], 4),
            round(r["Cost_Basis"] / r["Qty"], 4) if r["Qty"] else "",
            round(px, 4) if px is not None else "",
            round(pct, 2) if pct is not None else "",
            round(val, 2) if val is not None else "",
            round(r["Cost_Basis"], 2),
            round(upl, 2) if upl is not None else "",
            round(pcnt, 2) if pcnt is not None else "",
            r["Note"],
        ])
    if out:
        g = tot_val - tot_cost
        out.append(["TOTAL", "", "", "", "", round(tot_val, 2), round(tot_cost, 2),
                    round(g, 2),
                    round(100.0 * g / tot_cost, 2) if tot_cost else "", ""])
    return out


def write_tab(book, quotes, extract, log=print) -> int:
    """Rewrite the Chitra tab. Returns the number of holding rows."""
    rows = load(log=log)
    if not rows:
        return 0
    try:
        ws = book.worksheet(TAB)
    except Exception:
        ws = book.add_worksheet(title=TAB, rows=200, cols=len(COLS) + 2)

    m = meta()
    body = build_table(rows, quotes, extract, log=log)
    # The provenance line is not decoration. Every number but Live_Price comes
    # from a statement, and a tab that looked live would invite trading off a
    # stale quantity.
    head = [[f"CHITRA — {m.get('Account', 'account')}",
             f"qty & cost basis as of {m.get('As_Of', 'unknown')} · "
             f"prices live · managed by Chakravarti · "
             f"edit chitra_holdings.csv in the repo, not this tab"],
            COLS]
    ws.clear()
    ws.update(values=head + body,
              range_name=f"A1:{chr(64 + len(COLS))}{len(head) + len(body)}")
    try:
        ws.format(f"A1:{chr(64 + len(COLS))}2", {"textFormat": {"bold": True}})
    except Exception:
        pass                     # cosmetics never cost the data
    log(f"Chitra tab: {len(rows)} holding(s), {m.get('As_Of', 'no date')}")
    return len(rows)


def dashboard_row(ticker: str, rows, quotes, extract, width: int) -> list | None:
    """One CHITRA line for a Dashboard block, or None if she does not hold it.

    Shaped to POS_HDR with the coverage columns left BLANK — there is no
    Schwab connection to this account, so `N` would be asserting something
    this code cannot see.
    """
    hit = next((r for r in rows if r["Ticker"] == ticker.upper()), None)
    if hit is None:
        return None
    px, pct = _price(hit, quotes, extract)
    val = (px * hit["Qty"]) if px is not None else None
    upl = (val - hit["Cost_Basis"]) if val is not None else None
    row = [""] * width
    row[1:9] = [hit["Ticker"], ACCT_LABEL, round(hit["Qty"], 4),
                round(hit["Cost_Basis"] / hit["Qty"], 4) if hit["Qty"] else "",
                round(px, 4) if px is not None else "",
                round(pct, 2) if pct is not None else "",
                round(val, 2) if val is not None else "",
                round(upl, 2) if upl is not None else ""]
    return row
