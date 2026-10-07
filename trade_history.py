#!/usr/bin/env python3
"""
trade_history.py
================
Every trade that ACTUALLY EXECUTED, newest first, on the History tab.

    .venv/bin/python trade_history.py --list      # what is recorded
    .venv/bin/python trade_history.py --rebuild   # re-render the tab from file

BOTH SOURCES, BECAUSE SCHWAB MAKES NO DISTINCTION
    A fill is a fill. Whether the order came from the Pi's engine or from
    Chakravarti tapping Schwab on his phone, it lands in the same
    transactions feed. So this records everything that executed and then
    ATTRIBUTES it, rather than trying to keep two separate logs in step.

        PI      the fill's order id appears in the engine's ledger
        MANUAL  it does not — placed by hand at Schwab
        ?       no order id came back on the fill, so it cannot be told

WHAT THIS IS NOT
    Not the Orders tab's archive. That tab holds INTENTS — things that might
    happen. This holds what did. An intent that never triggered leaves no
    trace here, and correctly so.

THE FILE IS THE RECORD, THE SHEET IS A VIEW
    ~/.local/state/myTrading/trade_history.csv is append-only and deduped on
    Schwab's activity id. The tab is rebuilt from it every cycle, sorted
    newest-first, so a row cannot be lost by someone editing the sheet and the
    "newest at the top" ordering costs nothing to maintain.

    Same arrangement as the reserve ledger, for the same reason: a sheet is a
    place to look, not a place to keep things.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE_DIR = Path(os.getenv("REMOTE_OPS_STATE",
                           Path.home() / ".local" / "state" / "myTrading"))
FILE = Path(os.getenv("TRADE_HISTORY_FILE", STATE_DIR / "trade_history.csv"))
TAB = "History"

COLS = ["Filled_At", "Acct", "Ticker", "Side", "Qty", "Price", "Amount",
        "Asset", "Source", "Row_ID", "Order_ID", "Ref"]
# How many rows the SHEET shows. The file keeps everything — this only bounds
# the write, because a tab with five years of fills is slow and unreadable.
SHEET_MAX = int(os.getenv("TRADE_HISTORY_ROWS", "400"))


def _num(v, default=None):
    try:
        if v is None or str(v).strip() == "":
            return default
        return float(str(v).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return default


def read(log=print) -> list[dict]:
    """Everything recorded so far. [] on anything unexpected."""
    if not FILE.exists():
        return []
    try:
        with FILE.open(newline="", encoding="utf-8") as fh:
            return [dict(r) for r in csv.DictReader(fh)]
    except Exception as e:
        log(f"⚠️  could not read {FILE.name}: {e}")
        return []


def _attribution(order_id: str, ledger_ids: dict) -> tuple[str, str]:
    """(Source, Row_ID) for one fill.

    An empty order id is reported as "?" rather than guessed at. Calling an
    unattributable fill MANUAL would be a claim, and the one thing this file
    is for is knowing which trades the machine made.
    """
    oid = str(order_id or "").strip()
    if not oid:
        return "?", ""
    row_id = ledger_ids.get(oid)
    return ("PI", row_id) if row_id else ("MANUAL", "")


def ledger_order_ids(log=print) -> dict:
    """{schwab_order_id: Row_ID} from the engine's ledger."""
    try:
        import order_exec_config as cfg
        if not cfg.LEDGER.exists():
            return {}
        with cfg.LEDGER.open(newline="", encoding="utf-8") as fh:
            return {str(r.get("schwab_order_id", "")).strip(): r.get("row_id", "")
                    for r in csv.DictReader(fh)
                    if str(r.get("schwab_order_id", "")).strip()
                    not in ("", "UNKNOWN")}
    except Exception as e:
        log(f"⚠️  could not read the order ledger for attribution: {e}")
        return {}


def record(fills_df, log=print) -> int:
    """Append new fills. Returns how many were added.

    DEDUPED ON Schwab's activity id, the same key cash_reserve's replay guard
    uses. The fills window is re-fetched every cycle, so without this the same
    trade would be recorded every 35 minutes forever.
    """
    if fills_df is None or getattr(fills_df, "empty", True):
        return 0
    have = {str(r.get("Ref", "")).strip() for r in read(log=log)}
    ids = ledger_order_ids(log=log)

    new = []
    for _, f in fills_df.iterrows():
        ref = str(f.get("Ref", "")).strip()
        if not ref or ref in have:
            continue
        src, row_id = _attribution(f.get("Order_ID"), ids)
        new.append({
            "Filled_At": str(f.get("Fill_Time", ""))[:19],
            "Acct": str(f.get("Account", "")), "Ticker": str(f.get("Ticker", "")),
            "Side": str(f.get("Side", "")),
            "Qty": _num(f.get("Fill_QTY"), ""), "Price": _num(f.get("Fill_Price"), ""),
            "Amount": _num(f.get("Net_Amount"), ""),
            "Asset": str(f.get("Asset_Type", "")),
            "Source": src, "Row_ID": row_id,
            "Order_ID": str(f.get("Order_ID", "") or ""), "Ref": ref,
        })
        have.add(ref)

    if not new:
        return 0
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        first = not FILE.exists()
        with FILE.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
            if first:
                w.writeheader()
            for r in new:
                w.writerow(r)
    except Exception as e:
        log(f"⚠️  could not append to {FILE.name}: {e}")
        return 0

    by_src = {}
    for r in new:
        by_src[r["Source"]] = by_src.get(r["Source"], 0) + 1
    log(f"Trade history: {len(new)} new fill(s) — "
        + ", ".join(f"{v} {k}" for k, v in sorted(by_src.items())))
    return len(new)


def newest_first(rows) -> list[dict]:
    """Sorted by fill time descending. Rows with no time sink to the bottom
    rather than being dropped — a fill with a missing timestamp is still a
    trade that happened."""
    return sorted(rows, key=lambda r: (str(r.get("Filled_At") or ""),
                                       str(r.get("Ref") or "")), reverse=True)


def write_tab(book, log=print) -> int:
    """Rebuild the History tab from the file, newest at the top."""
    rows = newest_first(read(log=log))
    if not rows:
        log("Trade history: nothing recorded yet")
        return 0
    shown = rows[:SHEET_MAX]
    try:
        ws = book.worksheet(TAB)
    except Exception:
        ws = book.add_worksheet(title=TAB, rows=SHEET_MAX + 20, cols=len(COLS) + 2)

    last = chr(ord("A") + len(COLS) - 1)
    head = [[f"HISTORY — every executed trade, newest first. "
             f"{len(rows)} recorded"
             + (f", showing {len(shown)}" if len(shown) < len(rows) else "")
             + ". Source PI = placed by the order engine, MANUAL = placed by "
               "hand at Schwab. Read-only; rebuilt from "
               "trade_history.csv every cycle."],
            COLS]
    body = [[r.get(c, "") for c in COLS] for r in shown]
    try:
        # Clear generously: a shorter list than last time must not leave the
        # tail of a longer one behind, which is the stale-data failure this
        # sheet has had before.
        ws.batch_clear([f"A1:{last}{max(len(body) + 2, SHEET_MAX + 2)}"])
        ws.update(values=head + body, range_name=f"A1:{last}{len(head) + len(body)}")
        ws.format(f"A1:{last}2", {"textFormat": {"bold": True}})
    except Exception as e:
        log(f"⚠️  History tab write failed: {e}")
        return 0
    log(f"History tab: {len(shown)} row(s) of {len(rows)}")
    return len(shown)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Executed-trade history")
    ap.add_argument("--list", action="store_true", help="print what is recorded")
    ap.add_argument("--rebuild", action="store_true",
                    help="re-render the History tab from the file")
    args = ap.parse_args()

    rows = read()
    if args.rebuild:
        from orders_sheet import _open_book
        return 0 if write_tab(_open_book()) else 1

    print(f"{len(rows)} fill(s) in {FILE}")
    by = {}
    for r in rows:
        by[r.get("Source", "?")] = by.get(r.get("Source", "?"), 0) + 1
    for k, v in sorted(by.items()):
        print(f"  {k:<8}{v}")
    if args.list:
        print()
        for r in newest_first(rows)[:40]:
            print(f"  {r.get('Filled_At',''):<20}{r.get('Acct',''):<5}"
                  f"{r.get('Ticker',''):<8}{r.get('Side',''):<5}"
                  f"{str(r.get('Qty','')):>9} @ {str(r.get('Price','')):<10}"
                  f"{r.get('Source','')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
