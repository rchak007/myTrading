#!/usr/bin/env python3
"""
orders_sheet_prices.py
======================
Refresh Live_Price and Day_% on the Dashboard tab, and nothing else.

    .venv/bin/python orders_sheet_prices.py

WHY THIS EXISTS SEPARATELY
    jobStocksSignals.py writes the whole sheet, but it takes ~9 minutes and
    runs at :15 and :50 — so its prices are already ~9 minutes old on arrival
    and up to 35 minutes old before the next write. That is fine for signals
    and useless for "is it at my trigger right now".

    This does one Schwab quotes call for every ticker on the Dashboard and
    writes two columns. Seconds, not minutes, so it can run every few minutes.

WHAT IT WILL NOT DO
    Touch any other cell, write any file, or talk to git. It is sheet-only, so
    gitpush.py never sees it and there is no interaction with the output repo.

SAFETY
    Run under `flock -n` on the SAME lock as jobStocksSignals.py, so it simply
    skips while the big job is rebuilding the tab rather than writing into a
    half-built layout:

        */3 6-13 * * 1-5 cd /home/rchak007/github/myTrading && \\
          set -a && . ./.env && set +a && \\
          flock -n /tmp/jobmytrading.lock timeout 120 \\
          .venv/bin/python orders_sheet_prices.py \\
          >> /home/rchak007/.local/state/myTrading/prices_cron.log 2>&1

    Market hours only (6-13 PDT covers 6:30am-1pm). Row positions are re-read
    every run and matched by ticker — never cached — because the full job
    regenerates the tab wholesale and every row moves.

Runs on Pi 1.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:          # before the import below, not inside main()
    sys.path.insert(0, str(HERE))

# Rows whose column C holds one of these are section furniture, not holdings.
NOT_A_TICKER = {"TICKER", "POSITIONS", "ORDERS", "ACCT", "DASHBOARD", ""}

# Price extraction and the quotes call live in schwab_quotes so this and
# jobStocksSignals cannot drift apart — they must agree on what "the current
# price" means, or the Dashboard and stocks_signals.csv will disagree.
from schwab_quotes import extract_price, fetch_quotes  # noqa: E402


def main() -> int:
    import jobStocksSignals as job
    from orders_sheet import _open_book, _now, COL_LIVE_PRICE, COL_DAY_PCT
    log = job.log

    book = _open_book()
    try:
        ws = book.worksheet("Dashboard")
    except Exception:
        log("Dashboard tab does not exist yet — run jobStocksSignals.py first")
        return 1

    rows = ws.get_all_values()

    # Detected fresh every run: the full job rebuilds this tab and every row
    # number changes, so a cached map would write into the wrong cells.
    #
    # Track which SECTION we are in rather than guessing from the cell
    # contents. An ORDERS row holds the account in column B and the side in C,
    # so `171 | SELL` reads exactly like a ticker with an account beside it —
    # pricing that row would write into the orders table.
    targets: dict[str, list[int]] = {}
    in_positions = False
    for i, row in enumerate(rows, start=1):
        cells = list(row) + [""] * 4
        b = str(cells[1]).strip().upper()
        c = str(cells[2]).strip()

        if b == "POSITIONS":
            in_positions = True
            continue
        if b == "ORDERS":
            in_positions = False
            continue
        if not in_positions or not c or b in NOT_A_TICKER:
            continue
        # A ticker has at least one letter; an account number does not.
        if not any(ch.isalpha() for ch in b):
            continue
        targets.setdefault(b, []).append(i)

    if not targets:
        log("no position rows found on the Dashboard — nothing to price")
        return 0

    symbols = sorted(targets)
    quotes = fetch_quotes(job.get_schwab_client(), symbols, log=log)
    if not quotes:
        return 1

    payload, priced, missing = [], 0, []
    for tkr, row_nums in targets.items():
        entry = quotes.get(tkr)
        if not entry:
            missing.append(tkr)
            continue
        price, pct = extract_price(entry)
        if price is None:
            missing.append(tkr)
            continue
        priced += 1
        for r in row_nums:
            payload.append({"range": f"{COL_LIVE_PRICE}{r}", "values": [[price]]})
            payload.append({"range": f"{COL_DAY_PCT}{r}",
                            "values": [[round(pct, 2) if pct is not None else ""]]})

    stamp = _now()
    payload.append({"range": "H1", "values": [[f"prices {stamp}"]]})

    ws.batch_update(payload, value_input_option="USER_ENTERED")
    log(f"Prices: {priced}/{len(symbols)} ticker(s) across {len(payload) // 2} row(s)"
        + (f" · no quote for {', '.join(sorted(missing))}" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
