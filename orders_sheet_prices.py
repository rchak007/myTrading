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

# Rows whose column C holds one of these are section furniture, not holdings.
NOT_A_TICKER = {"TICKER", "POSITIONS", "ORDERS", "ACCT", "DASHBOARD", ""}


def _num(v):
    try:
        if v is None or v == "":
            return None
        return float(str(v).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def extract_price(entry: dict) -> tuple[float | None, float | None]:
    """(price, day_percent) from one Schwab quote entry.

    The payload splits into `quote` (regular session), `extended` (pre/post)
    and `regular`. Measured 2026-09-23: `realtime: true`, and after hours the
    `extended` block is the one that moves — which is the whole point of using
    quotes rather than dividing marketValue by quantity.

    Preference order is regular-session last price first, because that is what
    a chart shows during the day; extended only fills in when the session is
    closed and `quote` has stopped moving.
    """
    q = entry.get("quote") or {}
    ext = entry.get("extended") or {}
    reg = entry.get("regular") or {}

    def _stamp(src) -> float:
        """Most recent activity in this block, epoch ms."""
        return max(_num(src.get("tradeTime")) or 0,
                   _num(src.get("quoteTime")) or 0)

    def _last(src):
        for key in ("lastPrice", "mark"):
            v = _num(src.get(key))
            if v and v > 0:
                return v
        return None

    # Pick the block that traded MOST RECENTLY, rather than a fixed order.
    #
    # Preferring `quote` first was wrong after hours: at 23:54 PT it still held
    # the 16:00 close (AMD 614.61) while `extended` had the live overnight
    # print (608.00). Preferring `extended` first would be equally wrong during
    # the session, when it holds the stale pre-market number. The timestamp is
    # the only thing that answers "which of these is the latest price".
    candidates = [(_stamp(ext), _last(ext)), (_stamp(q), _last(q))]
    candidates = [(t, p) for t, p in candidates if p]
    price = max(candidates, key=lambda c: c[0])[1] if candidates else None

    if price is None:                       # nothing live; fall back to closes
        for src, key in ((reg, "regularMarketLastPrice"), (q, "closePrice")):
            v = _num(src.get(key))
            if v and v > 0:
                price = v
                break

    # Percent moves with whichever block the price came from, so an overnight
    # price is not paired with the regular session's move.
    pct = None
    from_ext = bool(candidates) and price == _last(ext) and _stamp(ext) >= _stamp(q)
    order = ((ext, "netPercentChange"), (q, "netPercentChange"),
             (reg, "regularMarketPercentChange")) if from_ext else \
            ((q, "netPercentChange"), (reg, "regularMarketPercentChange"),
             (ext, "netPercentChange"))
    for src, key in order:
        v = _num(src.get(key))
        if v is not None:
            pct = v
            break
    return price, pct


def fetch_quotes(client_wrapper, symbols: list[str], log=print) -> dict:
    """One call for every symbol.

    Measured with probe_quotes_api.py on 2026-09-23: the method is `quotes`
    on the schwabdev client (reachable via the wrapper's get_client(), since
    the wrapper itself forwards only fetch_positions), and it takes a single
    COMMA-JOINED STRING — the same shape `types` needed for transactions.
    """
    if not symbols:
        return {}
    inner = client_wrapper
    getter = getattr(client_wrapper, "get_client", None)
    if callable(getter):
        inner = getter() or client_wrapper

    meth = getattr(inner, "quotes", None)
    if not callable(meth):
        log("⚠️  this schwabdev has no quotes() — prices not refreshed")
        return {}

    resp = meth(",".join(symbols))
    body = resp.json() if hasattr(resp, "json") else resp
    if not isinstance(body, dict):
        log(f"⚠️  unexpected quotes response: {type(body).__name__}")
        return {}
    # Anything that is not keyed by a symbol we asked for is not a quote.
    return {k: v for k, v in body.items() if isinstance(v, dict)}


def main() -> int:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))

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

    # Position rows are those whose column B is a ticker and column C an
    # account. Detected fresh every run: the full job rebuilds this tab and
    # every row number changes, so a cached map would write into the wrong
    # cells for up to three minutes.
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
