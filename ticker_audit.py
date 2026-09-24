#!/usr/bin/env python3
"""
ticker_audit.py
===============
Which tickers do I actually hold that STOCK_TICKERS has never heard of?

    .venv/bin/python ticker_audit.py

STOCK_TICKERS in app.py is hand-maintained, so it drifts: you buy something,
forget to add it, and from that moment the signals pipeline is blind to it —
no RSI, no Supertrend, no exit signal, no coverage row in stocks_signals.csv.
The Dashboard still shows it (that reads Schwab positions directly), which is
why the two counts disagree and why the gap is easy to miss.

Read-only. Prints; changes nothing.

Runs on Pi 1 — it needs Schwab credentials.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def master_list() -> list[str]:
    """Parse STOCK_TICKERS out of app.py without importing it.

    app.py pulls in Streamlit and a great deal else; this only needs a literal
    list, and literal_eval cannot execute anything.
    """
    src = (HERE / "app.py").read_text(encoding="utf-8")
    m = re.search(r"STOCK_TICKERS\s*=\s*\[(.*?)\]", src, re.S)
    if not m:
        sys.exit("could not find STOCK_TICKERS in app.py")
    return [str(t).strip().upper() for t in ast.literal_eval("[" + m.group(1) + "]")]


def main() -> int:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))

    import jobStocksSignals as job
    from orders_sheet import fetch_positions_detailed

    master = master_list()
    dupes = sorted({t for t in master if master.count(t) > 1})

    pos = fetch_positions_detailed(job.get_schwab_client(), log=job.log)
    if pos is None or pos.empty:
        print("no positions returned — nothing to compare against")
        return 1

    held = {}
    for _, r in pos.iterrows():
        t = str(r["Ticker"]).strip().upper()
        held[t] = held.get(t, 0.0) + float(r["Market_Value"] or 0)

    missing = sorted(set(held) - set(master), key=lambda t: -held[t])
    listed_unheld = sorted(set(master) - set(held))

    print(f"\nSTOCK_TICKERS entries : {len(master)}"
          + (f"   ⚠️  duplicates: {', '.join(dupes)}" if dupes else ""))
    print(f"tickers held at Schwab: {len(held)}")
    print(f"held but NOT listed   : {len(missing)}\n")

    if missing:
        # Biggest position first — that is the one whose missing exit signal
        # costs the most.
        print("MISSING — held but absent from STOCK_TICKERS, by position size:")
        for t in missing:
            print(f"   {t:<8} ${held[t]:>12,.2f}")
        print("\npaste-ready:\n")
        print("    " + ", ".join(f'"{t}"' for t in missing))
    else:
        print("Nothing missing — every holding is in the list.")

    print(f"\nlisted but not held: {len(listed_unheld)} "
          f"(watchlist entries, expected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
