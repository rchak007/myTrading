#!/usr/bin/env python3
"""
diagnose_missing_exits.py
=========================
Confirm why our FIFO share count exceeds Schwab's on 23 tickers.

    ../.venv/bin/python diagnose_missing_exits.py

Read-only. Offline — uses the cached transactions, calls Schwab for nothing.

THE HYPOTHESIS
    ticker_txns.csv (what the engine kept) shows 25 TRANSFER_IN and 17
    XFER_IN_BASIS, with ZERO of either OUT. A "System transfer" moves shares
    between two of your own accounts: one leg +N, the other -N. Seventeen ins
    and no outs cannot be real.

    pl_engine.py:232 drops JOURNAL_IN/JOURNAL_OUT entirely, on the reasoning
    that a move between your own accounts nets to zero. That holds only if
    BOTH legs are classified as journals. If the in-leg arrives as a TRADE
    ("System transfer", zero net cash -> XFER_IN_BASIS) while the out-leg
    arrives as a JOURNAL, the in-leg is counted and the out-leg is thrown
    away — shares added, never removed.

    That would explain every symptom: the error is always in one direction,
    the amounts are round numbers, and realized P&L is understated because
    the basis stays in open_cost_basis instead of being realized.

WHAT THIS PRINTS
    The action histogram BEFORE the engine drops anything, then per mismatched
    ticker the net quantity each action contributes — so the dropped
    JOURNAL_OUT volume can be compared directly against the mismatch.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pandas as pd                                    # noqa: E402

import txn_cache                                       # noqa: E402
import txn_parser                                      # noqa: E402
from schwab_client import REPORT_DIR                    # noqa: E402


def main() -> int:
    cache = txn_cache.load_all_cached()
    if cache.empty:
        print("cache is empty — run build_pl_report.py first")
        return 1
    print(f"cached transactions: {len(cache)}\n")

    led = txn_parser.parse_ledger(cache)
    if led.empty:
        print("parser produced no rows")
        return 1

    print("=== action histogram, BEFORE the engine drops anything ===")
    print(led["action"].value_counts().to_string())

    dropped = led[led["action"].isin(["JOURNAL_IN", "JOURNAL_OUT"])]
    print(f"\nrows the engine discards (JOURNAL_IN/OUT): {len(dropped)}")
    if not dropped.empty:
        net = dropped.groupby("action")["qty"].agg(["count", "sum"])
        print(net.to_string())
        print("\nIf these netted to zero per symbol the discard would be safe.")

    # Which tickers are mismatched, straight from the report.
    anom = REPORT_DIR / "anomalies.csv"
    if not anom.exists():
        print(f"\n{anom} not found — run build_pl_report.py first")
        return 1
    a = pd.read_csv(anom)
    col = next((c for c in ("symbol", "Symbol") if c in a.columns), None)
    mis = next((c for c in ("qty_mismatch", "mismatch") if c in a.columns), None)
    if not col:
        print(f"\nunexpected anomalies.csv columns: {list(a.columns)}")
        return 1
    bad = a[[col] + ([mis] if mis else [])].copy()

    print("\n=== per mismatched ticker: net qty by action ===")
    print("(dropped JOURNAL rows are marked *)\n")
    for _, row in bad.iterrows():
        sym = str(row[col])
        s = led[led["symbol"] == sym]
        if s.empty:
            continue
        gap = row[mis] if mis else None
        net = s.groupby("action")["qty"].sum()
        parts = ", ".join(
            f"{'*' if k.startswith('JOURNAL') else ''}{k}={v:+.2f}"
            for k, v in net.items())
        jrn = net.get("JOURNAL_OUT", 0.0) + net.get("JOURNAL_IN", 0.0)
        verdict = ""
        if gap is not None and abs(jrn) > 0.005:
            verdict = ("   <-- dropped journals EXPLAIN the gap"
                       if abs(abs(jrn) - abs(float(gap))) < 0.01
                       else f"   (dropped journals net {jrn:+.2f})")
        print(f"{sym:<18} mismatch={float(gap):>9.2f}{verdict}" if gap is not None
              else f"{sym:<18}")
        print(f"    {parts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
