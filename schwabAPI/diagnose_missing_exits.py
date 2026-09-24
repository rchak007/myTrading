#!/usr/bin/env python3
"""
diagnose_missing_exits.py
=========================
Confirm why our FIFO share count exceeds Schwab's on 23 tickers.

    ../.venv/bin/python diagnose_missing_exits.py

Read-only. Offline — uses the cached transactions, calls Schwab for nothing.

WHAT WE KNOW SO FAR
    The first hypothesis — that the engine's discarding of JOURNAL_IN/OUT was
    losing out-legs — is DISPROVED. There are 8 of each and they net to exactly
    zero (+4415 / -4415), per symbol as well as overall. Discarding them is safe.

    The real gap is that there are **zero** TRANSFER_OUT and **zero**
    XFER_OUT_BASIS rows, against 25 TRANSFER_IN and 17 XFER_IN_BASIS. Positions
    that left the household — ACATS out to another broker, a closed account —
    have no out-leg in any form the engine recognises.

    Two ways that can happen, and the descriptions tell them apart:
      * the out-leg exists but its wording is not matched by RE_XFER_EXT /
        RE_SYS_XFER, so _classify_zero_cash falls through to SPLIT_REMOVE and
        the shares leave WITHOUT realizing P&L; or
      * the out-leg was never fetched at all, in which case nothing non-trade
        appears for that symbol.

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

    # The mismatched tickers come from ticker_pl_summary.csv, NOT anomalies.csv
    # — the first version read anomalies.csv, which holds the 26 UNKNOWN_BASIS
    # entries and none of the 23 quantity gaps.
    summ = REPORT_DIR / "ticker_pl_summary.csv"
    if not summ.exists():
        print(f"\n{summ} not found — run build_pl_report.py first")
        return 1
    a = pd.read_csv(summ)
    if "qty_mismatch" not in a.columns:
        print(f"\nunexpected columns: {list(a.columns)}")
        return 1
    bad = a[a["qty_mismatch"].abs() > 0.005][["symbol", "qty_mismatch",
                                              "open_qty", "schwab_qty"]]
    print(f"\n=== {len(bad)} ticker(s) where our count != Schwab's ===\n")

    for _, row in bad.iterrows():
        sym = str(row["symbol"])
        s_rows = led[led["symbol"] == sym]
        gap = float(row["qty_mismatch"])
        net = s_rows.groupby("action")["qty"].sum()
        parts = ", ".join(f"{k}={v:+.2f}" for k, v in net.items())
        print(f"{sym:<20} ours={row['open_qty']:>10.2f}  "
              f"schwab={row['schwab_qty']:>10.2f}  gap={gap:+.2f}")
        print(f"    {parts}")

        # The raw wording is the thing that decides classification, so print it
        # for every non-trade row. An unrecognised description falls through to
        # SPLIT_ADD/SPLIT_REMOVE, which is how a transfer-out or a ticker
        # rename can be silently booked as a corporate action.
        nt = s_rows[~s_rows["action"].isin(["BUY", "SELL", "DIVIDEND", "INTEREST"])]
        for _, r in nt.iterrows():
            print(f"      {str(r['date'])[:10]}  {r['action']:<14} "
                  f"qty={r['qty']:>10.2f}  type={r['txn_type']:<20} "
                  f"{str(r['description'])[:70]}")
        if nt.empty:
            print("      (no non-trade rows at all — the exit was never recorded"
                  " in any form)")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
