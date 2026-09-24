#!/usr/bin/env python3
"""
cache_coverage.py
=================
Show what the transaction cache actually covers, per account, per month.

    ../.venv/bin/python cache_coverage.py
    ../.venv/bin/python cache_coverage.py --symbol AEHR

Read-only, offline.

WHY
    Schwab's own history shows an AEHR transfer pair on 2023-09-05 in account
    ...171, and `dump_raw_txn.py AEHR --date 2023-09` finds NOTHING in the
    cache. So the rows were never fetched — this is not a parsing problem.

    The watermark only ever moves forward: sync_account() reads last_date and
    fetches from last_date - OVERLAP_DAYS onward, then _save_state() writes the
    new maximum. A window missed during the original cold start — a transient
    error, a chunk that came back empty — is therefore never retried. The gap
    is permanent until the state file is deleted and the account refetched.

    A month with zero transactions is normal in a dormant account. A month with
    zero transactions surrounded by busy months, in an account that was clearly
    being traded, is a hole.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pandas as pd                                    # noqa: E402

import txn_cache                                       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Transaction cache coverage by month")
    ap.add_argument("--symbol", help="also list every cached date for this ticker")
    args = ap.parse_args()

    cache = txn_cache.load_all_cached()
    if cache.empty:
        print("cache is empty — run build_pl_report.py first")
        return 1

    cache = cache.copy()
    cache["date"] = pd.to_datetime(cache["date"], errors="coerce")
    cache = cache.dropna(subset=["date"])
    cache["month"] = cache["date"].dt.to_period("M")

    acct_col = next((c for c in ("account_hash", "account_number") if c in cache.columns),
                    None)
    if acct_col is None:
        print(f"no account column; have {list(cache.columns)}")
        return 1

    print(f"cached transactions: {len(cache)}\n")

    for acct, grp in cache.groupby(acct_col):
        lo, hi = grp["month"].min(), grp["month"].max()
        months = pd.period_range(lo, hi, freq="M")
        counts = grp["month"].value_counts()
        empty = [str(m) for m in months if counts.get(m, 0) == 0]
        print(f"=== {str(acct)[:8]}  {lo} .. {hi}  "
              f"({len(grp)} txns over {len(months)} months)")
        if empty:
            # Runs of empty months are what matter; a single quiet month is
            # unremarkable, a six-month silence in a traded account is not.
            print(f"    {len(empty)} month(s) with ZERO transactions:")
            print("      " + ", ".join(empty))
        else:
            print("    no empty months")
        print()

    if args.symbol:
        sym = args.symbol.upper()
        hits = cache[cache["raw_json"].astype(str).str.upper().str.contains(sym, na=False)]
        print(f"=== every cached date mentioning {sym}: {len(hits)} row(s)")
        if hits.empty:
            print("    none at all")
        else:
            for acct, g in hits.groupby(acct_col):
                ds = sorted({str(d.date()) for d in g["date"]})
                print(f"    {str(acct)[:8]}  {len(ds)} date(s): "
                      f"{ds[0]} .. {ds[-1]}")
                print(f"      {', '.join(ds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
