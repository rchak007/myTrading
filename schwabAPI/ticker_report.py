#!/usr/bin/env python3
"""
ticker_report.py
----------------
Print every transaction the P&L engine saw for one ticker, in date order, with
the running share count and cost basis after each row -- so a number can be
checked line by line against Schwab's own transaction history.

This reads the report CSVs produced by build_pl_report.py. It never calls
Schwab and never touches the cache.

    python ticker_report.py TSLA
    python ticker_report.py TSLA --all        # include options on TSLA
    python ticker_report.py TSLA --out ~/pl_out

Validating against Schwab:
  * `position_after` is what we think you held after that row. Compare it to
    the account history in Schwab's UI on the same date.
  * The first row where `position_after` diverges from reality is the
    transaction we mishandled -- or the one Schwab never gave us.
  * `qty_mismatch` in the summary is the total error today. If it equals a
    single missing SELL or transfer, this listing will show where.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULT_OUT = Path.home() / "pl_out"


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-ticker transaction listing")
    ap.add_argument("ticker")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="report directory written by build_pl_report.py")
    ap.add_argument("--all", action="store_true",
                    help="also include options and any symbol starting with the ticker")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    txns_path = out / "ticker_txns.csv"
    summary_path = out / "ticker_pl_summary.csv"

    if not txns_path.exists():
        print(f"No {txns_path}. Run build_pl_report.py first.", file=sys.stderr)
        return 1

    tick = args.ticker.upper().strip()
    df = pd.read_csv(txns_path)
    df["symbol"] = df["symbol"].astype(str)

    sel = df["symbol"].str.startswith(tick) if args.all else df["symbol"] == tick
    rows = df[sel].copy()
    if rows.empty:
        print(f"No transactions for {tick} in {txns_path}")
        print("(try --all to include options, or check the spelling)")
        return 1

    rows = rows.sort_values(["date", "activity_id"])

    show = ["date", "account", "symbol", "action", "qty", "price", "gross",
            "fees", "realized_pl", "position_after", "cost_basis_after"]
    show = [c for c in show if c in rows.columns]

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", 46)

    print(f"\n=== {tick}: {len(rows)} transactions ===")
    print(rows[show].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print(f"\n=== descriptions (what Schwab actually called each row) ===")
    for _, r in rows.iterrows():
        print(f"  {r['date']}  {str(r['action']):<16} {str(r.get('description',''))[:90]}")

    # ------------------------------------------------------------- summary
    if summary_path.exists():
        s = pd.read_csv(summary_path)
        s["symbol"] = s["symbol"].astype(str)
        srow = s[s["symbol"].str.startswith(tick)] if args.all else s[s["symbol"] == tick]
        if not srow.empty:
            print(f"\n=== summary ===")
            for _, r in srow.iterrows():
                print(f"\n  {r['symbol']}")
                for c in srow.columns:
                    if c == "symbol":
                        continue
                    v = r[c]
                    if pd.isna(v):
                        continue
                    print(f"      {c:<18} {v}")

    # ------------------------------------------------------------- hint
    net = rows.loc[rows["action"].isin(["BUY"]), "qty"].sum() if "action" in rows else 0
    print("\n--- how to validate ---")
    print("  1. Open the same account's history in Schwab for this ticker.")
    print("  2. Walk the rows above in date order and compare position_after.")
    print("  3. The FIRST date where they disagree is the problem row; everything")
    print("     after it inherits the error. Send me that date and I'll trace it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
