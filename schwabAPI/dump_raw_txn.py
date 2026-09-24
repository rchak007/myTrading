#!/usr/bin/env python3
"""
dump_raw_txn.py
===============
Print the raw Schwab JSON for cached transactions, to see why one was dropped.

    ../.venv/bin/python dump_raw_txn.py AEHR
    ../.venv/bin/python dump_raw_txn.py AEHR --date 2023-09-05
    ../.venv/bin/python dump_raw_txn.py --grep "TRANSFER OF SECURITY OR OPTION OUT"

Read-only, offline.

WHY
    Schwab's own history shows a matched pair on 2023-09-05 in account ...171:

        Journaled Shares   TDA TRAN - TRANSFER OF SECURITY OR OPTION OUT (AEHR)  -109
        Internal Transfer  AEHR TEST SYSTEMS                                     +109

    Neither reaches our ledger — AEHR shows no non-trade rows at all — even
    though the first description matches RE_XFER_EXT and should classify as
    TRANSFER_OUT. Something drops them before classification.

    parse_ledger keeps an item only if its instrument's assetType is in
    SECURITY_ASSETS and its symbol is not a cash symbol. So the likely culprits
    are an unexpected assetType, an absent instrument, or empty transferItems.
    This prints the payload so we stop guessing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import txn_cache                                       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Dump raw Schwab transaction JSON")
    ap.add_argument("symbol", nargs="?", help="ticker to match inside the payload")
    ap.add_argument("--date", help="only rows whose date starts with this, e.g. 2023-09")
    ap.add_argument("--grep", help="match this text in the raw payload instead")
    ap.add_argument("--limit", type=int, default=8, help="max payloads to print")
    args = ap.parse_args()

    if not args.symbol and not args.grep:
        ap.error("give a SYMBOL or --grep")

    cache = txn_cache.load_all_cached()
    if cache.empty:
        print("cache is empty — run build_pl_report.py first")
        return 1
    print(f"cached transactions: {len(cache)}\n")

    needle = (args.grep or args.symbol or "").upper()
    shown = 0
    for _, r in cache.iterrows():
        raw = r.get("raw_json")
        if not isinstance(raw, str) or needle not in raw.upper():
            continue
        if args.date and not str(r.get("date", "")).startswith(args.date):
            continue
        try:
            t = json.loads(raw)
        except Exception:
            continue

        items = t.get("transferItems") or []
        print("=" * 78)
        print(f"date={r.get('date')}  type={t.get('type')}  "
              f"activityId={t.get('activityId')}")
        print(f"description: {t.get('description')}")
        print(f"netAmount  : {t.get('netAmount')}")
        print(f"transferItems: {len(items)}")
        for it in items:
            inst = it.get("instrument") or {}
            # assetType and symbol are exactly what decide whether parse_ledger
            # keeps this leg at all.
            print(f"   assetType={inst.get('assetType')!r:<26} "
                  f"symbol={inst.get('symbol')!r:<20} "
                  f"amount={it.get('amount')!r}  cost={it.get('cost')!r}")
        if not items:
            print("   (no transferItems — nothing for the parser to read)")
        shown += 1
        if shown >= args.limit:
            print(f"\n(stopped at --limit {args.limit})")
            break

    if not shown:
        print(f"no cached payload contains {needle!r}"
              + (f" with date starting {args.date}" if args.date else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
