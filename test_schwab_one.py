#!/usr/bin/env python3
"""
test_schwab_one.py
------------------
Standalone test: fetch QTY and VALUE for ONE ticker from your Schwab account.

Reuses the same create_schwab_client / SchwabAuthError as app.py so behavior is
identical to the dashboard. Designed to run on BOTH WSL and the Pi to isolate
where Schwab fetching breaks.

USAGE
-----
    # WSL (from your app dir, where tokens.json lives)
    python test_schwab_one.py NVDA

    # Pi
    cd ~/github/myTrading   # or wherever app.py + tokens.json live
    python3 test_schwab_one.py NVDA

    # Verbose: dumps all accounts/positions before filtering to your ticker
    python3 test_schwab_one.py NVDA --verbose

    # Point at a specific app dir if you're running from elsewhere
    python3 test_schwab_one.py NVDA --app-dir /home/pi/github/myTrading
"""
from __future__ import annotations

import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone


def main():
    ap = argparse.ArgumentParser(description="Schwab one-ticker QTY/VALUE test")
    ap.add_argument("ticker", help="Single ticker to look up (e.g. NVDA, AAPL)")
    ap.add_argument(
        "--app-dir",
        default=None,
        help="Path to dir containing app.py + tokens.json (defaults to script's dir)",
    )
    ap.add_argument("--verbose", "-v", action="store_true", help="Dump every position")
    ap.add_argument(
        "--user-id",
        default="main",
        help="USER_ID label (matches app.py default of 'main')",
    )
    args = ap.parse_args()

    ticker = args.ticker.upper().strip()

    # Resolve app dir
    app_dir = Path(args.app_dir).resolve() if args.app_dir else Path(__file__).resolve().parent
    print(f"[INFO] app_dir = {app_dir}")
    print(f"[INFO] ticker  = {ticker}")
    print(f"[INFO] user_id = {args.user_id}")

    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))

    # Load .env from app_dir (same as app.py does via load_dotenv() at startup).
    # The Schwab helper reads app_key / app_secret / callback_url from env.
    env_path = app_dir / ".env"
    try:
        from dotenv import load_dotenv
        # override=True so a value already in shell env doesn't silently win
        loaded = load_dotenv(dotenv_path=env_path, override=False)
        print(f"[INFO] .env  = {env_path}  (exists={env_path.exists()}, loaded={loaded})")
    except ImportError:
        print("[WARN] python-dotenv not installed — relying on shell env vars only")

    # Sanity-check the three Schwab vars (don't print secrets, just presence/length)
    for k in ("app_key", "app_secret", "callback_url"):
        v = os.getenv(k, "")
        if k == "callback_url":
            print(f"[INFO] env {k:13s} = {v if v else '<MISSING>'}")
        else:
            print(f"[INFO] env {k:13s} = {('SET (' + str(len(v)) + ' chars)') if v else '<MISSING>'}")

    # Same token paths app.py uses
    token_paths = [
        app_dir / "tokens.json",
        app_dir / "data" / "schwab" / "tokens.json",
    ]
    print("[INFO] Token paths:")
    for p in token_paths:
        if p.exists():
            sz = p.stat().st_size
            mt = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
            print(f"        ✓ {p}  ({sz} bytes, mtime={mt})")
        else:
            print(f"        ✗ {p}  (missing)")

    # Import the same helper app.py uses
    try:
        from data.schwab.schwab_helper import create_schwab_client, SchwabAuthError
    except Exception as e:
        print(f"[FATAL] Could not import data.schwab.schwab_helper: {e}")
        print("        Run from your app dir, or pass --app-dir.")
        sys.exit(2)

    # LOCAL_ONLY=1 → never touch Supabase (matches your Pi setup)
    local_only = os.getenv("LOCAL_ONLY", "1") != "0"
    print(f"[INFO] LOCAL_ONLY = {local_only}")

    # Build client + fetch positions
    try:
        client = create_schwab_client(args.user_id, token_paths, local_only=local_only)
        data = client.fetch_positions()
    except SchwabAuthError as e:
        print(f"[AUTH-ERROR] {e}")
        print("             → On Pi: copy a fresh tokens.json from WSL after re-auth.")
        sys.exit(3)
    except Exception as e:
        print(f"[ERROR] fetch_positions failed: {type(e).__name__}: {e}")
        sys.exit(4)

    print(f"[OK]  Got {len(data)} account(s) from Schwab")

    # Walk accounts/positions, find the ticker
    matches = []
    total_positions = 0
    for i, acct in enumerate(data):
        sa = acct.get("securitiesAccount", {}) or {}
        acct_type = sa.get("type", "UNKNOWN")
        positions = sa.get("positions", []) or []
        total_positions += len(positions)

        if args.verbose:
            print(f"\n[ACCT {i+1}] type={acct_type}  positions={len(positions)}")

        for pos in positions:
            inst = pos.get("instrument", {}) or {}
            sym = inst.get("symbol")
            atype = inst.get("assetType")
            long_qty = float(pos.get("longQuantity") or 0.0)
            short_qty = float(pos.get("shortQuantity") or 0.0)
            qty = long_qty - short_qty
            mkt_value = float(pos.get("marketValue") or 0.0)

            if args.verbose:
                print(f"    - {sym:8s} {atype:22s}  qty={qty:>10.4f}  value=${mkt_value:>12,.2f}")

            if sym and sym.upper() == ticker:
                matches.append({
                    "account_idx": i + 1,
                    "account_type": acct_type,
                    "asset_type": atype,
                    "qty": qty,
                    "value": mkt_value,
                })

    print(f"\n[INFO] Scanned {total_positions} total positions across {len(data)} account(s)")

    if not matches:
        print(f"\n[RESULT] {ticker}: NOT FOUND in any Schwab account")
        print("         → Either you don't hold it, or asset_type filter is hiding it.")
        print("         → Re-run with --verbose to dump every position.")
        sys.exit(1)

    # Aggregate (in case it sits in multiple accounts)
    total_qty = sum(m["qty"] for m in matches)
    total_val = sum(m["value"] for m in matches)

    print(f"\n[RESULT] {ticker}")
    for m in matches:
        print(f"   acct#{m['account_idx']} ({m['account_type']:8s}) {m['asset_type']:8s}"
              f"  qty={m['qty']:>10.4f}  value=${m['value']:>12,.2f}")
    print(f"   {'TOTAL':38s}  qty={total_qty:>10.4f}  value=${total_val:>12,.2f}")


if __name__ == "__main__":
    main()