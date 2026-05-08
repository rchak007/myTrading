#!/usr/bin/env python3
"""
backfill_trade_log_usd.py

One-shot migration: adds three USD-value columns to existing bot trade CSVs.

  token_value_usd, stable_value_usd, total_value_usd

Inserted between `amount_ccy` and `tx_sig` in:
  - per-bot logs:   bot_trades_<bot_id>.csv
  - master log:     bot_trades_MASTER_4h.csv

Computation rule (same as the live log_trade function):
  - amount_ccy in {USDC, USDT, DAI, USD}  →  BUY  → all three USD columns = amount
  - otherwise                              →  SELL → all three USD columns = amount * price

Behavior:
  - Backs up each file to <file>.bak before rewriting.
  - Skips files that are already migrated (header already has token_value_usd).
  - Reads dir from BOT_TRADE_LOG_DIR env var, defaults to ./outputs.

Usage:
  python3 backfill_trade_log_usd.py                    # backfill ./outputs
  python3 backfill_trade_log_usd.py /path/to/outputs   # backfill custom dir
  python3 backfill_trade_log_usd.py --dry-run          # show what would change
"""
from __future__ import annotations

import csv
import os
import shutil
import sys
from typing import List

STABLE_CCYS = {"USDC", "USDT", "DAI", "USD"}

OLD_PER_BOT_HEADER = [
    "timestamp", "bot_name", "blockchain", "action", "regime_from", "regime_to",
    "price", "amount", "amount_ccy", "tx_sig", "dry_run",
]
NEW_PER_BOT_HEADER = [
    "timestamp", "bot_name", "blockchain", "action", "regime_from", "regime_to",
    "price", "amount", "amount_ccy",
    "token_value_usd", "stable_value_usd", "total_value_usd",
    "tx_sig", "dry_run",
]

OLD_MASTER_HEADER = [
    "timestamp", "bot_id", "bot_name", "blockchain", "action", "regime_from", "regime_to",
    "price", "amount", "amount_ccy", "tx_sig", "dry_run",
]
NEW_MASTER_HEADER = [
    "timestamp", "bot_id", "bot_name", "blockchain", "action", "regime_from", "regime_to",
    "price", "amount", "amount_ccy",
    "token_value_usd", "stable_value_usd", "total_value_usd",
    "tx_sig", "dry_run",
]


def compute_usd_values(price_str: str, amount_str: str, amount_ccy: str) -> tuple[str, str, str]:
    """Return (token_value_usd, stable_value_usd, total_value_usd) as formatted strings."""
    try:
        price  = float(price_str)
        amount = float(amount_str)
    except ValueError:
        return ("", "", "")

    if amount_ccy.strip().upper() in STABLE_CCYS:
        # BUY: amount is stable spent; token leg is worth ~amount USD at swap price
        stable_usd = amount
        token_usd  = amount
    else:
        # SELL: amount is tokens sold; stable leg is worth ~amount * price USD
        token_usd  = amount * price
        stable_usd = amount * price
    total_usd = token_usd
    return (f"{token_usd:.2f}", f"{stable_usd:.2f}", f"{total_usd:.2f}")


def detect_format(header: List[str]) -> str:
    """Return 'per_bot_old', 'per_bot_new', 'master_old', 'master_new', or 'unknown'."""
    if header == NEW_PER_BOT_HEADER:
        return "per_bot_new"
    if header == OLD_PER_BOT_HEADER:
        return "per_bot_old"
    if header == NEW_MASTER_HEADER:
        return "master_new"
    if header == OLD_MASTER_HEADER:
        return "master_old"
    return "unknown"


def migrate_file(path: str, dry_run: bool = False) -> str:
    """Migrate a single CSV file. Returns a status string for the report."""
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception as e:
        return f"  ERROR reading: {e}"

    if not rows:
        return "  empty file — skipped"

    header = rows[0]
    fmt = detect_format(header)

    if fmt in ("per_bot_new", "master_new"):
        return f"  already migrated ({fmt}) — skipped"

    if fmt == "unknown":
        return f"  UNKNOWN header (cols={len(header)}) — skipped"

    # Determine column indices in the OLD header for price, amount, amount_ccy
    if fmt == "per_bot_old":
        new_header = NEW_PER_BOT_HEADER
    else:  # master_old
        new_header = NEW_MASTER_HEADER

    price_idx     = header.index("price")
    amount_idx    = header.index("amount")
    ccy_idx       = header.index("amount_ccy")
    tx_sig_idx    = header.index("tx_sig")  # we insert right before this

    new_rows: list[list[str]] = [new_header]
    rebuilt = 0
    skipped_malformed = 0

    for row in rows[1:]:
        if len(row) != len(header):
            # Malformed — pad or skip; safer to keep the row but mark blanks
            skipped_malformed += 1
            # Pad to old header length so indices are safe
            row = row + [""] * (len(header) - len(row))

        try:
            tok_v, stab_v, tot_v = compute_usd_values(
                row[price_idx], row[amount_idx], row[ccy_idx]
            )
        except IndexError:
            tok_v, stab_v, tot_v = ("", "", "")

        # Build new row by inserting the 3 USD cols just before tx_sig
        new_row = row[:tx_sig_idx] + [tok_v, stab_v, tot_v] + row[tx_sig_idx:]
        new_rows.append(new_row)
        rebuilt += 1

    if dry_run:
        return f"  [DRY RUN] would rewrite ({fmt} → migrated): {rebuilt} rows" + (
            f", {skipped_malformed} malformed padded" if skipped_malformed else ""
        )

    # Backup + write
    bak_path = path + ".bak"
    try:
        shutil.copyfile(path, bak_path)
    except Exception as e:
        return f"  ERROR creating backup: {e}"

    try:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(new_rows)
    except Exception as e:
        # Try to restore from backup
        try:
            shutil.copyfile(bak_path, path)
        except Exception:
            pass
        return f"  ERROR writing (restored from backup): {e}"

    extra = f", {skipped_malformed} malformed padded" if skipped_malformed else ""
    return f"  migrated ({fmt} → new): {rebuilt} rows, backup at {os.path.basename(bak_path)}{extra}"


def main():
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    args = [a for a in args if a != "--dry-run"]

    if args:
        log_dir = args[0]
    else:
        log_dir = os.getenv("BOT_TRADE_LOG_DIR", "./outputs")

    log_dir = os.path.abspath(log_dir)
    if not os.path.isdir(log_dir):
        print(f"ERROR: directory not found: {log_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Trade log dir: {log_dir}")
    if dry_run:
        print("Mode: DRY RUN (no files will be modified)")
    print("-" * 60)

    # Find all bot_trades_*.csv (skip .bak files and any non-csv)
    candidates = [
        f for f in os.listdir(log_dir)
        if f.startswith("bot_trades_") and f.endswith(".csv")
    ]
    candidates.sort()

    if not candidates:
        print("No bot_trades_*.csv files found.")
        return

    for fname in candidates:
        fpath = os.path.join(log_dir, fname)
        print(fname)
        status = migrate_file(fpath, dry_run=dry_run)
        print(status)

    print("-" * 60)
    print("Done.")
    if not dry_run:
        print("Backups saved with .bak suffix. Delete them once you've verified the new files.")


if __name__ == "__main__":
    main()