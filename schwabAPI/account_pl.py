#!/usr/bin/env python3
"""
account_pl2.py - Stock P&L Analyzer (reads Schwab transaction CSVs)
=====================================================================
For each symbol shows every BUY/SELL trade, open position, and
computes realized + unrealized P&L in $ and % return.

Usage:
    python account_pl2.py <file.csv>
    python account_pl2.py 0422CE...csv

    # Or auto-find all CSVs in current directory:
    python account_pl2.py
"""

from __future__ import annotations

import sys
import json
import glob
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────
# 1. PARSE  all trade rows from CSV
# ─────────────────────────────────────────────
def parse_trades(csv_path: str) -> pd.DataFrame:
    """
    Read CSV and extract every real (non-cash) trade item.
    Handles the multi-item transferItems structure correctly —
    iterates ALL items per transaction, not just item[0].
    """
    df = pd.read_csv(csv_path)
    trades = df[df["type"] == "TRADE"].dropna(subset=["raw_json"])

    rows = []
    for _, row in trades.iterrows():
        try:
            tx = json.loads(row["raw_json"])
        except Exception:
            continue

        date = str(row["date"])[:10]

        # ── transferItems (most Schwab trades use this) ──
        for item in tx.get("transferItems", []):
            inst = item.get("instrument", {})
            sym  = inst.get("symbol", "")
            if not sym or sym == "CURRENCY_USD":
                continue
            asset_type = inst.get("assetType", "")
            pos_effect  = item.get("positionEffect", "")
            qty   = float(item.get("amount", 0) or 0)
            price = float(item.get("price",  0) or 0)
            cost  = float(item.get("cost",   0) or 0)
            rows.append({
                "date":          date,
                "symbol":        sym,
                "asset_type":    asset_type,
                "positionEffect": pos_effect,
                "qty":           qty,
                "price":         price,
                "cost":          cost,
            })

        # ── transactionItem (older Schwab format, fallback) ──
        if "transactionItem" in tx and not tx.get("transferItems"):
            item = tx["transactionItem"]
            inst = item.get("instrument", {})
            sym  = inst.get("symbol", "")
            if sym and sym != "CURRENCY_USD":
                qty   = float(item.get("amount", 0) or 0)
                price = float(item.get("price",  0) or 0)
                cost  = float(item.get("cost",   0) or 0)
                pos_effect = item.get("instruction", "") or item.get("positionEffect", "")
                rows.append({
                    "date":          date,
                    "symbol":        sym,
                    "asset_type":    inst.get("assetType", ""),
                    "positionEffect": pos_effect,
                    "qty":           qty,
                    "price":         price,
                    "cost":          cost,
                })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    return out


# ─────────────────────────────────────────────
# 2. FETCH current price via yfinance
# ─────────────────────────────────────────────
def fetch_price(symbol: str) -> float | None:
    """Fetch latest price. Returns None on failure."""
    try:
        import yfinance as yf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fi = yf.Ticker(symbol).fast_info
            p = getattr(fi, "last_price", None)
            if p and float(p) > 0:
                return round(float(p), 4)
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
# 3. COMPUTE P&L per symbol  (FIFO lot matching)
# ─────────────────────────────────────────────
def compute_pl(trades: pd.DataFrame) -> list[dict]:
    """
    For every symbol:
      - List every BUY and SELL trade
      - Track open lots (FIFO)
      - Compute realized P&L on closed lots
      - Compute unrealized P&L on remaining open lots using live price
    """
    results = []

    for sym, grp in trades.groupby("symbol"):
        grp = grp.sort_values("date").reset_index(drop=True)

        trade_rows   = []   # individual trade log
        open_lots    = []   # list of (qty, cost_per_share) for FIFO
        realized_pl  = 0.0
        total_bought = 0.0  # total $ invested in BUYs
        total_sold   = 0.0  # total $ received from SELLs

        for _, t in grp.iterrows():
            pos   = t["positionEffect"]
            qty   = abs(float(t["qty"]))
            price = abs(float(t["price"]))
            cost  = abs(float(t["cost"]))
            date  = t["date"].strftime("%Y-%m-%d")

            if pos == "OPENING" or (pos == "" and float(t["qty"]) > 0):
                # ── BUY ──
                dollars = cost if cost > 0 else qty * price
                open_lots.append({"qty": qty, "cost_per_sh": dollars / qty if qty else 0})
                total_bought += dollars
                trade_rows.append({
                    "date":   date,
                    "action": "BUY",
                    "qty":    qty,
                    "price":  price,
                    "dollars": dollars,
                    "note":   "",
                })

            elif pos == "CLOSING" or (pos == "" and float(t["qty"]) < 0):
                # ── SELL ──
                proceeds = cost if cost > 0 else qty * price
                total_sold += proceeds

                # FIFO match
                remaining   = qty
                cost_basis  = 0.0
                while remaining > 0 and open_lots:
                    lot = open_lots[0]
                    if lot["qty"] <= remaining:
                        cost_basis += lot["qty"] * lot["cost_per_sh"]
                        remaining  -= lot["qty"]
                        open_lots.pop(0)
                    else:
                        cost_basis += remaining * lot["cost_per_sh"]
                        lot["qty"] -= remaining
                        remaining   = 0

                trade_pl = proceeds - cost_basis
                realized_pl += trade_pl
                trade_rows.append({
                    "date":    date,
                    "action":  "SELL",
                    "qty":     qty,
                    "price":   price,
                    "dollars": proceeds,
                    "note":    f"realized ${trade_pl:+.2f}",
                })

            else:
                # System transfer / unknown — treat as BUY at cost
                dollars = cost if cost > 0 else qty * price
                if qty > 0 and dollars > 0:
                    open_lots.append({"qty": qty, "cost_per_sh": dollars / qty})
                    total_bought += dollars
                    trade_rows.append({
                        "date":   date,
                        "action": "TRANSFER-IN",
                        "qty":    qty,
                        "price":  price,
                        "dollars": dollars,
                        "note":   "transferred in",
                    })

        # ── Open position summary ──
        open_qty       = sum(l["qty"] for l in open_lots)
        open_cost_basis = sum(l["qty"] * l["cost_per_sh"] for l in open_lots)
        avg_cost       = open_cost_basis / open_qty if open_qty > 0 else 0.0

        # Live price + unrealized P&L
        current_price  = fetch_price(sym) if open_qty > 0 else None
        market_value   = (current_price * open_qty) if current_price and open_qty > 0 else None
        unrealized_pl  = (market_value - open_cost_basis) if market_value is not None else None

        # Total return = realized + unrealized vs total invested
        total_invested = total_bought   # gross cash in
        total_return   = realized_pl + (unrealized_pl or 0.0)
        pct_return     = (total_return / total_invested * 100) if total_invested > 0 else None

        results.append({
            "symbol":          sym,
            "trade_rows":      trade_rows,
            "open_qty":        open_qty,
            "avg_cost":        avg_cost,
            "open_cost_basis": open_cost_basis,
            "current_price":   current_price,
            "market_value":    market_value,
            "realized_pl":     realized_pl,
            "unrealized_pl":   unrealized_pl,
            "total_invested":  total_invested,
            "total_return":    total_return,
            "pct_return":      pct_return,
        })

    return results


# ─────────────────────────────────────────────
# 4. PRINT report
# ─────────────────────────────────────────────
def _fmt(val, prefix="$", decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if prefix == "$":
        return f"${val:+,.{decimals}f}" if val != 0 else "$0.00"
    if prefix == "%":
        return f"{val:+.{decimals}f}%"
    return f"{val:.{decimals}f}"


def print_report(results: list[dict], account_label: str):
    print()
    print("=" * 90)
    print(f"  ACCOUNT P&L REPORT  —  {account_label}")
    print("=" * 90)

    total_realized   = 0.0
    total_unrealized = 0.0
    total_invested   = 0.0

    for r in sorted(results, key=lambda x: x["symbol"]):
        sym = r["symbol"]
        print()
        print(f"  {'─'*86}")
        print(f"  {sym}  |  open qty: {r['open_qty']:.4g}  |  avg cost: ${r['avg_cost']:.4f}")
        print(f"  {'─'*86}")
        print(f"  {'DATE':<12} {'ACTION':<12} {'QTY':>8} {'PRICE':>10} {'DOLLARS':>12}  NOTE")
        print(f"  {'-'*80}")

        for tr in r["trade_rows"]:
            print(
                f"  {tr['date']:<12} {tr['action']:<12} {tr['qty']:>8.4g}"
                f" {tr['price']:>10.4f} {tr['dollars']:>12.2f}  {tr['note']}"
            )

        print()
        # Open position
        if r["open_qty"] > 0:
            cp  = f"${r['current_price']:.4f}" if r["current_price"] else "N/A"
            mv  = f"${r['market_value']:,.2f}"  if r["market_value"] is not None else "N/A"
            ucb = f"${r['open_cost_basis']:,.2f}"
            print(f"  OPEN POSITION  qty={r['open_qty']:.4g}  cost_basis={ucb}  current_price={cp}  market_value={mv}")
        else:
            print(f"  POSITION FULLY CLOSED")

        # P&L summary
        print(f"  Realized P&L   : {_fmt(r['realized_pl'])}")
        if r["open_qty"] > 0:
            print(f"  Unrealized P&L : {_fmt(r['unrealized_pl'])}  (live price)")
        print(f"  Total Invested : ${r['total_invested']:,.2f}")
        print(f"  Total Return   : {_fmt(r['total_return'])}  ({_fmt(r['pct_return'], '%')})")

        total_realized   += r["realized_pl"]
        total_unrealized += r["unrealized_pl"] or 0.0
        total_invested   += r["total_invested"]

    # ── Grand total ──
    grand_total = total_realized + total_unrealized
    grand_pct   = grand_total / total_invested * 100 if total_invested > 0 else 0.0

    print()
    print("=" * 90)
    print(f"  SUMMARY  —  {account_label}")
    print("=" * 90)
    print(f"  Total Invested (gross buys)  : ${total_invested:>12,.2f}")
    print(f"  Total Realized P&L           : {_fmt(total_realized):>12}")
    print(f"  Total Unrealized P&L         : {_fmt(total_unrealized):>12}  (open positions at live price)")
    print(f"  ─────────────────────────────────────────")
    print(f"  Grand Total Return           : {_fmt(grand_total):>12}  ({grand_pct:+.2f}%)")
    print("=" * 90)
    print()


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────
def main():
    # Resolve which CSV(s) to process
    if len(sys.argv) >= 2:
        csv_files = [sys.argv[1]]
    else:
        # Auto-discover all account CSVs in current directory
        csv_files = sorted(
            f for f in glob.glob("*.csv")
            if not f.startswith("bot_") and not f.startswith("supertrend")
        )
        if not csv_files:
            print("No CSV files found. Pass a filename as argument:")
            print("  python account_pl2.py <account_hash>.csv")
            sys.exit(1)
        print(f"Auto-discovered {len(csv_files)} CSV file(s): {csv_files}")

    for csv_path in csv_files:
        if not Path(csv_path).exists():
            print(f"File not found: {csv_path}")
            continue

        label = Path(csv_path).stem[:16] + "..."
        print(f"\nProcessing: {csv_path}")

        trades = parse_trades(csv_path)
        if trades.empty:
            print(f"  No trade rows found in {csv_path}")
            continue

        print(f"  Parsed {len(trades)} trade items across {trades['symbol'].nunique()} symbols")
        print(f"  Fetching live prices for open positions...")

        results = compute_pl(trades)
        print_report(results, label)


if __name__ == "__main__":
    main()