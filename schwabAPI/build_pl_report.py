"""
build_pl_report.py
------------------
End-to-end:  Schwab -> transaction cache -> ledger -> FIFO P&L -> CSV/XLSX.

Usage
-----
    python build_pl_report.py                # sync from Schwab, then report
    python build_pl_report.py --offline      # rebuild report from cache only
    python build_pl_report.py --no-prices    # skip the live quote call

Writes to  ~/github/jobMyTrading/outputs/portfolio/  (override with
PORTFOLIO_OUT_DIR) so the existing Streamlit dashboard picks it up:

    ticker_pl_summary.csv   one row per ticker ever traded (open AND closed)
    ticker_txns.csv         every transaction, with running position + realized
    open_lots.csv           the FIFO lots still open
    anomalies.csv           anything the engine could not explain -- READ THIS
    portfolio_pl.xlsx       all of the above as tabs

This script does NOT git push. gitpush.py remains the sole git writer.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from schwab_client import REPORT_DIR, make_client, stray_token_check
import txn_cache
import txn_parser
import pl_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pl_report")

# Instruments that are cash, not investments
EXCLUDE = {"CURRENCY_USD", "MMDA1", "MMDA2", "CASH"}


def fetch_market_prices(client) -> tuple[dict, dict]:
    """Returns (price_by_symbol, qty_by_symbol) from the live positions API."""
    try:
        resp = client.account_details_all(fields="positions")
    except TypeError:
        resp = client.account_details_all(fields=["positions"])

    prices, qtys = {}, {}
    for acct in resp.json():
        sa = acct.get("securitiesAccount", {}) or {}
        for p in sa.get("positions", []) or []:
            inst = p.get("instrument", {}) or {}
            sym = inst.get("symbol")
            if not sym:
                continue
            q = float(p.get("longQuantity") or 0.0) - float(p.get("shortQuantity") or 0.0)
            mv = float(p.get("marketValue") or 0.0)
            qtys[sym] = qtys.get(sym, 0.0) + q
            if abs(q) > 1e-6:
                prices[sym] = mv / q
    return prices, qtys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use the on-disk cache, do not call Schwab")
    ap.add_argument("--no-prices", action="store_true", help="skip live quotes (unrealized will be blank)")
    ap.add_argument("--out", default=None, help="override output directory")
    args = ap.parse_args()

    stray_token_check()
    out_dir = Path(args.out) if args.out else REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    client = None
    hash_to_acct: dict = {}

    if args.offline:
        logger.info("OFFLINE: rebuilding from cache only")
        cache = txn_cache.load_all_cached()
    else:
        client = make_client()
        accounts = txn_cache.get_linked_accounts(client)
        hash_to_acct = {a["hash"]: a["account_number"] for a in accounts}
        logger.info("Syncing %d accounts ...", len(accounts))
        cache = txn_cache.sync_all(client)

    if cache.empty:
        logger.error("No transactions. Nothing to do.")
        return 1
    logger.info("Cached transactions: %d", len(cache))

    # ---------------------------------------------------------------- ledger
    ledger = txn_parser.parse_ledger(cache, hash_to_acct)
    ledger = ledger[~ledger["symbol"].isin(EXCLUDE)]
    logger.info("Ledger rows (security + income legs): %d", len(ledger))

    summary, txns, lots, anomalies = pl_engine.compute_pl(ledger)

    # ---------------------------------------------------------------- prices
    have_prices = (not args.no_prices) and client is not None
    if have_prices:
        prices, live_qty = fetch_market_prices(client)
        summary["market_price"] = summary["symbol"].map(prices).fillna(0.0).round(4)
        summary["schwab_qty"] = summary["symbol"].map(live_qty).fillna(0.0).round(4)
        summary["market_value"] = (summary["open_qty"] * summary["market_price"]).round(2)
        summary["unrealized_pl"] = (summary["market_value"] - summary["open_cost_basis"]).round(2)
        summary.loc[summary["open_qty"].abs() < 1e-6, "unrealized_pl"] = 0.0
        # does our FIFO share count match what Schwab says you actually hold?
        summary["qty_mismatch"] = (summary["open_qty"] - summary["schwab_qty"]).round(4)
    else:
        # no live quotes -> leave these BLANK rather than pretending they are zero
        for c in ("market_price", "schwab_qty", "market_value", "unrealized_pl", "qty_mismatch"):
            summary[c] = pd.NA

    unreal = summary["unrealized_pl"].fillna(0.0) if have_prices else 0.0
    summary["total_pl"] = (summary["realized_pl"] + unreal + summary["dividends"]).round(2)

    invested = summary["open_cost_basis"].abs().where(lambda s: s > 0)
    summary["total_pl_pct"] = (summary["total_pl"] / invested * 100).round(2)

    cols = [
        "symbol", "asset_type", "status", "accounts", "first_txn", "last_txn",
        "bought_qty", "sold_qty", "open_qty", "schwab_qty", "qty_mismatch",
        "avg_cost", "open_cost_basis", "market_price", "market_value",
        "unrealized_pl", "realized_pl", "dividends", "fees",
        "total_pl", "total_pl_pct", "basis_flag",
    ]
    summary = summary[[c for c in cols if c in summary.columns]]
    summary = summary.sort_values("total_pl", ascending=False, na_position="last")

    # ---------------------------------------------------------------- write
    summary.to_csv(out_dir / "ticker_pl_summary.csv", index=False)
    txns.sort_values(["symbol", "date"]).to_csv(out_dir / "ticker_txns.csv", index=False)
    lots.to_csv(out_dir / "open_lots.csv", index=False)
    anomalies.to_csv(out_dir / "anomalies.csv", index=False)

    try:
        with pd.ExcelWriter(out_dir / "portfolio_pl.xlsx", engine="openpyxl") as xl:
            summary.to_excel(xl, sheet_name="Summary", index=False)
            txns.sort_values(["symbol", "date"]).to_excel(xl, sheet_name="Transactions", index=False)
            lots.to_excel(xl, sheet_name="OpenLots", index=False)
            anomalies.to_excel(xl, sheet_name="Anomalies", index=False)
    except Exception as e:  # openpyxl not installed on the Pi? CSVs still land.
        logger.warning("xlsx not written: %r", e)

    # ---------------------------------------------------------------- print
    eq = summary[summary["asset_type"].isin(["EQUITY", "COLLECTIVE_INVESTMENT"])]
    opn = eq[eq["status"] == "OPEN"]

    print("\n=== TOP 15 by total P&L ===")
    print(summary.head(15).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n=== TOTALS (equity + ETF) ===")
    print(f"  realized      {eq['realized_pl'].sum():>15,.2f}")
    if have_prices:
        print(f"  unrealized    {eq['unrealized_pl'].sum():>15,.2f}")
    print(f"  dividends     {eq['dividends'].sum():>15,.2f}")
    print(f"  fees          {eq['fees'].sum():>15,.2f}")
    print(f"  TOTAL P&L     {eq['total_pl'].sum():>15,.2f}")
    if have_prices:
        print(f"  open mkt val  {opn['market_value'].sum():>15,.2f}")

    bad = summary[summary["basis_flag"] == "UNKNOWN_BASIS"]
    if not bad.empty:
        print(f"\n!! {len(bad)} tickers have NO cost basis from Schwab "
              f"(transferred in from another broker).")
        print("   Their P&L is understated. Fill in manual_basis.csv:")
        print("   " + ", ".join(bad["symbol"].tolist()))

    mism = summary[summary["qty_mismatch"].abs() > 0.01] if have_prices else summary.iloc[0:0]
    if not mism.empty:
        print(f"\n!! {len(mism)} tickers where our FIFO share count != Schwab's. "
              f"See anomalies.csv:")
        print(mism[["symbol", "open_qty", "schwab_qty", "qty_mismatch"]].to_string(index=False))

    print(f"\nWrote 4 CSVs + portfolio_pl.xlsx to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())