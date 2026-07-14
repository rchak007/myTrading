"""
txn_parser.py
-------------
Turns the raw Schwab transaction JSON into a flat, normalized ledger.

One row per (transaction, security leg). Cash-only legs (CURRENCY_USD,
MMDA1 sweeps, Cash Alternatives) are dropped except that their fees are
folded back into the trade they belong to.

Ledger columns
--------------
date, account_hash, account_number, activity_id, txn_type, action,
symbol, asset_type, underlying, qty, price, gross, fees, cash,
description, basis_known

action is one of:
    BUY / SELL                     real trades (positionEffect OPENING/CLOSING)
    SPLIT_ADD / SPLIT_REMOVE       cost-free share count changes
    TRANSFER_IN / TRANSFER_OUT     ACATS in/out from another broker (NO BASIS)
    JOURNAL_IN / JOURNAL_OUT       share movement between YOUR OWN accounts
    OPT_EXPIRE                     option removed at expiration
    DIVIDEND / INTEREST            cash, attributed to a symbol where possible
"""

import json
import logging
import re
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

SECURITY_ASSETS = {"EQUITY", "COLLECTIVE_INVESTMENT", "OPTION", "FIXED_INCOME", "INDEX", "MUTUAL_FUND"}
CASH_ASSETS = {"CURRENCY", "CASH_EQUIVALENT"}

# Instruments that are cash in disguise -- never part of an equity ledger.
CASH_SYMBOLS = {"CURRENCY_USD", "MMDA1", "MMDA2", "SWVXX", "SNSXX"}

# Description fingerprints for cost-free (zero-cash) security movements.
RE_SPLIT = re.compile(r"SPLIT", re.I)
RE_XFER_EXT = re.compile(r"Transfer of Security or Option (In|Out)", re.I)
RE_XFER_INT = re.compile(r"Internal Transfer between accounts", re.I)
RE_EXPIRE = re.compile(r"Removed due to Expiration|Expiration", re.I)
RE_MERGER = re.compile(r"Mandatory|Merger|Exchange|Redemption of|Reorganization", re.I)
# Schwab books account consolidations as TRADE with netAmount==0 and the COST
# BASIS (positive) sitting in `cost`. It never reports the outgoing leg.
RE_SYS_XFER = re.compile(r"System transfer", re.I)

# Dividend descriptions look like "Qualified Dividend~MKSI"
RE_DIV_SYMBOL = re.compile(r"~\s*([A-Z0-9./-]+)\s*$")

# Option symbols: "CRDO  260116C00260000"
RE_OPTION = re.compile(r"^([A-Z.]+)\s+\d{6}[CP]\d{8}$")


def _fees_of(items: List[Dict]) -> float:
    """Total fees on a transaction. Schwab stores them as negative `cost`."""
    tot = 0.0
    for it in items:
        if it.get("feeType"):
            tot += abs(float(it.get("cost") or 0.0))
    return round(tot, 4)


def _underlying(symbol: str) -> str:
    m = RE_OPTION.match(symbol or "")
    return m.group(1) if m else (symbol or "")


def _classify_zero_cash(desc: str, qty: float) -> str:
    d = desc or ""
    if RE_EXPIRE.search(d):
        return "OPT_EXPIRE"
    if RE_XFER_INT.search(d):
        return "JOURNAL_IN" if qty > 0 else "JOURNAL_OUT"
    if RE_XFER_EXT.search(d):
        return "TRANSFER_IN" if qty > 0 else "TRANSFER_OUT"
    if RE_SPLIT.search(d) or RE_MERGER.search(d):
        return "SPLIT_ADD" if qty > 0 else "SPLIT_REMOVE"
    # Unlabeled zero-cash movement (ticker re-registrations, name changes).
    # Treated as a split-style adjustment; surfaced in the anomaly report.
    return "SPLIT_ADD" if qty > 0 else "SPLIT_REMOVE"


def parse_ledger(cache_df: pd.DataFrame, hash_to_acct: Dict[str, str] | None = None) -> pd.DataFrame:
    hash_to_acct = hash_to_acct or {}
    rows: List[Dict] = []

    for _, r in cache_df.iterrows():
        raw = r.get("raw_json")
        if not isinstance(raw, str):
            continue
        try:
            t = json.loads(raw)
        except Exception:
            continue

        items = t.get("transferItems") or []
        txn_type = t.get("type")
        desc = t.get("description") or ""
        acct_hash = r.get("account_hash")
        acct_num = t.get("accountNumber") or hash_to_acct.get(acct_hash)
        date = r.get("date")
        aid = t.get("activityId")
        net_amount = float(t.get("netAmount") or 0.0)
        fees = _fees_of(items)

        sec_items = [
            it
            for it in items
            if (it.get("instrument") or {}).get("assetType") in SECURITY_ASSETS
            and (it.get("instrument") or {}).get("symbol") not in CASH_SYMBOLS
        ]

        # ---------- dividends / interest (no security leg; symbol is in the text)
        if txn_type == "DIVIDEND_OR_INTEREST" and not sec_items:
            m = RE_DIV_SYMBOL.search(desc)
            sym = m.group(1) if m else "CASH"
            amt = sum(
                float(it.get("amount") or 0.0)
                for it in items
                if (it.get("instrument") or {}).get("assetType") in CASH_ASSETS
            )
            if not amt:
                amt = net_amount
            rows.append(
                dict(
                    date=date, account_hash=acct_hash, account_number=acct_num, activity_id=aid,
                    txn_type=txn_type,
                    action="DIVIDEND" if m else "INTEREST",
                    symbol=sym, asset_type="CASH", underlying=sym,
                    qty=0.0, price=0.0, gross=0.0, fees=0.0, cash=round(amt, 4),
                    description=desc, basis_known=True,
                )
            )
            continue

        # ---------- security legs
        n_sec = len(sec_items) or 1
        for it in sec_items:
            inst = it.get("instrument") or {}
            sym = inst.get("symbol")
            if not sym:
                continue
            atype = inst.get("assetType")
            qty = float(it.get("amount") or 0.0)
            gross = float(it.get("cost") or 0.0)   # signed cash: -ve on a buy
            price = float(it.get("price") or 0.0)

            if txn_type == "TRADE" and RE_SYS_XFER.search(desc) and abs(net_amount) < 0.005:
                # account consolidation: no cash moved, `cost` is the carried basis
                action = "XFER_IN_BASIS" if qty > 0 else "XFER_OUT_BASIS"
                gross = -abs(gross) if qty > 0 else abs(gross)
            elif txn_type == "TRADE":
                # sign of qty is authoritative; positionEffect=OPENING on a
                # sold-to-open call still has qty < 0
                action = "BUY" if qty > 0 else "SELL"
            else:
                action = _classify_zero_cash(desc, qty)

            rows.append(
                dict(
                    date=date, account_hash=acct_hash, account_number=acct_num, activity_id=aid,
                    txn_type=txn_type, action=action,
                    symbol=sym, asset_type=atype, underlying=_underlying(sym),
                    qty=qty, price=price, gross=round(gross, 4),
                    fees=round(fees / n_sec, 4),
                    cash=round(net_amount / n_sec, 4),
                    description=desc,
                    basis_known=(action in ("BUY", "SELL", "XFER_IN_BASIS", "XFER_OUT_BASIS")),
                )
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["date", "activity_id"]).reset_index(drop=True)
    return df