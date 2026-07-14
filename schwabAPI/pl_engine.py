"""
pl_engine.py
------------
Household-level FIFO P&L.

All accounts collapse into ONE ledger per symbol, so share movements between
your own Schwab accounts (JOURNAL_IN / JOURNAL_OUT) cancel out and are ignored.
Every transaction row still carries the account it happened in.

Lot convention
--------------
    lot.qty   signed   >0 long, <0 short
    lot.cost  positive  cash PAID to open a long, or PREMIUM RECEIVED on a short
    cost_per_share = cost / abs(qty)

Realized on a close:
    long  : proceeds - basis_used
    short : premium_used - buyback_cost
Both fall out of  realized = trade_cash_share  +/-  lot_cost_share  below.

Handled:
  * long AND short positions (a sold-to-open covered call has qty < 0)
  * option expiry            -> closes the lot at 0, realizing the full premium
  * forward / reverse splits -> share count changes, TOTAL BASIS PRESERVED
  * symbol changes / mergers -> symbol_map in corporate_actions.json
  * ACATS transfer-in        -> Schwab reports cost 0. Uses manual_basis.csv if
                                you supply a price, otherwise flags the ticker.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
CORP_ACTIONS_FILE = HERE / "corporate_actions.json"
MANUAL_BASIS_FILE = HERE / "manual_basis.csv"

EPS = 1e-6


# ------------------------------------------------------------------ config


def load_symbol_map() -> Dict[str, str]:
    if not CORP_ACTIONS_FILE.exists():
        return {}
    cfg = json.loads(CORP_ACTIONS_FILE.read_text())
    return {k.upper(): v.upper() for k, v in (cfg.get("symbol_map") or {}).items()}


def load_manual_basis() -> Dict[str, float]:
    """manual_basis.csv:  symbol,cost_per_share  (for ACATS transfer-ins)."""
    if not MANUAL_BASIS_FILE.exists():
        return {}
    df = pd.read_csv(MANUAL_BASIS_FILE, comment="#")
    out: Dict[str, float] = {}
    for _, r in df.iterrows():
        try:
            px = float(r["cost_per_share"])
        except (TypeError, ValueError, KeyError):
            continue
        if px > 0:
            out[str(r["symbol"]).strip().upper()] = px
    return out


# ------------------------------------------------------------------ lots


class Lot:
    __slots__ = ("date", "qty", "cost", "unknown")

    def __init__(self, date, qty: float, cost: float, unknown: bool = False):
        self.date = date
        self.qty = qty            # signed
        self.cost = abs(cost)     # positive: paid (long) or received (short)
        self.unknown = unknown

    @property
    def cps(self) -> float:
        return self.cost / abs(self.qty) if abs(self.qty) > EPS else 0.0


def _pos(lots: List[Lot]) -> float:
    return sum(l.qty for l in lots)


def _basis(lots: List[Lot]) -> float:
    """Signed: +cost for longs, -credit for shorts."""
    return sum(l.cost * (1 if l.qty > 0 else -1) for l in lots)


def _prune(lots: List[Lot]) -> None:
    lots[:] = [l for l in lots if abs(l.qty) > EPS]


def _rescale(lots: List[Lot], new_total: float) -> bool:
    """Split: change share count, keep total cost. Returns False if impossible."""
    cur = _pos(lots)
    if abs(cur) < EPS:
        return False
    f = new_total / cur
    if f <= 0:
        return False
    for l in lots:
        l.qty *= f
    return True


def _remove_fifo(lots: List[Lot], qty: float) -> None:
    """Take |qty| shares out FIFO without realizing anything (transfer out)."""
    need = abs(qty)
    for l in lots:
        if need < EPS:
            break
        take = min(abs(l.qty), need)
        if take < EPS:
            continue
        l.cost -= l.cps * take
        l.qty -= take * (1 if l.qty > 0 else -1)
        need -= take
    _prune(lots)


def _close_fifo(lots: List[Lot], qty: float, cash: float) -> Tuple[float, float, bool]:
    """
    Close |qty| against opposite-side lots, FIFO.
    `cash` is the signed net cash of the whole trade (+ on a sell, - on a buy).
    Returns (realized, leftover_qty, touched_unknown_basis).
    """
    need = abs(qty)
    cash_ps = cash / abs(qty) if abs(qty) > EPS else 0.0   # signed cash per share
    realized = 0.0
    unknown = False

    for l in lots:
        if need < EPS:
            break
        avail = abs(l.qty)
        if avail < EPS:
            continue
        take = min(avail, need)
        if l.unknown:
            unknown = True

        lot_cost = l.cps * take
        trade_cash = cash_ps * take

        if l.qty > 0:
            # closing a long: trade_cash is the (positive) proceeds
            realized += trade_cash - lot_cost
            l.qty -= take
        else:
            # closing a short: lot_cost is the premium received,
            # trade_cash is the (negative) buy-back
            realized += lot_cost + trade_cash
            l.qty += take
        l.cost -= lot_cost
        need -= take

    _prune(lots)
    leftover = ((1 if qty > 0 else -1) * need) if need > EPS else 0.0
    return realized, leftover, unknown


# ------------------------------------------------------------------ engine


def _collapse_adjustments(g: pd.DataFrame) -> pd.DataFrame:
    """
    Net all zero-cash share adjustments (splits, re-registrations, renames)
    per DAY before applying them.

    Schwab books a ticker re-registration as  -50 / +50 / +50 / -50  in four
    separate transactions on one day.  Applied one at a time these momentarily
    drive the position to zero and destroy the cost basis.  Netted per day they
    correctly collapse to 0, and a real 20:1 reverse split (-9087 / +1135)
    correctly collapses to -7952.
    """
    adj = g["action"].isin(["SPLIT_ADD", "SPLIT_REMOVE"])
    if adj.sum() < 2:
        return g

    keep = g[~adj]
    merged = []
    for date, d in g[adj].groupby("date"):
        net = d["qty"].sum()
        row = d.iloc[0].copy()
        row["qty"] = net
        row["action"] = "SPLIT_ADD" if net >= 0 else "SPLIT_REMOVE"
        row["description"] = " | ".join(pd.unique(d["description"].astype(str)))[:120]
        merged.append(row)

    out = pd.concat([keep, pd.DataFrame(merged)], ignore_index=True)
    return out.sort_values(["date", "activity_id"])


def compute_pl(ledger: pd.DataFrame):
    """Returns (summary_df, transactions_df, open_lots_df, anomalies_df)."""
    symbol_map = load_symbol_map()
    manual_basis = load_manual_basis()

    led = ledger.copy()
    for col in ("symbol", "underlying"):
        led[col] = led[col].astype(str).str.upper().map(lambda s: symbol_map.get(s, s))

    anomalies: List[Dict] = []
    txn_rows: List[Dict] = []
    lot_rows: List[Dict] = []
    summary: List[Dict] = []

    # internal journals must net to zero at household level
    jr = led[led["action"].isin(["JOURNAL_IN", "JOURNAL_OUT"])]
    for sym, g in jr.groupby("symbol"):
        net = g["qty"].sum()
        if abs(net) > EPS:
            anomalies.append(dict(
                symbol=sym, date=str(g["date"].max().date()),
                issue="unbalanced_internal_journal",
                detail=f"net {net:+g} sh — did shares leave the household?",
                description=str(g["description"].iloc[0])[:60],
            ))

    div_by_sym = led[led["action"] == "DIVIDEND"].groupby("symbol")["cash"].sum().to_dict()
    interest_total = float(led[led["action"] == "INTEREST"]["cash"].sum())

    trades = led[~led["action"].isin(["JOURNAL_IN", "JOURNAL_OUT", "DIVIDEND", "INTEREST"])]

    for sym, g in trades.groupby("symbol", sort=False):
        g = _collapse_adjustments(g.sort_values(["date", "activity_id"]))
        lots: List[Lot] = []
        realized = fees_total = bought = sold = 0.0
        basis_flag = "OK"
        asset_type = g["asset_type"].iloc[0]
        underlying = g["underlying"].iloc[0]
        accounts = sorted({str(a) for a in g["account_number"].dropna().unique()})

        for _, r in g.iterrows():
            action = r["action"]
            qty = float(r["qty"])
            date = r["date"]
            gross = float(r["gross"])
            fees = float(r["fees"])
            desc = str(r["description"])
            row_realized = 0.0
            fees_total += fees

            if action in ("SPLIT_ADD", "SPLIT_REMOVE"):
                if abs(qty) < EPS:
                    continue                      # re-registration, nets to zero
                cur = _pos(lots)
                if abs(cur) < EPS:
                    if qty > 0:
                        lots.append(Lot(date, qty, 0.0, unknown=True))
                        basis_flag = "UNKNOWN_BASIS"
                        anomalies.append(dict(
                            symbol=sym, date=str(date.date()), issue="shares_from_nothing",
                            detail=f"+{qty:g} sh, no open lots, no cost reported",
                            description=desc[:60]))
                elif not _rescale(lots, cur + qty):
                    anomalies.append(dict(
                        symbol=sym, date=str(date.date()), issue="bad_split_adjustment",
                        detail=f"position {cur:g} -> {cur + qty:g}", description=desc[:60]))

            elif action == "XFER_IN_BASIS":
                # consolidation from another account; Schwab DID carry the basis
                lots.append(Lot(date, qty, gross))
                bought += qty

            elif action == "XFER_OUT_BASIS":
                _remove_fifo(lots, qty)
                sold += abs(qty)

            elif action == "TRANSFER_IN":
                px = manual_basis.get(sym)
                lots.append(Lot(date, qty, qty * px if px else 0.0, unknown=not px))
                bought += qty
                if not px:
                    basis_flag = "UNKNOWN_BASIS"
                    anomalies.append(dict(
                        symbol=sym, date=str(date.date()), issue="transfer_in_no_basis",
                        detail=f"{qty:g} sh in from another broker; Schwab cost = 0",
                        description=desc[:60]))

            elif action == "TRANSFER_OUT":
                _remove_fifo(lots, qty)
                sold += abs(qty)
                anomalies.append(dict(
                    symbol=sym, date=str(date.date()), issue="transfer_out",
                    detail=f"{abs(qty):g} sh left Schwab; no P&L realized",
                    description=desc[:60]))

            else:  # BUY / SELL / OPT_EXPIRE
                if action == "OPT_EXPIRE":
                    gross = fees = 0.0
                cash = gross - fees
                pos = _pos(lots)

                if abs(pos) < EPS or (pos > 0) == (qty > 0):
                    lots.append(Lot(date, qty, cash))
                else:
                    row_realized, leftover, hit_unknown = _close_fifo(lots, qty, cash)
                    realized += row_realized
                    basis_flag = "UNKNOWN_BASIS" if hit_unknown else basis_flag
                    if abs(leftover) > EPS:
                        cps = abs(cash / qty) if abs(qty) > EPS else 0.0
                        lots.append(Lot(date, leftover, abs(leftover) * cps))

                if qty > 0:
                    bought += qty
                else:
                    sold += abs(qty)

            txn_rows.append(dict(
                date=date.date(), account=r["account_number"], symbol=sym,
                asset_type=asset_type, action=action, qty=round(qty, 4),
                price=round(float(r["price"]), 4), gross=round(gross, 2),
                fees=round(fees, 2), realized_pl=round(row_realized, 2),
                position_after=round(_pos(lots), 4),
                cost_basis_after=round(_basis(lots), 2),
                description=desc, activity_id=r["activity_id"],
            ))

        open_qty = round(_pos(lots), 6)
        open_basis = round(_basis(lots), 2)

        for l in lots:
            lot_rows.append(dict(
                symbol=sym, open_date=l.date.date(), qty=round(l.qty, 4),
                cost_basis=round(l.cost * (1 if l.qty > 0 else -1), 2),
                cost_per_share=round(l.cps, 4), basis_known=not l.unknown,
            ))

        summary.append(dict(
            symbol=sym, asset_type=asset_type, underlying=underlying,
            accounts=",".join(accounts),
            first_txn=g["date"].min().date(), last_txn=g["date"].max().date(),
            bought_qty=round(bought, 4), sold_qty=round(sold, 4),
            open_qty=open_qty, open_cost_basis=open_basis,
            avg_cost=round(open_basis / open_qty, 4) if abs(open_qty) > EPS else 0.0,
            realized_pl=round(realized, 2),
            dividends=round(float(div_by_sym.get(sym, 0.0)), 2),
            fees=round(fees_total, 2),
            status="OPEN" if abs(open_qty) > EPS else "CLOSED",
            basis_flag=basis_flag,
        ))

    summary_df = pd.DataFrame(summary)
    if not summary_df.empty:
        summary_df.attrs["interest_total"] = interest_total

    # dividends on tickers we never traded (e.g. transferred-in, sold before)
    known = set(summary_df["symbol"]) if not summary_df.empty else set()
    orphan = [s for s in div_by_sym if s not in known and s != "CASH"]
    if orphan:
        extra = pd.DataFrame([
            dict(symbol=s, asset_type="EQUITY", underlying=s, accounts="",
                 first_txn=None, last_txn=None, bought_qty=0.0, sold_qty=0.0,
                 open_qty=0.0, open_cost_basis=0.0, avg_cost=0.0, realized_pl=0.0,
                 dividends=round(float(div_by_sym[s]), 2), fees=0.0,
                 status="CLOSED", basis_flag="DIVIDEND_ONLY")
            for s in orphan
        ])
        summary_df = pd.concat([summary_df, extra], ignore_index=True)

    return summary_df, pd.DataFrame(txn_rows), pd.DataFrame(lot_rows), pd.DataFrame(anomalies)