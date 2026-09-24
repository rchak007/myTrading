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
MANUAL_ADJ_FILE = HERE / "manual_adjustments.csv"
MANUAL_TXN_FILE = HERE / "manual_transactions.csv"

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


def load_manual_adjustments() -> pd.DataFrame:
    """manual_adjustments.csv: symbol,date,qty,proceeds,note

    Shares that left the account without Schwab's transactions API ever
    reporting it. Measured 2026-09-24: the API returns nothing for account
    ...171 between 2023-10 and 2024-08 while the web UI shows trades all
    through it, and a full refetch from 2019 returned byte-identical data. The
    rows are simply not obtainable — not a chunk failure, not a watermark gap,
    not the `types` filter, all four excluded by measurement.

    `qty` is POSITIVE — how many shares to remove. `proceeds` is the total cash
    received; leave it BLANK when unknown and the shares exit at cost, which
    makes the share count and unrealized P&L correct while adding nothing false
    to realized P&L. Every applied row is reported in anomalies.csv, so an
    approximate figure can never quietly pass for a measured one.
    """
    cols = ["symbol", "date", "qty", "proceeds", "note"]
    if not MANUAL_ADJ_FILE.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(MANUAL_ADJ_FILE, comment="#")
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["proceeds"] = pd.to_numeric(df["proceeds"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df[df["qty"].notna() & (df["qty"] > 0)][cols]


def load_manual_transactions() -> pd.DataFrame:
    """manual_transactions.csv: symbol,date,action,qty,amount,account,note

    A COMPLETE, hand-entered history for one ticker, taken from Schwab's own
    account pages. Any symbol appearing here has ALL its API rows replaced by
    these — the file is authoritative, not additive, so there is no way to
    double-count.

    This is stronger than manual_adjustments.csv and is the right tool when the
    API is missing BUYS as well as sells. SOFI is the example: the API gap of
    2023-10..2024-08 swallowed three sells (70, 350, 367) and a buy (142), so
    netting the share difference would have left realized P&L wrong even once
    the quantity was right.

    qty is always POSITIVE; `action` carries the direction. `amount` is the
    signed cash exactly as Schwab shows it — negative on a buy, positive on a
    sell — so it can be copied straight off the screen without arithmetic.
    Internal transfer pairs that net to zero within one account are omitted.
    """
    cols = ["symbol", "date", "action", "qty", "amount", "account", "note"]
    if not MANUAL_TXN_FILE.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(MANUAL_TXN_FILE, comment="#")
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["action"] = df["action"].astype(str).str.strip().str.upper()
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").abs()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    ok = df["qty"].notna() & df["date"].notna() & df["action"].isin(["BUY", "SELL"])
    return df[ok][cols]


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

    # NOTE: the merger pair is netted by _collapse_adjustments() further down,
    # which nets same-day SPLIT_ADD/SPLIT_REMOVE per symbol and was already
    # doing this correctly. The ASST bug was purely the missing symbol_map
    # entry: until 862945102 folded onto ASST the two legs were different
    # symbols, so they could never net. One mechanism, not two.

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

    # Hand-entered complete histories REPLACE the API rows for those symbols.
    # Authoritative, not additive — the API rows for an overridden symbol are
    # dropped entirely, so nothing can be counted twice.
    man = load_manual_transactions()
    if not man.empty:
        overridden = sorted(man["symbol"].unique())
        trades = trades[~trades["symbol"].isin(overridden)]
        rows = []
        for _, m in man.iterrows():
            qty = float(m["qty"]) * (1 if m["action"] == "BUY" else -1)
            cash = float(m["amount"]) if pd.notna(m["amount"]) else 0.0
            rows.append(dict(
                date=m["date"], account_number=str(m.get("account") or "MANUAL"),
                account_hash="MANUAL",
                activity_id=f"man:{m['symbol']}:{m['date']:%Y%m%d}:{m['qty']:g}",
                txn_type="MANUAL", action=m["action"],
                symbol=m["symbol"], asset_type="EQUITY", underlying=m["symbol"],
                qty=qty, price=0.0, gross=cash, fees=0.0, cash=cash,
                description=str(m.get("note") or "hand-entered from Schwab"),
                basis_known=True))
        trades = pd.concat([trades, pd.DataFrame(rows)], ignore_index=True)
        logger.info("Manual histories override %d symbol(s): %s",
                    len(overridden), ", ".join(overridden))
        for sym in overridden:
            anomalies.append(dict(
                symbol=sym, date="", issue="manual_history",
                detail="API rows replaced by hand-entered history",
                description=""))

    # Shares Schwab's API never reported leaving. Appended as synthetic rows so
    # they flow through the same FIFO path as a real sell — no special case in
    # the loop, and they appear in ticker_txns.csv where they can be seen.
    adj = load_manual_adjustments()
    if not adj.empty and not trades.empty:
        extra = []
        for _, a in adj.iterrows():
            sym = a["symbol"]
            g = trades[trades["symbol"] == sym]
            if g.empty:
                anomalies.append(dict(
                    symbol=sym, date="", issue="manual_adjustment_unused",
                    detail=f"{a['qty']:g} sh listed but no transactions for {sym}",
                    description=str(a.get("note") or "")[:60]))
                continue
            # Undated rows land after everything else for that symbol, which is
            # the only ordering that cannot consume lots opened later.
            when = a["date"] if pd.notna(a["date"]) else g["date"].max()
            proceeds = 0.0 if pd.isna(a["proceeds"]) else float(a["proceeds"])
            extra.append(dict(
                date=when, account_number="MANUAL", account_hash="MANUAL",
                activity_id=f"manual:{sym}:{a['qty']:g}",
                txn_type="MANUAL", action="MANUAL_EXIT",
                symbol=sym, asset_type=g["asset_type"].iloc[0],
                underlying=g["underlying"].iloc[0],
                qty=-abs(float(a["qty"])), price=0.0,
                gross=proceeds, fees=0.0, cash=proceeds,
                description=str(a.get("note") or "manual adjustment"),
                basis_known=pd.notna(a["proceeds"]),
            ))
        if extra:
            trades = pd.concat([trades, pd.DataFrame(extra)], ignore_index=True)
            logger.info("Applied %d manual adjustment(s)", len(extra))

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

            elif action == "MANUAL_EXIT":
                # Known to have left, proceeds possibly unknown. With a figure
                # it closes like a sell; without one it exits AT COST, so the
                # share count and unrealized P&L become right while realized
                # P&L gains nothing invented. Always flagged.
                if abs(gross) > EPS:
                    row_realized, leftover, unk = _close_fifo(lots, qty, gross)
                    realized += row_realized
                    if unk:
                        basis_flag = "UNKNOWN_BASIS"
                else:
                    _remove_fifo(lots, qty)
                sold += abs(qty)
                anomalies.append(dict(
                    symbol=sym, date=str(date.date()), issue="manual_adjustment",
                    detail=(f"{abs(qty):g} sh removed by hand; "
                            + (f"proceeds ${gross:,.2f}" if abs(gross) > EPS
                               else "NO proceeds given — exited at cost, "
                                    "realized P&L understated")),
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