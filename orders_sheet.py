#!/usr/bin/env python3
"""
orders_sheet.py
===============
Writes the Pi-1-owned tabs of myTrading-ORDERS-pi1: Positions, Cash, and the
status header block on Orders.

Phase 1 of Documentation/ordersSheetDesign-9-18-26.md. It places no orders and
touches no human-owned column. The worst it can do is write a wrong number into
a cell.

DESIGN — same injection contract as stocks_cash.py / stocks_orders.py. This
module owns *only* sheet composition. Everything shared is injected:

    Schwab client  -> jobStocksSignals.get_schwab_client()
    open orders    -> stocks_orders.build_orders_table()
    cash           -> stocks_cash.build_cash_table()
    prices         -> the merged signals frame
    log()          -> jobStocksSignals.log

The one thing it resolves itself is the spreadsheet, from GSHEET_ORDERS_ID and
REMOTE_OPS_CREDS — same as remote_ops.py, and for the same reason: the caller
has no business knowing about Sheets.

WHY IT RE-FETCHES POSITIONS
    jobStocksSignals.fetch_schwab_holdings() groups by Ticker and drops the
    account, and never captures averagePrice. Both are required here — coverage
    is meaningless without the account, see below — so this module extracts its
    own rows from the same fetch_positions() payload. One extra call per cycle.

WHY COVERAGE IS PER ACCOUNT
    A stop resting in ...431 protects only the shares in ...431. Rolling
    Has_Stop up across accounts would report a position as protected while half
    of it is naked — not a less precise answer, a false one. So the flags are
    computed per (ticker, account) and left blank on TOTAL rows.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

POSITIONS_COLS = [
    "Ticker", "Acct", "Qty", "Avg_Cost", "Market_Value", "Unrealized_PL",
    "Has_Stop", "Has_Trim", "Has_Dip", "Has_Breakout", "Seed_Reserved", "Updated",
]
CASH_COLS = [
    "Acct", "Nickname", "Cash", "Cash_In_Open_Orders", "Cash_After_Open_Orders",
    "Seed_Reserved", "Free_To_Deploy", "Updated",
]

TOTAL = "TOTAL"
YES, NO, BLANK = "Y", "N", ""


# ───────────────────────────────────────────────────────────── helpers
def _num(v, default=None):
    try:
        if v is None or v == "" or (isinstance(v, float) and pd.isna(v)):
            return default
        return float(str(v).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return default


def acct_key(a) -> str:
    """Last 3 digits, matching cash_reserve.acct_key and stocks_cash._mask."""
    s = str(a or "").strip().lstrip(".")
    return s[-3:] if len(s) >= 3 else s


def _now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


# ───────────────────────────────────────────────────── positions fetch
def fetch_positions_detailed(client_wrapper, log=print) -> pd.DataFrame:
    """
    One row per (ticker, account), with cost basis — what the aggregated
    fetch_schwab_holdings() throws away.
    """
    cols = ["Ticker", "Acct", "Qty", "Avg_Cost", "Market_Value", "Unrealized_PL"]
    try:
        data = client_wrapper.fetch_positions()
    except Exception as e:
        log(f"⚠️  positions fetch failed: {e}")
        return pd.DataFrame(columns=cols)

    rows = []
    for acct in data or []:
        sa = acct.get("securitiesAccount", {}) or {}
        acct_no = acct_key(sa.get("accountNumber"))
        for pos in sa.get("positions", []) or []:
            inst = pos.get("instrument", {}) or {}
            sym = inst.get("symbol")
            if not sym or inst.get("assetType") not in ("EQUITY", "COLLECTIVE_INVESTMENT", "ETF"):
                continue
            qty = (_num(pos.get("longQuantity"), 0.0) or 0.0) - (_num(pos.get("shortQuantity"), 0.0) or 0.0)
            if abs(qty) < 1e-9:
                continue
            mv = _num(pos.get("marketValue"), 0.0) or 0.0
            avg = _num(pos.get("averagePrice"), 0.0) or 0.0
            upl = _num(pos.get("longOpenProfitLoss"))
            if upl is None:
                upl = mv - (avg * qty)
            rows.append({"Ticker": sym.upper(), "Acct": acct_no, "Qty": round(qty, 4),
                         "Avg_Cost": round(avg, 4), "Market_Value": round(mv, 2),
                         "Unrealized_PL": round(upl, 2)})

    df = pd.DataFrame(rows, columns=cols)
    log(f"Positions: {len(df)} (ticker, account) rows across "
        f"{df['Acct'].nunique() if not df.empty else 0} accounts")
    return df


# ────────────────────────────────────────────────────── coverage flags
def coverage_for(ticker: str, acct: str, price: float | None,
                 orders_df: pd.DataFrame | None) -> dict:
    """
    Which of the four protective/entry orders exist for this pair.

    Classification mirrors sell_guard.py: a SELL below the current price is
    protection, a SELL above it is a profit target. Reads live Schwab orders,
    not the Orders tab — the question is "is this position protected", not
    "did the sheet arrange it", and orders get placed by hand too.
    """
    out = {"Has_Stop": NO, "Has_Trim": NO, "Has_Dip": NO, "Has_Breakout": NO}
    if orders_df is None or getattr(orders_df, "empty", True) or not price:
        return {k: BLANK for k in out} if not price else out

    m = (orders_df["Ticker"].astype(str).str.upper() == ticker)
    if "Account" in orders_df.columns:
        m &= orders_df["Account"].map(acct_key) == acct
    legs = orders_df[m]
    if legs.empty:
        return out

    for _, o in legs.iterrows():
        side = str(o.get("Side", "")).upper()
        stop = _num(o.get("Stop_Price"))
        limit = _num(o.get("Limit_Price"))
        px = stop if stop else limit
        if not px:
            continue
        is_sell = side.startswith("SELL")
        if is_sell and px < price:
            out["Has_Stop"] = YES          # downside protection
        elif is_sell and px >= price:
            out["Has_Trim"] = YES          # profit target
        elif not is_sell and px <= price:
            out["Has_Dip"] = YES           # buy the pullback
        elif not is_sell and px > price:
            out["Has_Breakout"] = YES      # buy strength
    return out


# ──────────────────────────────────────────────────── positions table
def build_positions_table(positions_df, signals_df=None, orders_df=None,
                          reserves=None, log=print) -> pd.DataFrame:
    """
    Per (ticker, account), plus a TOTAL row for any ticker held in more than
    one account. Avg_Cost on a TOTAL row is QUANTITY-WEIGHTED — a mean of the
    per-account averages is wrong whenever the accounts hold different sizes,
    and wrong in a way that still looks plausible.
    """
    if positions_df is None or positions_df.empty:
        return pd.DataFrame(columns=POSITIONS_COLS)

    prices = {}
    if signals_df is not None and not getattr(signals_df, "empty", True):
        pcol = next((c for c in ("Current Price", "Last Close", "Price")
                     if c in signals_df.columns), None)
        if pcol:
            prices = {str(t).upper(): _num(p)
                      for t, p in zip(signals_df["Ticker"], signals_df[pcol])}

    reserves = reserves or {}
    stamp = _now()
    out = []

    for ticker in sorted(positions_df["Ticker"].unique()):
        grp = positions_df[positions_df["Ticker"] == ticker].sort_values("Acct")
        px = prices.get(ticker)
        for _, r in grp.iterrows():
            flags = coverage_for(ticker, r["Acct"], px, orders_df)
            out.append({
                "Ticker": ticker, "Acct": r["Acct"], "Qty": r["Qty"],
                "Avg_Cost": r["Avg_Cost"], "Market_Value": r["Market_Value"],
                "Unrealized_PL": r["Unrealized_PL"], **flags,
                "Seed_Reserved": reserves.get((r["Acct"], ticker), ""),
                "Updated": stamp,
            })

        if len(grp) > 1:
            qty = grp["Qty"].sum()
            wavg = (grp["Qty"] * grp["Avg_Cost"]).sum() / qty if qty else 0.0
            seeds = [_num(reserves.get((a, ticker)), 0.0) or 0.0 for a in grp["Acct"]]
            out.append({
                "Ticker": ticker, "Acct": TOTAL, "Qty": round(qty, 4),
                "Avg_Cost": round(wavg, 4),
                "Market_Value": round(grp["Market_Value"].sum(), 2),
                "Unrealized_PL": round(grp["Unrealized_PL"].sum(), 2),
                # Blank, not N: coverage is per account and has no honest
                # aggregate value.
                "Has_Stop": BLANK, "Has_Trim": BLANK,
                "Has_Dip": BLANK, "Has_Breakout": BLANK,
                "Seed_Reserved": round(sum(seeds), 2) if any(seeds) else "",
                "Updated": stamp,
            })

    df = pd.DataFrame(out, columns=POSITIONS_COLS)
    naked = df[(df["Acct"] != TOTAL) & (df["Has_Stop"] == NO)]["Ticker"].tolist()
    if naked:
        log(f"⚠️  {len(naked)} holding(s) with no protective stop: {', '.join(naked[:12])}")
    return df


# ───────────────────────────────────────────────────────── cash table
def build_cash_rows(cash_df, reserves=None, log=print) -> pd.DataFrame:
    """Free_To_Deploy = Cash_After_Open_Orders − seed reserved for that account."""
    if cash_df is None or getattr(cash_df, "empty", True):
        return pd.DataFrame(columns=CASH_COLS)

    per_acct = {}
    for (a, _t), amt in (reserves or {}).items():
        per_acct[a] = per_acct.get(a, 0.0) + (_num(amt, 0.0) or 0.0)

    stamp, out = _now(), []
    for _, r in cash_df.iterrows():
        a = acct_key(r.get("Account"))
        if not a or a.upper() == TOTAL:
            continue
        after = _num(r.get("Cash_After_Open_Orders"), 0.0) or 0.0
        seed = round(per_acct.get(a, 0.0), 2)
        out.append({
            "Acct": a, "Nickname": r.get("Nickname", ""),
            "Cash": _num(r.get("Cash"), 0.0),
            "Cash_In_Open_Orders": _num(r.get("Cash_In_Open_Orders"), 0.0),
            "Cash_After_Open_Orders": after,
            "Seed_Reserved": seed,
            "Free_To_Deploy": round(after - seed, 2),
            "Updated": stamp,
        })
    return pd.DataFrame(out, columns=CASH_COLS)


# ──────────────────────────────────────────────────────── sheet write
def _open_book():
    from google.oauth2.service_account import Credentials
    import gspread

    sid = os.getenv("GSHEET_ORDERS_ID", "")
    creds = os.getenv("REMOTE_OPS_CREDS",
                      os.getenv("GSHEET_CREDS", "/etc/myTrading/gsheets-ops.json"))
    if not sid:
        raise RuntimeError("GSHEET_ORDERS_ID is not set")
    if not Path(creds).exists():
        raise RuntimeError(f"service-account key not found at {creds}")
    return gspread.authorize(
        Credentials.from_service_account_file(creds, scopes=SCOPES)
    ).open_by_key(sid)


def _put(ws, df, cols, log):
    ws.batch_clear([f"A2:{chr(64 + len(cols))}1000"])
    if df is None or df.empty:
        return 0
    body = [[("" if pd.isna(v) else v) for v in row] for row in df[cols].values.tolist()]
    ws.update(values=body, range_name=f"A2:{chr(64 + len(cols))}{len(body) + 1}")
    return len(body)


def write_orders_sheet(*, client_wrapper, signals_df=None, orders_df=None,
                       cash_df=None, reserves=None, token_status=None,
                       log=print) -> None:
    """
    Refresh Positions, Cash and the header block. Never touches columns A-K of
    Orders — those are the human's.
    """
    book = _open_book()

    pos_raw = fetch_positions_detailed(client_wrapper, log=log)
    positions = build_positions_table(pos_raw, signals_df, orders_df, reserves, log=log)
    cash = build_cash_rows(cash_df, reserves, log=log)

    n_pos = _put(book.worksheet("Positions"), positions, POSITIONS_COLS, log)
    n_cash = _put(book.worksheet("Cash"), cash, CASH_COLS, log)

    # ---- header block. LAST POLL is the health check: if this stops moving,
    # ---- whatever runs this module has died.
    free = cash["Free_To_Deploy"].sum() if not cash.empty else 0.0
    naked = positions[(positions["Acct"] != TOTAL)
                      & (positions["Has_Stop"] == NO)]["Ticker"].nunique()

    tok = token_status or {}
    state = str(tok.get("state", "UNKNOWN"))
    days = tok.get("days_left")
    tok_line = (f"{state} — expires {tok.get('expires', '?')}"
                + (f" ({days} days)" if days is not None else ""))
    ok = state == "OK"

    header = [
        ["SYSTEM STATUS", "OK" if ok else f"ACTION REQUIRED — SCHWAB TOKEN {state}"],
        ["SCHWAB TOKEN", tok_line],
        ["LAST POLL", _now()],
        ["TRADING", "DISABLED (kill switch)" if Path("/etc/myTrading/TRADING_DISABLED").exists()
                    else "reporting only — phase 1, nothing is placed"],
        ["FREE TO DEPLOY", f"${free:,.2f} across {len(cash)} account(s)"],
        ["ALERTS", f"{naked} holding(s) with no protective stop" if naked else "none"],
    ]
    book.worksheet("Orders").update(values=header, range_name="A1:B6")

    log(f"Orders sheet updated: {n_pos} position row(s), {n_cash} cash row(s), "
        f"free ${free:,.2f}, {naked} unprotected")
