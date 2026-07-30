#!/usr/bin/env python3
"""
stocks_orders.py
================
Flattens OPEN (working / pending) Schwab orders into a table keyed to the
tickers you track.

DESIGN: this module owns *only* order parsing. Everything else is injected by
the caller so there is exactly one definition of each thing in the codebase:

    paths / JOB_DIR      -> jobStocksSignals.OUT_ORDERS_CSV / OUT_ORDERS_HTML
    log()                -> jobStocksSignals.log
    HTML rendering       -> jobStocksSignals.build_html_table
    token paths / client -> jobStocksSignals.get_schwab_client()

Importable (normal path — client comes from the caller):
    from stocks_orders import build_orders_table, write_orders_outputs

Standalone (resolves the above by loading jobStocksSignals.py, so still no
duplicated config):
    python3 stocks_orders.py               # open orders, watchlist only
    python3 stocks_orders.py --all         # include filled/cancelled
    python3 stocks_orders.py --any-ticker  # ignore STOCK_TICKERS filter
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

# Statuses that mean "still live on the book"
OPEN_STATUSES = {
    "AWAITING_PARENT_ORDER", "AWAITING_CONDITION", "AWAITING_STOP_CONDITION",
    "AWAITING_MANUAL_REVIEW", "ACCEPTED", "AWAITING_UR_OUT", "PENDING_ACTIVATION",
    "QUEUED", "WORKING", "NEW", "AWAITING_RELEASE_TIME", "PENDING_ACKNOWLEDGEMENT",
    "PENDING_RECALL",
}

ORDER_COLS = [
    "Ticker", "Symbol", "Asset_Type", "Side", "Order_Type", "Status",
    "QTY", "Filled_QTY", "Remaining_QTY", "Limit_Price", "Stop_Price",
    "Est_Value", "Duration", "Entered_Time", "Account", "Order_ID",
    "Strategy", "Cancelable",
]


# ─────────────────────────────────────────────────────────────────────
# Raw fetch — takes the client the caller already built
# ─────────────────────────────────────────────────────────────────────
def _raw_client(client_wrapper):
    """
    schwab_helper's SchwabClient wrapper exposes fetch_positions() but not orders.
    Reach the underlying schwabdev Client. The wrapper provides get_client();
    fall back to probing common attribute names for other wrapper versions.
    """
    if hasattr(client_wrapper, "account_linked"):
        return client_wrapper

    # Preferred: explicit accessor method. Trust whatever it returns —
    # the wrapper is explicitly handing us its underlying client.
    getter = getattr(client_wrapper, "get_client", None)
    if callable(getter):
        c = getter()
        if c is not None:
            return c

    # Fallback: attribute probing (older/other wrapper shapes)
    for attr in ("_client", "client", "schwab", "_schwab", "raw", "c"):
        c = getattr(client_wrapper, attr, None)
        if c is not None and hasattr(c, "account_linked"):
            return c

    attrs = [a for a in dir(client_wrapper) if not a.startswith("__")][:25]
    raise RuntimeError(
        f"No raw schwabdev client on {type(client_wrapper).__name__}; attrs={attrs}"
    )


def _call_orders_all(client, start, end):
    """account_orders_all across schwabdev naming variants."""
    for name in ("account_orders_all", "orders_all", "get_orders_all"):
        fn = getattr(client, name, None)
        if fn is None:
            continue
        try:
            resp = fn(start, end)
        except TypeError:
            resp = fn(fromEnteredTime=start, toEnteredTime=end)
        return resp.json() if hasattr(resp, "json") else resp
    return None


def _call_orders_per_account(client, start, end):
    """Fallback: iterate linked accounts."""
    resp = client.account_linked()
    accts = resp.json() if hasattr(resp, "json") else resp
    out = []
    for a in accts or []:
        if not isinstance(a, dict):
            continue
        h = a.get("hashValue")
        if not h:
            continue
        for name in ("account_orders", "orders"):
            fn = getattr(client, name, None)
            if fn is None:
                continue
            try:
                r = fn(h, start, end)
            except TypeError:
                r = fn(accountHash=h, fromEnteredTime=start, toEnteredTime=end)
            data = r.json() if hasattr(r, "json") else r
            if isinstance(data, list):
                out.extend(data)
            break
    return out


def fetch_schwab_orders(client_wrapper, days_back: int = 90, log=print) -> list:
    """Raw list of Schwab order dicts. Empty list on any failure (never raises)."""
    if client_wrapper is None:
        log("⚠️  No Schwab client supplied — skipping orders")
        return []
    try:
        client = _raw_client(client_wrapper)
        start = datetime.now(timezone.utc) - timedelta(days=days_back)
        end   = datetime.now(timezone.utc) + timedelta(days=1)

        data = _call_orders_all(client, start, end)
        if data is None:
            log("account_orders_all unavailable — per-account fallback")
            data = _call_orders_per_account(client, start, end)

        if data is None or (isinstance(data, list) and len(data) == 0):
            order_methods = [m for m in dir(client)
                             if "order" in m.lower() and not m.startswith("__")]
            log(f"    (client order-related methods: {order_methods})")

        if not isinstance(data, list):
            log(f"⚠️  Unexpected orders payload: {type(data).__name__}")
            return []

        log(f"Schwab orders fetched: {len(data)} (last {days_back}d)")
        return data
    except Exception as e:
        log(f"⚠️  Error fetching Schwab orders: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────
# Flatten
# ─────────────────────────────────────────────────────────────────────
def _iter_orders(orders):
    """Yield every order, recursing into OCO / bracket childOrderStrategies."""
    for o in orders or []:
        if not isinstance(o, dict):
            continue
        yield o
        yield from _iter_orders(o.get("childOrderStrategies") or [])


def _flatten_order(o: dict) -> list[dict]:
    """One row per order leg."""
    status     = str(o.get("status") or "")
    otype      = str(o.get("orderType") or o.get("orderTypeName") or "")
    limit_px   = o.get("price")
    stop_px    = o.get("stopPrice")
    tot_qty    = float(o.get("quantity") or 0.0)
    filled     = float(o.get("filledQuantity") or 0.0)
    remaining  = float(o.get("remainingQuantity") or max(tot_qty - filled, 0.0))
    entered    = str(o.get("enteredTime") or "")[:19].replace("T", " ")
    duration   = str(o.get("duration") or "")
    strategy   = str(o.get("orderStrategyType") or "")
    order_id   = o.get("orderId")
    acct       = o.get("accountNumber")
    cancelable = bool(o.get("cancelable", False))

    rows = []
    for leg in (o.get("orderLegCollection") or [{}]):
        inst   = leg.get("instrument", {}) or {}
        symbol = inst.get("symbol") or ""
        atype  = inst.get("assetType") or ""
        under  = inst.get("underlyingSymbol") or ""
        ticker = (under or symbol).split()[0].upper() if (under or symbol) else ""

        leg_qty = float(leg.get("quantity") or tot_qty or 0.0)
        px      = limit_px if limit_px not in (None, 0) else stop_px
        mult    = 100.0 if atype == "OPTION" else 1.0
        est_val = round(float(px) * leg_qty * mult, 2) if px not in (None, "") else None

        rows.append({
            "Ticker": ticker,
            "Symbol": symbol,
            "Asset_Type": atype,
            "Side": str(leg.get("instruction") or ""),
            "Order_Type": otype,
            "Status": status,
            "QTY": leg_qty,
            "Filled_QTY": filled,
            "Remaining_QTY": remaining,
            "Limit_Price": limit_px,
            "Stop_Price": stop_px,
            "Est_Value": est_val,
            "Duration": duration,
            "Entered_Time": entered,
            "Account": acct,
            "Order_ID": order_id,
            "Strategy": strategy,
            "Cancelable": cancelable,
        })
    return rows


def build_orders_table(
    client_wrapper,
    tickers: list[str] | None = None,
    *,
    days_back: int = 90,
    open_only: bool = True,
    restrict_to_tickers: bool = True,
    log=print,
) -> pd.DataFrame:
    """
    DataFrame of orders, one row per leg.

    client_wrapper     — from jobStocksSignals.get_schwab_client() (reused, not rebuilt)
    tickers            — STOCK_TICKERS; options match on underlyingSymbol
    open_only          — keep only live/working statuses
    restrict_to_tickers— False to see every open order regardless of watchlist
    """
    rows = []
    for o in _iter_orders(fetch_schwab_orders(client_wrapper, days_back, log)):
        rows.extend(_flatten_order(o))

    if not rows:
        log("No orders returned.")
        return pd.DataFrame(columns=ORDER_COLS)

    df = pd.DataFrame(rows)

    if open_only:
        before = len(df)
        df = df[df["Status"].str.upper().isin(OPEN_STATUSES)]
        log(f"Open-order filter: {before} → {len(df)} legs")

    if tickers and restrict_to_tickers:
        tset = {t.upper() for t in tickers}
        before = len(df)
        df = df[df["Ticker"].isin(tset)]
        log(f"Watchlist filter: {before} → {len(df)} legs")

    df = df[[c for c in ORDER_COLS if c in df.columns]]
    df = df.sort_values(["Ticker", "Entered_Time"], ascending=[True, False]).reset_index(drop=True)
    log(f"Orders table: {len(df)} rows, {df['Ticker'].nunique() if not df.empty else 0} tickers")
    return df


# ─────────────────────────────────────────────────────────────────────
# Output — paths and renderer injected, nothing defined here
# ─────────────────────────────────────────────────────────────────────
def write_orders_outputs(df, updated_pst, *, out_csv, out_html, html_builder, log=print) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    out_html.write_text(
        html_builder(df, "Open Schwab Orders", updated_pst), encoding="utf-8"
    )
    log(f"Orders outputs written: {out_csv.name} / {out_html.name}")


# ─────────────────────────────────────────────────────────────────────
# CLI — pulls every dependency from jobStocksSignals, defines none
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, importlib.util, sys
    from pathlib import Path
    import pytz

    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include filled/cancelled/rejected")
    ap.add_argument("--any-ticker", action="store_true", help="ignore STOCK_TICKERS filter")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--no-write", action="store_true", help="print only, no CSV/HTML")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    spec = importlib.util.spec_from_file_location("_job_stocks", str(here / "jobStocksSignals.py"))
    job  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)

    tickers = None
    try:
        aspec = importlib.util.spec_from_file_location("myTrading_app", str(job.MYTRADING_DIR / "app.py"))
        amod  = importlib.util.module_from_spec(aspec)
        aspec.loader.exec_module(amod)
        tickers = getattr(amod, "STOCK_TICKERS", None)
    except Exception as e:
        print(f"⚠️  Could not load STOCK_TICKERS: {e}")

    d = build_orders_table(
        job.get_schwab_client(),
        tickers,
        days_back=args.days,
        open_only=not args.all,
        restrict_to_tickers=not args.any_ticker,
        log=job.log,
    )
    print(d.to_string(index=False) if not d.empty else "(no orders)")

    if not args.no_write:
        updated = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z")
        write_orders_outputs(
            d, updated,
            out_csv=job.OUT_ORDERS_CSV, out_html=job.OUT_ORDERS_HTML,
            html_builder=job.build_html_table, log=job.log,
        )