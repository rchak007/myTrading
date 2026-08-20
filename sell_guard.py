#!/usr/bin/env python3
"""
sell_guard.py
=============
Shared helper: SELL_ORDER protection column.

For every ticker you actually HOLD (QTY / VALUE > 0), check the open Schwab
orders table for a live SELL leg priced BELOW the current price — i.e. a real
downside protective order (stop, stop-limit, or trailing stop that has resolved
to a price). A sell LIMIT above the current price is a profit target, not
protection, so it does NOT satisfy the check.

    🟢 SELL      — at least one protective SELL leg below current price
    🔴 NO_SELL   — position held, nothing protecting it
    ⚪ NO_PX     — position held but no usable current price to compare
    ""           — no position, nothing to check

Options legs are ignored by default: a covered call (SELL_TO_OPEN on an OPTION)
is income, not downside protection on the shares.

Public API mirrors hhll.py / ath.py:
    compute_sell_guard_table(df, df_orders)             -> DataFrame[Ticker, SELL_ORDER]
    attach_sell_order_column(df, df_orders, after=...)  -> (df_with_col, message)
"""
from __future__ import annotations

import pandas as pd

# Instruction values that reduce/close a long equity position
SELL_SIDES = {"SELL", "SELL_TO_CLOSE"}

# Icons / labels (kept here so there is exactly one definition)
FLAG_OK      = "🟢 SELL"
FLAG_MISSING = "🔴 NO_SELL"
FLAG_NO_PX   = "⚪ NO_PX"
FLAG_NONE    = ""

PRICE_COLS = ("Current Price", "Last Close", "Price")
QTY_COLS   = ("QTY", "VALUE")


def _num(v):
    """float or None — tolerates '', None, 'None', NaN, strings from CSV."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def _protective_prices(row) -> list[float]:
    """Trigger prices on an order leg that could sit below the market."""
    out = []
    for c in ("Stop_Price", "Limit_Price"):
        p = _num(row.get(c))
        if p is not None and p > 0:
            out.append(p)
    return out


def _live_sell_legs(df_orders: pd.DataFrame, equity_only: bool = True) -> pd.DataFrame:
    """Open SELL legs with quantity still working, keyed by Ticker."""
    if df_orders is None or len(df_orders) == 0 or "Ticker" not in df_orders.columns:
        return pd.DataFrame(columns=["Ticker"])

    d = df_orders.copy()
    d["Ticker"] = d["Ticker"].astype(str).str.upper().str.strip()
    d["Side"] = d.get("Side", "").astype(str).str.upper().str.strip()
    d = d[d["Side"].isin(SELL_SIDES)]

    if equity_only and "Asset_Type" in d.columns:
        d = d[d["Asset_Type"].astype(str).str.upper() == "EQUITY"]

    # Remaining_QTY is the live size; fall back to QTY when absent/blank.
    if "Remaining_QTY" in d.columns:
        rem = d["Remaining_QTY"].map(_num)
        qty = d["QTY"].map(_num) if "QTY" in d.columns else pd.Series([None] * len(d), index=d.index)
        live = rem.fillna(qty).fillna(0.0)
        d = d[live > 0]

    return d


def compute_sell_guard_table(
    df: pd.DataFrame,
    df_orders: pd.DataFrame,
    *,
    equity_only: bool = True,
    price_cols=PRICE_COLS,
    qty_cols=QTY_COLS,
) -> pd.DataFrame:
    """Return DataFrame[Ticker, SELL_ORDER] — one row per ticker in `df`."""
    if "Ticker" not in df.columns:
        return pd.DataFrame(columns=["Ticker", "SELL_ORDER"])

    px_col  = next((c for c in price_cols if c in df.columns), None)
    qty_col = next((c for c in qty_cols if c in df.columns), None)

    sells = _live_sell_legs(df_orders, equity_only=equity_only)
    by_ticker = {t: g for t, g in sells.groupby("Ticker")} if len(sells) else {}

    rows = []
    for _, r in df.iterrows():
        tkr = str(r.get("Ticker", "")).upper().strip()

        held = True
        if qty_col is not None:
            q = _num(r.get(qty_col)) or 0.0
            held = q > 0
        if not held:
            rows.append({"Ticker": r.get("Ticker"), "SELL_ORDER": FLAG_NONE})
            continue

        px = _num(r.get(px_col)) if px_col else None
        if px is None or px <= 0:
            rows.append({"Ticker": r.get("Ticker"), "SELL_ORDER": FLAG_NO_PX})
            continue

        legs = by_ticker.get(tkr)
        protected = False
        if legs is not None:
            for _, leg in legs.iterrows():
                if any(p < px for p in _protective_prices(leg)):
                    protected = True
                    break

        rows.append({"Ticker": r.get("Ticker"),
                     "SELL_ORDER": FLAG_OK if protected else FLAG_MISSING})

    return pd.DataFrame(rows)


def attach_sell_order_column(
    df: pd.DataFrame,
    df_orders: pd.DataFrame,
    after="Current Price",
    log_fn=print,
    *,
    equity_only: bool = True,
):
    """
    Add a SELL_ORDER column to df, placed immediately after the `after` column.

    `after` may be a single column name or a list of candidates — the first one
    present in df is used. Falls back to appending at the end.

    Returns (df, message). Never raises on bad/missing order data.
    """
    if "Ticker" not in df.columns:
        return df, "⚠️  SELL_ORDER: no 'Ticker' column — skipped"

    guard = compute_sell_guard_table(df, df_orders, equity_only=equity_only)
    if guard.empty:
        return df, "⚠️  SELL_ORDER: nothing computed"

    out = df.drop(columns=["SELL_ORDER"], errors="ignore").copy()
    out["SELL_ORDER"] = guard["SELL_ORDER"].values

    cols = [c for c in out.columns if c != "SELL_ORDER"]
    candidates = [after] if isinstance(after, str) else list(after)
    anchor = next((c for c in candidates if c in cols), None)

    if anchor is not None:
        cols.insert(cols.index(anchor) + 1, "SELL_ORDER")
    else:
        cols.append("SELL_ORDER")

    out = out[cols]

    n_held    = int((out["SELL_ORDER"] != FLAG_NONE).sum())
    n_missing = int((out["SELL_ORDER"] == FLAG_MISSING).sum())
    n_ok      = int((out["SELL_ORDER"] == FLAG_OK).sum())
    where = anchor if anchor else "end"
    return out, (f"SELL_ORDER: {n_held} held — {n_ok} protected, "
                 f"{n_missing} unprotected (placed after '{where}')")


# ─────────────────────────────────────────────────────────────────────
# CLI — read the two CSVs already on disk and report, no job import needed
#     python3 sell_guard.py [signals_csv] [orders_csv]
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sig_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("stocks_signals.csv")
    ord_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("stocks_orders.csv")

    d = pd.read_csv(sig_path)
    o = pd.read_csv(ord_path) if ord_path.exists() else pd.DataFrame()
    d, msg = attach_sell_order_column(d, o)
    print(msg)

    cols = [c for c in ("Ticker", "Current Price", "SELL_ORDER", "QTY", "VALUE") if c in d.columns]
    held = d[d["SELL_ORDER"] != FLAG_NONE]
    print(held[cols].to_string(index=False) if not held.empty else "(no positions)")