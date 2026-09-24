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
    """Pacific, explicitly — the Pi's own clock may be on UTC, and LAST POLL
    is read against market hours. Falls back to local time if the tz database
    is missing rather than failing the write."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Los_Angeles")).strftime(
            "%Y-%m-%d %H:%M:%S %Z")
    except Exception:
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
    bare = df[(df["Acct"] != TOTAL) & (df["Has_Stop"] == NO)]
    # Unique tickers, not rows — a ticker held in three accounts is one
    # problem listed three times, and the header line counts uniques too.
    names = sorted(bare["Ticker"].unique())
    if names:
        shown = ", ".join(names[:12]) + (" …" if len(names) > 12 else "")
        log(f"⚠️  {len(names)} ticker(s) unprotected in {len(bare)} (ticker, account) "
            f"row(s): {shown}")
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
        # Test the RAW value before masking. build_cash_table() appends a TOTAL
        # row, and acct_key("TOTAL") is "TAL" — which matches nothing, so the
        # total was counted as a seventh account and Free_To_Deploy came out
        # exactly doubled.
        raw = str(r.get("Account") or "").strip().upper()
        if not raw or raw == TOTAL:
            continue
        a = acct_key(raw)
        if not a:
            continue
        after = _num(r.get("Cash_After_Open_Orders"), 0.0) or 0.0
        seed = round(per_acct.get(a, 0.0), 2)
        free = round(after - seed, 2)
        if free < 0:
            # Nothing validates a seed against real cash (PROJECT_PLAN §2), so
            # this is where over-fencing first becomes visible. Say it out loud
            # rather than leaving a negative number in a cell nobody scrolls to.
            log(f"⚠️  {a} is over-fenced: ${seed:,.2f} reserved against "
                f"${after:,.2f} available — short ${abs(free):,.2f}")
        out.append({
            "Acct": a, "Nickname": r.get("Nickname", ""),
            "Cash": _num(r.get("Cash"), 0.0),
            "Cash_In_Open_Orders": _num(r.get("Cash_In_Open_Orders"), 0.0),
            "Cash_After_Open_Orders": after,
            "Seed_Reserved": seed,
            "Free_To_Deploy": free,
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


def load_reserves(log=print) -> dict:
    """{(acct_last3, TICKER): reserved_cash} folded from cash_reserve's ledger.

    Fail-soft on purpose. A missing ledger, an unimportable module or a bad row
    must cost the Seed_Reserved column, never the whole sheet write — positions
    and coverage flags are the part you cannot get anywhere else.

    Pairs folding to zero (a closed or fully deployed reserve) are dropped, so
    the column shows blank rather than a misleading $0.00 next to a ticker that
    has no reserve at all.
    """
    try:
        import cash_reserve
    except Exception as e:
        log(f"⚠️  cash_reserve unavailable — Seed_Reserved left blank: {e}")
        return {}

    try:
        folded = cash_reserve.fold_balances()
    except Exception as e:
        log(f"⚠️  could not fold the reserve ledger — Seed_Reserved blank: {e}")
        return {}

    if folded is None or folded.empty:
        return {}

    out = {}
    for _, r in folded.iterrows():
        amt = float(r["Reserved_Cash"])
        if abs(amt) < 0.005:                 # closed, or spent down to nothing
            continue
        out[(acct_key(r["Account"]), str(r["Ticker"]).strip().upper())] = amt

    if out:
        log(f"Reserves loaded: {len(out)} fenced pair(s) · "
            f"${sum(out.values()):,.2f} reserved")
    return out


def _scrub(v):
    """NaN/NaT/None -> "". Anything else passes through untouched.

    pd.isna() raises on a list or array rather than returning a scalar, so the
    guard is not decoration — it keeps a stray sequence from taking the whole
    write down.
    """
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return v


def _put(ws, df, cols, log):
    ws.batch_clear([f"A2:{chr(64 + len(cols))}1000"])
    if df is None or df.empty:
        return 0
    body = [[_scrub(v) for v in row] for row in df[cols].values.tolist()]
    ws.update(values=body, range_name=f"A2:{chr(64 + len(cols))}{len(body) + 1}")
    return len(body)


# Live_Price and Day_% sit next to Avg_Cost on purpose: paid / now / today's
# move reads left to right. Both are filled by orders_sheet_prices.py on its own
# faster schedule, so they are written blank here rather than with a stale value.
POS_HDR = ["Ticker", "Acct", "Qty", "Avg_Cost", "Live_Price", "Day_%",
           "Market_Value", "Unrealized_PL",
           "Has_Stop", "Has_Trim", "Has_Dip", "Has_Breakout", "Seed_Reserved"]
# Column letters within a block (column A is the ticker label, so POS_HDR
# starts at B). orders_sheet_prices.py writes into these two.
COL_LIVE_PRICE = "F"
COL_DAY_PCT = "G"


def _last_col() -> str:
    """Rightmost column letter of a dashboard block.

    Derived, not hardcoded: adding Live_Price and Day_% moved it from L to N,
    and three separate range strings said "L".
    """
    n, s = DASH_WIDTH, ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s
ORD_HDR = ["Acct", "Side", "Type", "Qty", "Limit_Price", "Stop_Price",
           "Status", "Entered", "Order_ID"]
DASH_WIDTH = 1 + len(POS_HDR)          # column A holds the ticker label


def build_dashboard(positions: pd.DataFrame, orders_df=None) -> tuple[list, dict]:
    """
    One block per ticker: the positions mini-table, then the live Schwab orders
    for that ticker.

    Read-only by construction. Nothing typed by a human lives here, which is the
    whole reason it can be regenerated wholesale every cycle without a merge
    step — and without any risk of eating an intent.

    Returns (rows, marks) where marks records which row numbers are ticker
    headers / section labels / column headers, so formatting can be applied in
    one batch afterwards.
    """
    rows: list[list] = []
    marks = {"ticker": [], "label": [], "header": [], "title": []}

    def add(vals):
        # Scrub NaN the way _put() does. Google's API rejects the entire
        # batch on a single non-compliant float, so one NaN in one order leg
        # silently blanks the whole tab. Order legs are full of them: a stop
        # order has no Limit_Price, a limit order has no Stop_Price.
        clean = [_scrub(v) for v in vals]
        rows.append(clean + [""] * (DASH_WIDTH - len(clean)))
        return len(rows)                      # 1-based row number

    n_tickers = positions.loc[positions["Acct"] != TOTAL, "Ticker"].nunique()
    marks["title"].append(add(["DASHBOARD", f"updated {_now()}"]))
    add(["", f"{n_tickers} ticker(s) · read-only, regenerated every cycle · "
             f"ORDERS rows are live Schwab orders, not the Orders tab"])
    add([])

    for ticker in sorted(positions["Ticker"].unique()):
        grp = positions[positions["Ticker"] == ticker]

        marks["ticker"].append(add([ticker, "POSITIONS"]))
        marks["header"].append(add([""] + POS_HDR))
        for _, r in grp.iterrows():
            add(["", r["Ticker"], r["Acct"], r["Qty"], r["Avg_Cost"],
                 "", "",                      # Live_Price, Day_% — see below
                 r["Market_Value"], r["Unrealized_PL"], r["Has_Stop"],
                 r["Has_Trim"], r["Has_Dip"], r["Has_Breakout"], r["Seed_Reserved"]])

        add([])
        marks["label"].append(add(["", "ORDERS"]))
        marks["header"].append(add([""] + ORD_HDR))

        legs = pd.DataFrame()
        if orders_df is not None and not getattr(orders_df, "empty", True):
            legs = orders_df[orders_df["Ticker"].astype(str).str.upper() == ticker]
        if legs.empty:
            add(["", "— no open orders —"])
        else:
            for _, o in legs.iterrows():
                add(["", acct_key(o.get("Account")), o.get("Side", ""),
                     o.get("Order_Type", ""), o.get("Remaining_QTY", o.get("QTY", "")),
                     o.get("Limit_Price", ""), o.get("Stop_Price", ""),
                     o.get("Status", ""), str(o.get("Entered_Time", ""))[:16],
                     o.get("Order_ID", "")])
        add([])
        add([])

    return rows, marks


# Chakravarti's saved TradingView layout — it already carries his indicators,
# so the link only has to hand it a symbol. Override if the layout changes.
TV_CHART = os.getenv("TRADINGVIEW_CHART", "ajSFidjP")


def _tv_url(ticker: str) -> str:
    """No exchange prefix on purpose.

    The obvious form is NASDAQ:AEHR, but these holdings span NASDAQ, NYSE and
    NYSE Arca (IBIT, ARKB, HODL...), and a hardcoded prefix breaks every ticker
    that is not on that exchange. A bare symbol lets TradingView resolve the
    primary listing itself.
    """
    return f"https://www.tradingview.com/chart/{TV_CHART}/?symbol={ticker}"


def _link_tickers(ws, rows, marks, log):
    """Turn each block's ticker label into a TradingView hyperlink.

    Written separately from the bulk update, and only over the ticker cells,
    because a formula needs USER_ENTERED while the data wants RAW — sending the
    whole table as USER_ENTERED would let Sheets reinterpret values it has no
    business touching, such as an order's Entered timestamp becoming a date.

    Best-effort, like _paint: a link is a convenience, the data is not.
    """
    try:
        payload = []
        for r in marks.get("ticker", []):
            t = str(rows[r - 1][0]).strip()
            if not t:
                continue
            payload.append({"range": f"A{r}",
                            "values": [[f'=HYPERLINK("{_tv_url(t)}","{t}")']]})
        if payload:
            ws.batch_update(payload, value_input_option="USER_ENTERED")
            log(f"Dashboard: {len(payload)} ticker(s) linked to TradingView")
    except Exception as e:
        log(f"⚠️  dashboard TradingView links skipped (data is fine): {e}")


def _paint(ws, marks, log):
    """Cosmetics. Best-effort — never let a formatting call lose the data."""
    cyan = {"backgroundColor": {"red": 0.80, "green": 0.95, "blue": 1.0},
            "textFormat": {"bold": True}}
    yellow = {"backgroundColor": {"red": 1.0, "green": 0.95, "blue": 0.60},
              "textFormat": {"bold": True}}
    bold = {"textFormat": {"bold": True}}
    title = {"backgroundColor": {"red": 0.20, "green": 0.25, "blue": 0.35},
             "textFormat": {"bold": True, "fontSize": 12,
                            "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}}}
    # Column A carries the TradingView HYPERLINK. Sheets styles a link blue and
    # underlined, but a later textFormat write silently replaces that with the
    # cell's own colour — so the link stayed clickable while looking like plain
    # text, and therefore looked broken. Style A explicitly, B separately.
    cyan_link = {"backgroundColor": cyan["backgroundColor"],
                 "textFormat": {"bold": True, "underline": True,
                                "foregroundColor": {"red": 0.05, "green": 0.25,
                                                    "blue": 0.75}}}
    try:
        if marks.get("title"):
            ws.format([f"A{r}:{_last_col()}{r}" for r in marks["title"]], title)
        if marks["ticker"]:
            ws.format([f"A{r}" for r in marks["ticker"]], cyan_link)
            ws.format([f"B{r}" for r in marks["ticker"]], cyan)
        if marks["label"]:
            ws.format([f"B{r}" for r in marks["label"]], yellow)
        if marks["header"]:
            ws.format([f"B{r}:{_last_col()}{r}" for r in marks["header"]], bold)
    except Exception as e:
        log(f"⚠️  dashboard formatting skipped (data is fine): {e}")


def write_dashboard(book, positions: pd.DataFrame, orders_df=None, log=print) -> int:
    if positions is None or positions.empty:
        return 0
    try:
        ws = book.worksheet("Dashboard")
    except Exception:
        ws = book.add_worksheet(title="Dashboard", rows=1000, cols=DASH_WIDTH + 2)

    rows, marks = build_dashboard(positions, orders_df)
    ws.clear()
    ws.update(values=rows, range_name=f"A1:{_last_col()}{len(rows)}")
    ws.freeze(rows=0)
    _link_tickers(ws, rows, marks, log)
    _paint(ws, marks, log)          # after the links, so the cyan survives
    return len(marks["ticker"])


def write_orders_sheet(*, client_wrapper, signals_df=None, orders_df=None,
                       cash_df=None, reserves=None, token_status=None,
                       positions_raw=None, log=print) -> None:
    """
    Refresh Positions, Cash and the header block. Never touches columns A-K of
    Orders — those are the human's.
    """
    book = _open_book()

    # Caller may inject reserves (tests, or a future caller that already has
    # them); otherwise read the ledger, which is the authority.
    if reserves is None:
        reserves = load_reserves(log=log)

    # Caller may have fetched positions already (the reserves step needs the
    # same frame); one Schwab round-trip is worth avoiding.
    pos_raw = (positions_raw if positions_raw is not None
               else fetch_positions_detailed(client_wrapper, log=log))
    positions = build_positions_table(pos_raw, signals_df, orders_df, reserves, log=log)
    cash = build_cash_rows(cash_df, reserves, log=log)

    n_pos = _put(book.worksheet("Positions"), positions, POSITIONS_COLS, log)
    n_cash = _put(book.worksheet("Cash"), cash, CASH_COLS, log)
    n_dash = write_dashboard(book, positions, orders_df, log=log)

    # ---- header block. LAST POLL is the health check: if this stops moving,
    # ---- whatever runs this module has died.
    free = cash["Free_To_Deploy"].sum() if not cash.empty else 0.0
    naked = positions[(positions["Acct"] != TOTAL)
                      & (positions["Has_Stop"] == NO)]["Ticker"].nunique()

    over = (cash[cash["Free_To_Deploy"] < 0]["Acct"].tolist()
            if not cash.empty else [])
    seeded_total = cash["Seed_Reserved"].sum() if not cash.empty else 0.0

    alerts = []
    if naked:
        alerts.append(f"{naked} holding(s) with no protective stop")
    if over:
        alerts.append(f"OVER-FENCED: {', '.join(over)} reserved beyond cash")

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
        ["FREE TO DEPLOY", f"${free:,.2f} across {len(cash)} account(s)"
                           + (f" · ${seeded_total:,.2f} seed-reserved"
                              if seeded_total else "")],
        ["ALERTS", " · ".join(alerts) if alerts else "none"],
    ]
    book.worksheet("Orders").update(values=header, range_name="A1:B6")

    log(f"Orders sheet updated: {n_pos} position row(s), {n_cash} cash row(s), "
        f"{n_dash} dashboard block(s), free ${free:,.2f}, {naked} unprotected")
