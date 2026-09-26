#!/usr/bin/env python3
"""
order_engine.py
===============
Evaluate armed rows against the daily close and submit what triggers.

    .venv/bin/python order_engine.py                 # evaluate, honour DRY RUN
    .venv/bin/python order_engine.py --status        # show state, touch nothing
    .venv/bin/python order_engine.py --force-time    # ignore the 13:15 PT gate

Phase 3 of Documentation/ordersSheetDesign-9-18-26.md, inheriting the security
model of orderExecutionDesign-9-7-26.md §6.

THE ONE THING TO UNDERSTAND
    A close trigger submits the NEXT TRADING MORNING. The daily bar has to be
    final before it can be evaluated, and by then the market is shut. If you
    need the fill at that close, this is the wrong mechanism.

WHAT BELONGS HERE, AND WHAT DOES NOT
    Here:     "buy AVGO if it CLOSES above 245" — Schwab cannot express that.
    Not here: "buy AVGO at 245" — that is a resting limit order. Place it at
              Schwab, where it works whether or not Pi 1 is awake.

    The difference is real: a limit order fills the instant price TOUCHES the
    level, wick included. A close trigger waits for the bar to finish, so a
    spike that reverses does not fire it.

NO SIGNING KEY. Decided 2026-09-26. Intents are typed straight into the sheet
from a phone; requiring a laptop to sign each one defeats the point of having a
sheet. The accepted risk is that sheet access implies the ability to cause
TRADES — not to move money out, which a brokerage order cannot do.

WHAT PROTECTS YOU, IN ORDER OF HOW MUCH IT MATTERS
    1. LIVE_TRADING defaults to FALSE. Nothing reaches Schwab until someone
       sets ORDER_ENGINE_LIVE=1 on Pi 1, deliberately.
    2. The kill switch file blocks every submission, checked immediately before
       each one rather than once at startup.
    3. Hard caps on notional per order, per day, and submissions per day. With
       no signature these are the primary control, so they start small.
    4. Guards that read the ACTUAL account: never sell more than is held,
       never trade a symbol that is neither held nor watched, never place a
       buy that exceeds free cash. A mis-typed quantity fails these.
    5. Account allowlist, failing CLOSED when unset.
    6. A validation line written back into the sheet EVERY cycle, so a bad row
       says so within minutes of being typed rather than at the close.
    7. Write-ahead to the ledger before any submit, so a crash mid-call is
       detectable rather than silently repeatable.
    8. Triggers fire on a completed daily CLOSE, never an intrabar touch —
       matching the HHLL/pivot analysis, which is close-based throughout.

THE REALISTIC FAILURE is a mis-typed cell, not a break-in: a formula
autofilling down, 10 becoming 100, a paste landing one row off. Guards 3, 4
and 6 exist for that.

Runs on Pi 1.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import order_intent as oc                            # noqa: E402
import order_exec_config as cfg                         # noqa: E402

DATA_START_ROW = 9          # Orders tab: header block 1-6, banner 7, cols 8

# Engine-owned cells, by letter. Named rather than inlined: the columns shifted
# named rather than inlined because a write-back aimed at the wrong column
# silently overwrites a different field.
COL_STATUS = "K"
COL_STATUS_DATE = "L"
COL_VALIDATION = "M"
COL_ENGINE_NOTE = "S"
COL_LAST_CHECKED = "T"
SHEET_COLS = ["Row_ID", "Date", "Acct", "Ticker", "Side", "Close_Is",
              "Trigger_Price", "Limit_Price", "Qty", "Expires_On"]

LEDGER_COLS = ["ts", "row_id", "fingerprint", "state", "note", "acct", "ticker",
               "side", "close_is", "qty", "trigger_price", "limit_price",
               "trigger_close", "idem_key", "submit_attempted", "schwab_order_id"]


def _now():
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/Los_Angeles"))


def _log(msg: str) -> None:
    print(f"[{_now():%Y-%m-%d %H:%M:%S %Z}] {msg}")


def audit(**fields) -> None:
    fields.setdefault("ts", _now().isoformat())
    cfg.STATE_DIR.mkdir(parents=True, exist_ok=True)
    with cfg.AUDIT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(fields, default=str) + "\n")


# ────────────────────────────────────────────────────────────── ledger
def read_ledger() -> list[dict]:
    if not cfg.LEDGER.exists():
        return []
    with cfg.LEDGER.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def append_ledger(row: dict) -> None:
    """Append-only. The ledger is the record of what happened; the sheet is a
    mirror of it. Never rewrite a line — a correction is a new line."""
    cfg.STATE_DIR.mkdir(parents=True, exist_ok=True)
    new = not cfg.LEDGER.exists()
    with cfg.LEDGER.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({**{c: "" for c in LEDGER_COLS},
                    "ts": _now().isoformat(), **row})


def latest_states(ledger: list[dict]) -> dict[str, dict]:
    """Last line wins per Row_ID — the fold over an append-only log."""
    out = {}
    for r in ledger:
        if r.get("row_id"):
            out[r["row_id"]] = r
    return out


def todays_submissions(ledger: list[dict]) -> tuple[int, float]:
    """Counters live in the ledger, not memory, so a restart cannot reset them."""
    today = _now().date().isoformat()
    n, notional = 0, 0.0
    for r in ledger:
        if not str(r.get("ts", "")).startswith(today):
            continue
        if r.get("state") in ("SUBMITTED", "FILLED"):
            n += 1
            try:
                notional += abs(float(r.get("qty") or 0)) * \
                            float(r.get("limit_price") or r.get("trigger_price") or 0)
            except (TypeError, ValueError):
                pass
    return n, notional


# ────────────────────────────────────────────────────────── preflight
def preflight(force_time: bool) -> list[str]:
    """Reasons not to run. Empty list means proceed.

    Fails CLOSED: anything unverifiable is a stop, not a warning. A cheap
    missed cycle beats an order placed on a assumption.
    """
    stop = []

    if not force_time:
        h, m = cfg.EVALUATE_AFTER_PT
        now = _now()
        if (now.hour, now.minute) < (h, m):
            stop.append(f"too early — the daily bar is not final until "
                        f"{h:02d}:{m:02d} PT (now {now:%H:%M})")
        if now.weekday() >= 5:
            stop.append("weekend — no daily bar to evaluate")

    if not cfg.ACCOUNT_ALLOWLIST:
        stop.append("ORDER_ACCOUNT_ALLOWLIST is empty — every row would be "
                    "blocked. Set it to the accounts confirmed to accept "
                    "order placement.")
    return stop


# ─────────────────────────────────────────────────────────── the sheet
def open_orders_tab():
    from google.oauth2.service_account import Credentials
    import gspread

    sid = os.getenv("GSHEET_ORDERS_ID", "")
    creds = os.getenv("REMOTE_OPS_CREDS",
                      os.getenv("GSHEET_CREDS", "/etc/myTrading/gsheets-ops.json"))
    if not sid:
        raise RuntimeError("GSHEET_ORDERS_ID is not set")
    if not Path(creds).exists():
        raise RuntimeError(f"service-account key not found at {creds}")
    gc = gspread.authorize(Credentials.from_service_account_file(
        creds, scopes=["https://www.googleapis.com/auth/spreadsheets"]))
    return gc.open_by_key(sid).worksheet("Orders")


def read_rows(ws) -> list[dict]:
    """Intent rows from the sheet, with their 1-based sheet row number."""
    values = ws.get_all_values()
    out = []
    for i, raw in enumerate(values, start=1):
        if i < DATA_START_ROW:
            continue
        cells = list(raw) + [""] * len(SHEET_COLS)
        rec = {c: str(cells[j]).strip() for j, c in enumerate(SHEET_COLS)}
        if not rec["Row_ID"] and not rec["Ticker"]:
            continue                                # blank spacer
        rec["_sheet_row"] = i
        out.append(rec)
    return out


# ──────────────────────────────────────────────────────── evaluation
def daily_close(client_wrapper, ticker: str, log=print) -> tuple[float | None, str]:
    """(close, note) for the most recent COMPLETED daily bar.

    Schwab, not yfinance: the hotspot blocks Cloudflare and yfinance goes down
    with it. A failed fetch must leave the row ARMED — never advance a row on
    a price we could not read.
    """
    inner = client_wrapper
    getter = getattr(client_wrapper, "get_client", None)
    if callable(getter):
        try:
            inner = getter() or client_wrapper
        except Exception as e:
            return None, f"schwab client unavailable: {e}"

    meth = getattr(inner, "price_history", None)
    if not callable(meth):
        return None, "this schwabdev exposes no price_history()"

    try:
        resp = meth(ticker, periodType="month", period=1,
                    frequencyType="daily", frequency=1)
        body = resp.json() if hasattr(resp, "json") else resp
    except Exception as e:
        return None, f"price history failed: {type(e).__name__}: {e}"

    candles = (body or {}).get("candles") or []
    if not candles:
        return None, "no candles returned"

    last = candles[-1]
    close = last.get("close")
    when = datetime.fromtimestamp(last.get("datetime", 0) / 1000).date()
    if close is None:
        return None, "last candle has no close"

    # Guard against acting on a stale bar — a holiday, or a data source that
    # has stopped updating. Three days covers a long weekend.
    if (date.today() - when).days > 3:
        return None, f"last daily bar is {when}, too old to act on"
    return float(close), f"close {close} on {when}"


def cross_check(close: float, ticker: str, log=print) -> str | None:
    """Compare against stocks_signals.csv; refuse to act if they disagree.

    Two independent sources agreeing is weak evidence; two disagreeing is
    strong evidence that one is wrong, and we cannot tell which.
    """
    csv_path = Path.home() / "github" / "jobMyTrading" / "stocks_signals.csv"
    if not csv_path.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        row = df[df["Ticker"].astype(str).str.upper() == ticker]
        if row.empty:
            return None
        col = next((c for c in ("Last Close", "Current Price") if c in df.columns), None)
        if not col:
            return None
        other = float(row.iloc[0][col])
    except Exception:
        return None
    if other <= 0:
        return None
    diff = abs(close - other) / other * 100
    if diff > cfg.PRICE_DISAGREEMENT_PCT:
        return (f"PRICE_DISAGREEMENT: Schwab {close:.2f} vs signals {other:.2f} "
                f"({diff:.1f}% apart) — not acting")
    return None


def _known_tickers() -> set[str]:
    """STOCK_TICKERS out of app.py without importing it."""
    try:
        import ast
        import re as _re
        src = (HERE / "app.py").read_text(encoding="utf-8")
        m = _re.search(r"STOCK_TICKERS\s*=\s*\[(.*?)\]", src, _re.S)
        return {str(t).strip().upper()
                for t in ast.literal_eval("[" + m.group(1) + "]")} if m else set()
    except Exception:
        return set()


def fp_of(rec: dict) -> str:
    """Fingerprint, or "" when the row is too malformed to have one.

    finish() is called for MALFORMED rows too, so this must never raise —
    losing the audit line would be worse than losing the fingerprint.
    """
    try:
        return oc.fingerprint(rec)
    except Exception:
        return ""


def triggered(close_is: str, close: float, trigger: float) -> bool:
    """Strictly through the level. A close exactly ON the trigger does not
    fire — "above 245" means above, and equality is the one case where doing
    nothing is always defensible."""
    return close > trigger if close_is == "ABOVE" else close < trigger


# ─────────────────────────────────────────────────────────── guards
def account_state(client_wrapper, log=print) -> dict:
    """What the account ACTUALLY holds, per (acct, ticker), plus free cash.

    Without a signature this is the real protection. A mis-typed "sell 1000"
    is caught here — not because 1000 looks odd, but because the account holds
    26 and the engine can see that.
    """
    out = {"positions": {}, "cash": {}, "tickers": set()}
    try:
        from orders_sheet import fetch_positions_detailed
        pos = fetch_positions_detailed(client_wrapper, log=lambda *a: None)
        for _, r in pos.iterrows():
            out["positions"][(str(r["Acct"]), str(r["Ticker"]).upper())] = float(r["Qty"])
            out["tickers"].add(str(r["Ticker"]).upper())
    except Exception as e:
        log(f"⚠️  could not read positions: {e}")
        return {}                      # empty means "unknown" -> fail closed

    try:
        import pandas as pd
        cash_csv = Path.home() / "github" / "jobMyTrading" / "cash.csv"
        if cash_csv.exists():
            df = pd.read_csv(cash_csv)
            for _, r in df.iterrows():
                acct = str(r.get("Account") or "").strip().upper()
                if not acct or acct == "TOTAL":
                    continue
                key = acct[-3:] if len(acct) >= 3 else acct
                val = r.get("Cash_After_Open_Orders", r.get("Cash"))
                try:
                    out["cash"][key] = float(str(val).replace(",", "").replace("$", ""))
                except (TypeError, ValueError):
                    pass
    except Exception as e:
        log(f"⚠️  could not read cash.csv: {e}")
    return out


def check_guards(n: dict, close: float | None, submitted_today: int,
                 notional_today: float, acct_state: dict | None = None) -> str | None:
    """The reason this must NOT be submitted, or None.

    `close` may be None during validation, before any price is known; the
    price-dependent checks are then skipped and everything else still runs, so
    a bad row is reported the moment it is typed.
    """
    if cfg.kill_switch_on():
        return f"KILL_SWITCH: {cfg.KILL_SWITCH} exists"

    if n["Acct"] not in cfg.ACCOUNT_ALLOWLIST:
        return (f"ACCOUNT_NOT_ALLOWED: {n['Acct']} is not in "
                f"{sorted(cfg.ACCOUNT_ALLOWLIST)}")

    if cfg.TICKER_ALLOWLIST and n["Ticker"] not in cfg.TICKER_ALLOWLIST:
        return f"TICKER_NOT_ALLOWED: {n['Ticker']}"

    if not n["Limit_Price"] and not cfg.ALLOW_MARKET_ORDERS:
        return ("NO_LIMIT_PRICE: market orders are disabled "
                "(ALLOW_MARKET_ORDERS=0)")

    qty = float(n["Qty"])
    px = float(n["Limit_Price"] or n["Trigger_Price"])
    notional = qty * px
    side = n["Side"]

    if notional > cfg.MAX_NOTIONAL_PER_ORDER:
        return (f"CAP_PER_ORDER: ${notional:,.2f} exceeds "
                f"${cfg.MAX_NOTIONAL_PER_ORDER:,.2f}")
    if notional_today + notional > cfg.MAX_NOTIONAL_PER_DAY:
        return (f"CAP_PER_DAY: ${notional_today:,.2f} already + ${notional:,.2f} "
                f"exceeds ${cfg.MAX_NOTIONAL_PER_DAY:,.2f}")
    if submitted_today >= cfg.MAX_SUBMISSIONS_PER_DAY:
        return (f"CAP_SUBMISSIONS: {submitted_today} orders already placed "
                f"today, limit is {cfg.MAX_SUBMISSIONS_PER_DAY}. This guards "
                f"against a runaway loop, so check the ledger before raising "
                f"it. To raise: MAX_SUBMISSIONS_PER_DAY=25 in .env")

    # ---- guards that read the real account ----------------------------
    if acct_state is not None:
        if not acct_state:
            return "ACCOUNT_UNKNOWN: could not read positions — refusing to act blind"

        held = acct_state["positions"].get((n["Acct"], n["Ticker"]))

        if side == "SELL":
            if held is None:
                return (f"NOTHING_TO_SELL: no {n['Ticker']} position in "
                        f"{n['Acct']}")
            if qty > held + 1e-6:
                return (f"OVERSELL: {qty:g} shares but only {held:g} held in "
                        f"{n['Acct']} — check for a typo")

        if side == "BUY":
            # Unheld and unwatched is almost always a mistyped symbol.
            if held is None and n["Ticker"] not in acct_state["tickers"]:
                # Parsed, not imported: app.py pulls in Streamlit, which is
                # absent on some machines, and the import failing would
                # silently skip this check rather than announcing itself.
                known = _known_tickers()
                if known and n["Ticker"] not in known:
                    return (f"UNKNOWN_TICKER: {n['Ticker']} is neither held nor "
                            f"in STOCK_TICKERS — likely a typo")
            free = acct_state["cash"].get(n["Acct"])
            if free is not None and notional > free:
                return (f"INSUFFICIENT_CASH: ${notional:,.2f} needed, "
                        f"${free:,.2f} available in {n['Acct']}")

    # ---- price-dependent, only once a close is known ------------------
    if close is not None:
        lim = float(n["Limit_Price"]) if n["Limit_Price"] else None
        if lim:
            through = ((close - lim) / lim * 100) if side == "BUY" else \
                      ((lim - close) / lim * 100)
            if through > cfg.GAP_THROUGH_PCT:
                return (f"GAPPED_THROUGH: close {close:.2f} is {through:.1f}% "
                        f"past the {side} limit {lim:.2f}")
    return None


def expired(n: dict) -> str | None:
    if not n["Expires_On"]:
        return None
    try:
        exp = datetime.strptime(n["Expires_On"], "%Y-%m-%d").date()
    except ValueError:
        return "BAD_EXPIRY"
    if exp < date.today():
        return f"EXPIRED on {exp}"
    if exp > date.today() + timedelta(days=cfg.MAX_EXPIRY_DAYS):
        return f"EXPIRY_TOO_FAR: max {cfg.MAX_EXPIRY_DAYS} days out"
    return None


# ─────────────────────────────────────────────────────────── submit
def submit(client_wrapper, n: dict, idem: str, log=print) -> tuple[str | None, str]:
    """Place the order. Returns (schwab_order_id, note).

    NOT IMPLEMENTED ON PURPOSE. This is step 11 of the design's implementation
    order, and every step before it has to have run clean first. The function
    exists so the surrounding machinery — write-ahead, caps, reconciliation —
    can be exercised end to end against a stub that cannot spend anything.

    When it is written it must: build the order JSON, POST once, and return the
    id. It must never retry on an ambiguous failure — that is what the
    write-ahead ledger entry is for.
    """
    return None, ("SUBMIT_NOT_IMPLEMENTED — the engine evaluated this row and "
                  "would have placed it. Wiring the Schwab call is the last "
                  "step, after a clean dry run.")


# ───────────────────────────────────────────────────────────── main
def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate close-triggered orders")
    ap.add_argument("--status", action="store_true",
                    help="print configuration and ledger state, change nothing")
    ap.add_argument("--force-time", action="store_true",
                    help="skip the 13:15 PT gate (testing only)")
    args = ap.parse_args()

    print()
    print(cfg.summary())
    print()

    ledger = read_ledger()
    states = latest_states(ledger)
    n_sub, notional_today = todays_submissions(ledger)

    if args.status:
        _log(f"ledger: {len(ledger)} line(s), {len(states)} distinct row(s)")
        _log(f"today: {n_sub} submission(s), ${notional_today:,.2f}")
        for rid, r in sorted(states.items()):
            print(f"   {rid:<26} {r.get('state',''):<10} {r.get('note','')[:60]}")
        return 0

    stop = preflight(args.force_time)
    if stop:
        for s in stop:
            _log(f"⛔ {s}")
        return 1

    try:
        ws = open_orders_tab()
        rows = read_rows(ws)
    except Exception as e:
        _log(f"⛔ cannot read the Orders tab: {type(e).__name__}: {e}")
        return 1

    _log(f"{len(rows)} intent row(s) in the sheet")
    if len(rows) > cfg.MAX_ARMED_ROWS:
        _log(f"⛔ {len(rows)} rows exceeds MAX_ARMED_ROWS={cfg.MAX_ARMED_ROWS}")
        return 1

    import jobStocksSignals as job
    client = job.get_schwab_client()

    # Read the real account ONCE. Every guard that catches a mis-typed cell
    # depends on knowing what is actually held.
    acct_state = account_state(client, log=_log)
    if acct_state:
        _log(f"account: {len(acct_state['positions'])} position(s), "
             f"cash known for {len(acct_state['cash'])} account(s)")
    else:
        _log("⚠️  account state unknown — every row will be held, not acted on")

    if n_sub >= cfg.WARN_SUBMISSIONS_PER_DAY:
        _log(f"⚠️  {n_sub} order(s) already placed today — the cap is "
             f"{cfg.MAX_SUBMISSIONS_PER_DAY}. Normal days are well under this; "
             f"if you did not expect it, read {cfg.LEDGER}")
        audit(event="submission_warning", count=n_sub,
              cap=cfg.MAX_SUBMISSIONS_PER_DAY)

    after_close = args.force_time or \
        (_now().hour, _now().minute) >= cfg.EVALUATE_AFTER_PT
    if not after_close:
        _log("before 13:15 PT — validating rows only, not evaluating triggers")

    updates: list[dict] = []

    for rec in rows:
        rid = rec["Row_ID"] or f"(sheet row {rec['_sheet_row']})"

        prior = states.get(rec["Row_ID"], {})
        if prior.get("state") in cfg.TERMINAL_STATES:
            continue                       # history still sitting in the sheet

        def note_only(validation: str):
            """Feedback without a state change — the whole point of running
            often. A bad row says so within minutes of being typed instead of
            failing silently at the close."""
            updates.append(dict(row=rec["_sheet_row"], validation=validation))

        def finish(state: str, note: str, **extra):
            append_ledger(dict(row_id=rec["Row_ID"], fingerprint=fp_of(rec),
                               state=state, note=note, acct=rec["Acct"],
                               ticker=rec["Ticker"], side=rec["Side"],
                               close_is=rec["Close_Is"],
                               qty=rec["Qty"], trigger_price=rec["Trigger_Price"],
                               limit_price=rec["Limit_Price"], **extra))
            audit(row_id=rec["Row_ID"], state=state, note=note)
            updates.append(dict(row=rec["_sheet_row"], state=state, note=note))
            _log(f"   {rid:<26} {state:<10} {note}")

        # 1. Is it even a well-formed intent?
        try:
            n = oc.normalize(rec)
        except oc.IntentError as e:
            finish("VOID", f"MALFORMED: {e}")
            continue

        fp = oc.fingerprint(rec)

        # 2. Has a row we already acted on since been EDITED? Without a
        #    signature this is what stops an executed intent quietly becoming a
        #    second, different order under the same Row_ID.
        if prior and prior.get("state") in cfg.LIVE_STATES \
                and prior.get("fingerprint") and prior["fingerprint"] != fp:
            finish("VOID", "INTENT_CHANGED: this row was edited after the "
                           "engine acted on it. Use a new Row_ID rather than "
                           "editing a live row.")
            continue

        # 3. Expiry.
        why = expired(n)
        if why:
            finish("EXPIRED" if why.startswith("EXPIRED") else "VOID", why)
            continue

        # 4. Validate NOW, price-independently, and say so in the sheet. This
        #    is what replaced the signing key: a row that could never execute
        #    announces itself immediately rather than at the close.
        problem = check_guards(n, None, n_sub, notional_today, acct_state)
        if problem:
            note_only(f"⛔ {problem}")
            _log(f"   {rid:<26} WOULD BLOCK  {problem}")
            continue
        msg = f"✅ {oc.describe(rec)}"
        if not n["Limit_Price"] and n["Side"] == "BUY" and cfg.WARN_MARKET_BUY:
            # Not an error, but the one place a blank limit usually is a
            # mistake: an entry has no urgency, so chasing a gap up is all
            # cost and no benefit.
            msg += ("  ⚠️ MARKET BUY — will pay whatever it opens at. A gap up "
                    "means overpaying for a setup that no longer exists. "
                    "Consider a limit.")
            _log(f"   {rid:<26} ⚠️  market BUY with no limit price")
        note_only(msg)

        if not after_close:
            continue

        # 5. The close.
        close, note = daily_close(client, n["Ticker"])
        if close is None:
            # Leave it ARMED. Never advance a row on a price we could not read.
            _log(f"   {rid:<26} ARMED      no price: {note}")
            continue

        dis = cross_check(close, n["Ticker"])
        if dis:
            _log(f"   {rid:<26} ARMED      {dis}")
            continue

        if not triggered(n["Close_Is"], close, float(n["Trigger_Price"])):
            _log(f"   {rid:<26} ARMED      close {close:.2f} is not "
                 f"{n['Close_Is'].lower()} {float(n['Trigger_Price']):.2f}")
            note_only(f"⏳ waiting — close {close:.2f}, needs "
                      f"{n['Close_Is'].lower()} {float(n['Trigger_Price']):.2f}")
            continue

        # 6. Triggered. Every guard again, now with the close.
        block = check_guards(n, close, n_sub, notional_today, acct_state)
        if block:
            finish("BLOCKED", block, trigger_close=f"{close:.2f}")
            continue

        # 7. Write-ahead BEFORE the call, so a crash mid-submit is detectable
        #    rather than silently repeatable.
        idem = oc.idempotency_key(n["Row_ID"], fp)
        append_ledger(dict(row_id=n["Row_ID"], fingerprint=fp,
                           state="TRIGGERED", note=f"close {close:.2f}",
                           acct=n["Acct"], ticker=n["Ticker"], side=n["Side"],
                           close_is=n["Close_Is"],
                           qty=n["Qty"], trigger_price=n["Trigger_Price"],
                           limit_price=n["Limit_Price"],
                           trigger_close=f"{close:.2f}", idem_key=idem,
                           submit_attempted="1" if cfg.LIVE_TRADING else ""))

        if not cfg.LIVE_TRADING:
            finish("TRIGGERED",
                   f"DRY RUN — would submit at close {close:.2f}. "
                   f"Set ORDER_ENGINE_LIVE=1 to arm for real.",
                   trigger_close=f"{close:.2f}", idem_key=idem)
            continue

        oid, snote = submit(client, n, idem)
        if oid:
            n_sub += 1
            if n_sub >= cfg.WARN_SUBMISSIONS_PER_DAY:
                _log(f"⚠️  that was order {n_sub} of a possible "
                     f"{cfg.MAX_SUBMISSIONS_PER_DAY} today")
            notional_today += float(n["Qty"]) * float(n["Limit_Price"] or close)
            finish("SUBMITTED", snote, trigger_close=f"{close:.2f}",
                   idem_key=idem, schwab_order_id=oid, submit_attempted="1")
        else:
            finish("BLOCKED", snote, trigger_close=f"{close:.2f}", idem_key=idem)

    # Mirror state back into the engine columns. Best effort: the ledger is the
    # record, the sheet is a view of it.
    try:
        payload = []
        stamp = _now().strftime("%Y-%m-%d %H:%M:%S %Z")
        for u in updates:
            r = u["row"]
            if "validation" in u:
                payload.append({"range": f"{COL_VALIDATION}{r}",
                                "values": [[u["validation"][:400]]]})
            if "state" in u:
                payload.append({"range": f"{COL_STATUS}{r}:{COL_STATUS_DATE}{r}",
                                "values": [[u["state"], stamp]]})
                payload.append({"range": f"{COL_ENGINE_NOTE}{r}:{COL_LAST_CHECKED}{r}",
                                "values": [[u["note"][:400], stamp]]})
            payload.append({"range": f"{COL_LAST_CHECKED}{r}", "values": [[stamp]]})
        if payload:
            ws.batch_update(payload, value_input_option="RAW")
    except Exception as e:
        _log(f"⚠️  sheet write-back failed (ledger is still correct): {e}")

    _log(f"done — {len(updates)} row(s) changed state")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
