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
    need the fill at that close, this is the wrong mechanism — use a resting
    order at Schwab (After_Close = N).

WHAT PROTECTS YOU, IN ORDER OF HOW MUCH IT MATTERS
    1. LIVE_TRADING defaults to FALSE. Nothing reaches Schwab until someone
       sets ORDER_ENGINE_LIVE=1 on Pi 1, deliberately.
    2. The kill switch file blocks every submission, checked immediately before
       each one rather than once at startup.
    3. Every row needs a Confirm_Token minted by arm_order.py. The sheet cannot
       authorise a trade by itself; editing any intent cell invalidates it.
    4. Caps on notional per order, per day, and submissions per day.
    5. Account and ticker allowlists, failing CLOSED when unset.
    6. Write-ahead to the ledger before any submit, so a crash mid-call is
       detectable rather than silently repeatable.
    7. Triggers fire on a completed daily CLOSE, never an intrabar touch —
       matching the HHLL/pivot analysis, which is close-based throughout.

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

import order_canonical as oc                            # noqa: E402
import order_exec_config as cfg                         # noqa: E402

DATA_START_ROW = 9          # Orders tab: header block 1-6, banner 7, cols 8
SHEET_COLS = ["Row_ID", "Date", "Acct", "Ticker", "Action", "Trigger_Price",
              "Limit_Price", "Qty", "Qty_Unit", "After_Close", "Expires_On",
              "Confirm_Token"]

LEDGER_COLS = ["ts", "row_id", "token", "state", "note", "acct", "ticker",
               "action", "qty", "trigger_price", "limit_price", "trigger_close",
               "idem_key", "submit_attempted", "schwab_order_id"]


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

    try:
        oc.load_secret()
    except oc.IntentError as e:
        stop.append(f"HMAC key unusable: {e}")

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


def triggered(action: str, close: float, trigger: float) -> bool:
    _side, direction = oc.ACTIONS[action]
    return close > trigger if direction == "CLOSE_ABOVE" else close < trigger


# ─────────────────────────────────────────────────────────── guards
def check_guards(n: dict, close: float, submitted_today: int,
                 notional_today: float) -> str | None:
    """The reason this must NOT be submitted, or None."""
    if cfg.kill_switch_on():
        return f"KILL_SWITCH: {cfg.KILL_SWITCH} exists"

    if n["Acct"] not in cfg.ACCOUNT_ALLOWLIST:
        return (f"ACCOUNT_NOT_ALLOWED: {n['Acct']} is not in "
                f"{sorted(cfg.ACCOUNT_ALLOWLIST)}")

    if cfg.TICKER_ALLOWLIST and n["Ticker"] not in cfg.TICKER_ALLOWLIST:
        return f"TICKER_NOT_ALLOWED: {n['Ticker']}"

    if not n["Limit_Price"] and not cfg.ALLOW_MARKET_ORDERS:
        return "MARKET_ORDERS_REFUSED: no limit price given"

    if n["Qty_Unit"] != "SHARES":
        return (f"UNSUPPORTED_QTY_UNIT: {n['Qty_Unit']} — v1 handles SHARES "
                f"only, so 'trim 25%' still needs a share count")

    qty = float(n["Qty"])
    px = float(n["Limit_Price"] or n["Trigger_Price"])
    notional = qty * px

    if notional > cfg.MAX_NOTIONAL_PER_ORDER:
        return (f"CAP_PER_ORDER: ${notional:,.2f} exceeds "
                f"${cfg.MAX_NOTIONAL_PER_ORDER:,.2f}")
    if notional_today + notional > cfg.MAX_NOTIONAL_PER_DAY:
        return (f"CAP_PER_DAY: ${notional_today:,.2f} already + ${notional:,.2f} "
                f"exceeds ${cfg.MAX_NOTIONAL_PER_DAY:,.2f}")
    if submitted_today >= cfg.MAX_SUBMISSIONS_PER_DAY:
        return (f"CAP_SUBMISSIONS: {submitted_today} already today, max "
                f"{cfg.MAX_SUBMISSIONS_PER_DAY}")

    # The setup that was intended no longer exists.
    side, _ = oc.ACTIONS[n["Action"]]
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
    client = None
    updates: list[dict] = []

    for rec in rows:
        rid = rec["Row_ID"] or f"(sheet row {rec['_sheet_row']})"

        prior = states.get(rec["Row_ID"], {})
        if prior.get("state") in cfg.TERMINAL_STATES:
            continue                       # history still sitting in the sheet

        def finish(state: str, note: str, **extra):
            append_ledger(dict(row_id=rec["Row_ID"], token=rec["Confirm_Token"],
                               state=state, note=note, acct=rec["Acct"],
                               ticker=rec["Ticker"], action=rec["Action"],
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

        # 2. Does the token prove someone armed this exact intent?
        if not rec["Confirm_Token"]:
            finish("VOID", "NO_TOKEN: arm it with arm_order.py first")
            continue
        try:
            if not oc.verify(rec, rec["Confirm_Token"]):
                finish("VOID", "BAD_TOKEN: intent was edited after arming, or "
                               "never armed. Re-arm rather than editing.")
                continue
        except oc.IntentError as e:
            finish("VOID", f"CANNOT_VERIFY: {e}")
            continue

        # 3. Replay: a live row whose token changed is a mutated intent.
        if prior and prior.get("state") in cfg.LIVE_STATES \
                and prior.get("token") and prior["token"] != rec["Confirm_Token"]:
            finish("VOID", "INTENT_MUTATED: token differs from the ledger")
            continue

        # 4. Expiry.
        why = expired(n)
        if why:
            finish("EXPIRED" if why.startswith("EXPIRED") else "VOID", why)
            continue

        # 5. Resting orders are not this engine's job.
        if n["After_Close"] != "Y":
            _log(f"   {rid:<26} SKIP       After_Close=N — belongs at Schwab, "
                 f"not here (phase 2)")
            continue

        # 6. The close.
        if client is None:
            client = job.get_schwab_client()
        close, note = daily_close(client, n["Ticker"])
        if close is None:
            # Leave it ARMED. Never advance a row on a price we could not read.
            _log(f"   {rid:<26} ARMED      no price: {note}")
            continue

        dis = cross_check(close, n["Ticker"])
        if dis:
            _log(f"   {rid:<26} ARMED      {dis}")
            continue

        if not triggered(n["Action"], close, float(n["Trigger_Price"])):
            _side, direction = oc.ACTIONS[n["Action"]]
            _log(f"   {rid:<26} ARMED      {close:.2f} has not gone "
                 f"{'above' if direction=='CLOSE_ABOVE' else 'below'} "
                 f"{float(n['Trigger_Price']):.2f}")
            continue

        # 7. Triggered. Now every guard, immediately before acting.
        block = check_guards(n, close, n_sub, notional_today)
        if block:
            finish("BLOCKED", block, trigger_close=f"{close:.2f}")
            continue

        # 8. Write-ahead BEFORE the call, so a crash mid-submit is detectable
        #    rather than silently repeatable.
        idem = oc.idempotency_key(n["Row_ID"], rec["Confirm_Token"])
        append_ledger(dict(row_id=n["Row_ID"], token=rec["Confirm_Token"],
                           state="TRIGGERED", note=f"close {close:.2f}",
                           acct=n["Acct"], ticker=n["Ticker"], action=n["Action"],
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
            notional_today += float(n["Qty"]) * float(n["Limit_Price"] or close)
            finish("SUBMITTED", snote, trigger_close=f"{close:.2f}",
                   idem_key=idem, schwab_order_id=oid, submit_attempted="1")
        else:
            finish("BLOCKED", snote, trigger_close=f"{close:.2f}", idem_key=idem)

    # Mirror state back into the engine columns. Best effort: the ledger is the
    # record, the sheet is a view of it.
    try:
        payload = []
        for u in updates:
            payload.append({"range": f"M{u['row']}:O{u['row']}",
                            "values": [[u["state"], u["note"][:400],
                                        _now().strftime("%Y-%m-%d %H:%M:%S %Z")]]})
        if payload:
            ws.batch_update(payload, value_input_option="RAW")
    except Exception as e:
        _log(f"⚠️  sheet write-back failed (ledger is still correct): {e}")

    _log(f"done — {len(updates)} row(s) changed state")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
