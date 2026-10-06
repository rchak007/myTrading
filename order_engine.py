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
from datetime import date, datetime, timedelta, timezone
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
COL_STATUS = "L"
COL_STATUS_DATE = "M"
COL_VALIDATION = "N"
COL_ENGINE_NOTE = "T"
COL_LAST_CHECKED = "U"
SHEET_COLS = ["Row_ID", "Date", "Acct", "Ticker", "Side", "Close_Is",
              "Trigger_Price", "Limit_Price", "Qty", "Expires_On", "Notes"]
COL_ROW_ID = "A"

LEDGER_COLS = ["ts", "row_id", "fingerprint", "state", "note", "acct", "ticker",
               "side", "close_is", "qty", "trigger_price", "limit_price",
               "trigger_close", "idem_key", "submit_attempted", "schwab_order_id",
               # Set when submit() returned a failure it is CERTAIN about —
               # the order never reached Schwab. Without it the write-ahead
               # marker alone makes every clean rejection permanent.
               "submit_cleared"]


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


def orphaned_attempts(ledger: list[dict]) -> list[dict]:
    """Write-ahead rows that never got an outcome written after them.

    A run that dies BETWEEN the write-ahead and finish() leaves exactly this:
    state TRIGGERED, submit_attempted=1, no order id, and no later row. The
    order may or may not have reached Schwab, and nothing here can tell which
    — only the broker knows.

    Found on LITE 2026-10-06: the engine reported "already submitted as (id
    unknown)" every cycle afterwards, which is true but reads like routine
    idempotency rather than "a run died and you must go and look". Surfacing
    it is the whole difference between a known unknown and an invisible one.
    """
    last, out = {}, []
    for r in ledger:
        if r.get("row_id"):
            last[r["row_id"]] = r
    for rid, r in last.items():
        if (r.get("submit_attempted") and not r.get("schwab_order_id")
                and r.get("state") not in ("SUBMITTED", "BLOCKED", "CANCELLED",
                                           "EXPIRED", "VOID", "FILLED")
                and not r.get("submit_cleared")):
            out.append(r)
    return out


def submitted_before(ledger: list[dict], row_id: str) -> dict | None:
    """The ledger row proving this intent already went out, or None.

    THREE STATES, not two. A confirmed submission has an order id or state
    SUBMITTED and must never be resent. An attempt whose outcome is UNKNOWN
    must also never be resent — that is what the write-ahead marker is for.
    But an attempt that failed CERTAINLY, before anything was sent, has to be
    retryable, or one bad limit price retires the row forever.

    Scanned newest-first so the most recent outcome decides: a cleared failure
    after a genuine submission would be a different row_id entirely.
    """
    for r in reversed(ledger):
        if r.get("row_id") != row_id:
            continue
        if r.get("state") == "SUBMITTED" or r.get("schwab_order_id"):
            return r
        if r.get("submit_cleared"):
            return None            # we know it never left the building
        if r.get("submit_attempted"):
            return r               # in flight or ambiguous — do not resend
    return None


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

    # NO TIME CHECK HERE. Being before 13:15 is not a reason to abandon the
    # run — it is a reason not to evaluate TRIGGERS, which the after_close gate
    # in main() handles. Putting it here aborted the whole cycle, so the
    # validation feedback loop never ran during the trading day: rows sat
    # unchecked all morning and only got looked at once. Preflight is for
    # conditions that make running UNSAFE, and the clock is not one.

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
# A bar older than this many TRADING days means the data source has stopped
# updating, not that the market was shut. Two allows for running before today's
# bar has settled while still catching a feed that died last week.
MAX_STALE_TRADING_DAYS = 2


def _trading_days_between(a: date, b: date) -> int:
    """Weekdays strictly after `a`, up to and including `b`.

    Holidays are not modelled — overstating staleness by a day around a
    Thanksgiving is harmless, and a holiday calendar is a dependency this does
    not need.
    """
    if b <= a:
        return 0
    n, cur = 0, a
    while cur < b:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            n += 1
    return n


def daily_close(client_wrapper, ticker: str, log=print) -> tuple[float | None, str]:
    """(close, note) for the most recent COMPLETED daily bar.

    Schwab, not yfinance: the hotspot blocks Cloudflare and yfinance goes down
    with it. A failed fetch must leave the row ARMED — never advance a row on
    a price we could not read.
    """
    inner = _unwrap(client_wrapper)

    meth = getattr(inner, "price_history", None)
    if not callable(meth):
        return None, "this schwabdev exposes no price_history()"

    # endDate IS REQUIRED to get TODAY's bar. Measured 2026-09-28 with
    # probe_price_today.py: the same call without it returned 23 candles
    # ending FRIDAY, omitting a Monday close that had been final for eight
    # hours. A relative period alone means completed days only.
    #
    #   month/1 daily                 newest 2026-09-25  <- yesterday's world
    #   month/1 + explicit endDate    newest 2026-09-28  <- correct
    #
    # This is the difference between a close trigger acting tonight and acting
    # a day late, so it is not optional and must not be "simplified" away.
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    try:
        resp = meth(ticker, periodType="month", period=1,
                    frequencyType="daily", frequency=1, endDate=end_ms)
        if getattr(resp, "status_code", 200) != 200:
            return None, (f"price history HTTP {resp.status_code}: "
                          f"{str(getattr(resp, 'text', ''))[:120]}")
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


def stamp_row_ids(rows: list[dict], used: set | None = None) -> list[dict]:
    """Give every un-stamped row a permanent Row_ID.

    You leave column A blank; Pi 1 fills it once and never changes it. That id
    — not the row's position — is the row's identity everywhere else, which is
    what makes inserting new rows at the TOP safe. Sheet positions shift; ids
    do not.

    Deterministic: the same blank row regenerates the same id next cycle, so a
    failed write-back does not produce a duplicate under a second name.
    """
    # Ids already in the LEDGER count as taken, not just those in the sheet.
    # Otherwise clearing column A regenerates the same -01 the ledger has
    # already marked terminal, and the "new" row is skipped as history — which
    # is exactly what happened on 2026-09-28 and read as the engine ignoring a
    # live order. A Row_ID is spent for good once used.
    taken = {r["Row_ID"] for r in rows if r["Row_ID"]} | set(used or ())
    for rec in rows:
        if rec["Row_ID"] or not rec.get("Ticker"):
            continue
        day = (rec.get("Date") or date.today().isoformat())[:10]
        try:                                    # tolerate 9/26/2026 etc.
            day = datetime.fromisoformat(day).date().isoformat()
        except ValueError:
            day = date.today().isoformat()
        base = f"{day}-{rec['Ticker'].strip().upper()}"
        for i in range(1, 100):
            cand = f"{base}-{i:02d}"
            if cand not in taken:
                rec["Row_ID"] = cand
                rec["_new_id"] = True
                taken.add(cand)
                break
    return rows


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


_INNER: dict = {}


def _unwrap(client_wrapper):
    """The schwabdev client behind our wrapper, resolved ONCE per run.

    SchwabClient.get_client() builds a fresh client every call, each with its
    own auth round trip. A single cycle was creating five and spending most of
    its wall time doing it.
    """
    key = id(client_wrapper)
    if key not in _INNER:
        inner = client_wrapper
        getter = getattr(client_wrapper, "get_client", None)
        if callable(getter):
            try:
                inner = getter() or client_wrapper
            except Exception:
                inner = client_wrapper
        _INNER[key] = inner
    return _INNER[key]


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
    out = {"positions": {}, "cash": {}, "tickers": set(), "held": {}}
    try:
        from orders_sheet import fetch_positions_detailed
        pos = fetch_positions_detailed(client_wrapper, log=lambda *a: None)
        for _, r in pos.iterrows():
            key = (str(r["Acct"]), str(r["Ticker"]).upper())
            # SELLABLE, not Qty. Shares backing a short call are spoken for:
            # selling them turns a covered call naked, which in an IRA is not
            # a position that is allowed to exist. Falls back to Qty so an
            # older positions frame without the column still works.
            sellable = r["Sellable"] if "Sellable" in pos.columns else None
            out["positions"][key] = float(
                r["Qty"] if sellable is None or sellable != sellable else sellable)
            out["held"][key] = float(r["Qty"])
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
            # `held` is SELLABLE; `owned` is the real position. The two differ
            # by whatever backs a short call, and they are DIFFERENT mistakes:
            # more than you own is a typo, more than is free is a covered call
            # you forgot about. Checked in that order, or a typo gets blamed
            # on collateral.
            owned = (acct_state or {}).get("held", {}).get(
                (n["Acct"], n["Ticker"]), held)
            if qty > owned + 1e-6:
                return (f"OVERSELL: {qty:g} shares but only {owned:g} held in "
                        f"{n['Acct']} — check for a typo")
            if qty > held + 1e-6:
                return (f"COLLATERAL: {qty:g} shares but only {held:g} of "
                        f"{owned:g} are free in {n['Acct']} — "
                        f"{owned - held:g} back a short call")

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
    # GAPPED_THROUGH is deliberately NOT checked any more. It refused an order
    # whose limit sat away from the current price — which under DAY orders meant
    # "this can never fill". Under GOOD_TILL_CANCEL that is the normal case and
    # often the whole intent: "when TSLA closes below 400, rest a buy at 300"
    # is a coherent thing to want, and blocking it would be the engine second-
    # guessing a deliberate instruction. An unfillable-today GTC order simply
    # waits, which is what GTC is for.
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
def _num_or_none(v):
    try:
        return float(v) if str(v).strip() else None
    except (TypeError, ValueError):
        return None


def build_order_json(n: dict, limit_price: float | None = None) -> dict:
    """The Schwab order payload for one intent.

    Pure and testable — no client, no network — so the shape can be checked
    without risking a placement. Shape per Schwab's Trader API:

        limit_price      overrides the sheet's Limit_Price. Passed when the
                         sheet left it blank and the engine derived one from
                         the live book — see resolve_limit().

        orderType        LIMIT, always
        session          NORMAL (regular hours; a close trigger submits the
                         next session, so there is no reason to reach into
                         pre-market where spreads are worst)
        duration         GOOD_TILL_CANCEL
        instruction      BUY | SELL

    GTC AND LIMIT GO TOGETHER. A market order executes immediately, so
    "good till cancelled" has nothing to persist and brokers reject the
    combination. Chakravarti's call 2026-09-28: every order from this engine
    rests until filled or cancelled, which means every order needs a price.

    To exit REGARDLESS of price, do not reach for a market order — set the
    limit well THROUGH the market (a sell limit far below it). Such an order
    is marketable: it fills immediately at the best available bid, so you get
    the certainty of a market order AND a floor under a bad fill. That is
    strictly better than a market order, which has no floor at all.
    """
    qty = int(round(float(n["Qty"])))
    if qty <= 0:
        raise ValueError(f"quantity rounds to {qty}")
    px = limit_price or _num_or_none(n["Limit_Price"])
    if not px:
        raise ValueError("no limit price, and none could be derived from the "
                         "quote")

    return {
        "orderStrategyType": "SINGLE",
        "session": "NORMAL",
        "duration": "GOOD_TILL_CANCEL",
        "orderType": "LIMIT",
        "price": f"{px:.2f}",
        "orderLegCollection": [{
            "instruction": n["Side"],
            "quantity": qty,
            "instrument": {"symbol": n["Ticker"], "assetType": "EQUITY"},
        }],
    }


def _account_hash(client_wrapper, acct: str, log=print) -> str | None:
    """Schwab's opaque hash for an account. Order placement takes the hash,
    never the number."""
    inner = _unwrap(client_wrapper)
    try:
        for a in inner.linked_accounts().json() or []:
            if isinstance(a, dict) and str(a.get("accountNumber", ""))[-3:] == acct:
                return a.get("hashValue")
    except Exception as e:
        log(f"⚠️  could not resolve account hash: {type(e).__name__}: {e}")
    return None


def resolve_limit(client_wrapper, n: dict, log=print) -> tuple[float | None, str]:
    """The limit price to actually use.

    A price typed into the sheet wins — it is an explicit instruction. Left
    blank, the engine derives a MARKETABLE one from the live book at submit
    time: the ask for a buy, the bid for a sell, nudged through by a small
    buffer.

    Deriving beats typing for the usual case. A number typed days ago is stale
    by the time the trigger fires, and Chakravarti's point is the right one:
    "buy when it closes below 400" says nothing about what to pay, and the
    answer is simply whatever the market is asking when the order goes in.
    """
    typed = _num_or_none(n["Limit_Price"])
    if typed:
        return typed, f"limit {typed:.2f} as typed"

    try:
        from schwab_quotes import fetch_quotes, marketable_limit
        q = fetch_quotes(client_wrapper, [n["Ticker"]], log=lambda *a: None)
        entry = q.get(n["Ticker"]) or q.get(n["Ticker"].upper())
        if not entry:
            return None, f"no quote for {n['Ticker']} — cannot derive a limit"
        px, why = marketable_limit(entry, n["Side"], cfg.LIMIT_BUFFER_PCT)
        return px, why
    except Exception as e:
        return None, f"could not derive a limit: {type(e).__name__}: {e}"


def preview(client_wrapper, n: dict, log=print) -> tuple[bool, str]:
    """Ask Schwab to VALIDATE the order without placing it.

    preview_order is the real dry run. Checking the payload ourselves only
    proves we built what we intended; this proves Schwab accepts it — the
    symbol, the instruction, the quantity, the account. A payload that previews
    clean and then fails on placement is a much shorter list of possibilities.
    """
    inner = _unwrap(client_wrapper)

    meth = getattr(inner, "preview_order", None)
    if not callable(meth):
        return True, "no preview_order() on this client — payload unverified"

    h = _account_hash(client_wrapper, n["Acct"], log=log)
    if not h:
        return False, f"NO_ACCOUNT_HASH for {n['Acct']}"
    try:
        px, _why = resolve_limit(client_wrapper, n, log=log)
        body = build_order_json(n, limit_price=px)
        resp = meth(h, body)
    except Exception as e:
        return False, f"preview failed: {type(e).__name__}: {e}"

    code = getattr(resp, "status_code", None)
    if code not in (200, 201):
        return False, (f"Schwab REJECTED the preview: HTTP {code} "
                       f"{str(getattr(resp, 'text', ''))[:200]}")
    return True, "Schwab accepted the preview — the payload is valid"


def _acct_key(a) -> str:
    """Last 3 digits, matching orders_sheet.acct_key and cash_reserve."""
    t = str(a or "").strip().lstrip(".")
    return t[-3:] if len(t) >= 3 else t


def conflicting_sells(orders_df, acct: str, ticker: str, held: float,
                      want: float) -> tuple[list, str]:
    """Resting SELL orders that would starve this one of shares.

    Schwab reserves shares against an open sell order. Hold 23 with a trim
    resting for all 23, and a stop-triggered sell for 23 has nothing left to
    sell — the exit fails at exactly the moment it matters.

    Only a genuine shortfall counts. Hold 23, trim 10, sell 13 is fine and
    nothing is cancelled: the two orders coexist because the shares cover both.
    """
    if orders_df is None or getattr(orders_df, "empty", True) or not held:
        return [], ""
    m = ((orders_df["Ticker"].astype(str).str.upper() == ticker)
         & (orders_df["Side"].astype(str).str.upper().str.startswith("SELL")))
    if "Account" in orders_df.columns:
        m &= orders_df["Account"].map(_acct_key) == acct
    legs = orders_df[m]
    if legs.empty:
        return [], ""

    reserved = sum((_num_or_none(o.get("Remaining_QTY"))
                    or _num_or_none(o.get("QTY")) or 0.0)
                   for _, o in legs.iterrows())
    if reserved + want <= held + 1e-6:
        return [], (f"{reserved:g} sh already committed, {want:g} wanted, "
                    f"{held:g} held — both fit")

    out = []
    for _, o in legs.iterrows():
        oid = str(o.get("Order_ID") or "").strip()
        if not oid:
            continue
        out.append({"id": oid,
                    "qty": _num_or_none(o.get("Remaining_QTY")) or 0.0,
                    "price": (_num_or_none(o.get("Limit_Price"))
                              or _num_or_none(o.get("Stop_Price"))),
                    "cancelable": bool(o.get("Cancelable", True))})
    return out, (f"{reserved:g} sh committed to {len(out)} resting sell(s) + "
                 f"{want:g} wanted exceeds {held:g} held")


def cancel_order(client_wrapper, acct: str, order_id: str,
                 log=print) -> tuple[bool, str]:
    """Cancel one resting order. Returns (cancelled, note)."""
    inner = _unwrap(client_wrapper)
    meth = getattr(inner, "cancel_order", None)
    if not callable(meth):
        return False, "this schwabdev exposes no cancel_order()"
    h = _account_hash(client_wrapper, acct, log=log)
    if not h:
        return False, f"no account hash for {acct}"
    try:
        resp = meth(h, order_id)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    code = getattr(resp, "status_code", None)
    if code in (200, 201):
        return True, f"cancelled {order_id}"
    return False, f"HTTP {code} {str(getattr(resp, 'text', ''))[:120]}"


def submit(client_wrapper, n: dict, idem: str, log=print,
           conflicts: list | None = None) -> tuple[str | None, str, bool]:
    """Place the order. Returns (schwab_order_id, note, certain_not_placed).

    The third value is the one that matters when the first is None. Most
    failures happen BEFORE anything is sent — no limit price, a bad payload, a
    rejection, a cancel that would not clear — and those are safe to retry.
    Only an exception around the call itself is ambiguous, and only that must
    freeze the row for a human.

    NEVER RETRIES. If the outcome is ambiguous — a timeout, an unexpected
    status — this returns no id and the caller moves the row to BLOCKED for a
    human to resolve. Retrying a submit whose result is unknown is how one
    intent becomes two positions; the write-ahead ledger entry exists precisely
    so that state is recoverable by looking rather than by guessing.
    """
    inner = _unwrap(client_wrapper)

    # place_order(accountHash, order) — measured 2026-09-28 with
    # probe_order_api.py. It is NOT order_place; that guess cost a live run.
    place = getattr(inner, "place_order", None)
    if not callable(place):
        return None, True, ("NO_ORDER_API: this schwabdev exposes no place_order() — "
                      "run probe_order_api.py and report the method list")

    h = _account_hash(client_wrapper, n["Acct"], log=log)
    if not h:
        return None, True, f"NO_ACCOUNT_HASH for {n['Acct']} — is it linked to the app?"

    # CANCEL FIRST. Schwab reserves shares against a resting sell, so a trim
    # sitting on the whole position leaves a triggered exit with nothing to
    # sell — the order fails exactly when it is needed. A triggered exit
    # supersedes a profit target: getting out is the decision that was just
    # made, the trim is one made earlier under different conditions.
    if conflicts:
        log(f"   {len(conflicts)} resting sell(s) block this — cancelling first")
        for c in conflicts:
            if not c["cancelable"]:
                return None, True, (f"BLOCKED_BY_ORDER {c['id']} ({c['qty']:g} sh) "
                              f"is not cancelable — cancel it by hand")
            ok, note = cancel_order(client_wrapper, n["Acct"], c["id"], log=log)
            log(f"     {note}")
            if not ok:
                # Never place on top of an order we could not clear: the
                # rejection would be confusing and the position unchanged.
                return None, True, (f"CANCEL_FAILED for {c['id']}: {note}. "
                              f"Not placing on top of it.")

    px, why = resolve_limit(client_wrapper, n, log=log)
    if not px:
        return None, True, f"NO_LIMIT: {why}"
    try:
        body = build_order_json(n, limit_price=px)
    except ValueError as e:
        return None, True, f"BAD_ORDER: {e}"
    log(f"   limit: {why}")

    log(f"   submitting {n['Side']} {body['orderLegCollection'][0]['quantity']} "
        f"{n['Ticker']} {body['orderType']} "
        f"{body.get('price', 'at market')} in {n['Acct']}")

    try:
        resp = place(h, body)
    except Exception as e:
        return None, False, (f"SUBMIT_UNKNOWN: {type(e).__name__}: {e} — the order may "
                      f"or may not have reached Schwab. Check the account "
                      f"before doing anything with this row. idem={idem}")

    code = getattr(resp, "status_code", None)
    if code not in (200, 201):
        return None, True, (f"REJECTED: HTTP {code} "
                      f"{str(getattr(resp, 'text', ''))[:200]}")

    # Schwab returns the new order id in the Location header, not the body.
    oid = ""
    try:
        loc = (getattr(resp, "headers", {}) or {}).get("Location", "")
        oid = str(loc).rstrip("/").rsplit("/", 1)[-1] if loc else ""
    except Exception:
        pass
    if not oid:
        # Placed, but we cannot name it. Say so plainly rather than inventing
        # an id — reconciliation against open orders will adopt it.
        return "UNKNOWN", ("PLACED but no order id returned; reconcile against "
                           "open orders to attach one"), False
    return oid, f"placed, Schwab order {oid}", False


# ───────────────────────────────────────────────────────────── main
def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate close-triggered orders")
    ap.add_argument("--no-cancel", action="store_true",
                    help="never cancel a resting order; block instead")
    ap.add_argument("--status", action="store_true",
                    help="print configuration and ledger state, change nothing")
    ap.add_argument("--force-time", action="store_true",
                    help="evaluate triggers regardless of the clock, including "
                         "at a weekend (testing only)")
    args = ap.parse_args()

    print()
    print(cfg.summary())
    print()

    ledger = read_ledger()
    orders_df = None                 # lazily fetched by the SELL path below
    for o in orphaned_attempts(ledger):
        _log(f"🔴 ORPHANED SUBMIT  {o.get('row_id')}  {o.get('ticker')} "
             f"{o.get('side')} {o.get('qty')} in {o.get('acct')} at "
             f"{str(o.get('ts',''))[:19]} — a run died mid-submit. Only the "
             f"broker knows whether it landed. Check the account.")
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

    rows = stamp_row_ids(rows, used=set(states))
    fresh = [r for r in rows if r.get("_new_id")]
    if fresh:
        # Write the ids back FIRST and separately. Everything downstream keys
        # on Row_ID, so a row without one in the sheet would be re-stamped and
        # re-evaluated as if it were new on the next cycle.
        try:
            ws.batch_update([{"range": f"{COL_ROW_ID}{r['_sheet_row']}",
                              "values": [[r["Row_ID"]]]} for r in fresh],
                            value_input_option="RAW")
            _log(f"stamped {len(fresh)} new row(s) with a Row_ID")
        except Exception as e:
            _log(f"⛔ could not write Row_IDs ({e}) — stopping rather than "
                 f"acting on rows with no stable identity")
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

    now = _now()
    h, m = cfg.EVALUATE_AFTER_PT
    weekend = now.weekday() >= 5
    after_close = args.force_time or (
        not weekend and (now.hour, now.minute) >= (h, m))
    if not after_close:
        why = ("weekend — no daily bar to evaluate" if weekend
               else f"before {h:02d}:{m:02d} PT — the daily bar is not final")
        _log(f"{why}. Validating rows only; triggers are not evaluated.")

    updates: list[dict] = []

    for rec in rows:
        rid = rec["Row_ID"] or f"(sheet row {rec['_sheet_row']})"

        def note_only(validation: str):  # noqa: E306
            """Feedback without a state change — the whole point of running
            often. A bad row says so within minutes of being typed instead of
            failing silently at the close."""
            updates.append(dict(row_id=rec["Row_ID"], validation=validation))

        def finish(state: str, note: str, **extra):
            # Merge rather than splat: extra legitimately OVERRIDES a default
            # — a derived limit_price supersedes the blank one in the sheet —
            # and **extra alongside the same keyword is a TypeError, not an
            # override.
            row = dict(row_id=rec["Row_ID"], fingerprint=fp_of(rec),
                       state=state, note=note, acct=rec["Acct"],
                       ticker=rec["Ticker"], side=rec["Side"],
                       close_is=rec["Close_Is"], qty=rec["Qty"],
                       trigger_price=rec["Trigger_Price"],
                       limit_price=rec["Limit_Price"])
            row.update(extra)
            append_ledger(row)
            audit(row_id=rec["Row_ID"], state=state, note=note)
            updates.append(dict(row_id=rec["Row_ID"], state=state, note=note))
            _log(f"   {rid:<26} {state:<10} {note}")


        prior = states.get(rec["Row_ID"], {})

        # ALREADY SUBMITTED — never send it again. SUBMITTED is a live state
        # (the order can still be cancelled or rejected) so the row is still
        # re-read each cycle, but re-SUBMITTING it is a different thing
        # entirely: the trigger condition stays true after firing, so every
        # subsequent run placed the same order again.
        #
        # Observed 2026-09-29: one intent became THREE live TSLA orders at
        # 13:30, 13:45 and 14:05 — one per cron tick. The idempotency key was
        # being written to the ledger and never read back, which is the whole
        # reason it exists.
        done = submitted_before(ledger, rec["Row_ID"])
        if done:
            oid = done.get("schwab_order_id") or "(id unknown)"
            if (done.get("submit_attempted") and not done.get("schwab_order_id")
                    and done.get("state") not in ("SUBMITTED", "FILLED")):
                # A run died between the write-ahead and the outcome. Say that,
                # loudly — it is not routine idempotency, and only the broker
                # can resolve it.
                _log(f"🔴 {rid:<26} a previous run DIED MID-SUBMIT at "
                     f"{str(done.get('ts',''))[:19]}. The order may or may not "
                     f"have reached Schwab — CHECK THE ACCOUNT.")
                note_only(f"🔴 a run died mid-submit on "
                          f"{str(done.get('ts',''))[:19]} — the order may or "
                          f"may not have reached Schwab. CHECK THE ACCOUNT, "
                          f"then use a NEW row to order again.")
                continue
            _log(f"   {rid:<26} already submitted as {oid} — not resending")
            # Carry the ORIGINAL note. Replacing it with this one destroyed the
            # only record of why a submit failed — the sheet is often the first
            # and sometimes the only place that gets read.
            why = str(done.get("note", "")).strip()
            note_only(f"⏹ already submitted as {oid} on "
                      f"{str(done.get('ts',''))[:19]}. A Row_ID is placed ONCE; "
                      f"use a new row to order again."
                      + (f" [{why[:160]}]" if why else ""))
            continue

        if prior.get("state") in cfg.TERMINAL_STATES:
            # SAY SO. This exited silently and produced a run that read as
            # "nothing to do" when the row was in fact being ignored —
            # indistinguishable from a bug, and alarming when you are waiting
            # for an order. A terminal row is history, but the operator has no
            # way to know that without being told.
            _log(f"   {rid:<26} skipped — already {prior.get('state')} in the "
                 f"ledger. Give the row a NEW Row_ID to run it again.")
            note_only(f"⏹ already {prior.get('state')} — this Row_ID is spent. "
                      f"Clear column A AND change something, or use a new row.")
            continue

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
            # Validate against Schwab rather than merely asserting we would
            # have. A dry run that never touches the broker cannot tell you the
            # payload is wrong, which is the failure it most needs to catch.
            px, why = resolve_limit(client, n, log=_log)
            ok, pnote = preview(client, n, log=_log)
            finish("TRIGGERED" if ok else "BLOCKED",
                   f"DRY RUN at close {close:.2f} — would place GTC LIMIT "
                   f"{px if px else '??'} ({why}). {pnote}. "
                   f"Set ORDER_ENGINE_LIVE=1 to place for real.",
                   trigger_close=f"{close:.2f}", limit_price=px or "",
                   idem_key=idem)
            continue

        # Resting sells that would starve this one of shares.
        #
        # FETCHED LAZILY, ONCE. This used to reference an `orders_df` that was
        # never defined in main() — a NameError that crashed the engine the
        # FIRST time a SELL triggered, on 2026-10-06. It had gone unnoticed
        # because the only earlier trigger was a BUY, which never reaches this
        # branch. Fetching here rather than up front keeps the cost on the
        # path that actually needs it.
        conflicts, cnote = ([], "")
        if n["Side"] == "SELL" and not args.no_cancel:
            if orders_df is None:
                try:
                    from stocks_orders import build_orders_table
                    orders_df = build_orders_table(client, open_only=True,
                                                   restrict_to_tickers=False,
                                                   log=_log)
                    _log(f"   open orders at Schwab: "
                         f"{0 if orders_df is None else len(orders_df)}")
                except Exception as e:
                    # Fail CLOSED. Not knowing what rests at Schwab means not
                    # knowing whether the shares are free, and placing anyway
                    # is how an exit fails at the moment it matters.
                    finish("BLOCKED", f"NO_OPEN_ORDERS: cannot read resting "
                                      f"orders ({type(e).__name__}: {e}) — not "
                                      f"placing without knowing what they hold",
                           trigger_close=f"{close:.2f}", submit_cleared="1")
                    continue
            held = (acct_state or {}).get("positions", {}).get(
                (n["Acct"], n["Ticker"]), 0.0)
            conflicts, cnote = conflicting_sells(orders_df, n["Acct"],
                                                 n["Ticker"], held,
                                                 float(n["Qty"]))
            if cnote:
                _log(f"   {cnote}")

        oid, snote, certain_not_placed = submit(client, n, idem,
                                                conflicts=conflicts)
        if conflicts and oid:
            snote += (f" · cancelled {len(conflicts)} resting sell(s) first: "
                      + ", ".join(c["id"] for c in conflicts))
        if oid:
            n_sub += 1
            if n_sub >= cfg.WARN_SUBMISSIONS_PER_DAY:
                _log(f"⚠️  that was order {n_sub} of a possible "
                     f"{cfg.MAX_SUBMISSIONS_PER_DAY} today")
            notional_today += float(n["Qty"]) * float(n["Limit_Price"] or close)
            finish("SUBMITTED", snote, trigger_close=f"{close:.2f}",
                   idem_key=idem, schwab_order_id=oid, submit_attempted="1")
        else:
            # CLEAR THE WRITE-AHEAD when submit is certain nothing was sent.
            # The marker exists to catch a crash mid-call; treating a clean
            # rejection the same way made the row permanently unretryable and
            # then overwrote the real reason with "already submitted as (id
            # unknown)" on the next cycle. Observed on LITE, 2026-10-06.
            finish("BLOCKED", snote, trigger_close=f"{close:.2f}", idem_key=idem,
                   submit_cleared="1" if certain_not_placed else "")
            if certain_not_placed:
                _log(f"   nothing reached Schwab — this row can trigger again")

    # Mirror state back into the engine columns. Best effort: the ledger is the
    # record, the sheet is a view of it.
    try:
        # Re-read column A and map id -> CURRENT row. Rows may have been
        # inserted above these while the cycle ran — the whole point of adding
        # new work at the top — so a position cached minutes ago would now
        # write into the wrong row.
        live = {}
        for i, raw in enumerate(ws.col_values(1), start=1):
            if i >= DATA_START_ROW and str(raw).strip():
                live[str(raw).strip()] = i

        payload = []
        stamp = _now().strftime("%Y-%m-%d %H:%M:%S %Z")
        for u in updates:
            r = live.get(u["row_id"])
            if r is None:
                _log(f"⚠️  {u['row_id']} vanished from the sheet mid-cycle; "
                     f"ledger is still correct")
                continue
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

    # Distinct rows, not update entries: one row produces both a validation
    # note and a state change, so the raw count read as two rows for one.
    touched = len({u["row_id"] for u in updates})
    _log(f"done — {touched} row(s) touched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
