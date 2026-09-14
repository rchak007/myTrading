#!/usr/bin/env python3
"""
cash_reserve.py
===============
Per-(Account, Ticker) cash fencing. "Seed" a ticker with a slice of an
account's cash, then manage that slice over its life so that a BUY signal can
only ever spend money that was explicitly earmarked for that ticker.

DESIGN — same injection contract as stocks_cash.py / stocks_orders.py. This
module owns *only* reserve accounting. Everything shared is injected:

    output paths   -> jobStocksSignals.OUT_RESERVES_CSV / OUT_RESERVES_HTML
    log()          -> jobStocksSignals.log
    HTML rendering -> jobStocksSignals.build_html_table
    Schwab client  -> jobStocksSignals.get_schwab_client()
    raw client     -> stocks_orders._raw_client
    nicknames      -> stocks_cash.ACCOUNT_LABELS

State files are the exception: they are machine-local and live under
~/.local/state/myTrading/ (override with MYTRADING_STATE_DIR). They are NOT
in the repo — reserve_ledger.csv is money history, it does not belong in git.

TWO FILES, TWO ROLES
    reserves_config.csv   declarative INTENT. Hand-edited (or sheet-synced).
                          What you *want* fenced. Safe to rewrite.
    reserve_ledger.csv    APPEND-ONLY history. What actually happened.
                          Never rewritten, never sorted, never de-duplicated
                          in place. Balances are always a fold of this file.

IDEMPOTENCY — the whole point of the SKIP rows
    Every fill-derived event carries Ref = the Schwab activityId. Before
    applying fills we build the set of Refs already present in the ledger
    (including SKIP rows). Anything already seen is dropped.

    A fill that *cannot* be applied — ticker not fenced, account not fenced,
    fill predates the seed's Effective_From — is written as SKIP_BUY /
    SKIP_SELL with Amount 0 and the same Ref. That does two jobs:
      1. it stops the next run retro-debiting a reserve for a trade that
         happened before the reserve existed;
      2. it stops the same "unfenced fill" nudge firing every 35 minutes
         forever.

POLICY MODES (see the Target_Capital note in CASH_RESERVE_HANDOFF.md)
    CASH_ONLY       Seed_Cash is cash committed to the ticker. The reserve
                    balance is dry powder: buys debit it, sells credit it.
                    Available_To_Buy = reserve balance.
    TOTAL_CAPITAL   Target_Capital is total capital allocated to the ticker
                    (cash + market value of the open position).
                    Available_To_Buy = Target_Capital - Position_Value,
                    capped by the reserve balance.

Importable:
    from cash_reserve import (
        build_reserves_table, write_reserve_outputs,
        available_to_buy, seed, topup, withdraw, apply_fills,
    )

Standalone (resolves paths/log/client by loading jobStocksSignals.py):
    python3 cash_reserve.py --list
    python3 cash_reserve.py --seed     --account 431 --ticker NVDA --amount 5000
    python3 cash_reserve.py --topup    --account 431 --ticker NVDA --amount 1000
    python3 cash_reserve.py --withdraw --account 431 --ticker NVDA --amount 500
    python3 cash_reserve.py --close    --account 431 --ticker NVDA
    python3 cash_reserve.py --apply-fills --days 7
    python3 cash_reserve.py --check    --account 431 --ticker NVDA --amount 2000
    python3 cash_reserve.py --reconcile
    python3 cash_reserve.py --write

MASKING: Account is stored and displayed as the last 3 digits only ("431").
Full account numbers never enter these files.
"""
from __future__ import annotations

import csv
import fcntl
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

try:
    import pytz
    _PST = pytz.timezone("America/Los_Angeles")
except Exception:                                    # pragma: no cover
    _PST = None

# ─────────────────────────────────────────────────────────────────────
# Paths — machine-local state, overridable for tests
# ─────────────────────────────────────────────────────────────────────
STATE_DIR = Path(
    os.environ.get("MYTRADING_STATE_DIR", Path.home() / ".local" / "state" / "myTrading")
)
LEDGER_PATH = STATE_DIR / "reserve_ledger.csv"
CONFIG_PATH = STATE_DIR / "reserves_config.csv"

# ─────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────
LEDGER_COLS = [
    "Timestamp_PST",   # when the row was written
    "Event_ID",        # uuid4[:12], row identity
    "Account",         # last 3 digits
    "Ticker",
    "Event",           # see EVENTS below
    "Amount",          # SIGNED. credit > 0, debit < 0, skip == 0
    "Balance_After",   # advisory snapshot. fold_balances() is authoritative.
    "Ref",             # idempotency key (Schwab activityId, or op:<uuid>)
    "Source",          # cli | job | sheet | manual
    "Reason",          # free text
]

CONFIG_COLS = [
    "Account", "Ticker", "Policy",
    "Seed_Cash", "Target_Capital", "Max_Order_Pct",
    "Effective_From", "Active", "Notes",
]

RESERVE_COLS = [
    "Account", "Nickname", "Ticker", "Policy",
    "Seed_Cash", "Target_Capital",
    "Reserved_Cash", "Deployed", "Returned",
    "Position_Value", "Available_To_Buy",
    "Status", "Last_Event_PST",
]

CREDIT_EVENTS = {"SEED", "TOPUP", "SELL_FILL"}
DEBIT_EVENTS = {"WITHDRAW", "BUY_FILL", "CLOSE"}
SKIP_EVENTS = {"SKIP_BUY", "SKIP_SELL"}
FREE_EVENTS = {"ADJUST"}                     # signed either way, audit trail
EVENTS = CREDIT_EVENTS | DEBIT_EVENTS | SKIP_EVENTS | FREE_EVENTS

POLICIES = {"CASH_ONLY", "TOTAL_CAPITAL"}
DEFAULT_POLICY = "CASH_ONLY"

# Tolerance for float noise when comparing advisory vs folded balances.
_EPS = 0.005


# ─────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────
def _now_pst() -> str:
    if _PST is not None:
        return datetime.now(_PST).strftime("%Y-%m-%d %H:%M:%S %Z")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def acct_key(acct) -> str:
    """Everything in this module keys on the last 3 digits of the account."""
    s = str(acct or "").strip().lstrip(".")
    return s[-3:] if len(s) >= 3 else s


def tkr_key(t) -> str:
    return str(t or "").strip().upper()


def _f(v, default=None):
    try:
        if v is None or v == "" or (isinstance(v, float) and pd.isna(v)):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _nickname(last3: str) -> str:
    try:
        from stocks_cash import ACCOUNT_LABELS
        return ACCOUNT_LABELS.get(last3, "")
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────
# Ledger I/O — append-only, flock-guarded, fsync'd
# ─────────────────────────────────────────────────────────────────────
def read_ledger(ledger_path: Path = LEDGER_PATH) -> pd.DataFrame:
    """The full ledger. Empty frame with the right columns if it does not exist."""
    if not Path(ledger_path).exists():
        return pd.DataFrame(columns=LEDGER_COLS)
    df = pd.read_csv(ledger_path, dtype=str).fillna("")
    for c in LEDGER_COLS:
        if c not in df.columns:
            df[c] = ""
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    df["Account"] = df["Account"].map(acct_key)
    df["Ticker"] = df["Ticker"].map(tkr_key)
    return df[LEDGER_COLS]


def append_events(rows: list[dict], ledger_path: Path = LEDGER_PATH, log=print) -> int:
    """
    Append validated event rows under an exclusive flock. Never rewrites.
    Returns the number of rows written.
    """
    rows = [r for r in rows if r]
    if not rows:
        return 0

    for r in rows:
        ev = str(r.get("Event", "")).upper()
        if ev not in EVENTS:
            raise ValueError(f"Unknown reserve event: {ev!r}")
        amt = _f(r.get("Amount"), 0.0) or 0.0
        if ev in CREDIT_EVENTS and amt < 0:
            raise ValueError(f"{ev} must be a credit (>= 0), got {amt}")
        if ev in DEBIT_EVENTS and amt > 0:
            raise ValueError(f"{ev} must be a debit (<= 0), got {amt}")
        if ev in SKIP_EVENTS and amt != 0:
            raise ValueError(f"{ev} must carry Amount 0, got {amt}")

    p = Path(ledger_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new_file = not p.exists() or p.stat().st_size == 0

    with open(p, "a", newline="", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            w = csv.DictWriter(fh, fieldnames=LEDGER_COLS, extrasaction="ignore")
            if new_file:
                w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in LEDGER_COLS})
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    log(f"reserve_ledger: +{len(rows)} row(s) -> {p}")
    return len(rows)


def _event_row(account, ticker, event, amount, *, ref, reason="",
               source="cli", balance_after=None) -> dict:
    return {
        "Timestamp_PST": _now_pst(),
        "Event_ID": uuid.uuid4().hex[:12],
        "Account": acct_key(account),
        "Ticker": tkr_key(ticker),
        "Event": str(event).upper(),
        "Amount": round(float(amount), 2),
        "Balance_After": "" if balance_after is None else round(float(balance_after), 2),
        "Ref": ref,
        "Source": source,
        "Reason": reason,
    }


# ─────────────────────────────────────────────────────────────────────
# Fold — the ledger is the authority, balances are always derived
# ─────────────────────────────────────────────────────────────────────
def fold_balances(ledger: pd.DataFrame | None = None,
                  ledger_path: Path = LEDGER_PATH) -> pd.DataFrame:
    """
    Per (Account, Ticker):
        Reserved_Cash   net balance = sum of all signed Amounts
        Seeded          SEED + TOPUP
        Withdrawn       WITHDRAW + CLOSE (positive magnitude)
        Deployed        BUY_FILL  (positive magnitude)
        Returned        SELL_FILL
        Last_Event_PST  timestamp of the last row for that pair
    """
    df = read_ledger(ledger_path) if ledger is None else ledger.copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "Account", "Ticker", "Reserved_Cash", "Seeded", "Withdrawn",
            "Deployed", "Returned", "Last_Event_PST",
        ])

    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    ev = df["Event"].astype(str).str.upper()

    def _sum_where(mask, magnitude=False):
        s = df["Amount"].where(mask, 0.0)
        return s.abs() if magnitude else s

    df["_bal"] = df["Amount"]
    df["_seeded"] = _sum_where(ev.isin(["SEED", "TOPUP"]))
    df["_withdrawn"] = _sum_where(ev.isin(["WITHDRAW", "CLOSE"]), magnitude=True)
    df["_deployed"] = _sum_where(ev.eq("BUY_FILL"), magnitude=True)
    df["_returned"] = _sum_where(ev.eq("SELL_FILL"))

    g = df.groupby(["Account", "Ticker"], sort=True)
    out = g.agg(
        Reserved_Cash=("_bal", "sum"),
        Seeded=("_seeded", "sum"),
        Withdrawn=("_withdrawn", "sum"),
        Deployed=("_deployed", "sum"),
        Returned=("_returned", "sum"),
        Last_Event_PST=("Timestamp_PST", "last"),
    ).reset_index()

    for c in ("Reserved_Cash", "Seeded", "Withdrawn", "Deployed", "Returned"):
        out[c] = out[c].astype(float).round(2)
    return out


def balance_of(account, ticker, ledger: pd.DataFrame | None = None,
               ledger_path: Path = LEDGER_PATH) -> float:
    a, t = acct_key(account), tkr_key(ticker)
    b = fold_balances(ledger, ledger_path)
    if b.empty:
        return 0.0
    hit = b[(b["Account"] == a) & (b["Ticker"] == t)]
    return 0.0 if hit.empty else float(hit["Reserved_Cash"].iloc[0])


def seen_refs(ledger: pd.DataFrame | None = None,
              ledger_path: Path = LEDGER_PATH) -> set[str]:
    """Every Ref already in the ledger — including SKIP rows. The replay guard."""
    df = read_ledger(ledger_path) if ledger is None else ledger
    if df.empty:
        return set()
    return {r for r in df["Ref"].astype(str) if r and r != "nan"}


# ─────────────────────────────────────────────────────────────────────
# Config — declarative intent
# ─────────────────────────────────────────────────────────────────────
def read_config(config_path: Path = CONFIG_PATH, log=print) -> pd.DataFrame:
    if not Path(config_path).exists():
        log(f"reserves_config.csv not found at {config_path} — no tickers fenced")
        return pd.DataFrame(columns=CONFIG_COLS)

    df = pd.read_csv(config_path, dtype=str).fillna("")
    for c in CONFIG_COLS:
        if c not in df.columns:
            df[c] = ""
    df["Account"] = df["Account"].map(acct_key)
    df["Ticker"] = df["Ticker"].map(tkr_key)
    df["Policy"] = (df["Policy"].astype(str).str.upper().str.strip()
                    .replace("", DEFAULT_POLICY))
    bad = ~df["Policy"].isin(POLICIES)
    if bad.any():
        log(f"⚠️  {int(bad.sum())} config row(s) with unknown Policy — forced to {DEFAULT_POLICY}")
        df.loc[bad, "Policy"] = DEFAULT_POLICY

    df["Active"] = (df["Active"].astype(str).str.strip().str.upper()
                    .isin(["", "Y", "YES", "TRUE", "1"]))

    dupes = df[df["Active"]].duplicated(subset=["Account", "Ticker"], keep=False)
    if dupes.any():
        log(f"⚠️  {int(dupes.sum())} duplicate active (Account, Ticker) rows in config — "
            "last one wins, fix the file")
        df = df.drop_duplicates(subset=["Account", "Ticker"], keep="last")

    return df[CONFIG_COLS]


def ensure_config_row(account, ticker, *, policy=DEFAULT_POLICY, seed_cash=0.0,
                      target_capital=None, config_path: Path = CONFIG_PATH,
                      log=print) -> None:
    """
    Upsert one (Account, Ticker) row. The config is intent, not history —
    rewriting it is safe. The ledger is what must never be rewritten.
    """
    a, t = acct_key(account), tkr_key(ticker)
    df = read_config(config_path, log)
    mask = (df["Account"] == a) & (df["Ticker"] == t)

    if mask.any():
        i = df.index[mask][0]
        df.at[i, "Policy"] = policy
        df.at[i, "Seed_Cash"] = f"{_f(df.at[i, 'Seed_Cash'], 0.0) + float(seed_cash):.2f}"
        if target_capital is not None:
            df.at[i, "Target_Capital"] = f"{float(target_capital):.2f}"
        df.at[i, "Active"] = "Y"
    else:
        df = pd.concat([df, pd.DataFrame([{
            "Account": a, "Ticker": t, "Policy": policy,
            "Seed_Cash": f"{float(seed_cash):.2f}",
            "Target_Capital": "" if target_capital is None else f"{float(target_capital):.2f}",
            "Max_Order_Pct": "", "Effective_From": _now_pst(),
            "Active": "Y", "Notes": "",
        }])], ignore_index=True)

    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(config_path) + ".tmp")
    df[CONFIG_COLS].to_csv(tmp, index=False)
    os.replace(tmp, config_path)                    # atomic
    log(f"reserves_config: upserted {a}/{t}")


# ─────────────────────────────────────────────────────────────────────
# Management verbs
# ─────────────────────────────────────────────────────────────────────
def seed(account, ticker, amount, *, policy=DEFAULT_POLICY, target_capital=None,
         reason="", source="cli", ledger_path=LEDGER_PATH,
         config_path=CONFIG_PATH, log=print) -> float:
    """Fence `amount` of an account's cash to a ticker. Credits the reserve."""
    amount = float(amount)
    if amount <= 0:
        raise ValueError("seed amount must be positive")
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {sorted(POLICIES)}")

    ensure_config_row(account, ticker, policy=policy, seed_cash=amount,
                      target_capital=target_capital, config_path=config_path, log=log)

    bal = balance_of(account, ticker, ledger_path=ledger_path) + amount
    append_events([_event_row(account, ticker, "SEED", amount,
                              ref=f"op:{uuid.uuid4().hex[:12]}",
                              reason=reason or "initial seed",
                              source=source, balance_after=bal)],
                  ledger_path, log)
    log(f"SEED {acct_key(account)}/{tkr_key(ticker)} +${amount:,.2f} -> ${bal:,.2f}")
    return round(bal, 2)


def topup(account, ticker, amount, *, reason="", source="cli",
          ledger_path=LEDGER_PATH, config_path=CONFIG_PATH, log=print) -> float:
    amount = float(amount)
    if amount <= 0:
        raise ValueError("topup amount must be positive")
    ensure_config_row(account, ticker, seed_cash=amount, config_path=config_path, log=log)
    bal = balance_of(account, ticker, ledger_path=ledger_path) + amount
    append_events([_event_row(account, ticker, "TOPUP", amount,
                              ref=f"op:{uuid.uuid4().hex[:12]}",
                              reason=reason or "top-up", source=source,
                              balance_after=bal)], ledger_path, log)
    log(f"TOPUP {acct_key(account)}/{tkr_key(ticker)} +${amount:,.2f} -> ${bal:,.2f}")
    return round(bal, 2)


def withdraw(account, ticker, amount, *, reason="", source="cli",
             allow_negative=False, ledger_path=LEDGER_PATH, log=print) -> float:
    """Release fenced cash back to the account's free pool."""
    amount = abs(float(amount))
    cur = balance_of(account, ticker, ledger_path=ledger_path)
    if amount > cur and not allow_negative:
        raise ValueError(
            f"cannot withdraw ${amount:,.2f} — {acct_key(account)}/{tkr_key(ticker)} "
            f"holds ${cur:,.2f}. Pass allow_negative=True to force."
        )
    bal = cur - amount
    append_events([_event_row(account, ticker, "WITHDRAW", -amount,
                              ref=f"op:{uuid.uuid4().hex[:12]}",
                              reason=reason or "withdraw", source=source,
                              balance_after=bal)], ledger_path, log)
    log(f"WITHDRAW {acct_key(account)}/{tkr_key(ticker)} -${amount:,.2f} -> ${bal:,.2f}")
    return round(bal, 2)


def close(account, ticker, *, reason="", source="cli",
          ledger_path=LEDGER_PATH, config_path=CONFIG_PATH, log=print) -> float:
    """Zero the reserve and deactivate the config row. History is retained."""
    cur = balance_of(account, ticker, ledger_path=ledger_path)
    if abs(cur) > _EPS:
        append_events([_event_row(account, ticker, "CLOSE", -cur,
                                  ref=f"op:{uuid.uuid4().hex[:12]}",
                                  reason=reason or "close reserve", source=source,
                                  balance_after=0.0)], ledger_path, log)

    df = read_config(config_path, log)
    m = (df["Account"] == acct_key(account)) & (df["Ticker"] == tkr_key(ticker))
    if m.any():
        df.loc[m, "Active"] = "N"
        tmp = Path(str(config_path) + ".tmp")
        df[CONFIG_COLS].to_csv(tmp, index=False)
        os.replace(tmp, config_path)

    log(f"CLOSE {acct_key(account)}/{tkr_key(ticker)} released ${cur:,.2f}")
    return 0.0


# ─────────────────────────────────────────────────────────────────────
# The gate the order engine calls
# ─────────────────────────────────────────────────────────────────────
def available_to_buy(account, ticker, *, positions_df=None,
                     config: pd.DataFrame | None = None,
                     ledger_path=LEDGER_PATH, config_path=CONFIG_PATH,
                     log=print) -> dict:
    """
    How much may be spent on this ticker right now.

    Returns {"allowed": bool, "available": float, "policy": str, "reason": str}.

    Fail-closed: an unfenced (Account, Ticker) returns allowed=False. A ticker
    with no reserve row is a ticker with no permission to spend.
    """
    a, t = acct_key(account), tkr_key(ticker)
    cfg = read_config(config_path, log) if config is None else config
    row = cfg[(cfg["Account"] == a) & (cfg["Ticker"] == t)]

    if row.empty:
        return {"allowed": False, "available": 0.0, "policy": "",
                "reason": f"{a}/{t} is not fenced in reserves_config.csv"}
    row = row.iloc[0]
    if not bool(row["Active"]):
        return {"allowed": False, "available": 0.0, "policy": row["Policy"],
                "reason": f"{a}/{t} reserve is inactive"}

    bal = balance_of(a, t, ledger_path=ledger_path)
    policy = row["Policy"]

    if policy == "TOTAL_CAPITAL":
        target = _f(row["Target_Capital"])
        if target is None:
            return {"allowed": False, "available": 0.0, "policy": policy,
                    "reason": f"{a}/{t} is TOTAL_CAPITAL but Target_Capital is blank"}
        pos_val = position_value(a, t, positions_df)
        headroom = max(0.0, target - pos_val)
        avail = min(headroom, max(0.0, bal))
        reason = (f"target ${target:,.2f} - position ${pos_val:,.2f} "
                  f"= ${headroom:,.2f}, capped by reserve ${bal:,.2f}")
    else:
        avail = max(0.0, bal)
        reason = f"reserve balance ${bal:,.2f}"

    pct = _f(row["Max_Order_Pct"])
    if pct and pct > 0:
        cap = avail * (pct / 100.0)
        reason += f", per-order cap {pct:g}% = ${cap:,.2f}"
        avail = cap

    return {"allowed": avail > 0, "available": round(avail, 2),
            "policy": policy, "reason": reason}


def position_value(account, ticker, positions_df=None) -> float:
    """
    Market value of the open position, for TOTAL_CAPITAL mode.
    positions_df is expected to carry Account, Ticker and VALUE (the column
    jobStocksSignals already builds). Absent frame -> 0.0.
    """
    if positions_df is None or getattr(positions_df, "empty", True):
        return 0.0
    df = positions_df
    if "Ticker" not in df.columns:
        return 0.0
    m = df["Ticker"].map(tkr_key) == tkr_key(ticker)
    if "Account" in df.columns:
        m &= df["Account"].map(acct_key) == acct_key(account)
    val_col = next((c for c in ("VALUE", "Value", "Market_Value") if c in df.columns), None)
    if val_col is None:
        return 0.0
    return float(pd.to_numeric(df.loc[m, val_col], errors="coerce").fillna(0.0).sum())


# ─────────────────────────────────────────────────────────────────────
# Fills -> ledger events (with the SKIP guard)
# ─────────────────────────────────────────────────────────────────────
def apply_fills(fills_df, *, config: pd.DataFrame | None = None,
                ledger_path=LEDGER_PATH, config_path=CONFIG_PATH,
                dry_run=False, log=print) -> pd.DataFrame:
    """
    Turn executed fills into BUY_FILL / SELL_FILL / SKIP_* ledger events.

    fills_df columns (missing ones are tolerated):
        Ref          Schwab activityId — REQUIRED, the idempotency key
        Account      any form, keyed to last 3
        Ticker
        Side         BUY* / SELL*
        Fill_QTY, Fill_Price   used when Net_Amount is absent
        Net_Amount   Schwab netAmount, preferred when present
        Asset_Type   OPTION -> x100 multiplier on qty*price
        Fill_Time    ISO/parseable; compared against Effective_From

    Returns the frame of events that were (or would be) appended.
    """
    empty = pd.DataFrame(columns=LEDGER_COLS)
    if fills_df is None or getattr(fills_df, "empty", True):
        log("No fills to apply.")
        return empty

    cfg = read_config(config_path, log) if config is None else config
    known = seen_refs(ledger_path=ledger_path)

    fills = fills_df.copy()
    if "Ref" not in fills.columns:
        log("⚠️  fills frame has no Ref column — refusing to apply (would double-count)")
        return empty

    fills["Ref"] = fills["Ref"].astype(str).str.strip()
    fills = fills[fills["Ref"].ne("") & fills["Ref"].ne("nan")]

    before = len(fills)
    fills = fills[~fills["Ref"].isin(known)]
    if before != len(fills):
        log(f"Replay guard: {before - len(fills)} already-processed fill(s) dropped")
    if fills.empty:
        log("All fills already in the ledger — nothing to do.")
        return empty

    rows: list[dict] = []
    running: dict[tuple[str, str], float] = {}

    for _, f in fills.iterrows():
        a, t = acct_key(f.get("Account")), tkr_key(f.get("Ticker"))
        side = str(f.get("Side", "")).upper().strip()
        is_buy = side.startswith("BUY")
        is_sell = side.startswith("SELL")
        ref = str(f["Ref"])

        if not (is_buy or is_sell):
            log(f"⚠️  fill {ref}: unrecognised Side {side!r} — ignored, not ledgered")
            continue

        amt = _f(f.get("Net_Amount"))
        if amt is None:
            qty = _f(f.get("Fill_QTY"), 0.0) or 0.0
            px = _f(f.get("Fill_Price"), 0.0) or 0.0
            mult = 100.0 if str(f.get("Asset_Type", "")).upper() == "OPTION" else 1.0
            amt = qty * px * mult
        amt = abs(float(amt))

        cfg_row = cfg[(cfg["Account"] == a) & (cfg["Ticker"] == t)]
        skip_reason = ""
        if cfg_row.empty:
            skip_reason = f"{a}/{t} not fenced"
        elif not bool(cfg_row.iloc[0]["Active"]):
            skip_reason = f"{a}/{t} reserve inactive"
        else:
            eff = str(cfg_row.iloc[0]["Effective_From"] or "").strip()
            ft = str(f.get("Fill_Time", "") or "").strip()
            if eff and ft:
                try:
                    if pd.to_datetime(ft, errors="raise", utc=True) < \
                       pd.to_datetime(eff, errors="raise", utc=True):
                        skip_reason = f"fill predates Effective_From ({eff})"
                except Exception:
                    pass                       # unparseable -> do not skip on it

        if skip_reason:
            rows.append(_event_row(a, t, "SKIP_BUY" if is_buy else "SKIP_SELL", 0.0,
                                   ref=ref, reason=skip_reason, source="job"))
            continue

        key = (a, t)
        if key not in running:
            running[key] = balance_of(a, t, ledger_path=ledger_path)
        signed = -amt if is_buy else amt
        running[key] += signed

        if is_buy and running[key] < -_EPS:
            log(f"⚠️  BREACH {a}/{t}: fill {ref} (${amt:,.2f}) drove the reserve to "
                f"${running[key]:,.2f}. Ledgered as-is — the ledger records reality.")

        rows.append(_event_row(a, t, "BUY_FILL" if is_buy else "SELL_FILL", signed,
                               ref=ref, reason=f"fill {side}", source="job",
                               balance_after=running[key]))

    out = pd.DataFrame(rows, columns=LEDGER_COLS)
    if out.empty:
        return empty

    n_skip = int(out["Event"].isin(SKIP_EVENTS).sum())
    log(f"Fills -> {len(out)} event(s): {len(out) - n_skip} applied, {n_skip} skipped")

    if dry_run:
        log("dry_run — nothing written")
        return out

    append_events(out.to_dict("records"), ledger_path, log)
    return out


def fetch_fills(client_wrapper, *, days_back=7, log=print) -> pd.DataFrame:
    """
    Pull executed trades from Schwab and normalise them into the shape
    apply_fills() expects.

    Schwab returns `activityId` (NOT transactionId) and `netAmount` (NOT
    amount). Method names vary by schwabdev version, so they are probed the
    same way stocks_cash.fetch_account_details does.
    """
    cols = ["Ref", "Account", "Ticker", "Side", "Fill_QTY", "Fill_Price",
            "Net_Amount", "Asset_Type", "Fill_Time"]
    if client_wrapper is None:
        log("⚠️  No Schwab client supplied — skipping fills")
        return pd.DataFrame(columns=cols)

    try:
        from stocks_orders import _raw_client
        client = _raw_client(client_wrapper)
    except Exception as e:
        log(f"⚠️  Could not unwrap Schwab client: {e}")
        return pd.DataFrame(columns=cols)

    end = datetime.utcnow()
    start = end - timedelta(days=int(days_back))
    payloads = []

    accounts = []
    try:
        from stocks_cash import fetch_account_details
        for a in fetch_account_details(client_wrapper, log):
            sa = a.get("securitiesAccount", a) if isinstance(a, dict) else {}
            if sa.get("accountNumber"):
                accounts.append(str(sa["accountNumber"]))
    except Exception as e:
        log(f"⚠️  Could not list accounts for fills: {e}")

    for name in ("transactions", "account_transactions", "transactions_all"):
        fn = getattr(client, name, None)
        if fn is None:
            continue
        for acct in (accounts or [None]):
            for kwargs in (
                dict(accountHash=acct, startDate=start, endDate=end, types="TRADE"),
                dict(account=acct, startDate=start, endDate=end, types="TRADE"),
                dict(startDate=start, endDate=end),
            ):
                try:
                    resp = fn(**{k: v for k, v in kwargs.items() if v is not None})
                    data = resp.json() if hasattr(resp, "json") else resp
                    if isinstance(data, dict):
                        data = [data]
                    if isinstance(data, list) and data:
                        payloads.extend(data)
                        break
                except Exception:
                    continue
        if payloads:
            break

    if not payloads:
        methods = [m for m in dir(client)
                   if "transact" in m.lower() and not m.startswith("__")]
        log(f"⚠️  No transactions returned. Candidate methods on client: {methods}")
        return pd.DataFrame(columns=cols)

    rows = []
    for tx in payloads:
        if not isinstance(tx, dict):
            continue
        if str(tx.get("type", "")).upper() not in ("TRADE", ""):
            continue
        acct = tx.get("accountNumber") or tx.get("accountId") or ""
        for leg in (tx.get("transferItems") or tx.get("transactionItems") or []):
            if not isinstance(leg, dict):
                continue
            inst = leg.get("instrument") or {}
            sym = inst.get("symbol") or inst.get("underlyingSymbol")
            if not sym:
                continue
            amt = leg.get("amount")
            qty = _f(amt if amt is not None else leg.get("quantity"), 0.0) or 0.0
            side = str(leg.get("instruction") or ("SELL" if qty < 0 else "BUY")).upper()
            rows.append({
                "Ref": str(tx.get("activityId") or tx.get("tradeId") or ""),
                "Account": acct,
                "Ticker": sym,
                "Side": side,
                "Fill_QTY": abs(qty),
                "Fill_Price": _f(leg.get("price"), 0.0) or 0.0,
                "Net_Amount": _f(tx.get("netAmount")),
                "Asset_Type": str(inst.get("assetType") or ""),
                "Fill_Time": tx.get("tradeDate") or tx.get("time") or "",
            })

    df = pd.DataFrame(rows, columns=cols)
    df = df[df["Ref"].astype(str).str.strip().ne("")]
    log(f"Fills fetched: {len(df)} leg(s) over {days_back}d")
    return df


# ─────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────
def build_reserves_table(*, positions_df=None, cash_df=None,
                         ledger_path=LEDGER_PATH, config_path=CONFIG_PATH,
                         log=print) -> pd.DataFrame:
    """One row per fenced (Account, Ticker), plus a TOTAL row."""
    cfg = read_config(config_path, log)
    bal = fold_balances(ledger_path=ledger_path)

    if cfg.empty and bal.empty:
        return pd.DataFrame(columns=RESERVE_COLS)

    # Include ledger pairs that have dropped out of the config so money is
    # never invisible just because someone deleted a config line.
    keys = pd.concat([
        cfg[["Account", "Ticker"]],
        bal[["Account", "Ticker"]] if not bal.empty else pd.DataFrame(columns=["Account", "Ticker"]),
    ], ignore_index=True).drop_duplicates()

    df = (keys.merge(cfg, on=["Account", "Ticker"], how="left")
              .merge(bal, on=["Account", "Ticker"], how="left"))

    for c in ("Reserved_Cash", "Deployed", "Returned"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0.0)
    df["Policy"] = df["Policy"].fillna("").replace("", DEFAULT_POLICY)
    df["Active"] = df["Active"].fillna(False)
    df["Nickname"] = df["Account"].map(_nickname)
    df["Seed_Cash"] = pd.to_numeric(df.get("Seed_Cash"), errors="coerce")
    df["Target_Capital"] = pd.to_numeric(df.get("Target_Capital"), errors="coerce")
    df["Last_Event_PST"] = df.get("Last_Event_PST", "").fillna("")

    df["Position_Value"] = [
        round(position_value(a, t, positions_df), 2)
        for a, t in zip(df["Account"], df["Ticker"])
    ]

    avail, status = [], []
    for _, r in df.iterrows():
        res = available_to_buy(r["Account"], r["Ticker"], positions_df=positions_df,
                               config=cfg, ledger_path=ledger_path,
                               config_path=config_path, log=lambda *_: None)
        avail.append(res["available"])
        if r["Reserved_Cash"] < -_EPS:
            status.append("BREACH")
        elif not r["Active"]:
            status.append("INACTIVE")
        elif r["Reserved_Cash"] <= _EPS and r["Deployed"] <= _EPS:
            status.append("UNFUNDED")
        elif res["available"] <= 0:
            status.append("EMPTY")
        else:
            status.append("OK")
    df["Available_To_Buy"] = avail
    df["Status"] = status

    df = df.sort_values(["Account", "Ticker"], kind="mergesort")
    df = df[RESERVE_COLS].reset_index(drop=True)

    total = {c: None for c in RESERVE_COLS}
    total.update({"Account": "TOTAL", "Nickname": "", "Ticker": "", "Policy": "",
                  "Status": "", "Last_Event_PST": ""})
    for c in ("Seed_Cash", "Target_Capital", "Reserved_Cash", "Deployed",
              "Returned", "Position_Value", "Available_To_Buy"):
        s = pd.to_numeric(df[c], errors="coerce")
        total[c] = round(float(s.sum()), 2) if s.notna().any() else None
    df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)

    if cash_df is not None and not getattr(cash_df, "empty", True):
        for w in overcommit_warnings(df, cash_df):
            log(w)

    log(f"Reserves: {len(df) - 1} fenced pair(s) · reserved "
        f"${total['Reserved_Cash'] or 0:,.2f} · available "
        f"${total['Available_To_Buy'] or 0:,.2f}")
    return df


def overcommit_warnings(reserves_df, cash_df) -> list[str]:
    """Fenced cash in an account must not exceed that account's spendable cash."""
    out = []
    try:
        r = reserves_df[reserves_df["Account"].ne("TOTAL")]
        per_acct = (pd.to_numeric(r["Reserved_Cash"], errors="coerce").fillna(0.0)
                    .groupby(r["Account"]).sum())
        c = cash_df[cash_df["Account"].astype(str).ne("TOTAL")].copy()
        c["_k"] = c["Account"].map(acct_key)
        col = "Cash_After_Open_Orders" if "Cash_After_Open_Orders" in c.columns else "Cash"
        spendable = pd.to_numeric(c[col], errors="coerce").fillna(0.0).groupby(c["_k"]).sum()
        for a, fenced in per_acct.items():
            have = float(spendable.get(a, 0.0))
            if fenced - have > 0.01:
                out.append(f"⚠️  OVERCOMMIT ...{a}: ${fenced:,.2f} fenced vs "
                           f"${have:,.2f} spendable (short ${fenced - have:,.2f})")
    except Exception as e:
        out.append(f"⚠️  overcommit check failed: {e}")
    return out


def reconcile(ledger_path=LEDGER_PATH, log=print) -> pd.DataFrame:
    """
    Replay the ledger and compare the folded balance against the advisory
    Balance_After snapshots. Any mismatch means a row was hand-edited or a
    write interleaved — the fold always wins, this only tells you where.
    """
    df = read_ledger(ledger_path)
    if df.empty:
        log("Ledger empty — nothing to reconcile.")
        return pd.DataFrame()

    bad, running = [], {}
    for _, r in df.iterrows():
        k = (r["Account"], r["Ticker"])
        running[k] = running.get(k, 0.0) + float(r["Amount"])
        snap = _f(r["Balance_After"])
        if snap is not None and abs(snap - running[k]) > _EPS:
            bad.append({**r.to_dict(), "Folded_Balance": round(running[k], 2),
                        "Drift": round(snap - running[k], 2)})

    if bad:
        log(f"⚠️  {len(bad)} ledger row(s) drift from the fold — inspect before trusting")
    else:
        log(f"Ledger clean: {len(df)} rows, {len(running)} pair(s), fold matches snapshots")
    return pd.DataFrame(bad)


def write_reserve_outputs(df, updated_pst, *, out_csv, out_html,
                          html_builder, log=print) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    out_html.write_text(
        html_builder(df, "Cash Reserves by Ticker", updated_pst), encoding="utf-8"
    )
    log(f"Reserve outputs written: {out_csv.name} / {out_html.name}")


# ─────────────────────────────────────────────────────────────────────
# CLI — every dependency pulled from jobStocksSignals, none defined here
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, importlib.util, sys

    ap = argparse.ArgumentParser(description="Seed and manage per-ticker cash reserves")
    ap.add_argument("--list", action="store_true", help="print the reserves table")
    ap.add_argument("--seed", action="store_true")
    ap.add_argument("--topup", action="store_true")
    ap.add_argument("--withdraw", action="store_true")
    ap.add_argument("--close", action="store_true")
    ap.add_argument("--check", action="store_true", help="can I spend --amount on this?")
    ap.add_argument("--apply-fills", action="store_true")
    ap.add_argument("--reconcile", action="store_true")
    ap.add_argument("--write", action="store_true", help="write reserves.csv/html")

    ap.add_argument("--account")
    ap.add_argument("--ticker")
    ap.add_argument("--amount", type=float)
    ap.add_argument("--policy", default=DEFAULT_POLICY, choices=sorted(POLICIES))
    ap.add_argument("--target", type=float, help="Target_Capital for TOTAL_CAPITAL")
    ap.add_argument("--reason", default="")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    spec = importlib.util.spec_from_file_location("_job_stocks", str(here / "jobStocksSignals.py"))
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    log = job.log

    try:
        from dotenv import load_dotenv
        load_dotenv(job.MYTRADING_DIR / ".env")
    except Exception:
        pass

    def _need(*names):
        missing = [n for n in names if getattr(args, n, None) in (None, "")]
        if missing:
            ap.error(f"missing required: {', '.join('--' + m for m in missing)}")

    if args.seed:
        _need("account", "ticker", "amount")
        seed(args.account, args.ticker, args.amount, policy=args.policy,
             target_capital=args.target, reason=args.reason, log=log)

    elif args.topup:
        _need("account", "ticker", "amount")
        topup(args.account, args.ticker, args.amount, reason=args.reason, log=log)

    elif args.withdraw:
        _need("account", "ticker", "amount")
        withdraw(args.account, args.ticker, args.amount, reason=args.reason, log=log)

    elif args.close:
        _need("account", "ticker")
        close(args.account, args.ticker, reason=args.reason, log=log)

    elif args.check:
        _need("account", "ticker")
        res = available_to_buy(args.account, args.ticker, log=log)
        want = args.amount
        ok = res["allowed"] and (want is None or want <= res["available"])
        print(f"{acct_key(args.account)}/{tkr_key(args.ticker)}  "
              f"policy={res['policy'] or '-'}  available=${res['available']:,.2f}"
              + (f"  requested=${want:,.2f}" if want is not None else ""))
        print(f"  -> {'ALLOWED' if ok else 'BLOCKED'}: {res['reason']}")
        sys.exit(0 if ok else 1)

    elif args.apply_fills:
        client = job.get_schwab_client()
        fills = fetch_fills(client, days_back=args.days, log=log)
        ev = apply_fills(fills, dry_run=args.dry_run, log=log)
        if not ev.empty:
            print(ev[["Account", "Ticker", "Event", "Amount", "Ref", "Reason"]]
                  .to_string(index=False))

    elif args.reconcile:
        bad = reconcile(log=log)
        if not bad.empty:
            print(bad.to_string(index=False))
            sys.exit(1)

    if args.list or args.write or not any([
        args.seed, args.topup, args.withdraw, args.close, args.check,
        args.apply_fills, args.reconcile,
    ]):
        cash = None
        try:
            cash = pd.read_csv(job.OUT_CASH_CSV)
        except Exception:
            pass
        d = build_reserves_table(cash_df=cash, log=log)
        print(d.to_string(index=False) if not d.empty else "(no reserves configured)")

        if args.write:
            updated = _now_pst()
            write_reserve_outputs(
                d, updated,
                out_csv=getattr(job, "OUT_RESERVES_CSV", job.JOB_DIR / "reserves.csv"),
                out_html=getattr(job, "OUT_RESERVES_HTML", job.JOB_DIR / "reserves.html"),
                html_builder=job.build_html_table, log=log,
            )
