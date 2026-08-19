#!/usr/bin/env python3
"""
stocks_cash.py
==============
Per-account cash / money-market balances from Schwab  ->  cash.csv + cash.html.
This is the "Cash & Cash Investments" block of the Schwab positions page.

DESIGN — identical contract to stocks_orders.py. This module owns *only*
balance parsing; everything else is injected by the caller so there is exactly
one definition of each thing in the codebase:

    paths / JOB_DIR      -> jobStocksSignals.OUT_CASH_CSV / OUT_CASH_HTML
    log()                -> jobStocksSignals.log
    HTML rendering       -> jobStocksSignals.build_html_table
    token paths / client -> jobStocksSignals.get_schwab_client()
    raw schwabdev client -> stocks_orders._raw_client   (one unwrapper, reused)

Importable (normal path — client comes from the caller):
    from stocks_cash import build_cash_table, write_cash_outputs

Standalone (resolves the above by loading jobStocksSignals.py):
    python3 stocks_cash.py              # masked accounts, writes cash.csv/html
    python3 stocks_cash.py --full       # full account numbers, print only
    python3 stocks_cash.py --raw        # dump the raw balances dict per account

MASKING: Account is written as the last 3 digits only (e.g. "...922"), which
matches the Acct column in the myTrading Google Sheet. jobMyTrading is a
GitHub repo — full account numbers should not land in it.

CAVEATS to verify once against the Schwab web page:
  * Only accounts linked to your Schwab developer app come back. Workplace
    PCRA/custodial accounts often are not linked and will simply be absent.
  * Cash_Total = Cash + MoneyMarket. If your sweep is already inside
    cashBalance, that double-counts — check one account against the web UI
    and, if so, drop MoneyMarket from the sum in _row_from_balances().
"""
from __future__ import annotations

import pandas as pd

# Optional friendly names, keyed by the LAST 3 DIGITS of the account number.
# Purely cosmetic; unknown accounts just render with a blank Nickname.
ACCOUNT_LABELS = {
    "922": "PCRA Custodial",
    "171": "PCRA Custodial",
    "431": "Rollover IRA",
    "482": "Roth Contributory IRA",
    "505": "Designated Bene Individual",
    "885": "Designated Bene Individual",
}

CASH_COLS = [
    "Account", "Nickname", "Type",
    "Cash", "MoneyMarket", "Cash_Total",
    "Available_To_Trade", "Available_To_Withdraw", "Buying_Power",
    "Long_Market_Value", "Account_Value",
]

# Numeric columns that are summed into the TOTAL row.
_SUM_COLS = [c for c in CASH_COLS if c not in ("Account", "Nickname", "Type")]


# ─────────────────────────────────────────────────────────────────────
# Raw fetch — takes the client the caller already built
# ─────────────────────────────────────────────────────────────────────
def fetch_account_details(client_wrapper, log=print) -> list:
    """
    Raw list of Schwab account dicts (balances only, no positions).
    Empty list on any failure (never raises).
    """
    if client_wrapper is None:
        log("⚠️  No Schwab client supplied — skipping cash")
        return []
    try:
        from stocks_orders import _raw_client   # one unwrapper for the codebase
        client = _raw_client(client_wrapper)

        fn = None
        for name in ("account_details_all", "accounts_details_all", "accounts_all"):
            fn = getattr(client, name, None)
            if fn is not None:
                break
        if fn is None:
            methods = [m for m in dir(client)
                       if "account" in m.lower() and not m.startswith("__")]
            log(f"⚠️  No account_details_all on client; account methods: {methods}")
            return []

        resp = fn()
        data = resp.json() if hasattr(resp, "json") else resp

        if isinstance(data, dict):                 # single-account shape
            data = [data]
        if not isinstance(data, list):
            log(f"⚠️  Unexpected accounts payload: {type(data).__name__}")
            return []

        log(f"Schwab accounts fetched: {len(data)}")
        return data
    except Exception as e:
        log(f"⚠️  Error fetching Schwab account details: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────
# Flatten
# ─────────────────────────────────────────────────────────────────────
def _num(v) -> float | None:
    try:
        return None if v is None or v == "" else float(v)
    except (TypeError, ValueError):
        return None


def _pick(*sources_and_keys) -> float | None:
    """
    _pick(dict_a, dict_b, "keyOne", "keyTwo") -> first non-None numeric value
    found by trying every key against every dict, in order.
    """
    dicts = [d for d in sources_and_keys if isinstance(d, dict)]
    keys  = [k for k in sources_and_keys if isinstance(k, str)]
    for k in keys:
        for d in dicts:
            v = _num(d.get(k))
            if v is not None:
                return v
    return None


def _mask(acct: str) -> str:
    s = str(acct or "").strip()
    return f"...{s[-3:]}" if len(s) >= 3 else s


def _row_from_balances(sa: dict, *, mask: bool = True) -> dict:
    """One row per securitiesAccount. Missing fields stay None, not 0."""
    cur  = sa.get("currentBalances") or {}
    init = sa.get("initialBalances") or {}
    proj = sa.get("projectedBalances") or {}

    acct_raw = str(sa.get("accountNumber") or "")
    last3    = acct_raw[-3:] if len(acct_raw) >= 3 else acct_raw

    cash = _pick(cur, init, "cashBalance", "totalCash", "cashAvailableForTrading")
    mmkt = _pick(cur, init, "moneyMarketFund", "bankSweep", "savings")

    total = None
    if cash is not None or mmkt is not None:
        total = round((cash or 0.0) + (mmkt or 0.0), 2)

    return {
        "Account":  _mask(acct_raw) if mask else acct_raw,
        "Nickname": ACCOUNT_LABELS.get(last3, ""),
        "Type":     str(sa.get("type") or ""),
        "Cash":                  cash,
        "MoneyMarket":           mmkt,
        "Cash_Total":            total,
        "Available_To_Trade":    _pick(cur, proj, "cashAvailableForTrading", "availableFunds"),
        "Available_To_Withdraw": _pick(cur, proj, "cashAvailableForWithdrawal", "availableFundsNonMarginableTrade"),
        "Buying_Power":          _pick(cur, proj, "buyingPower"),
        "Long_Market_Value":     _pick(cur, init, "longMarketValue"),
        "Account_Value":         _pick(cur, init, "liquidationValue", "accountValue", "equity"),
    }


def build_cash_table(client_wrapper, *, mask: bool = True, log=print) -> pd.DataFrame:
    """
    DataFrame of cash balances, one row per linked account, plus a TOTAL row.

    client_wrapper — from jobStocksSignals.get_schwab_client() (reused, not rebuilt)
    mask           — write "...922" instead of the full account number
    """
    rows = []
    for acct in fetch_account_details(client_wrapper, log):
        if not isinstance(acct, dict):
            continue
        sa = acct.get("securitiesAccount") or acct
        if not isinstance(sa, dict) or not sa.get("accountNumber"):
            continue
        rows.append(_row_from_balances(sa, mask=mask))

    if not rows:
        log("No account balances returned.")
        return pd.DataFrame(columns=CASH_COLS)

    df = pd.DataFrame(rows)[CASH_COLS]
    df = df.sort_values("Account", kind="mergesort").reset_index(drop=True)

    total = {c: None for c in CASH_COLS}
    total.update({"Account": "TOTAL", "Nickname": "", "Type": ""})
    for c in _SUM_COLS:
        s = pd.to_numeric(df[c], errors="coerce")
        total[c] = round(float(s.sum()), 2) if s.notna().any() else None
    df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)

    cash_total = total.get("Cash_Total")
    if cash_total is None:
        log(f"Cash table: {len(df) - 1} accounts (no cash fields returned)")
    else:
        log(f"Cash table: {len(df) - 1} accounts, total cash ${cash_total:,.2f}")
    return df


# ─────────────────────────────────────────────────────────────────────
# Output — paths and renderer injected, nothing defined here
# ─────────────────────────────────────────────────────────────────────
def write_cash_outputs(df, updated_pst, *, out_csv, out_html, html_builder, log=print) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    out_html.write_text(
        html_builder(df, "Schwab Cash & Cash Investments", updated_pst), encoding="utf-8"
    )
    log(f"Cash outputs written: {out_csv.name} / {out_html.name}")


# ─────────────────────────────────────────────────────────────────────
# CLI — pulls every dependency from jobStocksSignals, defines none
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, importlib.util, json, sys
    from datetime import datetime
    from pathlib import Path
    import pytz

    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="unmasked account numbers (implies --no-write)")
    ap.add_argument("--raw", action="store_true", help="dump raw balances dicts and exit")
    ap.add_argument("--no-write", action="store_true", help="print only, no CSV/HTML")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    spec = importlib.util.spec_from_file_location("_job_stocks", str(here / "jobStocksSignals.py"))
    job  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)

    client = job.get_schwab_client()

    if args.raw:
        for a in fetch_account_details(client, job.log):
            sa = a.get("securitiesAccount", a)
            print(f"\n=== {sa.get('accountNumber')} ({sa.get('type')}) ===")
            print(json.dumps({k: v for k, v in sa.items() if "Balances" in k}, indent=2))
        sys.exit(0)

    d = build_cash_table(client, mask=not args.full, log=job.log)
    print(d.to_string(index=False) if not d.empty else "(no accounts)")

    if not args.no_write and not args.full:
        updated = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z")
        write_cash_outputs(
            d, updated,
            out_csv=job.OUT_CASH_CSV, out_html=job.OUT_CASH_HTML,
            html_builder=job.build_html_table, log=job.log,
        )