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
    python3 stocks_cash.py --no-orders  # balances only, skip the orders fetch

MASKING: Account is written as the last 3 digits only (e.g. "...922"), which
matches the Acct column in the myTrading Google Sheet. jobMyTrading is a
GitHub repo — full account numbers should not land in it.

COLUMNS (deliberately minimal):
    Account | Nickname | Cash | Cash_In_Open_Orders | Cash_After_Open_Orders
    plus a TOTAL row.

  Cash                   = currentBalances.cashBalance + moneyMarketFund
  Cash_In_Open_Orders    = est. cost of OPEN **BUY** legs in that account.
                           Sells are ignored — they add cash on fill, they do
                           not reserve it.
  Cash_After_Open_Orders = Cash - Cash_In_Open_Orders. This is what is really
                           spendable without borrowing on margin.

Open orders are NOT re-fetched here — the caller passes the DataFrame that
stocks_orders.build_orders_table() already produced (one Schwab orders call
per run, not two). Accounts are joined on the last 3 digits.

CAVEATS:
  * Only accounts linked to your Schwab developer app come back. Workplace
    PCRA/custodial accounts often are not linked and will simply be absent.
  * Verified on account ...885: moneyMarketFund is 0.00 and cashBalance alone
    matches the web UI, so the sum does not double-count the sweep.
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
    "Account", "Nickname",
    "Cash", "Cash_In_Open_Orders", "Cash_After_Open_Orders",
]

# Numeric columns that are summed into the TOTAL row.
_SUM_COLS = [c for c in CASH_COLS if c not in ("Account", "Nickname")]

# Join key between the cash table and stocks_orders: last 3 digits of the
# account number. Works whether or not the account is masked for output.
_KEY = "_acct_key"


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

    acct_raw = str(sa.get("accountNumber") or "")
    last3    = acct_raw[-3:] if len(acct_raw) >= 3 else acct_raw

    # initialBalances is a start-of-day snapshot and carries stale/zeroed
    # fields on margin accounts — currentBalances is the authority.
    cash = _pick(cur, init, "cashBalance", "totalCash")
    mmkt = _pick(cur, init, "moneyMarketFund", "bankSweep", "savings")

    total = None
    if cash is not None or mmkt is not None:
        total = round((cash or 0.0) + (mmkt or 0.0), 2)

    return {
        _KEY:       last3,
        "Account":  _mask(acct_raw) if mask else acct_raw,
        "Nickname": ACCOUNT_LABELS.get(last3, ""),
        "Cash":     total,
        # filled in by build_cash_table() once orders are known
        "Cash_In_Open_Orders":    None,
        "Cash_After_Open_Orders": None,
    }


# ─────────────────────────────────────────────────────────────────────
# Open BUY orders → cash reserved per account
# ─────────────────────────────────────────────────────────────────────
def open_buy_cash_by_account(orders_df, log=print) -> dict[str, float]:
    """
    {last3: reserved_cash} from the orders table stocks_orders already built.

    Only BUY-side legs count — BUY, BUY_TO_OPEN and BUY_TO_COVER all reduce
    spendable cash on fill. SELL legs are ignored on purpose.
    """
    if orders_df is None or getattr(orders_df, "empty", True):
        return {}

    df = orders_df.copy()
    if "Account" not in df.columns or "Side" not in df.columns:
        log("⚠️  Orders table missing Account/Side — cash reservation skipped")
        return {}

    # Defensive: the caller normally passes an already open-only table.
    if "Status" in df.columns:
        from stocks_orders import OPEN_STATUSES
        df = df[df["Status"].astype(str).str.upper().isin(OPEN_STATUSES)]

    side = df["Side"].astype(str).str.upper().str.strip()
    df = df[side.str.startswith("BUY")]
    if df.empty:
        return {}

    qty = pd.to_numeric(df.get("Remaining_QTY"), errors="coerce")
    qty = qty.where(qty.fillna(0) > 0, pd.to_numeric(df.get("QTY"), errors="coerce"))

    px = pd.to_numeric(df.get("Limit_Price"), errors="coerce")
    px = px.where(px.fillna(0) > 0, pd.to_numeric(df.get("Stop_Price"), errors="coerce"))

    mult = pd.Series(1.0, index=df.index)
    if "Asset_Type" in df.columns:
        mult = mult.mask(df["Asset_Type"].astype(str).str.upper() == "OPTION", 100.0)

    # Prefer a freshly computed qty x price (respects partial fills); fall back
    # to the Est_Value stocks_orders already worked out.
    val = qty * px * mult
    if "Est_Value" in df.columns:
        val = val.fillna(pd.to_numeric(df["Est_Value"], errors="coerce"))
    val = val.fillna(0.0)

    keys = df["Account"].map(lambda a: str(a).strip()[-3:])
    out = {k: round(float(v), 2) for k, v in val.groupby(keys).sum().items()}
    log(f"Open BUY orders reserving cash in {len(out)} account(s): "
        f"${sum(out.values()):,.2f}")
    return out


def build_cash_table(client_wrapper, orders_df=None, *, mask: bool = True, log=print) -> pd.DataFrame:
    """
    Cash per linked account, net of open BUY orders, plus a TOTAL row.

    client_wrapper — from jobStocksSignals.get_schwab_client() (reused, not rebuilt)
    orders_df      — the frame stocks_orders.build_orders_table() already built.
                     Pass it in so Schwab's orders endpoint is hit once per run.
                     If None, Cash_In_Open_Orders is 0 and Cash_After_Open_Orders
                     equals Cash.
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

    reserved = open_buy_cash_by_account(orders_df, log)

    for r in rows:
        held = float(reserved.pop(r[_KEY], 0.0))
        cash = r["Cash"]
        r["Cash_In_Open_Orders"]    = round(held, 2)
        r["Cash_After_Open_Orders"] = None if cash is None else round(cash - held, 2)

    # Orders in an account Schwab didn't return balances for — would silently
    # vanish from the totals, so say so rather than swallow it.
    for k, v in reserved.items():
        log(f"⚠️  ${v:,.2f} of open BUY orders in account ...{k} — no balance row")

    df = pd.DataFrame(rows).sort_values("Account", kind="mergesort")
    df = df.drop(columns=[_KEY])[CASH_COLS].reset_index(drop=True)

    total = {c: None for c in CASH_COLS}
    total.update({"Account": "TOTAL", "Nickname": ""})
    for c in _SUM_COLS:
        s = pd.to_numeric(df[c], errors="coerce")
        total[c] = round(float(s.sum()), 2) if s.notna().any() else None
    df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)

    log(f"Cash table: {len(df) - 1} accounts · cash ${total['Cash'] or 0:,.2f} · "
        f"in open buys ${total['Cash_In_Open_Orders'] or 0:,.2f} · "
        f"available ${total['Cash_After_Open_Orders'] or 0:,.2f}")
    return df


# ─────────────────────────────────────────────────────────────────────
# Output — paths and renderer injected, nothing defined here
# ─────────────────────────────────────────────────────────────────────
def write_cash_outputs(df, updated_pst, *, out_csv, out_html, html_builder, log=print) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    out_html.write_text(
        html_builder(df, "Schwab Cash (net of open BUY orders)", updated_pst), encoding="utf-8"
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
    ap.add_argument("--no-orders", action="store_true", help="skip the open-orders fetch")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    spec = importlib.util.spec_from_file_location("_job_stocks", str(here / "jobStocksSignals.py"))
    job  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)

    from dotenv import load_dotenv 
    load_dotenv(job.MYTRADING_DIR / ".env")

    client = job.get_schwab_client()

    if args.raw:
        for a in fetch_account_details(client, job.log):
            sa = a.get("securitiesAccount", a)
            print(f"\n=== {sa.get('accountNumber')} ({sa.get('type')}) ===")
            print(json.dumps({k: v for k, v in sa.items() if "Balances" in k}, indent=2))
        sys.exit(0)

    # Same orders table the job builds — fetched here only because this CLI
    # runs outside the job, and only when we actually need it.
    orders = None
    if not args.no_orders:
        try:
            from stocks_orders import build_orders_table
            orders = build_orders_table(
                client, None,
                days_back=90, open_only=True, restrict_to_tickers=False, log=job.log,
            )
        except Exception as e:
            job.log(f"⚠️  Could not load open orders: {e}")

    d = build_cash_table(client, orders, mask=not args.full, log=job.log)
    print(d.to_string(index=False) if not d.empty else "(no accounts)")

    if not args.no_write and not args.full:
        updated = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z")
        write_cash_outputs(
            d, updated,
            out_csv=job.OUT_CASH_CSV, out_html=job.OUT_CASH_HTML,
            html_builder=job.build_html_table, log=job.log,
        )