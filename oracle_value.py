#!/usr/bin/env python3
"""
oracle_value.py
===============
Weekly OracleValue for every ticker — the slow half, cached like the Benter
targets and for the same reason.

    .venv/bin/python oracle_value.py --refresh     # fetch + compute + cache
    .venv/bin/python oracle_value.py               # show the cache
    .venv/bin/python oracle_value.py --one AAOI    # one name, verbose

WEEKLY, ON MONDAY. Chakravarti, 2026-11-01: *"lets do that every Monday 1st
run."* The inputs barely move — revenue, margins, cash flow and share count
change once a quarter — so recomputing hourly would cost 135 network calls to
learn nothing. Price is the only fast-moving term, and the sheet already has
it live, so the BAND is recomputed every cycle against a cached value.

MOAT OVERRIDES. core/oracle.infer_moat() is a proxy; StockOracle's rating is
community-voted and we cannot read it. Where we DO know the real one, it is
pinned here rather than guessed — the proxy put SiTime at NONE on a 2% net
margin while StockOracle rates it Narrow.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from core.oracle import value as oracle_value          # noqa: E402

# In the repo, not the state dir — same transport as benter_targets.csv:
# Pi 2 computes, git carries it, Pi 1 reads. Versioned on purpose, so "what
# did the model think in November" stays answerable.
FILE = Path(os.getenv("ORACLE_FILE", HERE / "oracle_values.csv"))
COLS = ["Ticker", "Fetched", "Price_At_Fetch", "Oracle_Value", "Oracle_Low",
        "Oracle_High", "Oracle_Band", "Moat", "Oracle_Why"]

# Known StockOracle ratings, from the report Chakravarti sent 2026-11-01.
# Add to this whenever a real rating is seen; never guess into it.
MOAT_OVERRIDE = {
    "MSFT": "WIDE",
    "SITM": "NARROW",
    "AAOI": "NONE",
}

MAX_AGE_DAYS = int(os.getenv("ORACLE_MAX_AGE_DAYS", "14"))


def tickers() -> list[str]:
    """STOCK_TICKERS out of app.py by AST, never by import — app.py drags in
    schwabdev and Streamlit, which this job does not need and Pi 2 lacks."""
    import ast
    tree = ast.parse((HERE / "app.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "STOCK_TICKERS" for t in node.targets):
            return sorted({str(e.value) for e in node.value.elts
                           if isinstance(e, ast.Constant)})
    raise RuntimeError("no STOCK_TICKERS in app.py")


def load(log=print) -> dict:
    if not FILE.exists():
        return {}
    try:
        with FILE.open(newline="", encoding="utf-8") as fh:
            return {str(r["Ticker"]).upper(): r for r in csv.DictReader(fh)
                    if r.get("Ticker")}
    except Exception as e:                               # noqa: BLE001
        log(f"⚠️  could not read {FILE.name}: {e}")
        return {}


def compute_one(sym: str, info: dict | None = None, price=None):
    """One ticker -> Oracle. Returns None when yfinance gives us nothing."""
    import yfinance as yf
    i = info if info is not None else (yf.Ticker(sym).info or {})
    px = price or i.get("currentPrice") or i.get("previousClose")
    if not px:
        return None
    fpe = i.get("forwardPE")
    return oracle_value(
        px,
        shares=i.get("sharesOutstanding"),
        revenue=i.get("totalRevenue"),
        net_income=i.get("netIncomeToCommon"),
        operating_cf=i.get("operatingCashflow"),
        free_cf=i.get("freeCashflow"),
        gross_margin=i.get("grossMargins"),
        net_margin=i.get("profitMargins"),
        roe=i.get("returnOnEquity"),
        market_cap=i.get("marketCap"),
        growth=i.get("revenueGrowth"),
        beta=i.get("beta"),
        total_debt=i.get("totalDebt"),
        total_cash=i.get("totalCash"),
        forward_eps=(px / fpe) if fpe and fpe > 0 else None,
        moat=MOAT_OVERRIDE.get(sym.upper()),
    )


def due_today(log=print) -> bool:
    """Monday, or the cache is missing or stale. Chakravarti asked for
    'every Monday 1st run' — so the job is safe to schedule daily and will
    simply do nothing on the other six days.

    Also fires when the cache has gone past MAX_AGE_DAYS, so a Monday the Pi
    happened to be down does not leave the column blank all week.
    """
    cache = load(log=lambda *_: None)
    if not cache:
        log("no cache — refreshing regardless of weekday")
        return True
    ages = [a for a in (age_days(r) for r in cache.values()) if a is not None]
    if ages and min(ages) > MAX_AGE_DAYS:
        log(f"cache is {min(ages)}d old (> {MAX_AGE_DAYS}) — refreshing")
        return True
    if date.today().weekday() == 0:
        if ages and min(ages) == 0:
            log("already refreshed today")
            return False
        return True
    log(f"not Monday ({date.today():%A}) and cache is "
        f"{min(ages) if ages else '?'}d old — nothing to do")
    return False


def refresh(syms=None, log=print) -> int:
    """Recompute and REPLACE the cache. A failed ticker keeps its old row."""
    import yfinance as yf
    syms = syms or tickers()
    prev, out, failed = load(log=lambda *_: None), [], []
    today = date.today().isoformat()

    for n, s in enumerate(syms, 1):
        try:
            i = yf.Ticker(s).info or {}
            o = compute_one(s, i)
            if o is None or o.value is None:
                raise ValueError(o.why if o else "no price")
            out.append({
                "Ticker": s, "Fetched": today,
                "Price_At_Fetch": i.get("currentPrice") or i.get("previousClose") or "",
                "Oracle_Value": o.value, "Oracle_Low": o.low,
                "Oracle_High": o.high, "Oracle_Band": o.band,
                "Moat": o.moat, "Oracle_Why": o.why,
            })
        except Exception as e:                           # noqa: BLE001
            failed.append(f"{s} ({type(e).__name__})")
            if s in prev:
                out.append(prev[s])                      # keep, never drop
        if n % 25 == 0:
            log(f"   …{n}/{len(syms)}")

    try:
        FILE.parent.mkdir(parents=True, exist_ok=True)
        with FILE.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(out)
    except Exception as e:                               # noqa: BLE001
        log(f"⛔ could not write {FILE}: {e}")
        return 0

    fresh = sum(1 for r in out if str(r.get("Fetched", "")).startswith(today))
    log(f"{len(out)} row(s) cached — {fresh} fresh, {len(out) - fresh} kept")
    if failed:
        log(f"⚠️  no value for {len(failed)}: {', '.join(failed[:12])}"
            + (" …" if len(failed) > 12 else ""))
    return len(out)


def age_days(row) -> int | None:
    from datetime import datetime
    try:
        return (date.today() - datetime.strptime(
            str(row["Fetched"])[:10], "%Y-%m-%d").date()).days
    except Exception:                                    # noqa: BLE001
        return None


def for_sheet(ticker, price=None, cache=None) -> tuple[str, str]:
    """(cell_text, band) for the Dashboard RECOMMENDED row.

    The VALUE is cached weekly; the BAND is recomputed here against whatever
    price is passed in, so a stock that falls 20% during the week shows as
    cheaper the same day rather than waiting for Monday.
    """
    from core.oracle import BANDS
    cache = load(log=lambda *_: None) if cache is None else cache
    r = cache.get(str(ticker).upper())
    if not r:
        return "", ""
    a = age_days(r)
    if a is None or a > MAX_AGE_DAYS:
        return "", "STALE"
    try:
        lo, hi, val = float(r["Oracle_Low"]), float(r["Oracle_High"]), float(r["Oracle_Value"])
    except (TypeError, ValueError):
        return "", ""
    text = (f"${lo:,.0f}–${hi:,.0f}" if hi > lo * 1.02 else f"${val:,.2f}")
    band = r.get("Oracle_Band", "")
    try:
        if price and float(price) > 0 and val > 0:
            band = next(b for lim, b in BANDS if float(price) / val <= lim)
    except Exception:                                    # noqa: BLE001
        pass
    return text, band


def main() -> int:
    ap = argparse.ArgumentParser(description="Weekly conservative fair value")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--weekly", action="store_true",
                    help="refresh only if due (Monday, or cache stale). "
                         "Safe to run daily from cron.")
    ap.add_argument("--one", metavar="TICKER")
    a = ap.parse_args()

    if a.one:
        o = compute_one(a.one.upper())
        if not o or o.value is None:
            print(f"{a.one}: {o.why if o else 'no data'}")
            return 1
        print(f"{a.one.upper()}  OracleValue {o.as_text()}  ({o.band})")
        print(f"   point ${o.value:,.2f} · moat {o.moat} · gap {o.gap_pct:+.1f}%")
        print(f"   {o.why}")
        return 0

    if a.weekly:
        if not due_today():
            return 0
        print(f"computing OracleValue → {FILE}")
        return 0 if refresh() else 1

    if a.refresh:
        print(f"computing OracleValue → {FILE}")
        return 0 if refresh() else 1

    cache = load()
    if not cache:
        print(f"nothing cached at {FILE} — run with --refresh")
        return 1
    ages = [x for x in (age_days(r) for r in cache.values()) if x is not None]
    print(f"{len(cache)} cached, {min(ages)}-{max(ages)} days old\n")
    print(f"{'':7}{'value':>12}{'band':>13}  moat")
    for t, r in sorted(cache.items()):
        print(f"{t:<7}{r['Oracle_Value']:>12}{r['Oracle_Band']:>13}  {r['Moat']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
