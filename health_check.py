#!/usr/bin/env python3
"""
health_check.py
===============
Is every scheduled job actually running, and is every output actually fresh?

    .venv/bin/python health_check.py            # human readable
    .venv/bin/python health_check.py --quiet    # print only problems
    .venv/bin/python health_check.py --json     # machine readable

Read-only. Touches no sheet, no repo, no API.

WHY
    There are now eight scheduled jobs writing files, two Google Sheets and two
    git repos. A silent failure is the dangerous kind: a stale CSV looks exactly
    like a fresh one, and a cron that stopped firing produces no error anywhere.

    Staleness is the signal. Every job leaves a file behind, so the age of that
    file says whether the job ran. Nothing here needs the job to cooperate.

Exit code 0 when everything is within tolerance, 1 when anything is stale or
missing — so cron can mail on failure alone.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HOME = Path.home()
REPO = Path(os.getenv("MYTRADING_REPO", HOME / "github" / "myTrading"))
JOBS = Path(os.getenv("JOBS_REPO", HOME / "github" / "jobMyTrading"))
STATE = Path(os.getenv("REMOTE_OPS_STATE", HOME / ".local" / "state" / "myTrading"))

# (label, path, max age in hours, weekday-only)
#
# Tolerances are deliberately loose — roughly two missed runs, so a single
# hiccup is not an alarm but a stopped job is. Weekday-only entries are not
# checked on a weekend, when their cron does not fire at all.
CHECKS = [
    ("stocks signals",   JOBS / "stocks_signals.csv",            2,   True),
    ("stocks orders",    JOBS / "stocks_orders.csv",             2,   True),
    ("cash",             JOBS / "cash.csv",                      2,   True),
    ("reserves",         JOBS / "reserves.csv",                 30,  False),
    ("crypto signals",   JOBS / "crypto_signals.csv",            3,   False),
    ("macro",            JOBS / "macro.csv",                     2,   True),
    ("45 signal AM",     JOBS / "45_signal_full_morning.csv",   30,   True),
    ("45 signal PM",     JOBS / "45_signal_full_evening.csv",   30,   True),
    ("P&L summary",      JOBS / "outputs/portfolio/ticker_pl_summary.csv", 30, False),
    ("job log",          JOBS / "job_stocks.log",                2,   True),
    ("ops poller log",   STATE / "remote_ops_cron.log",          1,  False),
    ("gitpush log",      HOME / "gitpush_cron.log",              1,  False),
    ("price updater",    STATE / "prices_cron.log",              1,   True),
]


def _age_hours(p: Path) -> float | None:
    try:
        return (datetime.now(timezone.utc)
                - datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
                ).total_seconds() / 3600
    except FileNotFoundError:
        return None


def _is_weekend() -> bool:
    return datetime.now().weekday() >= 5


def check_files() -> list[dict]:
    out, weekend = [], _is_weekend()
    for label, path, max_h, weekday_only in CHECKS:
        age = _age_hours(path)
        if age is None:
            state, detail = "MISSING", "file does not exist"
        elif weekday_only and weekend:
            # Its cron does not fire Sat/Sun, so age is meaningless here.
            state, detail = "OK", f"{age:.1f}h old (weekend — not scheduled)"
        elif age > max_h:
            state, detail = "STALE", f"{age:.1f}h old, expected under {max_h}h"
        else:
            state, detail = "OK", f"{age:.1f}h old"
        out.append(dict(check=label, state=state, detail=detail, path=str(path)))
    return out


def check_crons() -> list[dict]:
    """Every job below writes something CHECKS watches. A missing cron line is
    why a file goes stale, so name it directly instead of making it inferred."""
    expect = {
        "jobStocksSignals.py": "stocks signals, orders, cash, reserves, orders sheet",
        "jobCryptoSignals.py": "crypto signals",
        "45_Signal.py":        "45 degree scans",
        "gitpush.py":          "pushes jobMyTrading to GitHub",
        "remote_ops.py":       "ops sheet poller",
        "orders_sheet_prices.py": "live prices on the Dashboard",
        "build_pl_report.py":  "historical P&L",
    }
    try:
        tab = subprocess.run(["crontab", "-l"], capture_output=True, text=True,
                             timeout=20).stdout
    except Exception as e:
        return [dict(check="crontab", state="MISSING", detail=str(e), path="")]

    live = [l for l in tab.splitlines()
            if l.strip() and not l.lstrip().startswith("#")]
    out = []
    for script, what in expect.items():
        hit = any(script in l for l in live)
        out.append(dict(
            check=f"cron: {script}",
            state="OK" if hit else "MISSING",
            detail=what if hit else f"no ACTIVE crontab line — {what} will not run",
            path=""))
    return out


def check_token() -> list[dict]:
    """The refresh token is a hard 7-day cap and everything Schwab dies with
    it, so it belongs in the same report as the jobs it would take down."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    try:
        import schwab_auth
        st = schwab_auth.status()
    except Exception as e:
        return [dict(check="schwab token", state="UNKNOWN",
                     detail=f"{type(e).__name__}: {e}", path="")]
    days = st.get("days_left")
    state = st.get("state", "UNKNOWN")
    bad = state in ("EXPIRED", "MISSING", "UNKNOWN") or (days is not None and days < 1)
    return [dict(check="schwab token",
                 state="STALE" if bad else "OK",
                 detail=f"{state}"
                        + (f", {days} day(s) left, expires {st.get('expires')}"
                           if days is not None else ""),
                 path="")]


def check_pl_quality() -> list[dict]:
    """A report that runs on schedule can still be quietly wrong, so the known
    defects are surfaced here rather than only in the run log nobody reads."""
    f = JOBS / "outputs/portfolio/ticker_pl_summary.csv"
    if not f.exists():
        return [dict(check="P&L reconciliation", state="MISSING",
                     detail="no summary yet", path=str(f))]
    try:
        import pandas as pd
        df = pd.read_csv(f)
        n = int((df["qty_mismatch"].abs() > 0.005).sum())
        unk = int((df["basis_flag"] == "UNKNOWN_BASIS").sum())
    except Exception as e:
        return [dict(check="P&L reconciliation", state="UNKNOWN",
                     detail=f"{type(e).__name__}: {e}", path=str(f))]
    return [
        dict(check="P&L share counts", state="OK" if n == 0 else "STALE",
             detail=(f"{n} ticker(s) disagree with Schwab — see anomalies.csv"
                     if n else "every ticker matches Schwab"), path=""),
        dict(check="P&L cost basis", state="OK" if unk == 0 else "STALE",
             detail=(f"{unk} ticker(s) with no cost basis — fill manual_basis.csv"
                     if unk else "every ticker has a basis"), path=""),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Check every scheduled job and output")
    ap.add_argument("--quiet", action="store_true", help="print only problems")
    ap.add_argument("--json", action="store_true", help="machine readable")
    args = ap.parse_args()

    rows = check_crons() + check_files() + check_token() + check_pl_quality()
    bad = [r for r in rows if r["state"] != "OK"]

    if args.json:
        print(json.dumps(dict(
            generated=datetime.now().astimezone().isoformat(),
            problems=len(bad), rows=rows), indent=2))
        return 1 if bad else 0

    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"\nmyTrading health — {stamp}\n")
    icon = {"OK": "✅", "STALE": "⚠️ ", "MISSING": "🔴", "UNKNOWN": "❓"}
    for r in rows:
        if args.quiet and r["state"] == "OK":
            continue
        print(f"{icon.get(r['state'], '  ')} {r['check']:<28} {r['detail']}")

    print()
    if bad:
        print(f"{len(bad)} problem(s). Exit 1.")
    else:
        print("All checks passed.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
