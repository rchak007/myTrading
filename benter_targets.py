#!/usr/bin/env python3
"""
benter_targets.py
=================
Analyst consensus for every ticker, cached weekly — the slow half of the
Benter rating.

    .venv/bin/python benter_targets.py --refresh    # fetch and cache (weekly)
    .venv/bin/python benter_targets.py              # show what is cached
    .venv/bin/python benter_targets.py --score      # cached targets + live price

WHY SPLIT SLOW FROM FAST. Chakravarti, 2026-10-16: *"we don't need to check
analysts consensus too often - maybe weekly once & store it & only current
price is fluctuating."* Exactly right, and it is what makes the rating
affordable at all.

    TARGETS    move when an analyst publishes — days to weeks apart.
               One network call per ticker. 135 tickers is slow.
    PRICE      moves every tick, and is already in the signals table.
    ATR        already computed every cycle.

So the gap is recomputed every run from a live price against a cached target.
A week-old target against today's price is the honest combination: the target
IS a week old, and pretending otherwise by refetching it hourly would cost 135
API calls to learn nothing.

STALENESS IS REPORTED, NOT HIDDEN. Every row carries the date it was fetched,
and attach() refuses to score from a cache older than MAX_AGE_DAYS rather
than quietly rating on numbers from last quarter.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import date, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

STATE_DIR = Path(os.getenv("REMOTE_OPS_STATE",
                           Path.home() / ".local" / "state" / "myTrading"))
FILE = Path(os.getenv("BENTER_TARGETS_FILE", STATE_DIR / "benter_targets.csv"))
COLS = ["Ticker", "Fetched", "Price_At_Fetch", "Target_Mean", "Target_High",
        "Target_Low", "Analysts", "Recommendation", "Sector"]

# A target older than this is not scored. Analysts revise after earnings, so
# a quarter-old consensus can be anchored to a thesis the company has already
# updated — the one way this screen fails silently.
MAX_AGE_DAYS = int(os.getenv("BENTER_MAX_AGE_DAYS", "21"))


def tickers() -> list[str]:
    """STOCK_TICKERS from app.py's SOURCE — importing it drags in schwabdev,
    which Pi 2 does not have and should not need."""
    import ast
    tree = ast.parse((HERE / "app.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "STOCK_TICKERS" for t in node.targets):
            return sorted({str(e.value) for e in node.value.elts
                           if isinstance(e, ast.Constant)})
    raise RuntimeError("no STOCK_TICKERS list in app.py")


def load(log=print) -> dict:
    """{TICKER: row}. Empty on anything unexpected."""
    if not FILE.exists():
        return {}
    try:
        with FILE.open(newline="", encoding="utf-8") as fh:
            return {str(r["Ticker"]).upper(): r for r in csv.DictReader(fh)
                    if r.get("Ticker")}
    except Exception as e:
        log(f"⚠️  could not read {FILE.name}: {e}")
        return {}


def age_days(row) -> int | None:
    try:
        return (date.today()
                - datetime.strptime(str(row["Fetched"])[:10],
                                    "%Y-%m-%d").date()).days
    except Exception:
        return None


def refresh(syms=None, log=print) -> int:
    """Fetch consensus for each ticker and REPLACE the cache. Returns rows.

    A ticker that fails keeps its previous row rather than vanishing — a
    transient lookup failure should not silently un-rate a position.
    """
    import yfinance as yf
    syms = syms or tickers()
    prev, out, failed = load(log=lambda *_: None), [], []
    today = date.today().isoformat()

    for i, t in enumerate(syms, 1):
        try:
            info = yf.Ticker(t).info or {}
            mean = info.get("targetMeanPrice")
            if not mean:
                raise ValueError("no targetMeanPrice")
            out.append({
                "Ticker": t, "Fetched": today,
                "Price_At_Fetch": info.get("currentPrice")
                or info.get("previousClose") or "",
                "Target_Mean": mean,
                "Target_High": info.get("targetHighPrice") or "",
                "Target_Low": info.get("targetLowPrice") or "",
                "Analysts": info.get("numberOfAnalystOpinions") or "",
                "Recommendation": info.get("recommendationKey") or "",
                "Sector": info.get("sector") or "",
            })
        except Exception as e:
            failed.append(f"{t} ({type(e).__name__})")
            if t in prev:
                out.append(prev[t])          # keep the stale row, do not drop
        if i % 25 == 0:
            log(f"   …{i}/{len(syms)}")

    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with FILE.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(out)
    except Exception as e:
        log(f"⛔ could not write {FILE}: {e}")
        return 0

    fresh = sum(1 for r in out if str(r.get("Fetched", "")).startswith(today))
    log(f"{len(out)} row(s) cached — {fresh} fresh, {len(out) - fresh} kept "
        f"from a previous fetch")
    if failed:
        log(f"⚠️  no consensus for {len(failed)}: {', '.join(failed[:12])}"
            + (" …" if len(failed) > 12 else ""))
    return len(out)


def attach(signals_df, log=print):
    """Add Benter_Rating / Benter_Band / Benter_Gap_% / Benter_Why in place.

    Called from the signals job, where price and ATR are already computed.
    Leaves the columns blank for anything it cannot honestly score — a ticker
    with no coverage, or a cache too old to trust.
    """
    if signals_df is None or getattr(signals_df, "empty", True):
        return signals_df
    from core.benter import score

    cache = load(log=log)
    if not cache:
        log("Benter: no cached targets — run benter_targets.py --refresh")
        for c in ("Benter_Rating", "Benter_Band", "Benter_Gap_%", "Benter_Why"):
            signals_df[c] = ""
        return signals_df

    pcol = next((c for c in ("Current Price", "Last Close", "Price")
                 if c in signals_df.columns), None)
    rate, band, gap, why = [], [], [], []
    scored = stale = uncovered = 0

    for _, r in signals_df.iterrows():
        t = str(r.get("Ticker", "")).upper()
        row = cache.get(t)
        px = r.get(pcol) if pcol else None
        a = age_days(row) if row else None
        if not row or a is None:
            rate.append(""); band.append(""); gap.append(""); why.append("")
            uncovered += 1
            continue
        if a > MAX_AGE_DAYS:
            rate.append(""); band.append(""); gap.append("")
            why.append(f"target {a}d old — refresh needed")
            stale += 1
            continue
        atr = r.get("ATR")
        atr_pct = None
        try:
            if atr and px and float(px) > 0:
                atr_pct = 100.0 * float(atr) / float(px)
        except (TypeError, ValueError):
            pass
        b = score(px, row.get("Target_Mean"),
                  target_high=row.get("Target_High"),
                  target_low=row.get("Target_Low"),
                  analysts=row.get("Analysts"), atr_pct=atr_pct)
        if b.score is None:
            rate.append(""); band.append(""); gap.append("")
            why.append(b.why)
            uncovered += 1
            continue
        rate.append(b.score); band.append(b.band)
        gap.append(round(100 * b.gap, 1))
        why.append(f"{b.why} (target {a}d old)")
        scored += 1

    signals_df["Benter_Rating"] = rate
    signals_df["Benter_Band"] = band
    signals_df["Benter_Gap_%"] = gap
    signals_df["Benter_Why"] = why
    log(f"Benter: {scored} scored, {stale} stale, {uncovered} without coverage")
    return signals_df


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyst consensus cache")
    ap.add_argument("--refresh", action="store_true", help="fetch and cache")
    ap.add_argument("--score", action="store_true",
                    help="score the cache against live prices")
    args = ap.parse_args()

    if args.refresh:
        print(f"refreshing analyst targets → {FILE}")
        return 0 if refresh() else 1

    cache = load()
    if not cache:
        print(f"nothing cached at {FILE} — run with --refresh")
        return 1
    ages = [a for a in (age_days(r) for r in cache.values()) if a is not None]
    print(f"{len(cache)} ticker(s) cached, "
          f"{min(ages)}-{max(ages)} days old" if ages else "")

    if not args.score:
        return 0

    import warnings
    warnings.filterwarnings("ignore")
    import yfinance as yf
    import pandas as pd
    from core.benter import score, concentration_note

    rows = []
    print(f"\n{'':<7}{'price':>10}{'target':>9}{'gap':>7}{'score':>7}  band")
    for t, r in sorted(cache.items()):
        try:
            d = yf.download(t, period="3mo", interval="1d", progress=False,
                            auto_adjust=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            px = float(d["Close"].iloc[-1])
            pc = d["Close"].shift(1)
            tr = pd.concat([d["High"] - d["Low"], (d["High"] - pc).abs(),
                            (d["Low"] - pc).abs()], axis=1).max(axis=1)
            ap = 100 * float(tr.ewm(alpha=0.1, adjust=False,
                                    min_periods=10).mean().iloc[-1]) / px
        except Exception:
            continue
        b = score(px, r.get("Target_Mean"), target_high=r.get("Target_High"),
                  target_low=r.get("Target_Low"), analysts=r.get("Analysts"),
                  atr_pct=ap)
        if b.score is None:
            continue
        rows.append({"Ticker": t, "Sector": r.get("Sector"), "score": b.score})
        print(f"{t:<7}{px:>10,.2f}{float(r['Target_Mean']):>9,.0f}"
              f"{100 * b.gap:>6.0f}%{b.score:>7.1f}  {b.band}")
    note = concentration_note(rows)
    if note:
        print(f"\n{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
