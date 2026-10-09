#!/usr/bin/env python3
"""
backtest_structure.py
=====================
What would core/structure.py actually have flagged?

    .venv/bin/python backtest_structure.py --fetch        # pull 2y of bars
    .venv/bin/python backtest_structure.py                # score them
    .venv/bin/python backtest_structure.py --ticker JOBY  # one chart, in detail
    .venv/bin/python backtest_structure.py --sweep        # try other thresholds

WHY
    BALANCE_BARS 5, BALANCE_MAX 1.5 ATR, LEG_MIN 2.0 ATR were a first guess
    and had never been checked against a chart. A detector nobody has
    backtested is an opinion. Chakravarti's call (2026-10-09): test it before
    anything depends on it.

WHAT IT MEASURES
    For every ticker and every day with enough history, run classify() on the
    bars available UP TO THAT DAY — never later ones — and when it says
    TRIGGERED, record what the next N days did. Entry is the NEXT session's
    open, because a close-triggered signal cannot be filled at that close.

    Hit rate is not the point. The point is whether the thing fires at a
    sane RATE and whether the P edge is real — if P, b and D all return the
    same forward numbers, the shape is not information and the feature should
    not be built.

HONEST ABOUT WHAT THIS IS NOT
    Daily bars from Yahoo, no slippage, no commission, no position sizing, no
    overlap control — a name can signal three times in a week and be counted
    three times. Two years is ~500 bars per ticker and this list is heavy on
    2024-26 AI/nuclear/quantum names, so the sample sits inside one of the
    strongest momentum tapes on record. It will flatter any long breakout
    rule. Read the b/D comparison, not the absolute return.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pandas as pd                                        # noqa: E402

from core import structure as ST                           # noqa: E402

CACHE = Path(os.getenv("STRUCT_BT_CACHE", HERE / "scratch" / "bt_bars"))
ATR_PERIOD = 10          # matches core/indicators.compute_supertrend
HORIZONS = (5, 10, 20)   # trading days forward to score


# ── bars ────────────────────────────────────────────────────────────
def tickers() -> list[str]:
    """STOCK_TICKERS, read out of app.py's SOURCE rather than imported.

    Importing app.py drags in schwabdev, which Pi 2 does not have and should
    not need — the whole point of a pure detector is that it can be tested on
    the machine with no credentials.
    """
    import ast
    tree = ast.parse((HERE / "app.py").read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(getattr(t, "id", "") == "STOCK_TICKERS" for t in node.targets):
            return sorted({str(e.value) for e in node.value.elts
                           if isinstance(e, ast.Constant)})
    raise RuntimeError("no STOCK_TICKERS list found in app.py")


def fetch(syms, period="2y", log=print) -> int:
    """Download daily bars once and park them as CSV. Re-runnable."""
    import yfinance as yf
    CACHE.mkdir(parents=True, exist_ok=True)
    got = 0
    for i in range(0, len(syms), 20):                 # batches, to be polite
        chunk = syms[i:i + 20]
        try:
            raw = yf.download(chunk, period=period, interval="1d",
                              progress=False, auto_adjust=False,
                              group_by="ticker", threads=True)
        except Exception as e:
            log(f"  batch {chunk[0]}..{chunk[-1]} failed: {e}")
            continue
        for s in chunk:
            try:
                df = raw[s] if len(chunk) > 1 else raw
                df = df.dropna(subset=["Close"])
                if len(df) < ST.MIN_BARS + max(HORIZONS):
                    log(f"  {s:<6} only {len(df)} bars — skipped")
                    continue
                df.to_csv(CACHE / f"{s}.csv")
                got += 1
            except Exception as e:
                log(f"  {s:<6} {type(e).__name__}: {e}")
        log(f"  …{i + len(chunk)}/{len(syms)}")
    return got


def load(sym):
    p = CACHE / f"{sym}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    for c in ("Open", "High", "Low", "Close"):
        if c not in df.columns:
            return None
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Open", "High", "Low", "Close"])


def atr_series(df, period=ATR_PERIOD):
    """Wilder ATR, the same way compute_supertrend does it."""
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# ── the walk ────────────────────────────────────────────────────────
def scan(sym, df, **kw) -> list[dict]:
    """Every day's verdict, computed only from bars up to that day."""
    atr = atr_series(df)
    hi, lo, cl = df["High"].tolist(), df["Low"].tolist(), df["Close"].tolist()
    op = df["Open"].tolist()
    dates = [str(d)[:10] for d in df.index]
    out = []
    for i in range(ST.MIN_BARS - 1, len(df)):
        a = atr.iloc[i]
        if a != a or a <= 0:
            continue
        s = ST.classify(hi[:i + 1], lo[:i + 1], cl[:i + 1], a, **kw)
        row = {"Ticker": sym, "Date": dates[i], "State": s.state,
               "Shape": s.shape or "", "Close": cl[i], "ATR": a,
               "ATR_pct": 100.0 * a / cl[i] if cl[i] else None,
               "Boundary": s.boundary, "Floor": s.floor,
               "Leg_ATR": s.leg_atr, "Range_ATR": s.range_atr,
               "Break_ATR": s.break_atr, "Why": s.why}
        # Forward returns from the NEXT open — a close-triggered signal
        # cannot be filled at that close.
        if i + 1 < len(op):
            entry = op[i + 1]
            row["Entry"] = entry
            for n in HORIZONS:
                j = i + n
                row[f"R{n}"] = (100.0 * (cl[j] - entry) / entry
                                if j < len(cl) and entry else None)
            # Worst close between entry and the longest horizon: what a stop
            # would have had to sit through.
            j = min(i + max(HORIZONS), len(cl) - 1)
            if entry and j > i:
                row["MAE"] = 100.0 * (min(cl[i + 1:j + 1]) - entry) / entry
        out.append(row)
    return out


def walk(syms, log=print, **kw) -> pd.DataFrame:
    rows, missing = [], []
    for s in syms:
        df = load(s)
        if df is None or len(df) < ST.MIN_BARS + 2:
            missing.append(s)
            continue
        rows.extend(scan(s, df, **kw))
    if missing:
        log(f"no bars for {len(missing)}: {', '.join(missing)}")
    return pd.DataFrame(rows)


# ── reporting ───────────────────────────────────────────────────────
def _n(v, w=7, d=1, suffix=""):
    return "—".rjust(w) if v is None or v != v else f"{v:,.{d}f}{suffix}".rjust(w)


def report(all_days: pd.DataFrame, log=print):
    fired = all_days[all_days.State == ST.TRIGGERED]
    armed = all_days[all_days.State == ST.P_FORMED]
    n_t = all_days.Ticker.nunique()
    days = len(all_days)

    log(f"\n{'='*74}\n  {days:,} ticker-days over {n_t} tickers "
        f"({all_days.Date.min()} → {all_days.Date.max()})\n{'='*74}")

    log(f"\n  TRIGGERED   {len(fired):>6,}  "
        f"{100.0*len(fired)/days:>5.2f}% of days   "
        f"{len(fired)/max(n_t,1):>5.1f} per ticker over the period")
    log(f"  P_FORMED    {len(armed):>6,}  {100.0*len(armed)/days:>5.2f}% "
        f"— a P sitting there, waiting")
    bal = all_days[all_days.Shape != ""]
    log(f"  balanced    {len(bal):>6,}  {100.0*len(bal)/days:>5.2f}% "
        f"— ≤{ST.BALANCE_MAX_ATR} ATR over {ST.BALANCE_BARS} bars")
    for sh in ("P", "b", "D"):
        k = bal[bal.Shape == sh]
        log(f"      {sh:<3}     {len(k):>6,}  "
            f"{100.0*len(k)/max(len(bal),1):>5.1f}% of balances")

    # ── the question that decides whether to build this ──
    log("\n  FORWARD RETURN FROM THE NEXT OPEN, by what broke out")
    log(f"  {'':<22}{'n':>7}{'5d':>9}{'10d':>9}{'20d':>9}"
        f"{'win20':>8}{'MAE':>9}")
    rows = []
    for sh, label in (("P", "P broke out  ← the rule"),
                      ("b", "b broke out"), ("D", "D broke out")):
        k = all_days[(all_days.Shape == sh) & (all_days.Break_ATR.notna())
                     & (all_days.Break_ATR > 0)]
        rows.append((label, k))
    base = all_days[all_days.Entry.notna()]
    rows.append(("every day (baseline)", base))
    for label, k in rows:
        if not len(k):
            log(f"  {label:<22}{'0':>7}")
            continue
        w = k.R20.dropna()
        log(f"  {label:<22}{len(k):>7,}"
            f"{_n(k.R5.mean(), 9, 2, '%')}{_n(k.R10.mean(), 9, 2, '%')}"
            f"{_n(k.R20.mean(), 9, 2, '%')}"
            f"{_n(100.0*(w > 0).mean() if len(w) else None, 8, 0, '%')}"
            f"{_n(k.MAE.mean(), 9, 1, '%')}")
    log("\n  MAE = average worst close in the 20 days after entry. It is where\n"
        "  a stop would have had to sit to survive the trade.")

    # Are these independent trades, or one market move counted many times?
    # 148 fires across 135 correlated AI/semi names could be a dozen days.
    if len(fired):
        per_day = fired.groupby("Date").size().sort_values(ascending=False)
        n_days = len(per_day)
        log(f"\n  CLUSTERING  {len(fired)} fires fell on {n_days} distinct "
            f"dates — {len(fired)/n_days:.1f} per firing day")
        log("  busiest:  " + ",  ".join(f"{d} ×{int(n)}"
                                        for d, n in per_day.head(6).items()))
        log("  A day with many simultaneous fires is ONE market move, not many\n"
            "  signals. These names are heavily correlated, so treat the n\n"
            "  above as optimistic.")


def by_ticker(all_days, log=print, top=25):
    fired = all_days[all_days.State == ST.TRIGGERED]
    if fired.empty:
        log("\n  nothing fired")
        return
    g = fired.groupby("Ticker").agg(
        Fires=("Date", "count"), Last=("Date", "max"),
        R20=("R20", "mean"), ATRpct=("ATR_pct", "mean")).sort_values(
        "Fires", ascending=False)
    log(f"\n  WHICH NAMES FIRE (top {top} of {len(g)})")
    log(f"  {'Ticker':<8}{'fires':>6}{'ATR%':>7}{'avg 20d':>10}  last")
    for t, r in g.head(top).iterrows():
        log(f"  {t:<8}{int(r.Fires):>6}{_n(r.ATRpct, 7, 1)}"
            f"{_n(r.R20, 10, 2, '%')}  {r.Last}")
    never = sorted(set(all_days.Ticker) - set(fired.Ticker))
    if never:
        log(f"\n  never fired in the period ({len(never)}): {', '.join(never)}")


def today_state(all_days, log=print):
    """Where every ticker stands on the last bar — the live view."""
    last = all_days.sort_values("Date").groupby("Ticker").tail(1)
    log(f"\n{'='*74}\n  AS OF THE LAST BAR ({last.Date.max()})\n{'='*74}")
    for st in (ST.TRIGGERED, ST.P_FORMED):
        k = last[last.State == st]
        log(f"\n  {st}  ({len(k)})")
        if k.empty:
            log("    none")
            continue
        for _, r in k.sort_values("Ticker").iterrows():
            log(f"    {r.Ticker:<7}{r.Date}  close {r.Close:>9,.2f}  "
                f"boundary {_n(r.Boundary, 9, 2)}  floor {_n(r.Floor, 9, 2)}  "
                f"leg {_n(r.Leg_ATR, 6, 1)} ATR")
    k = last[last.State == ST.WATCHING]
    log(f"\n  WATCHING  ({len(k)})  — no P right now")


def detail(sym, log=print, **kw):
    df = load(sym)
    if df is None:
        log(f"no bars cached for {sym} — run --fetch")
        return 1
    rows = scan(sym, df, **kw)
    log(f"\n{sym} — {len(df)} bars, last {len(rows) and rows[-1]['Date']}")
    log(f"  {'Date':<12}{'Close':>9}{'ATR':>7}{'leg':>7}{'rng':>6}"
        f"{'bound':>9}  state")
    for r in rows[-45:]:
        mark = ("  🔥" if r["State"] == ST.TRIGGERED
                else "  ⏳" if r["State"] == ST.P_FORMED else "    ")
        log(f"  {r['Date']:<12}{r['Close']:>9,.2f}{_n(r['ATR'], 7, 2)}"
            f"{_n(r['Leg_ATR'], 7, 1)}{_n(r['Range_ATR'], 6, 1)}"
            f"{_n(r['Boundary'], 9, 2)}{mark} {r['State']}")
    fires = [r for r in rows if r["State"] == ST.TRIGGERED]
    log(f"\n  fired {len(fires)}x over the period")
    for r in fires:
        log(f"    {r['Date']}  {r['Why']}")
        log(f"      → entry {_n(r.get('Entry'), 8, 2)}  "
            f"5d {_n(r.get('R5'), 7, 1, '%')}  10d {_n(r.get('R10'), 7, 1, '%')}"
            f"  20d {_n(r.get('R20'), 7, 1, '%')}  MAE {_n(r.get('MAE'), 7, 1, '%')}")
    return 0


def sweep(syms, log=print):
    """Do the thresholds matter, and in which direction?"""
    log(f"\n{'='*74}\n  THRESHOLD SWEEP\n{'='*74}")
    log(f"  {'bars':>5}{'maxATR':>8}{'legATR':>8}{'brkATR':>8}"
        f"{'fires':>8}{'%days':>7}{'5d':>8}{'10d':>8}{'20d':>8}{'win20':>7}")
    grid = []
    for bars in (4, 5, 7):
        for bmax in (1.0, 1.5, 2.0):
            grid.append((bars, bmax, 2.0, 0.0))
    for leg in (1.0, 1.5, 2.5, 3.0):
        grid.append((5, 1.5, leg, 0.0))
    for brk in (0.1, 0.25, 0.5):
        grid.append((5, 1.5, 2.0, brk))
    for bars, bmax, leg, brk in grid:
        d = walk(syms, log=lambda *a: None, balance_bars=bars,
                 balance_max_atr=bmax, leg_min_atr=leg, break_min_atr=brk)
        if d.empty:
            continue
        f = d[d.State == ST.TRIGGERED]
        w = f.R20.dropna() if len(f) else pd.Series(dtype=float)
        log(f"  {bars:>5}{bmax:>8.1f}{leg:>8.1f}{brk:>8.2f}{len(f):>8,}"
            f"{100.0*len(f)/len(d):>7.2f}"
            f"{_n(f.R5.mean() if len(f) else None, 8, 2)}"
            f"{_n(f.R10.mean() if len(f) else None, 8, 2)}"
            f"{_n(f.R20.mean() if len(f) else None, 8, 2)}"
            f"{_n(100.0*(w > 0).mean() if len(w) else None, 7, 0)}")
    log("\n  Rows 1-9 vary the balance, 10-13 the leg, 14-16 how far past the\n"
        "  boundary the close must settle. Defaults are bars 5 / 1.5 / 2.0 / 0.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest core/structure.py")
    ap.add_argument("--fetch", action="store_true", help="download bars first")
    ap.add_argument("--period", default="2y", help="how much history (yf period)")
    ap.add_argument("--ticker", help="one chart, bar by bar")
    ap.add_argument("--sweep", action="store_true", help="vary the thresholds")
    ap.add_argument("--csv", help="write every ticker-day here")
    args = ap.parse_args()

    syms = tickers()
    if args.fetch:
        print(f"fetching {args.period} of daily bars for {len(syms)} tickers "
              f"→ {CACHE}")
        print(f"\n{fetch(syms, period=args.period)} cached")

    have = sorted(p.stem for p in CACHE.glob("*.csv"))
    if not have:
        print(f"no bars in {CACHE} — run with --fetch first")
        return 1

    if args.ticker:
        return detail(args.ticker.upper())
    if args.sweep:
        sweep(have)
        return 0

    d = walk(have)
    if d.empty:
        print("no evaluable ticker-days")
        return 1
    report(d)
    by_ticker(d)
    today_state(d)
    if args.csv:
        d.to_csv(args.csv, index=False)
        print(f"\nwrote {len(d):,} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
