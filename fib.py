#!/usr/bin/env python3
"""
fib.py
======
Fibonacci BUY and TRIM ladders per ticker, anchored automatically.

Chakravarti, 2026-11-22, after a weekend of drawing them by hand on MU, BE,
GOOG and BTC: *"build some FIB series - based on current situation ... if its
time to sell or buy - Make the highlighted row RED or GREEN - if neutral keep
blue"*. The rules below are the ones settled in that conversation and printed
on reminder cards 09 and 10 — not a new method.

THE ANCHORS
-----------
A  the MEANINGFUL swing low: the base of the most recent big correction
   that (1) was deep — at least `thr`, scaled to the stock's own volatility,
   so a 15% dip counts on SPY and only a ~30% one on MU; (2) was COMPLETED —
   price later went above the peak it corrected from; and (3) has NEVER BEEN
   UNDERCUT since. Rule 3 is the BTC lesson: 74.4k was the obvious base until
   price fell to 55k, at which point the structure built on it was broken and
   the next base down (49.1k) is the one that still holds.
B  the highest high after A.
C  the lowest low after B — the pullback. Blank while B is the latest bar.

  "It is an art — nobody knows which fib will hold." These are the anchors a
  careful person would most often pick, chosen the same way every time so the
  levels do not move because of a mood. They are a cross-check on his own
  swing lines, never a substitute for them.

THE BUY LADDER  (retracement of A→B, card 09)
    0.382  0.5  0.618  0.786  1.0   →  10 / 25 / 50 / 75 / 100 % deployed
    1.0 is NOT a floor: below A the fib has failed.

THE TRIM LADDER  (trend-based extension, card 10)
    1.272  1.618  2.0  2.618        →  10 / 25 / 35 / 50 % of the non-core
  Projected from C when the pullback is real (as deep as the correction that
  qualified A). While the stock is still running there is no real C, so it is
  the PREVIOUS swing (A0 → the high A corrected from) launched from A — which
  is the extension that actually caps a stock in price discovery. Measured on
  MU 2026: from A0 193 / peak 471 / A 312 it put 1.618 at ~762 and 2.618 at
  ~1,040; the stock topped at 1,255.

Public API mirrors ath.py:
    compute_fib(ohlc)                        -> dict of Fib_* fields
    classify(fields, price)                  -> (action, text)
    attach_fib_columns(df, tickers, ...)     -> (df, message)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

RETRACE = (0.382, 0.5, 0.618, 0.786, 1.0)
EXTEND = (1.272, 1.618, 2.0, 2.618)
# Cumulative share of the planned cash / of the non-core slice, by the deepest
# level reached. Straight off cards 09 and 10.
DEPLOYED = {0.382: 10, 0.5: 25, 0.618: 50, 0.786: 75, 1.0: 100}
TRIMMED = {1.272: 10, 1.618: 25, 2.0: 35, 2.618: 50}

PERIOD = "5y"            # needs the swing BEFORE A for the near-high case
PEAK_LOOKBACK = 126      # bars searched for the high a trough corrected from
PIVOT_W = 5              # a trough must be the lowest within ±5 bars

FIB_COLS = ["Fib_A", "Fib_A_Date", "Fib_B", "Fib_B_Date", "Fib_C",
            "Fib_C_Date", "Fib_382", "Fib_500", "Fib_618", "Fib_786",
            "Fib_1000", "Fib_Ext_From", "Fib_E1272", "Fib_E1618",
            "Fib_E2000", "Fib_E2618"]

_R_COL = dict(zip(RETRACE, ["Fib_382", "Fib_500", "Fib_618", "Fib_786",
                            "Fib_1000"]))
_E_COL = dict(zip(EXTEND, ["Fib_E1272", "Fib_E1618", "Fib_E2000",
                           "Fib_E2618"]))


def _r(x: float) -> float:
    """Rounded for display: cents on a stock, whole units on BTC."""
    return round(float(x), 2 if abs(x) < 10_000 else 0)


def _threshold(close: pd.Series) -> float:
    """How deep a correction must be to count as MEANINGFUL, from the
    stock's own volatility. Half the annualised vol, between 15% and 30%."""
    rets = np.log(close).diff().dropna().tail(252)
    vol = float(rets.std() * math.sqrt(252)) if len(rets) > 20 else 0.4
    return min(0.30, max(0.15, 0.5 * vol))


def _troughs(high: np.ndarray, low: np.ndarray, thr: float) -> list[dict]:
    """Every confirmed swing low that was a correction of at least `thr`,
    oldest first, each with the peak it corrected from."""
    out = []
    n = len(low)
    for j in range(PIVOT_W, n - PIVOT_W):
        if low[j] != low[j - PIVOT_W: j + PIVOT_W + 1].min():
            continue
        # The peak it corrected FROM is the high since price was last this
        # low — not simply the highest bar in the window. Otherwise a minor
        # higher low just after a crash inherits the crash's depth: GOOG's
        # July-2025 174 was scored against February's 207 and beat April's
        # 142, the real base, as "the most recent big correction".
        lo = max(0, j - PEAK_LOOKBACK)
        under = np.nonzero(low[lo:j] < low[j])[0]
        if under.size:
            lo = lo + int(under[-1]) + 1
        if j - lo < 2:
            continue
        pk = lo + int(np.argmax(high[lo:j]))
        if 1.0 - low[j] / high[pk] < thr:
            continue
        out.append({"bar": j, "low": float(low[j]),
                    "peak_bar": pk, "peak": float(high[pk])})
    return out


def compute_fib(ohlc: pd.DataFrame) -> dict:
    """Fib_* fields for one ticker's daily bars, or {} when there is no
    swing structure to anchor to (a stock that has only gone down)."""
    df = ohlc[["High", "Low", "Close"]].dropna()
    # 60 bars, not a year: CBRS, SKHY and SPCX are held and are all under
    # six months old. With less history the search simply looks back less.
    if len(df) < 60:
        return {}
    high, low = df["High"].to_numpy(float), df["Low"].to_numpy(float)
    dates = [d.strftime("%Y-%m-%d") for d in df.index]
    n = len(df)
    thr = _threshold(df["Close"])
    tr = _troughs(high, low, thr)

    def valid(t, upto=None):
        """Completed (later beat its peak) and never undercut afterwards."""
        end = n if upto is None else upto
        later_hi = high[t["bar"] + 1: end]
        later_lo = low[t["bar"] + 1: end]
        return (later_hi.size and later_hi.max() > t["peak"]
                and (later_lo.size == 0 or later_lo.min() > t["low"]))

    a = next((t for t in reversed(tr) if valid(t)), None)
    if a is None:
        # No qualifying correction: fall back to the window's lowest low,
        # but only if a real move has been made off it since.
        j = int(np.argmin(low))
        if j >= n - 1 or high[j + 1:].max() / low[j] - 1.0 < thr:
            return {}
        a = {"bar": j, "low": float(low[j]), "peak_bar": None, "peak": None}

    A = a["low"]
    b_bar = a["bar"] + 1 + int(np.argmax(high[a["bar"] + 1:]))
    B = float(high[b_bar])
    leg = B - A
    if leg <= 0:
        return {}

    C = c_bar = None
    if b_bar < n - 1:
        c_bar = b_bar + 1 + int(np.argmin(low[b_bar + 1:]))
        C = float(low[c_bar])

    out = {"Fib_A": _r(A), "Fib_A_Date": dates[a["bar"]],
           "Fib_B": _r(B), "Fib_B_Date": dates[b_bar],
           "Fib_C": _r(C) if C is not None else np.nan,
           "Fib_C_Date": dates[c_bar] if c_bar is not None else ""}
    for r, col in _R_COL.items():
        out[col] = _r(B - r * leg)

    # TRIM projection. A real pullback → trend-based from C. Still at the
    # highs → the previous swing, launched from A.
    # "Real" means a correction as deep as the one that made A — the same
    # volatility-scaled bar. A fraction-of-the-leg test flipped MU's trims
    # from 975 to 1,844 on a 22% June dip mid-rally, turning a TRIM blue for
    # a fortnight and back again.
    if C is not None and (B - C) / B >= thr:
        base, size, how = C, leg, "C"
    else:
        prior = None
        if a.get("peak_bar") is not None:
            prior = next((t for t in reversed(tr)
                          if t["bar"] < a["peak_bar"] and valid(t, a["bar"] + 1)
                          and t["low"] < A), None)
            if prior is None:
                j0 = int(np.argmin(low[: a["peak_bar"]])) if a["peak_bar"] else None
                if j0 is not None and low[j0] < A:
                    prior = {"low": float(low[j0])}
        if prior is not None:
            base, size, how = A, a["peak"] - prior["low"], "prior swing"
        else:
            base, size, how = A, leg, "A→B"          # plain 2-point extension
    out["Fib_Ext_From"] = how
    for k, col in _E_COL.items():
        out[col] = _r(base + k * size)
    return out


def classify(f, price) -> tuple[str, str]:
    """(BUY | TRIM | HOLD | FAILED | "", one line of why) at `price`.

    `f` is anything with .get — a signals row or the dict compute_fib
    returned. Blank action when the fields are missing.
    """
    def g(k):
        try:
            v = float(f.get(k))
            return v if math.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    A, B = g("Fib_A"), g("Fib_B")
    try:
        P = float(price)
    except (TypeError, ValueError):
        P = None
    if A is None or B is None or P is None or not math.isfinite(P):
        return "", ""
    lv = {r: g(c) for r, c in _R_COL.items()}
    ex = {k: g(c) for k, c in _E_COL.items()}
    fmt = (lambda x: f"{x:,.0f}") if B >= 10_000 else (lambda x: f"{x:,.2f}")

    if P < A:
        return ("FAILED", f"below A {fmt(A)} — this fib has failed: redraw, "
                          f"and check the business before any deep buy")
    if lv[0.382] is not None and P <= lv[0.382]:
        hit = max(r for r in RETRACE if lv[r] is not None and P <= lv[r])
        nxt = next((r for r in RETRACE if r > hit), None)
        return ("BUY", f"past {hit:g} ({fmt(lv[hit])}) — ladder "
                       f"{DEPLOYED[hit]}% deployed"
                       + (f" · next {nxt:g} @ {fmt(lv[nxt])}" if nxt else "")
                       + (" · 1.0 is not a floor: business check first"
                          if hit >= 0.786 else ""))
    if ex[1.272] is not None and P >= ex[1.272]:
        hit = max(k for k in EXTEND if ex[k] is not None and P >= ex[k])
        nxt = next((k for k in EXTEND if k > hit), None)
        return ("TRIM", f"past {hit:g} ext ({fmt(ex[hit])}) — "
                        f"{TRIMMED[hit]}% of the non-core trimmed, core kept"
                        + (f" · next {nxt:g} @ {fmt(ex[nxt])}" if nxt else ""))
    # Between the ladders. Say which side is nearer.
    note = (f"between ladders — buy starts {fmt(lv[0.382])}, "
            f"trim starts {fmt(ex[1.272])}")
    C = g("Fib_C")
    if C is not None and P >= 0.97 * B:
        note = (f"back at the old high {fmt(B)} — optional small trim if it "
                f"stalls; planned trims start {fmt(ex[1.272])}")
    return "HOLD", note


def ladder_text(f, price) -> str:
    """Every level on ONE line, for the Dashboard's FIB row.

    Chakravarti, 2026-11-23: *"give me all level on that line ... separate
    with semi colon"*. Per-tranche sizes, not the cumulative ones on the rows
    below — the line answers "how much do I buy HERE": cards 09 and 10 as
    10/15/25/25/25 in and 10/15/10/15 of the non-core out. ✓ marks a level
    price has reached, ◀ next the one it has not.
    """
    def g(k):
        try:
            v = float(f.get(k))
            return v if math.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    try:
        P = float(price)
    except (TypeError, ValueError):
        P = None
    B = g("Fib_B")
    if B is None:
        return ""
    fmt = (lambda x: f"{x:,.0f}") if B >= 10_000 else (lambda x: f"{x:,.2f}")

    def side(levels, cols, sizes, reached, _sign):
        parts, flagged = [], False
        for lvl, col, size in zip(levels, cols, sizes):
            x = g(col)
            if x is None:
                continue
            mark = ""
            if P is not None and reached(P, x):
                mark = " ✓"
            elif not flagged:
                mark, flagged = " ◀ next", True
            tail = " only if thesis holds" if lvl == 1.0 else ""
            name = f"{lvl:.1f}" if lvl in (1.0, 2.0) else f"{lvl:g}"
            parts.append(f"{name} @ {fmt(x)} → {size}%{tail}{mark}")
        return "; ".join(parts)

    buy = side(RETRACE, [_R_COL[r] for r in RETRACE], (10, 15, 25, 25, 25),
               lambda p, x: p <= x, "")
    trim = side(EXTEND, [_E_COL[k] for k in EXTEND], (10, 15, 10, 15),
                lambda p, x: p >= x, "")
    return f"BUY: {buy}   ‖   TRIM: {trim}"


def _ohlc_for(raw, ticker: str):
    if raw is None or len(raw) == 0:
        return None
    cols = raw.columns
    if isinstance(cols, pd.MultiIndex):
        for lvl in (0, 1):
            if ticker in set(cols.get_level_values(lvl)):
                return raw.xs(ticker, axis=1, level=lvl)
        return None
    return raw


def compute_fib_table(tickers, log_fn=print, chunk_size: int = 40) -> pd.DataFrame:
    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    rows = []
    if yf is None:
        log_fn("⚠️  FIB: yfinance not available — skipping")
        return pd.DataFrame(columns=["Ticker"] + FIB_COLS)
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            raw = yf.download(chunk, period=PERIOD, interval="1d",
                              progress=False, auto_adjust=False,
                              group_by="ticker", threads=True)
        except Exception as e:                          # noqa: BLE001
            log_fn(f"⚠️  FIB: batch download failed ({len(chunk)}): {e}")
            raw = None
        for t in chunk:
            try:
                sub = _ohlc_for(raw, t)
                if sub is None or sub.dropna(how="all").empty:
                    sub = _ohlc_for(yf.download(t, period=PERIOD, interval="1d",
                                                progress=False,
                                                auto_adjust=False), t)
                f = compute_fib(sub) if sub is not None else {}
            except Exception as e:                      # noqa: BLE001
                log_fn(f"⚠️  FIB: {t} failed: {e}")
                f = {}
            rows.append({"Ticker": t, **f})
    return pd.DataFrame(rows)


def attach_fib_columns(df: pd.DataFrame, tickers, log_fn=print):
    """Merge the Fib_* columns onto the signals table, appended at the END —
    they are read by name by the Dashboard, never by position."""
    if "Ticker" not in df.columns:
        return df, "⚠️  FIB: no 'Ticker' column — skipped"
    fib = compute_fib_table(tickers, log_fn=log_fn)
    if fib.empty:
        return df, "⚠️  FIB: no data computed"
    keep = ["Ticker"] + [c for c in FIB_COLS if c in fib.columns]
    out = df.drop(columns=[c for c in FIB_COLS if c in df.columns])
    out = out.merge(fib[keep], on="Ticker", how="left")
    n = int(out["Fib_A"].notna().sum()) if "Fib_A" in out.columns else 0
    return out, f"FIB: {n}/{len(df)} tickers anchored"


if __name__ == "__main__":                              # quick look
    import sys
    for t in sys.argv[1:] or ["MU", "GOOG", "BE", "BTC-USD"]:
        raw = yf.download(t, period=PERIOD, interval="1d", progress=False,
                          auto_adjust=False)
        f = compute_fib(_ohlc_for(raw, t))
        last = float(_ohlc_for(raw, t)["Close"].dropna().iloc[-1])
        print(t, f"last {last:,.2f}", classify(f, last))
        print("   ", {k: f.get(k) for k in FIB_COLS})
