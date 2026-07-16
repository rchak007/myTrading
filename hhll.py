#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hhll.py — Higher High / Lower Low structure scanner
(Python port of LonesomeTheBlue-style "Higher High Lower Low" TradingView indicator)

Logic:
  1. Pivot detection (two engines):
       - "window" (default): pivot high = highest of prd bars left + self + prd bars
         right. Confirmed prd bars after the pivot bar. Exact TradingView replica.
       - "bos": hybrid — same window rule, BUT a pending pivot confirms EARLY the
         moment a close breaks the prior opposite pivot (Break of Structure),
         mirroring rizzy.py's confirmation philosophy.
  2. Zigzag alternation: pivots must alternate H/L; same-direction extremes extend
     the current leg instead of creating a new one.
  3. Labels: each pivot compared to the previous pivot of the same type:
       pivot high  -> HH if higher than last pivot high,  else LH
       pivot low   -> HL if higher than last pivot low,   else LL
  4. S/R lines: every confirmed pivot spawns a level (high -> Resistance,
     low -> Support). A level is retired when a bar CLOSES across it
     (close-based, never wicks).
  5. Regime: close above an active Resistance -> BULL. Close below an active
     Support -> BEAR. Persists until the opposite break (= the indicator's
     persistent blue/dark bar coloring).

Usage:
    python hhll.py                          # stocks + crypto from app.py, prd=3
    python hhll.py --period 10              # TradingView default period
    python hhll.py --engine bos             # rizzy-style early confirmation
    python hhll.py --tickers MU,AVGO,NVDA   # ad-hoc list (skips app.py)
    python hhll.py --stocks-only / --crypto-only

Outputs:
    outputs/hhll_stocks.csv
    outputs/hhll_crypto.csv
"""

from __future__ import annotations

import sys
import argparse
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
OUT_DIR = APP_DIR / "../jobMyTrading/outputs/hhll"

DEFAULT_PERIOD = 2          # ZigZag Period (user default; TradingView default is 10)
DEFAULT_ENGINE = "window"   # "window" = TV replica | "bos" = rizzy-style hybrid
SCAN_PERIOD = "420d"        # yfinance history window (matches 45_Signal convention)

# HHLL structure columns exported for merging into the signals tables
# (jobStocksSignals.py / jobCryptoSignals.py). Order here is the display order.
HHLL_MERGE_COLS = [
    "Regime", "Structure", "Last_Label", "Last_Label_Price",
    "Prev_Label", "Prev_Label_Price", "Nearest_Support", "Nearest_Resistance",
    "Bars_Since_Flip", "Active_Supports", "Active_Resistances",
]


# ═══════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Pivot:
    bar: int            # bar index of the pivot itself
    price: float
    kind: str           # "H" or "L"
    confirmed_bar: int  # bar index at which the pivot became known
    label: str = ""     # HH / LH / HL / LL (filled during labeling)


@dataclass
class Level:
    price: float
    kind: str           # "R" (from pivot high) or "S" (from pivot low)
    born_bar: int       # bar the level became active (pivot confirmation bar)
    pivot_bar: int
    broken_bar: int | None = None   # bar whose close retired the level


@dataclass
class HHLLResult:
    pivots: list[Pivot] = field(default_factory=list)
    levels: list[Level] = field(default_factory=list)
    regime: list[str] = field(default_factory=list)       # per-bar: BULL/BEAR/NONE
    flip_bar: int | None = None                            # last regime flip bar
    events: list[dict] = field(default_factory=list)      # break events log


# ═══════════════════════════════════════════════════════════════════
# Pivot engines
# ═══════════════════════════════════════════════════════════════════

def _window_pivots(highs: np.ndarray, lows: np.ndarray, prd: int) -> list[Pivot]:
    """
    TradingView-style symmetric window pivots.
    Pivot high at i: high[i] >= max(high[i-prd : i+prd+1])   (ties allowed, like Pine)
    Confirmed at bar i + prd.
    """
    pivots: list[Pivot] = []
    n = len(highs)
    for i in range(prd, n - prd):
        wh = highs[i - prd: i + prd + 1]
        wl = lows[i - prd: i + prd + 1]
        if highs[i] >= wh.max():
            pivots.append(Pivot(bar=i, price=float(highs[i]), kind="H",
                                confirmed_bar=i + prd))
        if lows[i] <= wl.min():
            pivots.append(Pivot(bar=i, price=float(lows[i]), kind="L",
                                confirmed_bar=i + prd))
    pivots.sort(key=lambda p: (p.bar, p.kind))
    return pivots


def _apply_bos_confirmation(pivots: list[Pivot], closes: np.ndarray) -> list[Pivot]:
    """
    Hybrid BOS confirmation (rizzy philosophy):
    a pending pivot can confirm EARLIER than bar+prd the moment a close breaks
    the prior opposite pivot's price.
      - pending pivot LOW confirms early when close > previous pivot HIGH price
      - pending pivot HIGH confirms early when close < previous pivot LOW price
    confirmed_bar = min(window confirmation, first BOS bar after the pivot).
    """
    out: list[Pivot] = []
    prev_h: Pivot | None = None
    prev_l: Pivot | None = None
    n = len(closes)
    for p in pivots:
        ref = prev_h if p.kind == "L" else prev_l
        if ref is not None:
            for t in range(p.bar + 1, min(p.confirmed_bar, n)):
                c = closes[t]
                if p.kind == "L" and c > ref.price:
                    p.confirmed_bar = t
                    break
                if p.kind == "H" and c < ref.price:
                    p.confirmed_bar = t
                    break
        if p.kind == "H":
            prev_h = p
        else:
            prev_l = p
        out.append(p)
    return out


def _zigzag_alternate(pivots: list[Pivot]) -> list[Pivot]:
    """
    Enforce alternation: consecutive same-kind pivots collapse to the more
    extreme one (higher price for H, lower price for L). The kept pivot keeps
    the EARLIEST confirmation knowledge that is still truthful: if a later,
    more extreme same-kind pivot replaces an earlier one, its own confirmation
    bar applies (we can't know the extension before it happens).
    """
    zz: list[Pivot] = []
    for p in pivots:
        if zz and zz[-1].kind == p.kind:
            last = zz[-1]
            if (p.kind == "H" and p.price >= last.price) or \
               (p.kind == "L" and p.price <= last.price):
                zz[-1] = p           # extend the leg with the more extreme pivot
            # else: weaker same-direction pivot -> ignore
        else:
            zz.append(p)
    return zz


def _label_pivots(zz: list[Pivot]) -> None:
    """HH/LH for highs, HL/LL for lows vs previous same-kind pivot (in place)."""
    last_h: Pivot | None = None
    last_l: Pivot | None = None
    for p in zz:
        if p.kind == "H":
            p.label = "H" if last_h is None else ("HH" if p.price > last_h.price else "LH")
            last_h = p
        else:
            p.label = "L" if last_l is None else ("HL" if p.price > last_l.price else "LL")
            last_l = p


# ═══════════════════════════════════════════════════════════════════
# Core: S/R lifecycle + regime state machine
# ═══════════════════════════════════════════════════════════════════

def compute_hhll(df: pd.DataFrame, prd: int = DEFAULT_PERIOD,
                 engine: str = DEFAULT_ENGINE) -> HHLLResult:
    """
    df: OHLC DataFrame with columns High, Low, Close (daily bars).
    Returns HHLLResult with pivots (labeled), levels, per-bar regime, events.
    """
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    n = len(df)

    raw = _window_pivots(highs, lows, prd)
    if engine == "bos":
        raw = _apply_bos_confirmation(raw, closes)
    zz = _zigzag_alternate(raw)
    _label_pivots(zz)

    res = HHLLResult(pivots=zz)

    # Map: confirmation bar -> pivots confirmed on that bar
    by_conf: dict[int, list[Pivot]] = {}
    for p in zz:
        by_conf.setdefault(p.confirmed_bar, []).append(p)

    active: list[Level] = []
    regime = "NONE"
    regimes: list[str] = []
    flip_bar: int | None = None

    for t in range(n):
        # 1) spawn levels for pivots confirmed on this bar
        for p in by_conf.get(t, []):
            active.append(Level(price=p.price,
                                kind="R" if p.kind == "H" else "S",
                                born_bar=t, pivot_bar=p.bar))
        # 2) close-based break check against every active level
        c = closes[t]
        still: list[Level] = []
        for lv in active:
            if lv.kind == "R" and c > lv.price:
                lv.broken_bar = t
                res.levels.append(lv)
                res.events.append({"bar": t, "type": "RES_BREAK",
                                   "level": lv.price, "close": c})
                if regime != "BULL":
                    regime, flip_bar = "BULL", t
            elif lv.kind == "S" and c < lv.price:
                lv.broken_bar = t
                res.levels.append(lv)
                res.events.append({"bar": t, "type": "SUP_BREAK",
                                   "level": lv.price, "close": c})
                if regime != "BEAR":
                    regime, flip_bar = "BEAR", t
            else:
                still.append(lv)
        active = still
        regimes.append(regime)

    res.levels.extend(active)           # unbroken levels (broken_bar=None)
    res.regime = regimes
    res.flip_bar = flip_bar
    return res


def summarize(df: pd.DataFrame, res: HHLLResult) -> dict:
    """One output row per ticker for the CSV."""
    n = len(df)
    close = float(df["Close"].iloc[-1])
    regime = res.regime[-1] if res.regime else "NONE"

    active = [lv for lv in res.levels if lv.broken_bar is None]
    sups = sorted((lv.price for lv in active if lv.kind == "S" and lv.price < close),
                  reverse=True)
    ress = sorted(lv.price for lv in active if lv.kind == "R" and lv.price > close)

    last = res.pivots[-1] if res.pivots else None
    prev = res.pivots[-2] if len(res.pivots) > 1 else None

    # Structure call from the last two labels (matches how we read the chart)
    labels = [p.label for p in res.pivots if p.label in ("HH", "HL", "LH", "LL")]
    struct = "UNDEFINED"
    if len(labels) >= 2:
        recent = set(labels[-2:])
        if recent <= {"HH", "HL"}:
            struct = "BULLISH"
        elif recent <= {"LH", "LL"}:
            struct = "BEARISH"
        else:
            struct = "MIXED"

    return {
        "Close": round(close, 2),
        "Regime": regime,
        "Structure": struct,
        "Last_Label": last.label if last else "",
        "Last_Label_Price": round(last.price, 2) if last else np.nan,
        "Prev_Label": prev.label if prev else "",
        "Prev_Label_Price": round(prev.price, 2) if prev else np.nan,
        "Nearest_Support": round(sups[0], 2) if sups else np.nan,
        "Nearest_Resistance": round(ress[0], 2) if ress else np.nan,
        "Bars_Since_Flip": (n - 1 - res.flip_bar) if res.flip_bar is not None else np.nan,
        "Active_Supports": len([lv for lv in active if lv.kind == "S"]),
        "Active_Resistances": len([lv for lv in active if lv.kind == "R"]),
    }


# ═══════════════════════════════════════════════════════════════════
# Callable API — compute HHLL in-process and fold it into a signals table
# (imported by jobStocksSignals.py / jobCryptoSignals.py — no CSV involved)
# ═══════════════════════════════════════════════════════════════════

def compute_hhll_table(tickers: "list[str]",
                       prd: int = DEFAULT_PERIOD,
                       engine: str = DEFAULT_ENGINE,
                       verbose: bool = False,
                       log_fn=None) -> pd.DataFrame:
    """
    Compute HHLL structure rows for a list of tickers, in-process.

    Fetches daily bars (yfinance), runs the pivot/regime engine, and summarizes
    one row per ticker. Returns a DataFrame with 'Ticker' + HHLL_MERGE_COLS
    (plus 'Close'). Tickers that fail to fetch/compute are skipped silently
    (or logged if verbose). No files are read or written.

    This is the same work `scan()` does for the CLI, minus the CSV write and the
    'List' tag column — factored out so the cron jobs can call it directly.
    """
    emit = log_fn or (print if verbose else (lambda *_: None))
    rows: list[dict] = []
    total = len(tickers)
    for i, tk in enumerate(tickers, 1):
        dfd = fetch_daily(tk)
        if dfd is None:
            emit(f"  [{i:>3}/{total}] {tk:<14} — no data, skipped")
            continue
        try:
            res = compute_hhll(dfd, prd=prd, engine=engine)
            rows.append({"Ticker": tk, **summarize(dfd, res)})
        except Exception as e:
            emit(f"  [{i:>3}/{total}] {tk:<14} — compute failed ({e})")
    return pd.DataFrame(rows)


def merge_hhll_columns(df: pd.DataFrame,
                       hhll_df: pd.DataFrame,
                       after: str = "VALUE") -> "tuple[pd.DataFrame, str]":
    """
    Left-merge a precomputed HHLL table (from compute_hhll_table) onto a signals
    DataFrame, keyed on 'Ticker', positioning the HHLL block immediately after
    the `after` column.

    Parameters
    ----------
    df       : signals table (must contain a 'Ticker' column).
    hhll_df  : output of compute_hhll_table (Ticker + HHLL columns).
    after    : column to insert the HHLL block right after. If absent from df,
               the block is appended at the end.

    Returns
    -------
    (df_out, message) : merged DataFrame + short status string for logging.
                        If df has no 'Ticker', or hhll_df is empty, df is returned
                        unchanged with an explanatory message — the job never breaks.
    """
    if "Ticker" not in df.columns:
        return df, "HHLL merge skipped: signals table has no 'Ticker' column."
    if hhll_df is None or hhll_df.empty or "Ticker" not in hhll_df.columns:
        return df, "HHLL merge skipped: HHLL table is empty."

    keep = ["Ticker"] + [c for c in HHLL_MERGE_COLS if c in hhll_df.columns]
    hh = hhll_df[keep].drop_duplicates(subset="Ticker", keep="last")

    # Let the HHLL columns own their requested names: drop any pre-existing
    # same-named columns from the signals table before merging.
    overlap = [c for c in HHLL_MERGE_COLS if c in df.columns]
    if overlap:
        df = df.drop(columns=overlap)

    merged = df.merge(hh, on="Ticker", how="left")

    # Reposition the HHLL block right after `after` (or append if `after` absent).
    hh_cols = [c for c in HHLL_MERGE_COLS if c in merged.columns]
    base = [c for c in merged.columns if c not in hh_cols]
    idx = base.index(after) + 1 if after in base else len(base)
    new_order = base[:idx] + hh_cols + base[idx:]

    matched = int(merged[hh_cols[0]].notna().sum()) if hh_cols else 0
    anchor = after if after in base else f"end ('{after}' not found)"
    return (merged[new_order],
            f"HHLL merged: {len(hh_cols)} cols, {matched}/{len(merged)} tickers "
            f"matched, inserted after {anchor}.")


def attach_hhll_columns(df: pd.DataFrame,
                        tickers: "list[str]",
                        after: str = "VALUE",
                        prd: int = DEFAULT_PERIOD,
                        engine: str = DEFAULT_ENGINE,
                        verbose: bool = False,
                        log_fn=None) -> "tuple[pd.DataFrame, str]":
    """
    One-call convenience: compute HHLL for `tickers` and merge it into `df`.

    This is what the cron jobs import:

        from hhll import attach_hhll_columns
        df, msg = attach_hhll_columns(df, STOCK_TICKERS, after="VALUE", log_fn=log)

    No CSV is read or written — everything happens in-process.
    """
    hhll_df = compute_hhll_table(tickers, prd=prd, engine=engine,
                                 verbose=verbose, log_fn=log_fn)
    return merge_hhll_columns(df, hhll_df, after=after)


# ═══════════════════════════════════════════════════════════════════
# Universe loading (same pattern as jobStocksSignals.py)
# ═══════════════════════════════════════════════════════════════════

def load_app_lists() -> tuple[list[str], list[str]]:
    app_py = APP_DIR / "app.py"
    spec = importlib.util.spec_from_file_location("mytrading_app", app_py)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)                      # type: ignore
        return (list(getattr(mod, "STOCK_TICKERS")),
                list(getattr(mod, "CRYPTO_TICKERS")))
    except Exception as e:
        raise RuntimeError(f"Could not load ticker lists from app.py: {e}")


def fetch_daily(ticker: str) -> pd.DataFrame | None:
    import yfinance as yf
    try:
        df = yf.download(ticker, period=SCAN_PERIOD, interval="1d",
                         auto_adjust=True, progress=False)
        if df is None or df.empty or len(df) < 30:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as e:
        print(f"  ⚠️  {ticker}: fetch failed ({e})")
        return None


def scan(tickers: list[str], prd: int, engine: str, tag: str) -> pd.DataFrame:
    rows = []
    total = len(tickers)
    for i, tk in enumerate(tickers, 1):
        df = fetch_daily(tk)
        if df is None:
            print(f"  [{i:>3}/{total}] {tk:<14} — no data, skipped")
            continue
        try:
            res = compute_hhll(df, prd=prd, engine=engine)
            row = {"Ticker": tk, **summarize(df, res)}
            rows.append(row)
            print(f"  [{i:>3}/{total}] {tk:<14} {row['Regime']:<5} "
                  f"{row['Structure']:<9} last={row['Last_Label']:<3}"
                  f"@{row['Last_Label_Price']}")
        except Exception as e:
            print(f"  [{i:>3}/{total}] {tk:<14} — compute failed ({e})")
    out = pd.DataFrame(rows)
    if not out.empty:
        out.insert(1, "List", tag)
    return out


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="HH/HL/LH/LL structure scanner")
    ap.add_argument("--period", type=int, default=DEFAULT_PERIOD,
                    help=f"ZigZag pivot period (default {DEFAULT_PERIOD})")
    ap.add_argument("--engine", choices=["window", "bos"], default=DEFAULT_ENGINE,
                    help="Pivot confirmation engine (default window = TV replica)")
    ap.add_argument("--tickers", type=str, default=None,
                    help="Comma-separated ad-hoc list (skips app.py)")
    ap.add_argument("--stocks-only", action="store_true")
    ap.add_argument("--crypto-only", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    print(f"\nHHLL scanner — period={args.period}, engine={args.engine}\n" + "=" * 60)

    if args.tickers:
        lst = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        df = scan(lst, args.period, args.engine, "ADHOC")
        path = OUT_DIR / "hhll_adhoc.csv"
        df.to_csv(path, index=False)
        print(f"\n✅ {len(df)} rows -> {path}")
        return

    stock_list, crypto_list = load_app_lists()

    if not args.crypto_only:
        print(f"\nStocks ({len(stock_list)}):")
        df_s = scan(stock_list, args.period, args.engine, "STOCKS")
        p = OUT_DIR / "hhll_stocks.csv"
        df_s.to_csv(p, index=False)
        print(f"✅ {len(df_s)} rows -> {p}")

    if not args.stocks_only:
        print(f"\nCrypto ({len(crypto_list)}):")
        df_c = scan(crypto_list, args.period, args.engine, "CRYPTO")
        p = OUT_DIR / "hhll_crypto.csv"
        df_c.to_csv(p, index=False)
        print(f"✅ {len(df_c)} rows -> {p}")


if __name__ == "__main__":
    main()