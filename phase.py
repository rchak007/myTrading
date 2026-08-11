#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase.py — Wyckoff-style market phase classifier  →  column: SmartMoneyPhase

Labels each ticker ACCUMULATION / MARKUP / DISTRIBUTION / MARKDOWN / UNDEFINED
by scoring five conditions per phase and taking the best scorer.

NOTHING here re-implements an indicator. Everything is derived from code we
already own:

    hhll.fetch_daily()            → daily OHLCV (same 420d window as HHLL)
    hhll.load_app_lists()         → STOCK_TICKERS / CRYPTO_TICKERS from app.py
                                     (identical to how jobStocksSignals.py and
                                      jobCryptoSignals.py load them)
    hhll.compute_hhll()           → per-bar Regime + HH/HL/LH/LL pivot labels
    core.indicators.apply_indicators()
                                  → Supertrend, ADXR, MRC bands/zone, Avg_Volume
    core.config.INDICATOR_PARAMS  → shared params (filtered to what
                                     apply_indicators actually accepts)

Public API mirrors hhll.py / ath.py:
    compute_phase_table(tickers)                  -> DataFrame[Ticker + PHASE_MERGE_COLS]
    merge_phase_columns(df, phase_df, after=...)  -> (df, message)
    attach_phase_columns(df, tickers, after=...)  -> (df, message)

CLI:
    python phase.py                       # stocks + crypto from app.py
    python phase.py --stocks-only
    python phase.py --crypto-only
    python phase.py --tickers MU,NVDA,AVGO
    python phase.py --tickers NVDA --explain   # per-condition breakdown

Outputs (CLI only):
    ../jobMyTrading/outputs/phase/phase_stocks.csv
    ../jobMyTrading/outputs/phase/phase_crypto.csv
"""

from __future__ import annotations

import sys
import argparse
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

OUT_DIR = APP_DIR / "../jobMyTrading/outputs/phase"

# ── Reuse: data fetch, ticker lists, structure engine ──────────────────────────
from hhll import (                     # noqa: E402
    fetch_daily,
    load_app_lists,
    compute_hhll,
    DEFAULT_PERIOD as HHLL_PERIOD,
    DEFAULT_ENGINE as HHLL_ENGINE,
)

# ── Reuse: indicators + shared params ─────────────────────────────────────────
from core.indicators import apply_indicators          # noqa: E402
try:
    from core.config import INDICATOR_PARAMS          # noqa: E402
except Exception:                                     # pragma: no cover
    INDICATOR_PARAMS = {}


# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════

PHASE_COL = "SmartMoneyPhase"
PHASE_MERGE_COLS = [PHASE_COL, "Phase_Confidence", "Bars_In_Phase", "Phase_Why"]

PHASES = ["ACCUMULATION", "MARKUP", "DISTRIBUTION", "MARKDOWN"]

# Tie-break order — with the mandatory gates below, ties are rare; when they
# happen the trend reads win over the transition reads.
PHASE_PRIORITY = ["MARKUP", "MARKDOWN", "DISTRIBUTION", "ACCUMULATION"]

# Emoji so the phase survives raw-CSV / GitHub previews (same convention as MRC_Zone).
PHASE_EMOJI = {
    "ACCUMULATION": "🟦",
    "MARKUP":       "🟩",
    "DISTRIBUTION": "🟧",
    "MARKDOWN":     "🟥",
    "UNDEFINED":    "⬜",
}

MIN_CONFIDENCE   = 0.60   # < 3 of 5 conditions → UNDEFINED
PCT_WINDOW       = 250    # lookback for band-width / volume-erratic percentiles
ADXR_SLOPE_BARS  = 5      # ADXR rising/falling measured over this many bars
COMPRESSED_RANK  = 0.35   # band width in bottom 35% → range contraction
ERRATIC_RANK     = 0.60   # volume-ratio stdev in top 40% → erratic volume
BODY_SHRINK      = 0.90   # 10-bar mean body/range vs 50-bar mean
NEAR_HIGH_TOL    = 0.98   # within 2% of the 60-bar high
HIGH_LOOKBACK    = 60


def _indicator_kwargs() -> dict:
    """INDICATOR_PARAMS filtered to the keys apply_indicators actually accepts.

    INDICATOR_PARAMS carries signal-layer keys (vol_multiplier, rsi_buy_threshold)
    that apply_indicators doesn't take — same filtering data/stocks.py does.
    """
    accepted = set(inspect.signature(apply_indicators).parameters)
    return {k: v for k, v in INDICATOR_PARAMS.items() if k in accepted}


def _adxr_low() -> float:
    return float(INDICATOR_PARAMS.get("adxr_low_threshold", 20.0))


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _rolling_pct_rank(s: pd.Series, window: int = PCT_WINDOW) -> pd.Series:
    """Percentile rank of each value within its trailing window (0..1)."""
    n = len(s)
    w = int(min(window, max(30, n // 2)))
    minp = max(20, w // 4)
    try:
        return s.rolling(w, min_periods=minp).rank(pct=True)
    except Exception:  # pandas < 1.4 has no Rolling.rank
        return s.rolling(w, min_periods=minp).apply(
            lambda x: float((x[:-1] <= x[-1]).mean()), raw=True
        )


def _per_bar_labels(res, n: int) -> tuple[pd.Series, pd.Series]:
    """
    Forward-fill HHLL pivot labels by their CONFIRMATION bar so every bar knows
    the last two structure labels that were knowable at that time (no lookahead).
    Reuses compute_hhll's pivots — no pivot logic re-implemented here.
    """
    last = [""] * n
    prev = [""] * n
    cur_last, cur_prev = "", ""
    by_conf: dict[int, list[str]] = {}
    for p in res.pivots:
        if p.label in ("HH", "HL", "LH", "LL"):
            by_conf.setdefault(p.confirmed_bar, []).append(p.label)
    for t in range(n):
        for lab in by_conf.get(t, []):
            cur_prev, cur_last = cur_last, lab
        last[t], prev[t] = cur_last, cur_prev
    return pd.Series(last), pd.Series(prev)


def _structure(last: pd.Series, prev: pd.Series) -> pd.Series:
    """BULLISH / BEARISH / MIXED / UNDEFINED from the last two labels."""
    bull = last.isin(["HH", "HL"]) & prev.isin(["HH", "HL"])
    bear = last.isin(["LH", "LL"]) & prev.isin(["LH", "LL"])
    both = (last != "") & (prev != "")
    out = pd.Series("UNDEFINED", index=last.index, dtype=object)
    out[both] = "MIXED"
    out[bull] = "BULLISH"
    out[bear] = "BEARISH"
    return out


# ═══════════════════════════════════════════════════════════════════
# Core: per-bar phase classification
# ═══════════════════════════════════════════════════════════════════

def classify_phases(df: pd.DataFrame,
                    prd: int = HHLL_PERIOD,
                    engine: str = HHLL_ENGINE) -> pd.DataFrame:
    """
    df: daily OHLCV (Open, High, Low, Close, Volume) — e.g. from hhll.fetch_daily.

    Returns a DataFrame indexed like df with:
        SmartMoneyPhase, Phase_Confidence, Phase_Why
        + one score column per phase (Score_ACCUMULATION, ...)
    """
    ind = apply_indicators(df, **_indicator_kwargs())
    res = compute_hhll(df, prd=prd, engine=engine)
    n = len(df)

    idx = df.index
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)

    regime = pd.Series(res.regime, index=idx)
    last_lab, prev_lab = _per_bar_labels(res, n)
    last_lab.index, prev_lab.index = idx, idx
    struct = _structure(last_lab, prev_lab)

    adxr = pd.to_numeric(ind["ADXR"], errors="coerce")
    adxr_slope = adxr.diff(ADXR_SLOPE_BARS)
    adxr_low = _adxr_low()

    zone = ind["MRC_Zone"].astype(str)
    dist = pd.to_numeric(ind["MRC_Dist_Pct"], errors="coerce")
    band_w = (ind["MRC_R1"] - ind["MRC_S1"]) / ind["MRC_Mean"]
    bw_rank = _rolling_pct_rank(band_w)

    vol_ratio = pd.to_numeric(ind["Volume"], errors="coerce") / pd.to_numeric(
        ind["Avg_Volume"], errors="coerce")
    vol_erratic_rank = _rolling_pct_rank(vol_ratio.rolling(10).std())

    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs() / rng
    body_shrink = body.rolling(10).mean() / body.rolling(50).mean()

    st_buy = ind["Supertrend_Signal"].astype(str).eq("BUY")
    near_high = c >= (c.rolling(HIGH_LOOKBACK, min_periods=10).max() * NEAR_HIGH_TOL)

    below_zone = zone.isin(["Below_Mean", "Near_Mean", "OS", "Strong_OS"])
    above_zone = zone.isin(["OB", "Strong_OB"])

    # ── Condition sets (5 per phase, equal weight) ───────────────────────────
    conds: dict[str, dict[str, pd.Series]] = {
        "ACCUMULATION": {
            "adxr_low":    adxr < adxr_low,
            "range_tight": bw_rank <= COMPRESSED_RANK,
            "at_lows":     below_zone,
            "no_new_lows": last_lab.ne("LL"),
            "vol_dry":     vol_ratio < 1.0,
        },
        "MARKUP": {
            "st_buy":      st_buy,
            "regime_bull": regime.eq("BULL"),
            "adxr_up":     (adxr > adxr_low) & (adxr_slope > 0),
            "struct_bull": struct.eq("BULLISH"),
            "above_mean":  dist > 0,
        },
        "DISTRIBUTION": {
            "at_highs":    near_high | above_zone,
            "adxr_fade":   (adxr > adxr_low) & (adxr_slope < 0),
            "body_shrink": body_shrink < BODY_SHRINK,
            "vol_erratic": vol_erratic_rank >= ERRATIC_RANK,
            "lower_high":  last_lab.eq("LH") | struct.eq("MIXED"),
        },
        "MARKDOWN": {
            "st_sell":     ~st_buy,
            "regime_bear": regime.eq("BEAR"),
            "struct_bear": struct.eq("BEARISH"),
            "below_mean":  dist < 0,
            "impulse_dn":  (adxr > adxr_low) & (adxr_slope > 0),
        },
    }

    # Mandatory gates — a phase cannot be claimed out of its own half of the
    # cycle. Without these, DISTRIBUTION steals bars from a healthy MARKUP
    # (any pullback looks like "erratic volume + shrinking bodies").
    gates: dict[str, pd.Series] = {
        "ACCUMULATION": ~near_high,
        "MARKUP":       st_buy,
        "DISTRIBUTION": near_high | above_zone,
        "MARKDOWN":     ~st_buy,
    }

    scores = pd.DataFrame(
        {p: (sum(s.fillna(False).astype(int) for s in d.values()) / len(d))
             * gates[p].fillna(False).astype(int)
         for p, d in conds.items()},
        index=idx,
    )

    ranked = sorted(PHASES, key=lambda p: PHASE_PRIORITY.index(p))
    best_idx = scores[ranked].to_numpy().argmax(axis=1)
    best = pd.Series([ranked[i] for i in best_idx], index=idx)
    conf = pd.Series(scores[ranked].to_numpy().max(axis=1), index=idx)

    phase = best.where(conf >= MIN_CONFIDENCE, "UNDEFINED")

    why = pd.Series("", index=idx, dtype=object)
    for p in PHASES:
        hit = phase.eq(p)
        if not hit.any():
            continue
        parts = pd.DataFrame(
            {k: v.fillna(False) for k, v in conds[p].items()}, index=idx)
        why[hit] = parts[hit].apply(
            lambda r: "+".join([k for k, ok in r.items() if ok]), axis=1)

    out = pd.DataFrame({
        PHASE_COL: phase,
        "Phase_Confidence": conf.round(2),
        "Phase_Why": why,
    }, index=idx)
    for p in PHASES:
        out[f"Score_{p}"] = scores[p].round(2)
    return out


def summarize_phase(phases: pd.DataFrame) -> dict:
    """Last-bar summary row: phase, confidence, and consecutive bars in it."""
    if phases.empty:
        return {PHASE_COL: "UNDEFINED", "Phase_Confidence": np.nan,
                "Bars_In_Phase": np.nan, "Phase_Why": ""}
    ser = phases[PHASE_COL]
    cur = str(ser.iloc[-1])
    bars = 0
    for v in ser.iloc[::-1]:
        if str(v) != cur:
            break
        bars += 1
    return {
        PHASE_COL:          cur,
        "Phase_Confidence": float(phases["Phase_Confidence"].iloc[-1]),
        "Bars_In_Phase":    int(bars),
        "Phase_Why":        str(phases["Phase_Why"].iloc[-1]),
    }


# ═══════════════════════════════════════════════════════════════════
# Callable API — what the cron jobs import
# ═══════════════════════════════════════════════════════════════════

def compute_phase_table(tickers: "list[str]",
                        prd: int = HHLL_PERIOD,
                        engine: str = HHLL_ENGINE,
                        verbose: bool = False,
                        log_fn=None) -> pd.DataFrame:
    """
    One row per ticker: Ticker + PHASE_MERGE_COLS. Fetch failures are skipped
    (logged if verbose). No files read or written.
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
            ph = classify_phases(dfd, prd=prd, engine=engine)
            rows.append({"Ticker": tk, **summarize_phase(ph)})
        except Exception as e:
            emit(f"  [{i:>3}/{total}] {tk:<14} — phase compute failed ({e})")
    return pd.DataFrame(rows)


def merge_phase_columns(df: pd.DataFrame,
                        phase_df: pd.DataFrame,
                        after: str = "VALUE",
                        decorate: bool = True) -> "tuple[pd.DataFrame, str]":
    """
    Left-merge a precomputed phase table onto a signals DataFrame on 'Ticker',
    positioning the block right after `after` (appended if `after` is absent).

    `after` may be a single name or a list of candidates (first match wins),
    matching ath.attach_ath_columns' behaviour.

    Never raises — returns df unchanged with an explanatory message on failure.
    """
    if "Ticker" not in df.columns:
        return df, "Phase merge skipped: signals table has no 'Ticker' column."
    if phase_df is None or phase_df.empty or "Ticker" not in phase_df.columns:
        return df, "Phase merge skipped: phase table is empty."

    keep = ["Ticker"] + [c for c in PHASE_MERGE_COLS if c in phase_df.columns]
    ph = phase_df[keep].drop_duplicates(subset="Ticker", keep="last").copy()

    if decorate and PHASE_COL in ph.columns:
        ph[PHASE_COL] = ph[PHASE_COL].map(decorate_phase)

    overlap = [c for c in PHASE_MERGE_COLS if c in df.columns]
    if overlap:
        df = df.drop(columns=overlap)

    merged = df.merge(ph, on="Ticker", how="left")

    new_cols = [c for c in PHASE_MERGE_COLS if c in merged.columns]
    base = [c for c in merged.columns if c not in new_cols]

    candidates = [after] if isinstance(after, str) else list(after)
    anchor = next((c for c in candidates if c in base), None)
    idx = base.index(anchor) + 1 if anchor else len(base)
    new_order = base[:idx] + new_cols + base[idx:]

    matched = int(merged[new_cols[0]].notna().sum()) if new_cols else 0
    where = anchor if anchor else f"end ('{candidates[0]}' not found)"
    return (merged[new_order],
            f"SmartMoneyPhase merged: {len(new_cols)} cols, {matched}/{len(merged)} "
            f"tickers matched, inserted after {where}.")


def attach_phase_columns(df: pd.DataFrame,
                         tickers: "list[str]",
                         after: str = "VALUE",
                         prd: int = HHLL_PERIOD,
                         engine: str = HHLL_ENGINE,
                         decorate: bool = True,
                         verbose: bool = False,
                         log_fn=None) -> "tuple[pd.DataFrame, str]":
    """
    One-call convenience for the cron jobs:

        from phase import attach_phase_columns
        df, msg = attach_phase_columns(df, STOCK_TICKERS, after="VALUE", log_fn=log)
    """
    phase_df = compute_phase_table(tickers, prd=prd, engine=engine,
                                   verbose=verbose, log_fn=log_fn)
    return merge_phase_columns(df, phase_df, after=after, decorate=decorate)


def decorate_phase(val: object) -> str:
    """'MARKUP' → '🟩 MARKUP' (mirrors _decorate_mrc_zone in jobStocksSignals.py)."""
    s = "" if val is None else str(val)
    emoji = PHASE_EMOJI.get(s, "")
    return f"{emoji} {s}".strip() if emoji else s


def phase_key(val: object) -> str:
    """Recover the raw phase from a possibly-decorated value."""
    s = "" if val is None else str(val)
    for emoji in PHASE_EMOJI.values():
        if s.startswith(emoji + " "):
            return s[len(emoji) + 1:]
    return s


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def scan(tickers: list[str], prd: int, engine: str, tag: str,
         explain: bool = False) -> pd.DataFrame:
    rows = []
    total = len(tickers)
    for i, tk in enumerate(tickers, 1):
        dfd = fetch_daily(tk)
        if dfd is None:
            print(f"  [{i:>3}/{total}] {tk:<14} — no data, skipped")
            continue
        try:
            ph = classify_phases(dfd, prd=prd, engine=engine)
            row = {"Ticker": tk, **summarize_phase(ph)}
            rows.append(row)
            print(f"  [{i:>3}/{total}] {tk:<14} {row[PHASE_COL]:<13} "
                  f"conf={row['Phase_Confidence']:.2f} bars={row['Bars_In_Phase']:<4} "
                  f"{row['Phase_Why']}")
            if explain:
                cols = [PHASE_COL, "Phase_Confidence"] + [f"Score_{p}" for p in PHASES]
                print(ph[cols].tail(10).to_string())
        except Exception as e:
            print(f"  [{i:>3}/{total}] {tk:<14} — compute failed ({e})")
    out = pd.DataFrame(rows)
    if not out.empty:
        out.insert(1, "List", tag)
    return out


def main():
    ap = argparse.ArgumentParser(description="Wyckoff-style SmartMoneyPhase scanner")
    ap.add_argument("--period", type=int, default=HHLL_PERIOD,
                    help=f"HHLL pivot period (default {HHLL_PERIOD})")
    ap.add_argument("--engine", choices=["window", "bos"], default=HHLL_ENGINE)
    ap.add_argument("--tickers", type=str, default=None,
                    help="Comma-separated ad-hoc list (skips app.py)")
    ap.add_argument("--stocks-only", action="store_true")
    ap.add_argument("--crypto-only", action="store_true")
    ap.add_argument("--explain", action="store_true",
                    help="Print the last 10 bars of per-phase scores")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSmartMoneyPhase scanner — period={args.period}, engine={args.engine}\n"
          + "=" * 70)

    if args.tickers:
        lst = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        df = scan(lst, args.period, args.engine, "ADHOC", explain=args.explain)
        p = OUT_DIR / "phase_adhoc.csv"
        df.to_csv(p, index=False)
        print(f"\n✅ {len(df)} rows -> {p}")
        return

    # Same source of truth as jobStocksSignals.py / jobCryptoSignals.py
    stock_list, crypto_list = load_app_lists()

    if not args.crypto_only:
        print(f"\nStocks ({len(stock_list)}):")
        df_s = scan(stock_list, args.period, args.engine, "STOCKS", explain=args.explain)
        p = OUT_DIR / "phase_stocks.csv"
        df_s.to_csv(p, index=False)
        print(f"✅ {len(df_s)} rows -> {p}")

    if not args.stocks_only:
        print(f"\nCrypto ({len(crypto_list)}):")
        df_c = scan(crypto_list, args.period, args.engine, "CRYPTO", explain=args.explain)
        p = OUT_DIR / "phase_crypto.csv"
        df_c.to_csv(p, index=False)
        print(f"✅ {len(df_c)} rows -> {p}")


if __name__ == "__main__":
    main()