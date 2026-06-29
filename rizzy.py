#!/usr/bin/env python3
"""
rizzy.py — Swing-structure state scanner.

Classifies every ticker in STOCK_TICKERS and CRYPTO_TICKERS (loaded from
myTrading/app.py) into one of:

    RIZZY_UP    higher highs + higher lows, structure intact  (long setup)
    RIZZY_DOWN  lower highs  + lower lows,  structure intact  (short setup)
    NO_RIZZY    mixed / chopping / structure just broke
    NO_DATA     fetch failed or not enough bars

Swings are CONFIRMED PIVOTS — a bar that dominates `prd` bars on each side —
NOT single candles (same idea as data._find_pivots). The current state is read
off the two most recent confirmed swing highs and swing lows:

    RIZZY_DOWN  last 2 highs stepping down AND last 2 lows stepping down,
                AND price hasn't closed back above the most recent lower high.
    RIZZY_UP    mirror: last 2 highs up AND last 2 lows up,
                AND price hasn't closed below the most recent higher low.

A close beyond the most recent swing = break of structure = drops to NO_RIZZY.

Timeframes:   stocks -> 1d (fetch_stock_1d_df)   crypto -> 4h (fetch_crypto_4h_df)

Write-only: produces a CSV + console summary. No git push (the single pusher
owns git).

Usage:
    python rizzy.py                 # full scan, stocks + crypto
    python rizzy.py --crypto-only
    python rizzy.py --stocks-only
    python rizzy.py --limit 10      # first 10 of each (quick test)
    python rizzy.py --prd 5         # override pivot strength for both classes
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Paths — mirror jobStocksSignals.py / jobCryptoSignals.py
# ----------------------------------------------------------------------
MYTRADING_DIR = Path.home() / "github" / "myTrading"
JOB_DIR       = Path.home() / "github" / "jobMyTrading"
OUT_CSV       = JOB_DIR / "rizzy_signals.csv"
LOG_FILE      = JOB_DIR / "rizzy.log"

# ----------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------
# Pivot strength: a swing high must be the highest of `prd` bars on each side
# (mirror for swing lows). Higher prd = fewer, cleaner swings, but more lag
# (a pivot can't be confirmed until `prd` bars later).
# NOTE: 4h crypto is choppier than daily stocks. If crypto looks too twitchy,
# bump PIVOT_PRD_CRYPTO to 5.
PIVOT_PRD_STOCK  = 3
PIVOT_PRD_CRYPTO = 3

# Need at least this many confirmed highs AND lows before calling a direction
# (your "first low and first high, then continuation").
MIN_SWINGS = 2


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ======================================================================
# Core swing-structure logic  (pure, no I/O — unit-testable)
# ======================================================================
def find_swings(df: pd.DataFrame, prd: int) -> tuple[list[float], list[float]]:
    """
    Return (swing_highs, swing_lows) as chronological price lists of CONFIRMED
    pivots.

    Bar i is a pivot high if its High is the SOLE max of the window
    [i-prd, i+prd]; pivot low if its Low is the SOLE min. The sole-extreme
    guard avoids double-counting flat tops/bottoms (common in crypto).
    The last `prd` bars can never be confirmed — that's the structural lag.
    """
    highs = df["High"].to_numpy(dtype=float)
    lows  = df["Low"].to_numpy(dtype=float)
    n = len(highs)

    swing_highs: list[float] = []
    swing_lows:  list[float] = []

    for i in range(prd, n - prd):
        win_h = highs[i - prd: i + prd + 1]
        win_l = lows[i - prd: i + prd + 1]
        if highs[i] == win_h.max() and (win_h == highs[i]).sum() == 1:
            swing_highs.append(float(highs[i]))
        if lows[i] == win_l.min() and (win_l == lows[i]).sum() == 1:
            swing_lows.append(float(lows[i]))

    return swing_highs, swing_lows


def _trailing_run(seq: list[float], direction: str) -> int:
    """
    Count consecutive trailing steps in `direction` ('down' or 'up').
    [10,9,8,7] down -> 3 ;  [10,9,11] down -> 0.
    """
    count = 0
    for i in range(len(seq) - 1, 0, -1):
        if direction == "down" and seq[i] < seq[i - 1]:
            count += 1
        elif direction == "up" and seq[i] > seq[i - 1]:
            count += 1
        else:
            break
    return count


def classify(df: pd.DataFrame | None, prd: int) -> dict:
    """Classify one ticker's current swing-structure state."""
    base = {
        "state": "NO_DATA", "last_close": np.nan,
        "last_swing_high": np.nan, "last_swing_low": np.nan,
        "hi_run": 0, "lo_run": 0, "legs": 0,
        "break_level": np.nan, "n_highs": 0, "n_lows": 0, "note": "",
    }
    if df is None or df.empty:
        base["note"] = "no data"
        return base

    sh, sl = find_swings(df, prd)
    base["last_close"] = round(float(df["Close"].iloc[-1]), 4)
    base["n_highs"] = len(sh)
    base["n_lows"]  = len(sl)

    if len(sh) < MIN_SWINGS or len(sl) < MIN_SWINGS:
        base["state"] = "NO_RIZZY"
        base["note"] = "not enough confirmed swings"
        return base

    last_close = base["last_close"]
    last_sh, prev_sh = sh[-1], sh[-2]
    last_sl, prev_sl = sl[-1], sl[-2]
    base["last_swing_high"] = round(last_sh, 4)
    base["last_swing_low"]  = round(last_sl, 4)

    lower_high  = last_sh < prev_sh
    lower_low   = last_sl < prev_sl
    higher_high = last_sh > prev_sh
    higher_low  = last_sl > prev_sl

    # RIZZY_DOWN: lower high + lower low, price hasn't reclaimed last lower high
    if lower_high and lower_low and last_close < last_sh:
        base["state"] = "RIZZY_DOWN"
        base["hi_run"] = _trailing_run(sh, "down")
        base["lo_run"] = _trailing_run(sl, "down")
        base["legs"] = min(base["hi_run"], base["lo_run"])
        base["break_level"] = round(last_sh, 4)   # close ABOVE -> exit short
        base["note"] = "short setup; exit on close above last lower high"
        return base

    # RIZZY_UP: higher high + higher low, price hasn't broken last higher low
    if higher_high and higher_low and last_close > last_sl:
        base["state"] = "RIZZY_UP"
        base["hi_run"] = _trailing_run(sh, "up")
        base["lo_run"] = _trailing_run(sl, "up")
        base["legs"] = min(base["hi_run"], base["lo_run"])
        base["break_level"] = round(last_sl, 4)   # close BELOW -> exit long
        base["note"] = "long setup; exit on close below last higher low"
        return base

    base["state"] = "NO_RIZZY"
    base["note"] = "mixed / chop / structure broken"
    return base


# ======================================================================
# Wiring into myTrading (tickers + data fetchers)
# ======================================================================
def load_tickers() -> tuple[list[str], list[str]]:
    """Load STOCK_TICKERS and CRYPTO_TICKERS from myTrading/app.py."""
    import importlib.util
    app_path = MYTRADING_DIR / "app.py"
    spec = importlib.util.spec_from_file_location("myTrading_app", str(app_path))
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot load {app_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    stocks  = list(getattr(mod, "STOCK_TICKERS", []))
    cryptos = list(getattr(mod, "CRYPTO_TICKERS", []))
    return stocks, cryptos


def get_fetchers():
    """Import the existing data fetchers from the myTrading package."""
    if str(MYTRADING_DIR) not in sys.path:
        sys.path.insert(0, str(MYTRADING_DIR))
    from data.stocks import fetch_stock_1d_df
    from data.crypto import fetch_crypto_4h_df
    return fetch_stock_1d_df, fetch_crypto_4h_df


# ======================================================================
# Scan + report
# ======================================================================
COLS = ["ticker", "asset_class", "timeframe", "state", "legs",
        "hi_run", "lo_run", "last_close", "last_swing_high",
        "last_swing_low", "break_level", "n_highs", "n_lows", "note"]


def scan(stocks: list[str], cryptos: list[str],
         prd_stock: int, prd_crypto: int, limit: int | None = None) -> pd.DataFrame:
    fetch_stock, fetch_crypto = get_fetchers()

    if limit:
        stocks = stocks[:limit]
        cryptos = cryptos[:limit]

    log(f"Scanning {len(stocks)} stocks (1d, prd={prd_stock}) + "
        f"{len(cryptos)} crypto (4h, prd={prd_crypto})")

    rows = []
    for klass, tickers, fetch, prd, tf in (
        ("stock",  stocks,  fetch_stock,  prd_stock,  "1d"),
        ("crypto", cryptos, fetch_crypto, prd_crypto, "4h"),
    ):
        for t in tickers:
            try:
                r = classify(fetch(t), prd)
            except Exception as e:
                r = classify(None, prd)
                r["note"] = f"error: {e}"
            r.update({"ticker": t, "asset_class": klass, "timeframe": tf})
            rows.append(r)
            log(f"  {t:<14} {r['state']:<10} legs={r['legs']}")

    df_out = pd.DataFrame(rows)[COLS]
    state_order = {"RIZZY_DOWN": 0, "RIZZY_UP": 1, "NO_RIZZY": 2, "NO_DATA": 3}
    df_out["_o"] = df_out["state"].map(state_order).fillna(9)
    df_out = (df_out.sort_values(["_o", "legs"], ascending=[True, False])
                    .drop(columns="_o").reset_index(drop=True))
    return df_out


def print_summary(df_out: pd.DataFrame) -> None:
    counts = df_out["state"].value_counts().to_dict()
    log("")
    log("==================== RIZZY SUMMARY ====================")
    for s in ["RIZZY_UP", "RIZZY_DOWN", "NO_RIZZY", "NO_DATA"]:
        log(f"  {s:<11}: {counts.get(s, 0)}")
    log("------------------------------------------------------")
    for s, arrow in [("RIZZY_DOWN", "v SHORT"), ("RIZZY_UP", "^ LONG")]:
        sub = df_out[df_out["state"] == s]
        if sub.empty:
            continue
        log(f"\n{arrow}  ({s})  — deepest structure first:")
        for _, r in sub.iterrows():
            log(f"   {r['ticker']:<14} {r['asset_class']:<6} "
                f"legs={int(r['legs'])}  close={r['last_close']}  "
                f"break@{r['break_level']}")
    log("======================================================")


def main() -> None:
    ap = argparse.ArgumentParser(description="Rizzy swing-structure scanner")
    ap.add_argument("--prd", type=int, default=None,
                    help="override pivot strength for BOTH classes")
    ap.add_argument("--prd-stock", type=int, default=PIVOT_PRD_STOCK)
    ap.add_argument("--prd-crypto", type=int, default=PIVOT_PRD_CRYPTO)
    ap.add_argument("--stocks-only", action="store_true")
    ap.add_argument("--crypto-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N of each (quick test)")
    ap.add_argument("--out", type=str, default=str(OUT_CSV))
    args = ap.parse_args()

    prd_stock  = args.prd if args.prd is not None else args.prd_stock
    prd_crypto = args.prd if args.prd is not None else args.prd_crypto

    stocks, cryptos = load_tickers()
    if args.stocks_only:
        cryptos = []
    if args.crypto_only:
        stocks = []

    df_out = scan(stocks, cryptos, prd_stock, prd_crypto, limit=args.limit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print_summary(df_out)
    log(f"\nWrote {len(df_out)} rows -> {out_path}")


if __name__ == "__main__":
    main()