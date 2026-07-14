#!/usr/bin/env python3
"""
rizzy.py — Swing-structure state scanner with measured-move targets.

Classifies every ticker in STOCK_TICKERS and CRYPTO_TICKERS (loaded from
myTrading/app.py) into one of:

    RIZZY_UP    higher highs + higher lows, structure intact  (long setup)
    RIZZY_DOWN  lower highs  + lower lows,  structure intact  (short setup)
    NO_RIZZY    mixed / chopping / structure just broke
    NO_DATA     fetch failed or not enough bars

Swings are pivots that dominate `prd` bars on each side.

PROVISIONAL PIVOTS (asymmetric confirmation):
    A fully CONFIRMED pivot needs `prd` bars on the left AND right.
    The LATEST swing in each direction may instead be PROVISIONAL:
    `prd` bars on the left but only 1..prd-1 closed bars on the right.
    This lets the scanner flag a fresh lower high (e.g. a failed bounce)
    1-2 bars earlier, at the cost that the pivot can be erased if price
    later exceeds it. The `confirm` column tells you which mode the
    current state rests on:
        FULL    all anchors fully confirmed
        PROV_2  newest anchor has 2 right-side bars
        PROV_1  newest anchor has 1 right-side bar

MEASURED-MOVE TARGET (RIZZY_DOWN example, mirror for UP):
    1. Trendline through the last two swing highs SH1 -> SH2 (price/bar slope).
    2. Deepest low (wick) among bars between SH1 and SH2 inclusive.
    3. channel_height = trendline value AT that low's bar - the low.
    4. rizzy_target   = low - channel_height   (parallel channel projection).

Timeframes:   stocks -> 1d (fetch_stock_1d_df)   crypto -> 4h (fetch_crypto_4h_df)

Write-only: produces a CSV + console summary. No git push (the single pusher
owns git).

Usage:
    python rizzy.py                 # full scan, stocks + crypto
    python rizzy.py --crypto-only
    python rizzy.py --stocks-only
    python rizzy.py --limit 10      # first 10 of each (quick test)
    python rizzy.py --prd 5         # override pivot strength for both classes
    python rizzy.py --no-provisional  # strict 3/3 confirmation only
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
RIZZY_VERSION = "2.1-zigzag"

MYTRADING_DIR = Path.home() / "github" / "myTrading"
JOB_DIR       = Path.home() / "github" / "jobMyTrading"
OUT_CSV       = JOB_DIR / "rizzy_signals.csv"
LOG_FILE      = JOB_DIR / "rizzy.log"

# ----------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------
# Pivot strength: a swing high must dominate `prd` bars on each side.
# Higher prd = fewer, cleaner swings, more lag.
PIVOT_PRD_STOCK  = 3
PIVOT_PRD_CRYPTO = 3

# Allow the newest swing in each direction to be provisional
# (prd bars left, >=1 bar right). Set False for strict anti-repaint mode.
ALLOW_PROVISIONAL = True

# Need at least this many highs AND lows before calling a direction.
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
def find_swings(df: pd.DataFrame, prd: int,
                allow_provisional: bool = True
                ) -> tuple[list[dict], list[dict]]:
    """
    Alternating ZigZag swing detection.

    A swing HIGH is the highest High since the last swing low, confirmed
    once `prd` bars close without exceeding it. A swing LOW is the lowest
    Low since the last swing high, confirmed once `prd` bars close without
    undercutting it. Highs and lows strictly alternate — this matches how
    structure is marked on a chart, and unlike a fixed left/right window
    it cannot miss the top of a sharp V-bounce just because the prior
    decline passed through the same price a few bars earlier.

    If price exceeds the last swing high before a new low confirms, the
    swing high RELOCATES to the new extreme (standard zigzag behavior);
    mirror for lows. This is also what makes provisional pivots
    self-erasing.

    Returns (swing_highs, swing_lows): chronological lists of
    {"idx": bar, "price": float, "rbars": bars closed after the pivot,
    capped at prd}. rbars == prd -> fully confirmed; rbars < prd ->
    provisional (only ever the final pivot, only if allow_provisional).

    Ties: strict comparisons everywhere, so the FIRST bar of an
    equal-extreme flat top/bottom owns the pivot.
    """
    highs = df["High"].to_numpy(dtype=float)
    lows  = df["Low"].to_numpy(dtype=float)
    n = len(highs)
    sh_i: list[int] = []   # swing high bar indices
    sl_i: list[int] = []   # swing low bar indices
    if n < prd + 2:
        return [], []

    seeking: str | None = None   # None until first pivot; then 'low'/'high'
    max_i, min_i = 0, 0
    i = 1
    bos: set[int] = set()        # pivots confirmed by break-of-structure

    # ---- seed: first pivot = whichever extreme goes stale first ----
    while i < n and seeking is None:
        if highs[i] > highs[max_i]:
            max_i = i
        if lows[i] < lows[min_i]:
            min_i = i
        hi_stale = (i - max_i) >= prd
        lo_stale = (i - min_i) >= prd
        if hi_stale and (not lo_stale or max_i < min_i):
            sh_i.append(max_i)
            seeking = "low"
            seg = lows[max_i + 1: i + 1]
            min_i = max_i + 1 + int(np.argmin(seg))
        elif lo_stale:
            sl_i.append(min_i)
            seeking = "high"
            seg = highs[min_i + 1: i + 1]
            max_i = min_i + 1 + int(np.argmax(seg))
        i += 1

    # ---- main alternation ----
    # Within each bar: (1) extend the running extreme, (2) confirm the
    # pending pivot by bar count, (3) break-of-structure: while seeking a
    # high, a bar undercutting the last swing low CONFIRMS the running
    # high as the (lower) swing high — price broke the low before
    # exceeding the high, which validates the swing structurally even if
    # fewer than prd bars have closed (mirror for seeking a low). Only if
    # NO running extreme has formed since the last pivot does the
    # undercut/overshoot relocate that pivot (pure trend continuation).
    while i < n:
        if seeking == "low":
            if lows[i] < lows[min_i]:
                min_i = i
            if (i - min_i) >= prd and min_i > sh_i[-1]:
                sl_i.append(min_i)              # bar-count confirmation
                seeking = "high"
                seg = highs[min_i + 1: i + 1]
                max_i = min_i + 1 + int(np.argmax(seg))
            elif highs[i] > highs[sh_i[-1]]:
                if min_i > sh_i[-1] and i > min_i:
                    sl_i.append(min_i)          # BOS confirmation (up)
                    bos.add(min_i)
                    seeking = "high"
                    seg = highs[min_i + 1: i + 1]
                    max_i = min_i + 1 + int(np.argmax(seg))
                else:
                    sh_i[-1] = i                # relocate swing high
                    bos.discard(i)
                    min_i = i
        elif seeking == "high":
            if highs[i] > highs[max_i]:
                max_i = i
            if (i - max_i) >= prd and max_i > sl_i[-1]:
                sh_i.append(max_i)              # bar-count confirmation
                seeking = "low"
                seg = lows[max_i + 1: i + 1]
                min_i = max_i + 1 + int(np.argmin(seg))
            elif lows[i] < lows[sl_i[-1]]:
                if max_i > sl_i[-1] and i > max_i:
                    sh_i.append(max_i)          # BOS confirmation (down)
                    bos.add(max_i)
                    seeking = "low"
                    seg = lows[max_i + 1: i + 1]
                    min_i = max_i + 1 + int(np.argmin(seg))
                else:
                    sl_i[-1] = i                # relocate swing low
                    bos.discard(i)
                    max_i = i
        i += 1

    # ---- provisional tail: current running extreme with >=1 closed bar ----
    if allow_provisional and seeking is not None:
        last_bar = n - 1
        if seeking == "low":
            r = last_bar - min_i
            if 1 <= r < prd and min_i > sh_i[-1]:
                sl_i.append(min_i)
        else:
            r = last_bar - max_i
            if 1 <= r < prd and (not sl_i or max_i > sl_i[-1]):
                sh_i.append(max_i)

    def _mk(idxs: list[int], arr: np.ndarray) -> list[dict]:
        out = []
        for j in idxs:
            rb = min(prd, (n - 1) - j)
            if rb < 1:      # a pivot needs at least one closed bar after it
                continue
            out.append({"idx": j, "price": float(arr[j]), "rbars": rb,
                        "bos": j in bos})
        return out

    return _mk(sh_i, highs), _mk(sl_i, lows)


def _trailing_run(prices: list[float], direction: str) -> int:
    """Consecutive trailing steps in `direction` ('down'|'up')."""
    count = 0
    for i in range(len(prices) - 1, 0, -1):
        if direction == "down" and prices[i] < prices[i - 1]:
            count += 1
        elif direction == "up" and prices[i] > prices[i - 1]:
            count += 1
        else:
            break
    return count


def _measured_target(df: pd.DataFrame, p1: dict, p2: dict,
                     direction: str) -> tuple[float, float, float]:
    """
    Measured-move / parallel-channel projection.

    direction 'down': p1,p2 = last two swing highs (p1 older/higher).
        Trendline through (p1.idx,p1.price)-(p2.idx,p2.price).
        Deepest wick low in [p1.idx, p2.idx];
        height = trendline value at that bar - low;
        target = low - height.
    direction 'up': mirror with swing lows and the highest wick.

    Returns (channel_extreme, channel_height, target).
    """
    i1, y1 = p1["idx"], p1["price"]
    i2, y2 = p2["idx"], p2["price"]
    if i2 == i1:
        return (np.nan, np.nan, np.nan)
    slope = (y2 - y1) / (i2 - i1)

    if direction == "down":
        seg = df["Low"].to_numpy(dtype=float)[i1: i2 + 1]
        k = int(np.argmin(seg))
        extreme = float(seg[k])
        line_at = y1 + slope * k          # trendline value above that bar
        height = line_at - extreme
        target = extreme - height
    else:
        seg = df["High"].to_numpy(dtype=float)[i1: i2 + 1]
        k = int(np.argmax(seg))
        extreme = float(seg[k])
        line_at = y1 + slope * k          # trendline value below that bar
        height = extreme - line_at
        target = extreme + height

    return (round(extreme, 4), round(height, 4), round(target, 4))


def classify(df: pd.DataFrame | None, prd: int,
             allow_provisional: bool = True) -> dict:
    """Classify one ticker's current swing-structure state + target."""
    base = {
        "state": "NO_DATA", "confirm": "",
        "legs": 0, "hi_run": 0, "lo_run": 0,
        "last_close": np.nan,
        "last_swing_high": np.nan, "last_swing_low": np.nan,
        "break_level": np.nan,
        "trend_p1": np.nan, "trend_p2": np.nan,
        "channel_extreme": np.nan, "channel_height": np.nan,
        "rizzy_target": np.nan,
        "n_highs": 0, "n_lows": 0, "note": "",
    }
    if df is None or df.empty:
        base["note"] = "no data"
        return base

    sh, sl = find_swings(df, prd, allow_provisional)
    base["last_close"] = round(float(df["Close"].iloc[-1]), 4)
    base["n_highs"] = len(sh)
    base["n_lows"]  = len(sl)

    if len(sh) < MIN_SWINGS or len(sl) < MIN_SWINGS:
        base["state"] = "NO_RIZZY"
        base["note"] = "not enough swings"
        return base

    last_close = base["last_close"]
    h1, h2 = sh[-2], sh[-1]           # h2 = most recent swing high
    l1, l2 = sl[-2], sl[-1]
    base["last_swing_high"] = round(h2["price"], 4)
    base["last_swing_low"]  = round(l2["price"], 4)

    closes = df["Close"].to_numpy(dtype=float)
    # Structure-break checks must look at EVERY close since the pivot,
    # not just the latest one. If any close after the lower high exceeded
    # it, the down-structure broke at that moment — a later pullback
    # under the level does not un-break it (mirror for the up case).
    down_intact = not (closes[h2["idx"] + 1:] > h2["price"]).any()
    up_intact   = not (closes[l2["idx"] + 1:] < l2["price"]).any()

    def _confirm_tag(*pivots: dict) -> str:
        # a pivot confirmed by break-of-structure is as solid as a
        # bar-count-confirmed one; otherwise report the weakest anchor
        weakest = min(prd if p.get("bos") else p["rbars"] for p in pivots)
        return "FULL" if weakest >= prd else f"PROV_{weakest}"

    sh_prices = [p["price"] for p in sh]
    sl_prices = [p["price"] for p in sl]

    # ---- RIZZY_DOWN: LH + LL, close still under the last lower high ----
    if h2["price"] < h1["price"] and l2["price"] < l1["price"] \
            and down_intact:
        base["state"] = "RIZZY_DOWN"
        base["confirm"] = _confirm_tag(h1, h2, l1, l2)
        base["hi_run"] = _trailing_run(sh_prices, "down")
        base["lo_run"] = _trailing_run(sl_prices, "down")
        base["legs"] = min(base["hi_run"], base["lo_run"])
        base["break_level"] = round(h2["price"], 4)
        base["trend_p1"] = round(h1["price"], 4)
        base["trend_p2"] = round(h2["price"], 4)
        ext, hgt, tgt = _measured_target(df, h1, h2, "down")
        base["channel_extreme"], base["channel_height"], base["rizzy_target"] = ext, hgt, tgt
        base["note"] = "short setup; exit close above last LH"
        return base

    # ---- RIZZY_UP: HH + HL, close still above the last higher low ----
    if h2["price"] > h1["price"] and l2["price"] > l1["price"] \
            and up_intact:
        base["state"] = "RIZZY_UP"
        base["confirm"] = _confirm_tag(h1, h2, l1, l2)
        base["hi_run"] = _trailing_run(sh_prices, "up")
        base["lo_run"] = _trailing_run(sl_prices, "up")
        base["legs"] = min(base["hi_run"], base["lo_run"])
        base["break_level"] = round(l2["price"], 4)
        base["trend_p1"] = round(l1["price"], 4)
        base["trend_p2"] = round(l2["price"], 4)
        ext, hgt, tgt = _measured_target(df, l1, l2, "up")
        base["channel_extreme"], base["channel_height"], base["rizzy_target"] = ext, hgt, tgt
        base["note"] = "long setup; exit close below last HL"
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
COLS = ["ticker", "asset_class", "timeframe", "state", "confirm", "legs",
        "hi_run", "lo_run", "last_close", "last_swing_high",
        "last_swing_low", "break_level",
        "trend_p1", "trend_p2", "channel_extreme", "channel_height",
        "rizzy_target", "n_highs", "n_lows", "note"]


def debug_ticker(ticker: str, prd: int, allow_provisional: bool = True,
                 last_n: int = 30) -> None:
    """Dump recent bars + every detected pivot for one ticker, then the
    classification. Use to verify the scanner against a chart."""
    fetch_stock, fetch_crypto = get_fetchers()
    stocks, cryptos = load_tickers()
    if ticker in cryptos:
        df, tf = fetch_crypto(ticker), "4h"
    else:
        df, tf = fetch_stock(ticker), "1d"
    if df is None or df.empty:
        print(f"{ticker}: no data")
        return

    sh, sl = find_swings(df, prd, allow_provisional)
    sh_idx = {p["idx"]: p for p in sh}
    sl_idx = {p["idx"]: p for p in sl}

    def _tag(p: dict) -> str:
        return "FULL" if p["rbars"] >= prd else f"PROV_{p['rbars']}"

    print(f"rizzy {RIZZY_VERSION}")
    print(f"\n=== {ticker} ({tf}, prd={prd}, "
          f"provisional={'ON' if allow_provisional else 'OFF'}) — "
          f"last {last_n} bars ===")
    start = max(0, len(df) - last_n)
    for i in range(start, len(df)):
        row = df.iloc[i]
        marks = []
        if i in sh_idx:
            marks.append(f"<== SWING HIGH {sh_idx[i]['price']} ({_tag(sh_idx[i])})")
        if i in sl_idx:
            marks.append(f"<== SWING LOW {sl_idx[i]['price']} ({_tag(sl_idx[i])})")
        ts = df.index[i]
        print(f"  [{i:>4}] {ts}  H={row['High']:<10.4f} L={row['Low']:<10.4f} "
              f"C={row['Close']:<10.4f} {' '.join(marks)}")

    print(f"\n  all swing highs: {[(p['idx'], p['price'], p['rbars']) for p in sh][-6:]}")
    print(f"  all swing lows : {[(p['idx'], p['price'], p['rbars']) for p in sl][-6:]}")

    r = classify(df, prd, allow_provisional)
    print("\n  classification:")
    for k in COLS:
        if k in r:
            print(f"    {k:>16}: {r[k]}")


def scan(stocks: list[str], cryptos: list[str],
         prd_stock: int, prd_crypto: int,
         allow_provisional: bool = True,
         limit: int | None = None) -> pd.DataFrame:
    fetch_stock, fetch_crypto = get_fetchers()

    if limit:
        stocks = stocks[:limit]
        cryptos = cryptos[:limit]

    log(f"rizzy {RIZZY_VERSION}")
    log(f"Scanning {len(stocks)} stocks (1d, prd={prd_stock}) + "
        f"{len(cryptos)} crypto (4h, prd={prd_crypto}) "
        f"provisional={'ON' if allow_provisional else 'OFF'}")

    rows = []
    for klass, tickers, fetch, prd, tf in (
        ("stock",  stocks,  fetch_stock,  prd_stock,  "1d"),
        ("crypto", cryptos, fetch_crypto, prd_crypto, "4h"),
    ):
        for t in tickers:
            try:
                r = classify(fetch(t), prd, allow_provisional)
            except Exception as e:
                r = classify(None, prd)
                r["note"] = f"error: {e}"
            r.update({"ticker": t, "asset_class": klass, "timeframe": tf})
            rows.append(r)
            tgt = r["rizzy_target"]
            tgt_s = f" target={tgt}" if pd.notna(tgt) else ""
            log(f"  {t:<14} {r['state']:<10} {r['confirm']:<7} "
                f"legs={r['legs']}{tgt_s}")

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
                f"{r['confirm']:<7} legs={int(r['legs'])}  "
                f"close={r['last_close']}  break@{r['break_level']}  "
                f"target={r['rizzy_target']}")
    log("======================================================")


def main() -> None:
    ap = argparse.ArgumentParser(description="Rizzy swing-structure scanner")
    ap.add_argument("--prd", type=int, default=None,
                    help="override pivot strength for BOTH classes")
    ap.add_argument("--prd-stock", type=int, default=PIVOT_PRD_STOCK)
    ap.add_argument("--prd-crypto", type=int, default=PIVOT_PRD_CRYPTO)
    ap.add_argument("--stocks-only", action="store_true")
    ap.add_argument("--crypto-only", action="store_true")
    ap.add_argument("--no-provisional", action="store_true",
                    help="strict prd/prd confirmation only (anti-repaint)")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N of each (quick test)")
    ap.add_argument("--debug-ticker", type=str, default=None,
                    help="dump bars + pivots + classification for ONE ticker")
    ap.add_argument("--out", type=str, default=str(OUT_CSV))
    args = ap.parse_args()

    prd_stock  = args.prd if args.prd is not None else args.prd_stock
    prd_crypto = args.prd if args.prd is not None else args.prd_crypto
    allow_prov = ALLOW_PROVISIONAL and not args.no_provisional

    if args.debug_ticker:
        t = args.debug_ticker
        stocks_all, cryptos_all = load_tickers()
        prd = prd_crypto if t in cryptos_all else prd_stock
        debug_ticker(t, prd, allow_prov)
        return

    stocks, cryptos = load_tickers()
    if args.stocks_only:
        cryptos = []
    if args.crypto_only:
        stocks = []

    df_out = scan(stocks, cryptos, prd_stock, prd_crypto,
                  allow_provisional=allow_prov, limit=args.limit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print_summary(df_out)
    log(f"\nWrote {len(df_out)} rows -> {out_path}")


if __name__ == "__main__":
    main()