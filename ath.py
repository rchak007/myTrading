#!/usr/bin/env python3
"""
ath.py
======
Shared helper: all-time-high (ATH) distance columns.

For each ticker, fetches FULL price history (yfinance period="max") and computes
how far the latest close sits below its all-time high, as a percentage:

    ATH_Dist_Pct = (last_close / all_time_high - 1) * 100

    e.g.  -45.0  → 45% below its ATH
           0.0   → sitting at a fresh ATH

Used by jobStocksSignals.py and 45_Signal.py (equities priced via yfinance).

NOT wired into the crypto job: crypto prices come from GeckoTerminal / CoinGecko
(and only ~70 days of history for on-chain pools), so a uniform true-ATH is not
available from that pipeline. See notes in jobCryptoSignals.py.

Public API mirrors hhll.py:
    compute_ath_table(tickers)                 -> DataFrame[Ticker, ATH, ATH_Dist_Pct]
    attach_ath_columns(df, tickers, after=...) -> (df_with_cols, message)
"""
from __future__ import annotations

import pandas as pd
import numpy as np

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _high_close_for(raw, ticker: str):
    """
    Pull (high_series, close_series) for one ticker out of a yfinance frame that
    may be single-index (one ticker) or MultiIndex columns (batched download).
    Returns (None, None) if not resolvable.
    """
    if raw is None or len(raw) == 0:
        return None, None
    cols = raw.columns
    if isinstance(cols, pd.MultiIndex):
        # Batched columns look like ('AAPL','High') or ('High','AAPL') depending
        # on group_by; handle either level carrying the ticker.
        lvl0 = set(cols.get_level_values(0))
        lvl1 = set(cols.get_level_values(1))
        if ticker in lvl0:
            sub = raw.xs(ticker, axis=1, level=0)
        elif ticker in lvl1:
            sub = raw.xs(ticker, axis=1, level=1)
        else:
            return None, None
    else:
        sub = raw
    high = sub["High"] if "High" in sub.columns else (sub["Close"] if "Close" in sub.columns else None)
    close = sub["Close"] if "Close" in sub.columns else None
    return high, close


def compute_ath_table(tickers, log_fn=print, chunk_size: int = 40) -> pd.DataFrame:
    """Return DataFrame with columns: Ticker, ATH, ATH_Dist_Pct."""
    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    rows: list[dict] = []

    if yf is None:
        log_fn("⚠️  ATH: yfinance not available — skipping")
        return pd.DataFrame(columns=["Ticker", "ATH", "ATH_Dist_Pct"])

    for chunk in _chunks(tickers, chunk_size):
        try:
            raw = yf.download(
                chunk, period="max", interval="1d",
                progress=False, auto_adjust=False,
                group_by="ticker", threads=True,
            )
        except Exception as e:
            log_fn(f"⚠️  ATH: batch download failed ({len(chunk)} tickers): {e}")
            raw = None

        for t in chunk:
            try:
                high, close = _high_close_for(raw, t) if raw is not None else (None, None)
                if high is None or close is None:
                    # Per-ticker fallback if the batch didn't carry this symbol.
                    single = yf.download(
                        t, period="max", interval="1d",
                        progress=False, auto_adjust=False,
                    )
                    high, close = _high_close_for(single, t)

                if high is None or close is None:
                    rows.append({"Ticker": t, "ATH": np.nan, "ATH_Dist_Pct": np.nan})
                    continue

                high_s = pd.to_numeric(pd.Series(high).squeeze(), errors="coerce").dropna()
                close_s = pd.to_numeric(pd.Series(close).squeeze(), errors="coerce").dropna()
                if high_s.empty or close_s.empty:
                    rows.append({"Ticker": t, "ATH": np.nan, "ATH_Dist_Pct": np.nan})
                    continue

                ath = float(high_s.max())
                last = float(close_s.iloc[-1])
                dist = (last / ath - 1.0) * 100.0 if ath > 0 else np.nan
                rows.append({
                    "Ticker": t,
                    "ATH": round(ath, 4),
                    "ATH_Dist_Pct": round(dist, 2) if pd.notna(dist) else np.nan,
                })
            except Exception as e:
                log_fn(f"⚠️  ATH: {t} failed: {e}")
                rows.append({"Ticker": t, "ATH": np.nan, "ATH_Dist_Pct": np.nan})

    return pd.DataFrame(rows)


def attach_ath_columns(df: pd.DataFrame, tickers, after="Current Price",
                       log_fn=print, include_ath_value: bool = True):
    """
    Merge ATH_Dist_Pct (and, by default, the absolute ATH) into df on Ticker,
    placing them immediately after the `after` column.

    `after` may be a single column name or a list of candidates — the first one
    present in df is used (handy across jobs: "Current Price" for stocks,
    "Price" for the 45° scanner). Falls back to appending at the end.

    Returns (df, message).
    """
    if "Ticker" not in df.columns:
        return df, "⚠️  ATH: no 'Ticker' column — skipped"

    ath_df = compute_ath_table(tickers, log_fn=log_fn)
    if ath_df.empty:
        return df, "⚠️  ATH: no data computed"

    if not include_ath_value:
        ath_df = ath_df.drop(columns=["ATH"], errors="ignore")

    out = df.merge(ath_df, on="Ticker", how="left")

    new_cols = [c for c in ["ATH_Dist_Pct", "ATH"] if c in out.columns]
    base_cols = [c for c in out.columns if c not in new_cols]

    # Resolve anchor column
    candidates = [after] if isinstance(after, str) else list(after)
    anchor = next((c for c in candidates if c in base_cols), None)

    if anchor is not None:
        idx = base_cols.index(anchor) + 1
        for c in reversed(new_cols):        # results in ATH_Dist_Pct, then ATH
            base_cols.insert(idx, c)
        ordered = base_cols
    else:
        ordered = base_cols + new_cols

    out = out[ordered]
    matched = int(out["ATH_Dist_Pct"].notna().sum())
    where = anchor if anchor else "end"
    return out, f"ATH: {matched}/{len(df)} tickers matched (placed after '{where}')"