"""
core/oracle_iq.py
=================
OracleIQ — the six-factor quality panel that sits beside OracleValue.

StockOracle's OracleIQ™ tab prints six ranks for a ticker. On TSLA
(2026-11-01) it read:

    Predictability      Medium
    Profitability       High
    Growth              High
    OracleMoat          Narrow
    Financial Strength  High
    Valuation           Overvalued

Five of those six are computable from free data. This module computes them.

WHY THIS EXISTS. OracleValue answers "what is it worth". It does not answer
"how much should I trust that number". A $320 fair value on a company whose
earnings land within 5% of estimate every quarter is a different object from
the same $320 on a company that misses by a third. OracleIQ is that second
question, and it is the part a price target alone silently drops.

PREDICTABILITY IS THE FIND. yfinance's `get_earnings_dates()` returns the
estimate and the actual for the last ~24 quarters, and it matches the
StockOracle Earnings tab essentially line for line (TSLA 2026: est 0.45 /
act 0.48, est 0.54 / act 0.33, est 0.35 / act 0.41, est 0.45 / act 0.50).
So the beat/miss record is free and exact.

The measure is median |actual - estimate| divided by the stock's own median
|EPS|. Two wrong denominators were tried first and both are recorded here so
nobody retries them:

  - raw Surprise(%) — explodes on a near-zero EPS base. SITM scored 635%,
    which says nothing about SITM and everything about dividing by ~0.
  - surprise scaled by PRICE — inverts the ranking. It made MSFT look worse
    than TSLA, because scaling EPS error by price just measures 1/PE.

Dividing by the stock's own typical EPS is scale-free and ranks correctly:

    MSFT 0.05  AVGO 0.02  AMD 0.02  NVDA 0.07  CRWD 0.08   -> HIGH
    TSLA 0.11  COHR 0.10  PLTR 0.11  MU 0.13  NET 0.13     -> MEDIUM
    ALAB 0.21  AAOI 0.25  SITM 0.32                        -> LOW

TSLA at 0.11 lands MEDIUM, which is exactly what their OracleIQ prints.
That is the only published rank we can check, and it checks out.

WHAT PREDICTABILITY DELIBERATELY DOES NOT DO: move the value. The two
published anchors tempt you —

    TSLA (Medium)  OracleValue / MeanPS = 0.909
    MSFT (High)    OracleValue / MeanPS = 1.048

— into reading a ~13% quality tilt off the mean-P/S anchor. Applying it
makes the fit WORSE on both: MSFT goes from 0% error to +5%, TSLA from -17%
to -24%. The mean-P/S anchor already absorbs it, because a market that
trusts a company has been paying up for its sales all along. So this is a
label and a confidence signal, never a multiplier. Two data points are not
a coefficient.

WHAT IS NOT COMPUTABLE, and is left alone rather than faked:

  - VMI™ — locked behind their paywall, proprietary.
  - The community moat bars — literally human votes.
  - The AI Insights narrative — an LLM writing prose. Not deterministic,
    and the AAOI episode is what this whole module exists to avoid.
"""
from __future__ import annotations

import statistics as st

HIGH, MEDIUM, LOW = "High", "Medium", "Low"

# Thresholds set so TSLA (0.11) lands MEDIUM, matching the published rank.
PRED_HIGH = 0.08
PRED_LOW = 0.18

MIN_QUARTERS = 4          # fewer than this and we say nothing
LOOKBACK_Q = 12           # three years; longer drags in a different company


def predictability(earnings=None, *, log=None):
    """Rank how reliably reported EPS lands near the estimate.

    `earnings` is the DataFrame from yfinance's `get_earnings_dates()`, or
    any iterable of (actual, estimate) pairs. Returns (rank, detail) or
    (None, reason) — never a guess.
    """
    pairs = _pairs(earnings)
    if len(pairs) < MIN_QUARTERS:
        return None, f"only {len(pairs)} quarters of estimates"

    pairs = pairs[:LOOKBACK_Q]
    actuals = [a for a, _ in pairs]
    # The stock's own typical EPS magnitude. A company that earns $0.02 a
    # share is allowed a smaller absolute miss than one that earns $3.
    scale = st.median([abs(a) for a in actuals])
    if scale <= 0:
        return None, "median |EPS| is zero — no scale to measure against"

    err = st.median([abs(a - e) for a, e in pairs]) / scale
    miss = sum(1 for a, e in pairs if a < e) / len(pairs)
    neg = sum(1 for a in actuals if a < 0) / len(actuals)

    if err < PRED_HIGH:
        rank = HIGH
    elif err < PRED_LOW:
        rank = MEDIUM
    else:
        rank = LOW

    # Chronic losses are their own kind of unpredictable, whatever the
    # estimates did. AAOI was in the red in 11 of its last 12 quarters.
    if neg >= 0.5 and rank != LOW:
        rank = LOW if neg >= 0.75 else MEDIUM

    detail = (f"median miss {err:.0%} of typical EPS over {len(pairs)}q, "
              f"below estimate {miss:.0%} of the time")
    if neg:
        detail += f", loss-making {neg:.0%} of quarters"
    return rank, detail


def profitability(*, net_margin=None, normalized_margin=None, roe=None,
                  gross_margin=None):
    """Rank earnings quality, judged on the NORMALIZED margin where we have
    one. TSLA's trailing net margin is 3.11% against a 5-year average of
    10.50% — valuing the trailing figure is the cyclical trap from Part D of
    the method, and StockOracle ranks TSLA profitability High, not Low.
    """
    margin = normalized_margin if normalized_margin is not None else net_margin
    if margin is None and roe is None:
        return None, "no margin or ROE"

    score = 0
    bits = []
    if margin is not None:
        bits.append(f"margin {margin:.1%}")
        score += 2 if margin >= 0.15 else 1 if margin >= 0.07 else \
            0 if margin >= 0 else -2
    if roe is not None:
        bits.append(f"ROE {roe:.1%}")
        # 0.18, not 0.20: TSLA's normalized ROE is 19.61% and their OracleIQ
        # calls profitability High. A fifth of equity a year is high.
        score += 2 if roe >= 0.18 else 1 if roe >= 0.10 else \
            0 if roe >= 0 else -1
    if gross_margin is not None and gross_margin >= 0.50:
        bits.append(f"gross {gross_margin:.0%}")
        score += 1

    rank = HIGH if score >= 3 else MEDIUM if score >= 1 else LOW
    return rank, ", ".join(bits)


def growth(*, revenue_growth=None, earnings_growth=None, revenue_cagr=None):
    """Rank top-line expansion. Prefers a multi-year CAGR to a single
    year-over-year print, because one quarter against an easy comparison is
    not growth.
    """
    g = revenue_cagr if revenue_cagr is not None else revenue_growth
    if g is None and earnings_growth is None:
        return None, "no growth figures"

    bits = []
    score = 0
    if g is not None:
        bits.append(f"revenue {g:+.1%}")
        score += 2 if g >= 0.20 else 1 if g >= 0.08 else 0 if g >= 0 else -2
    if earnings_growth is not None:
        bits.append(f"earnings {earnings_growth:+.1%}")
        score += 1 if earnings_growth >= 0.15 else \
            0 if earnings_growth >= 0 else -1

    rank = HIGH if score >= 2 else MEDIUM if score >= 1 else LOW
    return rank, ", ".join(bits)


def financial_strength(*, total_debt=None, total_cash=None, free_cf=None,
                       operating_cf=None, market_cap=None, ebitda=None,
                       current_ratio=None):
    """Rank the balance sheet. Net cash and positive free cash flow carry
    this; the question it answers is whether the company can fund itself
    without asking the market for money on the market's terms.
    """
    if total_debt is None and total_cash is None and free_cf is None:
        return None, "no balance-sheet figures"

    score = 0
    bits = []
    if total_debt is not None and total_cash is not None:
        net = total_cash - total_debt
        bits.append(("net cash " if net >= 0 else "net debt ")
                    + _money(abs(net)))
        if net >= 0:
            score += 2
        elif ebitda and ebitda > 0:
            lev = total_debt / ebitda
            bits.append(f"{lev:.1f}x EBITDA")
            score += 1 if lev < 2.0 else 0 if lev < 3.5 else -2
        elif market_cap:
            score += 0 if net > -0.25 * market_cap else -2

    if free_cf is not None:
        bits.append("FCF " + _money(free_cf))
        score += 1 if free_cf > 0 else -2
    elif operating_cf is not None:
        score += 1 if operating_cf > 0 else -2

    if current_ratio is not None:
        bits.append(f"current {current_ratio:.2f}")
        score += 1 if current_ratio >= 1.5 else 0 if current_ratio >= 1.0 \
            else -1

    rank = HIGH if score >= 3 else MEDIUM if score >= 0 else LOW
    return rank, ", ".join(bits)


def panel(*, earnings=None, moat=None, band=None, **kw):
    """All six factors as an ordered dict, in StockOracle's own order.

    Unavailable factors are present with a None rank and the reason, so a
    blank is always distinguishable from a zero. Fail loud, not empty.
    """
    out = {}
    out["Predictability"] = predictability(earnings)
    out["Profitability"] = profitability(
        net_margin=kw.get("net_margin"),
        normalized_margin=kw.get("normalized_margin"),
        roe=kw.get("roe"), gross_margin=kw.get("gross_margin"))
    out["Growth"] = growth(
        revenue_growth=kw.get("revenue_growth"),
        earnings_growth=kw.get("earnings_growth"),
        revenue_cagr=kw.get("revenue_cagr"))
    out["OracleMoat"] = ((moat.title(), "rated") if moat
                         else (None, "not rated — inferred only"))
    out["FinancialStrength"] = financial_strength(
        total_debt=kw.get("total_debt"), total_cash=kw.get("total_cash"),
        free_cf=kw.get("free_cf"), operating_cf=kw.get("operating_cf"),
        market_cap=kw.get("market_cap"), ebitda=kw.get("ebitda"),
        current_ratio=kw.get("current_ratio"))
    out["Valuation"] = ((band.title(), "vs OracleValue") if band
                        else (None, "no OracleValue"))
    return out


def summary(p) -> str:
    """One line for the sheet: 'Pred Med · Profit High · Growth High ...'"""
    short = {"Predictability": "Pred", "Profitability": "Profit",
             "Growth": "Growth", "OracleMoat": "Moat",
             "FinancialStrength": "Fin", "Valuation": "Val"}
    bits = [f"{short[k]} {v[0]}" for k, v in p.items() if v[0]]
    return " · ".join(bits)


def _pairs(earnings):
    """(actual, estimate) newest first, from a yfinance frame or an iterable.
    Rows without both numbers are dropped — a future quarter has an estimate
    and no actual, and counting that as a miss would be nonsense.
    """
    if earnings is None:
        return []
    try:                                     # a DataFrame from yfinance
        cols = set(getattr(earnings, "columns", []))
        if {"Reported EPS", "EPS Estimate"} <= cols:
            df = earnings.dropna(subset=["Reported EPS", "EPS Estimate"])
            return [(float(a), float(e)) for a, e
                    in zip(df["Reported EPS"], df["EPS Estimate"])]
    except Exception:                                        # noqa: BLE001
        pass
    out = []
    for row in earnings:
        try:
            a, e = float(row[0]), float(row[1])
        except (TypeError, ValueError, IndexError):
            continue
        # The NEXT quarter arrives with an estimate and a NaN actual.
        # NaN fails every comparison silently, so drop it explicitly.
        if a != a or e != e:
            continue
        out.append((a, e))
    return out


def _money(x) -> str:
    x = float(x)
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(x) >= div:
            return f"${x / div:,.1f}{unit}"
    return f"${x:,.0f}"
