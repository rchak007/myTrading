#!/usr/bin/env python3
"""
core/benter.py
==============
The Benter expected-value score, 0-10, for one stock.

    10  clean buy        5  neutral        0  clear don't buy

PURE. Numbers in, a number out. No network, no Sheets, no files — so it is
testable on Pi 2, same contract as core/recommend.py and core/structure.py.

WHAT IT IS
    Documentation/BENTER-METHOD.md §7. The gap between a fair-value anchor and
    the price, DISCOUNTED for how much that anchor can be trusted. The
    discount is the method: a large gap on a contested, unproven business is
    not a large edge.

WHY THE CODE IS BETTER THAN SCORING BY HAND
    Scoring by hand, "uncertainty" was a label I assigned — high, very high —
    and I gave NVDA, AVGO, AMD and MRVL the same one. They are not equally
    uncertain. Here it is MEASURED from three things the data already knows:

        target spread   (high - low) / mean      how much analysts disagree
        ATR %                                    how violently it actually moves
        analyst count                            how thin the coverage is

    Measured beats asserted, and it removes the step most likely to be
    rationalised after the fact.

AND THE BANDS ARE CONTINUOUS
    The first hand-built rubric used fixed bands and collapsed real
    differences: GOOGL on a 23% gap and MSFT on 10% both scored 6; SPCX on
    52% and AMD on 12% both scored 5. A scale that maps a 12% gap and a 52%
    gap to the same number has destroyed the thing it was built to measure.
    The base is interpolated instead, so 23% and 10% cannot collide.

WHAT IT CANNOT KNOW
    Whether you already own the theme. Benter never bet the same horse twice
    because the odds looked good in two races, and the three highest scores
    here are usually the same AI-semi trade. `concentration_note()` says so;
    acting on it is yours.

    And the headline caveat from the doc: CONSENSUS TARGETS ARE NOT AN EDGE.
    Everyone sees them. A big gap often just means targets have not caught up
    with price in either direction. This is a SCREEN.
"""
from __future__ import annotations

# (gap, base score). Interpolated between, flat outside.
#
#  0% gap is a 3, not a 5: fairly priced is not a reason to act, and the
#  method is about finding overlays, not holding opinions.
#
#  THE CURVE PEAKS AND THEN FALLS, which is the whole point. A first version
#  rose monotonically and scored RCAT 7.1 on a 237% gap, ONDS 8.2 on 197%,
#  STKE on 349% — $1-to-$6 names where consensus says the stock is worth
#  three times the price.
#
#  That is not a three-times edge. A gap that large means the anchor is not
#  credible: either the analysts are modelling a lottery ticket, or the market
#  is pricing a risk of zero that the target ignores. Benter's requirement was
#  a BETTER probability estimate than the crowd's, and a 237% disagreement
#  with the crowd is overwhelmingly evidence that the estimate is wrong rather
#  than that the crowd is.
#
#  So an enormous gap is treated as a RED FLAG. The sweet spot is 40-70%:
#  large enough to matter, small enough to be believable.
GAP_CURVE = [(-0.10, 0.0), (-0.02, 2.0), (0.00, 3.0), (0.10, 4.5),
             (0.20, 6.0), (0.30, 7.2), (0.40, 8.4), (0.55, 9.5), (0.70, 9.5),
             (1.00, 7.0), (1.50, 5.0), (2.50, 3.0), (4.00, 2.0)]

# CALIBRATED SO "ORDINARY" COSTS NOTHING. The first cut multiplied by 0.77
# for a 0.84 target spread and 0.83 for a 3.2% ATR — both of which are simply
# normal. Compounded, AVGO's 53% gap came out at 5.8/NEUTRAL and NOTHING in a
# nine-name sample could score above 6, which makes the top of the scale
# decoration.
#
# Measured across AVGO NVDA MRVL GOOGL MSFT AMD SPCX TSLA AAPL on 2026-10-16:
# target spread runs 0.41-1.50 of mean (median ~0.8), ATR 1.9-6.8%. So the
# discount is pinned at 1.0 through the middle of those and only bites in the
# tail, where disagreement or volatility is genuinely abnormal.
SPREAD_FREE = 0.80      # spread at or under this costs nothing
SPREAD_SLOPE = 0.35
SPREAD_FLOOR = 0.50     # the most disagreement can cost
VOL_FREE = 3.2          # ATR % at or under this costs nothing
VOL_SLOPE = 1 / 18.0
VOL_FLOOR = 0.60        # the most volatility can cost
THIN_COVERAGE = 5       # fewer analysts than this is thin
THIN_PENALTY = 0.85

BANDS = [(9.0, "CLEAN BUY", "Large gap, high confidence. Full size."),
         (7.0, "BUY", "Real gap, uncertainty understood. Normal size."),
         (6.0, "STARTER", "Gap is thin or confidence limited. Part size."),
         (4.5, "NEUTRAL", "Fairly priced. Hold what you own; no new money."),
         (2.5, "DON'T BUY", "Small gap for the risk. Not necessarily a sell."),
         (0.0, "AVOID", "At or above fair value. Trim if held and sized up.")]


def _f(v):
    try:
        if v is None:
            return None
        x = float(v)
        return None if x != x else x
    except (TypeError, ValueError):
        return None


def _interp(x: float, curve) -> float:
    if x <= curve[0][0]:
        return curve[0][1]
    if x >= curve[-1][0]:
        return curve[-1][1]
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return curve[-1][1]


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Benter:
    """A score and the reasons for it. `score` is None when it cannot be told."""

    __slots__ = ("score", "band", "action", "gap", "base", "confidence",
                 "spread", "atr_pct", "analysts", "why")

    def __init__(self, score=None, band="", action="", gap=None, base=None,
                 confidence=None, spread=None, atr_pct=None, analysts=None,
                 why=""):
        self.score, self.band, self.action = score, band, action
        self.gap, self.base, self.confidence = gap, base, confidence
        self.spread, self.atr_pct, self.analysts = spread, atr_pct, analysts
        self.why = why

    def __repr__(self):
        if self.score is None:
            return f"Benter(unscored: {self.why})"
        return (f"Benter({self.score:.1f} {self.band}, gap "
                f"{100 * (self.gap or 0):+.0f}%, conf "
                f"{(self.confidence or 0):.2f})")


def score(price, target_mean, *, target_high=None, target_low=None,
          analysts=None, atr_pct=None) -> Benter:
    """0-10 for one stock.

    price / target_mean are required; everything else only sharpens the
    confidence discount. Missing extras make the score MORE generous, never
    less — so a thinly-covered name is not flattered by its own lack of data.
    `analysts` below THIN_COVERAGE is penalised precisely for that reason.
    """
    p, t = _f(price), _f(target_mean)
    if not p or p <= 0:
        return Benter(why="no price")
    if not t or t <= 0:
        return Benter(why="no analyst target to compare against")

    gap = t / p - 1.0
    base = _interp(gap, GAP_CURVE)

    conf, bits = 1.0, []

    hi, lo = _f(target_high), _f(target_low)
    spread = None
    if hi and lo and hi > lo and t > 0:
        spread = (hi - lo) / t
        # A range as wide as the mean itself means the street has no shared
        # view; the average is then an artifact of where the outliers sit.
        f = _clamp(1.0 + (SPREAD_FREE - spread) * SPREAD_SLOPE,
                   SPREAD_FLOOR, 1.0)
        conf *= f
        if f < 0.97:
            bits.append(f"spread {spread:.2f}x mean ×{f:.2f}")

    a = _f(atr_pct)
    if a and a > 0:
        # Daily ATR as a percentage of price. 2% is calm, 8% is a name that
        # can erase the whole gap in a fortnight.
        f = _clamp(1.0 + (VOL_FREE - a) * VOL_SLOPE, VOL_FLOOR, 1.0)
        conf *= f
        if f < 0.97:
            bits.append(f"ATR {a:.1f}% ×{f:.2f}")

    n = _f(analysts)
    if n is not None and n < THIN_COVERAGE:
        conf *= THIN_PENALTY
        bits.append(f"only {int(n)} analyst(s) ×{THIN_PENALTY}")

    s = _clamp(base * conf, 0.0, 10.0)
    band, action = next((b, act) for cut, b, act in BANDS if s >= cut)
    return Benter(score=round(s, 1), band=band, action=action, gap=gap,
                  base=round(base, 1), confidence=round(conf, 2),
                  spread=spread, atr_pct=a, analysts=n,
                  why=(f"gap {100 * gap:+.0f}% → base {base:.1f}"
                       + (", " + ", ".join(bits) if bits else
                          ", no confidence discount")))


def concentration_note(rows, top=3, min_score=7.0) -> str:
    """Warn when the best scores are all the same trade.

    rows: [{Ticker, Sector or Theme, score}]. The method is about finding
    INDEPENDENT overlays; three correlated names showing one gap is one
    sector-wide question, not three edges. Benter never bet the same horse
    twice because the odds looked good in two races.
    """
    good = sorted((r for r in rows
                   if _f(r.get("score")) is not None
                   and _f(r.get("score")) >= min_score),
                  key=lambda r: -_f(r["score"]))[:top]
    if len(good) < 2:
        return ""
    themes = {str(r.get("Sector") or r.get("Theme") or "").strip().upper()
              for r in good}
    themes.discard("")
    if len(themes) == 1:
        only = next(iter(themes))
        names = ", ".join(str(r["Ticker"]) for r in good)
        return (f"⚠️  the top {len(good)} scores are ALL {only}: {names}. "
                f"That is one bet, not {len(good)} — ask whether the sector "
                f"has re-rated rather than treating these as independent.")
    return ""
