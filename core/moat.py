"""
core/moat.py
============
Our own moat rating, derived from filings. Not copied from anyone.

WHY THIS REPLACED THE OLD ONE. There were two moat paths in this repo and
both were really the same thing: a `MOAT_OVERRIDE` table holding four
ratings read off StockOracle's website, and an `infer_moat()` proxy whose
test was literally named "matches the three known ratings". One hardcoded
their answers; the other was fitted to them. Chakravarti, 2026-11-01:
*"i am developing this so we make our own independent ratings and not
hardcode from StockOracle"*. He supplied their screens so we could work out
WHICH MEASURES MATTER and compute them ourselves — not to copy the outputs.

WHAT A MOAT ACTUALLY IS. A durable competitive advantage shows up as
durable EXCESS RETURNS ON CAPITAL. Not as technology, market share, or a
product roadmap — those are capabilities, and capabilities only count once
they survive competitors trying to take them. So the evidence is:

  level        median return on equity across every year we can see
  durability   how OFTEN it clears a cost-of-equity hurdle, not just lately
  pricing      median gross margin -- can it charge more than it costs
  resilience   how STEADY that gross margin is. A margin that holds while
               rivals attack is pricing power; one that swings with the
               cycle is a commodity with good years.
  evidence     years of data. "Durable" is a claim about time, so a company
               with four years of filings cannot earn WIDE no matter how
               good those four look.

TTM FIGURES ARE NOT USED, DELIBERATELY. The old proxy fed on trailing
twelve months and rated TSLA NONE off a 3.8% trailing ROE, in a year its
margins were compressing. A cyclical trough is not the absence of a moat,
and a cyclical peak is not the presence of one. The whole question is what
survives the cycle, so the cycle is what we measure over.

CALIBRATED ON BUSINESS SENSE, NOT ON THEIR ANSWERS. The thresholds were set
by asking whether the output is defensible for companies whose competitive
position is not seriously disputed -- Microsoft and Apple wide, Ford none --
and then left alone. Where it disagrees with StockOracle, that is a result,
not a bug to tune away.
"""
from __future__ import annotations

import statistics as st

WIDE, NARROW, NONE = "WIDE", "NARROW", "NONE"

# Roughly cost of equity plus a real spread. Below this a company is not
# earning its keep, however profitable it looks in absolute dollars.
HURDLE = 0.12

MIN_YEARS_FOR_WIDE = 8
MIN_YEARS = 3

# HOW FAR BACK TO LOOK. Long enough to span a cycle, short enough that a
# company's startup decade does not define it. Over all 16 years of filings
# Tesla's median ROE is -14%, because 2009-2018 was a company losing money
# building factories -- a true fact about 2012 and a useless one about the
# business today. Over the last ten it is a different and fairer picture.
# A moat is a claim about what survives competition, not about who the
# company used to be.
WINDOW = 10


def rate(*, roe_by_year=None, gross_margin_by_year=None, log=None):
    """(rating, why). (None, reason) when there is not enough to judge.

    A moat rating with no evidence behind it is worse than none at all --
    it travels into the valuation and silently adds a premium.
    """
    roe = _vals(roe_by_year)
    gm = _vals(gross_margin_by_year)
    if len(roe) < MIN_YEARS:
        return None, f"only {len(roe)} years of returns — not enough to judge"

    med_roe = st.median(roe)
    clears = sum(1 for v in roe if v > HURDLE) / len(roe)
    n = len(roe)

    med_gm = st.median(gm) if len(gm) >= MIN_YEARS else None
    # Mean absolute deviation, not stdev: one restructuring year should not
    # read as a decade of volatility.
    gm_steady = (st.mean([abs(v - med_gm) for v in gm])
                 if med_gm is not None else None)

    bits = [f"ROE median {med_roe:.0%} over {n}y",
            f"clears {HURDLE:.0%} in {clears:.0%} of years"]
    if med_gm is not None:
        bits.append(f"gross margin {med_gm:.0%}"
                    + (f" ±{gm_steady:.0%}" if gm_steady is not None else ""))

    # ── WIDE ────────────────────────────────────────────────────────────
    # High returns, earned nearly every year, over a long record, on
    # margins wide enough and steady enough to show real pricing power.
    if (n >= MIN_YEARS_FOR_WIDE
            and med_roe >= 0.20
            and clears >= 0.80
            and med_gm is not None and med_gm >= 0.40
            and gm_steady is not None and gm_steady <= 0.06):
        return WIDE, " · ".join(bits)

    # ── NARROW ──────────────────────────────────────────────────────────
    # Genuinely profitable more often than not, but one leg is missing:
    # too short a record, returns that come and go, or margins that say
    # the pricing is competitive rather than protected.
    if med_roe >= HURDLE and clears >= 0.50:
        return NARROW, " · ".join(bits)
    # Strong, stable pricing power that has not yet converted into returns
    # is still an advantage -- a young company with 60% gross margins is
    # not Ford.
    if (med_gm is not None and med_gm >= 0.50
            and gm_steady is not None and gm_steady <= 0.06
            and med_roe > 0):
        return NARROW, " · ".join(bits)

    # ── NONE ────────────────────────────────────────────────────────────
    return NONE, " · ".join(bits)


def from_edgar(ticker, log=None):
    """Pull the series from SEC filings and rate. ({} -> (None, reason))."""
    try:
        from core import edgar
        ni = edgar.annual_series(ticker, edgar.NET_INCOME_TAGS)
        eq = edgar.annual_series(ticker, edgar.EQUITY_TAGS, instant=True)
        rev = edgar.annual_revenue(ticker)
        gp = edgar.annual_series(ticker, edgar.GROSS_PROFIT_TAGS)
    except Exception as e:                               # noqa: BLE001
        return None, f"no filings ({type(e).__name__})"

    roe = {y: ni[y] / eq[y] for y in sorted(set(ni) & set(eq))
           if eq.get(y) and eq[y] > 0}
    gm = {y: gp[y] / rev[y] for y in sorted(set(gp) & set(rev))
          if rev.get(y) and rev[y] > 0}
    return rate(roe_by_year=roe, gross_margin_by_year=gm, log=log)


def _vals(d):
    """The most recent WINDOW years, newest last.

    Takes the LAST years by key when given a {year: value} mapping, so the
    window means what it says. A plain sequence is assumed already ordered.
    """
    if not d:
        return []
    if hasattr(d, "items"):
        ordered = [v for _, v in sorted(d.items())]
    else:
        ordered = list(d)
    clean = [float(v) for v in ordered if v is not None and v == v]
    return clean[-WINDOW:]
