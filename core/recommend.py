#!/usr/bin/env python3
"""
core/recommend.py
=================
Four recommended PRICE LEVELS per ticker, derived from the indicators
jobStocksSignals already computes:

    Rec_Stop        a SELL below  — where the thesis is broken
    Rec_Trim        a SELL above  — where strength is worth rotating out of
    Rec_Dip         a BUY  below  — where weakness is worth adding into
    Rec_Breakout    a BUY  above  — where strength is worth adding into

They mirror the four Dashboard coverage flags (Has_Stop / Has_Trim / Has_Dip /
Has_Breakout), which is the point: the flag says whether an order EXISTS, the
recommendation says where one WOULD go.

PURE. Prices in, prices out. No Schwab, no Sheets, no files, no clock — so the
whole thing is testable on Pi 2, which has none of those.

─────────────────────────────────────────────────────────────────────────────
EVERYTHING IS MEASURED IN ATR, NEVER IN PERCENT
    A 5% stop on NVDA (ATR ≈ 2% of price) and a 5% stop on FCEL (ATR ≈ 6%) are
    completely different trades. Percent hides that; ATR is the unit the market
    actually moves in.

THE THREE TRAPS, measured across the book on 2026-10-01
    1. ATH is CORRUPT on 15 of 128 names — reverse splits are not adjusted, so
       FCEL reads 234,900 and RCAT 990,000. Rejected above 5x price.
    2. Supertrend is in SELL mode on 64 of 128 — the line then sits ABOVE
       price and is resistance. Using it as a stop would put the stop above
       the market on half the book.
    3. MRC_S2 goes NEGATIVE (FCEL -1.03, MSTX -0.42) and MRC_R1 sits a median
       +19.9% away. Fine as a stretch marker, useless as an order level
       without zone gating.

FENCED CHANGES THE MODE
    A ticker cannot sensibly carry both a stop and a dip bid — they are
    opposite intents. Measured: the dip level landed BELOW the stop on 36 of
    128 names, because Supertrend often sits above structural support.

    Chakravarti's own IA house rules settle it: "Always Hedge", not always
    stop. So a FENCED (core) holding keeps its dip bid and treats the stop as
    an advisory thesis-break line; an unfenced (trade) position does the
    reverse. See `fenced=` on recommend().

FAILS CLOSED, ALWAYS
    Every level is None rather than 0.0 when it cannot be computed, and every
    output is re-checked against the side of price it must be on before being
    returned. This system's recurring failure mode is a wrong answer that
    looks like a normal result — a 0.00 in a price cell eventually gets read
    as a price.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

# ── tunables ────────────────────────────────────────────────────────────────
# In ATR multiples. These are the only free parameters in the module.
STOP_MIN_ATR = 1.5      # closer than this is inside daily noise
STOP_MAX_ATR = 4.0      # further than this is not a stop, it is a hope
STOP_FALLBACK_ATR = 2.5  # pure-volatility stop when no structure survives
TRIM_MIN_ATR = 1.0
TRIM_MAX_ATR = 8.0
DIP_MIN_ATR = 1.0       # closer than this is not a dip
BRK_MIN_ATR = 0.5
PIVOT_CUSHION_ATR = 0.25  # how far off a pivot to sit, either side

ATH_MAX_MULTIPLE = 5.0  # above this, the ATH is a reverse-split artifact
BREAKOUT_MIN_SCORE = 50  # the system's own "consider" bar (see Regime_Action)

FIELDS = ("Rec_Stop", "Rec_Trim", "Rec_Dip", "Rec_Breakout")


@dataclass
class Rec:
    """Four levels and the reason for each. Basis is not decoration.

    Without a stated basis the numbers are unfalsifiable — you cannot tell a
    structural stop from a volatility fallback, and the two deserve different
    confidence. A blank level carries the reason it is blank.
    """
    Rec_Stop: float | None = None
    Rec_Trim: float | None = None
    Rec_Dip: float | None = None
    Rec_Breakout: float | None = None
    Stop_Basis: str = ""
    Trim_Basis: str = ""
    Dip_Basis: str = ""
    Breakout_Basis: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    def basis_summary(self) -> str:
        """One cell: the four bases in fixed order, blanks as a dash."""
        return "/".join(b or "-" for b in (self.Stop_Basis, self.Trim_Basis,
                                           self.Dip_Basis, self.Breakout_Basis))


def _num(v) -> float | None:
    """A usable positive price, or None.

    Rejects NaN by the self-inequality trick rather than importing pandas —
    this module stays dependency-free so it can be tested anywhere. Zero and
    negatives are rejected too: MRC_S2 genuinely goes negative, and a level at
    or below zero is not a price.
    """
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f <= 0:          # NaN, zero, negative
        return None
    return f


def tick_round(px: float | None) -> float | None:
    """Round to a price an exchange will actually accept.

    US equities quote in $0.01 above $1.00 and $0.0001 below it (SEC Rule
    612); a sub-penny limit on a $40 stock is rejected outright. These levels
    get read straight into the Orders tab, so a number that cannot be entered
    as a limit price is not a recommendation.
    """
    if px is None:
        return None
    return round(px, 2) if px >= 1.0 else round(px, 4)


def _zone(raw) -> str:
    """Strip the emoji the signals table decorates zones with.

    MRC_Zone arrives as '🔵 Near_Mean' or plain 'Above_Mean' depending on the
    zone, so splitting on whitespace and taking the last token is the only
    form that handles both.
    """
    return str(raw or "").strip().split()[-1] if str(raw or "").strip() else ""


def recommend(price, atr, *,
              supertrend=None, supertrend_signal="",
              nearest_support=None, nearest_resistance=None,
              mrc_zone="", mrc_r1=None, mrc_mean=None, mrc_s1=None,
              ath=None, score=None, structure="", regime="",
              fenced=False) -> Rec:
    """The four levels for one ticker.

    `price` and `atr` are required; everything else degrades. A recent IPO with
    no 200-bar MRC and no confirmed pivots still gets a volatility stop, which
    is the whole reason the fallbacks exist — CBRS and PBLS would otherwise
    come back entirely blank.
    """
    r = Rec()
    p = _num(price)
    a = _num(atr)
    if p is None or a is None:
        return r
    cushion = a * PIVOT_CUSHION_ATR

    st = _num(supertrend)
    sup = _num(nearest_support)
    res = _num(nearest_resistance)
    r1, mean, s1 = _num(mrc_r1), _num(mrc_mean), _num(mrc_s1)
    zone = _zone(mrc_zone)
    st_bull = str(supertrend_signal).upper().startswith("BUY")

    # ── STOP ─────────────────────────────────────────────────────────────
    # Among valid floors below price, take the HIGHEST. The nearest floor is
    # the one whose break actually means something; choosing a lower one just
    # donates the difference between them.
    floors = []
    if st_bull and st is not None and st < p:
        floors.append((st, "ST"))
    if sup is not None and sup < p:
        # UNDER the pivot, not at it — resting exactly on an obvious swing low
        # is where stop runs are aimed.
        floors.append((sup - cushion, "pivot"))
    if floors:
        lvl, basis = max(floors)
        clamped = min(lvl, p - a * STOP_MIN_ATR)
        clamped = max(clamped, p - a * STOP_MAX_ATR)
        # Say so when the clamp moved it. A level reported as `pivot` that is
        # really 4 ATR of pure volatility is the kind of plausible-looking
        # wrong answer this system keeps getting bitten by.
        if abs(clamped - lvl) > 1e-9:
            basis += "~cap"
        lvl = clamped
    else:
        lvl, basis = p - a * STOP_FALLBACK_ATR, "vol"
    if lvl > 0 and lvl < p:
        r.Rec_Stop, r.Stop_Basis = lvl, basis
    else:
        r.Stop_Basis = "-"

    # ── TRIM ─────────────────────────────────────────────────────────────
    # The ZONE is the stretch measurement, so let it choose the level. The
    # below-mean branch matters most: it covers 39 of 128 names, and for those
    # R1 is not a plan. FCEL at 16.81 has R1 at 29.26 (+74%) — trimming there
    # is a wish. Reverting to the mean is the realistic first exit.
    if zone == "Strong_OB":
        lvl, basis = p + a * 0.5, "at-mkt"       # maximally stretched: go now
    elif zone == "OB":
        lvl, basis = (res, "pivot") if (res and res > p) else (p + a, "vol")
    elif zone in ("Above_Mean", "Near_Mean"):
        lvl, basis = (r1, "R1") if r1 else (None, "-")
    elif zone in ("Below_Mean", "OS", "Strong_OS"):
        lvl, basis = (mean, "mean") if mean else (None, "-")
    else:                                         # N/A — MRC still warming up
        lvl, basis = (res, "pivot") if (res and res > p) else (None, "-")
    if lvl is not None:
        want = lvl
        if zone != "Strong_OB":
            lvl = max(lvl, p + a * TRIM_MIN_ATR)
        lvl = min(lvl, p + a * TRIM_MAX_ATR)
        if abs(lvl - want) > 1e-9:
            basis += "~cap"
        if lvl > p:
            r.Rec_Trim, r.Trim_Basis = lvl, basis
        else:
            r.Trim_Basis = "-"
    else:
        r.Trim_Basis = basis

    # ── DIP ──────────────────────────────────────────────────────────────
    # Just ABOVE the pivot, the mirror of the stop sitting just below it: one
    # level, two sides, defined risk. You want the fill before the crowd's
    # stops trigger, not after.
    if str(structure).upper() == "BEARISH" and str(regime).upper() == "BEAR":
        r.Dip_Basis = "broken"                    # no bid into a broken name
    else:
        bases = [(v, n) for v, n in ((sup, "pivot"), (s1, "S1"))
                 if v is not None and v < p]
        if not bases:
            r.Dip_Basis = "-"
        else:
            base, basis = max(bases)
            lvl = base + cushion
            if lvl > p - a * DIP_MIN_ATR:
                r.Dip_Basis = "too-near"
            elif (not fenced and r.Rec_Stop is not None
                  and lvl <= r.Rec_Stop):
                # Unfenced, this is a TRADE: bidding underneath your own stop
                # is incoherent. Fenced, it is a core holding that is not being
                # stopped out at all, so the stop does not veto the bid.
                r.Dip_Basis = "below-stop"
            else:
                r.Rec_Dip, r.Dip_Basis = lvl, basis

    # ── BREAKOUT ─────────────────────────────────────────────────────────
    # The strictest of the four on purpose. Breakout adds are the weakest of
    # these four trades without confirmation, and an add on a name the system
    # already rates EXIT would be noise wearing a price.
    if not st_bull:
        r.Breakout_Basis = "downtrend"
    elif _num(score) is None or float(score) < BREAKOUT_MIN_SCORE:
        r.Breakout_Basis = "score<%d" % BREAKOUT_MIN_SCORE
    else:
        lvl = basis = None
        if res is not None and res > p:
            lvl, basis = res + cushion, "pivot"
        else:
            a_th = _num(ath)
            # Above 5x price the ATH is a reverse-split artifact, not a high.
            if a_th is not None and p < a_th < p * ATH_MAX_MULTIPLE:
                lvl, basis = a_th + cushion, "ATH"
        if lvl is None:
            r.Breakout_Basis = "-"
        else:
            lvl = max(lvl, p + a * BRK_MIN_ATR)
            r.Rec_Breakout, r.Breakout_Basis = lvl, basis

    # Round LAST, so the clamps and comparisons above work on exact numbers and
    # only the published figure is quantised.
    for f in FIELDS:
        setattr(r, f, tick_round(getattr(r, f)))
    return r


def recommend_row(row, fenced: bool = False, price=None) -> Rec:
    """Adapt one signals-table row (dict or pandas Series) to recommend().

    The column names here are the signals CSV's, spaces and all. Kept in ONE
    place so the pure function above never learns them — callers with a
    differently shaped source can use recommend() directly.

    `price` overrides the row's close: the Dashboard has a live quote, and
    recommending levels around a stale close while displaying a live price
    would put the two columns visibly at odds.
    """
    g = row.get
    return recommend(
        price if price is not None else (g("Current Price") or g("Last Close")),
        g("ATR"),
        supertrend=g("Supertrend"),
        supertrend_signal=g("Supertrend Signal", ""),
        nearest_support=g("Nearest_Support"),
        nearest_resistance=g("Nearest_Resistance"),
        mrc_zone=g("MRC_Zone", ""),
        mrc_r1=g("MRC_R1"), mrc_mean=g("MRC_Mean"), mrc_s1=g("MRC_S1"),
        ath=g("ATH"), score=g("Score_Weighted"),
        structure=g("Structure", ""), regime=g("Regime", ""),
        fenced=fenced,
    )


def attach(signals_df, fenced_tickers=None):
    """Add the eight Rec_* / *_Basis columns to a signals DataFrame.

    Used by the 45° scan and anything else wanting the levels in bulk. The
    Dashboard does NOT use this — it recommends per block with the live quote.
    """
    import pandas as pd
    fenced_tickers = {str(t).upper() for t in (fenced_tickers or ())}
    recs = [recommend_row(r, fenced=str(r.get("Ticker", "")).upper() in fenced_tickers)
            for _, r in signals_df.iterrows()]
    return pd.concat(
        [signals_df.reset_index(drop=True),
         pd.DataFrame([x.as_dict() for x in recs])], axis=1)
