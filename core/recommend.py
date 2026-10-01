#!/usr/bin/env python3
"""
core/recommend.py
=================
Recommended PRICE LEVELS per ticker, in three tiers, derived from the
indicators jobStocksSignals already computes.

                 tier 1              tier 2               tier 3
    STOP         Rec_Stop            Rec_Stop_Hard          —
                 close-confirmed     disaster, resting
    TRIM         Rec_Trim            Rec_Trim2            Rec_Trim3
                 ── scale out in thirds ──
    DIP          Rec_Dip             Rec_Dip2               —
                 add / re-entry      the deeper one
    BREAKOUT     Rec_Breakout          —                    —

They mirror the four Dashboard coverage flags (Has_Stop / Has_Trim / Has_Dip /
Has_Breakout): the flag says whether an order EXISTS, the level says where one
WOULD go.

PURE. Prices in, prices out. No Schwab, no Sheets, no files, no clock — so the
whole thing is testable on Pi 2, which has none of those.

─────────────────────────────────────────────────────────────────────────────
TWO STOPS, NOT ONE — AND THEY USE DIFFERENT MECHANISMS
    A resting Schwab stop is a TOUCH trigger: a single wick takes you out at
    the worst price of the day, and you were right about the level. The Orders
    tab engine is CLOSE-triggered by construction, which wicks cannot reach.

    But close-confirmation is not free — it accepts gap risk. The night
    something halves you sell at the next open, far below your level.

    So both, at two distances:
        Rec_Stop       the technical level. Orders tab, SELL / CLOSE BELOW.
        Rec_Stop_Hard  ~1.5x further out. A resting Schwab STOP. Disaster only.

    Neither is a size. A stop level says where the thesis broke, not how much
    to sell, and at 1.5-4 ATR these get tagged by ordinary noise several times
    a year on the volatile names.

THE DIP IS A LADDER, AND IT HAS TWO TIERS
    The first version vetoed any dip below the stop, on the logic that bidding
    under your own stop is incoherent. That was wrong: the stop protects the
    shares you HOLD, the dip deploys FRESH capital at a better price. They are
    different money.

    Measured on 2026-10-01, that veto threw away real levels — PLTR had a
    pivot at 164.55 under a 171.78 stop, CRDO had MRC_S1 at 157.30 under a
    169.09 stop. Both are exactly where you want a bid.

        add       above the stop  — you still hold, this is an add
        re-entry  below the stop  — you were stopped out, this is the way back

    Candidates are a LADDER (pivot, MRC bands, ATR extensions), so one failed
    candidate never blanks the column.

TRIM IN THIRDS, NOT AT A PRICE
    A single trim level forces an all-or-nothing decision, and you will always
    feel you sold too early. Three levels capture the move.

THE THREE TRAPS, measured across the book on 2026-10-01
    1. ATH is CORRUPT on 15 of 128 names — reverse splits are not adjusted, so
       FCEL reads 234,900 against a $16 price. Rejected above 5x price.
    2. Supertrend is in SELL mode on 64 of 128 — the line then sits ABOVE
       price and is resistance, not support.
    3. MRC_S2 goes NEGATIVE, and MRC_R1 sits a median +19.9% away. Fine as a
       stretch marker, useless as an order level without zone gating.

FENCED IS NOT A KEEPER FLAG
    It means "when this sells, keep the proceeds earmarked to this ticker so I
    can pick it up later" — it PRESUMES a sale. Chakravarti, 2026-10-01:
    "fenced still does not mean i want to keep when its going down. It just
    means later i might pick it up."

    Nothing here means "hold through a drawdown". The system has no keeper
    concept at all — see Documentation/RECOMMENDED-LEVELS.md §4.

FAILS CLOSED, ALWAYS
    Every level is None rather than 0.0 when it cannot be computed, and every
    output is re-checked against the side of price it must be on. This
    system's recurring failure mode is a wrong answer that looks like a normal
    result — a 0.00 in a price cell eventually gets read as a price.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields as _dc_fields

# ── tunables ────────────────────────────────────────────────────────────────
# In ATR multiples. These are the only free parameters in the module.
STOP_MIN_ATR = 1.5        # closer than this is inside daily noise
STOP_MAX_ATR = 4.0        # further than this is not a stop, it is a hope
STOP_FALLBACK_ATR = 2.5   # pure-volatility stop when no structure survives
HARD_STOP_MULT = 1.5      # the disaster stop, as a multiple of the soft distance
HARD_STOP_MIN_ATR = 3.0   # ...but never nearer than this
HARD_STOP_MAX_ATR = 8.0   # ...and never further
# THE LADDER NEEDS ROOM TO BE A LADDER. With three rungs, a span of 2 ATR puts
# a full day's range between each — enough that price can plausibly stop
# between them. Without this, a soft stop clamped to STOP_MIN_ATR dragged the
# whole ladder into 1.5 ATR: TSLA came out 338.98 / 330.53 / 322.07, three
# rungs 0.75 ATR apart that would all trigger in the same two-day move.
STOP_LADDER_SPAN_ATR = 2.0
STOP_RUNG_GAP_ATR = 0.6   # minimum clearance for the middle rung
TRIM_MIN_ATR = 1.0
TRIM_MAX_ATR = 8.0        # ceiling for the FIRST trim
TRIM3_MAX_ATR = 16.0      # ceiling for the final third — it is allowed to reach
DIP_MIN_ATR = 1.0         # closer than this is not a dip
DIP_GAP_ATR = 0.5         # a re-entry must sit meaningfully below the stop
BRK_MIN_ATR = 0.5
PIVOT_CUSHION_ATR = 0.25  # how far off a pivot to sit, either side

ATH_MAX_MULTIPLE = 5.0    # above this, the ATH is a reverse-split artifact
BREAKOUT_MIN_SCORE = 50   # the system's own "consider" bar (see Regime_Action)

# Rendering order, by tier. `None` means that slot has no level in that tier.
TIER1 = ("Rec_Stop", "Rec_Trim", "Rec_Dip", "Rec_Breakout")
TIER2 = ("Rec_Stop2", "Rec_Trim2", "Rec_Dip2", None)
TIER3 = ("Rec_Stop_Hard", "Rec_Trim3", None, None)
FIELDS = TIER1          # kept: callers and tests use it as "the primary four"
ALL_LEVELS = tuple(n for tier in (TIER1, TIER2, TIER3) for n in tier if n)


@dataclass
class Rec:
    """Every level, and the reason for each of the four primary ones.

    Basis is not decoration. Without it the numbers are unfalsifiable — you
    cannot tell a structural stop from a volatility fallback, and the two
    deserve different confidence. A blank level carries the reason it is blank.
    """
    Rec_Stop: float | None = None
    Rec_Stop2: float | None = None
    Rec_Stop_Hard: float | None = None
    Rec_Trim: float | None = None
    Rec_Trim2: float | None = None
    Rec_Trim3: float | None = None
    Rec_Dip: float | None = None
    Rec_Dip2: float | None = None
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

    def tier(self, which: tuple) -> list:
        """The four slots of one tier, as values (None where empty)."""
        return [getattr(self, n) if n else None for n in which]


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
              mrc_zone="", mrc_r1=None, mrc_r2=None, mrc_mean=None,
              mrc_s1=None, mrc_s2=None,
              ath=None, score=None, structure="", regime="",
              fenced=False) -> Rec:
    """Every level for one ticker.

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
    r1, r2 = _num(mrc_r1), _num(mrc_r2)
    mean, s1, s2 = _num(mrc_mean), _num(mrc_s1), _num(mrc_s2)
    zone = _zone(mrc_zone)
    st_bull = str(supertrend_signal).upper().startswith("BUY")

    # ── STOP (tier 1: close-confirmed) ───────────────────────────────────
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
        clamped = max(min(lvl, p - a * STOP_MIN_ATR), p - a * STOP_MAX_ATR)
        # Say so when the clamp moved it. A level reported as `pivot` that is
        # really 4 ATR of pure volatility is the kind of plausible-looking
        # wrong answer this system keeps getting bitten by.
        if abs(clamped - lvl) > 1e-9:
            basis += "~cap"
        lvl = clamped
    else:
        lvl, basis = p - a * STOP_FALLBACK_ATR, "vol"
    soft_stop = lvl if (lvl > 0 and lvl < p) else None
    r.Rec_Stop, r.Stop_Basis = soft_stop, (basis if soft_stop else "-")

    # ── STOP (tiers 2 and 3) ─────────────────────────────────────────────
    # THE STOP IS A LADDER TOO: a third out at each of the first two levels,
    # everything at the third. Chakravarti's reading of the layout, and a
    # better design than the one it was describing.
    #
    # WHY SCALE OUT AT ALL. One level forces a binary decision on a position
    # you do not really want to leave, and stops get whipsawed. The trade-off
    # is explicit: in a REAL decline scaling costs you (you sell thirds at
    # -8%, -11%, -14% instead of all at -8%); in a WHIPSAW it saves you (only
    # a third left before the recovery). For a long-horizon book of quality
    # names whipsaws are the commoner event, so scaling wins on average.
    #
    # THE LAST RUNG IS DIFFERENT AND MUST STAY DIFFERENT. Tiers 1 and 2 are
    # CLOSE-confirmed, so wicks cannot reach them. Tier 3 is the disaster
    # stop: a resting Schwab STOP, a touch trigger, and a FULL exit. Leaving a
    # third on through a crash because the ladder said "a third at a time" is
    # exactly the wrong lesson to take from scaling out.
    if soft_stop is not None:
        soft_dist = p - soft_stop
        # The span term is what keeps the rungs apart. Proportional growth
        # alone collapses when the soft stop is tight: 1.5x of 1.5 ATR is
        # 2.25 ATR, so the whole ladder lived inside 0.75 ATR gaps.
        hard = p - max(soft_dist * HARD_STOP_MULT,
                       soft_dist + a * STOP_LADDER_SPAN_ATR,
                       a * HARD_STOP_MIN_ATR)
        hard = max(hard, p - a * HARD_STOP_MAX_ATR)
        if 0 < hard < soft_stop:
            r.Rec_Stop_Hard = hard
            # The middle rung must be meaningfully clear of BOTH neighbours —
            # three rungs inside one ATR is one stop pretending to be a plan.
            lo = hard + a * STOP_RUNG_GAP_ATR
            hi = soft_stop - a * STOP_RUNG_GAP_ATR
            if hi > lo:
                # Prefer REAL structure in that window: the next shelf down is
                # where a decline actually pauses. A shelf that hugs the soft
                # stop is no use, so it is filtered rather than accepted and
                # then rejected — otherwise one near-miss loses the whole rung.
                shelves = [v + cushion for v in (sup, s1, s2, mean)
                           if v is not None and lo <= v + cushion <= hi]
                r.Rec_Stop2 = max(shelves) if shelves else (soft_stop + hard) / 2.0

    # ── TRIM (three tiers: scale out) ────────────────────────────────────
    # The ZONE is the stretch measurement, so it chooses the FIRST level. The
    # below-mean branch matters most: it covers 39 of 128 names, and for those
    # R1 is not a plan. FCEL at 16.81 has R1 at 29.26 (+74%) — trimming there
    # is a wish. Reverting to the mean is the realistic first exit.
    if zone == "Strong_OB":
        t1, basis = p + a * 0.5, "at-mkt"       # maximally stretched: go now
    elif zone == "OB":
        t1, basis = (res, "pivot") if (res and res > p) else (p + a, "vol")
    elif zone in ("Above_Mean", "Near_Mean"):
        t1, basis = (r1, "R1") if r1 else (None, "-")
    elif zone in ("Below_Mean", "OS", "Strong_OS"):
        t1, basis = (mean, "mean") if mean else (None, "-")
    else:                                        # N/A — MRC still warming up
        t1, basis = (res, "pivot") if (res and res > p) else (None, "-")

    if t1 is not None:
        want = t1
        if zone != "Strong_OB":
            t1 = max(t1, p + a * TRIM_MIN_ATR)
        t1 = min(t1, p + a * TRIM_MAX_ATR)
        if abs(t1 - want) > 1e-9:
            basis += "~cap"
        if t1 > p:
            r.Rec_Trim, r.Trim_Basis = t1, basis
            # The far target: the outer band if it is sane and genuinely
            # beyond the first level, else a volatility extension. T2 splits
            # the difference, so the three are evenly spaced.
            t3 = r2 if (r2 and r2 > t1) else p + a * TRIM3_MAX_ATR
            t3 = min(t3, p + a * TRIM3_MAX_ATR)
            if t3 > t1 + a * 0.5:                # else the ladder is noise
                r.Rec_Trim3 = t3
                r.Rec_Trim2 = (t1 + t3) / 2.0
        else:
            r.Trim_Basis = "-"
    else:
        r.Trim_Basis = basis

    # ── DIP (a ladder, in two tiers) ─────────────────────────────────────
    # Structure levels get the cushion ABOVE them, the mirror of the stop
    # sitting just below: one level, two sides, defined risk. You want the
    # fill before the crowd's stops trigger, not after. Volatility extensions
    # need no cushion — they are not levels anyone else is watching.
    if str(structure).upper() == "BEARISH" and str(regime).upper() == "BEAR":
        r.Dip_Basis = "broken"                   # no bid into a broken name
    else:
        far_enough = lambda v: 0 < v <= p - a * DIP_MIN_ATR
        struct = sorted((v + cushion, n) for v, n in
                        ((sup, "pivot"), (s1, "S1"), (s2, "S2"), (mean, "mean"))
                        if v is not None and v < p and far_enough(v + cushion))
        # VOLATILITY RUNGS ARE A FALLBACK, NEVER A COMPETITOR. A line at
        # price-2ATR is not a level anyone else is watching, so it must not
        # outrank a real pivot just for being nearer — which it did on PLTR,
        # where 174.86 beat the 164.55 swing low.
        vol = sorted((p - a * m, f"{m:g}ATR") for m in (2.0, 3.0, 4.0)
                     if far_enough(p - a * m))

        # THE ADD TIER REQUIRES STRUCTURE, no volatility fallback. Adding at a
        # vol rung that happens to sit above the stop is the worst of both:
        # you buy, price keeps going, and 1 ATR later the stop takes out the
        # whole position including what you just added. If nothing structural
        # holds above the stop, the honest answer is a re-entry, not an add.
        above = [x for x in struct if soft_stop is None or x[0] > soft_stop]
        best_add = max(above) if above else None

        # A re-entry means you are OUT, and with a laddered stop you are not
        # fully out until the last rung. Measuring from the soft stop would
        # put a re-entry bid at a level where you still hold two thirds — and
        # worse, at the same shelf as your own second stop.
        exit_at = r.Rec_Stop_Hard or soft_stop
        reentry = ([x for x in struct if x[0] < exit_at - a * DIP_GAP_ATR]
                   if exit_at is not None else [])
        if not reentry and exit_at is not None:
            # Already out, and nothing structural below. A volatility rung is
            # at least a place to look.
            reentry = [x for x in vol if x[0] < exit_at - a * DIP_GAP_ATR]
        best_re = max(reentry) if reentry else None

        if best_add and best_re:
            r.Rec_Dip, r.Dip_Basis = best_add[0], "add:" + best_add[1]
            r.Rec_Dip2 = best_re[0]
        elif best_add:
            r.Rec_Dip, r.Dip_Basis = best_add[0], "add:" + best_add[1]
        elif best_re:
            # Stopped out first, then back in. This is the case the old veto
            # threw away — PLTR's 164.55 pivot under a 171.78 stop.
            r.Rec_Dip, r.Dip_Basis = best_re[0], "re:" + best_re[1]
            deeper = [x for x in reentry if x[0] < best_re[0] - a * DIP_GAP_ATR]
            r.Rec_Dip2 = max(deeper)[0] if deeper else None
        else:
            r.Dip_Basis = "-"

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
            r.Rec_Breakout = max(lvl, p + a * BRK_MIN_ATR)
            r.Breakout_Basis = basis

    # Round LAST, so the clamps and comparisons above work on exact numbers
    # and only the published figure is quantised.
    for f in ALL_LEVELS:
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
        mrc_r1=g("MRC_R1"), mrc_r2=g("MRC_R2"), mrc_mean=g("MRC_Mean"),
        mrc_s1=g("MRC_S1"), mrc_s2=g("MRC_S2"),
        ath=g("ATH"), score=g("Score_Weighted"),
        structure=g("Structure", ""), regime=g("Regime", ""),
        fenced=fenced,
    )


def earnings_soon(row) -> bool:
    """Whether this ticker has earnings inside the alert window.

    `Earnings_Alert` is computed by data/stock_scoring.get_earnings_alert and
    is either "🔴 EARNINGS SOON" or empty. Read here rather than in
    orders_sheet so there is one definition of the question.

    It does NOT move any level today. Holding a close-confirmed stop through
    an earnings gap is exactly how the soft stop fails, so it is surfaced as a
    warning and the judgement is left to the human — see PROJECT_PLAN.md §6.
    """
    v = row.get("Earnings_Alert")
    return bool(v) and str(v).strip().lower() not in ("nan", "none", "")


def attach(signals_df, fenced_tickers=None):
    """Add the Rec_* / *_Basis columns to a signals DataFrame.

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
