"""
core/oracle.py
==============
OracleValue — a conservative fair value per share.

WHAT THIS IS, AND IS NOT. StockOracle™ (Piranha Profits / Adam Khoo) computes
a proprietary, adaptive OracleValue™ we cannot see. Their own handoff note
says so: *"OracleValue is proprietary and adaptive; exact replication is not
expected. Focus on relative conservatism, moat impact, and growth
normalization."*

So this is OUR deterministic approximation, built to behave the way theirs
behaves on the three names we can check. It is a ballpark, it says so, and it
is reproducible — the same inputs give the same number every time, which an
LLM asked to "estimate like StockOracle" does not.

WHY DETERMINISTIC AND NOT A MODEL CALL. Chakravarti was shown a Grok estimate
of $80–$110 for AAOI at $69, called "modestly to moderately undervalued."
StockOracle's actual figure, in the screenshots he sent, is **$26.36 — 161.95%
OVERVALUED, No Moat**. Backwards, and out by ~3.5x. A plausible paragraph is
not a valuation, and there is no way to tell one from the other by reading it.

FOUR CALIBRATION POINTS (2026-10-30/11-01, from the StockOracle report):

    MSFT   price 521.91   OracleValue 564.54   Wide moat    7.6% under
    SITM   price 488.08   OracleValue 514.02   Narrow moat  5.3% under
    TSLA   price 399.81   OracleValue 320.80   Narrow moat 24.6% OVER
    AAOI   price  69.05   OracleValue  26.36   No moat    161.9% OVER

WHAT THE TSLA PAGE FINALLY EXPLAINED. Its Valuation Chart lists NINE
sub-models spanning $57.95 (PEG) to $858.10 (DCF-20), with OracleValue 320.80
sitting beside the Mean-Price-to-Sales model at 352.76. On MSFT the same
relationship holds: OracleValue 564.54 against a Mean-PS of 538.82. Better
still, TSLA's own 10-year average P/S of 10.63 times revenue per share of
29.25 gives $310.93 — three per cent from the published figure.

So OracleValue is NOT a universal fair multiple applied to everyone. It is
what THIS stock has historically been worth per dollar of sales, applied to
today's sales, with the earnings models ignored whenever earnings are
unreliable. That is their word "adaptive", and it is why SiTime earns 33x
sales and Applied Optoelectronics 3.8x on identical 3% net margins.

A universal BASE_PE framework cannot reproduce that. The first version of
this module used one and put TSLA at $22.67.

What those three teach, and what this module is built around:

  1. EARNINGS, NOT PROMISES. AAOI's forward P/E of 15 implies $4.60 of EPS on
     $7.02 of revenue per share — a 65% net margin for an optics manufacturer.
     The forward estimate is fiction; the trailing loss is fact. We anchor on
     achieved results and treat forward figures as a ceiling, never a floor.

  2. CASH BURN IS THE KILLER. AAOI burned $890M of free cash on $596M of
     revenue — 1.5x its own sales. No conservative model survives that intact,
     and the 3.76x sales OracleValue implies theirs does not either.

  3. MOAT MOVES THE MULTIPLE, not the method. MSFT and SITM both land within
     ~8% of price. AAOI lands at a third of it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────
# A business with no moat and no growth, earning steadily, is worth about
# this many times its sustainable earnings. Everything else adjusts from here.
BASE_PE = 15.0

# Moat is the single biggest lever, as it is for them.
MOAT_BONUS = {"WIDE": 9.0, "NARROW": 4.5, "NONE": 0.0}

# Growth is worth paying for, but NOT linearly and NOT forever. 84% growth
# does not earn 84 points of multiple — StockOracle's own SITM note calls the
# name "high-growth speculative" while still landing within 5% of price.
GROWTH_CAP = 1.30          # growth above 130% contributes nothing further
GROWTH_TO_PE = 28.0        # 130% capped growth -> at most +36 of PE

# Margin of safety. 1.00 after calibration — the conservatism already lives
# in BASE_PE 15, the growth discount for loss-makers, and the penalties.
# Stacking another haircut on top put MSFT 11% below a figure the rest of the
# model had already nailed to 1%.
MARGIN_OF_SAFETY = 1.00

# Discount rate for the compounded anchor. Deliberately ABOVE a risk-free
# rate and below the WACCs StockOracle shows (TSLA 15.84%, MSFT 6.61%) —
# a single rate the whole book shares, rather than a per-name WACC we cannot
# reproduce.
DISCOUNT = 0.11

# A company whose cash flow and earnings both fail gets valued on sales at a
# multiple set by the margin it ACTUALLY achieves, floored here so a single
# bad quarter does not mark it to zero.
MIN_SALES_MULT = 0.8
MAX_SALES_MULT = 18.0

# Penalties. Each is a multiplier on the final figure, and they compound.
P_FCF_BURN = 0.70          # free cash flow negative
P_LOSS = 0.80              # net income negative
P_DEBT = 0.85              # net debt > 3x operating cash flow
P_DILUTION = 0.85          # share count up more than 10% in a year
P_HIGH_BETA = 0.92

# Above this multiple of the share price, the output is a data artefact
# rather than a valuation.
#
# The book itself picks the line. Sorted by value/price there is a clean
# break: the plausible opinions stop around 2.6x (NVDA 1.22x, META 1.68x,
# SMCI 2.09x, APLD 2.57x) and past 3x it is all breakage — TSM at 25x on
# four years of history, SOFI 7x, RDDT 6x, CLSK 3,075x, and RCAT at
# 8,288,439x the share price. Nearly all of them share one cause: no mean
# P/S could be built, so the dominant anchor was missing and the earnings
# anchors ran unchecked on bad inputs.
#
# 3.0 suppresses 22 of 116. Between 2.5x and 4.0x only five names change
# hands, so the exact point is not delicate; what matters is that a
# CONSERVATIVE model claiming a stock is worth more than three times what
# the market pays is making an extraordinary claim, and ours was making it
# by accident.
MAX_PRICE_MULTIPLE = float(os.getenv("ORACLE_MAX_MULTIPLE", "3.0"))         # beta > 2.5

BANDS = [
    (0.85, "DEEP VALUE"),      # price <= 85% of value
    (0.95, "UNDERVALUED"),
    (1.08, "FAIR"),
    (1.35, "RICH"),
    (float("inf"), "OVERVALUED"),
]


@dataclass
class Oracle:
    value: float | None        # the point estimate, per share
    low: float | None          # the range Chakravarti asked for
    high: float | None
    band: str
    moat: str
    why: str
    gap_pct: float | None      # (value / price - 1) * 100

    def as_text(self) -> str:
        """The cell the sheet shows: '~$80–$110' style, or a single value."""
        if self.value is None:
            return ""
        if self.low and self.high and self.high > self.low * 1.02:
            return f"${self.low:,.0f}–${self.high:,.0f}"
        return f"${self.value:,.2f}"


def _n(v):
    try:
        f = float(v)
        return None if f != f else f          # NaN check without importing math
    except (TypeError, ValueError):
        return None


def infer_moat(gross_margin=None, roe=None, market_cap=None,
               net_margin=None) -> str:
    """A MOAT PROXY, because StockOracle's rating is community-voted and we
    cannot read it for an arbitrary ticker.

    Deliberately hard to earn WIDE. Checked against the three known ratings:
    MSFT (67.7% GM, 34% ROE, $3.9T) -> WIDE; SITM (high GM, thin margins,
    small) -> NARROW; AAOI (34.9% GM, negative) -> NONE.
    """
    gm, r, cap, nm = _n(gross_margin), _n(roe), _n(market_cap), _n(net_margin)
    score = 0
    # GROSS MARGIN IS THE CLEAREST SINGLE MOAT SIGNAL, so it can earn a
    # NARROW rating on its own. SiTime has a 2% net margin and a 1.3% ROE —
    # every other test scores it zero — but a 58.7% gross margin is pricing
    # power, and StockOracle's community rates it Narrow.
    if gm is not None:
        score += 3 if gm >= 0.55 else 1 if gm >= 0.40 else 0
    if r is not None:
        score += 2 if r >= 0.20 else 1 if r >= 0.10 else 0
    if nm is not None:
        score += 2 if nm >= 0.20 else 1 if nm >= 0.08 else 0
    if cap is not None:
        score += 2 if cap >= 5e11 else 1 if cap >= 2e10 else 0
    return "WIDE" if score >= 6 else "NARROW" if score >= 3 else "NONE"


def value(price=None, *, shares=None, revenue=None, net_income=None,
          operating_cf=None, free_cf=None, gross_margin=None,
          net_margin=None, roe=None, market_cap=None, growth=None,
          beta=None, total_debt=None, total_cash=None,
          share_growth=None, moat=None, forward_eps=None,
          normalized_margin=None, proj_growth=None, hist_ps=None,
          hist_ps_years=None, hist_ps_dispersion=None) -> Oracle:
    """A conservative fair value per share, plus the range and the reasoning.

    Every input is optional. Missing data narrows what can be asserted and is
    reported in `why` rather than silently assumed — a fair value computed
    from three fields and one from ten should not look identical.
    """
    px = _n(price)
    sh = _n(shares)
    rev, ni = _n(revenue), _n(net_income)
    ocf, fcf = _n(operating_cf), _n(free_cf)

    if not sh or sh <= 0 or px is None or px <= 0:
        return Oracle(None, None, None, "", "", "no share count or price", None)

    notes_unpaid = False
    moat = (moat or infer_moat(gross_margin, roe, market_cap, net_margin)).upper()
    if moat not in MOAT_BONUS:
        moat = "NONE"

    # ── the multiple ────────────────────────────────────────────────────
    g = max(0.0, min(_n(growth) or 0.0, GROWTH_CAP))
    # GROWTH EARNS LESS WHEN IT IS NOT PAID FOR. AAOI grew revenue 86% while
    # burning $887M of free cash; SITM grew 126% with earnings, however thin.
    # Giving both the same multiple bonus is how a model ends up calling a
    # cash furnace cheap — which is precisely the error in the Grok estimate.
    if (ni is not None and ni < 0) or (fcf is not None and fcf < 0):
        g *= 0.45
        if (_n(growth) or 0) > 0:
            notes_unpaid = True
    fair_pe = BASE_PE + MOAT_BONUS[moat] + g * GROWTH_TO_PE
    notes = [f"{moat.lower()} moat"]
    if notes_unpaid:
        notes.append("growth discounted (unprofitable)")
    if g > 0:
        notes.append(f"growth {100*g:.0f}%{' (capped)' if (_n(growth) or 0) > GROWTH_CAP else ''}")

    # ── the anchors ─────────────────────────────────────────────────────
    # Earnings first, because that is what a share finally is. Cash flow
    # second, as the check on whether the earnings are real. Sales last, and
    # only as the floor for a company the first two cannot value.
    anchors: list[tuple[str, float, float]] = []        # (name, value, weight)

    eps = (ni / sh) if ni is not None else None

    # NORMALIZED EARNINGS — the single biggest correction in this module.
    #
    # StockOracle's TSLA page reports Net Profit Margin TTM 3.11% beside a
    # 5-year average of 10.50%, ROE 3.80% beside 19.61%, ROIC 3.32% beside
    # 16.86%. They value the AVERAGE, not the trough. Reproduced exactly from
    # the income statement: TSLA's last five fiscal years ran 10.3 / 15.4 /
    # 15.5 / 7.3 / 4.0 %, mean 10.50%.
    #
    # This is the cyclical P/E trap out of WALL-STREET-LEVEL-STOCK-ANALYSIS.md
    # PART D, in code: "for cyclicals, E is a point on a wave, not a stable
    # number." Valuing TSLA on a $0.91 trough EPS produced $22.67 against a
    # published $320.80 — not conservative, just wrong.
    nm5 = _n(normalized_margin)
    norm_eps = None
    if nm5 is not None and nm5 > 0 and rev and rev > 0:
        norm_eps = (rev * nm5) / sh
        if eps is None or norm_eps > eps:
            notes.append(f"normalized on 5y margin {100*nm5:.1f}%")

    earn = norm_eps if norm_eps is not None else eps
    if earn is not None and earn > 0:
        anchors.append(("earnings", earn * fair_pe, 0.55))

        # COMPOUNDED, because that is what they are paying for. A terminal
        # multiple on earnings five years out, discounted back — the shape of
        # every DCF on their valuation-comparison table.
        gp = _n(proj_growth)
        if gp is not None and gp > 0:
            gp = min(gp, 0.60)                   # 60% for five years is already heroic
            terminal = (BASE_PE + MOAT_BONUS[moat]) * 1.15
            fwd = earn * (1 + gp) ** 5 * terminal / (1 + DISCOUNT) ** 5
            anchors.append(("5y compounded", fwd, 0.45))
            notes.append(f"5y @ {100*gp:.0f}%")

    # FORWARD EARNINGS, AS A CEILING — and only for a company already making
    # money. This is the line AAOI must not be allowed to cross: its forward
    # P/E of 15 implies $4.60 of EPS on $7.02 of revenue per share, a 65% net
    # margin for an optics manufacturer. Requiring positive TTM earnings to
    # use a forward figure keeps the fiction out while still letting a real
    # but temporarily-depressed earner (SITM at 798x trailing, 30x forward)
    # be valued on something other than its worst quarter.
    feps = _n(forward_eps)
    if feps is not None and feps > 0 and eps is not None and eps > 0:
        anchors.append(("forward earnings", feps * fair_pe * 0.80, 0.30))
        notes.append("fwd EPS used")

    if ocf is not None and ocf > 0:
        # Operating cash flow carries a lower multiple than earnings: it is
        # before the capex that keeps the business alive.
        anchors.append(("cash flow", (ocf / sh) * fair_pe * 0.62, 0.10))

    if rev is not None and rev > 0:
        # The sales multiple is EARNED, not granted: it is the margin the
        # company actually achieves times the multiple its quality supports.
        # A 40% net margin and a wide moat gets a high one honestly; a loss
        # gets the floor.
        # GROSS margin, not net. Net margin is where the company is in its
        # investment cycle; gross margin is what the business model can earn
        # when it stops spending. Using net crushed SITM to 0.8x sales on a
        # 2% net margin while StockOracle paid 33x, and that is a modelling
        # error rather than conservatism.
        gm = _n(gross_margin)
        if gm is None:
            nm = _n(net_margin)
            if nm is None and ni is not None and rev:
                nm = ni / rev
            gm = max(0.0, (nm or 0.0)) * 2.5      # crude, and flagged below
            notes.append("gross margin unknown")
        moat_f = {"WIDE": 1.30, "NARROW": 1.15, "NONE": 1.00}[moat]
        mult = min(MAX_SALES_MULT, max(MIN_SALES_MULT,
                                       max(0.0, gm) * BASE_PE * moat_f))
        # Weight depends on whether anything better exists. For a profitable
        # company sales is a sanity check; for a loss-maker it is the only
        # honest anchor left, so it carries the whole weight.
        w = 0.05 if (eps is not None and eps > 0) else 1.00
        anchors.append(("sales", (rev / sh) * mult, w))
        notes.append(f"sales {mult:.1f}x")

    # THE DOMINANT ANCHOR — see mean_ps() in oracle_value.py for why.
    # Weighted above everything else because it is the one thing that
    # reproduces all four published OracleValues; the rest are now sanity
    # checks on it rather than the valuation itself.
    hps = _n(hist_ps)
    if hps is not None and hps > 0 and rev and rev > 0:
        # DOMINANT, not one vote among many. TSLA's own sub-model table shows
        # the four sales/book models clustered at 345-385 while the four
        # earnings models sit at 95-148, and OracleValue 320.80 lands with the
        # first group. Their word for this is "adaptive": when earnings are
        # unreliable — TSLA at 439x trailing, SITM at 798x, AAOI negative —
        # the multiple the market has actually paid for this company's SALES
        # is the only stable anchor, and the earnings models get ignored.
        #
        # Weighted 6.0 against the others' 0.05-0.55 so they become sanity
        # checks rather than co-equal votes. Blending them equally pulled TSLA
        # to $199 when this anchor alone said $339 against a published $320.80.
        #
        # HOW MUCH TO TRUST IT. The weight of 6.0 assumes the mean is drawn
        # from a decent run of years and that those years agree with each
        # other. Neither is guaranteed, so both are measured and the weight
        # is cut when they fail:
        #
        #   few years -- a mean of four observations is a guess. NVDA's four
        #                were the AI bubble (34x 43x 29x 22x) and valued a
        #                $205 stock at $434. Across the four published
        #                anchors the relationship is monotonic:
        #                   SITM  7y -> 54% error
        #                   AAOI 10y -> 19%
        #                   MSFT 11y -> 15%
        #                   TSLA 16y ->  6%
        #
        # A discounted anchor hands weight to the earnings and cash-flow
        # models rather than shrinking the answer, because the re-weighting
        # below normalizes by whatever weight actually remains.
        #
        # DISPERSION IS MEASURED BUT DOES NOT MOVE THE VALUE. On four years
        # of history it looked decisive -- MSFT scattered 9% and we valued it
        # exactly, AAOI scattered 50% and we were 42% out -- so an earlier
        # version cut the weight by 40% above a 0.35 spread. Real history
        # kills the rule outright: TSLA has the WIDEST spread of the four
        # (0.76) and the SMALLEST error (6%), while SITM has the narrowest
        # (0.30) and the largest (54%). It also fired on two thirds of the
        # book, and a warning that common is not a warning. It stays in the
        # output as description -- a stock whose multiple has ranged 1x to
        # 6x is worth knowing about -- and out of the arithmetic.
        w = 6.00
        n = _n(hist_ps_years)
        cv = _n(hist_ps_dispersion)
        if n is not None and n < 8:
            w *= max(0.45, n / 10.0)
            notes.append(f"only {int(n)}y of sales history")
        if cv is not None and cv > 0.90:        # top quartile; descriptive
            notes.append(f"P/S has ranged widely (spread {cv:.0%})")
        anchors.append(("historical P/S", (rev / sh) * hps, w))
        notes.insert(0, f"own mean P/S {hps:.1f}x"
                        + (f" over {int(n)}y" if n else ""))

    if not anchors:
        return Oracle(None, None, None, "", moat, "no revenue, earnings or cash flow", None)

    # Re-weight to whatever anchors actually exist, so a missing one shifts
    # weight rather than quietly scaling the answer toward zero.
    tot = sum(w for _, _, w in anchors)
    raw = sum(v * w for _, v, w in anchors) / tot

    # ── penalties ───────────────────────────────────────────────────────
    penalties: list[tuple[float, str]] = []
    if fcf is not None and fcf < 0:
        penalties.append((P_FCF_BURN, "fcf"))
        burn = f" (${abs(fcf)/1e6:,.0f}M)" if abs(fcf) >= 1e6 else ""
        notes.append(f"FCF NEGATIVE{burn}")
    if ni is not None and ni < 0:
        penalties.append((P_LOSS, "loss"))
        notes.append("loss-making")
    nd = (_n(total_debt) or 0.0) - (_n(total_cash) or 0.0)
    if ocf and ocf > 0 and nd > 3 * ocf:
        penalties.append((P_DEBT, "debt"))
        notes.append("net debt > 3x OCF")
    sg = _n(share_growth)
    if sg is not None and sg > 0.10:
        penalties.append((P_DILUTION, "dilution"))
        notes.append(f"shares +{100*sg:.0f}%")
    b = _n(beta)
    if b is not None and b > 2.5:
        penalties.append((P_HIGH_BETA, "beta"))
        notes.append(f"beta {b:.1f}")

    # COMPOUNDING WAS THE BUG. Three penalties multiplied to 0.35x and put
    # AAOI at $1.79 against a known $26.36 — a model so harsh it is useless
    # is no better than one so loose it is wrong. The worst flag sets the
    # level; every additional one costs a further 8%.
    if penalties:
        worst = min(p for p, _ in penalties)
        mult = worst * (0.92 ** (len(penalties) - 1))
    else:
        mult = 1.0

    val = raw * mult * MARGIN_OF_SAFETY

    # ── the range ───────────────────────────────────────────────────────
    # Width reflects how much disagreement the anchors themselves carry. A
    # name whose earnings, cash flow and sales all say the same thing gets a
    # tight range; one that does not, gets an honest wide one.
    #
    # CAPPED AT 25%, NOT 45%. A band of +/-45% is not a range, it is a
    # refusal to answer wearing a number: TSLA printed "$203-$536" around a
    # $369 point, which rules out nothing a trader could act on. Where the
    # anchors disagree by more than the cap, that disagreement is now SAID
    # rather than drawn, because a reader cannot tell a wide honest range
    # from a wide broken one by looking at it.
    spread = 0.12
    if len(anchors) > 1:
        vals = [v for _, v, _ in anchors]
        raw_spread = (max(vals) - min(vals)) / (2 * raw) if raw else 0.0
        spread = max(0.10, min(0.25, raw_spread))
        if raw_spread > 0.25:
            notes.append(f"anchors disagree {raw_spread:.0%} — low confidence")
    low, high = val * (1 - spread), val * (1 + spread)

    # A value that rounds to nothing is not a valuation. A pre-revenue
    # biotech (PBLS) has no fair value by this method, and "$0.00 —
    # OVERVALUED" asserts one. Say nothing instead.
    if val < 0.01 or val < px * 0.02:
        return Oracle(None, None, None, "", moat,
                      "no honest value — " + " · ".join(notes), None)

    # THE SAME TEST AT THE OTHER END, which was missing. A conservative fair
    # value cannot be several times the market price; past some multiple the
    # model has broken, not spotted a bargain the whole market missed.
    #
    # SNDK is the case that forced it. SanDisk spun out of Western Digital in
    # February 2025, so it has 21 months of price history — too little for a
    # mean P/S, which left the DOMINANT anchor missing. The earnings anchors
    # then ran unchecked on a yfinance net income of $19.4B against $29.0B of
    # revenue: a 67% net margin for a memory manufacturer, which is spinoff
    # accounting, not profit. Result: $13,712 against a $1,151 share price,
    # printed on the Dashboard with no flag of any kind.
    #
    # Blank is the honest output. The same rule already applies at the bottom
    # end for pre-revenue names, and it should never have applied only there.
    if val > px * MAX_PRICE_MULTIPLE:
        return Oracle(None, None, None, "", moat,
                      f"no honest value — implies {val / px:.0f}x the share "
                      f"price, so an input is wrong"
                      + (" · " + " · ".join(notes) if notes else ""), None)

    ratio = px / val if val > 0 else 99.0
    band = next(b for lim, b in BANDS if ratio <= lim)
    gap = (val / px - 1) * 100

    return Oracle(round(val, 2), round(low, 2), round(high, 2), band, moat,
                  " · ".join(notes), round(gap, 1))
