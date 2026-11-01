#!/usr/bin/env python3
"""
test_oracle_iq.py
=================
The six-factor panel. The anchor test is TSLA: StockOracle's OracleIQ tab
prints Predictability **Medium**, and that is the one published rank we can
check our arithmetic against.
"""
import sys

from core import oracle_iq as iq


def _q(pairs):
    """Quarters as (actual, estimate), newest first."""
    return pairs


# TSLA's last 12 quarters, read off the StockOracle Earnings tab and
# confirmed against yfinance (they agree line for line).
TSLA = _q([(0.48, 0.45), (0.33, 0.54), (0.41, 0.35), (0.50, 0.45),
           (0.50, 0.56), (0.40, 0.40), (0.27, 0.41), (0.73, 0.77),
           (0.72, 0.60), (0.52, 0.62), (0.45, 0.49), (0.71, 0.74)])

MSFT = _q([(3.72, 3.66), (3.65, 3.56), (3.46, 3.22), (3.23, 3.11),
           (3.30, 3.10), (2.95, 2.93), (2.94, 2.78), (2.99, 2.78),
           (2.69, 2.65), (2.93, 2.81), (2.45, 2.55), (2.69, 2.65)])


def test_tsla_is_medium_the_one_rank_we_can_check():
    """The published OracleIQ rank for TSLA is Medium. If this ever fails,
    the thresholds drifted away from the only ground truth we have."""
    rank, why = iq.predictability(TSLA)
    assert rank == iq.MEDIUM, f"TSLA should be Medium, got {rank} ({why})"


def test_msft_beats_tsla_on_predictability():
    assert iq.predictability(MSFT)[0] == iq.HIGH
    assert iq.predictability(TSLA)[0] == iq.MEDIUM


def test_a_chronic_loss_maker_is_never_predictable():
    """AAOI lost money in 11 of 12 quarters. However tight the estimates
    were, that is not a company whose earnings you can plan around."""
    losses = [(-0.10, -0.09), (-0.12, -0.11), (-0.21, -0.20),
              (-0.15, -0.14), (-0.33, -0.31), (-0.26, -0.25),
              (-0.19, -0.18), (-0.40, -0.38), (-0.28, -0.27),
              (-0.22, -0.21), (0.02, 0.01), (-0.17, -0.16)]
    rank, why = iq.predictability(losses)
    assert rank == iq.LOW, f"got {rank} ({why})"
    assert "loss-making" in why


def test_too_few_quarters_says_so_rather_than_guessing():
    rank, why = iq.predictability([(1.0, 0.9), (1.1, 1.0)])
    assert rank is None
    assert "quarters" in why


def test_no_earnings_at_all_is_not_a_crash():
    assert iq.predictability(None)[0] is None
    assert iq.predictability([])[0] is None


def test_zero_eps_base_does_not_explode():
    """The bug that made SITM score 635%: percentage surprise on a near-zero
    base. A median |EPS| of zero must return None, not infinity."""
    rank, why = iq.predictability([(0.0, 0.01)] * 8)
    assert rank is None
    assert "zero" in why


def test_future_quarter_with_no_actual_is_dropped():
    """yfinance returns the NEXT quarter with an estimate and a NaN actual.
    Counting that as a miss would be nonsense."""
    import math
    rows = [(math.nan, 0.50)] + TSLA
    assert iq.predictability(rows)[0] == iq.MEDIUM


def test_profitability_uses_the_normalized_margin():
    """TSLA: trailing net margin 3.11%, five-year average 10.50%. Their
    OracleIQ says Profitability HIGH. Valuing the trailing figure is the
    cyclical trap; the normalized one must win."""
    trailing = iq.profitability(net_margin=0.0311, roe=0.0380)
    normalized = iq.profitability(net_margin=0.0311, roe=0.1961,
                                  normalized_margin=0.1050)
    assert normalized[0] == iq.HIGH, normalized
    assert trailing[0] != iq.HIGH


def test_financial_strength_rewards_net_cash():
    strong = iq.financial_strength(total_cash=41e9, total_debt=13e9,
                                   free_cf=3e9, current_ratio=2.0)
    assert strong[0] == iq.HIGH
    assert "net cash" in strong[1]


def test_financial_strength_punishes_a_cash_burner():
    """AAOI: burning free cash with more debt than cash."""
    weak = iq.financial_strength(total_cash=40e6, total_debt=300e6,
                                 free_cf=-887e6, market_cap=3e9)
    assert weak[0] == iq.LOW, weak


def test_missing_inputs_report_why_rather_than_ranking_blind():
    for rank, why in (iq.profitability(), iq.growth(),
                      iq.financial_strength()):
        assert rank is None
        assert why


def test_panel_has_all_six_in_stockoracle_order():
    p = iq.panel(earnings=TSLA, moat="narrow", band="overvalued",
                 normalized_margin=0.1050, roe=0.1961, net_margin=0.0311,
                 revenue_growth=0.22, total_cash=41e9, total_debt=13e9,
                 free_cf=-3.25e9, current_ratio=2.0)
    assert list(p) == ["Predictability", "Profitability", "Growth",
                       "OracleMoat", "FinancialStrength", "Valuation"]
    assert p["Predictability"][0] == iq.MEDIUM
    assert p["OracleMoat"][0] == "Narrow"
    assert p["Valuation"][0] == "Overvalued"
    line = iq.summary(p)
    assert "Pred Medium" in line and "Moat Narrow" in line


def test_panel_keeps_unavailable_factors_visible():
    """A blank must never be mistaken for a zero."""
    p = iq.panel(earnings=None)
    assert p["Predictability"][0] is None
    assert p["Predictability"][1]


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"  ok  {name}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{fails} failure(s)")
    sys.exit(1 if fails else 0)
