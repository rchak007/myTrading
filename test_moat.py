#!/usr/bin/env python3
"""test_moat.py — our own moat rating. No published rating is referenced."""
import sys
sys.path.insert(0, "/home/chakravarti/github/myTrading")
from core import moat


def _yrs(vals, start=2016):
    return {start + i: v for i, v in enumerate(vals)}


def test_a_long_record_of_high_steady_returns_is_wide():
    """Microsoft's shape: 30%+ ROE every year on a 68% gross margin that
    barely moves. If this is not a moat the word means nothing."""
    r, why = moat.rate(roe_by_year=_yrs([0.32] * 10),
                       gross_margin_by_year=_yrs([0.68] * 10))
    assert r == moat.WIDE, (r, why)


def test_wide_needs_a_long_record():
    """'Durable' is a claim about time. Four spectacular years cannot earn
    it, however good they look."""
    r, _ = moat.rate(roe_by_year=_yrs([0.40] * 4),
                     gross_margin_by_year=_yrs([0.70] * 4))
    assert r == moat.NARROW


def test_a_commodity_maker_is_not_wide_however_good_the_peak():
    """Memory: huge returns at the top of the cycle, losses at the bottom,
    gross margin swinging 20 points. That is a cycle, not a moat."""
    r, why = moat.rate(
        roe_by_year=_yrs([0.30, -0.10, 0.25, -0.15, 0.35, 0.05, -0.08,
                          0.28, 0.12, -0.05]),
        gross_margin_by_year=_yrs([0.45, 0.20, 0.40, 0.18, 0.50, 0.30,
                                   0.22, 0.46, 0.33, 0.25]))
    assert r != moat.WIDE, why


def test_losses_are_none():
    r, _ = moat.rate(roe_by_year=_yrs([-0.2] * 10),
                     gross_margin_by_year=_yrs([0.26] * 10))
    assert r == moat.NONE


def test_pricing_power_without_returns_yet_is_narrow():
    """A young company holding a 70% gross margin is not Ford, even before
    the returns arrive."""
    r, _ = moat.rate(roe_by_year=_yrs([0.03] * 6),
                     gross_margin_by_year=_yrs([0.70] * 6))
    assert r == moat.NARROW


def test_too_little_history_says_so_rather_than_rating_none():
    """'We cannot judge' and 'no moat' are different claims. Conflating them
    quietly penalises every recent listing."""
    r, why = moat.rate(roe_by_year=_yrs([0.30, 0.28]))
    assert r is None and "not enough" in why


def test_the_window_ignores_a_startup_decade():
    """Over 16 years Tesla's median ROE is negative, because 2009-2018 was a
    company building factories at a loss. True about 2012, useless about the
    business now."""
    early = [-0.5] * 8
    recent = [0.24, 0.28, 0.22, 0.26, 0.21, 0.23, 0.25, 0.22, 0.27, 0.24]
    r, why = moat.rate(roe_by_year=_yrs(early + recent, start=2008),
                       gross_margin_by_year=_yrs([0.55] * 18, start=2008))
    assert r == moat.WIDE, (r, why)
    assert "over 10y" in why


def test_no_rating_is_seeded_from_a_published_source():
    import inspect
    src = inspect.getsource(moat)
    # Published figures must not appear as thresholds or targets. The
    # docstring may NAME the removed override table; it may not contain a
    # rating to look up.
    for leaked in ("564.54", "514.02", "26.36", "320.80", "352.76"):
        assert leaked not in src, leaked
    import oracle_value as ov
    assert not hasattr(ov, "MOAT_OVERRIDE")


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"  ok  {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL  {name}: {e}")
    print(f"\n{fails} failure(s)")
    sys.exit(1 if fails else 0)
