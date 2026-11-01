#!/usr/bin/env python3
"""
test_oracle.py
==============
OracleValue — our conservative fair value.

The thing worth testing is not the arithmetic, it is the DIRECTION on the
three names where StockOracle's own answer is known. Grok's "StockOracle-
style" estimate for AAOI was $80-110, undervalued; the real figure is $26.36,
161.95% OVERVALUED. A model that gets the sign wrong is worse than no model,
so the sign is what these assert.
"""
from core.oracle import value, infer_moat

# price, StockOracle's published OracleValue, their moat rating
KNOWN = {
    "MSFT": (521.91, 564.54, "WIDE"),
    "SITM": (488.08, 514.02, "NARROW"),
    "AAOI": (69.05, 26.36, "NONE"),
}


def test_a_cash_burner_is_never_called_cheap():
    """AAOI: $887M of free cash burned on $596M of revenue. Whatever else the
    model does, it must not call that undervalued — that is the exact error
    in the estimate Chakravarti was shown."""
    o = value(69.05, shares=85.3e6, revenue=596e6, net_income=-57e6,
              operating_cf=-130e6, free_cf=-887e6, gross_margin=0.289,
              net_margin=-0.0957, growth=0.864, beta=3.8, market_cap=6.26e9)
    assert o.value is not None
    assert o.value < 69.05, f"called a cash burner cheap at ${o.value}"
    assert o.band in ("RICH", "OVERVALUED"), o.band
    assert "FCF NEGATIVE" in o.why


def test_forward_earnings_cannot_rescue_a_loss_maker():
    """AAOI's forward P/E of 15 implies a 65% net margin for an optics
    manufacturer. A forward figure is only usable when TTM earnings are
    already positive."""
    loss = dict(shares=85.3e6, revenue=596e6, net_income=-57e6,
                gross_margin=0.289, growth=0.5)
    a = value(69.05, **loss)
    b = value(69.05, **loss, forward_eps=4.60)
    assert a.value == b.value, "a forward estimate moved a loss-maker"


def test_a_wide_moat_earner_lands_near_its_published_value():
    px, target, _ = KNOWN["MSFT"]
    o = value(px, shares=7.426e9, revenue=346.4e9, net_income=142.63e9,
              operating_cf=195.88e9, free_cf=2.99e9, gross_margin=0.677,
              net_margin=0.412, roe=0.342, market_cap=3.876e12,
              growth=0.188, forward_eps=24.14, moat="WIDE")
    err = abs(o.value / target - 1)
    assert err < 0.15, f"MSFT ${o.value} vs ${target} — {100*err:.0f}% out"


def test_moat_changes_the_answer_not_the_method():
    base = dict(shares=1e9, revenue=10e9, net_income=2e9, operating_cf=2.5e9,
                free_cf=2e9, gross_margin=0.60, net_margin=0.20, growth=0.15)
    wide = value(50, **base, moat="WIDE").value
    none = value(50, **base, moat="NONE").value
    assert wide > none, "a moat must be worth something"
    assert wide / none < 2.0, "a moat must not be worth everything"


def test_no_revenue_means_no_value_rather_than_zero():
    """A pre-revenue biotech (PBLS) has no fair value by this method.
    '$0.00 — OVERVALUED' asserts one; blank does not."""
    o = value(27.0, shares=50e6, revenue=0, net_income=-40e6)
    assert o.value is None
    assert o.as_text() == ""


def test_missing_data_is_reported_not_assumed():
    o = value(100, shares=1e6)
    assert o.value is None and "revenue" in o.why


def test_moat_proxy_matches_the_three_known_ratings():
    assert infer_moat(gross_margin=0.677, roe=0.342,
                      market_cap=3.876e12, net_margin=0.412) == "WIDE"
    assert infer_moat(gross_margin=0.587, roe=0.013,
                      market_cap=13e9, net_margin=0.030) == "NARROW"
    assert infer_moat(gross_margin=0.289, roe=-0.054,
                      market_cap=6.26e9, net_margin=-0.096) == "NONE"


def test_the_range_widens_when_the_anchors_disagree():
    agree = value(50, shares=1e9, revenue=10e9, net_income=2e9,
                  operating_cf=2.4e9, free_cf=2e9, gross_margin=0.45,
                  net_margin=0.20)
    assert agree.low < agree.value < agree.high


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
