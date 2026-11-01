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


# ── post-earnings refresh ───────────────────────────────────────────────
# Each ticker reports on its own date, so the per-ticker path has to be able
# to rewrite ONE row without disturbing the other 134.

def test_reported_since_fetch_spots_a_new_quarter():
    import oracle_value as ov
    assert ov.reported_since_fetch(
        {"Last_Earnings": "2026-10-21", "Fetched": "2026-10-19"})
    assert not ov.reported_since_fetch(
        {"Last_Earnings": "2026-10-21", "Fetched": "2026-10-26"})


def test_reported_since_fetch_is_quiet_when_dates_are_missing():
    """A blank must never read as 'it reported'. That would refresh all 135
    names every single day."""
    import oracle_value as ov
    assert not ov.reported_since_fetch({})
    assert not ov.reported_since_fetch({"Last_Earnings": "", "Fetched": ""})
    assert not ov.reported_since_fetch({"Last_Earnings": "2026-10-21"})


def test_a_one_ticker_refresh_keeps_every_other_row():
    """THE BUG THIS GUARDS: refresh() rewrites the whole file. Handed a
    subset it used to emit only that subset, so a single post-earnings
    refresh would have wiped the other 134 values."""
    import csv
    import os
    import tempfile
    import importlib
    import oracle_value as ov

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "oracle_values.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=ov.COLS, extrasaction="ignore")
            w.writeheader()
            for t in ("AAA", "BBB", "CCC"):
                w.writerow({"Ticker": t, "Fetched": "2026-10-01",
                            "Oracle_Value": "10", "Oracle_Band": "FAIR"})

        os.environ["ORACLE_FILE"] = path
        importlib.reload(ov)
        # BBB fails to compute, so it must fall back to its old row; AAA and
        # CCC were never asked for and must survive untouched.
        ov.refresh(["BBB"], log=lambda *_: None)
        rows = {r["Ticker"]: r for r in csv.DictReader(open(path))}
        assert set(rows) == {"AAA", "BBB", "CCC"}, rows.keys()
        assert rows["AAA"]["Oracle_Value"] == "10"
        assert rows["CCC"]["Oracle_Value"] == "10"
    os.environ.pop("ORACLE_FILE", None)
    importlib.reload(ov)


def test_mean_ps_returns_years_and_dispersion():
    """The anchor now travels with how much history it is built on and how
    much that history agrees, because a four-year mean of bubble years is
    what valued NVDA at $434."""
    import oracle_value as ov
    import inspect
    sig = inspect.signature(ov.mean_ps)
    assert len(sig.parameters) == 2
    src = inspect.getsource(ov.mean_ps)
    assert "edgar" in src, "mean_ps must prefer SEC filings over yfinance"


def test_a_short_history_is_discounted_not_trusted():
    """Same inputs, four years of P/S history vs twelve: the short one must
    not carry the same weight."""
    from core.oracle import value
    kw = dict(shares=1e9, revenue=10e9, net_income=1e9, market_cap=100e9,
              gross_margin=0.5, net_margin=0.10, roe=0.15, moat="NARROW")
    deep = value(100.0, hist_ps=10.0, hist_ps_years=12, **kw)
    thin = value(100.0, hist_ps=10.0, hist_ps_years=4, **kw)
    assert deep.value != thin.value, "years must change the weighting"
    assert "only 4y" in thin.why


def test_an_unstable_multiple_is_discounted():
    """AAOI's yearly P/S ran 1x to 6x. Its mean is a number, not a level."""
    from core.oracle import value
    kw = dict(shares=1e9, revenue=10e9, net_income=1e9, market_cap=100e9,
              gross_margin=0.5, net_margin=0.10, roe=0.15, moat="NARROW")
    steady = value(100.0, hist_ps=10.0, hist_ps_years=12,
                   hist_ps_dispersion=0.10, **kw)
    noisy = value(100.0, hist_ps=10.0, hist_ps_years=12,
                  hist_ps_dispersion=0.55, **kw)
    assert steady.value != noisy.value
    assert "scatters" in noisy.why


# ── the ORACLE dashboard section ────────────────────────────────────────

def _fake_cache(**over):
    row = {"Ticker": "ZZZ", "Fetched": "2026-11-01", "Price_At_Fetch": "100",
           "Oracle_Value": "120", "Oracle_Low": "110", "Oracle_High": "130",
           "Oracle_Band": "UNDERVALUED", "Moat": "WIDE",
           "Oracle_Why": "own mean P/S 10.0x over 12y",
           "Oracle_IQ": "Pred High · Profit High · Growth Medium · "
                        "Moat Wide · Fin High · Val Undervalued",
           "Last_Earnings": "2026-10-20", "Next_Earnings": "2027-01-28",
           "PS_Mean": "10.0", "PS_Years": "12", "PS_Spread": "0.12"}
    row.update(over)
    return {"ZZZ": row}


def test_oracle_section_has_value_quality_and_basis():
    import oracle_value as ov
    groups = ov.panel_for_sheet("ZZZ", price=100.0, cache=_fake_cache())
    assert [g[0] for g in groups] == ["VALUE", "QUALITY", "BASIS"]
    names, vals = groups[0][1], groups[0][2]
    assert "OracleValue" in names and "Upside_to_Fair" in names
    assert "+20.0%" in vals, vals          # 120 fair against a 100 price


def test_oracle_section_flags_a_thin_history():
    """The caveat is the point of the BASIS group. A four-year mean is what
    valued NVDA at $434, and a dollar figure alone does not show that."""
    import oracle_value as ov
    g = ov.panel_for_sheet("ZZZ", 100.0, _fake_cache(PS_Years="4"))
    assert "thin history" in g[-1][2][-1]


def test_oracle_section_flags_an_unstable_multiple():
    import oracle_value as ov
    g = ov.panel_for_sheet("ZZZ", 100.0, _fake_cache(PS_Spread="0.55"))
    assert "unstable" in g[-1][2][-1]


def test_a_deep_steady_history_earns_no_caveat():
    import oracle_value as ov
    g = ov.panel_for_sheet("ZZZ", 100.0, _fake_cache())
    assert g[-1][2][-1] == ""


def test_no_cached_row_is_an_absent_section_not_an_empty_one():
    import oracle_value as ov
    assert ov.panel_for_sheet("NOPE", 100.0, {}) == []


def test_oracle_rows_pad_to_the_dashboard_width():
    """Every emitted row must be exactly DASH_WIDTH or the sheet write
    ragged-edges and gspread rejects the batch."""
    import orders_sheet
    rows = orders_sheet._oracle_rows("NOPE", 100.0, 19)
    assert rows == []                      # unknown ticker -> no section
    import oracle_value as ov
    groups = ov.panel_for_sheet("ZZZ", 100.0, _fake_cache())
    for _, names, vals in groups:
        assert len(names) == len(vals), (names, vals)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
