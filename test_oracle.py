#!/usr/bin/env python3
"""
test_oracle.py
==============
OracleValue — our conservative fair value.

The thing worth testing is not the arithmetic, it is the DIRECTION and the
refusals. Grok's "StockOracle-style" estimate for AAOI was $80-110 and
"modestly undervalued" for a company burning $887M of free cash on $596M of
revenue. A model that gets the sign wrong is worse than no model.

NO PUBLISHED VALUE IS ASSERTED HERE. An earlier version of this file carried
a KNOWN dict of StockOracle's figures and checked our MSFT output landed
within 15% of theirs, which made their number the definition of correct.
Chakravarti, 2026-11-01: *"remember not that Oracle value calculated by
stock oracle is any bible"*. These tests assert PROPERTIES instead -- a cash
burner is never called cheap, a forward estimate cannot rescue a loss-maker,
a thin history is discounted, a broken input blanks rather than prints.
"""
from core.oracle import value


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


def test_a_wide_moat_compounder_is_valued_near_its_price():
    """Microsoft on its own numbers: a wide moat earning 34% on equity at a
    68% gross margin is, by construction, roughly fairly priced. The test is
    that the model lands in that neighbourhood -- not that it reproduces any
    particular published figure."""
    o = value(521.91, shares=7.426e9, revenue=346.4e9, net_income=142.63e9,
              operating_cf=195.88e9, free_cf=2.99e9, gross_margin=0.677,
              net_margin=0.412, roe=0.342, market_cap=3.876e12,
              growth=0.188, forward_eps=24.14, moat="WIDE")
    assert o.value is not None
    assert 0.6 < o.value / 521.91 < 1.7, f"${o.value} against a $522 price"
    assert o.band not in ("DEEP VALUE",), o.band


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


def test_the_moat_rating_is_ours_and_not_a_lookup():
    """There was a MOAT_OVERRIDE table of four ratings copied off
    StockOracle feeding straight into the valuation. It is gone, and nothing
    may reintroduce a hardcoded rating."""
    import oracle_value as ov
    import inspect
    assert not hasattr(ov, "MOAT_OVERRIDE")
    assert "moat.from_edgar" in inspect.getsource(ov.our_moat)


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
        # BBB raises (a transient fetch failure), so it keeps its old row.
        # AAA and CCC were never asked for and must survive untouched.
        def _boom(sym, info=None, price=None):
            raise RuntimeError("network")
        ov.compute_one = _boom
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


def test_a_wide_spread_is_reported_but_does_not_move_the_value():
    """DISPERSION DESCRIBES, IT DOES NOT DISCOUNT. On four years of history
    the rule looked decisive; on real history it inverts -- TSLA has the
    widest spread of the four published anchors (0.76) and the smallest
    error (6%), SITM the narrowest (0.30) and the largest (54%). It also
    fired on two thirds of the book."""
    from core.oracle import value
    kw = dict(shares=1e9, revenue=10e9, net_income=1e9, market_cap=100e9,
              gross_margin=0.5, net_margin=0.10, roe=0.15, moat="NARROW")
    steady = value(100.0, hist_ps=10.0, hist_ps_years=12,
                   hist_ps_dispersion=0.10, **kw)
    noisy = value(100.0, hist_ps=10.0, hist_ps_years=12,
                  hist_ps_dispersion=0.95, **kw)
    assert steady.value == noisy.value, "spread must not move the number"
    assert "ranged widely" in noisy.why, "but it must still be reported"


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
    # The POINT value headlines; the range is a separate column beside it.
    # "$110-$130" in the OracleValue cell reads as saying nothing.
    assert "$120.00" in vals, vals
    assert "Range" in names and "$110-$130".replace("-", "\u2013") in vals


def test_oracle_section_flags_a_thin_history():
    """The caveat is the point of the BASIS group. A four-year mean is what
    valued NVDA at $434, and a dollar figure alone does not show that."""
    import oracle_value as ov
    g = ov.panel_for_sheet("ZZZ", 100.0, _fake_cache(PS_Years="4"))
    assert "thin history" in g[-1][2][-1]


def test_oracle_section_only_flags_a_truly_extreme_spread():
    """0.55 is around the median once history is deep -- flagging it made
    the caveat fire on two thirds of the book."""
    import oracle_value as ov
    assert ov.panel_for_sheet("ZZZ", 100.0, _fake_cache(PS_Spread="0.55"))[-1][2][-1] == ""
    assert "ranged widely" in ov.panel_for_sheet(
        "ZZZ", 100.0, _fake_cache(PS_Spread="0.95"))[-1][2][-1]


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


def test_a_value_many_times_the_price_is_suppressed():
    """SNDK: spun out of Western Digital in Feb 2025, so 21 months of price
    history and no mean P/S — the dominant anchor absent. The earnings
    anchors then ran on a yfinance net income of $19.4B against $29.0B of
    revenue (spinoff accounting, not profit) and printed $13,712 against a
    $1,151 share price, with no flag at all. Blank is the honest output."""
    from core.oracle import value
    o = value(1150.89, shares=142.3e6, revenue=28.97e9,
              net_income=19.42e9, gross_margin=0.40, net_margin=0.67,
              roe=0.50, market_cap=163e9, forward_eps=288.0, moat="WIDE")
    assert o.value is None, f"published ${o.value} on a $1,151 stock"
    assert "share price" in o.why


def test_a_genuine_deep_value_call_still_survives():
    """The guard must catch broken arithmetic, not an unpopular opinion.
    2x the price is a real call; 12x is a bug."""
    from core.oracle import value
    o = value(100.0, shares=1e9, revenue=50e9, net_income=8e9,
              operating_cf=9e9, free_cf=7e9, gross_margin=0.55,
              net_margin=0.16, roe=0.25, market_cap=100e9, hist_ps=4.0,
              hist_ps_years=12, moat="WIDE")
    assert o.value is not None and o.value > 100.0


def test_the_range_is_capped_so_it_can_be_acted_on():
    """+/-45% ruled out nothing. Where the anchors disagree by more than the
    cap, the disagreement is stated instead of drawn."""
    from core.oracle import value
    o = value(400.0, shares=3.2e9, revenue=95e9, net_income=3e9,
              operating_cf=15e9, free_cf=-3e9, gross_margin=0.18,
              net_margin=0.03, roe=0.08, market_cap=1.3e12, hist_ps=16.0,
              hist_ps_years=16, forward_eps=2.5, moat="NARROW")
    assert o.value
    half = (o.high - o.low) / o.value / 2
    assert half <= 0.2501, f"range is +/-{half:.0%}"
    assert "low confidence" in o.why


def test_a_refusal_drops_the_row_instead_of_restoring_the_old_one():
    """THE BUG THIS GUARDS. compute_one() refuses to value a ticker whose
    answer is absurd. refresh() used to treat that refusal as a fetch
    failure and restore yesterday's row 'to be safe' -- putting the very
    numbers the guard had just rejected back on the sheet. RCAT came back
    at 8,288,439x the share price, still flagged DEEP VALUE, after a
    refresh that had declined it. A refusal a retry can undo is not a
    guard."""
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
            w.writerow({"Ticker": "JUNK", "Fetched": "2026-10-01",
                        "Price_At_Fetch": "1", "Oracle_Value": "8288439",
                        "Oracle_Band": "DEEP VALUE"})
            w.writerow({"Ticker": "KEEP", "Fetched": "2026-10-01",
                        "Oracle_Value": "50", "Oracle_Band": "FAIR"})

        os.environ["ORACLE_FILE"] = path
        importlib.reload(ov)
        from core.oracle import Oracle
        ov.compute_one = lambda s, i=None, price=None: Oracle(
            None, None, None, "", None, "no honest value — refused", None)
        ov.refresh(["JUNK"], log=lambda *_: None)
        rows = {r["Ticker"]: r for r in csv.DictReader(open(path))}

    os.environ.pop("ORACLE_FILE", None)
    importlib.reload(ov)
    # The ROW survives so PE, PEG and the margin series stay readable --
    # FINANCIALS reads this same cache, and dropping the row took TSM and
    # SNDK off the Dashboard entirely. But the VALUE must be blank, or the
    # figure the guard rejected is simply back.
    assert "JUNK" in rows, "a refused ticker lost its fundamentals too"
    assert rows["JUNK"]["Oracle_Value"] == "", rows["JUNK"]["Oracle_Value"]
    assert "8288439" not in str(rows["JUNK"]), "the rejected value returned"
    assert "share price" in rows["JUNK"]["Oracle_Why"] or \
           "refused" in rows["JUNK"]["Oracle_Why"]
    assert rows["KEEP"]["Oracle_Value"] == "50", "an untouched row was lost"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
