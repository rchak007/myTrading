#!/usr/bin/env python3
"""
test_fib.py
===========
fib.py anchors and verdicts, and the Dashboard's FIB section — on synthetic
prices, so it runs offline and the expected anchors are known exactly.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/chakravarti/github/myTrading")
import fib                       # noqa: E402
import orders_sheet as o         # noqa: E402


def path(*legs):
    """Daily bars along straight legs: path((0,100),(100,150),...) with the
    first number the bar index and the second the close there."""
    xs = np.concatenate([np.linspace(a[1], b[1], b[0] - a[0], endpoint=False)
                         for a, b in zip(legs, legs[1:])] + [[legs[-1][1]]])
    idx = pd.bdate_range("2022-01-03", periods=len(xs))
    return pd.DataFrame({"High": xs * 1.01, "Low": xs * 0.99, "Close": xs},
                        index=idx)


# base 100 → 150, a 40% correction to 90 (T0), → 200, a 35% correction to 130
# (A), → 400 (B), then a pullback to 220 (C).
RUN = [(0, 100), (100, 150), (160, 90), (300, 200), (360, 130), (480, 400)]
FULL = path(*RUN, (600, 220))


def test_anchors_are_the_last_big_unbroken_base():
    f = fib.compute_fib(FULL)
    assert f["Fib_A"] == round(130 * 0.99, 2), f["Fib_A"]
    assert f["Fib_B"] == round(400 * 1.01, 2), f["Fib_B"]
    assert f["Fib_C"] == round(220 * 0.99, 2), f["Fib_C"]
    assert f["Fib_Ext_From"] == "C"


def test_levels_are_the_standard_retracement_and_extension():
    f = fib.compute_fib(FULL)
    A, B, C = f["Fib_A"], f["Fib_B"], f["Fib_C"]
    assert abs(f["Fib_618"] - (B - 0.618 * (B - A))) < 0.01
    assert abs(f["Fib_1000"] - A) < 0.01
    assert abs(f["Fib_E1618"] - (C + 1.618 * (B - A))) < 0.01


def test_an_undercut_base_is_dropped_for_the_next_one_down():
    """The BTC case: once price falls below A, that structure is broken and
    the next base below — still intact — becomes A."""
    f = fib.compute_fib(path(*RUN, (600, 110)))
    assert f["Fib_A"] == round(90 * 0.99, 2), f["Fib_A"]


def test_near_the_highs_trims_come_from_the_previous_swing():
    """No real pullback yet, so the extension is the PRIOR swing (T0 → the
    high A corrected from) launched from A."""
    f = fib.compute_fib(path(*RUN, (490, 390)))
    assert f["Fib_Ext_From"] == "prior swing", f["Fib_Ext_From"]
    A, T0, pk = 130 * 0.99, 90 * 0.99, 200 * 1.01
    assert abs(f["Fib_E1272"] - (A + 1.272 * (pk - T0))) < 0.05, f["Fib_E1272"]


def test_a_stock_that_only_fell_has_no_ladder():
    assert fib.compute_fib(path((0, 300), (400, 60))) == {}


def test_verdicts():
    f = fib.compute_fib(FULL)
    assert fib.classify(f, f["Fib_618"] - 1)[0] == "BUY"
    assert "50% deployed" in fib.classify(f, f["Fib_618"] - 1)[1]
    assert fib.classify(f, f["Fib_786"] - 1)[1].endswith("business check first")
    assert fib.classify(f, f["Fib_382"] + 1)[0] == "HOLD"
    assert fib.classify(f, f["Fib_E1618"] + 1)[0] == "TRIM"
    assert "25% of the non-core" in fib.classify(f, f["Fib_E1618"] + 1)[1]
    assert fib.classify(f, f["Fib_A"] - 1)[0] == "FAILED"
    assert fib.classify({}, 100) == ("", "")


POS = pd.DataFrame([dict(
    Ticker="NVDA", Acct="171", Qty=10, Avg_Cost=150, Market_Value=1800,
    Unrealized_PL=300, Has_Stop="N", Has_Trim="N", Has_Dip="N",
    Has_Breakout="N", Fenced="", Seed_Reserved="", Borrowed="", Owed="",
    Updated="x")], columns=o.POSITIONS_COLS)


def _sig(price):
    return pd.DataFrame([{"Ticker": "NVDA", "Current Price": price,
                          **fib.compute_fib(FULL)}])


def test_dashboard_section_is_painted_by_verdict():
    f = fib.compute_fib(FULL)
    for px, key in ((f["Fib_618"] - 1, "fib_buy"),
                    (f["Fib_E1272"] + 1, "fib_trim"),
                    (f["Fib_382"] + 1, "fib_hold")):
        rows, marks = o.build_dashboard(POS, signals_df=_sig(px),
                                        sections={"FIB"})
        assert len(marks[key]) == 1, (key, {k: v for k, v in marks.items()
                                            if k.startswith("fib")})
        lab = rows[marks[key][0] - 1]
        assert lab[1] == "FIB" and lab[2], lab[:4]
        for r in rows:
            assert len(r) == o.DASH_WIDTH, (len(r), r[:3])


def test_no_fib_columns_no_section():
    sig = pd.DataFrame([{"Ticker": "NVDA", "Current Price": 100}])
    rows, marks = o.build_dashboard(POS, signals_df=sig)
    assert not any(marks[k] for k in ("fib_buy", "fib_trim", "fib_hold"))
    assert not any(r[1] == "FIB" for r in rows)


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
