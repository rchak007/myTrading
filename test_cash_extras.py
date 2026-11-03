#!/usr/bin/env python3
"""
test_cash_extras.py
===================
The TOTAL line and seed breakdown under the Cash tab.

The thing worth guarding is that these rows stay OUT of the dataframe.
A TOTAL inside it gets added to its own components, which has already
happened here once: the upstream cash table carries a TOTAL whose
acct_key is "TAL", it matched nothing, counted as a seventh account, and
doubled Free_To_Deploy.
"""
import sys

import pandas as pd

sys.path.insert(0, "/home/chakravarti/github/myTrading")
import orders_sheet as os_        # noqa: E402

CASH = pd.DataFrame([
    dict(Acct="171", Nickname="PCRA", Cash=46941.99,
         Cash_In_Open_Orders=16467.0, Cash_After_Open_Orders=30474.99,
         Seed_Reserved=39673.29, Free_To_Deploy=-124.30, Updated="x"),
    dict(Acct="431", Nickname="IRA", Cash=4708.47,
         Cash_In_Open_Orders=1556.0, Cash_After_Open_Orders=3152.47,
         Seed_Reserved=5169.60, Free_To_Deploy=-1657.13, Updated="x"),
], columns=os_.CASH_COLS)

RES = {("431", "ALAB"): 2500.0, ("431", "MSTR"): 1800.0,
       ("431", "AAOI"): -330.40, ("171", "PLTR"): 816.0}


def _cells(rows):
    return [[str(c) for c in r] for r in rows]


def test_the_total_row_sums_the_money_columns():
    rows = os_.cash_extras(CASH, RES)
    total = next(r for r in rows if r and r[0] == os_.TOTAL)
    assert abs(float(total[2]) - (46941.99 + 4708.47)) < 0.01
    assert abs(float(total[6]) - (-124.30 + -1657.13)) < 0.01


def test_extras_never_enter_the_dataframe():
    """THE BUG THIS GUARDS. Several callers sum this frame; a TOTAL row
    inside it is added to its own components."""
    before = len(CASH)
    os_.cash_extras(CASH, RES)
    assert len(CASH) == before, "cash_extras mutated the frame"
    assert os_.TOTAL not in set(CASH["Acct"])


def test_the_breakdown_names_the_tickers_holding_the_reserve():
    """A negative Free_To_Deploy says an account is over-fenced; only this
    says what to unfence."""
    flat = " ".join(" ".join(x) for x in _cells(os_.cash_extras(CASH, RES)))
    assert "ALAB" in flat and "MSTR" in flat and "PLTR" in flat


def test_an_over_fenced_account_says_by_how_much():
    rows = os_.cash_extras(CASH, RES)
    sub = next(r for r in rows if r[0] == "431" and "reserved" in str(r[1]))
    # 2500 + 1800, with the breach floored out of the total
    assert abs(float(sub[5]) - 4300.0) < 0.01, sub
    assert "SHORT" in sub[6] and "1,147" in sub[6], sub[6]


def test_a_covered_account_says_so_rather_than_nothing():
    rows = os_.cash_extras(CASH, RES)
    sub = next(r for r in rows if r[0] == "171" and "reserved" in str(r[1]))
    assert "covered" in sub[6], sub[6]


def test_a_breach_is_shown_but_not_counted_as_reserved():
    """The negative IS the breach -- hiding it would be the worse bug. It
    just must not be spendable, so it stays out of the subtotal."""
    rows = os_.cash_extras(CASH, RES)
    aaoi = next(r for r in rows if r[1] == "AAOI")
    assert float(aaoi[5]) < 0 and "BREACH" in aaoi[6]
    sub = next(r for r in rows if r[0] == "431" and "reserved" in str(r[1]))
    assert float(sub[5]) == 4300.0, "a breach was netted into the reserve"


def test_every_row_fits_the_column_count():
    """A ragged row makes gspread reject the whole batch."""
    for r in os_.cash_extras(CASH, RES):
        assert len(r) == len(os_.CASH_COLS), r


def test_no_cash_means_no_extras():
    assert os_.cash_extras(pd.DataFrame(columns=os_.CASH_COLS), RES) == []


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
