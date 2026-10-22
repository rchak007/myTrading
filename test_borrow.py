#!/usr/bin/env python3
"""
test_borrow.py
==============
The borrow/owe ledger. Four events, two of which must never move money.

The thing worth testing here is not the arithmetic — it is the SEPARATION.
BORROW changes a reserve; OWE does not, and a bug that let OWE touch
Reserved_Cash would show up as free cash that does not exist. That is the
failure this file exists to catch.
"""
import tempfile
from pathlib import Path

import cash_reserve as cr

QUIET = lambda *a, **k: None


def _fresh():
    d = Path(tempfile.mkdtemp())
    return d / "ledger.csv", d / "config.csv"


def test_borrow_moves_cash_and_records_the_debt():
    L, C = _fresh()
    cr.seed("171", "MSTR", 2000, ledger_path=L, config_path=C, log=QUIET)
    cr.borrow("171", "MSTR", 1000, ledger_path=L, log=QUIET)

    assert cr.balance_of("171", "MSTR", ledger_path=L) == 1000.0
    assert cr.outstanding("171", "MSTR", "Borrowed", L) == 1000.0


def test_owe_never_touches_cash():
    """The whole point of the memo events."""
    L, C = _fresh()
    cr.seed("171", "ALAB", 500, ledger_path=L, config_path=C, log=QUIET)
    before = cr.balance_of("171", "ALAB", ledger_path=L)

    cr.owe("171", "ALAB", 1000, ledger_path=L, log=QUIET)

    assert cr.balance_of("171", "ALAB", ledger_path=L) == before
    assert cr.outstanding("171", "ALAB", "Owed", L) == 1000.0


def test_full_round_trip_returns_to_zero():
    L, C = _fresh()
    cr.seed("171", "MSTR", 2000, ledger_path=L, config_path=C, log=QUIET)
    cr.borrow("171", "MSTR", 1000, ledger_path=L, log=QUIET)
    cr.owe("171", "ALAB", 1000, ledger_path=L, log=QUIET)
    cr.return_borrow("171", "MSTR", 1000, ledger_path=L, log=QUIET)
    cr.paid_back("171", "ALAB", 1000, ledger_path=L, log=QUIET)

    assert cr.balance_of("171", "MSTR", ledger_path=L) == 2000.0
    assert cr.outstanding("171", "MSTR", "Borrowed", L) == 0.0
    assert cr.outstanding("171", "ALAB", "Owed", L) == 0.0


def test_partial_return_leaves_the_remainder():
    L, C = _fresh()
    cr.seed("171", "MSTR", 2000, ledger_path=L, config_path=C, log=QUIET)
    cr.borrow("171", "MSTR", 1000, ledger_path=L, log=QUIET)
    cr.return_borrow("171", "MSTR", 400, ledger_path=L, log=QUIET)

    assert cr.balance_of("171", "MSTR", ledger_path=L) == 1400.0
    assert cr.outstanding("171", "MSTR", "Borrowed", L) == 600.0


def test_every_overdraw_is_refused():
    """No force flag anywhere. Borrowing more than exists is a typo, and the
    lender is supposed to get it back — unlike withdraw, which legitimately
    releases a reserve you no longer want."""
    L, C = _fresh()
    cr.seed("171", "MSTR", 1000, ledger_path=L, config_path=C, log=QUIET)
    cr.borrow("171", "MSTR", 400, ledger_path=L, log=QUIET)
    cr.owe("171", "ALAB", 400, ledger_path=L, log=QUIET)

    for fn, args in ((cr.borrow, ("171", "MSTR", 5000)),
                     (cr.return_borrow, ("171", "MSTR", 999)),
                     (cr.paid_back, ("171", "ALAB", 999)),
                     (cr.borrow, ("171", "MSTR", 0)),
                     (cr.owe, ("171", "ALAB", -5))):
        try:
            fn(*args, ledger_path=L, log=QUIET)
        except ValueError:
            continue
        raise AssertionError(f"{fn.__name__}{args} was NOT refused")


def test_sheet_columns_blank_when_zero_and_total_rolls_up():
    import pandas as pd
    import orders_sheet as o

    raw = pd.DataFrame([
        {"Ticker": "MSTR", "Acct": "171", "Qty": 10, "Avg_Cost": 300,
         "Market_Value": 3000, "Unrealized_PL": 0},
        {"Ticker": "MSTR", "Acct": "922", "Qty": 5, "Avg_Cost": 300,
         "Market_Value": 1500, "Unrealized_PL": 0},
        {"Ticker": "ALAB", "Acct": "171", "Qty": 4, "Avg_Cost": 250,
         "Market_Value": 1000, "Unrealized_PL": 0},
    ])
    bo = {("171", "MSTR"): (1000.0, 0.0), ("922", "MSTR"): (250.0, 0.0),
          ("171", "ALAB"): (0.0, 1000.0)}
    df = o.build_positions_table(raw, borrow_owe=bo, log=QUIET)

    total = df[(df.Ticker == "MSTR") & (df.Acct == "TOTAL")].iloc[0]
    assert total["Borrowed"] == 1250.0, "TOTAL must roll borrowed up"
    # Blank, not 0.00 — a column of zeroes hides the rows that matter.
    assert df[df.Ticker == "ALAB"]["Borrowed"].iloc[0] == ""
    assert "Borrowed" in o.POS_HDR and "Owed" in o.POS_HDR


def test_every_verb_is_documented():
    """A new verb with no VERB_DOCS entry must be visible, not silent."""
    import remote_ops as r
    known = set(r.VERBS) | set(r.PY_VERBS)
    documented = {v for v, *_ in r.VERB_DOCS}
    assert not (known - documented), f"undocumented: {sorted(known - documented)}"
    assert not (documented - known), f"stale docs: {sorted(documented - known)}"


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
    sys.exit(0)
