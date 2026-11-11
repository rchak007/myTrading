#!/usr/bin/env python3
"""
test_rest_dashboard.py
======================
REST-DASHBOARD renders the tickers the main Dashboard does not.
"""
import sys

import pandas as pd

sys.path.insert(0, "/home/chakravarti/github/myTrading")
import orders_sheet as o        # noqa: E402

POS = pd.DataFrame([dict(
    Ticker="NVDA", Acct="171", Qty=10, Avg_Cost=150, Market_Value=1800,
    Unrealized_PL=300, Has_Stop="N", Has_Trim="N", Has_Dip="N",
    Has_Breakout="N", Fenced="", Seed_Reserved="", Borrowed="", Owed="",
    Updated="x")], columns=o.POSITIONS_COLS)


def blocks(rows):
    """Ticker labels in column A, skipping the title row."""
    return [r[0] for r in rows[1:] if r and str(r[0]).strip()
            and str(r[0]) != o.REST_TAB]


def test_only_renders_exactly_those_tickers():
    rows, _ = o.build_dashboard(POS, only=["ARBE", "SERV"],
                                title=o.REST_TAB)
    assert blocks(rows) == ["ARBE", "SERV"], blocks(rows)


def test_the_held_position_is_not_rendered_when_only_is_given():
    """NVDA is in `positions` and must NOT appear — otherwise the two tabs
    would show the same name twice and both totals would be wrong."""
    rows, _ = o.build_dashboard(POS, only=["ARBE"], title=o.REST_TAB)
    assert "NVDA" not in blocks(rows)


def test_the_title_is_the_tabs_own_name():
    """Both tabs are built by the same renderer; a REST tab headed
    'DASHBOARD' would be read as the real one."""
    rows, _ = o.build_dashboard(POS, only=["ARBE"], title=o.REST_TAB)
    assert rows[0][0] == o.REST_TAB
    plain, _ = o.build_dashboard(POS)
    assert plain[0][0] == "DASHBOARD"


def test_beth_watchlist_does_not_leak_into_the_rest_tab():
    """`only` replaces the watchlist, it does not add to it — a Beth name
    belongs on the Dashboard and must not be duplicated here."""
    rows, _ = o.build_dashboard(POS, only=["ARBE"], title=o.REST_TAB)
    beth = o._beth_tickers(log=lambda *_: None)
    assert not (set(blocks(rows)) & beth), set(blocks(rows)) & beth


def test_all_tickers_reads_app_py_and_finds_them():
    t = o.all_tickers()
    assert len(t) > 100, len(t)
    for want in ("NVDA", "ARBE", "SERV", "LWLG", "POET", "XLE"):
        assert want in t, want


def test_every_row_is_the_dashboard_width():
    """A ragged row makes the Sheets API reject the whole batch."""
    rows, _ = o.build_dashboard(POS, only=["ARBE", "SERV"],
                                title=o.REST_TAB)
    for r in rows:
        assert len(r) == o.DASH_WIDTH, (len(r), r[:3])


def test_an_empty_only_list_is_not_the_same_as_none():
    """`only=[]` means 'render nothing'; `only=None` means 'normal'. If
    those collapsed, an empty REST list would silently rebuild the whole
    Dashboard into the wrong tab."""
    empty, _ = o.build_dashboard(POS, only=[], title=o.REST_TAB)
    assert blocks(empty) == []
    normal, _ = o.build_dashboard(POS)
    assert "NVDA" in blocks(normal)


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
