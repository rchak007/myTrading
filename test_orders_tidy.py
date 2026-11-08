#!/usr/bin/env python3
"""
test_orders_tidy.py
===================
Sorting the Orders tab. Pure ordering only — no spreadsheet.
"""
import sys
sys.path.insert(0, "/home/chakravarti/github/myTrading")
import orders_tidy as t        # noqa: E402


def R(rid, status="", date="2026-11-01", ticker="AAA"):
    return {"Row_ID": rid, "Status": status, "Date": date, "Ticker": ticker}


def ids(rows):
    return [r["Row_ID"] for r in rows]


def test_live_rows_come_before_finished_ones():
    out = t.sort_rows([R("a", "FILLED"), R("b", "ARMED"),
                       R("c", "VOID"), R("d", "")])
    assert ids(out)[:2] == ["b", "d"], ids(out)
    assert set(ids(out)[2:]) == {"a", "c"}


def test_every_live_status_is_treated_as_live():
    for s in ("", "ARMED", "WAITING", "TRIGGERED", "SUBMITTED"):
        assert t.is_live(s), s
    for s in ("FILLED", "BLOCKED", "EXPIRED", "VOID"):
        assert not t.is_live(s), s


def test_status_case_and_spacing_do_not_matter():
    assert t.is_live("  armed ") and not t.is_live(" void ")


def test_newest_first_within_a_group():
    out = t.sort_rows([R("old", "ARMED", "2026-01-05"),
                       R("new", "ARMED", "2026-11-08"),
                       R("mid", "ARMED", "2026-06-30")])
    assert ids(out) == ["new", "mid", "old"]


def test_an_undated_row_sorts_last_not_first():
    """A missing date is not fresh news. Putting it on top would reward the
    mistake."""
    out = t.sort_rows([R("none", "ARMED", ""), R("dated", "ARMED",
                                                 "2026-01-01")])
    assert ids(out) == ["dated", "none"]


def test_same_day_rows_keep_the_order_they_were_typed():
    out = t.sort_rows([R("first", "ARMED"), R("second", "ARMED"),
                       R("third", "ARMED")])
    assert ids(out) == ["first", "second", "third"]


def test_spacer_rows_are_dropped():
    out = t.sort_rows([R("a", "ARMED"),
                       {"Row_ID": "", "Ticker": "", "Status": "",
                        "Date": ""},
                       R("b", "FILLED")])
    assert ids(out) == ["a", "b"]


def test_a_row_with_a_ticker_but_no_id_is_kept():
    """Half-typed rows are real work in progress, not spacers."""
    out = t.sort_rows([{"Row_ID": "", "Ticker": "BE", "Status": "",
                        "Date": "2026-11-08"}])
    assert len(out) == 1


def test_an_already_sorted_tab_is_unchanged():
    rows = [R("a", "ARMED", "2026-11-08"), R("b", "ARMED", "2026-11-01"),
            R("c", "FILLED", "2026-11-07")]
    assert ids(t.sort_rows(rows)) == ids(rows)


def test_nothing_in_nothing_out():
    assert t.sort_rows([]) == []


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
