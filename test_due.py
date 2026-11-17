#!/usr/bin/env python3
"""
test_due.py
===========
Due dates, and the order the DUE tab puts them in.

The ordering IS the feature — "past due first and empty next and then rest"
— so most of this is about the sort, and about the parser refusing anything
it cannot read rather than guessing a day.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/chakravarti/github/myTrading")

QUIET = lambda *_: None    # noqa: E731


def _log():
    return Path(tempfile.mkdtemp()) / "notes.csv"


# ---------------------------------------------------------------- parsing

def test_iso_and_slash_and_relative_all_parse():
    import ticker_notes as tn
    from datetime import date, timedelta
    t = date.today()
    assert tn.parse_due("2026-12-01") == "2026-12-01"
    assert tn.parse_due("today") == t.isoformat()
    assert tn.parse_due("tomorrow") == (t + timedelta(days=1)).isoformat()
    assert tn.parse_due("30d") == (t + timedelta(days=30)).isoformat()
    assert tn.parse_due("4w") == (t + timedelta(days=28)).isoformat()


def test_a_bare_month_day_means_the_NEXT_one():
    # Typing 1/15 in December must mean next January, not eleven months ago.
    import ticker_notes as tn
    from datetime import date
    t = date.today()
    got = tn.parse_due(f"{t.month}/{t.day}")
    assert got == t.isoformat(), got
    past = date(t.year, 1, 1)
    if t > past:
        assert tn.parse_due("1/1") == date(t.year + 1, 1, 1).isoformat()


def test_nonsense_is_refused_not_guessed():
    import ticker_notes as tn
    for bad in ("soon", "abc", "2026-13-01", "next week"):
        try:
            tn.parse_due(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should not have parsed")


def test_clearing_words_all_mean_empty():
    import ticker_notes as tn
    for s in ("-", "none", "clear", "x", "off", ""):
        assert tn.parse_due(s) == "", s


# ---------------------------------------------------------------- the fold

def test_the_last_row_wins():
    import ticker_notes as tn
    p = _log()
    tn.set_due("NVDA", "2026-12-01", path=p, log=QUIET)
    tn.set_due("NVDA", "2026-12-25", path=p, log=QUIET)
    assert tn.due_dates(p) == {"NVDA": "2026-12-25"}


def test_clearing_removes_the_key_rather_than_emptying_it():
    # Callers must only ever see present/absent. An empty string left in the
    # map reads as a date everywhere someone forgets to check for it.
    import ticker_notes as tn
    p = _log()
    tn.set_due("NVDA", "2026-12-01", path=p, log=QUIET)
    tn.set_due("NVDA", "-", path=p, log=QUIET)
    assert tn.due_dates(p) == {}


def test_clearing_does_not_erase_the_history():
    import ticker_notes as tn
    p = _log()
    tn.set_due("NVDA", "2026-12-01", path=p, log=QUIET)
    tn.set_due("NVDA", "-", path=p, log=QUIET)
    rows = [r for r in tn.read_log(p) if r["Action"] == "DUE"]
    assert len(rows) == 2 and rows[0]["Text"] == "2026-12-01"


def test_due_does_not_disturb_notes_or_attended():
    # DUE shares the ledger with ADD/REPLACE/DELETE/ATTENDED, so the folds
    # must not see each other's rows.
    import ticker_notes as tn
    p = _log()
    tn.add("NVDA", "keep 20 shares", path=p, log=QUIET)
    tn.attend("NVDA", path=p, log=QUIET)
    tn.set_due("NVDA", "2026-12-01", path=p, log=QUIET)
    assert tn.due_dates(p) == {"NVDA": "2026-12-01"}
    assert "keep 20 shares" in tn.render("NVDA", path=p)
    assert "NVDA" in tn.attended(p)


# ---------------------------------------------------------------- the sort

def test_the_three_buckets_are_in_his_order():
    import orders_sheet as os_
    k = os_._due_sort_key
    assert k(-5)[0] == 0, "past due leads"
    assert k(None)[0] == 1, "undated sits in the middle"
    assert k(3)[0] == 2, "future last"
    assert k(0)[0] == 2, "due today is not yet past due"


def test_most_overdue_leads_and_soonest_future_leads():
    import orders_sheet as os_
    k = os_._due_sort_key
    assert k(-30) < k(-2)
    assert k(1) < k(60)


def test_rows_come_out_past_due_then_empty_then_rest():
    import orders_sheet as os_
    import ticker_notes as tn
    from datetime import date, timedelta
    p = _log()
    t = date.today()
    tn.set_due("OLD", (t - timedelta(days=30)).isoformat(), path=p, log=QUIET)
    tn.set_due("LATE", (t - timedelta(days=1)).isoformat(), path=p, log=QUIET)
    tn.set_due("SOON", (t + timedelta(days=2)).isoformat(), path=p, log=QUIET)
    tn.set_due("FAR", (t + timedelta(days=90)).isoformat(), path=p, log=QUIET)

    rows, marks = os_.due_rows(
        {"OLD": "DASHBOARD", "LATE": "CRYPTO", "SOON": "DASHBOARD",
         "FAR": "REST-DASHBOARD", "NONE1": "CRYPTO", "NONE2": "DASHBOARD"},
        set(), log=QUIET, notes_path=p)

    order = [r[0] for r in rows if r[0] and r[0] not in ("DUE", "Ticker")]
    assert order == ["OLD", "LATE", "NONE1", "NONE2", "SOON", "FAR"], order
    assert len(marks["overdue"]) == 2
    assert len(marks["undated"]) == 2


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
