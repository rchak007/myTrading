#!/usr/bin/env python3
"""
test_parked.py
==============
Reserved money sitting in a stock instead of in cash.

The parcel is a fiction the broker does not share — Schwab sells SHARES,
not parcels — so the tests that matter are the ones about what happens
when the fiction and reality disagree.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/chakravarti/github/myTrading")
import parked as pk        # noqa: E402

QUIET = lambda *_: None    # noqa: E731


def _tmp():
    return Path(tempfile.mkdtemp()) / "parked.csv"


def test_park_then_read_back():
    p = _tmp()
    pk.park("431", "BE", "SPY", 2, cash=1000.0, path=p, log=QUIET)
    assert pk.outstanding("431", "BE", "SPY", p) == 2
    assert pk.for_owner("431", "BE", p) == "2 SPY"
    assert pk.for_vehicle("431", "SPY", p) == "2 BE"


def test_several_owners_in_one_vehicle():
    """BE and MU both park into SPY. A single number could not say how many
    shares to sell for which owner."""
    p = _tmp()
    pk.park("431", "BE", "SPY", 2, path=p, log=QUIET)
    pk.park("431", "MU", "SPY", 3, path=p, log=QUIET)
    assert pk.for_vehicle("431", "SPY", p) == "2 BE · 3 MU"
    assert pk.owed_qty("431", "SPY", p) == 5


def test_one_owner_across_several_vehicles():
    p = _tmp()
    pk.park("171", "BE", "SPY", 2, path=p, log=QUIET)
    pk.park("171", "BE", "QQQ", 3, path=p, log=QUIET)
    assert pk.for_owner("171", "BE", p) == "3 QQQ · 2 SPY"


def test_free_qty_is_what_you_may_actually_sell():
    """His question, 2026-11-08: 'what if i put sell order on SPY on its own
    and it encroaches into BE and MU?' — this is the number that answers
    it."""
    p = _tmp()
    pk.park("431", "BE", "SPY", 2, path=p, log=QUIET)
    pk.park("431", "MU", "SPY", 3, path=p, log=QUIET)
    assert pk.free_qty("431", "SPY", 12, p) == 7
    assert pk.free_qty("431", "SPY", 5, p) == 0


def test_a_sale_through_the_parcels_is_reported():
    """Prevention cannot be complete — a manual order at Schwab or a stop on
    the whole position will get through. Detection has to cover the rest."""
    p = _tmp()
    pk.park("431", "BE", "SPY", 2, path=p, log=QUIET)
    pk.park("431", "MU", "SPY", 3, path=p, log=QUIET)
    assert pk.breaches({("431", "SPY"): 5}, p) == []
    out = pk.breaches({("431", "SPY"): 1}, p)
    assert len(out) == 1 and "short 4" in out[0], out


def test_unpark_refuses_to_overdraw():
    """Unparking more than is parked is a typo, not a decision — it would
    leave a negative parcel that quietly frees shares nobody owns."""
    p = _tmp()
    pk.park("431", "BE", "SPY", 2, path=p, log=QUIET)
    try:
        pk.unpark("431", "BE", "SPY", 3, path=p, log=QUIET)
    except ValueError as e:
        assert "only 2" in str(e), e
    else:
        raise AssertionError("overdrew the parcel")


def test_unpark_to_zero_disappears():
    """'No longer parked' and 'parked nothing' are the same state and must
    not read differently on the sheet."""
    p = _tmp()
    pk.park("431", "BE", "SPY", 2, path=p, log=QUIET)
    pk.unpark("431", "BE", "SPY", 2, path=p, log=QUIET)
    assert pk.fold(p) == {}
    assert pk.for_owner("431", "BE", p) == ""


def test_a_ticker_cannot_be_parked_in_itself():
    """It would make free_qty subtract a holding from itself, so the
    position would look fully spoken for forever."""
    p = _tmp()
    try:
        pk.park("431", "SPY", "SPY", 2, path=p, log=QUIET)
    except ValueError as e:
        assert "itself" in str(e)
    else:
        raise AssertionError("parked a ticker in itself")


def test_negative_and_zero_are_refused():
    p = _tmp()
    for bad in (0, -1):
        for fn in (pk.park, pk.unpark):
            try:
                fn("431", "BE", "SPY", bad, path=p, log=QUIET)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{fn.__name__} accepted {bad}")


def test_the_log_is_append_only():
    """A delete writes a row, it does not erase one — so August stays
    answerable."""
    p = _tmp()
    pk.park("431", "BE", "SPY", 2, path=p, log=QUIET)
    pk.unpark("431", "BE", "SPY", 2, path=p, log=QUIET)
    rows = pk.read_log(p)
    assert len(rows) == 2
    assert [r["Event"] for r in rows] == ["PARK", "UNPARK"]


def test_no_file_is_empty_not_an_error():
    p = Path(tempfile.mkdtemp()) / "nope.csv"
    assert pk.fold(p) == {} and pk.read_log(p) == []
    assert pk.render(p) == ["nothing parked"]
    assert pk.breaches({("431", "SPY"): 0}, p) == []


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
