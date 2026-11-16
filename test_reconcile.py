#!/usr/bin/env python3
"""
test_reconcile.py
=================
What became of a submitted order.

The tests that matter here are the ones about RESTRAINT. Writing FILLED is
easy; the damage comes from writing it when the order has not filled, because
a chained row then arms and spends money on the strength of it. So most of
what follows checks that a row is left ALONE — still working, missing from
the history window, already finished.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/chakravarti/github/myTrading")

QUIET = lambda *_: None    # noqa: E731


def _fresh():
    """A reconciler pointed at an empty ledger in a throwaway directory."""
    tmp = Path(tempfile.mkdtemp())
    import order_exec_config as cfg
    cfg.LEDGER = tmp / "ledger.csv"
    cfg.STATE_DIR = tmp
    cfg.AUDIT = tmp / "audit.log"
    import order_engine as eng
    import order_reconcile as rec
    return eng, rec


def _submitted(eng, row_id, oid, **kw):
    eng.append_ledger(dict(row_id=row_id, state="SUBMITTED",
                           schwab_order_id=oid, acct="922", **kw))


def _run(rec, orders):
    import stocks_orders
    stocks_orders.fetch_schwab_orders = lambda *a, **k: orders
    return rec.reconcile(client_wrapper=None, ws=None, log=QUIET)


def test_a_filled_order_becomes_FILLED():
    eng, rec = _fresh()
    _submitted(eng, "R1", "111", ticker="QQQ", side="SELL", qty="1")
    out = _run(rec, [{"orderId": 111, "status": "FILLED", "filledQuantity": 1}])
    assert [u["state"] for u in out] == ["FILLED"], out
    assert eng.latest_states(eng.read_ledger())["R1"]["state"] == "FILLED"


def test_a_working_order_is_left_alone():
    eng, rec = _fresh()
    _submitted(eng, "R1", "111")
    assert _run(rec, [{"orderId": 111, "status": "WORKING"}]) == []
    assert eng.latest_states(eng.read_ledger())["R1"]["state"] == "SUBMITTED"


def test_an_unknown_status_is_left_alone():
    # The map is an allowlist. A status nobody has seen must not be guessed
    # at — "not in TERMINAL" has to mean "still live", never "probably done".
    eng, rec = _fresh()
    _submitted(eng, "R1", "111")
    assert _run(rec, [{"orderId": 111, "status": "SOME_NEW_STATUS"}]) == []
    assert eng.latest_states(eng.read_ledger())["R1"]["state"] == "SUBMITTED"


def test_an_order_missing_from_history_is_left_alone():
    # Absence of evidence. The lookback window is finite, so "not found" is a
    # statement about the window, not about the order.
    eng, rec = _fresh()
    _submitted(eng, "R1", "111")
    assert _run(rec, [{"orderId": 999, "status": "FILLED"}]) == []
    assert eng.latest_states(eng.read_ledger())["R1"]["state"] == "SUBMITTED"


def test_a_row_that_never_reached_schwab_is_not_touched():
    # No order id means no submission to ask about. orphaned_attempts already
    # reports these and they must not be swept up here.
    eng, rec = _fresh()
    eng.append_ledger(dict(row_id="R1", state="SUBMITTED", submit_attempted="1"))
    assert rec.pending(eng.read_ledger()) == {}
    assert _run(rec, [{"orderId": 111, "status": "FILLED"}]) == []


def test_an_already_finished_row_is_not_rechecked():
    eng, rec = _fresh()
    _submitted(eng, "R1", "111")
    eng.append_ledger(dict(row_id="R1", state="FILLED", schwab_order_id="111"))
    assert rec.pending(eng.read_ledger()) == {}


def test_cancelled_and_rejected_and_expired_are_terminal():
    for status, want in (("CANCELED", "CANCELLED"), ("CANCELLED", "CANCELLED"),
                         ("REJECTED", "REJECTED"), ("EXPIRED", "EXPIRED"),
                         ("REPLACED", "CANCELLED")):
        eng, rec = _fresh()
        _submitted(eng, "R1", "111")
        out = _run(rec, [{"orderId": 111, "status": status}])
        assert [u["state"] for u in out] == [want], (status, out)


def test_a_child_order_is_found():
    # A bracket's fill lives on the child, which carries its own id.
    eng, rec = _fresh()
    _submitted(eng, "R1", "222")
    out = _run(rec, [{"orderId": 111, "status": "WORKING",
                      "childOrderStrategies": [
                          {"orderId": 222, "status": "FILLED",
                           "filledQuantity": 5}]}])
    assert [u["state"] for u in out] == ["FILLED"], out
    assert out[0]["filled_qty"] == 5


def test_the_average_price_is_weighted_not_meaned():
    # 0.4 @ 660 and 0.6 @ 670 is 666, not 665. A plain mean misreports every
    # order that filled in uneven pieces, which is most of them.
    _, rec = _fresh()
    qty, px = rec.fill_details({"filledQuantity": 1, "orderActivityCollection": [
        {"executionLegs": [{"quantity": 0.4, "price": 660.0},
                           {"quantity": 0.6, "price": 670.0}]}]})
    assert qty == 1 and abs(px - 666.0) < 1e-9, (qty, px)


def test_executions_are_trusted_when_filledQuantity_is_absent():
    _, rec = _fresh()
    qty, px = rec.fill_details({"orderActivityCollection": [
        {"executionLegs": [{"quantity": 3, "price": 10.0}]}]})
    assert qty == 3 and px == 10.0, (qty, px)


def test_a_partial_fill_on_a_cancelled_order_is_recorded():
    # It ended CANCELLED, but shares changed hands and the quantity that
    # matters downstream is what executed.
    eng, rec = _fresh()
    _submitted(eng, "R1", "111")
    out = _run(rec, [{"orderId": 111, "status": "CANCELED", "filledQuantity": 3,
                      "orderActivityCollection": [
                          {"executionLegs": [{"quantity": 3, "price": 10.0}]}]}])
    assert out[0]["state"] == "CANCELLED" and out[0]["filled_qty"] == 3
    assert rec.PARTIAL_NOTE in out[0]["note"], out[0]["note"]


def test_no_submitted_orders_is_not_an_error():
    _, rec = _fresh()
    assert _run(rec, [{"orderId": 111, "status": "FILLED"}]) == []


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
