#!/usr/bin/env python3
"""
test_engine_resilience.py
=========================
The order engine must survive a bad row.

A CRDO row typed 2026-10-29 with Qty RESERVE raised ValueError out of
check_guards and killed main(). Every run from then until 2026-11-12 died
on the same row -- sixteen days with NO order for ANY ticker evaluated,
while the sheet header kept updating because a different job writes it.
"""
import sys

sys.path.insert(0, "/home/chakravarti/github/myTrading")
import order_exec_config as cfg        # noqa: E402
import order_engine as oe              # noqa: E402

cfg.ACCOUNT_ALLOWLIST = {"431"}
cfg.TICKER_ALLOWLIST = set()


def row(**over):
    n = {"Row_ID": "T-01", "_sheet_row": 9, "Acct": "431", "Ticker": "CRDO",
         "Side": "BUY", "Qty": "10", "Close_Is": "BELOW", "Date": "2026-10-29",
         "Limit_Price": "", "Trigger_Price": "140", "Expires_On": "",
         "Status": "", "Notes": "", "Trigger_ID": "", "After_Trigger_ID": ""}
    n.update(over)
    return n


def test_reserve_qty_does_not_raise():
    """THE EXACT CRASH. float('RESERVE') killed the run for 16 days."""
    assert oe.check_guards(row(Qty="RESERVE"), None, 0, 0.0, None) is None


def test_dollar_qty_does_not_raise():
    """$730 has the same shape -- unresolvable until there is a price."""
    assert oe.check_guards(row(Qty="$730"), None, 0, 0.0, None) is None


def test_an_unreadable_qty_is_blocked_not_crashed():
    """Garbage must produce a BLOCK with a readable reason, not a
    traceback."""
    p = oe.check_guards(row(Qty="banana"), None, 0, 0.0, None)
    assert p and "QTY_UNREADABLE" in p, p
    assert "banana" in p


def test_a_normal_quantity_still_works():
    assert oe.check_guards(row(Qty="10"), None, 0, 0.0, None) is None


def test_notional_caps_still_bite_when_qty_is_known():
    old = cfg.MAX_NOTIONAL_PER_ORDER
    try:
        cfg.MAX_NOTIONAL_PER_ORDER = 100.0
        p = oe.check_guards(row(Qty="10", Trigger_Price="140"),
                            None, 0, 0.0, None)
        assert p and "CAP_PER_ORDER" in p, p
    finally:
        cfg.MAX_NOTIONAL_PER_ORDER = old


def test_caps_are_skipped_not_misapplied_when_qty_is_unresolved():
    """A RESERVE row cannot be cap-checked before it is resolved. Skipping
    is correct; treating the qty as zero or one would let a large order
    through the cap it exists to enforce -- it is re-checked at 5b."""
    old = cfg.MAX_NOTIONAL_PER_ORDER
    try:
        cfg.MAX_NOTIONAL_PER_ORDER = 1.0
        assert oe.check_guards(row(Qty="RESERVE"), None, 0, 0.0,
                               None) is None
    finally:
        cfg.MAX_NOTIONAL_PER_ORDER = old


def test_the_loop_body_is_wrapped():
    """The structural fix: a row that blows up must be blocked, not fatal."""
    import inspect
    src = inspect.getsource(oe.main)
    i = src.index("for rec in rows:")
    after = src[i:]
    assert "try:" in after.split("\n", 6)[-1] or "try:" in after[:900], \
        "the per-row loop body is not wrapped"
    assert "ENGINE_ERROR" in after, "a crashed row does not report itself"


def test_rid_is_defined_before_the_try():
    """The handler reports against rid. A NameError inside an error handler
    is how a fix becomes a new bug."""
    import inspect
    lines = inspect.getsource(oe.main).splitlines()
    start = next(i for i, l in enumerate(lines)
                 if l.strip().startswith("for rec in rows:"))
    body = lines[start + 1:start + 40]
    rid = next(i for i, l in enumerate(body) if l.strip().startswith("rid ="))
    tr = next(i for i, l in enumerate(body) if l.strip() == "try:")
    assert rid < tr, f"rid at {rid}, try at {tr}"


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
