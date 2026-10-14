#!/usr/bin/env python3
"""
test_order_triggers.py
======================
Structure triggers (HH/HL/LL/LH) and order chaining. No network, no sheet.

    .venv/bin/python test_order_triggers.py

THE TWO FAILURES THESE GUARD AGAINST, both silent:

  FIRING ON A STALE PIVOT. "Buy the next HL" must mean new since the row was
  typed. Match on Last_Label alone and a row typed today fires instantly on a
  pivot from three weeks ago — it looks like the feature working.

  A CHAIN THAT CANNOT FIRE. A child whose parent VOIDed, or a loop, sits
  reading ARMED forever. Nothing errors; the order simply never happens, and
  you find out by noticing something that should have traded did not.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import order_intent as oc                                  # noqa: E402
from order_engine import triggered                         # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def _ok(fn) -> bool:
    """True if it validated, False if it raised IntentError."""
    try:
        fn()
        return True
    except oc.IntentError:
        return False


def fired(close_is, close, trigger=None, **kw):
    return triggered(close_is, close, trigger, **kw)[0]


def why(close_is, close, trigger=None, **kw):
    return triggered(close_is, close, trigger, **kw)[1]


print("\n── price triggers still behave exactly as before ──")
check("ABOVE fires strictly through", fired("ABOVE", 245.01, 245.0))
check("ABOVE does NOT fire exactly on the level", not fired("ABOVE", 245.0, 245.0))
check("BELOW fires strictly through", fired("BELOW", 244.99, 245.0))
check("BELOW does NOT fire exactly on the level", not fired("BELOW", 245.0, 245.0))

print("\n── a structure trigger needs a NEW pivot ──")
ARG = dict(side="BUY", label="HL", confirmed="2026-10-14", since="2026-10-10")
check("HL confirmed after the row fires", fired("HL", 6.10, **ARG))
# The headline bug: a row typed today, a pivot confirmed weeks ago.
check("an OLD HL does not fire",
      not fired("HL", 6.10, side="BUY", label="HL",
                confirmed="2026-09-20", since="2026-10-10"))
check("...and says it is waiting for the NEXT one",
      "NEXT" in why("HL", 6.10, side="BUY", label="HL",
                    confirmed="2026-09-20", since="2026-10-10"))
check("confirmed ON the row's own date does not fire (not newer)",
      not fired("HL", 6.10, side="BUY", label="HL",
                confirmed="2026-10-10", since="2026-10-10"))
check("the wrong label does not fire",
      not fired("HL", 6.10, side="BUY", label="LL",
                confirmed="2026-10-14", since="2026-10-10"))
check("...and names what it is waiting for",
      "waiting for HL" in why("HL", 6.10, side="BUY", label="LL",
                              confirmed="2026-10-14", since="2026-10-10"))

print("\n── missing data must WAIT, never fire ──")
check("no structure data at all", not fired("HL", 6.10, side="BUY"))
check("label but no confirmation date",
      not fired("HL", 6.10, side="BUY", label="HL", since="2026-10-10"))
check("no row Date to measure 'next' from",
      not fired("HL", 6.10, side="BUY", label="HL", confirmed="2026-10-14"))
check("...and says so rather than guessing",
      "no Date" in why("HL", 6.10, side="BUY", label="HL",
                       confirmed="2026-10-14"))

print("\n── Trigger_Price on a structure row is a BOUND ──")
# "Buy the next HL, but do not pay over 6.40."
check("BUY fires under the bound", fired("HL", 6.10, 6.40, **ARG))
check("BUY does NOT fire above the bound", not fired("HL", 6.55, 6.40, **ARG))
check("BUY does not fire exactly on the bound",
      not fired("HL", 6.40, 6.40, **ARG))
S = dict(side="SELL", label="LH", confirmed="2026-10-14", since="2026-10-10")
check("SELL fires above its bound", fired("LH", 7.50, 7.00, **S))
check("SELL does NOT fire below its bound", not fired("LH", 6.80, 7.00, **S))
check("no bound means the structure event alone is enough",
      fired("HL", 0.01, None, **ARG))

print("\n── all four labels work, on either side ──")
for lab in ("HH", "HL", "LL", "LH"):
    check(f"{lab} fires when confirmed fresh",
          fired(lab, 10.0, side="BUY", label=lab,
                confirmed="2026-10-14", since="2026-10-01"))
check("BUY on LL is allowed (buying capitulation is a real trade)",
      fired("LL", 10.0, side="BUY", label="LL",
            confirmed="2026-10-14", since="2026-10-01"))

print("\n── the intent layer accepts and rejects the right rows ──")
base = dict(Row_ID="2026-10-14-JOBY-01", Acct="171", Ticker="JOBY", Side="BUY",
            Close_Is="HL", Trigger_Price="", Limit_Price="", Qty="50",
            Expires_On="")
check("a structure row needs no Trigger_Price",
      oc.normalize(base)["Trigger_Price"] == "")
check("a price row still requires one",
      not _ok(lambda: oc.normalize(dict(base, Close_Is="ABOVE"))))
check("an unknown Close_Is is refused",
      not _ok(lambda: oc.normalize(dict(base, Close_Is="SIDEWAYS"))))
check("a zero bound is refused even though the bound is optional",
      not _ok(lambda: oc.normalize(dict(base, Trigger_Price="0"))))

print("\n── Qty may only be blank when chained ──")
check("blank Qty with no parent is refused",
      not _ok(lambda: oc.normalize(dict(base, Qty=""))))
check("blank Qty WITH a parent is allowed",
      oc.normalize(dict(base, Qty="", After_Trigger_ID="1"))["Qty"] == "")
check("a row cannot wait for its own Trigger_ID",
      not _ok(lambda: oc.normalize(dict(base, Trigger_ID="1",
                                        After_Trigger_ID="1"))))
check("a label he types is kept verbatim",
      oc.normalize(dict(base, Trigger_ID="joby-entry"))["Trigger_ID"]
      == "joby-entry")

print("\n── the fingerprint does not move for existing rows ──")
# Adding the chain columns to FIELDS would have VOIDed every live row on
# deploy with INTENT_CHANGED. This is the assertion that it did not.
priced = dict(base, Close_Is="BELOW", Trigger_Price="6.00")
check("an unchained row hashes identically with the fields absent or blank",
      oc.fingerprint(priced)
      == oc.fingerprint(dict(priced, Trigger_ID="", After_Trigger_ID="")))
check("a chained row hashes differently",
      oc.fingerprint(priced)
      != oc.fingerprint(dict(priced, After_Trigger_ID="1")))
check("changing the label it waits for changes the fingerprint",
      oc.fingerprint(dict(priced, After_Trigger_ID="1"))
      != oc.fingerprint(dict(priced, After_Trigger_ID="2")))
check("publishing a label changes it too",
      oc.fingerprint(priced) != oc.fingerprint(dict(priced, Trigger_ID="1")))

print("\n── chain faults are graph-level ──")


def R(rid, mine="", after=""):
    """A sheet row: its Pi-stamped Row_ID, the label it publishes, the one it
    waits for."""
    return {"Row_ID": rid, "Trigger_ID": mine, "After_Trigger_ID": after}


check("no chains at all is clean", not oc.chain_problems([R("A"), R("B")]))
check("a 3-row chain is allowed",
      not oc.chain_problems([R("A", "1"), R("B", "2", "1"), R("C", "", "2")]))
four = oc.chain_problems([R("A", "1"), R("B", "2", "1"),
                          R("C", "3", "2"), R("D", "", "3")])
check("a 4th row is too deep", "CHAIN_TOO_DEEP" in four.get("D", ""), str(four))
check("depth counts ROWS, not links", "D" in four and "C" not in four, str(four))
gone = oc.chain_problems([R("B", "", "GHOST")])
check("waiting on a label nobody publishes is caught",
      "CHAIN_BROKEN" in gone.get("B", ""), str(gone))
loop = oc.chain_problems([R("A", "1", "3"), R("B", "2", "1"), R("C", "3", "2")])
check("a loop is caught, on every row in it",
      all("CHAIN_LOOP" in loop.get(k, "") for k in "ABC"), str(loop))

print("\n── a Trigger_ID may not be reused while still live ──")
dup = oc.chain_problems([R("A", "1"), R("B", "1"), R("C", "", "1")])
check("two live rows publishing the same label is an error",
      any("TRIGGER_ID_IN_USE" in v for v in dup.values()), str(dup))
check("...and it names the offending label",
      any("'1'" in v for v in dup.values()), str(dup))
# His rule: free again once the first one has finished.
reuse = oc.chain_problems([R("A", "1"), R("B", "1"), R("C", "", "1")],
                          done_ids={"A"})
check("reusing a label whose row is DONE is fine", not reuse, str(reuse))
check("...and the live row is the one that now owns it",
      oc.parent_of([R("A", "1"), R("B", "1"), R("C", "", "1")],
                   done_ids={"A"}).get("C") == "B")

print("\n── labels resolve to real rows ──")
rows = [R("2026-10-14-JOBY-01", "1"), R("2026-10-14-JOBY-02", "", "1")]
check("the child's parent is found by label",
      oc.parent_of(rows).get("2026-10-14-JOBY-02") == "2026-10-14-JOBY-01")
check("an unchained row has no parent", "2026-10-14-JOBY-01"
      not in oc.parent_of(rows))
check("a dangling label resolves to nothing",
      not oc.parent_of([R("X", "", "NOPE")]))

print(f"\n{'ALL PASS' if not FAILED else str(len(FAILED)) + ' FAILED: ' + ', '.join(FAILED)}")
raise SystemExit(1 if FAILED else 0)
