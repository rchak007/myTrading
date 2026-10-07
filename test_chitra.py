#!/usr/bin/env python3
"""
test_chitra.py
==============
Tests for chitra.py — her positions, her Merrill orders, and the conditions
Chakravarti types into the tab. No network, no sheet, no Schwab.

    .venv/bin/python test_chitra.py

Her account has no API. Everything here is transcribed by hand from statements
and screenshots, so the one thing that must never happen is the tab claiming
something is arranged when it is not.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chitra                                            # noqa: E402

FAILED = []
EX = lambda e: (e["quote"]["lastPrice"], e["quote"]["netPercentChange"])
Q = {t: {"quote": {"lastPrice": p, "netPercentChange": 1.0}}
     for t, p in (("TSLA", 357.6), ("GOOG", 337.4), ("MRVL", 272.0),
                  ("MSFT", 516.7), ("MU", 1102.4), ("IBIT", 49.1))}
CLOSES = {"TSLA": 354.11, "GOOG": 334.93, "MRVL": 268.08, "MSFT": 512.80,
          "MU": 1097.39, "IBIT": 47.96}


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def cond(t, side, d, trig, qty=1, row=58):
    return {"row": row, "Ticker": t, "Side": side, "Close_Is": d,
            "Trigger_Price": str(trig), "Qty": str(qty), "Note": ""}


print("\n── a condition fires on the CLOSE, never the live price ──")
# "Closes below 214" is not "touched 214". TSLA's live price is 357.60 and its
# close 354.11; a trigger between the two must NOT fire.
check("a trigger above the close fires",
      chitra.evaluate(cond("TSLA", "SELL", "BELOW", 500), 354.11) == chitra.MET)
check("a trigger below the close does not",
      chitra.evaluate(cond("TSLA", "SELL", "BELOW", 300), 354.11) != chitra.MET)
check("a trigger BETWEEN live and close does not fire on the live price",
      chitra.evaluate(cond("TSLA", "SELL", "BELOW", 356), 354.11) == chitra.MET
      and chitra.evaluate(cond("TSLA", "SELL", "BELOW", 350), 354.11) != chitra.MET)
check("ABOVE fires the other way",
      chitra.evaluate(cond("GOOG", "SELL", "ABOVE", 300), 334.93) == chitra.MET
      and chitra.evaluate(cond("GOOG", "SELL", "ABOVE", 900), 334.93) != chitra.MET)

print("\n── a row that cannot be judged says so, and never fires ──")
for bad, why in ((cond("MSFT", "SELL", "SIDEWAYS", 5), "bad direction"),
                 (cond("MSFT", "HOLD", "BELOW", 5), "bad side"),
                 (cond("MSFT", "SELL", "BELOW", ""), "no trigger")):
    got = chitra.evaluate(bad, 512.8)
    check(f"{why} → explained, not MET", got != chitra.MET and "need" in got, got)
check("no close for the ticker → says so rather than guessing",
      "no close" in chitra.evaluate(cond("XYZ", "SELL", "BELOW", 10), None))

print("\n── an unjudgeable row must not claim protection ──")
# THE BUG THIS CAUGHT. A SIDEWAYS typo fell through to the price rule — SELL
# at 5 against a price of 516 reads as a stop — and MSFT showed Has_Stop = P
# from a row that can never act. Same shape as the Dashboard bugs: "covered"
# about the one order that cannot run.
broken = chitra.coverage("MSFT", 516.7, [], [cond("MSFT", "SELL", "SIDEWAYS", 5)], 2)
check("a SIDEWAYS typo claims nothing", broken["Has_Stop"] == "N", str(broken))
good = chitra.coverage("MSFT", 516.7, [], [cond("MSFT", "SELL", "BELOW", 480)], 2)
check("...but a valid condition does claim it", good["Has_Stop"] == "P", str(good))

print("\n── coverage uses the same rule as his Dashboard ──")
# Direction decides for a condition; price decides for a resting order.
check("BUY/BELOW is a dip, not a breakout",
      chitra.coverage("MRVL", 272.0, [], [cond("MRVL", "BUY", "BELOW", 200)], 9
                      )["Has_Dip"] == "P")
check("SELL/ABOVE is a trim",
      chitra.coverage("GOOG", 337.4, [], [cond("GOOG", "SELL", "ABOVE", 400)], 2
                      )["Has_Trim"] == "P")
resting = [{"Ticker": "MU", "Side": "SELL", "Type": "STOP", "Qty": "1",
            "Limit_Price": "", "Stop_Price": "900"}]
check("a resting Merrill SELL below price is a stop, and outranks a condition",
      chitra.coverage("MU", 1102.4, resting, [], 3)["Has_Stop"] == "Y")
check("an OVERSELL condition claims nothing",
      chitra.coverage("MU", 1102.4, [], [cond("MU", "SELL", "BELOW", 900, qty=99)],
                      3)["Has_Stop"] == "N")

print("\n── the positions block ──")
rows = chitra.load(log=lambda *_: None)
check("her holdings load", len(rows) >= 6, str(len(rows)))
pos = chitra.build_positions(rows, Q, EX, resting, [cond("TSLA", "SELL", "BELOW", 500)])
by = {r[0]: r for r in pos}
check("cash carries BLANK flags, not N",
      by["IIAXX"][9:13] == ["", "", "", ""], str(by["IIAXX"][9:13]))
check("MU picks up the resting Merrill stop", by["MU"][9] == "Y", by["MU"][9])
check("TSLA picks up the typed condition", by["TSLA"][9] == "P", by["TSLA"][9])
check("a TOTAL row is present", "TOTAL" in by)
check("every row is the full width",
      all(len(r) == len(chitra.POS_COLS) for r in pos),
      str({len(r) for r in pos}))

print("\n── fencing and the sweep ──")
res = chitra.load_reserves(log=lambda *_: None)
check("all six holdings are fenced",
      {t for t, v in res.items() if v["fenced"]}
      == {"MU", "MRVL", "TSLA", "MSFT", "IBIT", "GOOG"}, str(sorted(res)))
cash, seeded, open_buys, free = chitra.cash_position(rows, res)
check("the sweep is read from the CASH row, not guessed",
      abs(cash - 6764.23) < 0.01, f"{cash}")
check("fencing alone seeds nothing", seeded == 0.0, str(seeded))
check("free cash is the whole sweep while nothing is seeded",
      abs(free - cash) < 0.01)
# Over-commitment has to be VISIBLE, because nothing here can refuse a trade.
over = {"MU": {"seed": 5000.0, "fenced": True, "note": ""},
        "TSLA": {"seed": 3000.0, "fenced": True, "note": ""}}
_, s_over, _b, f_over = chitra.cash_position(rows, over)
check("seeding past the sweep makes free go NEGATIVE",
      s_over == 8000.0 and f_over < 0, f"seeded {s_over} free {f_over}")
fenced_pos = chitra.build_positions(rows, Q, EX, [], [], res)
by_f = {r[0]: r for r in fenced_pos}
check("a fenced ticker shows the lock", by_f["MU"][13] == "🔒", by_f["MU"][13])
check("cash is not fenced", by_f["IIAXX"][13] == "", by_f["IIAXX"][13])
check("Seed_Reserved is blank at zero, not 0.0",
      by_f["MU"][14] == "", repr(by_f["MU"][14]))
seeded_pos = chitra.build_positions(
    rows, Q, EX, [], [], {"MU": {"seed": 2000.0, "fenced": True, "note": ""}})
check("a seeded ticker shows its dollars",
      {r[0]: r for r in seeded_pos}["MU"][14] == 2000.0)

# An open BUY commits cash the moment it rests — the same overstatement the
# covered-call collateral was, in the other pocket.
live = chitra.load_orders(log=lambda *_: None)
_c, _s, buys, free_after = chitra.cash_position(rows, res, live)
check("an open BUY limit commits cash", buys == 360.0, str(buys))
check("...and comes out of free", abs(free_after - (cash - 360.0)) < 0.01,
      f"{free_after} vs {cash - 360.0}")
check("a SELL commits no cash",
      chitra.cash_position(rows, res,
                           [{"Side": "SELL", "Limit_Price": "412", "Qty": "1"}]
                           )[2] == 0.0)
check("an order with no price commits nothing rather than crashing",
      chitra.cash_position(rows, res,
                           [{"Side": "BUY", "Limit_Price": "", "Qty": "1"}]
                           )[2] == 0.0)

print("\n── the orders block ──")
empty = chitra.build_orders([])
check("an empty file says NONE AS OF A DATE, not just nothing",
      "none resting" in empty[0][0] and "as of" in empty[0][0], empty[0][0])
check("a real order renders", chitra.build_orders(resting)[0][0] == "MU")

print("\n── the layout cannot collide ──")
# His typed rows sit at fixed rows. If a machine section could grow into them
# the sheet would eat his conditions, which is the one unrecoverable failure.
check("positions cannot reach the orders label",
      chitra.POS_START + chitra.POS_MAX <= chitra.ORD_LABEL_ROW,
      f"{chitra.POS_START + chitra.POS_MAX} vs {chitra.ORD_LABEL_ROW}")
check("orders cannot reach the conditions label",
      chitra.ORD_START + chitra.ORD_MAX <= chitra.CON_LABEL_ROW,
      f"{chitra.ORD_START + chitra.ORD_MAX} vs {chitra.CON_LABEL_ROW}")
check("the human columns come before the machine ones",
      chitra.CON_COLS[:chitra.CON_HUMAN_N] == chitra.CON_HUMAN)
# SIZED FOR HER ACCOUNT. The first layout reserved 30 rows for 7 holdings and
# buried the orders at row 37 behind blank space, which is how Chakravarti
# came to report that orders he had just supplied were missing.
held = len([r for r in rows])
check(f"the positions cap ({chitra.POS_MAX}) is room to grow, not a canyon",
      held <= chitra.POS_MAX <= held * 2 + 2,
      f"{held} held, cap {chitra.POS_MAX}")
check("the orders section is visible without scrolling",
      chitra.ORD_LABEL_ROW <= 22, str(chitra.ORD_LABEL_ROW))
check("every section still fits on one screen",
      chitra.CON_START <= 40, str(chitra.CON_START))

print("\n── status for the sheet ──")
st = chitra.build_condition_status(
    [cond("TSLA", "SELL", "BELOW", 500, row=58),
     cond("GOOG", "SELL", "ABOVE", 900, row=59)], Q, EX, CLOSES)
check("one met, one waiting",
      st[58][2] == chitra.MET and st[59][2].startswith("waiting"),
      f"{st[58][2]} / {st[59][2]}")
check("both carry the live price AND the close, so the gap is visible",
      st[58][0] and st[58][1] and st[58][0] != st[58][1], str(st[58][:2]))

print(f"\n{'ALL PASS' if not FAILED else str(len(FAILED)) + ' FAILED: ' + ', '.join(FAILED)}")
raise SystemExit(1 if FAILED else 0)
