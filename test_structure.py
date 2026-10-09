#!/usr/bin/env python3
"""
test_structure.py
=================
Tests for core/structure.py. No network, no sheet, no Schwab.

    .venv/bin/python test_structure.py

The off-by-one is the thing to guard. If the balance window accidentally
includes today, `close > max(high)` needs the close to BE the high and the
detector simply never fires — a failure that looks exactly like "no setups
today", which is the failure mode this codebase keeps meeting.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import structure as ST                           # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def bars(closes, *, spread=0.2):
    """H/L/C from a close path, each bar a tight range around its close."""
    return ([c + spread for c in closes],
            [c - spread for c in closes],
            list(closes))


ATR = 1.0

# A leg up of 5 ATR over 5 bars, then 5 bars balanced inside ~0.6 ATR.
LEG_UP = [100, 101, 102.5, 104, 105]
BAL = [105.1, 104.9, 105.2, 105.0, 105.1]
# boundary = max high over BAL = 105.2 + 0.2 = 105.4

print("\n── the P ──")
h, lo, c = bars(LEG_UP + BAL + [105.3])       # today closes INSIDE the balance
s = ST.classify(h, lo, c, ATR)
check("balance + up-leg, close inside → P_FORMED", s.state == ST.P_FORMED, repr(s))
check("shape is P", s.shape == "P", str(s.shape))
check("boundary is the balance HIGH, not its close",
      abs(s.boundary - 105.4) < 1e-9, str(s.boundary))
check("floor is the balance low", abs(s.floor - 104.7) < 1e-9, str(s.floor))
# 5.1, not 5.0: the leg ENDS at the balance's first bar (105.1), which is the
# design's `close[-5] - close[-10]`. The leg is the move INTO the balance, so
# the balance's own first bar is its last point.
check("leg measured in ATR", abs(s.leg_atr - 5.1) < 1e-9, str(s.leg_atr))
check("armed, not fired", s.armed and not s.fired)

h, lo, c = bars(LEG_UP + BAL + [105.5])       # today closes ABOVE the boundary
s = ST.classify(h, lo, c, ATR)
check("a close above the boundary → TRIGGERED", s.state == ST.TRIGGERED, repr(s))
check("fired", s.fired and not s.armed)
check("break distance reported", s.break_atr and s.break_atr > 0, str(s.break_atr))

# THE OFF-BY-ONE. Today's own high must not raise the boundary it is tested
# against, or nothing can ever fire.
h, lo, c = bars(LEG_UP + BAL + [105.5])
h[-1] = 999.0
check("today's own HIGH cannot lift the boundary (the off-by-one)",
      ST.classify(h, lo, c, ATR).state == ST.TRIGGERED,
      repr(ST.classify(h, lo, c, ATR)))

print("\n── the shapes we do NOT buy ──")
h, lo, c = bars([105, 104, 102.5, 101, 100] + [99.9, 100.1, 99.8, 100.0, 99.9]
                + [100.4])
s = ST.classify(h, lo, c, ATR)
check("a leg DOWN into balance is b, and b does not fire",
      s.shape == "b" and s.state == ST.WATCHING, repr(s))
check("...but its boundary is still measured, for the control group",
      s.boundary is not None and s.break_atr is not None, repr(s))

# The flat run must sit at the SAME level as the balance. An earlier version
# of this fixture ran flat at 100 and then balanced at 105, which is a +5 ATR
# jump — it was a textbook P and the test was asserting the opposite.
FLAT = [105.0, 105.1, 104.9, 105.0, 105.1]
h, lo, c = bars(FLAT + BAL + [105.5])
s = ST.classify(h, lo, c, ATR)
check("no leg into balance is D, and D does not fire",
      s.shape == "D" and s.state == ST.WATCHING, repr(s))
check("...and D's leg really is small", abs(s.leg_atr) < ST.LEG_MIN_ATR,
      str(s.leg_atr))

h, lo, c = bars(LEG_UP + [105, 110, 103, 108, 104] + [111])
s = ST.classify(h, lo, c, ATR)
check("a wide range is not balance at all", s.state == ST.WATCHING
      and s.shape is None, repr(s))
check("...and says so in ATR", "not balanced" in s.why, s.why)

print("\n── a late breakout is not a signal ──")
# Broke out, then ran for four more days. The window now holds the breakout,
# so it is no longer a balance and must NOT re-fire.
h, lo, c = bars(LEG_UP + BAL + [105.5, 107, 109, 111, 113])
s = ST.classify(h, lo, c, ATR)
check("four days into the move, nothing fires", s.state == ST.WATCHING, repr(s))

print("\n── ATR is the unit, so price level is irrelevant ──")
cheap = ST.classify(*bars([x / 20 for x in LEG_UP + BAL + [105.5]],
                          spread=0.01), 1.0 / 20)
check("the same shape at $5 reads the same as at $105",
      cheap.state == ST.TRIGGERED, repr(cheap))
check("...and the leg is the same multiple of ATR",
      abs(cheap.leg_atr - 5.1) < 1e-6, str(cheap.leg_atr))

print("\n── fails closed ──")
check("too few bars is NO_DATA, not a verdict",
      ST.classify([1, 2], [1, 2], [1, 2], 1.0).state == ST.NO_DATA)
check("no ATR is NO_DATA", ST.classify(h, lo, c, None).state == ST.NO_DATA)
check("zero ATR is NO_DATA", ST.classify(h, lo, c, 0).state == ST.NO_DATA)
check("NaN ATR is NO_DATA", ST.classify(h, lo, c, float("nan")).state == ST.NO_DATA)
h2, lo2, c2 = bars(LEG_UP + BAL + [105.5])
c2[3] = None
check("a gap in the bars is NO_DATA, not a guess",
      ST.classify(h2, lo2, c2, ATR).state == ST.NO_DATA)
check("NO_DATA is never armed or fired",
      not ST.classify([], [], [], 1).armed
      and not ST.classify([], [], [], 1).fired)
check("MIN_BARS is what classify actually requires",
      ST.classify(*bars((LEG_UP + BAL + [105.5])[-ST.MIN_BARS:]),
                  ATR).state != ST.NO_DATA
      and ST.classify(*bars((LEG_UP + BAL + [105.5])[-(ST.MIN_BARS - 1):]),
                      ATR).state == ST.NO_DATA, str(ST.MIN_BARS))

print("\n── the dollar budget buys WHOLE shares ──")
check("$500 at $6.12 is 81 shares", ST.shares_for(500, 6.12) == 81,
      str(ST.shares_for(500, 6.12)))
check("never rounds up past the budget", ST.shares_for(500, 6.12) * 6.12 <= 500)
check("too small for one share is 0, not a fraction",
      ST.shares_for(500, 612.40) == 0)
check("an exact fit is not short-changed", ST.shares_for(600, 6.0) == 100)
check("junk is 0, never an exception",
      ST.shares_for(None, 5) == 0 and ST.shares_for(500, None) == 0
      and ST.shares_for(-5, 5) == 0 and ST.shares_for(500, 0) == 0
      and ST.shares_for("abc", 5) == 0)

print("\n── the DataFrame door ──")
try:
    import pandas as pd
    h, lo, c = bars(LEG_UP + BAL + [105.5])
    df = pd.DataFrame({"High": h, "Low": lo, "Close": c, "ATR": [ATR] * len(c)})
    check("classify_frame reads High/Low/Close + ATR",
          ST.classify_frame(df).state == ST.TRIGGERED, repr(ST.classify_frame(df)))
    # atr=100 makes the 5.1-point leg 0.05 ATR — no leg at all, so a D, so
    # WATCHING. That the verdict CHANGES is the proof the argument is used.
    check("an explicit ATR overrides the column",
          ST.classify_frame(df, atr=100.0).state == ST.WATCHING
          and ST.classify_frame(df, atr=100.0).shape == "D",
          repr(ST.classify_frame(df, atr=100.0)))
    check("no ATR anywhere is NO_DATA",
          ST.classify_frame(df.drop(columns=["ATR"])).state == ST.NO_DATA)
    check("a missing price column is NO_DATA, not a crash",
          ST.classify_frame(df.drop(columns=["High"])).state == ST.NO_DATA)
    check("an empty frame is NO_DATA",
          ST.classify_frame(pd.DataFrame()).state == ST.NO_DATA
          and ST.classify_frame(None).state == ST.NO_DATA)
except ImportError:
    print("  SKIP  pandas not installed")

print(f"\n{'ALL PASS' if not FAILED else str(len(FAILED)) + ' FAILED: ' + ', '.join(FAILED)}")
raise SystemExit(1 if FAILED else 0)
