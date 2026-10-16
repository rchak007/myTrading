#!/usr/bin/env python3
"""
test_benter.py
==============
The Benter 0-10 rating. No network, no sheet.

    .venv/bin/python test_benter.py

TWO MISCALIBRATIONS THIS PINS, both found by running the real 135-ticker list
rather than by reasoning about it:

  NOTHING COULD SCORE WELL. The first confidence discount charged 0.77 for an
  ordinary 0.84 target spread and 0.83 for an ordinary 3.2% ATR. Compounded,
  AVGO's 53% gap came out 5.8/NEUTRAL and no name in a nine-stock sample
  cleared 6 — a scale whose top half is unreachable is decoration.

  A BIGGER GAP SCORED BETTER, FOREVER. RCAT 7.1 on a 237% gap, ONDS 8.2 on
  197%, STKE on 349%. Those are $1-to-$6 names where consensus claims the
  stock is worth three times the price. That is not a three-times edge, it is
  an anchor nobody should trust, and the curve now falls away past ~70%.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.benter import score, concentration_note                # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def s(gap, spread=0.8, atr=3.0, analysts=20, price=100.0):
    """Score a hypothetical at a given gap, with otherwise ordinary inputs."""
    t = price * (1 + gap)
    return score(price, t, target_high=t * (1 + spread / 2),
                 target_low=t * (1 - spread / 2), analysts=analysts,
                 atr_pct=atr)


print("\n── the scale is actually reachable ──")
# The bug: with ordinary inputs nothing could clear 6.
check("an ordinary 55% gap is a CLEAN BUY", s(0.55).score >= 9.0,
      str(s(0.55).score))
check("an ordinary 40% gap is a BUY", 7.0 <= s(0.40).score < 9.0,
      str(s(0.40).score))
check("ordinary inputs cost no confidence", s(0.40).confidence >= 0.97,
      str(s(0.40).confidence))

print("\n── the ends of the scale ──")
check("fairly priced is ~3, not 5", 2.5 <= s(0.0).score <= 3.5, str(s(0.0).score))
check("above fair value scores near zero", s(-0.10).score <= 1.0,
      str(s(-0.10).score))
check("a negative gap never scores a buy", s(-0.05).score < 4.5,
      str(s(-0.05).score))
check("nothing exceeds 10 or drops below 0",
      all(0.0 <= s(g).score <= 10.0
          for g in (-0.5, -0.1, 0, 0.3, 0.6, 1.0, 3.0, 10.0)))

print("\n── an ENORMOUS gap is a red flag, not a bigger edge ──")
# RCAT at 237%, ONDS at 197%, STKE at 349% all scored 7-8 before this.
check("the curve peaks around 55-70%", s(0.60).score > s(1.20).score,
      f"{s(0.60).score} vs {s(1.20).score}")
check("a 237% gap scores WORSE than a 50% one", s(2.37).score < s(0.50).score,
      f"{s(2.37).score} vs {s(0.50).score}")
check("...and lands in DON'T BUY or worse", s(2.37).score < 4.5,
      str(s(2.37).score))
check("a 349% gap is AVOID", s(3.49).score < 2.5, str(s(3.49).score))
check("the fall is monotonic past the peak",
      s(0.70).score >= s(1.0).score >= s(1.5).score >= s(2.5).score)

print("\n── confidence is MEASURED, not asserted ──")
check("wide disagreement costs", s(0.50, spread=1.5).score < s(0.50, spread=0.5).score)
check("high volatility costs", s(0.50, atr=9.0).score < s(0.50, atr=2.0).score)
check("thin coverage costs", s(0.50, analysts=3).score < s(0.50, analysts=30).score)
check("...and the discount is floored, not unbounded",
      s(0.55, spread=5.0, atr=40.0).score > 0.5,
      str(s(0.55, spread=5.0, atr=40.0).score))
# The real pair: identical gaps, different trustworthiness.
calm, wild = s(0.50, spread=0.5, atr=2.0), s(0.50, spread=1.5, atr=9.0)
check("same gap, very different score when confidence differs",
      calm.score - wild.score >= 2.0, f"{calm.score} vs {wild.score}")

print("\n── the old rubric's collisions are gone ──")
# GOOGL 23% and MSFT 10% both scored 6; SPCX 52% and AMD 12% both scored 5.
check("a 23% gap and a 10% gap no longer tie",
      abs(s(0.23).score - s(0.10).score) >= 1.0,
      f"{s(0.23).score} vs {s(0.10).score}")
check("a 50% gap and a 12% gap no longer tie",
      abs(s(0.50).score - s(0.12).score) >= 2.0,
      f"{s(0.50).score} vs {s(0.12).score}")

print("\n── it refuses rather than guesses ──")
for lbl, b in (("no price", score(None, 100)),
               ("zero price", score(0, 100)),
               ("no target", score(100, None)),
               ("zero target", score(100, 0)),
               ("junk target", score(100, "n/a"))):
    check(f"{lbl} → unscored", b.score is None and b.why, repr(b))
check("missing extras still score", score(100, 150).score is not None)
check("...at full confidence, never penalised for absent data",
      score(100, 150).confidence == 1.0)

print("\n── every score carries its reasoning ──")
b = s(0.50, spread=1.5, atr=9.0)
check("why names the gap", "gap" in b.why and "%" in b.why, b.why)
check("why names each discount applied",
      "spread" in b.why and "ATR" in b.why, b.why)
check("a band and an action come with it", b.band and b.action, repr(b))

print("\n── the concentration warning ──")
rows = [{"Ticker": "AVGO", "Sector": "Technology", "score": 8.8},
        {"Ticker": "NVDA", "Sector": "Technology", "score": 8.2},
        {"Ticker": "STM", "Sector": "Technology", "score": 8.7}]
note = concentration_note(rows)
check("fires when the top scores are one sector", "TECHNOLOGY" in note, note)
check("...and names them", all(t in note for t in ("AVGO", "NVDA", "STM")), note)
mixed = [{"Ticker": "AVGO", "Sector": "Technology", "score": 8.8},
         {"Ticker": "SE", "Sector": "Consumer Cyclical", "score": 9.5}]
check("silent when the top scores are different sectors",
      not concentration_note(mixed))
check("silent when only one name scores well",
      not concentration_note([rows[0], {"Ticker": "X", "Sector": "Technology",
                                        "score": 3.0}]))

print(f"\n{'ALL PASS' if not FAILED else str(len(FAILED)) + ' FAILED: ' + ', '.join(FAILED)}")
raise SystemExit(1 if FAILED else 0)
