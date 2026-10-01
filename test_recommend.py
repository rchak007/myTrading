#!/usr/bin/env python3
"""
test_recommend.py
=================
Tests for core/recommend.py. Runs anywhere — no Schwab, no Sheets, no network.

    .venv/bin/python test_recommend.py

Each case here is a real trap the live data contains, not a hypothetical. The
numbers in the fixtures are the actual 2026-10-01 values for those tickers.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.recommend import recommend, recommend_row, Rec   # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def side_invariants(label, r: Rec, p: float):
    """The one property that must never break, whatever the inputs."""
    check(f"{label}: stop below price",
          r.Rec_Stop is None or r.Rec_Stop < p, f"{r.Rec_Stop} vs {p}")
    check(f"{label}: trim above price",
          r.Rec_Trim is None or r.Rec_Trim > p, f"{r.Rec_Trim} vs {p}")
    check(f"{label}: dip below price",
          r.Rec_Dip is None or r.Rec_Dip < p, f"{r.Rec_Dip} vs {p}")
    check(f"{label}: breakout above price",
          r.Rec_Breakout is None or r.Rec_Breakout > p, f"{r.Rec_Breakout} vs {p}")
    for f in ("Rec_Stop", "Rec_Trim", "Rec_Dip", "Rec_Breakout"):
        v = getattr(r, f)
        check(f"{label}: {f} is None or positive, never 0.0",
              v is None or v > 0, repr(v))


print("\n── the three traps the live data contains ──")

# TRAP 1: reverse-split ATH. FCEL's real ATH field reads 234900 at a price
# of 16.81. Anything that uses it unguarded emits a breakout at 234900.
r = recommend(16.81, 1.05, supertrend=19.95, supertrend_signal="SELL",
              nearest_support=14.71, nearest_resistance=18.33,
              mrc_zone="🔵 Near_Mean", mrc_r1=29.26, mrc_mean=20.39, mrc_s1=11.52,
              ath=234900.0, score=2, structure="BULLISH", regime="BEAR")
check("FCEL: corrupt ATH (234900) never becomes a level",
      r.Rec_Breakout is None, repr(r.Rec_Breakout))
check("FCEL: breakout blocked by downtrend first", r.Breakout_Basis == "downtrend",
      r.Breakout_Basis)
side_invariants("FCEL", r, 16.81)

# Same corrupt ATH but WITH an uptrend and a high score, so the ATH branch is
# actually reached — this is the case the guard exists for.
r2 = recommend(16.81, 1.05, supertrend=14.0, supertrend_signal="BUY",
               nearest_support=14.71, nearest_resistance=None,
               mrc_zone="Above_Mean", mrc_r1=29.26, ath=234900.0, score=80)
check("corrupt ATH rejected even when the ATH branch is reached",
      r2.Rec_Breakout is None and r2.Breakout_Basis == "-", r2.Breakout_Basis)

# A SANE ATH must still work, or the guard is just breaking the feature.
r3 = recommend(100.0, 3.0, supertrend=90.0, supertrend_signal="BUY",
               nearest_support=95.0, nearest_resistance=None,
               mrc_zone="Above_Mean", mrc_r1=130.0, ath=120.0, score=80)
check("sane ATH (1.2x price) IS used", r3.Breakout_Basis == "ATH", r3.Breakout_Basis)
check("sane ATH level sits above the ATH", r3.Rec_Breakout > 120.0, repr(r3.Rec_Breakout))

# TRAP 2: Supertrend in SELL mode sits ABOVE price — 64 of 128 names.
r = recommend(15.92, 0.68, supertrend=17.96, supertrend_signal="SELL",
              nearest_support=15.22, nearest_resistance=17.75,
              mrc_zone="🔵 Near_Mean", mrc_r1=19.75, mrc_mean=17.15, mrc_s1=14.54,
              score=13, structure="MIXED", regime="BEAR")
check("SOFI: SELL-mode Supertrend (17.96 > 15.92) never used as a stop",
      r.Rec_Stop < 15.92 and r.Stop_Basis != "ST", f"{r.Rec_Stop} {r.Stop_Basis}")
side_invariants("SOFI", r, 15.92)

# TRAP 3: negative MRC_S2 / unusable bands.
r = recommend(18.85, 1.4, supertrend=16.0, supertrend_signal="BUY",
              nearest_support=None, nearest_resistance=None,
              mrc_zone="Below_Mean", mrc_mean=None, mrc_s1=-0.42, score=10)
check("negative MRC_S1 (-0.42) never becomes a dip level",
      r.Rec_Dip is None or r.Rec_Dip > 0, repr(r.Rec_Dip))
check("...and the ladder still produces something usable",
      r.Rec_Dip is None or 0 < r.Rec_Dip < 18.85, repr(r.Rec_Dip))
side_invariants("MSTX-like", r, 18.85)

print("\n── two stops: close-confirmed, then disaster ──")
# A resting Schwab stop is a TOUCH trigger — one wick takes you out at the
# worst price of the day. The soft stop goes in the Orders tab (close
# confirmed); the hard one rests at Schwab, further out, for disasters only.
r = recommend(100.0, 2.0, supertrend=94.0, supertrend_signal="BUY",
              nearest_support=93.0, mrc_zone="Above_Mean", mrc_r1=112.0, score=10)
check("soft stop is the technical level", abs(r.Rec_Stop - 94.0) < 1e-9, repr(r.Rec_Stop))
check("hard stop sits BELOW the soft stop", r.Rec_Stop_Hard < r.Rec_Stop,
      f"{r.Rec_Stop_Hard} vs {r.Rec_Stop}")
check("the middle rung sits between them",
      r.Rec_Stop_Hard < r.Rec_Stop2 < r.Rec_Stop,
      f"{r.Rec_Stop_Hard} / {r.Rec_Stop2} / {r.Rec_Stop}")
check("hard stop is 1.5x the soft distance (6.0 → 91.0)",
      abs(r.Rec_Stop_Hard - 91.0) < 1e-9, repr(r.Rec_Stop_Hard))
# A tight soft stop still needs real room before "disaster" is the right word.
tight = recommend(100.0, 2.0, supertrend=99.0, supertrend_signal="BUY",
                  mrc_zone="Above_Mean", mrc_r1=112.0, score=10)
check("a tight soft stop does not give a tight hard stop",
      tight.Rec_Stop_Hard <= 100.0 - 2.0 * 3.0 + 1e-9,
      f"soft {tight.Rec_Stop} hard {tight.Rec_Stop_Hard}")
check("hard stop is capped at HARD_STOP_MAX_ATR",
      (lambda x: x.Rec_Stop_Hard >= 100.0 - 2.0 * 8.0 - 1e-9)(
          recommend(100.0, 2.0, nearest_support=60.0, supertrend_signal="SELL",
                    mrc_zone="Above_Mean", mrc_r1=112.0, score=10)))
check("no soft stop → no hard stop", recommend(100.0, None).Rec_Stop_Hard is None)

print("\n── the dip ladder, and its two tiers ──")
# THE CASE THAT PROMPTED THIS. PLTR had a real pivot at 164.55 under a 171.78
# Supertrend stop, and the first version threw it away as "below-stop". The
# stop protects shares you HOLD; the dip deploys FRESH capital. Different money.
pltr = recommend(186.82, 5.98, supertrend=171.78, supertrend_signal="BUY",
                 nearest_support=164.55, nearest_resistance=194.68,
                 mrc_zone="🟠 OB", mrc_r1=170.59, mrc_r2=203.04,
                 mrc_mean=147.66, mrc_s1=124.73, mrc_s2=92.28,
                 ath=207.52, score=50, structure="MIXED", regime="BULL")
check("PLTR: dip is populated, not blank", pltr.Rec_Dip is not None, pltr.Dip_Basis)
check("PLTR: dip is tagged a re-entry (it sits below the stop ladder)",
      pltr.Dip_Basis.startswith("re:"), pltr.Dip_Basis)
# The 164.55 pivot is now consumed by the stop ladder — it sits between the
# soft and hard stops, so bidding there would mean buying at a level where
# you are still selling. The re-entry drops to the next real shelf.
check("PLTR: dip clears the whole stop ladder",
      pltr.Rec_Dip < pltr.Rec_Stop_Hard,
      f"dip {pltr.Rec_Dip} vs hard {pltr.Rec_Stop_Hard}")
check("PLTR: dip is a STRUCTURAL level, not a volatility rung",
      not pltr.Dip_Basis.endswith("ATR"), pltr.Dip_Basis)
side_invariants("PLTR", pltr, 186.82)

# AEHR: the pivot is too near AND the next band is 40% down. Only the ATR
# ladder saves this one — proof that a single failed candidate must not blank.
aehr = recommend(99.10, 8.42, supertrend=105.67, supertrend_signal="SELL",
                 nearest_support=92.38, nearest_resistance=None,
                 mrc_zone="🔵 Near_Mean", mrc_r1=133.47, mrc_mean=96.40,
                 mrc_s1=59.29, mrc_s2=6.77, score=30,
                 structure="MIXED", regime="BULL")
check("AEHR: ladder finds a level where the pivot was too near",
      aehr.Rec_Dip is not None, aehr.Dip_Basis)
side_invariants("AEHR", aehr, 99.10)

# An add tier exists when a rung sits ABOVE the stop.
add = recommend(100.0, 2.0, supertrend=88.0, supertrend_signal="BUY",
                nearest_support=95.0, mrc_zone="Above_Mean", mrc_r1=112.0,
                mrc_s1=80.0, score=10)
check("a rung above the stop is tagged an add", add.Dip_Basis.startswith("add:"),
      add.Dip_Basis)
check("both tiers populate when both exist",
      add.Rec_Dip is not None and add.Rec_Dip2 is not None,
      f"{add.Rec_Dip} / {add.Rec_Dip2}")
check("the second dip is deeper than the first", add.Rec_Dip2 < add.Rec_Dip,
      f"{add.Rec_Dip} / {add.Rec_Dip2}")
check("a broken name still gets no bid",
      recommend(100.0, 2.0, nearest_support=95.0, mrc_zone="Below_Mean",
                mrc_mean=110.0, structure="BEARISH", regime="BEAR"
                ).Rec_Dip is None)

print("\n── trim in thirds ──")
lad = recommend(100.0, 2.0, supertrend=90.0, supertrend_signal="BUY",
                nearest_support=95.0, nearest_resistance=108.0,
                mrc_zone="Above_Mean", mrc_r1=112.0, mrc_r2=130.0, score=10)
check("three trim levels", None not in (lad.Rec_Trim, lad.Rec_Trim2, lad.Rec_Trim3),
      f"{lad.Rec_Trim}/{lad.Rec_Trim2}/{lad.Rec_Trim3}")
check("the ladder ascends", lad.Rec_Trim < lad.Rec_Trim2 < lad.Rec_Trim3,
      f"{lad.Rec_Trim}/{lad.Rec_Trim2}/{lad.Rec_Trim3}")
check("the far third reaches MRC_R2", abs(lad.Rec_Trim3 - 130.0) < 1e-9,
      repr(lad.Rec_Trim3))
check("the middle third splits the difference",
      abs(lad.Rec_Trim2 - (lad.Rec_Trim + lad.Rec_Trim3) / 2) < 0.01)
check("no first trim → no ladder at all",
      (lambda x: x.Rec_Trim is None and x.Rec_Trim2 is None and x.Rec_Trim3 is None)(
          recommend(100.0, 2.0, mrc_zone="Below_Mean", mrc_mean=None)))

print("\n── earnings ──")
from core.recommend import earnings_soon     # noqa: E402
check("a set alert reads as soon", earnings_soon({"Earnings_Alert": "🔴 EARNINGS SOON"}))
check("an empty alert does not", not earnings_soon({"Earnings_Alert": ""}))
check("a missing column does not", not earnings_soon({}))
check("a NaN read back from CSV does not",
      not earnings_soon({"Earnings_Alert": float("nan")}))

print("\n── trim follows the zone ──")
base = dict(atr=2.0, supertrend=90.0, supertrend_signal="BUY",
            nearest_support=95.0, nearest_resistance=108.0,
            mrc_r1=130.0, mrc_mean=112.0, mrc_s1=95.0, score=10)
check("Strong_OB trims at market (within 1 ATR)",
      (lambda x: x.Trim_Basis == "at-mkt" and x.Rec_Trim < 101.5)(
          recommend(100.0, mrc_zone="🔴 Strong_OB", **base)))
check("OB trims at the resistance pivot",
      (lambda x: x.Trim_Basis == "pivot" and abs(x.Rec_Trim - 108.0) < 1e-9)(
          recommend(100.0, mrc_zone="🟠 OB", **base)))
# R1 at 130 is 15 ATR away here, past the 8-ATR ceiling — so the level is the
# cap and the basis must SAY it is the cap, not claim to be R1.
capped = recommend(100.0, mrc_zone="Above_Mean", **base)
check("Above_Mean reaches for R1", capped.Trim_Basis.startswith("R1"),
      capped.Trim_Basis)
check("R1 beyond the ceiling is capped, and the basis admits it",
      capped.Trim_Basis == "R1~cap" and abs(capped.Rec_Trim - 116.0) < 1e-9,
      f"{capped.Trim_Basis} {capped.Rec_Trim}")
# ...and an R1 inside the ceiling is used untouched, with a clean basis.
near = recommend(100.0, mrc_zone="Above_Mean",
                 **{**base, "mrc_r1": 112.0})
check("R1 inside the ceiling is used as-is",
      near.Trim_Basis == "R1" and abs(near.Rec_Trim - 112.0) < 1e-9,
      f"{near.Trim_Basis} {near.Rec_Trim}")
# The branch that matters most: 39 of 128 names are below the mean, where R1
# is often +70% away and is not a plan.
below = recommend(100.0, mrc_zone="Below_Mean", **base)
check("Below_Mean trims at the MEAN, not at R1",
      below.Trim_Basis == "mean" and abs(below.Rec_Trim - 112.0) < 1e-9,
      f"{below.Trim_Basis} {below.Rec_Trim}")
check("Below_Mean trim is nearer than R1 would have been",
      below.Rec_Trim < 130.0)

print("\n── clamps ──")
# A pivot 40% below price must not become a 40% stop.
r = recommend(100.0, 2.0, supertrend=None, supertrend_signal="SELL",
              nearest_support=60.0, mrc_zone="Above_Mean", mrc_r1=130.0, score=10)
check("far pivot clamped to STOP_MAX_ATR (4 ATR = 92.0)",
      abs(r.Rec_Stop - 92.0) < 1e-9, repr(r.Rec_Stop))
check("...and the basis admits the clamp rather than claiming the pivot",
      r.Stop_Basis == "pivot~cap", r.Stop_Basis)
# A pivot 0.5% below price must not become a stop inside the noise.
r = recommend(100.0, 2.0, supertrend=None, supertrend_signal="SELL",
              nearest_support=99.5, mrc_zone="Above_Mean", mrc_r1=130.0, score=10)
check("near pivot pushed out to STOP_MIN_ATR (1.5 ATR = 97.0)",
      abs(r.Rec_Stop - 97.0) < 1e-9, repr(r.Rec_Stop))

print("\n── degraded inputs ──")
# A recent IPO: no MRC (needs 200 bars), no confirmed pivots. It must still
# get a stop — CBRS and PBLS would otherwise come back entirely blank.
ipo = recommend(42.0, 2.5, supertrend_signal="BUY", mrc_zone="N/A", score=55)
check("recent IPO still gets a volatility stop",
      ipo.Rec_Stop is not None and ipo.Stop_Basis == "vol", ipo.Stop_Basis)
check("recent IPO: no invented trim", ipo.Rec_Trim is None, repr(ipo.Rec_Trim))
side_invariants("IPO", ipo, 42.0)

check("no ATR → everything blank, nothing guessed",
      recommend(100.0, None) == Rec())
check("no price → everything blank", recommend(None, 2.0) == Rec())
check("NaN ATR → everything blank", recommend(100.0, float("nan")) == Rec())
check("zero price → everything blank", recommend(0.0, 2.0) == Rec())

print("\n── row adapter ──")
row = {"Ticker": "NVDA", "Current Price": 227.67, "ATR": 4.8,
       "Supertrend": 213.24, "Supertrend Signal": "BUY",
       "Nearest_Support": 221.09, "Nearest_Resistance": 229.98,
       "MRC_Zone": "Above_Mean", "MRC_R1": 234.55, "MRC_Mean": 212.09,
       "MRC_S1": 189.63, "ATH": 236.54, "Score_Weighted": 56,
       "Structure": "MIXED", "Regime": "BEAR"}
r = recommend_row(row)
check("recommend_row maps the CSV column names",
      r.Rec_Stop is not None and r.Rec_Trim is not None, r.basis_summary())
side_invariants("NVDA", r, 227.67)
# The Dashboard shows a LIVE price; recommending around a stale close while
# displaying a live quote would put the two columns visibly at odds.
r_live = recommend_row(row, price=240.0)
check("live-price override is honoured",
      r_live.Rec_Stop != r.Rec_Stop and r_live.Rec_Stop < 240.0,
      f"{r.Rec_Stop} vs {r_live.Rec_Stop}")
check("basis_summary has four slots", len(r.basis_summary().split("/")) == 4,
      r.basis_summary())

print(f"\n{'ALL PASS' if not FAILED else str(len(FAILED)) + ' FAILED: ' + ', '.join(FAILED)}")
raise SystemExit(1 if FAILED else 0)
