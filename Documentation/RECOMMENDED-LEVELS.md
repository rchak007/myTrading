# Recommended levels — what the four numbers mean and how they are derived

`core/recommend.py` turns the indicators `jobStocksSignals` already computes
into four **price levels** per ticker, shown on the Dashboard between each
block's holdings and its live orders.

| | ① sell ⅓ / buy | ② sell ⅓ / buy | ③ **ALL OUT** / final ⅓ |
|---|---|---|---|
| **STOP** | `Rec_Stop` close-confirmed | `Rec_Stop2` close-confirmed | `Rec_Stop_Hard` **touch, full exit** |
| **TRIM** | `Rec_Trim` | `Rec_Trim2` | `Rec_Trim3` |
| **DIP** | `Rec_Dip` add or re-entry | `Rec_Dip2` the deeper one | — |
| **BREAKOUT** | `Rec_Breakout` | — | — |

**Read the rows top to bottom as a scale-out: a third at ① , a third at ② ,
everything at ③ .** The exception is the whole point — see below.

They sit directly under `Has_Stop` / `Has_Trim` / `Has_Dip` / `Has_Breakout` on
purpose. **The flag says whether an order exists; the level says where one
would go.** A yellow `N` beside a blank level means nothing is arranged *and*
nothing is recommended; a yellow `N` above a number is the gap worth closing.

**They place nothing.** These are read-only advisory numbers. The Orders tab
remains the only path to execution, and typing one of these into it is a
deliberate human act.

---

## 1. Everything is measured in ATR

A 5% stop on NVDA (ATR ≈ 2.1% of price) and a 5% stop on FCEL (ATR ≈ 5.0%) are
not the same trade. Percent hides that; ATR is the unit the market moves in.

`compute_supertrend()` has always calculated a Wilder ATR(10) internally and
thrown it away. It is now returned as an `ATR` column and flows through to
`stocks_signals.csv`. Every distance below is a multiple of it.

The tunables, all in ATR multiples, live at the top of `core/recommend.py`:

| | | why |
|---|---|---|
| `STOP_MIN_ATR` | 1.5 | tighter than this is inside daily noise |
| `STOP_MAX_ATR` | 4.0 | wider than this is not a stop, it is a hope |
| `STOP_FALLBACK_ATR` | 2.5 | the pure-volatility stop when no structure survives |
| `TRIM_MIN_ATR` / `TRIM_MAX_ATR` | 1.0 / 8.0 | ceiling for the FIRST trim |
| `TRIM3_MAX_ATR` | 16.0 | ceiling for the final third — it is allowed to reach |
| `HARD_STOP_MULT` | 1.5 | the disaster stop, as a multiple of the soft distance |
| `HARD_STOP_MIN_ATR` / `HARD_STOP_MAX_ATR` | 3.0 / 8.0 | never nearer, never further |
| `STOP_LADDER_SPAN_ATR` | 2.0 | **the ladder needs room to be a ladder** — see below |
| `STOP_RUNG_GAP_ATR` | 0.6 | minimum clearance for the middle rung |
| `DIP_GAP_ATR` | 0.5 | a re-entry must sit meaningfully below the stop |
| `DIP_MIN_ATR` | 1.0 | nearer than this is not a dip |
| `BRK_MIN_ATR` | 0.5 | |
| `PIVOT_CUSHION_ATR` | 0.25 | how far off a pivot to sit, either side |

---

## 2. The three traps in the data

Measured across all 128 names on 2026-10-01. Each one silently corrupts an
obvious implementation.

**`ATH` is corrupt on 15 of 128 names.** Reverse splits are not adjusted, so
FCEL's all-time high reads **234,900** against a price of 16.81; RCAT reads
990,000 and ABTC 2,179,500. Rejected above `ATH_MAX_MULTIPLE` (5×) price.

**`Supertrend` is unusable as a stop on half the book.** 64 of 128 are in SELL
mode, where the line sits *above* price and is resistance, not support. Only
used when `Supertrend Signal == BUY` **and** the line is genuinely below price.

**`MRC_S2` goes negative** (FCEL −1.03, MSTX −0.42) and `MRC_R1` sits a median
**+19.9%** away — up to +74% on depressed names. Excellent as a stretch
marker, useless as an order level without zone gating.

---

## 3. The formulas

### Rec_Stop — two stops, two mechanisms

**A resting Schwab stop is a TOUCH trigger.** One wick takes you out at the
worst price of the day, and you were right about the level. The Orders tab
engine is CLOSE-triggered by construction, which wicks cannot reach.

But close-confirmation is not free: it accepts **gap risk**. The night
something halves, you sell at the next open, far below your level. So both, at
two distances:

| | mechanism | where it goes | size |
|---|---|---|---|
| `Rec_Stop` | close-confirmed | **Orders tab**, `SELL` / `CLOSE BELOW` | ⅓ |
| `Rec_Stop2` | close-confirmed | **Orders tab**, `SELL` / `CLOSE BELOW` | ⅓ |
| `Rec_Stop_Hard` | **touch** | **a resting Schwab `STOP`** | **everything left** |

**Why scale out of a stop at all.** One level forces a binary decision on a
position you do not really want to leave, and stops get whipsawed. The
trade-off is explicit: in a *real* decline scaling costs you — thirds at
−7.5%, −10.5%, −13.1% instead of all at −7.5%. In a *whipsaw* it saves you —
only a third gone before the recovery. For a long-horizon book of quality
names whipsaws are the commoner event, so scaling wins on average.

**The last rung is different and must stay different.** ① and ② are
close-confirmed, so wicks cannot reach them. ③ is a resting Schwab stop — a
touch trigger, and a **full exit**. Leaving a third on through a crash because
"the ladder says a third at a time" is exactly the wrong lesson to draw from
scaling out.

**The ladder needs room to be a ladder.** Proportional growth alone collapses
when the soft stop is tight: 1.5× of a 1.5-ATR stop is 2.25 ATR, which left
the three rungs 0.75 ATR apart — TSLA came out **338.98 / 330.53 / 322.07**,
all three of which trigger in the same two-day move. That is one stop with
extra steps, and Chakravarti spotted it on sight (2026-10-01).

So the hard stop takes the **widest** of three terms:

```
hard_distance = max( soft_distance × 1.5
                   , soft_distance + 2·ATR     ← guarantees the span
                   , 3·ATR )          capped at 8·ATR
```

Measured after the fix: gaps run 0.60–1.39 ATR, median **1.00** — a full day's
range between rungs, so price can plausibly stop between them. TSLA now reads
339.01 / 327.75 / 316.47 (−4.8% / −7.9% / −11.1%).

`Rec_Stop2` prefers a **real shelf** between the other two — the next
structural level down is where a decline actually pauses — and falls back to
the midpoint when there is none. It is dropped entirely unless all three rungs
are at least 0.4 ATR apart; three stops inside one ATR is one stop pretending
to be a plan.

```
floors = [ Supertrend                     if BUY mode and below price
         , Nearest_Support − 0.25·ATR ]   # UNDER the pivot, not on it
Rec_Stop = max(floors)                    # the highest real floor
clamp to [price − 4·ATR, price − 1.5·ATR]
no floors → price − 2.5·ATR               # basis "vol"

Rec_Stop_Hard = price − max(1.5 × soft_distance, 3·ATR)
clamp to price − 8·ATR
```

`max`, not `min`: among valid floors the nearest one is the one whose break
actually means something. Choosing a lower one just donates the difference.

The quarter-ATR cushion sits *below* the pivot because resting exactly on an
obvious swing low is where stop runs are aimed.

The hard stop is **proportional, not fixed**: a name whose technical level is
already 4 ATR out does not need another 4 ATR on top, and one with a tight
1.5 ATR stop needs more room than that before "disaster" is the right word.
Measured across the book: soft stops sit a median −7.5% from price, hard stops
−13.1%.

**Neither is a size.** At 1.5–4 ATR these get tagged by ordinary noise several
times a year on the volatile names.

### Rec_Trim — three levels, not one

**A single trim level forces an all-or-nothing decision, and you will always
feel you sold too early.** Three levels capture the move.

The MRC zone **is** the stretch measurement, so it chooses the *first* level:

| `MRC_Zone` | level | basis |
|---|---|---|
| `Strong_OB` (≥ R2) | `price + 0.5·ATR` | `at-mkt` — maximally stretched, go now |
| `OB` (R1→R2) | `Nearest_Resistance` | `pivot` |
| `Above_Mean` / `Near_Mean` | `MRC_R1` | `R1` |
| `Below_Mean` / `OS` / `Strong_OS` | **`MRC_Mean`** | `mean` |
| `N/A` (MRC warming up) | `Nearest_Resistance` | `pivot` |

**The below-mean branch is the one that matters.** It covers 39 of 128 names,
and for those R1 is not a plan — FCEL at 16.81 has R1 at 29.26, +74% away.
Trimming there is a wish. Reverting to the mean is the realistic first exit.

The other two thirds follow from it:

```
Rec_Trim3 = MRC_R2 if it is sane and beyond Rec_Trim, else price + 16·ATR
Rec_Trim2 = midpoint of Rec_Trim and Rec_Trim3     # evenly spaced
```

Dropped entirely when `Rec_Trim3` would land within half an ATR of
`Rec_Trim` — a ladder that tight is noise, not a plan.

### Rec_Dip — a ladder, in two tiers

The first version vetoed any dip below the stop, reasoning that bidding under
your own stop is incoherent. **That was wrong.** The stop protects the shares
you *hold*; the dip deploys *fresh* capital at a better price. Different money.

Measured on 2026-10-01, that veto discarded real levels: PLTR had a pivot at
164.55 under a 171.78 stop, CRDO had `MRC_S1` at 157.30 under a 169.09 stop.
Both are exactly where you want a bid.

```
Structure == BEARISH and Regime == BEAR  →  blank ("broken")

rungs: Nearest_Support, MRC_S1, MRC_S2, MRC_Mean   (+0.25·ATR each)
       price − 2/3/4·ATR                           (volatility fallback)
keep only rungs ≤ price − 1·ATR

add tier      highest STRUCTURAL rung above the stop        → "add:…"
re-entry tier highest rung below (HARD stop − 0.5·ATR)      → "re:…"
Rec_Dip  = the add tier if one exists, else the re-entry
Rec_Dip2 = the other one, or the next rung deeper
```

The cushion sits *above* each pivot, the mirror of the stop sitting below it:
**one level, two sides, defined risk.** You want the fill before the crowd's
stops trigger, not after.

**Volatility rungs are a fallback, never a competitor.** A line at
`price − 2·ATR` is not a level anyone else is watching, so it must not outrank
a real pivot for being nearer — which it did on PLTR, where 174.86 beat the
164.55 swing low until this was fixed.

**The add tier requires structure, with no volatility fallback at all.**
Adding at a vol rung that happens to sit above the stop is the worst of both:
you buy, price keeps falling, and 1 ATR later the stop takes out the whole
position including what you just added. If nothing structural holds above the
stop, the honest answer is a re-entry, not an add.

**A re-entry is measured from the HARD stop, not the soft one**, because with
a laddered stop you are not fully out until the last rung. Measuring from the
soft stop would put a bid at a level where you still hold two thirds — and
worse, on the same shelf as your own second stop. Measured: **zero** dips land
inside the stop ladder.

Result: `Rec_Dip` went from **36 of 128 populated to 96** — 47 adds, 49
re-entries. The remaining 32 are all `broken`, which is deliberate.

### Rec_Breakout

```
Supertrend Signal == SELL  →  blank ("downtrend")
Score_Weighted < 50        →  blank ("score<50")
Rec_Breakout = Nearest_Resistance + 0.25·ATR
   else ATH + 0.25·ATR     only if price < ATH < 5 × price
floor price + 0.5·ATR
```

`Score_Weighted ≥ 50` is the system's own "consider" bar — it is what
`Regime_Action` already keys off ("🟢 HOLD → consider (score≥50)"). Reusing it
beats inventing a second threshold that can drift from the first.

**This fires on roughly 5% of names, and that is correct.** 64 are in a
downtrend and 58 score under 50. A breakout add on FCEL (score 2, 🔴 EXIT)
would be noise wearing a price.

---

## 4. Fenced changes the mode — and fenced is NOT a keeper flag

**Fencing means "when this sells, keep the proceeds earmarked to this ticker
so I can pick it up later."** It *presumes* a sale. It says nothing about
wanting to hold through a decline — Chakravarti's own words, 2026-10-01:

> "fenced still does not mean i want to keep when its going down. It just
> means later i might pick it up."

That matters here because the dip level lands *below* the stop on 36 of 128
names (Supertrend often sits above structural support). HOOD is the clean
case: Supertrend says exit at 104.92 while structural support is 101.71.

| | Rec_Stop | Rec_Dip |
|---|---|---|
| 🔒 **fenced** | a real level | **kept** — it is the re-entry half of a round trip |
| not fenced | a real level | suppressed when it falls below the stop |

For a fenced ticker the two are not contradictory, they are **one rotation**:
out at the stop, back in at the dip, with the money already set aside to do
it. That is LILO. Unfenced there is no earmarked re-entry money, so the same
pair really is incoherent — you would be buying at the level that just told
you to sell — and the dip is dropped.

Fencing is recorded per (ticker, account); these levels are a property of the
**stock**, so a ticker fenced in *any* account is treated as fenced here. One
pair of rows per block, not per account — unlike the coverage flags above
them, which are per (ticker, account) because an order exists in exactly one.

### There is no keeper concept in this system

Nothing in `recommend.py` means "hold through a drawdown". Every held ticker
gets a `Rec_Stop`, and the module emits **no size** — a stop level says where
the thesis is broken, not how much to sell. At 1.5–4 ATR these levels get
tagged by ordinary noise several times a year on the volatile names, so
reading every one as "sell everything" would churn exactly the positions worth
keeping.

If a genuine never-stop-out list is wanted it has to be a **separate flag**,
because fencing already means something else. Open item in
[PROJECT_PLAN.md](PROJECT_PLAN.md) §6.

---

## 5. Reading the basis

The label column of the value row carries the ATR and a four-slot basis in
fixed `stop/trim/dip/breakout` order:

```
ATR 4.81 (2.1%) · pivot/R1/pivot/pivot
ATR 3.78 (3.3%) · ST/R1~cap/below-stop/score<50
```

**Without it the numbers are unfalsifiable** — you cannot tell a structural
stop from a volatility fallback, and the two deserve different confidence.

| basis | meaning |
|---|---|
| `ST` | the Supertrend line |
| `pivot` | an HHLL structural level, offset by the cushion |
| `R1` / `mean` / `S1` / `S2` | an MRC band |
| `add:…` | the dip sits ABOVE the stop — you still hold, this is an add |
| `re:…` | the dip sits BELOW the stop — you were stopped out, this is the way back |
| `2ATR` / `3ATR` / `4ATR` | a volatility rung. Nothing structural qualified |
| `ATH` | the all-time high, having passed the 5× sanity check |
| `vol` | no structure survived; pure ATR distance |
| `at-mkt` | already maximally stretched |
| `~cap` suffix | **a clamp moved the level.** `R1~cap` reached for R1 and got the 8-ATR ceiling instead |
| `downtrend` / `score<50` / `broken` / `too-near` / `below-stop` | why it is blank |

The `~cap` suffix exists because a level reported as `pivot` that is really
4 ATR of pure volatility is precisely the kind of plausible-looking wrong
answer this system keeps getting bitten by.

---

## 5a. Earnings

`Earnings_Alert` is computed by `data/stock_scoring.get_earnings_alert` and was
previously unused. It now appears in two places:

- **a banner at the top of the Dashboard**, listing every *held* ticker with
  earnings inside the window. Buried in a block it would be found only by
  someone already scrolled to that ticker, and the point is to be seen before
  deciding anything.
- **in the basis line of that ticker's block.**

**Earnings is the one event that defeats a close-confirmed stop** — the gap
happens before any close can confirm anything, so `Rec_Stop` cannot protect
you through it and `Rec_Stop_Hard` is what actually catches it.

It does **not** move any level today. Suppressing the soft stop into earnings
is a judgement call, and the warning puts it in front of the human rather than
making it silently — open item in [PROJECT_PLAN.md](PROJECT_PLAN.md) §6.

---

## 6. Failing closed

- Every level is **`None`, never `0.0`**, when it cannot be computed. A zero in
  a price cell eventually gets read as a price.
- Every output is **re-checked against the side of price it must be on** before
  being returned.
- **Recent IPOs still get a stop.** CBRS and PBLS have no 200-bar MRC and no
  confirmed pivots; they fall through to the volatility stop rather than coming
  back empty.
- **No ATR, or no price → everything blank.** Nothing is guessed.
- Levels are **tick-rounded** (`tick_round`): $0.01 above $1.00, $0.0001 below,
  per SEC Rule 612. These get read straight into the Orders tab, and a
  sub-penny limit on a $40 stock is rejected outright.

---

## 7. Where it lives

| file | role |
|---|---|
| `core/recommend.py` | the whole thing. **Pure** — prices in, prices out, no I/O, no clock |
| `core/indicators.py` | exports `ATR` from `compute_supertrend` |
| `data/stocks.py` | carries `ATR` into the signals frame on both the scoring and non-scoring paths |
| `orders_sheet.py` | `_rec_rows()` renders the two Dashboard lines; `build_dashboard(signals_df=...)` |
| `test_recommend.py` | every case is a real trap from the live data, not a hypothetical |

Being pure is what makes it testable on Pi 2, which has no Schwab credentials
and no Sheets write access.

**`attach(signals_df, fenced)`** adds the eight columns to a whole frame. It is
written and tested but deliberately **not wired into the published CSV** —
`stocks_signals.csv` feeds Streamlit, and widening that schema is a separate
decision.

---

## 8. What this does not do

- **No position sizing.** These are levels, not quantities — and this is the
  real missing half. A stop level without a size is not a risk decision. AEHR
  carries an **8.3% ATR**: a 1.5–4 ATR stop there is a 12–33% stop, which is a
  position-size problem that no amount of level-tuning fixes.
- **No execution.** Nothing here places, cancels or modifies an order.
- **Equity only**, like the rest of the order path. An options overlay would
  need its own treatment — see the note in `PROJECT_PLAN.md` §6.
- **No backtest.** The parameters are reasoned from the indicator definitions
  and sanity-checked against the live distribution; they have not been fitted
  to realised returns. Treat them as a starting calibration.
