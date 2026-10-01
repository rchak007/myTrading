# Recommended levels — what the four numbers mean and how they are derived

`core/recommend.py` turns the indicators `jobStocksSignals` already computes
into four **price levels** per ticker, shown on the Dashboard between each
block's holdings and its live orders.

| | side | question it answers |
|---|---|---|
| **Rec_Stop** | SELL below | where is the thesis broken? |
| **Rec_Trim** | SELL above | where is strength worth rotating out of? |
| **Rec_Dip** | BUY below | where is weakness worth adding into? |
| **Rec_Breakout** | BUY above | where is strength worth adding into? |

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
| `TRIM_MIN_ATR` / `TRIM_MAX_ATR` | 1.0 / 8.0 | |
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

### Rec_Stop

```
floors = [ Supertrend                     if BUY mode and below price
         , Nearest_Support − 0.25·ATR ]   # UNDER the pivot, not on it
Rec_Stop = max(floors)                    # the highest real floor
clamp to [price − 4·ATR, price − 1.5·ATR]
no floors → price − 2.5·ATR               # basis "vol"
```

`max`, not `min`: among valid floors the nearest one is the one whose break
actually means something. Choosing a lower one just donates the difference.

The quarter-ATR cushion sits *below* the pivot because resting exactly on an
obvious swing low is where stop runs are aimed.

### Rec_Trim

The MRC zone **is** the stretch measurement, so it chooses the level:

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

### Rec_Dip

```
Structure == BEARISH and Regime == BEAR  →  blank ("broken")
base = max(Nearest_Support, MRC_S1)  below price
Rec_Dip = base + 0.25·ATR                 # just ABOVE the pivot
require Rec_Dip ≤ price − 1·ATR
```

The cushion sits *above* the pivot, the mirror of the stop sitting below it:
**one level, two sides, defined risk.** You want the fill before the crowd's
stops trigger, not after.

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

## 4. Fenced changes the mode

**A ticker cannot sensibly carry both a stop and a dip bid — they are opposite
intents.** Measured: the dip level landed *below* the stop on 36 of 128 names,
because Supertrend often sits above structural support. HOOD is the clean
case — Supertrend says exit at 104.92 while structural support is 101.71, so
bidding 102.50 while holding a stop at 104.92 is incoherent.

Chakravarti's own IA house rules settle it: **"Always Hedge", not always
stop.** A core holding is not stopped out; it is hedged and added to.

| | Rec_Stop | Rec_Dip |
|---|---|---|
| 🔒 **Fenced** (core) | advisory — a thesis-break line, not an order | **primary**, never suppressed |
| not fenced (trade) | **primary** | suppressed when it falls below the stop |

Fencing is recorded per (ticker, account), but these levels are a property of
the **stock**, so a ticker fenced in *any* account is treated as core. One pair
of rows per block, not per account — unlike the coverage flags above them,
which are per (ticker, account) because an order exists in exactly one.

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
| `R1` / `mean` | an MRC band |
| `ATH` | the all-time high, having passed the 5× sanity check |
| `vol` | no structure survived; pure ATR distance |
| `at-mkt` | already maximally stretched |
| `~cap` suffix | **a clamp moved the level.** `R1~cap` reached for R1 and got the 8-ATR ceiling instead |
| `downtrend` / `score<50` / `broken` / `too-near` / `below-stop` | why it is blank |

The `~cap` suffix exists because a level reported as `pivot` that is really
4 ATR of pure volatility is precisely the kind of plausible-looking wrong
answer this system keeps getting bitten by.

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

- **No position sizing.** These are levels, not quantities.
- **No execution.** Nothing here places, cancels or modifies an order.
- **Equity only**, like the rest of the order path. An options overlay would
  need its own treatment — see the note in `PROJECT_PLAN.md` §6.
- **No backtest.** The parameters are reasoned from the indicator definitions
  and sanity-checked against the live distribution; they have not been fitted
  to realised returns. Treat them as a starting calibration.
