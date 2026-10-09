#!/usr/bin/env python3
"""
core/structure.py
=================
Is this chart forming a P, a b, or a D — and has it broken out?

The three market-profile shapes, read off daily bars:

    P    a strong leg UP, then balance at the top        ← what we buy
         buyers absorbed the supply and did not give the gain back

    b    a strong leg DOWN, then balance at the bottom
         the mirror image. Balance after a fall is not accumulation.

    D    balance with no leg into it
         fair value, two-sided, nothing to lean on

PURE. Bars in, a verdict out. No Schwab, no Sheets, no files, no clock — so
the whole thing is testable on Pi 2, which has none of those. Same contract as
core/recommend.py.

─────────────────────────────────────────────────────────────────────────────
STATELESS BY CONSTRUCTION, AND THAT IS THE WHOLE TRICK

    The design sketch had a state machine: WATCHING → P_FORMED → TRIGGERED,
    with the boundary written back into the sheet and an arrow returning to
    WATCHING when the P failed. It would have worked, and it would have been
    one more thing that can drift out of step with reality.

    This needs none of it. Ask the question fresh every cycle:

        do the N bars BEFORE today form a balance,
        was the leg into that balance strongly up,
        and did TODAY close above the balance's high?

    Eleven bars answer it. A P that failed stops being a P on its own, with no
    arrow to remember — the window slides and the structure is simply gone. A
    stored boundary is the thing that fires weeks later on something
    unrelated; there is no stored boundary here.

    The states below are therefore a DESCRIPTION of today, not a memory:
    recompute and you get the same answer, which is the property a sheet view
    wants.

WHY THE BOUNDARY EXCLUDES TODAY
    `close > max(high of the last N bars)` including today is nearly
    impossible — it needs the close to be the high. The balance is the N
    COMPLETED bars before today; today is the bar that either breaks it or
    does not. Off-by-one here does not fail loudly, it just never signals.

WHY A LATE BREAKOUT IS NOT A SIGNAL
    Break out on Monday, run all week, and by Thursday the window contains
    the breakout bar and is no longer a balance. Thursday reports WATCHING,
    not TRIGGERED. That is deliberate: this buys the day the range gives way,
    not a week into the move after the edge is gone.

EVERYTHING IN ATR
    A 1.50 range is balance on AVGO at $360 and a wild swing on JOBY at $5.75.
    Thresholds are multiples of ATR throughout, which is what lets one set of
    numbers cover a 135-ticker list.
"""
from __future__ import annotations

# ── the four numbers that define a P ────────────────────────────────
BALANCE_BARS = 5          # how many bars of sideways counts as balance
BALANCE_MAX_ATR = 1.5     # ...and how tight, high-to-low, in ATR
LEG_BARS = 5              # the move INTO the balance, measured over this many
LEG_MIN_ATR = 2.0         # ...and how big it must be, in ATR
BREAK_MIN_ATR = 0.0       # how far past the boundary a close must settle

# Bars needed before any verdict is possible.
MIN_BARS = LEG_BARS + BALANCE_BARS + 1

NO_DATA = "NO_DATA"
WATCHING = "WATCHING"       # no P right now
P_FORMED = "P_FORMED"       # a P sits here, boundary known, not yet broken
TRIGGERED = "TRIGGERED"     # today closed above the boundary


class Structure:
    """What the chart is doing, as of the last bar given.

    `state` is the one callers act on. The rest explains it, and goes on the
    sheet so a row that has been waiting three weeks can say WHY.
    """

    __slots__ = ("state", "shape", "boundary", "floor", "leg_atr",
                 "range_atr", "break_atr", "why")

    def __init__(self, state, shape=None, boundary=None, floor=None,
                 leg_atr=None, range_atr=None, break_atr=None, why=""):
        self.state = state
        self.shape = shape            # "P" | "b" | "D" | None
        self.boundary = boundary      # the balance high — the level to break
        self.floor = floor            # the balance low — where the P fails
        self.leg_atr = leg_atr        # signed, in ATR
        self.range_atr = range_atr    # balance height, in ATR
        self.break_atr = break_atr    # how far today closed past the boundary
        self.why = why

    @property
    def armed(self) -> bool:
        """A P is sitting here, waiting. Not a reason to buy yet."""
        return self.state == P_FORMED

    @property
    def fired(self) -> bool:
        return self.state == TRIGGERED

    def __repr__(self):
        bits = [self.state]
        if self.shape:
            bits.append(f"shape={self.shape}")
        if self.boundary is not None:
            bits.append(f"boundary={self.boundary:.2f}")
        if self.leg_atr is not None:
            bits.append(f"leg={self.leg_atr:+.2f}ATR")
        if self.range_atr is not None:
            bits.append(f"range={self.range_atr:.2f}ATR")
        return "Structure(" + ", ".join(bits) + ")"


def _f(v):
    """A float, or None. Bars arrive from pandas, CSVs and the Schwab API, so
    NaN and '' are the normal case, not corruption."""
    try:
        if v is None:
            return None
        x = float(v)
        return None if x != x else x          # NaN
    except (TypeError, ValueError):
        return None


def classify(highs, lows, closes, atr, *,
             balance_bars: int = BALANCE_BARS,
             balance_max_atr: float = BALANCE_MAX_ATR,
             leg_bars: int = LEG_BARS,
             leg_min_atr: float = LEG_MIN_ATR,
             break_min_atr: float = BREAK_MIN_ATR) -> Structure:
    """The structure as of the LAST bar in the sequences given.

    The last bar is "today": the one that breaks the balance or does not. The
    `balance_bars` before it are the balance; the `leg_bars` before THAT are
    the leg into it.

    Sequences must be in chronological order, oldest first — the order
    yfinance, Schwab and every CSV here already use. Pass more bars than
    needed and the extras are ignored.
    """
    atr = _f(atr)
    if not atr or atr <= 0:
        return Structure(NO_DATA, why="no usable ATR")

    need = leg_bars + balance_bars + 1
    h = [_f(x) for x in highs][-need:]
    lo = [_f(x) for x in lows][-need:]
    c = [_f(x) for x in closes][-need:]
    if len(h) < need or len(lo) < need or len(c) < need:
        return Structure(NO_DATA,
                         why=f"need {need} bars, have {min(len(h), len(lo), len(c))}")
    if any(x is None for x in h + lo + c):
        return Structure(NO_DATA, why="gaps in the bars")

    today = c[-1]

    # The balance: the completed bars before today. Today is excluded on
    # purpose — see the module docstring.
    bal_h, bal_l = h[-1 - balance_bars:-1], lo[-1 - balance_bars:-1]
    boundary, floor = max(bal_h), min(bal_l)
    rng_atr = (boundary - floor) / atr

    # The leg: close-to-close over the bars leading into the balance.
    leg_start = c[-1 - balance_bars - leg_bars]
    leg_end = c[-1 - balance_bars]
    leg_atr = (leg_end - leg_start) / atr

    is_balance = rng_atr <= balance_max_atr
    if leg_atr >= leg_min_atr:
        shape = "P"
    elif leg_atr <= -leg_min_atr:
        shape = "b"
    else:
        shape = "D"

    if not is_balance:
        return Structure(WATCHING, shape=None, leg_atr=leg_atr,
                         range_atr=rng_atr,
                         why=f"not balanced — {rng_atr:.1f} ATR over "
                             f"{balance_bars} bars, needs ≤{balance_max_atr}")

    # Measured for EVERY balance, not just a P. A b or a D that clears its
    # own boundary is the control group: if their forward returns match the
    # P's, then the leg is not information and the shape test is decoration.
    break_atr = (today - boundary) / atr

    common = dict(shape=shape, boundary=boundary, floor=floor,
                  leg_atr=leg_atr, range_atr=rng_atr, break_atr=break_atr)

    if shape != "P":
        word = ("a leg DOWN into the balance" if shape == "b"
                else "no real leg into the balance")
        return Structure(WATCHING,
                         why=f"balanced, but {shape} — {word} "
                             f"({leg_atr:+.1f} ATR, needs ≥{leg_min_atr})",
                         **common)

    if break_atr > break_min_atr:
        return Structure(TRIGGERED,
                         why=f"P broke — close {today:.2f} is {break_atr:.2f} "
                             f"ATR above the {boundary:.2f} balance high",
                         **common)

    return Structure(P_FORMED,
                     why=f"P formed on a {leg_atr:+.1f} ATR leg — needs a "
                         f"close above {boundary:.2f} "
                         f"(now {today:.2f}, {break_atr:+.2f} ATR)",
                     **common)


def classify_frame(df, atr=None, **kw) -> Structure:
    """`classify` for a DataFrame with High / Low / Close columns.

    ATR comes from an `ATR` column if not given — which is what
    core/indicators.compute_supertrend already exports.
    """
    if df is None or getattr(df, "empty", True):
        return Structure(NO_DATA, why="no bars")
    cols = {str(x).strip().lower(): x for x in df.columns}
    try:
        h, lo, c = cols["high"], cols["low"], cols["close"]
    except KeyError:
        return Structure(NO_DATA, why=f"no High/Low/Close in {list(df.columns)}")
    if atr is None:
        if "atr" not in cols:
            return Structure(NO_DATA, why="no ATR column and none given")
        atr = df[cols["atr"]].iloc[-1]
    return classify(df[h].tolist(), df[lo].tolist(), df[c].tolist(), atr, **kw)


def shares_for(budget, price) -> int:
    """Whole shares a dollar budget buys. 0 if it does not reach one share.

    Whole only, and never rounded up: the budget is a ceiling Chakravarti set,
    and a fractional share is not a thing the order path can place.
    """
    b, p = _f(budget), _f(price)
    if not b or not p or b <= 0 or p <= 0:
        return 0
    return int(b // p)
