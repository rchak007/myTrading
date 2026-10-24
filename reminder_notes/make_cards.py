#!/usr/bin/env python3
"""
make_cards.py
=============
Regenerates the info-note cards WE own, which ride at the end of every
reminder email:

    03-pbd-structure.png   P / b / D market structure + the breakout rule
    04-buy-cheap.png       the one discipline his own data says he breaks
    05-catch-winners.png   a pointer to the 17-step doc, so it is not lost
    06-benter-method.png   a pointer to the expected-value doc
    07-vix-regime.png      the VIX fear-gauge bands, with what the data says
    08-market-cycle.png    the four-phase Wyckoff cycle, and the tool per phase

    .venv/bin/python reminder_notes/make_cards.py

KEPT AS A SCRIPT, NOT JUST A PNG. The other two cards are screenshots of
someone else's slides and cannot be edited. This one is ours, so the source
of truth is code: change a line here and regenerate, rather than hunting for
whatever made the image.

Content is the PbD method (Tom Vorwald, via Patrick Nill) plus the breakout
rule Chakravarti asked to have rolled in. Drawn rather than written out,
because the whole point of the method is the SHAPE.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / "03-pbd-structure.png"
OUT_CHEAP = HERE / "04-buy-cheap.png"
OUT_WINNERS = HERE / "05-catch-winners.png"
OUT_BENTER = HERE / "06-benter-method.png"
OUT_VIX = HERE / "07-vix-regime.png"
OUT_CYCLE = HERE / "08-market-cycle.png"
W, H = 1600, 1120

BG = (13, 13, 16)
WHITE = (255, 255, 255)
MUTED = (156, 163, 175)
BLUE = (59, 130, 246)
RED = (239, 68, 68)
GREEN = (74, 222, 128)
AMBER = (250, 204, 21)
TEAL = (45, 212, 191)
VIOLET = (167, 139, 250)
LINE = (55, 55, 62)

F = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
font = lambda sz, bold=False: ImageFont.truetype(FB if bold else F, sz)


def shape(d, x, y, w, h, kind, colour):
    """A little price path. P rises then balances high, b falls then balances
    low, D just oscillates — the glyph IS the lesson."""
    mid, lo, hi = y + h // 2, y + h - 10, y + 10
    if kind == "P":
        d.line([(x, lo), (x + w * 0.45, hi)], fill=colour, width=7)
        d.rectangle([x + w * 0.45, hi - 6, x + w, hi + 46], outline=colour, width=4)
    elif kind == "b":
        d.line([(x, hi), (x + w * 0.45, lo)], fill=colour, width=7)
        d.rectangle([x + w * 0.45, lo - 46, x + w, lo + 6], outline=colour, width=4)
    else:                                            # D — balance, no trend
        d.rectangle([x, mid - 30, x + w, mid + 30], outline=colour, width=4)
        pts = [(x + w * f, mid + (26 if i % 2 else -26))
               for i, f in enumerate((0.08, 0.28, 0.48, 0.68, 0.9))]
        d.line(pts, fill=colour, width=5)


def main() -> Path:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    d.text((W // 2, 62), "P · b · D  —  Market Structure",
           font=font(54, True), fill=WHITE, anchor="mm")
    d.text((W // 2, 124), "strong move  →  balance  →  next opportunity",
           font=font(30), fill=MUTED, anchor="mm")

    rows = [("P", BLUE, "Strong move UP, then sideways near the top",
             "Buyers pushed it higher. Now consolidating."),
            ("b", RED, "Strong move DOWN, then sideways near the bottom",
             "Sellers pushed it lower. Now consolidating."),
            ("D", MUTED, "Back and forth inside a range",
             "Buyers and sellers balanced. No edge yet.")]
    y = 196
    for letter, colour, what, means in rows:
        d.text((96, y + 58), letter, font=font(76, True), fill=colour, anchor="mm")
        shape(d, 168, y + 8, 180, 100, letter, colour)
        d.text((400, y + 34), what, font=font(31, True), fill=WHITE)
        d.text((400, y + 78), means, font=font(27), fill=MUTED)
        y += 132

    d.line([(90, y + 14), (W - 90, y + 14)], fill=LINE, width=2)
    y += 56

    d.text((90, y), "THE BREAKOUT RULE", font=font(32, True), fill=WHITE)
    y += 58
    for colour, head, tail in (
            (GREEN, "Breaks ABOVE the range and HOLDS", "follow it up"),
            (RED, "Breaks BELOW the range and HOLDS", "follow it down"),
            (MUTED, "Stays inside", "nothing to follow yet")):
        d.text((100, y), "●", font=font(26), fill=colour)
        d.text((140, y), head, font=font(29, True), fill=WHITE)
        d.text((140 + d.textlength(head, font=font(29, True)) + 18, y),
               f"→  {tail}", font=font(29), fill=colour)
        y += 50

    y += 26
    d.rectangle([90, y, W - 90, y + 132], outline=(90, 70, 20), width=3)
    d.text((118, y + 26), "THE CRUCIAL PART IS  “AND HOLDS”",
           font=font(29, True), fill=(250, 204, 21))
    d.text((118, y + 70),
           "Price can poke outside the range and reverse — a false breakout.",
           font=font(26), fill=MUTED)
    d.text((118, y + 100),
           "Wait for a CLOSE outside, then a retest where the boundary holds.",
           font=font(26), fill=MUTED)

    # Placed relative to the box, not to H — the footer and the warning
    # overlapped the first time because one was measured from the bottom and
    # the other from the flow.
    d.text((W // 2, y + 184), "PbD method — Tom Vorwald, via Patrick Nill",
           font=font(22), fill=(90, 95, 105), anchor="mm")

    img.save(OUT, optimize=True)
    return OUT


def buy_cheap() -> Path:
    """A deliberately small tile. One instruction, one reason.

    It earns its place because it is the discipline his OWN record says he
    breaks: MU 2026 was trimmed at $379 and bought back at $444, $483, $990
    and $1,038 — in a year the stock went 3.1x, the churn still cost $3,487
    against simply holding. The level was always on the sheet.
    """
    w, h = 1600, 430
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 14, h], fill=GREEN)

    d.text((78, 104), "BUY CHEAP !", font=font(96, True), fill=GREEN, anchor="lm")
    d.text((78, 186), "Wait for Rec_Dip. The level is already on the sheet.",
           font=font(36), fill=WHITE, anchor="lm")

    d.line([(78, 244), (w - 78, 244)], fill=LINE, width=2)
    d.text((78, 292),
           "Chasing the re-entry is what costs you — not the trim.",
           font=font(30, True), fill=(250, 204, 21), anchor="lm")
    d.text((78, 344),
           "MU 2026: trimmed at $379, bought back at $444 · $483 · $990 · $1,038.",
           font=font(27), fill=MUTED, anchor="lm")
    d.text((78, 386),
           "The stock went 3.1x that year and the churn still cost $3,487.",
           font=font(27), fill=MUTED, anchor="lm")

    img.save(OUT_CHEAP, optimize=True)
    return OUT_CHEAP


def catch_winners() -> Path:
    """A POINTER, not a summary. Chakravarti's reason, 2026-10-12: *"so it
    keeps me not losing track of it."*

    A 17-step framework does not fit on a tile and should not be squeezed onto
    one — the doc exists for that. So the tile carries three things: the name
    of the file, where it lives, and the one line the whole framework
    compresses to. Enough to recognise it and go read it.

    It is last of the five on purpose. The cards above are things to DO while
    the market is open; this is homework for before you buy a new name.
    """
    w, h = 1600, 470
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 14, h], fill=TEAL)

    d.text((78, 96), "CATCH WINNERS EARLY", font=font(76, True), fill=TEAL,
           anchor="lm")
    d.text((78, 168), "See the doc before sizing into a NEW name.",
           font=font(36), fill=WHITE, anchor="lm")

    d.text((78, 232), "HOW-TO-CATCH-WINNING-STOCK.md", font=font(38, True),
           fill=AMBER, anchor="lm")
    d.text((78, 278), "in   myTrading / Documentation", font=font(27),
           fill=MUTED, anchor="lm")

    d.line([(78, 322), (w - 78, 322)], fill=LINE, width=2)
    d.text((78, 368),
           "space → leader → financials → moat → entry/exit",
           font=font(34, True), fill=WHITE, anchor="lm")
    d.text((78, 424),
           "17 steps · InvestAnswers, + book-to-bill.   "
           "Know the exit before the entry.",
           font=font(26), fill=MUTED, anchor="lm")

    img.save(OUT_WINNERS, optimize=True)
    return OUT_WINNERS


def benter() -> Path:
    """A POINTER, like 05. The method in one line, and where the doc lives.

    It pairs with catch-winners rather than repeating it: that one asks
    whether the business is good and early, this one whether the price is low
    enough relative to what it is worth. A name can pass one and fail the
    other — the doc's own AAPL example is exactly that.
    """
    w, h = 1600, 500
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 14, h], fill=VIOLET)

    d.text((78, 96), "BENTER METHOD", font=font(76, True), fill=VIOLET,
           anchor="lm")
    d.text((78, 168), "Is the PRICE low enough for what it is worth?",
           font=font(36), fill=WHITE, anchor="lm")

    d.text((78, 232), "BENTER-METHOD.md", font=font(38, True), fill=AMBER,
           anchor="lm")
    d.text((78, 278), "in   myTrading / Documentation", font=font(27),
           fill=MUTED, anchor="lm")

    d.line([(78, 318), (w - 78, 318)], fill=LINE, width=2)
    d.text((78, 360),
           "Don't pick winners — find where YOUR probability beats the price.",
           font=font(31, True), fill=WHITE, anchor="lm")
    # The scoring scale is on the tile because it is the bit that gets
    # skipped: an analysis without a number can hedge forever.
    # MEASURED, not a guessed offset. A fixed 452px put the scale on top of
    # the label the first time — textlength is the only way to know.
    lead, lf = "End every one with a SCORE /10", font(29, True)
    d.text((78, 410), lead, font=lf, fill=AMBER, anchor="lm")
    d.text((78 + d.textlength(lead, font=lf) + 26, 410),
           "10 clean buy  ·  5 neutral  ·  0 don't buy",
           font=font(27), fill=WHITE, anchor="lm")
    d.text((78, 450),
           "Pay for uncertainty: a big gap on an unproven thesis is not a big "
           "score.",
           font=font(25), fill=MUTED, anchor="lm")

    img.save(OUT_BENTER, optimize=True)
    return OUT_BENTER


def market_cycle() -> Path:
    """The Wyckoff four-phase cycle, from three Smart Money Decode X slides
    Chakravarti sent on 2026-10-24.

    THREE SCREENSHOTS, ONE IDEA — so one card. The slides were: a BTC chart
    with the four phases marked; "THE HEARTBEAT — REPEATING ENDLESSLY" showing
    the same shape across 2018-2024; and a wheel captioned "EVERY CONCEPT = A
    TOOL FOR THIS CYCLE" assigning one tool to each phase. Pasted as three
    images they would be three-quarters redundant, and the email already runs
    seven cards.

    THE REASON IT EARNS A SLOT is the last line, which is ours rather than
    theirs: his Structure column already names the phase. BULL+BULLISH is
    markup, BEAR+BEARISH is markdown, and MIXED is the honest answer during
    the two phases where nobody can tell accumulation from distribution until
    afterwards. A framework that maps onto a column he already reads every
    morning is worth more than one that needs new tooling.
    """
    w, h = 1600, 860
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 14, h], fill=VIOLET)

    # 58px ran off the right edge and clipped the final letter. Measured
    # rather than guessed: the string is long, so the size follows from it.
    d.text((78, 76), "THE MARKET CYCLE  —  REPEATING ENDLESSLY",
           font=font(48, True), fill=VIOLET, anchor="lm")
    d.text((78, 134),
           "Four phases, always the same order. Every tool you own is a tool "
           "for ONE of them.",
           font=font(31), fill=WHITE, anchor="lm")

    QUIET = (148, 163, 184)
    # The glyph IS the lesson, as with the P/b/D card: flat, up, chop, down.
    paths = [
        (QUIET, [(100, 405), (155, 388), (210, 412), (265, 390), (320, 410),
                 (375, 387), (430, 402), (470, 386)]),
        (GREEN, [(470, 386), (530, 332), (572, 356), (648, 272), (696, 296),
                 (768, 236), (810, 226)]),
        (AMBER, [(810, 226), (852, 203), (894, 247), (942, 208), (990, 252),
                 (1038, 212), (1086, 256), (1128, 218), (1170, 246)]),
        (RED,   [(1170, 246), (1228, 302), (1270, 282), (1340, 356),
                 (1382, 340), (1452, 396), (1500, 412)]),
    ]
    for colour, pts in paths:
        d.line(pts, fill=colour, width=7, joint="curve")

    for x in (470, 810, 1170):
        for y in range(190, 440, 16):
            d.line([(x, y), (x, y + 8)], fill=LINE, width=2)

    phases = [("ACCUMULATION", "sideways, quiet", 100, 470, QUIET),
              ("MARKUP", "the uptrend", 470, 810, GREEN),
              ("DISTRIBUTION", "choppy, toppy", 810, 1170, AMBER),
              ("MARKDOWN", "the downtrend", 1170, 1500, RED)]
    for name, sub, x0, x1, colour in phases:
        cx = (x0 + x1) // 2
        d.text((cx, 476), name, font=font(31, True), fill=colour, anchor="mm")
        d.text((cx, 512), sub, font=font(24), fill=MUTED, anchor="mm")

    d.line([(78, 556), (w - 78, 556)], fill=LINE, width=2)
    d.text((78, 598), "EVERY CONCEPT IS A TOOL FOR ONE PHASE",
           font=font(30, True), fill=AMBER, anchor="lm")

    tools = [("ACCUMULATION", "Volume analysis", QUIET),
             ("MARKUP", "Liquidity sweeps", GREEN),
             ("DISTRIBUTION", "Order blocks", AMBER),
             ("MARKDOWN", "Market structure", RED)]
    x = 78
    for name, tool, colour in tools:
        # Taller box and both lines anchored to their MIDDLE, so a descender
        # cannot drop through the border the way "Volume analysis" did.
        d.rectangle([x, 630, x + 348, 734], outline=colour, width=3)
        d.text((x + 20, 662), name, font=font(25, True), fill=colour,
               anchor="lm")
        d.text((x + 20, 703), tool, font=font(25), fill=WHITE, anchor="lm")
        x += 362

    d.text((78, 772),
           "YOUR SHEET ALREADY NAMES THE PHASE —  Structure: BULLISH = "
           "markup  ·  BEARISH = markdown",
           font=font(26, True), fill=TEAL, anchor="lm")
    d.text((78, 814),
           "MIXED is the honest answer in the other two: nobody separates "
           "accumulation from distribution until afterwards.",
           font=font(24), fill=MUTED, anchor="lm")

    img.save(OUT_CYCLE, optimize=True)
    return OUT_CYCLE


def vix_regime() -> Path:
    """The VIX bands, from Justin Banks (@RealJGBanks) 2026-10-17, plus what
    ten years of data actually says about them.

    Chakravarti asked for it knowing it cuts against the rest of the book:
    *"I know this contradicts our philosophy"*. It does, in one direction
    only, and the card says which — a tile that printed "TRIM below 20"
    unqualified would be advice to sell for 69% of all trading days.

    Checked over 10 years to 2026-10-16, SPY forward 60 days:

        > 40   +19.6%  (100% of the time, but only ~4 real episodes)
        30-40  + 7.0%
        20-30  + 4.2%
        < 20   + 2.4%  on 69% of all days — still positive

    So the BUY half is real and the TRIM half is not, and both belong on the
    card. The buy side is also the half that contradicts our stops: at VIX 40
    a close-confirmed stop has already sold.
    """
    w, h = 1600, 560
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 14, h], fill=RED)

    d.text((78, 82), "VIX  —  THE FEAR GAUGE", font=font(66, True), fill=RED,
           anchor="lm")
    d.text((78, 144), "Wall Street's fear gauge. Market-wide, not per-ticker.",
           font=font(32), fill=WHITE, anchor="lm")

    bands = [("> 40", "BUY THE PANIC", RED),
             ("30 - 40", "START BUYING", (251, 146, 60)),
             ("20 - 30", "HOLD POSITIONS", (148, 163, 184)),
             ("< 20", "TRIM / REDUCE RISK", GREEN)]
    x = 78
    for lvl, what, colour in bands:
        d.rectangle([x, 196, x + 348, 286], outline=colour, width=3)
        d.text((x + 20, 226), lvl, font=font(34, True), fill=colour)
        d.text((x + 20, 262), what, font=font(23), fill=WHITE)
        x += 362

    d.line([(78, 322), (w - 78, 322)], fill=LINE, width=2)
    d.text((78, 364), "10 YEARS OF DATA — SPY, 60 days forward:",
           font=font(29, True), fill=AMBER, anchor="lm")
    d.text((78, 410),
           "VIX > 40  →  +19.6%, every time.   30-40  →  +7.0%.   "
           "The BUY half is real.",
           font=font(27), fill=WHITE, anchor="lm")
    d.text((78, 452),
           "But VIX < 20 is 69% of ALL days and still +2.4% — that is not a "
           "sell signal.",
           font=font(27), fill=(252, 165, 165), anchor="lm")
    d.text((78, 506),
           "Market-wide overlay, not a stock picker. At VIX 40 a "
           "close-confirmed stop has already sold.",
           font=font(24), fill=MUTED, anchor="lm")

    img.save(OUT_VIX, optimize=True)
    return OUT_VIX


if __name__ == "__main__":
    for p in (main(), buy_cheap(), catch_winners(), benter(), vix_regime(),
              market_cycle()):
        print(f"wrote {p.name:<24} ({p.stat().st_size/1024:.0f} KB)")
