#!/usr/bin/env python3
"""
make_cards.py
=============
Regenerates the info-note cards WE own, which ride at the end of every
reminder email:

    03-pbd-structure.png   P / b / D market structure + the breakout rule
    04-buy-cheap.png       the one discipline his own data says he breaks

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
W, H = 1600, 1120

BG = (13, 13, 16)
WHITE = (255, 255, 255)
MUTED = (156, 163, 175)
BLUE = (59, 130, 246)
RED = (239, 68, 68)
GREEN = (74, 222, 128)
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


if __name__ == "__main__":
    for p in (main(), buy_cheap()):
        print(f"wrote {p.name:<24} ({p.stat().st_size/1024:.0f} KB)")
