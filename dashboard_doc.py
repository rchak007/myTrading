#!/usr/bin/env python3
"""
dashboard_doc.py
================
Fills the `DASHBOARD-DOC` tab: what every field in a Dashboard block means.

Chakravarti, 2026-11-09: *"i created a new tab - DASHBOARD-DOC - here for
Also for inDicators and oracle can you create documentation for each of them
what they mean."*

WHY A GENERATED TAB AND NOT A MARKDOWN FILE. The field list is not static —
INDICATORS comes from `orders_sheet.IND_GROUPS`, FINANCIALS from
`oracle_value.FIN_GROUPS`, and both are meant to grow. A hand-written page
drifts the first time a column is added and then quietly lies. This reads
the same structures the Dashboard renders from, so a documented field and a
displayed field cannot diverge: anything in the groups but missing from
MEANINGS below is reported as undocumented, at the bottom of the tab.

    .venv/bin/python dashboard_doc.py          # print
    .venv/bin/python dashboard_doc.py --write  # write the tab
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TAB = os.getenv("GSHEET_DASHBOARD_DOC_TAB", "DASHBOARD-DOC")

# field -> (what it is, how to read it)
MEANINGS = {
    # ── RATING ───────────────────────────────────────────────────────────
    "Benter": ("0-10 expected-value score vs analyst consensus",
               "Higher is better. The curve PEAKS around a 55-70% gap and "
               "falls after: a target 4x the price scores 2, not 10, because "
               "a gap that large usually means stale estimates."),
    "Band": ("The Benter score as words",
             "CLEAN BUY 9+ · BUY 7+ · STARTER 6+ · NEUTRAL 4.5+ · "
             "DON'T BUY 2.5+ · AVOID below."),
    "Gap_%": ("Analyst mean target vs the live price",
              "+20% means the street sees 20% upside. Raw, before the "
              "confidence discount that Benter applies."),
    "Score_W": ("Weighted technical score, 0-100",
                "Trend, momentum and structure combined. Not a price target."),
    "Regime_Action": ("What the macro regime says to do about size",
                      "HOLD / EXIT / ADD. Driven by VIX and SPY vs its "
                      "200-day, not by this ticker."),
    "Macro": ("Market regime", "BULL / NEUTRAL / BEAR, for the whole book."),
    "Signal": ("This ticker's combined signal", "BUY / HOLD / EXIT."),
    "Sell_Order": ("Whether a protective SELL rests at Schwab",
                   "NO_SELL in red means nothing is protecting the position."),
    "Earnings": ("Earnings inside the alert window",
                 "A gap jumps a close-confirmed stop. The HARD stop is what "
                 "actually catches it."),
    "Mkt_Cap_M": ("Market capitalisation, $ millions", ""),

    # ── TREND ────────────────────────────────────────────────────────────
    "Supertrend": ("The Supertrend line, in dollars",
                   "Price above it is the uptrend case; below, the down."),
    "ST_Sig": ("Supertrend's verdict", "BUY or SELL."),
    "MOST_MA": ("The moving average MOST follows", ""),
    "MOST_Line": ("MOST's own trailing line",
                  "It flips when the MA crosses this line."),
    "MOST_Sig": ("MOST's verdict", "BUY or SELL."),
    "ADXR": ("Trend STRENGTH, not direction",
             "RISING / FALLING. A strong signal in a falling ADXR is a "
             "signal with no wind behind it."),
    "ADXR_Sig": ("ADXR read as words", "TREND_OK or WEAK."),
    "ST+Vol": ("Supertrend confirmed by volume", "Agreement matters more "
               "than either alone."),
    "Combined": ("Supertrend and MOST together",
                 "They disagree often; when they agree it is worth more."),
    "Full": ("Every trend input combined", "The one-word summary."),

    # ── LEVELS ───────────────────────────────────────────────────────────
    "ATH": ("All-time high", ""),
    "ATH_Dist_%": ("How far below the all-time high",
                   "-39% means it would take a 64% rally to get back."),
    "MRC_Zone": ("Where price sits in the mean-reversion channel",
                 "Above_Mean / Below_Mean / OS (oversold) / OB (overbought)."),
    "MRC_Dist_%": ("Distance from the channel mean", ""),
    "MRC_R2": ("Upper channel band, far", "A trim level, not a target."),
    "MRC_R1": ("Upper channel band, near", ""),
    "MRC_Mean": ("The channel's centre line", "What it reverts toward."),
    "MRC_S1": ("Lower band, near", "A dip-buy level."),
    "MRC_S2": ("Lower band, far", "The deeper bid."),
    "RSI": ("Relative Strength Index, 0-100",
            "Under 30 oversold, over 70 overbought. Weak on its own, "
            "useful with structure."),

    # ── STRUCTURE ────────────────────────────────────────────────────────
    "Regime": ("Structural regime from pivots", "BULL / BEAR."),
    "Structure": ("Whether the pivots agree", "CLEAN or MIXED."),
    "Last_Label": ("The most recent confirmed pivot",
                   "HH higher high · HL higher low · LH lower high · "
                   "LL lower low. HL is the classic entry, LH the exit."),
    "At": ("Price of that pivot", ""),
    "Confirmed": ("The date it became KNOWABLE",
                  "Not the date of the low itself — a pivot is only "
                  "confirmed once a later close breaks structure. Orders "
                  "that wait for 'the next HL' measure from this."),
    "Prev": ("The pivot before it", ""),
    "Prev_At": ("Its price", ""),
    "Support": ("Nearest structural support", ""),
    "Resistance": ("Nearest structural resistance", ""),
    "Bars_Flip": ("Bars since the regime last flipped",
                  "A small number means the new regime is young."),

    # ── MOMENTUM ─────────────────────────────────────────────────────────
    "ATR": ("Average True Range — a day's typical move, in dollars",
            "Every recommended stop is a multiple of this. A 3% ATR name "
            "needs a wider stop than a 1% one."),
    "RET30_%": ("Return over 30 trading days", ""),
    "RET60_%": ("Return over 60 trading days", ""),
    "RET90_%": ("Return over 90 trading days", ""),
    "RET120_%": ("Return over 120 trading days", ""),
    "Score30": ("Momentum rank over 30 days, 0-100", "Versus the book."),
    "Score60": ("Momentum rank over 60 days", ""),
    "Score90": ("Momentum rank over 90 days", ""),
    "Score120": ("Momentum rank over 120 days", ""),
    "Volume": ("Shares traded, latest session", ""),

    # ── ORACLE ───────────────────────────────────────────────────────────
    "OracleValue": ("Our own conservative fair value per share",
                    "Built mainly on the stock's OWN mean price-to-sales "
                    "across every year of SEC filings. NOT StockOracle's "
                    "number and not tuned toward it — a screener, not a "
                    "price target."),
    "Upside_to_Fair": ("Fair value vs the live price",
                       "+20% means the model sees 20% upside."),
    "Range": ("The spread across the model's anchors",
              "Capped at +/-25%. Where they disagree by more, the Caveat "
              "says so instead of drawing a wider band."),
    "Moat": ("OUR moat rating, from ten years of filings",
             "WIDE / NARROW / NONE, judged on durable excess returns on "
             "capital: median ROE, how often it clears a 12% hurdle, and "
             "how steady gross margin is. Blank means too little history "
             "to judge, which is different from NONE."),
    "As_Of": ("When the value was computed",
              "Weekly on Monday, plus the day after a ticker reports. The "
              "BAND is recomputed against the live price every cycle."),
    "Pred": ("Predictability — how reliably reported EPS lands near "
             "estimate", "From the last 12 quarters of beats and misses."),
    "Profit": ("Profitability, on the FIVE-YEAR average",
               "Not trailing twelve months. A cyclical trough is not the "
               "absence of profitability."),
    "Growth": ("Revenue growth, compounded over available years", ""),
    "Fin": ("Financial strength", "Net cash, free cash flow, leverage."),
    "Val": ("Valuation band versus OracleValue", ""),
    "Mean_P/S": ("The stock's own average price-to-sales",
                 "The dominant input. Why SiTime earns 33x sales and "
                 "Applied Optoelectronics 3.8x on identical margins."),
    "Years": ("How many fiscal years that mean rests on",
              "Fewer than 8 and the anchor is discounted. Error falls as "
              "years rise: 7y was 54% out, 16y was 6%."),
    "P/S_Spread": ("How much those yearly multiples disagreed",
                   "Reported, NOT used to discount the value — on real "
                   "history it does not predict error."),
    "Next_Earnings": ("Next expected report date", ""),
    "Caveat": ("When to distrust the value",
               "'thin history' or 'multiple has ranged widely'. Blank is "
               "the model saying it has no reservation."),

    # ── FINANCIALS ───────────────────────────────────────────────────────
    "PE": ("Price / trailing 12-month earnings",
           "Blank for a loss-maker — there is no meaningful P/E on "
           "negative earnings. Cyclicals look CHEAPEST at the top of the "
           "cycle, when earnings peak."),
    "Fwd_PE": ("Price / next year's estimated earnings",
               "An estimate, so it carries the street's optimism."),
    "PEG": ("P/E divided by the growth rate",
            "Under 1 is the classic 'growth is cheap' marker. Breaks down "
            "when growth is erratic or negative."),
    "EPS": ("Trailing 12-month earnings per share", ""),
    "Net_Margin_%": ("Net income as a share of revenue, trailing 12 months",
                     "Read it beside Margin_5y_%. A wide gap means you are "
                     "looking at a point in a cycle, not a level."),
    "Margin_5y_%": ("Net margin averaged over five FISCAL years",
                    "The normalised figure. This is what the valuation and "
                    "the FINANCIALS verdict use — a trough is not the "
                    "absence of profitability and a peak is not its "
                    "presence."),
    "ROE_5y_%": ("Return on equity over five fiscal years",
                 "Same reason as the margin. MU: 88.3% trailing, 16.4% "
                 "normalised."),
    "ROE_%": ("Return on equity, trailing", ""),
    "Rev_Growth_%": ("Revenue growth, most recent year over year", ""),
    "Debt/Equity": ("Total debt against shareholders' equity",
                    "As a percentage. Over ~200 is leveraged."),
    "Div_Yield_%": ("Dividend yield", ""),
    "Mkt_Cap_M": ("Market capitalisation, $ millions", ""),
}


def build() -> list[list]:
    import orders_sheet
    import oracle_value

    data: list[list] = []

    def add(row):
        data.append((row + ["", "", ""])[:4])

    add(["DASHBOARD — WHAT EVERY FIELD MEANS", "",
         "generated, read-only", ""])
    add(["", "Regenerated from the same structures the Dashboard renders "
             "from, so this cannot drift from what you see.", "", ""])
    add([])

    documented = set()

    def section(title, note, groups):
        add([])
        add([title.upper(), note, "", ""])
        add(["GROUP", "FIELD", "WHAT IT IS", "HOW TO READ IT"])
        for label, fields in groups:
            for i, name in enumerate(fields):
                documented.add(name)
                what, how = MEANINGS.get(name, ("", ""))
                add([label if i == 0 else "", name, what, how])

    section("INDICATORS", "computed every cycle by jobStocksSignals",
            [(lbl, [n for n, _ in f]) for lbl, f in orders_sheet.IND_GROUPS])

    section("ORACLE", "our own fair value — a screener, not a price target",
            [("VALUE", ["OracleValue", "Band", "Upside_to_Fair", "Range",
                        "Moat", "As_Of"]),
             ("QUALITY", ["Pred", "Profit", "Growth", "Moat", "Fin", "Val"]),
             ("BASIS", ["Mean_P/S", "Years", "P/S_Spread", "Next_Earnings",
                        "Caveat"])])

    add([])
    add(["FINANCIALS COLOURS", "the label row is coloured by an overall "
         "verdict, with its reasons printed beside it", "", ""])
    for c, meaning in [
        ("GREEN — GOOD", "Three or more positives and nothing negative."),
        ("RED — BAD", "Two negatives, or negatives with nothing good."),
        ("BLUE — NEUTRAL", "Mixed. The reasons end with 'but ...' naming "
                           "what is dragging."),
        ("ORANGE cell", "PEG is between 0 and 1 — growth is cheap on that "
                        "measure. Check the growth is repeatable first."),
    ]:
        add(["", c, meaning, ""])
    add(["", "Scored on DURABLE things", "Normalised margin and ROE, "
         "revenue growth, debt/equity, PEG, and forward P/E below "
         "trailing. Deliberately NOT on how cheap the multiple is — a low "
         "P/E is the cyclical trap, not evidence of quality.", ""])

    section("FINANCIALS", "reported figures, as of the ORACLE As_Of date",
            [(lbl, [n for n, _ in f]) for lbl, f in oracle_value.FIN_GROUPS])

    # HOW TO READ THEM TOGETHER. Chakravarti, 2026-11-09: *"basically i
    # want to also learn a little how the fundamentals are also evaluated
    # etc."* Definitions alone do not teach that — the meaning is in the
    # COMBINATIONS, and in knowing which ratio lies in which situation.
    # Examples are from his own book so they can be checked against the
    # blocks above.
    add([])
    add(["HOW TO READ FUNDAMENTALS TOGETHER", "no single ratio decides "
         "anything", "", ""])
    add(["TRAP / PATTERN", "WHAT YOU SEE", "WHAT IT USUALLY MEANS",
         "WHAT TO DO"])
    for row in [
        ["The cyclical P/E trap",
         "A very LOW P/E on a semi or a miner",
         "Cyclicals look cheapest at the TOP of the cycle, because the E "
         "in P/E is at a peak that will not last. MU shows a 9.9x trailing "
         "P/E and 3.5x forward on a 379% revenue jump — that is a boom, "
         "not a bargain.",
         "Compare Net_Margin_% against Margin_5y_% in the QUALITY row "
         "right below. MU: 63.8% trailing vs 13.1% over five years — the "
         "trailing figure is the peak, the five-year one is the business. "
         "(BASIS/Years is NOT this: that is how many years of SALES data "
         "sit behind the price-to-sales anchor.)"],
        ["P/E and PEG disagree",
         "High P/E, low PEG",
         "You are paying up, but growth is covering it. PEG under 1 is the "
         "classic 'growth is cheap' marker.",
         "Check the growth is real and repeatable, not one easy "
         "comparison. PEG breaks down entirely when growth is erratic."],
        ["No P/E at all",
         "PE blank",
         "The company lost money. There is no meaningful P/E on negative "
         "earnings, so the field is blank rather than zero.",
         "Value it on SALES instead — that is exactly why OracleValue "
         "leans on price-to-sales."],
        ["Margin without returns",
         "High Net_Margin, low ROE",
         "Profitable on each sale but carrying a lot of capital to do it.",
         "Look at Debt/Equity. Leverage flatters ROE and hides risk."],
        ["Returns without margin",
         "Low Net_Margin, high ROE",
         "A thin-margin business turning capital over fast — retail and "
         "distribution look like this.",
         "Normal for the model. Judge against its own history, not "
         "against a software company."],
        ["The moat question",
         "ROE above ~15% for most of a decade",
         "Something is keeping competitors from competing the returns "
         "away. That is what our Moat rating measures — not technology or "
         "market share, which are capabilities, not protection.",
         "One good year is noise. The ORACLE Moat field uses ten."],
        ["Growth that is not growth",
         "Rev_Growth huge, one year only",
         "An easy comparison, an acquisition, or a recovery from a "
         "collapse.",
         "Prefer the compounded figure in ORACLE/Growth over a single "
         "year-over-year print."],
        ["Debt that matters",
         "Debt/Equity over ~200 with falling margins",
         "Leverage is fine while business is good and is what kills "
         "companies when it is not.",
         "Pair it with free cash flow — debt serviced from cash flow is a "
         "different thing from debt serviced by refinancing."],
    ]:
        add(row)

    add([])
    add(["THE ORDER I WOULD READ A BLOCK IN", "", "", ""])
    for i, step in enumerate([
        "Is it protected? Has_Stop on the POSITIONS row. Nothing else "
        "matters if a gap can take the position out.",
        "What does the market think? RATING — Benter and Band, from "
        "analyst consensus.",
        "What does the chart say? TREND and STRUCTURE. Last_Label HL is "
        "the classic entry, LH the classic exit.",
        "What is it worth? ORACLE — and read the BASIS line before the "
        "number. A value built on 4 years is a guess; on 16 it is not.",
        "What is the business actually doing? FINANCIALS — and apply the "
        "traps above rather than reading any one ratio alone.",
        "Where would I act? RECOMMENDED levels, sized by ATR.",
    ], start=1):
        add([f"{i}.", step, "", ""])

    # UNDOCUMENTED FIELDS ARE REPORTED, NOT HIDDEN. A field added to a group
    # and never described here would otherwise appear on the Dashboard with
    # no explanation anywhere, and nothing would ever say so.
    missing = sorted(documented - set(MEANINGS))
    add([])
    if missing:
        add(["⚠️ UNDOCUMENTED", f"{len(missing)} field(s) shown on the "
             f"Dashboard with no description here", "", ""])
        for m in missing:
            add(["", m, "add it to MEANINGS in dashboard_doc.py", ""])
    else:
        add(["✓", f"All {len(documented)} displayed fields are documented.",
             "", ""])
    return data


def write(book=None, log=print) -> str:
    import orders_sheet
    book = book or orders_sheet._open_book()
    data = build()
    try:
        ws = book.worksheet(TAB)
    except Exception:                                    # noqa: BLE001
        ws = book.add_worksheet(title=TAB, rows=len(data) + 40, cols=4)
    if ws.row_count < len(data) + 5:
        ws.add_rows(len(data) + 20 - ws.row_count)
    ws.batch_clear(["A1:D2000"])
    ws.update(values=data, range_name=f"A1:D{len(data)}",
              value_input_option="RAW")
    msg = f"wrote {len(data)} row(s) to '{TAB}'"
    log(msg)
    return msg


def main() -> int:
    ap = argparse.ArgumentParser(description="Dashboard field reference")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.write:
        write()
        return 0
    for r in build():
        print(" | ".join(str(c) for c in r if str(c).strip()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
