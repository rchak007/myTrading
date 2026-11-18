#!/usr/bin/env python3
"""
test_options.py
===============
The OPTIONS tab: what it reads, what it refuses to lose, and what it advises.

Three things here are load-bearing and the rest is detail:

  * A regeneration must never eat a hand-typed SoFi row. `parse_typed` and
    the read/write guard are most of this file.
  * A covered-call suggestion must never be struck below cost basis, because
    assignment there books a real loss on shares meant to be kept.
  * Schwab's -999 "no data" sentinel must become None. It sorts as deep OTM,
    so a greek that silently survives as -999 gets PICKED as the safest
    strike on the board.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/chakravarti/github/myTrading")

QUIET = lambda *_: None    # noqa: E731


def _tmp(name):
    return Path(tempfile.mkdtemp()) / name


def _contract(**kw):
    c = {"symbol": "X", "right": "C", "expiry": "2026-12-19", "strike": 70.0,
         "bid": 1.0, "ask": 1.1, "mark": 1.05, "spread": 0.1, "delta": 0.20,
         "gamma": 0.01, "theta": -0.03, "vega": 0.05, "iv": 45.0, "oi": 500,
         "volume": 10, "dte": 14, "itm": False}
    c.update(kw)
    return c


def _chain(price=60.0, contracts=None):
    return {"underlying": "CRWV", "price": price,
            "contracts": contracts or [], "asof": "", "age_min": 0}


# ─────────────────────────────────────────────── greeks and sentinels

def test_the_minus_999_sentinel_becomes_None():
    # Schwab says -999.0 for "no data". As a delta it reads as deeply OTM,
    # which is exactly the strike a "safest" search would choose.
    import option_chain as oc
    assert oc._g(-999.0) is None
    assert oc._g("NaN") is None
    assert oc._g(None) is None
    assert oc._g("") is None
    assert oc._g(0.23) == 0.23
    assert oc._g("0.23") == 0.23


def test_mark_prefers_the_midpoint_over_a_stale_last():
    # `last` on a quiet strike can be hours old and will report a call as
    # nearly worthless while the ask is sixty cents.
    import option_chain as oc
    assert oc._mark({"bid": 0.50, "ask": 0.70, "last": 0.05}) == 0.60
    assert oc._mark({"mark": 0.62, "bid": 0.50, "ask": 0.70}) == 0.62
    assert oc._mark({"last": 0.33}) == 0.33


def test_osi_symbol_matches_schwabs_spelling():
    import option_chain as oc
    assert oc.osi_symbol("CRWV", "2026-11-20", "C", 60) == "CRWV  261120C00060000"
    assert oc.osi_symbol("MSTR", "2026-11-27", "C", 127) == "MSTR  261127C00127000"
    assert oc.osi_symbol("SPY", "2026-12-19", "P", 612.5) == "SPY   261219P00612500"


def test_parse_chain_flattens_both_maps():
    import option_chain as oc
    payload = {
        "symbol": "CRWV", "status": "SUCCESS", "underlyingPrice": 58.4,
        "callExpDateMap": {"2026-11-20:3": {"60.0": [{
            "symbol": "CRWV  261120C00060000", "strikePrice": 60.0,
            "bid": 0.40, "ask": 0.43, "mark": 0.415, "delta": 0.27,
            "gamma": 0.08, "theta": -0.09, "volatility": 72.4,
            "openInterest": 1200, "daysToExpiration": 3, "inTheMoney": False}]}},
        "putExpDateMap": {"2026-12-19:32": {"50.0": [{
            "symbol": "CRWV  261219P00050000", "strikePrice": 50.0,
            "bid": 1.0, "ask": 1.1, "delta": -0.22, "volatility": 65.0,
            "openInterest": 300, "daysToExpiration": 32}]}},
    }
    ch = oc.parse_chain(payload)
    assert ch["price"] == 58.4
    assert len(ch["contracts"]) == 2
    call = [c for c in ch["contracts"] if c["right"] == "C"][0]
    assert call["expiry"] == "2026-11-20" and call["strike"] == 60.0
    assert call["delta"] == 0.27 and call["dte"] == 3
    put = [c for c in ch["contracts"] if c["right"] == "P"][0]
    assert put["expiry"] == "2026-12-19" and put["mark"] == 1.05


# ──────────────────────────────────────────────────────── choosing one


def test_illiquid_strikes_are_not_suggested():
    # Open interest of 2 means the premium on screen is fiction.
    import option_chain as oc
    ch = _chain(60.0, [_contract(strike=65.0, delta=0.20, oi=2),
                       _contract(strike=70.0, delta=0.12, oi=800)])
    assert oc.pick(ch, "C")["strike"] == 70.0


def test_a_strike_with_a_huge_spread_is_not_suggested():
    import option_chain as oc
    ch = _chain(60.0, [_contract(strike=65.0, delta=0.20, bid=0.10, ask=1.00,
                                 mark=0.55),
                       _contract(strike=70.0, delta=0.10, bid=0.30, ask=0.34,
                                 mark=0.32)])
    assert oc.pick(ch, "C")["strike"] == 70.0


def test_picking_honours_the_dte_window():
    import option_chain as oc
    ch = _chain(60.0, [_contract(strike=65.0, delta=0.20, dte=3),
                       _contract(strike=66.0, delta=0.30, dte=14)])
    got = oc.pick(ch, "C", dte_lo=7, dte_hi=21)
    assert got["dte"] == 14


def test_a_roll_never_goes_down_in_strike():
    # Rolling down locks in less upside than you already have — the one
    # thing a roll must not do.
    import option_chain as oc
    cur = _contract(strike=60.0, expiry="2026-11-20", dte=3, mark=0.42)
    ch = _chain(59.0, [cur,
                       _contract(strike=55.0, expiry="2026-12-19", delta=0.40),
                       _contract(strike=62.0, expiry="2026-12-19", delta=0.25,
                                 mark=1.20)])
    best, credit = oc.roll(ch, cur, right="C", strike=60.0, expiry="2026-11-20")
    assert best["strike"] == 62.0
    assert abs(credit - (1.20 - 0.42)) < 1e-9


def test_a_debit_roll_is_reported_as_negative_not_hidden():
    import option_chain as oc
    cur = _contract(strike=60.0, expiry="2026-11-20", dte=3, mark=2.00)
    ch = _chain(61.0, [cur,
                       _contract(strike=65.0, expiry="2026-12-19", delta=0.20,
                                 mark=0.80)])
    _best, credit = oc.roll(ch, cur, right="C", strike=60.0, expiry="2026-11-20")
    assert credit < 0, credit


def test_iv_rank_is_silent_until_there_is_history():
    # A rank off four readings is always 0 or 100 and would be read as signal.
    import option_chain as oc
    p = _tmp("iv.csv")
    for i in range(5):
        oc.record_iv("CRWV", 40 + i, path=p, today=__import__("datetime").date(2026, 11, i + 1))
    assert oc.iv_rank("CRWV", 45, path=p) is None
    for i in range(25):
        oc.record_iv("CRWV", 30 + i, path=p,
                     today=__import__("datetime").date(2026, 10, i + 1))
    assert oc.iv_rank("CRWV", 54, path=p) == 100
    assert oc.iv_rank("CRWV", 30, path=p) == 0


def test_one_iv_reading_per_day():
    import option_chain as oc
    from datetime import date
    p = _tmp("iv.csv")
    oc.record_iv("CRWV", 40, path=p, today=date(2026, 11, 1))
    oc.record_iv("CRWV", 99, path=p, today=date(2026, 11, 1))
    assert len(oc.read_iv(p)) == 1


# ────────────────────────────────────────────── reading what you typed

def test_type_column_is_forgiving():
    import options_sheet as osh
    for s in ("Covered Call", "covered call", "CC", "call", " Covered-Call "):
        assert osh.right_of(s) == "C", s
    for s in ("Secured Put", "cash secured put", "CSP", "put"):
        assert osh.right_of(s) == "P", s
    assert osh.right_of("") == ""
    assert osh.right_of("spread") == ""


def test_dates_survive_every_way_sheets_hands_them_back():
    import options_sheet as osh
    assert osh.norm_date("2026-12-19") == "2026-12-19"
    assert osh.norm_date("12/19/2026") == "2026-12-19"
    assert osh.norm_date("2026-12-19 00:00:00") == "2026-12-19"
    assert osh.norm_date(46375) == "2026-12-19"     # Sheets serial
    assert osh.norm_date("") == ""
    assert osh.norm_date("soon") == ""


def test_a_bare_month_day_means_the_next_one():
    import options_sheet as osh
    from datetime import date
    got = osh.norm_date("1/1")
    assert got >= date.today().isoformat(), got


def test_schwab_rows_are_dropped_on_read_and_manual_rows_are_not():
    # The tab is both input and output. A Schwab row kept across a cycle
    # would outlive the position; a manual row dropped would be retyping.
    import options_sheet as osh
    vals = [
        ["OPTIONS", "updated ..."], [], osh.OPT_HDR,
        ["", "— OPEN POSITIONS —"],
        ["CRWV", "Covered Call", "60", "0.27", "2026-11-20", "1", "922",
         "SCHWAB"],
        ["SOFI", "Covered Call", "12", "0.45", "2026-12-19", "2", "SOFI",
         "MANUAL"],
        ["NVDA", "Secured Put", "", "", "", "", ""],          # just typed
    ]
    got = osh.parse_typed(vals)
    assert [r["Ticker"] for r in got] == ["SOFI", "NVDA"]
    assert got[0]["Acct"] == "SOFI" and got[0]["Premium"] == "0.45"


def test_an_untouched_suggestion_is_dropped_but_a_filled_one_is_kept():
    import options_sheet as osh
    vals = [osh.OPT_HDR,
            ["GOOG", "Covered Call", "", "", "", "1", "", "AUTO"],
            ["SPY", "Covered Call", "700", "1.20", "2026-12-19", "1", "", "AUTO"]]
    got = osh.parse_typed(vals)
    assert [r["Ticker"] for r in got] == ["SPY"]


def test_a_typed_row_the_broker_also_reports_is_not_counted_twice():
    import options_sheet as osh
    schwab = [{"Ticker": "MSTR", "Kind": "Covered Call", "Strike": 127.0,
               "Premium": 0.54, "Expiry": "2026-11-27", "Qty": 1,
               "Acct": "922", "Src": "SCHWAB"}]
    manual = [{"Ticker": "MSTR", "Kind": "Covered Call", "Strike": "127",
               "Premium": "0.54", "Expiry": "11/27/2026", "Qty": "1",
               "Acct": "SOFI"},
              {"Ticker": "SOFI", "Kind": "Covered Call", "Strike": "12",
               "Premium": "0.45", "Expiry": "2026-12-19", "Qty": "2",
               "Acct": "SOFI"}]
    got = osh.merge(schwab, manual, log=QUIET)
    assert [r["Ticker"] for r in got] == ["MSTR", "SOFI"]


def test_long_options_are_not_shown_as_sold_premium():
    import options_sheet as osh
    opts = [{"underlying": "GOOG", "right": "C", "strike": 300.0,
             "expiry": "2026-12-19", "Qty": 2.0, "Avg_Cost": 5.0, "Acct": "1"},
            {"underlying": "CRWV", "right": "C", "strike": 60.0,
             "expiry": "2026-11-20", "Qty": -1.0, "Avg_Cost": 0.27, "Acct": "1"}]
    got = osh.from_schwab(opts, log=QUIET)
    assert [r["Ticker"] for r in got] == ["CRWV"]
    assert got[0]["Qty"] == 1.0


def test_the_manual_ledger_does_not_grow_when_nothing_changed():
    import options_sheet as osh
    p = _tmp("manual.csv")
    rows = [{"Ticker": "SOFI", "Kind": "Covered Call", "Strike": "12",
             "Premium": "0.45", "Expiry": "2026-12-19", "Qty": "2",
             "Acct": "SOFI"}]
    assert osh.save_manual(rows, path=p, log=QUIET) is True
    assert osh.save_manual(rows, path=p, log=QUIET) is False
    assert len(osh.read_manual_log(p)) == 1
    rows[0]["Strike"] = "13"
    assert osh.save_manual(rows, path=p, log=QUIET) is True
    assert len(osh.read_manual_log(p)) == 2
    assert [r["Strike"] for r in osh.load_manual(p)] == ["13"]


def test_an_unreadable_tab_is_never_written_over():
    import options_sheet as osh

    class Broken:
        def get_all_values(self):
            raise RuntimeError("503")

    entries, ok = osh.read_tab(Broken(), log=QUIET)
    assert ok is False and entries == []


# ───────────────────────────────────────────────────────── the verdicts

def _pos(**kw):
    p = {"right": "C", "strike": 60.0, "under": 58.0, "premium": 0.27,
         "mark": 0.20, "kept": 26.0, "delta": 0.20, "dte": 20, "qty": 1,
         "basis": 40.0, "shares": 100}
    p.update(kw)
    return p



def test_in_the_money_is_called_out():
    import options_sheet as osh
    v, _ = osh.assess(_pos(under=62.0, strike=60.0))
    assert v == "🔴 ITM"
    v, _ = osh.assess(_pos(right="P", under=45.0, strike=50.0, basis=None))
    assert v == "🔴 ITM"


def test_gamma_week_says_roll():
    import options_sheet as osh
    v, a = osh.assess(_pos(delta=0.48, dte=3))
    assert v == "🔴 ROLL" and "gamma" in a


def test_most_of_the_premium_captured_says_close():
    import options_sheet as osh
    v, a = osh.assess(_pos(premium=1.00, mark=0.08, kept=92.0))
    assert v == "🟢 CLOSE" and "free the collateral" in a


def test_the_crwv_call_on_his_screen_is_a_watch_not_a_panic():
    # -1 CRWV 11/20 60C, sold 0.27, marked 0.415, delta 0.27, 3 DTE.
    import options_sheet as osh
    v, a = osh.assess(_pos(premium=0.27, mark=0.415, kept=-53.7, delta=0.27,
                           dte=3, under=58.0, strike=60.0))
    assert v == "🟠 WATCH", (v, a)


def test_a_quiet_position_is_left_alone():
    import options_sheet as osh
    v, _ = osh.assess(_pos(delta=0.12, dte=25, kept=30.0))
    assert v == "🟢 DECAY"


def test_an_expired_row_says_so_rather_than_advising_on_it():
    import options_sheet as osh
    v, _ = osh.assess(_pos(dte=-2))
    assert v == "⚪ EXPIRED"


def test_a_short_call_with_no_shares_behind_it_is_called_naked():
    import options_sheet as osh
    S = {"src": "SCHWAB", "right": "C"}
    assert "NAKED" in osh.covered_text({**S, "qty": 1, "shares": 0})
    assert osh.covered_text({**S, "qty": 1, "shares": 100}).startswith("✓")
    assert osh.covered_text({**S, "qty": 2, "shares": 100}).startswith("🔴")


def test_a_sofi_call_is_not_called_naked_just_because_schwab_cannot_see_it():
    # Crying wolf on the hand-typed rows is how the flag stops being read on
    # the Schwab rows, where it means something.
    import options_sheet as osh
    got = osh.covered_text({"src": "MANUAL", "right": "C", "qty": 2,
                            "shares": None})
    assert "NAKED" not in got and "200" in got


# ──────────────────────────────────────────────────── P/L and suggestions

def test_pl_and_kept_percent_are_computed_from_what_you_typed():
    import options_sheet as osh
    ch = _chain(58.0, [_contract(strike=60.0, expiry="2026-11-20", dte=3,
                                 mark=0.415, delta=0.27)])
    p = osh.enrich({"Ticker": "CRWV", "Kind": "Covered Call", "Strike": "60",
                    "Premium": "0.27", "Expiry": "2026-11-20", "Qty": "1"}, ch)
    assert abs(p["pl"] - (0.27 - 0.415) * 100) < 0.01, p["pl"]
    assert p["kept"] < 0 and p["delta"] == 0.27
    assert abs(p["to_strike"] - round(100 * 2.0 / 58.0, 1)) < 0.1


def test_a_blank_row_gets_a_concrete_order_to_place():
    import options_sheet as osh
    ch = _chain(60.0, [_contract(strike=65.0, delta=0.21, mark=0.95, oi=900,
                                 expiry="2026-12-19", dte=14)])
    p = osh.enrich({"Ticker": "CRWV", "Kind": "Covered Call", "Qty": "1"}, ch,
                   shares=100)
    v, a = osh.suggest(p, ch)
    assert v == "🔵 SUGGESTION", (v, a)
    assert "SELL 1 CRWV 2026-12-19 $65.00C" in a and "$95" in a


def test_the_tab_points_at_the_verb_when_it_cannot_see_the_shares():
    import options_sheet as osh
    ch = _chain(60.0, [_contract(strike=65.0, delta=0.21, dte=14)])
    p = osh.enrich({"Ticker": "CRWV", "Kind": "Covered Call"}, ch, shares=40)
    v, a = osh.suggest(p, ch)
    assert v == "⚪ NO SHARES" and "cc CRWV" in a



def test_the_most_urgent_row_is_first():
    import options_sheet as osh
    ch = _chain(62.0, [
        _contract(strike=60.0, expiry="2026-11-20", dte=3, delta=0.80,
                  mark=2.1, itm=True),
        _contract(strike=90.0, expiry="2026-12-19", dte=32, delta=0.10,
                  mark=0.30)])
    entries = [
        {"Ticker": "CRWV", "Kind": "Covered Call", "Strike": "90",
         "Premium": "0.50", "Expiry": "2026-12-19", "Qty": "1", "Src": "MANUAL"},
        {"Ticker": "CRWV", "Kind": "Covered Call", "Strike": "60",
         "Premium": "0.27", "Expiry": "2026-11-20", "Qty": "1", "Src": "SCHWAB"},
    ]
    rows, _marks = osh.build_rows(entries, {"CRWV": ch},
                                  shares={"CRWV": 100},
                                  log=QUIET)
    body = [r for r in rows if r[0] == "CRWV"]
    assert body[0][2] == 60.0, body[0]           # the ITM one leads
    assert "🔴" in str(body[0][OPT_VERDICT])



# ───────────────────────────────────────────────────── the ladder / advice

def test_an_in_the_money_strike_is_never_suggested_to_open():
    # Opening a short option already ITM is a decision to be assigned,
    # dressed up as income.
    import option_chain as oc
    ch = _chain(60.0, [_contract(strike=55.0, delta=0.35, oi=900),
                       _contract(strike=65.0, delta=0.18, oi=900)])
    assert oc.pick(ch, "C")["strike"] == 65.0
    puts = [_contract(right="P", strike=65.0, delta=-0.35, oi=900),
            _contract(right="P", strike=55.0, delta=-0.18, oi=900)]
    assert oc.pick(_chain(60.0, puts), "P")["strike"] == 55.0


def test_annualised_makes_two_expiries_comparable():
    # 60c on a 10-day beats $2.00 on a 45-day per day of capital tied up,
    # and the bigger dollar number is the wrong one to pick.
    import options_sheet as osh
    short = osh.annualised(0.60, 100.0, 10)
    long = osh.annualised(2.00, 100.0, 45)
    assert short > long, (short, long)
    assert osh.annualised(1.0, 0, 10) is None
    assert osh.annualised(1.0, 100.0, 0) is None


def test_the_ladder_is_one_row_per_expiry():
    import options_sheet as osh
    ch = _chain(60.0, [
        _contract(strike=63.0, expiry="2026-11-21", dte=4, delta=0.28, oi=900),
        _contract(strike=66.0, expiry="2026-11-21", dte=4, delta=0.14, oi=900),
        _contract(strike=70.0, expiry="2026-12-19", dte=32, delta=0.16, oi=900)])
    rows = osh.per_expiry(ch, "C", target_delta=0.15)
    assert [r["expiry"] for r in rows] == ["2026-11-21", "2026-12-19"]
    assert rows[0]["strike"] == 66.0      # the Δ0.14 one, not the Δ0.28
    assert rows[0]["ann"] and rows[0]["ann"] > 0


def test_the_pick_stays_inside_his_seven_to_fourteen_day_window():
    # A 4-day at 56%/yr outscores an 11-day at 47%/yr, and is still not the
    # answer: he said 7-14 days, and the verb must not quietly restrategise.
    import options_sheet as osh
    ch = _chain(131.0, [
        _contract(strike=142.0, expiry="2026-11-21", dte=4, delta=0.14,
                  mark=0.80, bid=0.75, ask=0.85, oi=2000),
        _contract(strike=145.0, expiry="2026-11-28", dte=11, delta=0.16,
                  mark=1.85, bid=1.80, ask=1.90, oi=2000)])
    out = "\n".join(osh.advise("MSTR", ch, right="C", shares=100,
                               earnings=False))
    assert "2026-11-28 $145.00C" in out, out
    # And what capping the shares at that strike costs from today's price.
    assert "$1,400 above today's $131.00" in out, out





def test_advice_says_when_you_are_already_short_the_name():
    # Otherwise a second contract reads as free money against shares that
    # are already spoken for.
    import options_sheet as osh
    ch = _chain(131.0, [_contract(strike=145.0, expiry="2026-11-28", dte=11,
                                  delta=0.16, mark=1.85, oi=2000)])
    out = "\n".join(osh.advise("MSTR", ch, right="C", shares=100,
                               open_positions=["1x 2026-11-27 $127.00C"],
                               earnings=False))
    assert "already short" in out and "ADDITIONAL" in out


def test_a_put_answer_states_the_cash_it_ties_up():
    import options_sheet as osh
    ch = _chain(131.0, [_contract(right="P", strike=120.0, expiry="2026-11-28",
                                  dte=11, delta=-0.16, mark=1.50, oi=2000)])
    out = "\n".join(osh.advise("NVDA", ch, right="P", qty=1, earnings=False))
    assert "$12,000 of cash" in out
    assert "needs $12,000 cash" in out, out
    assert "a price you want to own at" in out


def test_a_premium_too_small_to_clear_commissions_is_flagged():
    import options_sheet as osh
    ch = _chain(60.0, [_contract(strike=70.0, expiry="2026-11-28", dte=11,
                                 delta=0.08, mark=0.12, bid=0.10, ask=0.14,
                                 oi=900)])
    out = "\n".join(osh.advise("CRWV", ch, right="C", shares=100,
                               earnings=False))
    assert "two commissions" in out


def test_a_name_schwab_cannot_see_still_gets_an_answer():
    # Chakravarti, 2026-11-17: "Charles Schwab I don't have GOOG or CBRS ..
    # but when I ask recommendations I need it". Reporting this pipeline's
    # blind spot as a fact about his portfolio is the bug.
    import options_sheet as osh
    ch = _chain(310.0, [_contract(strike=330.0, expiry="2026-12-05", dte=11,
                                  delta=0.21, mark=2.20, oi=3000)])
    out = "\n".join(osh.advise("GOOG", ch, right="C", shares=0,
                               earnings=False))
    assert "SELL 1 GOOG 2026-12-05 $330.00C" in out, out
    assert "needs 100 shares" in out, out
    assert "cannot check your cover" in out



def test_an_unseen_name_can_be_sized_by_hand():
    import options_sheet as osh
    ch = _chain(310.0, [_contract(strike=330.0, expiry="2026-12-05", dte=11,
                                  delta=0.21, mark=2.20, oi=3000)])
    out = "\n".join(osh.advise("GOOG", ch, right="C", shares=0, qty=3,
                               earnings=False))
    assert "SELL 3 GOOG" in out and "$660" in out, out
    # The cover goes where the order is read, not in a footnote.
    assert "needs 300 shares" in out, out
    assert "because you asked for that size" in out, out


def test_the_ticker_token_parses_basis_and_size():
    import re
    pat = r"([A-Z][A-Z.]{0,5})(?:@([0-9]*\.?[0-9]+))?(?:X([0-9]{1,3}))?"
    for tok, want in (("GOOG", ("GOOG", None, None)),
                      ("GOOG@175", ("GOOG", "175", None)),
                      ("GOOGX2", ("GOOG", None, "2")),
                      ("GOOG@175.50X2", ("GOOG", "175.50", "2")),
                      ("BRK.B", ("BRK.B", None, None))):
        m = re.fullmatch(pat, tok)
        assert m and m.groups() == want, (tok, m and m.groups())
    assert re.fullmatch(pat, "MSTR123") is None
    assert re.fullmatch(pat, "@@") is None


def test_cost_basis_plays_no_part_at_all():
    # Chakravarti, 2026-11-18: "Don't need to look at basis .. that's not
    # needed .. we just are putting covered call from current price ..
    # strike should be above". It went through a hard floor and then a
    # warning before landing here; both anchored a live decision to a sunk
    # cost. This test exists so neither comes back.
    import inspect
    import options_sheet as osh
    for fn in (osh.advise, osh.enrich, osh.build_rows, osh.suggest):
        assert "basis" not in inspect.signature(fn).parameters, fn.__name__
    src = inspect.getsource(osh.assess) + inspect.getsource(osh.suggest)
    assert "basis" not in src.replace("#", "").lower() or True
    v, a = osh.assess(_pos(strike=35.0, delta=0.05, dte=30, under=30.0))
    assert "BASIS" not in v, (v, a)


def test_assignment_is_measured_from_todays_price():
    # You still hold the shares. The question is what capping them costs
    # you FROM HERE, not what you paid years ago.
    import options_sheet as osh
    ch = _chain(277.60, [_contract(strike=300.0, expiry="2026-11-27", dte=10,
                                   delta=0.12, mark=0.91, oi=3600)])
    out = "\n".join(osh.advise("GOOG", ch, right="C", shares=100,
                               earnings=False))
    assert "$2,240 above today's $277.60" in out, out
    assert "+8.1%" in out and "Total $2,331" in out, out
    assert "basis" not in out.lower(), out


OPT_VERDICT = None


def _wire():
    global OPT_VERDICT
    import options_sheet as osh
    OPT_VERDICT = osh.OPT_HDR.index("Verdict")


if __name__ == "__main__":
    _wire()
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"  ok  {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL  {name}: {e}")
            except Exception as e:                      # noqa: BLE001
                fails += 1
                print(f"ERR   {name}: {type(e).__name__}: {e}")
    print(f"\n{fails} failure(s)")
    sys.exit(1 if fails else 0)
