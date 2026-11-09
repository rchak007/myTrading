#!/usr/bin/env python3
"""
test_hyperliquid.py
===================
Hyperliquid perp positions. No network — fixtures only.
"""
import sys
sys.path.insert(0, "/home/chakravarti/github/myTrading")
import hyperliquid as hl        # noqa: E402


def acct(value, notional, margin, free=0.0):
    return {"label": "hyp-test", "value": value, "notional": notional,
            "margin": margin, "free": free, "pnl": 0.0, "positions": [],
            "error": ""}


def test_label_is_the_last_four_of_the_address():
    assert hl.label("0xAeE99E63A56FF967F535e736Fa17Fcae29A3236a") == "hyp-236a"
    assert hl.label("0x6ae3214643266BD6eD95C863a077763b07690c53") == "hyp-0c53"


def test_one_x_is_not_leveraged_however_much_margin_is_used():
    """THE BUG THIS GUARDS. At 1x the API still reports marginUsed, and it
    equals the notional because the full value is posted. The first version
    divided margin by value, read 100%, and printed 'almost no margin left'
    on two accounts carrying no leverage at all."""
    a = acct(value=1750.56, notional=1766.11, margin=1750.56, free=0.0)
    assert abs(hl.leverage(a) - 1.01) < 0.01
    assert "⚠️" not in hl.summary(a), hl.summary(a)
    assert "leveraged" not in hl.summary(a)


def test_real_leverage_is_reported():
    a = acct(value=5298.10, notional=5827.69, margin=5043.95, free=1.44)
    assert abs(hl.leverage(a) - 1.10) < 0.01


def test_a_levered_account_with_no_free_margin_is_warned_about():
    a = acct(value=1000.0, notional=2000.0, margin=950.0, free=1.0)
    s = hl.summary(a)
    assert "⚠️" in s and "2.00x" in s, s


def test_a_levered_account_with_room_is_noted_not_warned():
    a = acct(value=1000.0, notional=2000.0, margin=950.0, free=400.0)
    s = hl.summary(a)
    assert "⚠️" not in s and "2.00x leveraged" in s, s


def test_an_empty_account_does_not_divide_by_zero():
    assert hl.leverage(acct(0.0, 0.0, 0.0)) == 0.0


def test_an_unreachable_wallet_says_so_rather_than_reading_as_empty():
    """A dead API must not look like 'no positions' — that is the same
    failure shape as querying the wrong dex."""
    a = acct(0, 0, 0)
    a["error"] = "URLError: timed out"
    assert "unreachable" in hl.summary(a)


def test_rows_drop_the_dex_prefix_and_total_correctly():
    a = acct(value=100, notional=300, margin=200)
    a["pnl"] = 7.0
    a["positions"] = [
        {"ticker": "NVDA", "size": 1.0, "side": "LONG", "leverage": 2.0,
         "lev_type": "isolated", "entry": 180.0, "notional": 200.0,
         "margin": 100.0, "pnl": 5.0, "roe": 2.5, "liq": 69.4,
         "funding": -1.09},
        {"ticker": "AMZN", "size": 2.0, "side": "LONG", "leverage": 1.0,
         "lev_type": "isolated", "entry": 220.0, "notional": 100.0,
         "margin": 100.0, "pnl": 2.0, "roe": 2.0, "liq": 0.0,
         "funding": 0.5},
    ]
    r = hl.rows(a)
    assert r[0] == hl.HDR
    assert r[1][0] == "NVDA" and r[1][3] == "2x"
    assert r[-1][0] == "TOTAL" and r[-1][5] == 300.0 and r[-1][6] == 200.0
    # A zero liquidation price is blank, not "0.0" — it means no liq level,
    # not a liquidation at zero.
    assert r[2][9] == ""


def test_the_default_dex_is_xyz_not_the_main_one():
    """Querying without a dex returns accountValue 0 and no positions, which
    is indistinguishable from holding nothing."""
    assert hl.DEX == "xyz"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"  ok  {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL  {name}: {e}")
    print(f"\n{fails} failure(s)")
    sys.exit(1 if fails else 0)
