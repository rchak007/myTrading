#!/usr/bin/env python3
"""
test_order_json.py
==================
The Schwab order payload. No network, no client, no placement.

    .venv/bin/python test_order_json.py

build_order_json is pure precisely so this can run on Pi 2, where there are no
credentials — the shape of a live order gets checked without one being sent.

WHY THIS FILE EXISTS
    The session field was NORMAL for three weeks. Every close-triggered order
    submits after 16:00 ET by construction, so every one of them slept until
    the next morning: the EOSE sell on 2026-10-09 submitted at 13:3x PDT and
    sat overnight. Nothing errored. Nothing looked wrong on the sheet — the
    row said SUBMITTED, because it WAS submitted. It just could not fill.

    That is this codebase's recurring failure: a setting that is wrong in a
    way that looks like a normal result. So it gets a test.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import order_engine as oe                                  # noqa: E402
import order_exec_config as cfg                            # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def intent(**kw):
    n = {"Qty": "700", "Limit_Price": "2.65", "Side": "SELL", "Ticker": "EOSE"}
    n.update(kw)
    return n


print("\n── GTC + extended hours ──")
b = oe.build_order_json(intent())
check("session is SEAMLESS, not NORMAL", b["session"] == "SEAMLESS", b["session"])
check("duration is GOOD_TILL_CANCEL",
      b["duration"] == "GOOD_TILL_CANCEL", b["duration"])
check("the config is what the payload uses",
      b["session"] == cfg.ORDER_SESSION, f"{b['session']} vs {cfg.ORDER_SESSION}")
# SEAMLESS is only legal because every order here is a LIMIT — Schwab rejects
# market orders in extended hours. These two facts must stay true together.
check("orderType is LIMIT, which is what makes extended hours legal",
      b["orderType"] == "LIMIT", b["orderType"])
check("market orders are still disallowed", cfg.ALLOW_MARKET_ORDERS is False)

print("\n── the rest of the payload ──")
leg = b["orderLegCollection"][0]
check("single order", b["orderStrategyType"] == "SINGLE")
check("price is a 2dp string", b["price"] == "2.65", b["price"])
check("side carried through", leg["instruction"] == "SELL")
check("quantity is a whole number", leg["quantity"] == 700
      and isinstance(leg["quantity"], int), str(leg["quantity"]))
check("symbol carried through", leg["instrument"]["symbol"] == "EOSE")
check("equity asset type", leg["instrument"]["assetType"] == "EQUITY")

print("\n── the limit override ──")
b2 = oe.build_order_json(intent(Limit_Price=""), limit_price=2.71)
check("a derived limit is used when the sheet left it blank",
      b2["price"] == "2.71", b2["price"])
b3 = oe.build_order_json(intent(Limit_Price="2.65"), limit_price=2.71)
check("an explicit limit_price argument wins", b3["price"] == "2.71", b3["price"])
check("rounding is to the cent",
      oe.build_order_json(intent(), limit_price=2.6549)["price"] == "2.65",
      oe.build_order_json(intent(), limit_price=2.6549)["price"])
check("a BUY builds too",
      oe.build_order_json(intent(Side="BUY"))["orderLegCollection"][0]
      ["instruction"] == "BUY")

print("\n── refuses rather than guesses ──")
for label, kw in (("no limit price anywhere", dict(Limit_Price="")),
                  ("zero quantity", dict(Qty="0")),
                  ("negative quantity", dict(Qty="-5")),
                  ("quantity rounds to zero", dict(Qty="0.4"))):
    try:
        oe.build_order_json(intent(**kw))
        check(f"{label} raises", False, "built an order instead")
    except (ValueError, TypeError, KeyError):
        check(f"{label} raises", True)

print(f"\n{'ALL PASS' if not FAILED else str(len(FAILED)) + ' FAILED: ' + ', '.join(FAILED)}")
raise SystemExit(1 if FAILED else 0)
