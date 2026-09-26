#!/usr/bin/env python3
"""
arm_order.py
============
Mint the Confirm_Token for one order intent, and print the row to paste.

    .venv/bin/python arm_order.py \\
        --row-id 2026-09-25-AVGO-01 --acct 431 --ticker AVGO \\
        --action BUY-BREAKOUT --trigger 245.00 --limit 246.23 \\
        --qty 30 --after-close Y --expires 2026-10-31

Run it where the HMAC key is: Pi 1, or the Dell. Never on Pi 2 — that machine
deliberately holds no credentials.

WHY ARMING IS A SEPARATE STEP
    Typing a row into the sheet is not authorisation. The sheet is shared, is
    edited from a phone, and is exactly the sort of surface where a stray paste
    or an errant formula becomes a number. The token proves that a specific
    intent was approved by someone holding the secret.

    So the flow is deliberately two-handed: decide the trade here, then paste
    the result. Anything the engine finds in the sheet WITHOUT a matching token
    is refused, not executed.

It prints; it writes nothing. Confirming the summary before pasting is the
point of the exercise — read it back against what you meant to do.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import order_canonical as oc                            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Mint the Confirm_Token for an order intent",
        epilog="Actions: " + " | ".join(sorted(oc.ACTIONS)))
    ap.add_argument("--row-id", required=True,
                    help="unique and never reused, e.g. 2026-09-25-AVGO-01")
    ap.add_argument("--acct", required=True, help="LAST 3 DIGITS, e.g. 431")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--action", required=True, choices=sorted(oc.ACTIONS))
    ap.add_argument("--trigger", required=True, help="the condition price")
    ap.add_argument("--limit", default="",
                    help="limit price. Omit only if you mean a market order, "
                         "which the engine refuses by default")
    ap.add_argument("--qty", required=True)
    ap.add_argument("--qty-unit", default="SHARES", choices=sorted(oc.QTY_UNITS))
    ap.add_argument("--after-close", default="N", choices=["Y", "N"],
                    help="Y = Pi 1 watches the daily close and submits the next "
                         "morning. N = the order rests at Schwab (preferred)")
    ap.add_argument("--expires", default="", help="YYYY-MM-DD")
    args = ap.parse_args()

    intent = {
        "Row_ID": args.row_id, "Acct": args.acct, "Ticker": args.ticker,
        "Action": args.action, "Trigger_Price": args.trigger,
        "Limit_Price": args.limit, "Qty": args.qty, "Qty_Unit": args.qty_unit,
        "After_Close": args.after_close, "Expires_On": args.expires,
    }

    try:
        n = oc.normalize(intent)
        token = oc.token_for(intent)
    except oc.IntentError as e:
        print(f"\n❌ {e}\n", file=sys.stderr)
        return 2

    print()
    print("  " + oc.describe(intent))
    print()
    if n["After_Close"] == "Y":
        # Stated every time, because it is the single most surprising property
        # of a close-based trigger and the easiest to forget.
        print("  ⚠️  A close trigger submits the NEXT TRADING MORNING, not at")
        print("      the close that fired it. The daily bar has to be final")
        print("      before it can be evaluated.")
        print()
    if not n["Limit_Price"]:
        print("  ⚠️  No limit price. The engine refuses market orders by")
        print("      default — a gap open turns 'buy above 245' into a fill")
        print("      at 261. Add --limit.")
        print()

    print(f"  canonical      : {oc.canonical(intent)}")
    print(f"  Confirm_Token  : {token}")
    print()
    print("  Paste into the Orders tab, columns A-L:")
    print()
    cols = ["Row_ID", "Date", "Acct", "Ticker", "Action", "Trigger_Price",
            "Limit_Price", "Qty", "Qty_Unit", "After_Close", "Expires_On",
            "Confirm_Token"]
    from datetime import date
    values = [n["Row_ID"], str(date.today()), n["Acct"], n["Ticker"],
              n["Action"], n["Trigger_Price"], n["Limit_Price"], n["Qty"],
              n["Qty_Unit"], n["After_Close"], n["Expires_On"], token]
    print("    " + "\t".join(values))
    print()
    print("    " + "  ".join(f"{c}={v}" for c, v in zip(cols, values) if v))
    print()
    print("  Any edit to an intent cell invalidates the token and the engine")
    print("  will refuse the row. Re-arm rather than editing in place.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
