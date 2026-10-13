#!/usr/bin/env python3
"""
test_reserve_fills.py
=====================
A buy must never invent a negative reserve. No network, no sheet, no Schwab.

    .venv/bin/python test_reserve_fills.py

THE BUG THIS PINS, 2026-10-13
    Chakravarti bought SNDK, SKHY and CBRS by hand in 171. All three had an
    ACTIVE reserve row holding $0 — fenced, or seeded and long since spent.
    apply_fills debited the FULL purchase against that $0:

        SNDK  -2,970.94      exactly the purchase
        SKHY    -921.17      exactly the purchase
        CBRS  -1,476.90      exactly the purchase

    Then build_cash_rows summed those negatives into the account's reserved
    total, which SUBTRACTS — so Free_To_Deploy read $7,580.71 when the honest
    figure was $2,211.70. The money had genuinely left the account at Schwab,
    and the phantom negative handed it straight back as spendable.

    Both ends are tested here: the debit that creates the negative, and the
    sum that turns it into free cash.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd                                        # noqa: E402
import cash_reserve as cr                                  # noqa: E402
from orders_sheet import build_cash_rows                   # noqa: E402

FAILED = []
QUIET = lambda *a, **k: None                               # noqa: E731


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def bench(seed_amount, tickers=("SNDK",)):
    """A ledger + config with one seeded reserve per ticker."""
    d = Path(tempfile.mkdtemp())
    led, cfgp = d / "ledger.csv", d / "config.csv"
    lines = ["Account,Ticker,Active,Policy,Seed_Cash,Target_Capital,Effective_From"]
    for t in tickers:
        lines.append(f"171,{t},True,,{seed_amount},,")
    cfgp.write_text("\n".join(lines) + "\n")
    for t in tickers:
        if seed_amount:
            cr.seed("171", t, seed_amount, ledger_path=led, config_path=cfgp,
                    log=QUIET)
    return led, cfgp, cr.read_config(cfgp)


def fill(ref, ticker, side, amt, when="2026-10-12T10:00:00"):
    return {"Ref": ref, "Account": "171", "Ticker": ticker, "Side": side,
            "Fill_QTY": 1, "Fill_Price": amt, "Net_Amount": amt,
            "Asset_Type": "EQUITY", "Fill_Time": when}


print("\n── a buy against an EMPTY reserve ──")
led, cfgp, cfg = bench(0)
out = cr.apply_fills(pd.DataFrame([fill("A1", "SNDK", "BUY", 2970.94)]),
                     config=cfg, ledger_path=led, config_path=cfgp, log=QUIET)
bal = cr.balance_of("171", "SNDK", ledger_path=led)
check("the reserve stays at zero, not -2,970.94", abs(bal) < 0.005, f"{bal}")
check("nothing is debited", abs(float(out.iloc[0]["Amount"])) < 0.005,
      str(out.iloc[0]["Amount"]))
check("...and the row SAYS where the money came from",
      "ordinary cash" in str(out.iloc[0]["Reason"]), str(out.iloc[0]["Reason"]))

print("\n── a buy that OVERDRAWS a funded reserve ──")
led, cfgp, cfg = bench(1000.0)
out = cr.apply_fills(pd.DataFrame([fill("B1", "SNDK", "BUY", 2970.94)]),
                     config=cfg, ledger_path=led, config_path=cfgp, log=QUIET)
bal = cr.balance_of("171", "SNDK", ledger_path=led)
check("debits exactly what was reserved",
      abs(float(out.iloc[0]["Amount"]) + 1000.0) < 0.005,
      str(out.iloc[0]["Amount"]))
check("lands on zero, never below", abs(bal) < 0.005, f"{bal}")
check("the split is named in the reason",
      "1,000.00" in str(out.iloc[0]["Reason"])
      and "1,970.94" in str(out.iloc[0]["Reason"]), str(out.iloc[0]["Reason"]))

print("\n── a buy WITHIN the reserve is untouched ──")
# The live case that proved the model: AMD held 1,826.96, he bought 1,719.00,
# and the Dashboard read 107.96. Exactly.
led, cfgp, cfg = bench(1826.96, tickers=("AMD",))
out = cr.apply_fills(pd.DataFrame([fill("C1", "AMD", "BUY", 1719.00)]),
                     config=cfg, ledger_path=led, config_path=cfgp, log=QUIET)
bal = cr.balance_of("171", "AMD", ledger_path=led)
check("debits the full amount", abs(float(out.iloc[0]["Amount"]) + 1719.0) < 0.005,
      str(out.iloc[0]["Amount"]))
check("leaves 107.96, matching the live Dashboard", abs(bal - 107.96) < 0.005,
      f"{bal}")
check("no overdraft note on a normal buy",
      "ordinary cash" not in str(out.iloc[0]["Reason"]), str(out.iloc[0]["Reason"]))

print("\n── SELLS are not clamped ──")
# Fencing exists so sale proceeds come BACK to the ticker. Clamping a credit
# would break the thing reserves are for.
led, cfgp, cfg = bench(0)
out = cr.apply_fills(pd.DataFrame([fill("D1", "SNDK", "SELL", 1500.0)]),
                     config=cfg, ledger_path=led, config_path=cfgp, log=QUIET)
check("a sell credits in full even from zero",
      abs(float(out.iloc[0]["Amount"]) - 1500.0) < 0.005, str(out.iloc[0]["Amount"]))
check("the reserve rises to the proceeds",
      abs(cr.balance_of("171", "SNDK", ledger_path=led) - 1500.0) < 0.005)

print("\n── an unfenced ticker is still skipped entirely ──")
led, cfgp, cfg = bench(0)
out = cr.apply_fills(pd.DataFrame([fill("E1", "NVDA", "BUY", 5000.0)]),
                     config=cfg, ledger_path=led, config_path=cfgp, log=QUIET)
check("no reserve row means SKIP_BUY, not a debit",
      str(out.iloc[0]["Event"]) == "SKIP_BUY", str(out.iloc[0]["Event"]))

print("\n── the replay guard still holds ──")
led, cfgp, cfg = bench(5000.0)
f1 = pd.DataFrame([fill("F1", "SNDK", "BUY", 1000.0)])
cr.apply_fills(f1, config=cfg, ledger_path=led, config_path=cfgp, log=QUIET)
cr.apply_fills(f1, config=cfg, ledger_path=led, config_path=cfgp, log=QUIET)
check("the same fill applied twice debits once",
      abs(cr.balance_of("171", "SNDK", ledger_path=led) - 4000.0) < 0.005,
      str(cr.balance_of("171", "SNDK", ledger_path=led)))

print("\n── a negative reserve is NOT spendable cash ──")
cash = pd.DataFrame([{"Account": "171", "Nickname": "PCRA", "Cash": 44472.86,
                      "Cash_In_Open_Orders": 9275.0,
                      "Cash_After_Open_Orders": 35197.86}])
# The real 2026-10-13 numbers.
res = {("171", "SNDK"): -2970.94, ("171", "SKHY"): -921.17,
       ("171", "CBRS"): -1476.90, ("171", "REST"): 32986.16}
row = build_cash_rows(cash, res, log=QUIET).iloc[0]
check("negatives do not reduce the reserved total",
      abs(float(row["Seed_Reserved"]) - 32986.16) < 0.005,
      str(row["Seed_Reserved"]))
check("Free_To_Deploy is 2,211.70, not 7,580.71",
      abs(float(row["Free_To_Deploy"]) - 2211.70) < 0.005,
      str(row["Free_To_Deploy"]))
check("an all-positive account is unaffected",
      abs(float(build_cash_rows(
          cash, {("171", "A"): 1000.0, ("171", "B"): 2000.0},
          log=QUIET).iloc[0]["Free_To_Deploy"]) - 32197.86) < 0.005)

print(f"\n{'ALL PASS' if not FAILED else str(len(FAILED)) + ' FAILED: ' + ', '.join(FAILED)}")
raise SystemExit(1 if FAILED else 0)
