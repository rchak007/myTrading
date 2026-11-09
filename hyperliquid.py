#!/usr/bin/env python3
"""
hyperliquid.py
==============
Perp positions from Hyperliquid, read-only, for the Dashboard.

Chakravarti, 2026-11-08, after sending three wallet addresses: *"Yes I want
these on dashboard & you know we acct number instead say it's hyp-236a for
last 4 digits of address."*

PUBLIC DATA, NO KEYS. An address is the whole input — `clearinghouseState`
needs no signature and no API key. Nothing here can move funds, and nothing
here needs a secret on Pi 1. The addresses are on-chain and public already;
they are identifiers, not credentials.

THE POSITIONS ARE NOT ON THE MAIN DEX. Querying `clearinghouseState` with
just an address returns an account value of $0.00 and no positions, which
looks exactly like "nothing held". They are on **xyz**, one of eleven
builder-deployed perp DEXes (xyz, flx, vntl, hyna, km, abcd, cash, para,
mkts, io). The `dex` key is required, and omitting it fails silently with a
plausible-looking empty answer — the worst shape of bug.

LEVERAGE: SHOW BOTH NUMBERS, NEVER ONE. His question, and the right one:
*"it's a little complicated when I have a 2x level perpetual... how to still
show actually only the position value."* Two different quantities, and the
API gives both, so neither has to be derived:

    positionValue   EXPOSURE — what moves when the price moves.
                    NVDA 11.473 shares = $2,070.65.
    marginUsed      CAPITAL — what is actually committed to hold it.
                    The same position = $1,293.93.

Printing notional beside a Schwab cash balance would overstate how much
money is deployed; printing margin alone would hide that a 2x position
moves twice as fast as the capital behind it. Both, labelled, always.

FUNDING IS A REAL COST AND IS INVISIBLE IN PnL. A perp pays or receives
funding every hour, and `cumFunding.sinceOpen` is what this position has
paid since it was opened. A spot holding has no equivalent, so it is easy
to forget it is there; it is carried as its own column.
"""
from __future__ import annotations

import json
import os
import urllib.request

API = "https://api.hyperliquid.xyz/info"
DEX = os.getenv("HYPERLIQUID_DEX", "xyz")
TIMEOUT = int(os.getenv("HYPERLIQUID_TIMEOUT", "30"))

# Public wallet addresses, not secrets. Overridable so a new wallet needs no
# code change — comma-separated in HYPERLIQUID_ADDRESSES.
ADDRESSES = [a.strip() for a in os.getenv(
    "HYPERLIQUID_ADDRESSES",
    "0xAeE99E63A56FF967F535e736Fa17Fcae29A3236a,"
    "0x6ae3214643266BD6eD95C863a077763b07690c53,"
    "0x1d826e8c2461e12f3c3195720f0b0f25e42130a9").split(",") if a.strip()]


def label(address: str) -> str:
    """'hyp-236a' — his naming, the last four of the address.

    Short enough to sit in the Acct column beside 171 and 431, and unique
    enough in practice; a full 42-character address would not fit anywhere
    useful.
    """
    return f"hyp-{str(address).strip()[-4:].lower()}"


def _post(body: dict) -> dict:
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "User-Agent": "myTrading research"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def _f(v, d=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def account(address: str, dex: str = DEX, log=None) -> dict:
    """One wallet: {'label','address','value','notional','margin','free',
    'positions':[...]}. Raises nothing — a dead API costs the block.
    """
    out = {"label": label(address), "address": address, "value": 0.0,
           "notional": 0.0, "margin": 0.0, "free": 0.0, "pnl": 0.0,
           "positions": [], "error": ""}
    try:
        body = {"type": "clearinghouseState", "user": address}
        if dex:
            body["dex"] = dex
        d = _post(body)
    except Exception as e:                               # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        if log:
            log(f"   hyperliquid {out['label']}: {out['error']}")
        return out

    m = d.get("marginSummary") or {}
    out["value"] = _f(m.get("accountValue"))
    out["notional"] = _f(m.get("totalNtlPos"))
    out["margin"] = _f(m.get("totalMarginUsed"))
    out["free"] = _f(d.get("withdrawable"))

    for ap in d.get("assetPositions") or []:
        p = ap.get("position") or {}
        sz = _f(p.get("szi"))
        if not sz:
            continue
        lev = (p.get("leverage") or {})
        pnl = _f(p.get("unrealizedPnl"))
        out["pnl"] += pnl
        out["positions"].append({
            # "xyz:NVDA" -> "NVDA". The dex prefix is the same for every row
            # in the block and only costs width.
            "ticker": str(p.get("coin", "")).split(":")[-1].upper(),
            "size": sz,
            "side": "LONG" if sz > 0 else "SHORT",
            "leverage": _f(lev.get("value"), 1.0),
            "lev_type": str(lev.get("type", "")),
            "entry": _f(p.get("entryPx")),
            "notional": _f(p.get("positionValue")),
            "margin": _f(p.get("marginUsed")),
            "pnl": pnl,
            "roe": _f(p.get("returnOnEquity")) * 100.0,
            "liq": _f(p.get("liquidationPx")),
            "funding": _f((p.get("cumFunding") or {}).get("sinceOpen")),
        })
    out["positions"].sort(key=lambda r: -abs(r["notional"]))
    return out


def fetch(addresses=None, dex: str = DEX, log=None) -> list[dict]:
    return [account(a, dex, log) for a in (addresses or ADDRESSES)]


def leverage(acct: dict) -> float:
    """Exposure as a multiple of account value. 1.0 means no borrowing.

    NOT margin-used-over-value, which was the first version and was wrong.
    Chakravarti, 2026-11-08: *"when I buy 1x leverage, I don't have any
    margin."* Exactly — at 1x the API still reports marginUsed, but it
    equals the notional (SPCX: $2,041.01 notional against $2,035.91
    margin), because the full value is posted and nothing is borrowed.

    So margin utilisation reads 100% on a wholly unlevered account and the
    first draft of this file printed "almost no margin left" on two
    accounts carrying no leverage whatsoever. Notional over value is the
    honest number: 1.0x is fully invested cash, and only above that is
    anything actually borrowed.
    """
    v = _f(acct.get("value"))
    return (_f(acct.get("notional")) / v) if v > 0 else 0.0


HDR = ["Ticker", "Side", "Size", "Lev", "Entry", "Notional", "Margin",
       "PnL", "ROE_%", "Liq", "Funding"]


def rows(acct: dict) -> list[list]:
    """[[header], [position...], [TOTAL]] for one wallet."""
    out = [list(HDR)]
    for p in acct["positions"]:
        out.append([
            p["ticker"], p["side"], round(p["size"], 4),
            f"{p['leverage']:g}x", round(p["entry"], 2),
            round(p["notional"], 2), round(p["margin"], 2),
            round(p["pnl"], 2), round(p["roe"], 1),
            round(p["liq"], 2) if p["liq"] else "",
            round(p["funding"], 2),
        ])
    if acct["positions"]:
        out.append(["TOTAL", "", "", "", "",
                    round(acct["notional"], 2), round(acct["margin"], 2),
                    round(acct["pnl"], 2), "", "", ""])
    return out


def summary(acct: dict) -> str:
    if acct.get("error"):
        return f"{acct['label']}: unreachable — {acct['error']}"
    lev = leverage(acct)
    s = (f"{acct['label']}  value ${acct['value']:,.2f} · "
         f"exposure ${acct['notional']:,.2f} ({lev:.2f}x) · "
         f"free ${acct['free']:,.2f} · "
         f"unrealized ${acct['pnl']:+,.2f}")
    # Only borrowed money gets a warning. Being fully invested is a choice;
    # being fully invested with leverage on top is a risk.
    if lev > 1.25 and _f(acct.get("free")) < 0.02 * _f(acct.get("value")):
        s += f"  ⚠️ {lev:.2f}x leveraged with no free margin"
    elif lev > 1.25:
        s += f"  · {lev:.2f}x leveraged"
    return s


def by_ticker(accounts) -> dict[str, list[dict]]:
    """{TICKER: [position, ...]} across every wallet, for the Dashboard."""
    out: dict[str, list[dict]] = {}
    for a in accounts or []:
        for p in a["positions"]:
            out.setdefault(p["ticker"], []).append(dict(p, label=a["label"]))
    return out


def dashboard_rows(ticker: str, index: dict, width: int) -> list[list]:
    """The hyp-xxxx line(s) for one ticker block on the Dashboard.

    OUTSIDE HIS EQUITY TOTAL, like Chitra's line, but for a different
    reason. Hers is excluded because the money is not his; these are his,
    and are excluded because a 2x perp is not a share — it carries
    leverage, funding and a liquidation price, and adding its notional to a
    column of share counts would make the total mean nothing.

    Inside the ticker's block all the same, because that is where the
    question gets answered: long NVDA at Schwab AND 2x long NVDA here is
    one exposure, and seeing it in two places is how it gets missed.

    Coverage columns stay blank: there is no stop to place on a perp
    through this system, and asserting a flag would imply there is.
    """
    out = []
    for p in index.get(str(ticker).upper(), []):
        row = [""] * width
        # Ticker, Acct, Qty, Avg_Cost, Live_Price, Day_%, Market_Value,
        # Unrealized_PL — matching POS_HDR's leading columns.
        row[1:9] = [p["ticker"], p["label"], round(p["size"], 4),
                    round(p["entry"], 4), "", "",
                    round(p["notional"], 2), round(p["pnl"], 2)]
        note = f"{p['leverage']:g}x {p['lev_type'] or 'perp'}"
        if p["liq"]:
            note += f" · liq {p['liq']:,.2f}"
        if p["funding"]:
            note += f" · funding {p['funding']:+,.2f}"
        row[9] = note
        out.append(row)
    return out


def summary_rows(accounts, width: int, shown=None) -> list[list]:
    """The HYPERLIQUID section: one line per wallet, then a total.

    `shown` is the set of tickers that got a Dashboard block. Any perp in a
    name NOT in it is listed explicitly, because the per-ticker lines hang
    off existing blocks and a position in something held ONLY here would
    otherwise appear nowhere at all — the silent-empty failure this file
    already warns about for the wrong-dex query.
    """
    accounts = [a for a in (accounts or []) if a["positions"] or a["error"]]
    if not accounts:
        return []
    out = [_pad(["", "HYPERLIQUID", "perps — outside the equity total"],
                width),
           _pad(["", "Wallet", "Value", "Exposure", "Leverage", "Free",
                 "Unrealized"], width)]
    tv = tn = tp = 0.0
    for a in accounts:
        if a["error"]:
            out.append(_pad(["", a["label"], f"unreachable — {a['error']}"],
                            width))
            continue
        tv += a["value"]
        tn += a["notional"]
        tp += a["pnl"]
        out.append(_pad(["", a["label"], round(a["value"], 2),
                         round(a["notional"], 2), f"{leverage(a):.2f}x",
                         round(a["free"], 2), round(a["pnl"], 2)], width))
    out.append(_pad(["", "TOTAL", round(tv, 2), round(tn, 2),
                     f"{(tn / tv):.2f}x" if tv else "", "",
                     round(tp, 2)], width))

    if shown is not None:
        seen = {t.upper() for t in shown}
        orphan = [(a["label"], p) for a in accounts
                  for p in a["positions"] if p["ticker"] not in seen]
        if orphan:
            out.append(_pad(["", "ONLY ON HYPERLIQUID",
                             "no equity block for these"], width))
            for lab, p in sorted(orphan, key=lambda x: x[1]["ticker"]):
                out.append(_pad(
                    ["", p["ticker"], lab, round(p["size"], 4),
                     f"{p['leverage']:g}x", round(p["notional"], 2),
                     round(p["pnl"], 2)], width))
    return out


def _pad(cells, width):
    return list(cells)[:width] + [""] * max(0, width - len(cells))


def main() -> int:
    for a in fetch(log=print):
        print(summary(a))
        for r in rows(a):
            print("   " + " ".join(f"{str(c):>10}" for c in r))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
