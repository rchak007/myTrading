#!/usr/bin/env python3
"""
option_chain.py — option quotes, greeks and roll candidates from Schwab.
========================================================================

WHY A CHAIN AND NOT A QUOTE. The contract you are short and the contracts you
might roll INTO come out of the same `/marketdata/v1/chains` response, so one
call answers both "where am I" and "where could I go". Quoting the single
symbol you hold would tell you the first and leave you blind to the second,
for the same network cost.

ONE CALL PER UNDERLYING, CACHED. Eight or nine underlyings, three runs a day
is nothing — but the options reminder channel fires at 09:30, 11:00 and 12:30
PT and a human will also run it by hand, so a short TTL keeps a burst of runs
from turning into a burst of calls.

GREEKS ARE SCHWAB'S, NOT OURS. Nothing here prices an option. Black-Scholes
with the wrong rate and the wrong dividend assumption produces a confident
number that disagrees with the broker you actually trade at, which is worse
than no number. `_g()` exists only to turn Schwab's sentinels into None.

IV RANK IS ACCUMULATED, NOT FETCHED. Nobody sells you your own history. One
ATM implied-vol reading per underlying per day is appended to a CSV, and the
rank is this reading's percentile within it. It is worthless until roughly a
month of readings exist, and `iv_rank()` returns None rather than a flattering
number until then.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

STATE_DIR = Path(os.environ.get(
    "MYTRADING_STATE_DIR", Path.home() / ".local" / "state" / "myTrading"))
CACHE_DIR = Path(os.getenv("OPTION_CACHE_DIR", STATE_DIR / "option_chains"))
IV_PATH = Path(os.getenv("OPTION_IV_HISTORY", STATE_DIR / "option_iv.csv"))

TTL_MIN = float(os.getenv("OPTION_CHAIN_TTL_MIN", "15"))
DAYS_OUT = int(os.getenv("OPTION_CHAIN_DAYS", "75"))
IV_MIN_READINGS = 20          # below this, a "rank" is just noise with a %

# Schwab reports an unavailable greek as -999.0, and occasionally as the
# STRING "NaN". Both must become None: -999 delta sorts as deep OTM and would
# quietly be chosen as the safest strike on the board.
_SENTINELS = {-999.0, -999, 999.0}


# ───────────────────────────────────────────────────────────── symbols
def osi_symbol(root: str, expiry: str, right: str, strike: float) -> str:
    """OSI symbol as Schwab spells it: root padded to 6, then YYMMDD, C/P,
    strike x1000 in 8 digits.

        CRWV  261120C00060000
        └root┘└YYMMDD┘│└strike┘
                      └ C or P
    """
    y, m, d = expiry.split("-")
    return (f"{root.upper():<6}{y[2:]}{m}{d}"
            f"{right.upper()[:1]}{int(round(float(strike) * 1000)):08d}")


# ───────────────────────────────────────────────────────────── parsing
def _g(v):
    """A greek, or None. See the module docstring on sentinels."""
    if v is None:
        return None
    if isinstance(v, str):
        if v.strip().upper() in ("NAN", "", "N/A"):
            return None
        try:
            v = float(v)
        except ValueError:
            return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f in _SENTINELS or math.isnan(f) or math.isinf(f):
        return None
    return f


def _mark(c: dict):
    """What it would actually cost to close.

    MID, NOT LAST. `last` can be hours old on a quiet strike and will
    cheerfully tell you a call is worthless when the ask is sixty cents.
    Schwab's own `mark` is used when present; otherwise the bid/ask midpoint;
    `last` is the final fallback and is flagged by being the only path that
    can disagree with a live quote.
    """
    m = _g(c.get("mark"))
    if m is not None and m > 0:
        return m
    bid, ask = _g(c.get("bid")), _g(c.get("ask"))
    if bid is not None and ask is not None and ask > 0:
        return round((bid + ask) / 2.0, 4)
    return _g(c.get("last"))


def parse_chain(payload: dict) -> dict:
    """Schwab's nested expiry->strike->[contract] maps, flattened.

    Pure, so the whole scoring layer can be tested against a fixture without
    a broker. Returns {"underlying","price","contracts":[...],"asof"}.
    """
    out = {"underlying": str(payload.get("symbol") or "").upper(),
           "price": _g(payload.get("underlyingPrice")),
           "contracts": [], "asof": datetime.now().isoformat(timespec="seconds")}
    if out["price"] is None:
        u = payload.get("underlying") or {}
        out["price"] = _g(u.get("last")) or _g(u.get("mark"))

    for key, right in (("callExpDateMap", "C"), ("putExpDateMap", "P")):
        for exp_key, strikes in (payload.get(key) or {}).items():
            # "2026-11-20:3" — the date, then days to expiration.
            exp = str(exp_key).split(":")[0]
            for _strike, lst in (strikes or {}).items():
                for c in lst or []:
                    bid, ask = _g(c.get("bid")), _g(c.get("ask"))
                    out["contracts"].append({
                        "symbol": str(c.get("symbol") or "").strip(),
                        "right": right,
                        "expiry": exp,
                        "strike": _g(c.get("strikePrice")),
                        "bid": bid,
                        "ask": ask,
                        "mark": _mark(c),
                        "spread": (round(ask - bid, 4)
                                   if bid is not None and ask is not None else None),
                        "delta": _g(c.get("delta")),
                        "gamma": _g(c.get("gamma")),
                        "theta": _g(c.get("theta")),
                        "vega": _g(c.get("vega")),
                        "iv": _g(c.get("volatility")),      # percent, e.g. 48.3
                        "oi": _g(c.get("openInterest")) or 0,
                        "volume": _g(c.get("totalVolume")) or 0,
                        "dte": int(_g(c.get("daysToExpiration")) or 0),
                        "itm": bool(c.get("inTheMoney")),
                    })
    out["contracts"] = [c for c in out["contracts"]
                        if c["strike"] is not None and c["symbol"]]
    return out


# ───────────────────────────────────────────────────────────── fetching
def _raw_client(client_wrapper):
    """The underlying schwabdev client. Same probe stocks_orders uses."""
    from stocks_orders import _raw_client as rc
    return rc(client_wrapper)


def _call_chain(client, symbol: str, days_out: int):
    """schwabdev has renamed this method across versions; try each name."""
    frm = date.today()
    to = frm + timedelta(days=days_out)
    last = None
    for name in ("option_chains", "option_chain", "optionChains", "chains"):
        fn = getattr(client, name, None)
        if not callable(fn):
            continue
        try:
            return fn(symbol=symbol, contractType="ALL", strategy="SINGLE",
                      includeUnderlyingQuote=True,
                      fromDate=frm.isoformat(), toDate=to.isoformat())
        except TypeError as e:                          # different kwarg names
            last = e
            try:
                return fn(symbol, contractType="ALL",
                          fromDate=frm.isoformat(), toDate=to.isoformat())
            except Exception as e2:                     # noqa: BLE001
                last = e2
        except Exception as e:                          # noqa: BLE001
            last = e
    raise RuntimeError(f"no usable option-chain method on the client ({last})")


def _cache_file(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}.json"


def fetch_chain(client_wrapper, underlying: str, *, days_out: int = DAYS_OUT,
                ttl_min: float = TTL_MIN, log=print) -> dict | None:
    """One underlying's chain, cached.

    ON FAILURE, RETURNS THE STALE CACHE. A chain from an hour ago still says
    which strikes exist and roughly what they pay, and "roughly" beats a blank
    row. The age is carried on the result so the caller can say so out loud.
    """
    sym = str(underlying).upper().strip()
    f = _cache_file(sym)
    if ttl_min > 0 and f.exists():
        age_min = (time.time() - f.stat().st_mtime) / 60.0
        if age_min < ttl_min:
            try:
                d = json.loads(f.read_text())
                d["age_min"] = round(age_min, 1)
                return d
            except Exception:                           # noqa: BLE001
                pass

    try:
        resp = _call_chain(_raw_client(client_wrapper), sym, days_out)
        payload = resp.json() if hasattr(resp, "json") else resp
        if str(payload.get("status") or "").upper() not in ("SUCCESS", ""):
            raise RuntimeError(f"status={payload.get('status')}")
        chain = parse_chain(payload)
        if not chain["contracts"]:
            raise RuntimeError("no contracts in response")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(chain))
        tmp.replace(f)
        chain["age_min"] = 0.0
        return chain
    except Exception as e:                              # noqa: BLE001
        if f.exists():
            try:
                d = json.loads(f.read_text())
                d["age_min"] = round((time.time() - f.stat().st_mtime) / 60.0, 1)
                log(f"⚠️  {sym} chain fetch failed ({e}) — using cache "
                    f"{d['age_min']:.0f} min old")
                return d
            except Exception:                           # noqa: BLE001
                pass
        log(f"⚠️  {sym} chain unavailable: {e}")
        return None


# ───────────────────────────────────────────────────── implied-vol rank
def atm_iv(chain: dict, *, dte_lo: int = 20, dte_hi: int = 45):
    """One number for "how expensive is this underlying's vol right now".

    The nearest-the-money call in a 20-45 day window — the part of the curve
    a monthly premium seller actually lives on. A 2-day contract's IV is
    mostly gamma panic and says nothing about whether selling a month out is
    worth it.
    """
    px = chain.get("price")
    if not px:
        return None
    best, best_key = None, None
    for c in chain.get("contracts", []):
        if c["right"] != "C" or c["iv"] is None or c["iv"] <= 0:
            continue
        if not (dte_lo <= c["dte"] <= dte_hi):
            continue
        key = abs(c["strike"] - px)
        if best_key is None or key < best_key:
            best, best_key = c["iv"], key
    return best


def record_iv(underlying: str, iv, *, path: Path = IV_PATH, today=None) -> None:
    """Append today's reading, once per day. Append-only, like every other
    ledger here — a rank computed off a file that gets rewritten is a rank
    whose history you cannot audit."""
    if iv is None:
        return
    day = (today or date.today()).isoformat()
    rows = read_iv(path)
    if any(r["Date"] == day and r["Ticker"] == underlying.upper() for r in rows):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["Date", "Ticker", "IV"])
        if new:
            w.writeheader()
        w.writerow({"Date": day, "Ticker": underlying.upper(),
                    "IV": round(float(iv), 4)})


def read_iv(path: Path = IV_PATH) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            return [r for r in csv.DictReader(fh) if r.get("Ticker")]
    except Exception:                                   # noqa: BLE001
        return []


def iv_rank(underlying: str, iv, *, path: Path = IV_PATH, rows=None):
    """Percentile of `iv` within this ticker's own history, or None.

    None until IV_MIN_READINGS days exist. A "rank" off four readings is
    always 0 or 100 and would be read as a signal.
    """
    if iv is None:
        return None
    hist = [float(r["IV"]) for r in (rows if rows is not None else read_iv(path))
            if r.get("Ticker", "").upper() == underlying.upper()
            and r.get("IV") not in (None, "")]
    if len(hist) < IV_MIN_READINGS:
        return None
    lo, hi = min(hist), max(hist)
    if hi - lo < 1e-9:
        return None
    return round(100.0 * (float(iv) - lo) / (hi - lo))


# ────────────────────────────────────────────────────────── selection
def find(chain: dict, right: str, expiry: str, strike: float):
    """The exact contract, or None. Strike match is tolerant to a cent
    because a sheet round-trips 72.5 as 72.50000000001 often enough."""
    r = str(right).upper()[:1]
    for c in chain.get("contracts", []):
        if (c["right"] == r and c["expiry"] == expiry
                and abs(c["strike"] - float(strike)) < 0.01):
            return c
    return None


def _liquid(c) -> bool:
    """Tradeable, not merely listed.

    A strike with no open interest fills at the market maker's mercy, and the
    premium shown for it is fiction. This is the single biggest difference
    between a screen that looks clever and one you can act on.
    """
    if c["mark"] is None or c["mark"] <= 0.02:
        return False
    if (c["oi"] or 0) < 10:
        return False
    if c["bid"] is None or c["ask"] is None or c["bid"] <= 0:
        return False
    # A spread wider than a third of the mid means you lose the edge getting in.
    mid = (c["bid"] + c["ask"]) / 2.0
    return mid > 0 and (c["ask"] - c["bid"]) / mid <= 0.35


def expiries(chain: dict, right: str, *, dte_lo: int, dte_hi: int) -> list[str]:
    r = str(right).upper()[:1]
    return sorted({c["expiry"] for c in chain.get("contracts", [])
                   if c["right"] == r and dte_lo <= c["dte"] <= dte_hi})


def pick(chain: dict, right: str, *, target_delta: float = 0.20,
         dte_lo: int = 7, dte_hi: int = 21, floor_strike=None,
         min_premium: float = 0.0):
    """The best contract to SELL, on one rule: closest to `target_delta`.

    DELTA IS THE KNOB, NOT STRIKE. Delta already contains the volatility, the
    time left and the distance to the money, which is why "0.20 delta" means
    the same risk on MSTR and on SPY while "$5 out of the money" does not.
    ~0.20 is Chakravarti's stated 0.10-0.15 band loosened a little, because
    0.10 on a quiet name often pays nothing worth the assignment risk.

    `floor_strike` refuses to sell a covered call below it — the cost basis,
    so assignment can never lock in a loss on shares meant to be kept. This
    is a HARD filter, not a penalty: no amount of premium justifies being
    forced to sell at a loss something you said you wanted to hold.
    """
    r = str(right).upper()[:1]
    cands = []
    for c in chain.get("contracts", []):
        if c["right"] != r or not (dte_lo <= c["dte"] <= dte_hi):
            continue
        if c["delta"] is None or not _liquid(c):
            continue
        if c["mark"] < min_premium:
            continue
        d = abs(c["delta"])
        if d >= 0.60 or d <= 0.03:      # not a premium sale; not worth selling
            continue
        # OUT OF THE MONEY ONLY. Opening a short option already in the money
        # is a decision to be assigned, dressed up as income. Chakravarti's
        # objective is stated as "retain shares and collect income", so an
        # ITM strike is never the answer to "what should I sell".
        px = chain.get("price")
        if px:
            if r == "C" and c["strike"] <= px:
                continue
            if r == "P" and c["strike"] >= px:
                continue
        if floor_strike is not None:
            if r == "C" and c["strike"] < float(floor_strike):
                continue
            if r == "P" and c["strike"] > float(floor_strike):
                continue
        cands.append(c)
    if not cands:
        return None
    # Tie-break on open interest: same delta, take the one that will fill.
    return min(cands, key=lambda c: (abs(abs(c["delta"]) - target_delta),
                                     -(c["oi"] or 0)))


def roll(chain: dict, cur: dict, *, right: str, strike: float, expiry: str,
         target_delta: float = 0.20, min_days_out: int = 5,
         max_days_out: int = 60, floor_strike=None):
    """Where to roll a threatened short, and what it pays.

    Returns (contract, net_credit_per_share) or (None, None). Net credit is
    `new mark - cost to buy back`, so it is NEGATIVE when the roll costs you
    money — and a negative roll is reported, not hidden, because paying a
    debit to escape assignment is sometimes right and should be a decision
    rather than a surprise.

    ROLLS GO OUT AND UP, NEVER DOWN. A new strike below the current one locks
    in less upside than you already have, which is the one thing a roll must
    not do.
    """
    buyback = cur.get("mark") if cur else None
    r = str(right).upper()[:1]
    floor = floor_strike
    if r == "C":
        floor = max(float(strike), float(floor_strike or 0))
    best = None
    for c in chain.get("contracts", []):
        if c["right"] != r or c["expiry"] <= expiry:
            continue
        # Far enough out to be worth the two commissions. Rolling a 2-day
        # call into a 4-day call buys two days and pays for neither.
        if c["dte"] < min_days_out or c["dte"] > max_days_out:
            continue
        if c["delta"] is None or not _liquid(c):
            continue
        if r == "C" and c["strike"] < floor:
            continue
        if r == "P" and floor_strike is not None and c["strike"] > float(floor_strike):
            continue
        if abs(c["delta"]) >= 0.50:
            continue
        if best is None or (abs(abs(c["delta"]) - target_delta)
                            < abs(abs(best["delta"]) - target_delta)):
            best = c
    if best is None:
        return None, None
    credit = None
    if buyback is not None and best["mark"] is not None:
        credit = round(best["mark"] - buyback, 4)
    return best, credit
