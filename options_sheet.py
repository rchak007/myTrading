#!/usr/bin/env python3
"""
options_sheet.py — the OPTIONS tab.
===================================

Chakravarti, 2026-11-17: *"if you can create columns for Strike, price i paid
just to know P&L from there for future decision, Date of expiry. then you add
some columns to advise me. Also put a column first for TICKER, then type of
Option - i can put Covered Call or Secured Put.  then if i still did not sell
any calls - i will keep it blank and you can tell me which option date i
should pick and strike that is best in my interest."*

TWO SOURCES, ONE TAB.

  SCHWAB   positions the broker reports. Never typed, never editable — the
           facts are refetched every cycle, so a hand-edit would be silently
           reverted and that is worse than being read-only.
  MANUAL   SoFi, and anything else no API reaches here. You type columns
           A-G and the tab fills the rest in around them.

THE TAB IS BOTH INPUT AND OUTPUT, which is the one genuinely dangerous thing
about it: the write regenerates every row, so a cycle that read nothing and
wrote anyway would erase hand-typed positions. Two guards. The typed rows are
mirrored to an append-only ledger on every successful read, and `write_tab`
refuses to write at all if the read failed — a stale tab is recoverable, an
erased one is retyping.

WHAT THE ADVICE IS. Rules firing on numbers, not a forecast. Every verdict in
`assess()` can be read back to the row that produced it, and none of them
needs an opinion about where the stock is going. The one judgement embedded
here is Chakravarti's own, stated twice in his reminders file: these are
shares he wants to KEEP, so assignment is the thing being managed, not an
acceptable outcome to be optimised into.
"""
from __future__ import annotations

import csv
import os
from datetime import date, datetime
from pathlib import Path

import option_chain as oc

STATE_DIR = Path(os.environ.get(
    "MYTRADING_STATE_DIR", Path.home() / ".local" / "state" / "myTrading"))
MANUAL_PATH = Path(os.getenv("OPTIONS_MANUAL_PATH",
                             STATE_DIR / "options_manual.csv"))
MANUAL_COLS = ["Timestamp", "Batch", "Ticker", "Kind", "Strike", "Premium",
               "Expiry", "Qty", "Acct", "Note"]

OPTIONS_TAB = os.getenv("GSHEET_OPTIONS_TAB", "OPTIONS")

# Columns A-G are yours. Everything from H rightward is regenerated.
TYPED = ["Ticker", "Type", "Strike", "Premium", "Expiry", "Qty", "Acct"]
COMPUTED = ["Src", "DTE", "Under", "Mark", "P/L $", "Kept %",
            "Delta", "Gamma", "Theta/day", "IV", "IV_Rank", "To_Strike %",
            "Verdict", "Action", "Roll_To", "Roll_Credit", "Covered", "Note"]
OPT_HDR = TYPED + COMPUTED
N_TYPED = len(TYPED)

CALL_WORDS = ("COVERED CALL", "CC", "CALL", "C")
PUT_WORDS = ("SECURED PUT", "CSP", "CASH SECURED PUT", "PUT", "P")

# Chakravarti's own stated programme, 2026-11-17: delta 0.10-0.15,
# expiration 7-14 days, buy back at 80-95% profit, roll when the strike is
# threatened, objective "retain shares and collect income". These are HIS
# numbers, not defaults chosen here — the midpoint of his delta band and the
# exact window he named. Changing them is a decision for him, so they are
# env-overridable and otherwise left alone.
TARGET_DELTA = float(os.getenv("OPTIONS_TARGET_DELTA", "0.15"))
DTE_LO = int(os.getenv("OPTIONS_DTE_LO", "7"))
DTE_HI = int(os.getenv("OPTIONS_DTE_HI", "14"))
LADDER_MAX_DTE = int(os.getenv("OPTIONS_LADDER_DTE", "60"))

CLOSE_AT = 85.0       # % of premium captured — take it off, free the shares
GAMMA_DTE = 7         # inside this, delta moves fast enough to matter
MULT = 100.0


# ───────────────────────────────────────────────────────────── helpers
def _f(v, default=None):
    if v is None:
        return default
    s = str(v).strip().replace("$", "").replace(",", "").replace("%", "")
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def _i(v, default=None):
    f = _f(v, None)
    return default if f is None else int(round(f))


def right_of(kind: str) -> str:
    """'Covered Call' -> 'C'. Forgiving, because this is typed by hand."""
    k = str(kind or "").strip().upper().replace("-", " ")
    k = " ".join(k.split())
    if k in CALL_WORDS or k.endswith("CALL"):
        return "C"
    if k in PUT_WORDS or k.endswith("PUT"):
        return "P"
    return ""


def kind_of(right: str) -> str:
    return {"C": "Covered Call", "P": "Secured Put"}.get(
        str(right).upper()[:1], "")


def norm_date(v) -> str:
    """Whatever the sheet handed back, as YYYY-MM-DD, or ''.

    Sheets will return a typed 12/19 as a serial number, as 12/19/2026, or as
    the string you typed, depending on the cell's format. All three have to
    land on the same day or a position silently stops matching its contract.
    """
    s = str(v or "").strip()
    if not s:
        return ""
    # A bare Sheets serial (days since 1899-12-30).
    if s.replace(".", "").isdigit() and 10000 < float(s) < 80000:
        from datetime import timedelta
        return (date(1899, 12, 30) + timedelta(days=int(float(s)))).isoformat()
    s = s.split(" ")[0].split("T")[0]
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d-%b-%Y", "%b %d %Y",
                "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    # 12/19 with no year — the NEXT one, same rule the `due` verb uses.
    parts = s.replace("-", "/").split("/")
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        m, d = int(parts[0]), int(parts[1])
        today = date.today()
        try:
            cand = date(today.year, m, d)
        except ValueError:
            return ""
        if cand < today:
            cand = date(today.year + 1, m, d)
        return cand.isoformat()
    return ""


def dte_of(expiry: str, today=None):
    if not expiry:
        return None
    try:
        y, m, d = (int(x) for x in expiry.split("-"))
        return (date(y, m, d) - (today or date.today())).days
    except Exception:                                   # noqa: BLE001
        return None


# ──────────────────────────────────────────────── the manual-row ledger
def read_manual_log(path: Path = MANUAL_PATH) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            return [r for r in csv.DictReader(fh) if r.get("Ticker")]
    except Exception:                                   # noqa: BLE001
        return []


def load_manual(path: Path = MANUAL_PATH) -> list[dict]:
    """The most recent batch — what the tab held the last time it was read.

    The file is append-only so the history of what you typed survives, but
    only the newest batch is current. Reading it is the recovery path for a
    cycle where the tab could not be read.
    """
    rows = read_manual_log(path)
    if not rows:
        return []
    last = rows[-1].get("Batch")
    return [r for r in rows if r.get("Batch") == last]


def save_manual(entries: list[dict], *, path: Path = MANUAL_PATH,
                log=print) -> bool:
    """Append a batch, but ONLY if it differs from the last one.

    Without the comparison this file would grow by every row on every cycle,
    three times a day forever, and the history it exists to preserve would be
    unreadable.
    """
    def key(rows):
        return sorted((str(r.get("Ticker", "")).upper(),
                       str(r.get("Kind", "")), str(r.get("Strike", "")),
                       str(r.get("Premium", "")), str(r.get("Expiry", "")),
                       str(r.get("Qty", "")), str(r.get("Acct", "")))
                      for r in rows)

    if key(entries) == key(load_manual(path)):
        return False
    # MICROSECONDS, not seconds. Two saves inside the same second shared
    # a batch id, and load_manual() then returned both of them as current
    # — the superseded row reappearing next to the one that replaced it.
    stamp = datetime.now().isoformat(timespec="microseconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANUAL_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        for e in entries:
            w.writerow({**{c: "" for c in MANUAL_COLS}, **e,
                        "Timestamp": stamp, "Batch": stamp})
    log(f"OPTIONS: {len(entries)} manual row(s) mirrored to {path.name}")
    return True


# ──────────────────────────────────────────────────── reading the tab
def parse_typed(values: list[list]) -> list[dict]:
    """Hand-typed rows out of the raw tab values. Pure.

    WHICH ROWS SURVIVE A REGENERATION:
      SCHWAB  dropped — refetched from the broker, so keeping it would let a
              stale copy outlive the position it describes.
      MANUAL  kept, always.
      AUTO    kept only if a strike or premium has been typed into it. These
              are suggestions the tab proposed; once you act on one and fill
              it in, it stops being a suggestion and becomes a position.
      blank   kept — a row you have just typed, before the tab has ever
              labelled it.
    """
    out = []
    try:
        hdr_i = next(i for i, r in enumerate(values)
                     if r and str(r[0]).strip() == "Ticker")
    except StopIteration:
        hdr_i = -1
    for row in values[hdr_i + 1:]:
        if not row:
            continue
        cells = [str(c).strip() for c in row] + [""] * len(OPT_HDR)
        tkr = cells[0].upper()
        if not tkr or tkr.startswith("—") or tkr == "TICKER":
            continue
        src = cells[N_TYPED].upper()
        if src == "SCHWAB":
            continue
        strike, premium = cells[2], cells[3]
        if src == "AUTO" and not (strike or premium):
            continue
        out.append({"Ticker": tkr, "Kind": cells[1], "Strike": strike,
                    "Premium": premium, "Expiry": cells[4], "Qty": cells[5],
                    "Acct": cells[6] or "MANUAL"})
    return out


def read_tab(ws, log=print):
    """(entries, ok). `ok` False means DO NOT WRITE — see the module docstring."""
    try:
        values = ws.get_all_values()
    except Exception as e:                              # noqa: BLE001
        log(f"⚠️  OPTIONS: could not read the tab ({e}) — not writing, so "
            f"nothing typed there can be lost")
        return [], False
    return parse_typed(values), True


# ──────────────────────────────────────────────────── building positions
def from_schwab(options: list, log=print) -> list[dict]:
    """Short option positions as tab entries.

    LONG OPTIONS ARE SKIPPED. This tab is about premium sold — a long call is
    a different trade with different management, and showing it here under
    headings like "Kept %" would be actively misleading.
    """
    out = []
    for o in options or []:
        qty = _f(o.get("Qty"), 0.0) or 0.0
        if qty >= 0:
            continue
        out.append({
            "Ticker": str(o.get("underlying") or "").upper(),
            "Kind": kind_of(o.get("right")),
            "Strike": o.get("strike"),
            # averagePrice on a short option is the credit per share.
            "Premium": o.get("Avg_Cost"),
            "Expiry": o.get("expiry"),
            "Qty": abs(qty),
            "Acct": o.get("Acct") or "",
            "Src": "SCHWAB",
            "Symbol": o.get("symbol"),
        })
    return out


def _key(e):
    return (str(e.get("Ticker", "")).upper(), right_of(e.get("Kind")),
            round(_f(e.get("Strike"), 0) or 0, 2), norm_date(e.get("Expiry")))


def merge(schwab: list[dict], manual: list[dict], log=print) -> list[dict]:
    """Broker rows first; hand-typed rows that the broker also reports are
    dropped as duplicates.

    Dedupe ignores the account deliberately. Typing a position the broker
    already reports is a mistake whichever account you file it under, and
    double-counting a short call means the covered-ness check passes when it
    should not."""
    out = list(schwab)
    have = {_key(e) for e in schwab if _key(e)[1]}
    for m in manual:
        k = _key(m)
        if k[1] and k[2] and k[3] and k in have:
            log(f"OPTIONS: {m['Ticker']} {m.get('Expiry')} "
                f"{m.get('Strike')} is reported by Schwab — the typed copy "
                f"is ignored so it is not counted twice")
            continue
        out.append({**m, "Src": m.get("Src") or "MANUAL"})
    return out


# ────────────────────────────────────────────────────────── the advice
def assess(p: dict) -> tuple[str, str]:
    """(verdict, action) for one position. Pure, and the whole point.

    Ordered, first match wins. The order IS the priority: being in the money
    outranks having captured most of the premium, because the shares are what
    you are protecting and the premium is only what you were paid to risk them.
    """
    dte, delta = p.get("dte"), p.get("delta")
    kept, px, strike = p.get("kept"), p.get("under"), p.get("strike")
    right, mark, qty = p.get("right"), p.get("mark"), p.get("qty") or 1
    d = abs(delta) if delta is not None else None
    cost = f"${mark * MULT * qty:,.0f}" if mark is not None else "?"

    if dte is not None and dte < 0:
        return "⚪ EXPIRED", "Gone. Clear the row, or re-type it if you rolled."

    itm = (px is not None and strike is not None
           and ((right == "C" and px > strike) or (right == "P" and px < strike)))

    if itm:
        if right == "C":
            return ("🔴 ITM", f"Shares are being called at ${strike:,.2f}. "
                              f"Roll up and out, or accept assignment. "
                              f"Buy-back costs ~{cost}.")
        return ("🔴 ITM", f"You are on the hook to buy at ${strike:,.2f}. "
                          f"Roll down and out, or take the shares.")

    if d is not None and dte is not None and dte <= GAMMA_DTE and d >= 0.45:
        return ("🔴 ROLL", f"Δ{d:.2f} with {dte}d left — gamma week. "
                           f"Roll before Thursday, not on expiry day.")

    if kept is not None and kept >= CLOSE_AT:
        return ("🟢 CLOSE", f"{kept:.0f}% of the premium is captured; "
                            f"~{cost} left on the table. Buy it back and "
                            f"free the collateral.")

    # 0.25, not 0.30, inside three days. A quarter-delta with a weekend in
    # the way is a real chance of assignment and the last window in which
    # a roll is still cheap — exactly the CRWV 60C case.
    if d is not None and dte is not None and dte <= 3 and d >= 0.25:
        return ("🟠 WATCH", f"Δ{d:.2f} at {dte}d. Decide on the close before "
                            f"expiry, not during it.")

    if d is not None and d >= 0.40:
        return ("🟠 WATCH", f"Δ{d:.2f} — drifting toward the strike. "
                            f"Roll if it closes through ${strike:,.2f}.")

    # Only away from expiry. "No action" is sound advice at 25 days and
    # negligent at 2, where there is no time left for theta to deliver.
    if kept is not None and kept < 0 and (dte is None or dte > 3):
        return ("🟡 UNDERWATER", f"Down {-kept:.0f}% on the premium but Δ"
                                 f"{d:.2f} is still safe. Theta is on your "
                                 f"side; no action.")
    return ("🟢 DECAY", "On track. Let theta work.")


def covered_text(p: dict) -> str:
    """Whether the obligation is actually backed."""
    right, qty = p.get("right"), p.get("qty") or 1
    if right == "C":
        shares, need = p.get("shares"), qty * 100
        # NAKED can only be claimed where the broker sees everything. Shares
        # held at SoFi are invisible to this pipeline, so a hand-typed row
        # states the obligation and leaves the check to you — calling a
        # properly covered SoFi call "naked" would train you to ignore the
        # flag on the Schwab rows, where it is real.
        if p.get("src") != "SCHWAB":
            return (f"needs {need:,.0f} sh"
                    + (f" · {shares:,.0f} at Schwab" if shares else ""))
        if shares is None:
            return "?"
        return (f"✓ {shares:,.0f} sh" if shares >= need
                else f"🔴 NAKED — {shares:,.0f} sh vs {need:,.0f} needed")
    strike = p.get("strike")
    if not strike:
        return ""
    return f"${strike * 100 * qty:,.0f} cash"


# ─────────────────────────────────────────────────────────── enrichment
def enrich(entry: dict, chain, iv_rows=None, shares=None,
           today=None) -> dict:
    """One tab entry plus everything the chain knows about it."""
    right = right_of(entry.get("Kind"))
    strike = _f(entry.get("Strike"))
    expiry = norm_date(entry.get("Expiry"))
    prem = _f(entry.get("Premium"))
    qty = _i(entry.get("Qty"), 1) or 1
    p = {"ticker": str(entry.get("Ticker", "")).upper(), "right": right,
         "strike": strike, "expiry": expiry, "premium": prem, "qty": qty,
         "acct": entry.get("Acct") or "", "src": entry.get("Src") or "MANUAL",
         "kind": entry.get("Kind") or kind_of(right),
         "shares": shares,
         "dte": dte_of(expiry, today), "under": None, "mark": None,
         "delta": None, "gamma": None, "theta": None, "iv": None,
         "ivr": None, "kept": None, "pl": None, "to_strike": None,
         "roll_to": "", "roll_credit": None, "note": ""}

    if chain:
        p["under"] = chain.get("price")
        c = oc.find(chain, right, expiry, strike) if (right and expiry and strike) else None
        if c:
            p["mark"] = c["mark"]
            p["delta"] = c["delta"]
            p["gamma"] = c["gamma"]
            p["theta"] = c["theta"]
            p["iv"] = c["iv"]
            p["ivr"] = oc.iv_rank(p["ticker"], c["iv"], rows=iv_rows)
            if c["dte"] is not None:
                p["dte"] = c["dte"]
        elif right and expiry and strike:
            p["note"] = "contract not in the chain window"

    if p["premium"] is not None and p["mark"] is not None:
        p["pl"] = round((p["premium"] - p["mark"]) * MULT * qty, 2)
        if p["premium"] > 0:
            p["kept"] = round(100.0 * (p["premium"] - p["mark"]) / p["premium"], 1)
    if p["under"] and strike:
        gap = (strike - p["under"]) if right == "C" else (p["under"] - strike)
        p["to_strike"] = round(100.0 * gap / p["under"], 1)
    return p


def suggest(p: dict, chain, *, target_delta=TARGET_DELTA,
            dte_lo=DTE_LO, dte_hi=DTE_HI) -> tuple[str, str]:
    """(verdict, action) for a row with no contract typed into it.

    Chakravarti: *"if i still did not sell any calls - i will keep it blank
    and you can tell me which option date i should pick and strike that is
    best in my interest."* "In my interest" is the constraint that does the
    work: the strike must be ABOVE the current price. Chakravarti,
    2026-11-18: *"we just are putting covered call from current price ..
    strike should be above"*. Cost basis plays no part — it is a sunk cost,
    and anchoring a live decision to what you once paid is how a position
    stops being managed on its own merits.
    """
    right = p.get("right")
    if not right:
        return "⚪ ?", "Type 'Covered Call' or 'Secured Put' in column B."
    if not chain:
        return "⚪ NO DATA", "No chain for this underlying."

    qty = p.get("qty") or 1
    if right == "C":
        shares = p.get("shares") or 0
        if shares < 100:
            # On the TAB a blank row with no visible shares really is nothing
            # to act on, so the refusal stays — but it points at the verb,
            # which answers without needing to see the shares at all.
            return ("⚪ NO SHARES",
                    f"Schwab shows {shares:,.0f} shares. If you hold 100 "
                    f"elsewhere, run `cc {p.get('ticker', '')}` — that "
                    f"answers without needing to see them.")
        qty = min(qty, int(shares // 100))

    best = oc.pick(chain, right, target_delta=target_delta,
                   dte_lo=dte_lo, dte_hi=dte_hi)
    if best is None:
        return ("⚪ NOTHING LIQUID",
                f"No liquid strike near Δ{target_delta:.2f} in {dte_lo}-{dte_hi}d.")

    gross = (best["mark"] or 0) * MULT * qty
    ann = ""
    if best["mark"] and best["dte"] and p.get("under"):
        # Premium as a yield on the capital the position ties up, annualised.
        # The honest comparison between "60 cents on a 10-day" and "two
        # dollars on a 45-day", which are not otherwise comparable.
        base = best["strike"] if right == "P" else p["under"]
        ann = (f" · {365.0 / best['dte'] * best['mark'] / base * 100:.0f}%/yr"
               if base else "")
    msg = (f"SELL {qty} {p['ticker']} {best['expiry']} "
           f"${best['strike']:,.2f}{right} @ ~${best['mark']:.2f} "
           f"→ ${gross:,.0f} (Δ{abs(best['delta']):.2f}, {best['dte']}d"
           f", OI {best['oi']:,.0f}){ann}")
    return ("🔵 SUGGESTION", msg)


def plan_roll(p: dict, chain) -> tuple[str, str]:
    """(label, credit text) for where this position could roll to."""
    if not chain or not p.get("right") or not p.get("expiry") or not p.get("strike"):
        return "", ""
    cur = oc.find(chain, p["right"], p["expiry"], p["strike"])
    if not cur:
        return "", ""
    best, credit = oc.roll(chain, cur, right=p["right"], strike=p["strike"],
                           expiry=p["expiry"], target_delta=TARGET_DELTA)
    if best is None:
        return "", ""
    qty = p.get("qty") or 1
    label = (f"{best['expiry']} ${best['strike']:,.2f}{p['right']} "
             f"(Δ{abs(best['delta']):.2f})")
    if credit is None:
        return label, ""
    net = credit * MULT * qty
    return label, (f"+${net:,.0f} credit" if net >= 0 else f"−${-net:,.0f} DEBIT")


# ───────────────────────────────────────────────────────── rendering
VERDICT_RANK = {
    "🔴 ITM": 1, "🔴 ROLL": 2,
    "🟢 CLOSE": 3,                      # actionable: money sitting there
    "🟠 WATCH": 4,
    "🟡 UNDERWATER": 5, "🟢 DECAY": 6,
    "⚪ EXPIRED": 7,
}


def _scrub(v):
    """json-safe cell. See orders_sheet._scrub — one numpy scalar anywhere in
    the payload makes the WHOLE gspread write raise, so this is load-bearing."""
    if v is None:
        return ""
    item = getattr(v, "item", None)
    if callable(item) and getattr(v, "ndim", None) == 0:
        try:
            return item()
        except Exception:                               # noqa: BLE001
            return str(v)
    if isinstance(v, float) and v != v:                 # NaN
        return ""
    return v


def _pct(v, nd=1):
    return "" if v is None else f"{v:,.{nd}f}%"


def _money(v):
    return "" if v is None else f"${v:,.2f}"


def build_rows(entries, chains, *, shares=None, iv_rows=None,
               auto=(), log=print, today=None, stale=None):
    """(rows, marks) for the whole tab. Pure — no sheet, no network.

    `entries` are tab entries (typed or from Schwab), `chains` is
    {TICKER: chain or None}, `auto` is extra tickers to propose a contract
    for. Everything that decides anything lives in assess()/suggest(); this
    only lays it out.
    """
    shares = shares or {}
    open_pos, wanted = [], []
    for e in entries:
        t = str(e.get("Ticker", "")).upper()
        p = enrich(e, chains.get(t), iv_rows=iv_rows,
                   shares=shares.get(t), today=today)
        if p["strike"] and p["expiry"]:
            p["verdict"], p["action"] = assess(p)
            p["roll_to"], p["roll_credit"] = plan_roll(p, chains.get(t))
        else:
            p["verdict"], p["action"] = suggest(p, chains.get(t))
        (open_pos if p["strike"] and p["expiry"] else wanted).append(p)

    # Tickers that write calls but have none open right now.
    covered = {p["ticker"] for p in open_pos if p["right"] == "C"}
    asked = {p["ticker"] for p in wanted}
    for t in sorted(set(auto) - covered - asked):
        if (shares.get(t) or 0) < 100:
            continue
        p = enrich({"Ticker": t, "Kind": "Covered Call", "Qty": 1,
                    "Acct": "", "Src": "AUTO"}, chains.get(t),
                   iv_rows=iv_rows, shares=shares.get(t), today=today)
        p["src"] = "AUTO"
        p["verdict"], p["action"] = suggest(p, chains.get(t))
        wanted.append(p)

    open_pos.sort(key=lambda p: (VERDICT_RANK.get(p["verdict"], 8),
                                 p["dte"] if p["dte"] is not None else 999,
                                 p["ticker"]))
    wanted.sort(key=lambda p: (p["src"] != "MANUAL", p["ticker"]))

    marks = {"title": [], "header": [], "label": [], "red": [], "amber": [],
             "green": [], "suggest": [], "typed": []}
    rows = []

    def add(vals):
        rows.append([_scrub(v) for v in vals]
                    + [""] * (len(OPT_HDR) - len(vals)))
        return len(rows)

    n_red = sum(1 for p in open_pos if p["verdict"].startswith("🔴"))
    n_amb = sum(1 for p in open_pos if p["verdict"].startswith("🟠"))
    pl = sum(p["pl"] for p in open_pos if p["pl"] is not None)
    credit = sum((p["premium"] or 0) * MULT * (p["qty"] or 1) for p in open_pos)

    marks["title"].append(add(["OPTIONS", f"updated {_now()}"]))
    add(["", f"{len(open_pos)} short position(s) · ${credit:,.0f} premium "
             f"collected · ${pl:,.0f} open P/L · {n_red} needing action · "
             f"{n_amb} to watch"
             + (f" · ⚠️ chain data {stale}" if stale else "")])
    add(["", "Columns A-G are YOURS — type SoFi and any other off-Schwab "
             "position there. H onward is regenerated every cycle. Schwab "
             "rows are read-only: edit them at the broker, not here."])
    add([])
    marks["header"].append(add(OPT_HDR))

    def emit(p):
        theta_day = (None if p["theta"] is None
                     else -p["theta"] * MULT * (p["qty"] or 1))
        r = add([
            p["ticker"], p["kind"],
            p["strike"] if p["strike"] else "",
            p["premium"] if p["premium"] is not None else "",
            p["expiry"], p["qty"], p["acct"],
            p["src"], p["dte"] if p["dte"] is not None else "",
            _money(p["under"]), _money(p["mark"]),
            "" if p["pl"] is None else round(p["pl"], 2),
            _pct(p["kept"], 0),
            "" if p["delta"] is None else round(abs(p["delta"]), 3),
            "" if p["gamma"] is None else round(p["gamma"], 4),
            "" if theta_day is None else f"${theta_day:,.2f}",
            _pct(p["iv"], 0), "" if p["ivr"] is None else f"{p['ivr']}%",
            _pct(p["to_strike"], 1),
            p["verdict"], p["action"], p["roll_to"], p["roll_credit"],
            covered_text(p), p["note"],
        ])
        v = p["verdict"]
        if v.startswith("🔴"):
            marks["red"].append(r)
        elif v.startswith("🟠") or v.startswith("🟡"):
            marks["amber"].append(r)
        elif v.startswith("🔵"):
            marks["suggest"].append(r)
        elif v.startswith("🟢"):
            marks["green"].append(r)
        marks["typed"].append(r)

    if open_pos:
        marks["label"].append(add(["", "— OPEN POSITIONS — most urgent first"]))
        for p in open_pos:
            emit(p)
    if wanted:
        marks["label"].append(add(
            ["", "— NO CONTRACT WRITTEN — what to sell, if you want to"]))
        for p in wanted:
            emit(p)
    if not open_pos and not wanted:
        add(["", "Nothing to show. Type a ticker in column A and "
                 "'Covered Call' in column B to get a suggestion."])

    add([])
    marks["label"].append(add(["", "— HOW TO READ IT —"]))
    for line in (
        "🔴 act today · 🟠 decide on the close · 🟢 on track · 🔵 a contract "
        "you could sell",
        "Delta is roughly the chance of finishing in the money. "
        "Kept % is how much of the premium you have already earned.",
        "Theta/day is what the position earns you per day if nothing moves. "
        "Roll_Credit is NET of buying the current one back.",
        "Every suggested strike is ABOVE the current price. Cost basis "
        "plays no part — it is a sunk cost, not a reason to hold or sell.",
        "IV_Rank needs ~20 daily readings before it says anything; blank "
        "means not enough history yet.",
    ):
        add(["", line])
    return rows, marks


def _now():
    try:
        import pytz
        return datetime.now(pytz.timezone("America/Los_Angeles")).strftime(
            "%Y-%m-%d %H:%M:%S %Z")
    except Exception:                                   # noqa: BLE001
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _col(n: int) -> str:
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _paint(ws, marks, last_col, log=print) -> None:
    """Cosmetic only — never let it cost the write."""
    try:
        typed_end = _col(N_TYPED)
        fmts = [
            ("title", {"textFormat": {"bold": True, "fontSize": 12}}),
            ("header", {"textFormat": {"bold": True},
                        "backgroundColor": {"red": .85, "green": .87, "blue": .92}}),
            ("label", {"textFormat": {"bold": True, "italic": True},
                       "backgroundColor": {"red": .93, "green": .93, "blue": .93}}),
            ("red", {"backgroundColor": {"red": 1.0, "green": .80, "blue": .80}}),
            ("amber", {"backgroundColor": {"red": 1.0, "green": .93, "blue": .76}}),
            ("green", {"backgroundColor": {"red": .85, "green": .95, "blue": .85}}),
            ("suggest", {"backgroundColor": {"red": .85, "green": .91, "blue": 1.0}}),
        ]
        for name, fmt in fmts:
            spans = [f"A{r}:{last_col}{r}" for r in marks.get(name, [])]
            if spans:
                ws.format(spans, fmt)
        # The typed block gets a tint of its own, so which columns are yours
        # is visible without reading the note at the top.
        yours = [f"A{r}:{typed_end}{r}" for r in marks.get("typed", [])]
        if yours:
            ws.format(yours, {"textFormat": {"bold": True}})
    except Exception as e:                              # noqa: BLE001
        log(f"⚠️  OPTIONS: could not paint ({e})")


# ─────────────────────────────────────────────────────────── the write
def write_tab(book, positions, options, client_wrapper, log=print) -> int:
    """Refresh the OPTIONS tab. Returns the number of positions shown."""
    try:
        ws = book.worksheet(OPTIONS_TAB)
    except Exception:                                   # noqa: BLE001
        log(f"OPTIONS: no '{OPTIONS_TAB}' tab — skipping")
        return 0

    typed, ok = read_tab(ws, log=log)
    if not ok:
        return 0                     # never write over rows we could not read
    if typed:
        save_manual(typed, log=log)
    else:
        # An empty read is ambiguous — a genuinely empty tab, or a tab that
        # came back blank. The ledger decides, and it says so out loud.
        back = load_manual()
        if back:
            log(f"OPTIONS: tab has no typed rows but the ledger holds "
                f"{len(back)} — restoring them. Delete them from the ledger "
                f"if they are really gone.")
            typed = back

    entries = merge(from_schwab(options, log=log), typed, log=log)

    # Share counts, summed across accounts. A covered call in one account is
    # still covered by shares in another only if you can move them, so this
    # is deliberately the AGGREGATE and `Covered` says which.
    shares = {}
    try:
        import pandas as pd
        live = positions[positions["Acct"] != "TOTAL"]
        for t, grp in live.groupby(live["Ticker"].astype(str).str.upper()):
            shares[t] = float(
                pd.to_numeric(grp["Qty"], errors="coerce").fillna(0).sum())
    except Exception as e:                              # noqa: BLE001
        log(f"⚠️  OPTIONS: share counts unavailable ({e}) — the covered "
            f"check will read '?'")

    # Which underlyings to pull a chain for: everything with a position or a
    # typed row, plus anything this book has written against before. Bounded
    # by history rather than by the whole watchlist, so it cannot quietly
    # grow into ninety calls a cycle.
    universe = {str(e.get("Ticker", "")).upper() for e in entries}
    universe |= {str(r.get("Ticker", "")).upper() for r in read_manual_log()}
    universe |= {t.strip().upper()
                 for t in os.getenv("OPTIONS_WATCH", "").split(",") if t.strip()}
    universe.discard("")

    chains, stale = {}, []
    for t in sorted(universe):
        ch = oc.fetch_chain(client_wrapper, t, log=log)
        chains[t] = ch
        if ch:
            age = ch.get("age_min") or 0
            if age > 90:
                stale.append(f"{t} {age:.0f}m")
            try:
                oc.record_iv(t, oc.atm_iv(ch))
            except Exception:                           # noqa: BLE001
                pass
    iv_rows = oc.read_iv()

    rows, marks = build_rows(entries, chains, shares=shares,
                             iv_rows=iv_rows, auto=universe, log=log,
                             stale=", ".join(stale) if stale else None)

    need = len(rows) + 30
    if int(getattr(ws, "row_count", 0) or 0) < need:
        ws.resize(rows=need)
    last = _col(len(OPT_HDR))
    if int(getattr(ws, "col_count", 0) or 0) < len(OPT_HDR):
        ws.resize(cols=len(OPT_HDR))
    ws.batch_clear([f"A1:{last}{ws.row_count}"])
    ws.update(values=rows, range_name=f"A1:{last}{len(rows)}",
              value_input_option="USER_ENTERED")
    _paint(ws, marks, last, log)
    n = sum(1 for e in entries if _f(e.get("Strike")) and e.get("Expiry"))
    log(f"OPTIONS: {n} position(s), {len(universe)} chain(s), "
        f"{len(typed)} typed row(s)")
    return n


# ──────────────────────────────────────────── "what should I sell today"
def _m(v) -> str:
    """Dollars, with the minus OUTSIDE the sign. `$-1,300` reads as a price;
    `-$1,300` reads as a loss, which is what it is."""
    if v is None:
        return ""
    return f"${v:,.0f}" if v >= 0 else f"\u2212${-v:,.0f}"


def annualised(premium, base, dte):
    """Premium as a yield on the capital tied up, per year.

    THE ONLY WAY TO COMPARE TWO EXPIRIES. 60 cents on a 10-day and $2.00 on a
    45-day are not otherwise comparable, and picking the bigger dollar number
    is how a premium seller ends up locking their shares up for six weeks to
    earn less per day than a weekly would have paid.
    """
    if not premium or not base or not dte or dte <= 0:
        return None
    return round(365.0 / dte * premium / base * 100.0, 1)


def per_expiry(chain, right, *, target_delta=TARGET_DELTA, floor_strike=None,
               max_dte=LADDER_MAX_DTE, underlying_px=None):
    """The best strike at each expiry — one row per date, for a ladder.

    `oc.pick` answers "what should I sell"; this answers "what does the whole
    board look like", which is the question actually being asked when someone
    says "which date should I pick".
    """
    out = []
    for exp in sorted({c["expiry"] for c in chain.get("contracts", [])
                       if c["right"] == right and 0 < c["dte"] <= max_dte}):
        # pick() searches a DTE window; narrow it to this one expiry so each
        # ladder rung is the best strike for that date, not the best strike
        # overall repeated down the column.
        sub = {"contracts": [c for c in chain["contracts"]
                             if c["expiry"] == exp], "price": chain.get("price")}
        best = oc.pick(sub, right, target_delta=target_delta,
                       dte_lo=0, dte_hi=max_dte, floor_strike=floor_strike)
        if best is None:
            continue
        base = best["strike"] if right == "P" else (underlying_px
                                                    or chain.get("price"))
        out.append({**best, "ann": annualised(best["mark"], base, best["dte"])})
    return out


def _earnings_before(ticker, iso_expiry):
    """Next earnings date if it lands on or before this expiry, else "".

    A call written THROUGH earnings is a different trade: one print can gap
    the stock past the strike overnight, and no amount of theta compensates
    for that. Wrapped because it is a yfinance call and must never be the
    reason an answer fails to come back.
    """
    try:
        import oracle_value
        _last, nxt = oracle_value.earnings_dates(ticker)
        nxt = str(nxt or "")[:10]
        return nxt if nxt and nxt <= iso_expiry else ""
    except Exception:                                   # noqa: BLE001
        return ""


def advise(ticker, chain, *, right="C", shares=None, qty=None,
           open_positions=(), target_delta=TARGET_DELTA, dte_lo=DTE_LO,
           dte_hi=DTE_HI, iv_rows=None, earnings=True) -> list[str]:
    """Plain-text answer to "what covered call should I write on MSTR today".

    Returns lines, because this goes into an ops-sheet Status cell and a log,
    not a spreadsheet. A ladder first so the choice is visible, then ONE
    recommendation, then what happens if it is assigned — which is the part
    that decides whether the premium was worth it and the part a bare
    "sell the 0.15 delta" never tells you.
    """
    t = str(ticker).upper()
    L = []
    if not chain:
        return [f"{t}: no option chain available."]
    px = chain.get("price")
    iv = oc.atm_iv(chain)
    ivr = oc.iv_rank(t, iv, rows=iv_rows)
    want_call = right == "C"

    head = f"{t}  ${px:,.2f}" if px else t
    if iv:
        head += f"  ·  IV {iv:.0f}%" + (f" (rank {ivr}%)" if ivr is not None
                                        else " (rank: not enough history)")
    if shares is not None:
        # "0 sh" next to "SELL 2" on the same screen is how a reader is left
        # to reconcile two numbers that disagree. Name the source.
        head += f"  ·  {shares:,.0f} sh at Schwab"
    L += [head, ""]

    unseen = False
    if want_call:
        n = int((shares or 0) // 100)
        if n < 1:
            # ASK AND YOU GET AN ANSWER. Corrected 2026-11-17 — Chakravarti:
            # *"Charles Schwab I don't have GOOG or CBRS .. but when I ask
            # recommendations I need it"*. Schwab cannot see a SoFi holding,
            # and "you have no shares" is this pipeline reporting its own
            # blind spot as though it were a fact about his portfolio.
            #
            # The premium, the greeks and the right strike do not depend on
            # where the shares are custodied. Only the COVER does, and that
            # is his to confirm — so the answer is given and the assumption
            # is stated out loud.
            unseen, n = True, (qty or 1)
        qty = min(qty or n, n)
        if unseen:
            L.append(f"⚠  Schwab shows {shares or 0:,.0f} shares of {t}, so "
                     f"it cannot check your cover. Sizing assumes "
                     f"{qty} contract{'s' if qty != 1 else ''} = "
                     f"{qty * 100:,.0f} shares"
                     + (" — because you asked for that size."
                        if (qty or 1) > 1 else ".")
                     + " The strike and premium are right either way; the "
                       "share count is yours to confirm.")
            L.append("")
        for p in open_positions:
            L.append(f"NOTE: you are already short {p}. Anything below is "
                     f"ADDITIONAL, and you only have {n} contract(s) of "
                     f"cover in total.")
        if open_positions:
            L.append("")

    # COST BASIS PLAYS NO PART. Removed 2026-11-18 — Chakravarti: *"Don't
    # need to look at basis .. that's not needed .. we just are putting
    # covered call from current price .. strike should be above"*.
    #
    # He is right, and it went through two wrong versions first: a hard
    # floor at basis, then a warning at basis. Both anchored a live decision
    # to a sunk cost. The only rule a covered call needs is that the strike
    # is above the CURRENT price — that is what decides whether you keep the
    # shares, and `oc.pick` enforces it. What you paid is history.
    rows = per_expiry(chain, right, target_delta=target_delta,
                      underlying_px=px)
    if not rows:
        return L + [f"Nothing liquid near Δ{target_delta:.2f} in the next "
                    f"{LADDER_MAX_DTE} days."]

    kind = "COVERED CALL" if want_call else "CASH-SECURED PUT"
    L.append(f"{kind} LADDER — closest strike to Δ{target_delta:.2f} at each "
             f"expiry, all above the ${px:,.2f} spot")
    L.append("")
    L.append(f"   {'EXPIRY':<12}{'DTE':>4}{'STRIKE':>11}{'DELTA':>8}"
             f"{'MID':>9}{'PER CT':>10}{'%/YR':>8}{'OI':>9}")
    for c in rows:
        L.append(f"   {c['expiry']:<12}{c['dte']:>4}"
                 f"{'$' + format(c['strike'], ',.2f'):>11}"
                 f"{abs(c['delta']):>8.2f}"
                 f"{'$' + format(c['mark'], ',.2f'):>9}"
                 f"{'$' + format(c['mark'] * MULT, ',.0f'):>10}"
                 f"{(format(c['ann'], '.0f') + '%') if c['ann'] else '—':>8}"
                 f"{c['oi']:>9,.0f}"
)
    L.append("")

    # THE PICK. His stated window first; the ladder's best annualised yield
    # inside it. Only if that window is empty do we look outside it, and then
    # we say so rather than quietly changing his strategy for him.
    inwin = [c for c in rows if dte_lo <= c["dte"] <= dte_hi]
    note_window = ""
    if not inwin:
        inwin = [c for c in rows if c["dte"] <= dte_hi + 21]
        note_window = (f"   (nothing liquid in your usual {dte_lo}-{dte_hi} "
                       f"day window, so this looks further out)")
    if not inwin:
        inwin = rows
    best = max(inwin, key=lambda c: (c["ann"] or 0))

    gross = best["mark"] * MULT * (qty or 1)
    # THE COVER GOES IN THE ORDER LINE. Added 2026-11-18 after a demo
    # printed "0 sh" and "SELL 2 GOOG" four lines apart and expected the
    # reader to reconcile them. Chakravarti: *"I don't have 200 shares of
    # Google lol. Only 100 in SoFi"*. A contract count is abstract; "needs
    # 200 shares" is the thing you can check against your own account in
    # one second, so it belongs where the order is read, not in a note.
    need = (f"  — needs {(qty or 1) * 100:,.0f} shares"
            if want_call else
            f"  — needs {_m(best['strike'] * 100 * (qty or 1))} cash")
    L.append(f"→ TODAY: SELL {qty or 1} {t} {best['expiry']} "
             f"${best['strike']:,.2f}{right} — LIMIT ${best['mark']:.2f} "
             f"(${gross:,.0f}){need}")
    if note_window:
        L.append(note_window)
    bits = [f"Δ{abs(best['delta']):.2f}", f"{best['dte']} days"]
    if best["ann"]:
        bits.append(f"{best['ann']:.0f}%/yr")
    if px:
        bits.append(f"{(best['strike'] - px) / px * 100:+.1f}% from spot")
    L.append("   " + "  ·  ".join(bits))
    L.append(f"   Bid ${best['bid']:.2f} / ask ${best['ask']:.2f} — work the "
             f"mid, do not hit the bid.")

    # What assignment means, measured from TODAY'S PRICE. That is the real
    # comparison: you still hold the shares, so the question is what you give
    # up from here by capping them — not what you paid for them years ago.
    if want_call:
        up = (best["strike"] - px) * 100 * (qty or 1)
        L.append("")
        L.append(f"   If assigned: you sell {100 * (qty or 1):,.0f} shares at "
                 f"${best['strike']:,.2f} — {_m(up)} above today's "
                 f"${px:,.2f} ({(best['strike'] / px - 1) * 100:+.1f}%), "
                 f"plus {_m(gross)} premium. Total {_m(up + gross)}.")
    else:
        cash = best["strike"] * 100 * (qty or 1)
        L.append("")
        L.append(f"   If assigned: you buy {100 * (qty or 1):,.0f} shares at "
                 f"${best['strike']:,.2f} — ${cash:,.0f} of cash must be "
                 f"free and that has to be a price you want to own at.")

    warn = []
    if earnings:
        e = _earnings_before(t, best["expiry"])
        if e:
            warn.append(f"⚠️  earnings {e} falls before this expiry — one "
                        f"print can gap straight through the strike.")
    if ivr is not None and ivr < 25:
        warn.append(f"⚠️  IV rank {ivr}% — premium is cheap by this ticker's "
                    f"own standards. Selling here pays you little for the "
                    f"upside you give up.")
    if best["mark"] * MULT < 30:
        warn.append(f"⚠️  ${best['mark'] * MULT:.0f} a contract barely clears "
                    f"two commissions. Not worth the assignment risk.")
    if warn:
        L.append("")
        L += ["   " + w for w in warn]
    return L
