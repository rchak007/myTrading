#!/usr/bin/env python3
"""
chitra.py
=========
Chitra's account — the sister's portfolio Chakravarti manages. ONE TAB that is
the whole management surface, because there is no API to trade it through.

    ┌─ POSITIONS ─────────── machine, from chitra_holdings.csv
    │   with the same Has_Stop / Has_Trim / Has_Dip / Has_Breakout flags his
    │   own Dashboard carries, computed by the same function
    ├─ BROKERAGE ORDERS ──── machine, from chitra_orders.csv
    │   what is actually resting at Merrill, transcribed from his screenshots
    └─ MY CONDITIONS ─────── HUMAN. Pi 1 never writes columns A-F here.
        he types "SELL / BELOW / 214"; Pi 1 fills in the live price, the last
        close, and paints the row GREEN the moment the condition is met.

NOTHING IS PLACED. There is no Merrill API and none is wanted. A green row
means go and do it by hand.

─────────────────────────────────────────────────────────────────────────────
THE LAYOUT IS FIXED ON PURPOSE
    His typed conditions sit at known rows and the machine sections are capped,
    so a position appearing or disappearing can never shift his rows out from
    under him. The alternative — writing the tab top to bottom — would move
    the human block every time she bought something.

    This mirrors the Orders tab's contract exactly: "YOU FILL A-F", "PI 1 FILLS
    G-J — DO NOT TYPE HERE". It is the one arrangement in this system that has
    survived contact with a human editing the same sheet a program writes.

NEVER ws.clear()
    That is what the first version did, and it would erase every condition he
    had typed. Each machine section clears only its OWN range.

WHERE THE TRUTH IS
    chitra_holdings.csv  positions      ← from her statement
    chitra_orders.csv    resting orders ← from his Merrill screenshots
    chitra_reserves.csv  fencing + cash earmarks against the IIAXX sweep
    the SHEET            conditions     ← the only thing he types, and the only
                                          thing the repo does not own
"""
from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILE = Path(os.getenv("CHITRA_FILE", HERE / "chitra_holdings.csv"))
ORDERS_FILE = Path(os.getenv("CHITRA_ORDERS_FILE", HERE / "chitra_orders.csv"))
RESERVES_FILE = Path(os.getenv("CHITRA_RESERVES_FILE", HERE / "chitra_reserves.csv"))

ACCT_LABEL = "CHITRA"
TAB = "Chitra"

# ── the fixed layout ────────────────────────────────────────────────────────
# Row numbers are 1-based, as the sheet counts them. Caps are generous; a
# section that overflows is logged rather than silently truncated.
POS_LABEL_ROW = 3
POS_HDR_ROW = 4
POS_START = 5
POS_MAX = 30                       # rows 5..34

ORD_LABEL_ROW = 37
ORD_HDR_ROW = 38
ORD_START = 39
ORD_MAX = 15                       # rows 39..53

CON_LABEL_ROW = 56
CON_HDR_ROW = 57
CON_START = 58
CON_MAX = 60                       # rows 58..117

POS_COLS = ["Ticker", "Qty", "Avg_Cost", "Live_Price", "Day_%", "Market_Value",
            "Cost_Basis", "Unrealized_PL", "Unrealized_%",
            "Has_Stop", "Has_Trim", "Has_Dip", "Has_Breakout",
            "Fenced", "Seed_Reserved", "Note"]
ORD_COLS = ["Ticker", "Side", "Type", "Qty", "Limit_Price", "Stop_Price",
            "Duration", "Entered", "Order_ID", "Note"]
# A-F are his. G-J are Pi 1's.
CON_HUMAN = ["Ticker", "Side", "Close_Is", "Trigger_Price", "Qty", "Note"]
CON_MACHINE = ["Live_Price", "Last_Close", "Status", "Checked"]
CON_COLS = CON_HUMAN + CON_MACHINE
CON_HUMAN_N = len(CON_HUMAN)

WIDTH = max(len(POS_COLS), len(ORD_COLS), len(CON_COLS))
MET = "🟢 CONDITION MET"
WAIT = "waiting"


def _col(n: int) -> str:
    """1-based column number to letter. Single letters are enough at 16 wide."""
    return chr(ord("A") + n - 1)


LAST_COL = _col(WIDTH)


def _num(v, default=None):
    try:
        if v is None or str(v).strip() == "":
            return default
        return float(str(v).replace(",", "").replace("$", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return default


# ── the two repo files ──────────────────────────────────────────────────────
def _rows(path: Path, log) -> list[dict]:
    """CSV rows with the comment header skipped. [] on anything unexpected."""
    if not path.exists():
        log(f"Chitra: no {path.name}")
        return []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            body = [l for l in fh if not l.lstrip().startswith("#")]
        return [dict(r) for r in csv.DictReader(body)]
    except Exception as e:
        log(f"⚠️  Chitra: could not read {path.name}: {e}")
        return []


def meta(path: Path | None = None) -> dict:
    """`Key  Value` pairs from a file's comment header — Account, As_Of."""
    path = path or FILE
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("#"):
            break
        parts = line.lstrip("# ").split(None, 1)
        if len(parts) == 2 and parts[0] in ("Account", "As_Of"):
            out[parts[0]] = parts[1].strip()
    return out


def load(log=print) -> list[dict]:
    """Her positions. Never raises — her account is an addition, not a
    prerequisite, so a bad file costs her rows and nothing else."""
    out = []
    for r in _rows(FILE, log):
        t = str(r.get("Ticker", "")).strip().upper()
        qty = _num(r.get("Qty"))
        if not t or qty is None:
            continue
        out.append({"Ticker": t, "Qty": qty,
                    "Cost_Basis": _num(r.get("Cost_Basis"), 0.0) or 0.0,
                    "Asset": str(r.get("Asset", "EQUITY")).strip().upper(),
                    # Fencing lives in chitra_reserves.csv, not here. Two
                    # places to declare it would be one place to forget.
                    "Note": str(r.get("Note", "")).strip()})
    return out


def load_reserves(log=print) -> dict:
    """{TICKER: {"seed": float, "fenced": bool, "note": str}}.

    A NOTE, NOT A LEDGER. On his Schwab accounts cash_reserve.py sees fills
    and the engine refuses an over-spend; neither exists here, so this records
    intent and makes over-commitment visible. It enforces nothing, and it is
    only as current as the last screenshot.
    """
    out = {}
    for r in _rows(RESERVES_FILE, log):
        t = str(r.get("Ticker", "")).strip().upper()
        if not t:
            continue
        out[t] = {"seed": _num(r.get("Seed"), 0.0) or 0.0,
                  "fenced": str(r.get("Fenced", "")).strip().upper().startswith("Y"),
                  "note": str(r.get("Note", "")).strip()}
    return out


def cash_position(rows, reserves, orders=None) -> tuple:
    """(cash, seeded, in_open_buys, free) for the sweep.

    An open BUY limit commits cash the moment it rests, exactly as it does on
    his Schwab accounts where the Cash tab carries Cash_In_Open_Orders. Her
    $360 TSLA bid is $360 she cannot spend twice, and showing the sweep as
    fully free would be the same overstatement the covered-call collateral
    was.

    `free` goes negative when over-committed, which is the number worth
    seeing — nothing here can refuse a trade, so visibility is the only
    protection.
    """
    cash = sum(r["Cost_Basis"] for r in rows if r.get("Asset") == "CASH")
    seeded = sum(v["seed"] for v in reserves.values())
    open_buys = 0.0
    for o in (orders or []):
        if str(o.get("Side", "")).upper() != "BUY":
            continue
        px = _num(o.get("Limit_Price")) or _num(o.get("Stop_Price"))
        qty = _num(o.get("Qty"))
        if px and qty:
            open_buys += px * qty
    return cash, seeded, open_buys, cash - seeded - open_buys


def load_orders(log=print) -> list[dict]:
    """Orders resting at Merrill, per the last screenshot transcribed."""
    out = []
    for r in _rows(ORDERS_FILE, log):
        t = str(r.get("Ticker", "")).strip().upper()
        if not t:
            continue
        out.append({k: str(r.get(k, "")).strip() for k in ORD_COLS} | {"Ticker": t})
    return out


# ── pricing ─────────────────────────────────────────────────────────────────
def _price(row, quotes, extract) -> tuple:
    """(price, day_pct). CASH is held at $1.00 and never quoted — asking for a
    quote on a sweep fund returns nothing, which would read as a priceless row
    rather than as cash."""
    if row.get("Asset") == "CASH":
        return 1.0, 0.0
    if not quotes:
        return None, None
    entry = quotes.get(row["Ticker"])
    return extract(entry) if entry else (None, None)


def _closes(signals_df) -> dict:
    """{TICKER: last completed daily close}.

    Conditions are evaluated on the CLOSE, matching the Orders tab's
    semantics: "closes below 214" is not "touched 214", and a wick must not
    turn a row green.
    """
    if signals_df is None or getattr(signals_df, "empty", True):
        return {}
    out = {}
    for _, r in signals_df.iterrows():
        c = _num(r.get("Last Close")) or _num(r.get("Current Price"))
        if c:
            out[str(r.get("Ticker", "")).upper()] = c
    return out


# ── conditions: read what he typed, decide if it has fired ──────────────────
def read_conditions(ws, log=print) -> list[dict]:
    """His typed rows, with their sheet row number. Blank rows are skipped.

    A row needs a Ticker, a Side and a Trigger_Price to mean anything; one
    missing is a half-typed row, not an instruction.
    """
    try:
        grid = ws.get(f"A{CON_START}:{_col(CON_HUMAN_N)}{CON_START + CON_MAX - 1}")
    except Exception as e:
        log(f"⚠️  Chitra: could not read conditions: {e}")
        return []
    out = []
    for i, raw in enumerate(grid or []):
        cells = list(raw) + [""] * CON_HUMAN_N
        rec = {c: str(cells[j]).strip() for j, c in enumerate(CON_HUMAN)}
        if not rec["Ticker"] or not rec["Side"] or not _num(rec["Trigger_Price"]):
            continue
        rec["row"] = CON_START + i
        rec["Ticker"] = rec["Ticker"].upper()
        rec["Side"] = rec["Side"].upper()
        rec["Close_Is"] = rec["Close_Is"].upper()
        out.append(rec)
    return out


def evaluate(cond: dict, close: float | None) -> str:
    """MET, waiting, or why it cannot be judged.

    Evaluated on the CLOSE, never the live price. "Closes below 214" is a
    different instruction from "touches 214", and conflating them is how a
    wick turns into a trade.
    """
    trig = _num(cond.get("Trigger_Price"))
    d = cond.get("Close_Is", "")
    if d not in ("ABOVE", "BELOW"):
        return "need Close_Is ABOVE or BELOW"
    if cond.get("Side") not in ("BUY", "SELL"):
        return "need Side BUY or SELL"
    if trig is None:
        return "need a Trigger_Price"
    if close is None:
        return "no close for this ticker"
    hit = close < trig if d == "BELOW" else close > trig
    return MET if hit else f"{WAIT} — close {close:,.2f}, needs {d.lower()} {trig:,.2f}"


# ── coverage, using the SAME rule his Dashboard uses ────────────────────────
def coverage(ticker, price, orders, conditions, held_qty):
    """Has_Stop / Has_Trim / Has_Dip / Has_Breakout for one of her positions.

    Delegates to orders_sheet.coverage_for so there is ONE definition of what
    counts as protection. A second implementation here would drift, and the
    first thing to drift would be the thing the colour depends on.
    """
    try:
        import pandas as pd
        from orders_sheet import coverage_for
    except Exception:
        return {k: "" for k in ("Has_Stop", "Has_Trim", "Has_Dip", "Has_Breakout")}

    odf = pd.DataFrame([{"Ticker": o["Ticker"], "Side": o["Side"],
                         "Stop_Price": _num(o.get("Stop_Price")),
                         "Limit_Price": _num(o.get("Limit_Price"))}
                        for o in orders if o["Ticker"] == ticker]) \
        if orders else None
    # ONLY conditions that could actually fire. A half-typed row — a missing
    # direction, a bad Side — cannot be judged, and counting it as protection
    # would be the Dashboard's recurring failure in miniature: saying covered
    # about the one row that can never act.
    usable = [c for c in (conditions or [])
              if c["Ticker"] == ticker
              and c.get("Close_Is") in ("ABOVE", "BELOW")
              and c.get("Side") in ("BUY", "SELL")
              and _num(c.get("Trigger_Price")) is not None]
    cdf = pd.DataFrame([{"Ticker": c["Ticker"], "Side": c["Side"],
                         "Close_Is": c["Close_Is"],
                         "Trigger_Price": _num(c["Trigger_Price"]),
                         "Qty": _num(c.get("Qty"))}
                        for c in usable]) if usable else None
    return coverage_for(ticker, ACCT_LABEL, price, odf, cdf, held_qty=held_qty)


# ── the three blocks ────────────────────────────────────────────────────────
def build_positions(rows, quotes, extract, orders, conditions,
                    reserves=None) -> list[list]:
    """The positions block, with a TOTAL line."""
    reserves = reserves or {}
    out, tot_val, tot_cost = [], 0.0, 0.0
    for r in rows:
        px, pct = _price(r, quotes, extract)
        val = (px * r["Qty"]) if px is not None else None
        upl = (val - r["Cost_Basis"]) if val is not None else None
        pcnt = (100.0 * upl / r["Cost_Basis"]) if (upl is not None
                                                   and r["Cost_Basis"]) else None
        if val is not None:
            tot_val += val
            tot_cost += r["Cost_Basis"]
        # Cash has nothing to protect, so the flags are blank rather than N —
        # the same reason a TOTAL row carries blanks on his Dashboard.
        cov = ({k: "" for k in ("Has_Stop", "Has_Trim", "Has_Dip", "Has_Breakout")}
               if r["Asset"] == "CASH"
               else coverage(r["Ticker"], px, orders, conditions, r["Qty"]))
        out.append([
            r["Ticker"], round(r["Qty"], 4),
            round(r["Cost_Basis"] / r["Qty"], 4) if r["Qty"] else "",
            round(px, 4) if px is not None else "",
            round(pct, 2) if pct is not None else "",
            round(val, 2) if val is not None else "",
            round(r["Cost_Basis"], 2),
            round(upl, 2) if upl is not None else "",
            round(pcnt, 2) if pcnt is not None else "",
            cov["Has_Stop"], cov["Has_Trim"], cov["Has_Dip"], cov["Has_Breakout"],
            "🔒" if reserves.get(r["Ticker"], {}).get("fenced") else "",
            round(reserves.get(r["Ticker"], {}).get("seed", 0.0), 2) or "",
            r["Note"],
        ])
    if out:
        g = tot_val - tot_cost
        row = ["TOTAL", "", "", "", "", round(tot_val, 2), round(tot_cost, 2),
               round(g, 2),
               round(100.0 * g / tot_cost, 2) if tot_cost else ""]
        out.append(row + [""] * (len(POS_COLS) - len(row)))
    return out


def build_orders(orders) -> list[list]:
    if not orders:
        return [["— none resting at Merrill as of "
                 + (meta(ORDERS_FILE).get("As_Of") or "?") + " —"]
                + [""] * (len(ORD_COLS) - 1)]
    return [[o.get(c, "") for c in ORD_COLS] for o in orders]


def build_condition_status(conditions, quotes, extract, closes) -> dict:
    """{sheet row: [live, close, status, checked]} for the machine columns."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    out = {}
    for c in conditions:
        t = c["Ticker"]
        px, _ = _price({"Ticker": t, "Asset": "EQUITY"}, quotes, extract)
        close = closes.get(t)
        out[c["row"]] = [round(px, 4) if px is not None else "",
                         round(close, 4) if close is not None else "",
                         evaluate(c, close), now]
    return out


# ── writing ─────────────────────────────────────────────────────────────────
def write_tab(book, quotes, extract, signals_df=None, log=print) -> int:
    """Rewrite the machine sections. NEVER touches columns A-F of conditions."""
    try:
        ws = book.worksheet(TAB)
    except Exception:
        ws = book.add_worksheet(title=TAB,
                                rows=CON_START + CON_MAX + 20, cols=WIDTH + 2)

    rows = load(log=log)
    orders = load_orders(log=log)
    reserves = load_reserves(log=log)
    conditions = read_conditions(ws, log=log)
    closes = _closes(signals_df)

    pos = build_positions(rows, quotes, extract, orders, conditions, reserves)
    ords = build_orders(orders)
    status = build_condition_status(conditions, quotes, extract, closes)
    for block, cap, name in ((pos, POS_MAX, "positions"),
                             (ords, ORD_MAX, "orders")):
        if len(block) > cap:
            log(f"⚠️  Chitra: {len(block)} {name} rows exceeds the {cap}-row "
                f"section — the layout needs widening, showing the first {cap}")
            del block[cap:]

    m, mo = meta(), meta(ORDERS_FILE)
    cash, seeded, open_buys, free = cash_position(rows, reserves, orders)
    # Seeded against the sweep, on the face of the tab. Over-committing is
    # visible rather than discovered — nothing here can refuse a trade, so
    # seeing it is the only protection there is.
    money = (f"sweep ${cash:,.2f} · seeded ${seeded:,.2f}"
             + (f" · ${open_buys:,.2f} in open buys" if open_buys else "")
             + " · " + (f"free ${free:,.2f}" if free >= 0
                        else f"⚠️ OVER-COMMITTED by ${-free:,.2f}"))
    head = [[f"CHITRA — {m.get('Account', 'account')}",
             f"positions as of {m.get('As_Of', '?')} · orders as of "
             f"{mo.get('As_Of', '?')} · prices live · managed by Chakravarti"],
            ["", money + "  ·  NOTHING IS PLACED FROM HERE: a green row below "
                         "means go and do it at Merrill by hand."]]

    # Each section clears only its OWN range. ws.clear() would erase every
    # condition he has typed, which is the whole reason this is not one write.
    pad = lambda r, n=WIDTH: list(r) + [""] * (n - len(r))
    try:
        ws.batch_clear([f"A1:{LAST_COL}{POS_START + POS_MAX - 1}",
                        f"A{ORD_LABEL_ROW}:{LAST_COL}{ORD_START + ORD_MAX - 1}",
                        f"A{CON_LABEL_ROW}:{LAST_COL}{CON_HDR_ROW}",
                        f"{_col(CON_HUMAN_N + 1)}{CON_START}:"
                        f"{LAST_COL}{CON_START + CON_MAX - 1}"])

        ws.update(values=[pad(h, 2) for h in head], range_name="A1:B2")
        ws.update(values=[["POSITIONS"], pad(POS_COLS)],
                  range_name=f"A{POS_LABEL_ROW}:{LAST_COL}{POS_HDR_ROW}")
        if pos:
            ws.update(values=[pad(r) for r in pos],
                      range_name=f"A{POS_START}:{LAST_COL}{POS_START + len(pos) - 1}")

        ws.update(values=[["BROKERAGE ORDERS — actually resting at Merrill "
                           "(from Chakravarti's screenshots)"], pad(ORD_COLS)],
                  range_name=f"A{ORD_LABEL_ROW}:{LAST_COL}{ORD_HDR_ROW}")
        ws.update(values=[pad(r) for r in ords],
                  range_name=f"A{ORD_START}:{LAST_COL}{ORD_START + len(ords) - 1}")

        ws.update(values=[[f"MY CONDITIONS — type in A–{_col(CON_HUMAN_N)} only. "
                           f"Pi 1 fills {_col(CON_HUMAN_N + 1)}–{_col(len(CON_COLS))} "
                           f"and will overwrite anything typed there."],
                          pad(CON_COLS)],
                  range_name=f"A{CON_LABEL_ROW}:{LAST_COL}{CON_HDR_ROW}")
        if status:
            lo, hi = min(status), max(status)
            grid = [status.get(r, [""] * len(CON_MACHINE)) for r in range(lo, hi + 1)]
            ws.update(values=grid,
                      range_name=f"{_col(CON_HUMAN_N + 1)}{lo}:"
                                 f"{_col(len(CON_COLS))}{hi}")
    except Exception as e:
        log(f"⚠️  Chitra tab write failed: {e}")
        return 0

    _paint(ws, status, len(pos), len(ords), log)
    met = sum(1 for v in status.values() if v[2] == MET)
    log(f"Chitra tab: {len(rows)} position(s), {len(orders)} resting order(s), "
        f"{len(conditions)} condition(s)" + (f", {met} MET 🟢" if met else ""))
    if free < 0:
        log(f"⚠️  Chitra: OVER-COMMITTED by ${-free:,.2f} — ${seeded:,.2f} "
            f"seeded + ${open_buys:,.2f} in open buys against a ${cash:,.2f} "
            f"sweep")
    return len(rows)


def _paint(ws, status, n_pos, n_ords, log):
    """Cosmetics. Best-effort — never let a formatting call lose the data."""
    bold = {"textFormat": {"bold": True}}
    label = {"backgroundColor": {"red": 1.0, "green": 0.95, "blue": 0.60},
             "textFormat": {"bold": True}}
    human = {"backgroundColor": {"red": 1.0, "green": 1.0, "blue": 0.90}}
    green = {"backgroundColor": {"red": 0.42, "green": 0.84, "blue": 0.49},
             "textFormat": {"bold": True,
                            "foregroundColor": {"red": 0, "green": 0.15, "blue": 0}}}
    plain = {"backgroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0},
             "textFormat": {"bold": False, "italic": False}}
    try:
        # Reset the condition rows absolutely before colouring. A row that was
        # green yesterday and is waiting today must not stay green — stale
        # formatting has outlived its data on this sheet before.
        ws.format(f"A{CON_START}:{LAST_COL}{CON_START + CON_MAX - 1}", plain)
        ws.format(f"A1:{LAST_COL}1", bold)
        ws.format([f"A{POS_LABEL_ROW}", f"A{ORD_LABEL_ROW}", f"A{CON_LABEL_ROW}"],
                  label)
        ws.format([f"A{r}:{LAST_COL}{r}"
                   for r in (POS_HDR_ROW, ORD_HDR_ROW, CON_HDR_ROW)], bold)
        # His columns tinted, so the boundary is visible rather than remembered.
        ws.format(f"A{CON_START}:{_col(CON_HUMAN_N)}{CON_START + CON_MAX - 1}",
                  human)
        hits = [r for r, v in status.items() if v[2] == MET]
        if hits:
            ws.format([f"A{r}:{LAST_COL}{r}" for r in hits], green)
            log(f"Chitra: {len(hits)} condition(s) met — painted green")
    except Exception as e:
        log(f"⚠️  Chitra formatting skipped (data is fine): {e}")


def dashboard_row(ticker: str, rows, quotes, extract, width: int) -> list | None:
    """One CHITRA line for a block on HIS Dashboard, or None.

    Coverage columns stay BLANK there: her flags live on her own tab, and
    asserting them inside his block would mix two accounts' protection in one
    place.
    """
    hit = next((r for r in rows if r["Ticker"] == ticker.upper()), None)
    if hit is None:
        return None
    px, pct = _price(hit, quotes, extract)
    val = (px * hit["Qty"]) if px is not None else None
    upl = (val - hit["Cost_Basis"]) if val is not None else None
    row = [""] * width
    row[1:9] = [hit["Ticker"], ACCT_LABEL, round(hit["Qty"], 4),
                round(hit["Cost_Basis"] / hit["Qty"], 4) if hit["Qty"] else "",
                round(px, 4) if px is not None else "",
                round(pct, 2) if pct is not None else "",
                round(val, 2) if val is not None else "",
                round(upl, 2) if upl is not None else ""]
    return row
