#!/usr/bin/env python3
"""
order_reconcile.py — close the loop between a submitted order and what the
broker actually did with it.

WHY THIS EXISTS
---------------
The engine could place an order and could never find out what became of it.
`order_engine.py` writes TRIGGERED, SUBMITTED, BLOCKED, VOID and EXPIRED to
the ledger — and nothing, anywhere in the repo, ever wrote FILLED. Nothing
wrote `Filled_Qty` either, though `order_engine` READS it and its own comment
says it is *"an engine column that reconcile writes back"*. That reconcile
did not exist.

Two things were broken by the gap, both silently:

  CHAINING NEVER ARMED. A row with After_Trigger_ID waits for its parent to
  reach FILLED. No code path produced FILLED, so every chained row waited
  for ever, showing "⏸ WAITING — it is SUBMITTED, arms when it FILLS" and
  looking healthy while being permanently inert. 2026-10-02-AAOI-02 has sat
  like that since October.

  THE SHEET LIED BY OMISSION. A filled order kept reading SUBMITTED, so the
  tab could not distinguish "resting at Schwab" from "done days ago".

WHAT IT DOES NOT DO
-------------------
It places nothing, cancels nothing and modifies no order. It reads order
status and writes down what it finds. The only state it can create is a
record of something that already happened — which is why it is safe to run
on every cycle, and why it takes no --dry-run flag worth the name.

It is also deliberately CONSERVATIVE about what counts as finished. Anything
Schwab reports that is not a clearly terminal status leaves the row exactly
as it is. A row stuck on SUBMITTED is visible and fixable; a row wrongly
marked FILLED arms a chain that spends money.

THE LEDGER HEADER IS NOT EXTENDED
---------------------------------
Fill quantity and price go into the ledger's `note` and into the SHEET's
Filled_Qty / Fill_Price columns, not into new ledger columns. `append_ledger`
writes a header only when the file does not exist, so adding a column to
LEDGER_COLS would make every new line wider than the header of the existing
file and silently shift every field after it. The sheet is where the engine
already looks for Filled_Qty, so that is where it goes.
"""
from __future__ import annotations

import argparse

import order_engine as eng

# Schwab status -> our ledger state. ONLY terminal, unambiguous outcomes.
#
# Everything absent from this map leaves the row untouched: WORKING, QUEUED,
# ACCEPTED, PENDING_ACTIVATION, AWAITING_PARENT_ORDER and friends all mean
# "still live", and a GTC order rests in one of them for days by design.
TERMINAL = {
    "FILLED": "FILLED",
    "CANCELED": "CANCELLED",          # Schwab spells it with one L
    "CANCELLED": "CANCELLED",
    "REJECTED": "REJECTED",
    "EXPIRED": "EXPIRED",
    # REPLACED means the order was superseded by another one. It will never
    # fill under THIS id, so leaving it live would keep a chain waiting for
    # ever — the exact failure this module exists to end.
    "REPLACED": "CANCELLED",
}

# A partial fill on a CANCELLED order still bought shares. Recorded, because
# the quantity that matters to a chain is what executed, not what was asked.
PARTIAL_NOTE = "partially filled before it ended"


def _f(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _iter_orders(orders):
    """Every order including OCO / bracket children, which carry their own
    ids and statuses and are where a real fill often lives."""
    for o in orders or []:
        if not isinstance(o, dict):
            continue
        yield o
        yield from _iter_orders(o.get("childOrderStrategies") or [])


def fill_details(o: dict) -> tuple[float, float | None]:
    """(filled quantity, average fill price) for one Schwab order dict.

    The average is WEIGHTED by leg quantity. A plain mean over executions
    would misreport any order that filled in uneven pieces, which is most of
    them on a thin name — and the price is the number that decides whether
    the trade was any good.
    """
    qty = _f(o.get("filledQuantity"))
    paid = 0.0
    shares = 0.0
    for act in o.get("orderActivityCollection") or []:
        if not isinstance(act, dict):
            continue
        for leg in act.get("executionLegs") or []:
            if not isinstance(leg, dict):
                continue
            q, px = _f(leg.get("quantity")), _f(leg.get("price"), None)
            if q and px:
                paid += q * px
                shares += q
    if shares and not qty:
        # filledQuantity absent but executions present: trust the executions.
        qty = shares
    return qty, (paid / shares if shares else None)


def pending(ledger: list[dict]) -> dict[str, dict]:
    """{schwab_order_id: ledger row} for every row still believed live.

    Keyed on the ORDER ID rather than the Row_ID because that is what the
    broker's answer is keyed on, and a row with no id never reached Schwab —
    `orphaned_attempts` already reports those and they are not this module's
    problem.
    """
    out = {}
    for rid, r in eng.latest_states(ledger).items():
        if r.get("state") != "SUBMITTED":
            continue
        oid = str(r.get("schwab_order_id") or "").strip()
        if oid:
            out[oid] = r
    return out


def reconcile(client_wrapper, ws=None, cols=None, log=print,
              days_back: int = 120) -> list[dict]:
    """Find out what happened to every SUBMITTED order. Returns the updates.

    Each update is {row_id, state, note, filled_qty, fill_price}. The ledger
    is written here; the sheet is written by `write_back` so a caller that
    already holds the tab open can do it in one batch with its own updates.
    """
    import stocks_orders

    ledger = eng.read_ledger()
    live = pending(ledger)
    if not live:
        log("reconcile: nothing is awaiting an outcome")
        return []
    log(f"reconcile: {len(live)} submitted order(s) to check")

    raw = stocks_orders.fetch_schwab_orders(client_wrapper,
                                            days_back=days_back, log=log)
    by_id = {}
    for o in _iter_orders(raw):
        oid = o.get("orderId")
        if oid is not None:
            by_id[str(oid)] = o

    updates = []
    for oid, prev in live.items():
        rid = prev.get("row_id", "")
        o = by_id.get(oid)
        if o is None:
            # NOT an error, and NOT a reason to change the row. The lookback
            # window is finite and Schwab's history is not ours to assume
            # about. Said out loud so a genuinely vanished order is visible.
            log(f"   {rid:<26} order {oid} not in the last {days_back}d of "
                f"Schwab history — left as SUBMITTED")
            continue
        status = str(o.get("status") or "").strip().upper()
        state = TERMINAL.get(status)
        if not state:
            log(f"   {rid:<26} still {status or 'UNKNOWN'}")
            continue

        qty, px = fill_details(o)
        bits = [f"Schwab says {status}"]
        if qty:
            bits.append(f"{qty:g} share(s)")
        if px:
            bits.append(f"@ {px:,.4f}")
        if state != "FILLED" and qty:
            bits.append(PARTIAL_NOTE)
        note = " · ".join(bits)

        # The ledger first, always. It is the record; the sheet is a view of
        # it, and a crash between the two must leave the truth written.
        eng.append_ledger(dict(
            row_id=rid, fingerprint=prev.get("fingerprint", ""),
            state=state, note=note, acct=prev.get("acct", ""),
            ticker=prev.get("ticker", ""), side=prev.get("side", ""),
            close_is=prev.get("close_is", ""), qty=prev.get("qty", ""),
            trigger_price=prev.get("trigger_price", ""),
            limit_price=prev.get("limit_price", ""),
            schwab_order_id=oid))
        eng.audit(event="reconciled", row_id=rid, order_id=oid,
                  state=state, filled_qty=qty, fill_price=px)
        log(f"   {rid:<26} {state:<10} {note}")
        updates.append({"row_id": rid, "state": state, "note": note,
                        "filled_qty": qty, "fill_price": px})

    if ws is not None and updates:
        write_back(ws, cols, updates, log=log)
    return updates


def write_back(ws, cols, updates, log=print) -> int:
    """Mirror the outcomes into the Orders tab. Best effort, never raises.

    Row_ID -> row number is re-read here rather than cached, for the reason
    the engine learned the hard way: new work is added at the TOP of the tab,
    so a row position from minutes ago now points at someone else's order.
    """
    if not updates:
        return 0
    try:
        if cols is None:
            import orders_columns as oco
            cols = oco.read(ws, log=log)
        live = {}
        for i, raw in enumerate(ws.col_values(1), start=1):
            if i >= eng.DATA_START_ROW and str(raw).strip():
                live[str(raw).strip()] = i

        stamp = eng._now().strftime("%Y-%m-%d %H:%M:%S %Z")
        payload = []
        for u in updates:
            r = live.get(u["row_id"])
            if r is None:
                log(f"⚠️  {u['row_id']} is no longer on the tab; the ledger "
                    f"still has its outcome")
                continue

            def cell(name, val):
                # A column the tab does not carry is skipped, not fatal.
                # Filled_Qty and Fill_Price arrived after some sheets were
                # created, and an older tab must still get its Status.
                if name in cols:
                    payload.append({"range": f"{cols.col(name)}{r}",
                                    "values": [[val]]})

            cell("Status", u["state"])
            cell("Status_Date", stamp)
            cell("Engine_Note", u["note"][:400])
            if u.get("filled_qty"):
                cell("Filled_Qty", f"{u['filled_qty']:g}")
            if u.get("fill_price"):
                cell("Fill_Price", f"{u['fill_price']:.4f}")
            cell("Last_Checked", stamp)
        if payload:
            ws.batch_update(payload, value_input_option="RAW")
        return len(updates)
    except Exception as e:                                  # noqa: BLE001
        log(f"⚠️  reconcile sheet write-back failed (ledger is correct): {e}")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Record what Schwab did with every submitted order")
    ap.add_argument("--no-sheet", action="store_true",
                    help="write the ledger only, leave the Orders tab alone")
    ap.add_argument("--days-back", type=int, default=120,
                    help="how far back to search Schwab order history")
    args = ap.parse_args()

    import jobStocksSignals as job
    client = job.get_schwab_client()
    ws = cols = None
    if not args.no_sheet:
        import orders_columns as oco
        ws = eng.open_orders_tab()
        cols = oco.read(ws, log=print)
    out = reconcile(client, ws=ws, cols=cols, days_back=args.days_back)
    print(f"\n{len(out)} order(s) reached a final state")
    for u in out:
        print(f"   {u['row_id']:<26} {u['state']:<10} {u['note']}")
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
