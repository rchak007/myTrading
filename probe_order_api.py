#!/usr/bin/env python3
"""
probe_order_api.py
==================
Find out how this schwabdev places, lists and cancels orders.

    .venv/bin/python probe_order_api.py

PLACES NOTHING. It only inspects method names and signatures, and optionally
READS existing orders. Nothing here can spend money.

WHY
    Three guesses about this library have been wrong already — `tokens_file`
    had become `tokens_db`, `account_linked` had become `linked_accounts`, and
    `types` turned out to need a comma-joined string. Each cost a round trip to
    Pi 1 and a production failure. Order placement is the one call where
    guessing wrong is expensive rather than merely annoying, so it gets
    measured like the others did.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def main() -> int:
    import jobStocksSignals as job
    client = job.get_schwab_client()
    inner = client
    getter = getattr(client, "get_client", None)
    if callable(getter):
        inner = getter() or client

    print(f"\nprobing: {type(inner)}\n")

    names = sorted(n for n in dir(inner)
                   if not n.startswith("_")
                   and any(k in n.lower() for k in ("order", "place", "cancel")))
    if not names:
        print("no order-ish methods. Full public surface:")
        pub = sorted(n for n in dir(inner) if not n.startswith("_"))
        for i in range(0, len(pub), 5):
            print("   " + "  ".join(f"{n:<26}" for n in pub[i:i + 5]))
        return 1

    print("order-related methods, with their signatures:\n")
    for n in names:
        fn = getattr(inner, n, None)
        try:
            sig = str(inspect.signature(fn))
        except (TypeError, ValueError):
            sig = "(signature unavailable)"
        doc = (inspect.getdoc(fn) or "").strip().splitlines()
        print(f"  {n}{sig}")
        for line in doc[:3]:
            print(f"      {line}")
        print()

    # A read-only call, to confirm the account-hash shape the place call wants.
    try:
        accts = inner.linked_accounts().json()
        print("linked accounts (hash is what order_place takes):")
        for a in accts or []:
            if isinstance(a, dict):
                print(f"   ...{str(a.get('accountNumber'))[-3:]}  "
                      f"hash={str(a.get('hashValue'))[:12]}...")
    except Exception as e:
        print(f"linked_accounts failed: {type(e).__name__}: {e}")

    print("\nNothing was placed. Paste the method list back.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
