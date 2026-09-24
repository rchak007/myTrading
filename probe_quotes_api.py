#!/usr/bin/env python3
"""
probe_quotes_api.py
===================
Find out how current schwabdev exposes quotes, and what it returns.

    .venv/bin/python probe_quotes_api.py

Measure, do not guess. Guessing this API has cost us twice already:
`tokens_file` had become `tokens_db`, and `account_linked()` had become
`linked_accounts()` — both discovered only by a run failing in production.

Prints the method that works and the shape of the response, so the price
updater can be written against what is actually there.

Read-only. Runs on Pi 1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYMBOLS = ["AAPL", "MU", "MSTR"]


def main() -> int:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))

    import jobStocksSignals as job
    client = job.get_schwab_client()

    # data.schwab.schwab_helper.SchwabClient wraps schwabdev.Client and stores
    # it as _client (underscore). The first version of this probe looked for
    # `.client`, found the wrapper itself, and reported "NONE FOUND" — the
    # wrapper has no quote methods because it only forwards what it was asked
    # to forward. Unwrap properly, and say which object is being probed.
    # get_client() first: SchwabClient builds schwabdev lazily, so ._client is
    # None until something asks for it. Reading the attribute finds None,
    # falls through, and leaves you probing the wrapper — which exposes only
    # fetch_positions, because that is all it was written to forward.
    inner = client
    getter = getattr(client, "get_client", None)
    if callable(getter):
        try:
            cand = getter()
            if cand is not None and cand is not client:
                inner = cand
                print("unwrapped via .get_client()")
        except Exception as e:
            print(f"get_client() raised: {type(e).__name__}: {e}")

    if inner is client:
        for attr in ("_client", "client", "schwab", "_schwab"):
            cand = getattr(client, attr, None)
            if cand is not None and cand is not client:
                inner = cand
                print(f"unwrapped via .{attr}")
                break

    print(f"\nwrapper type: {type(client)}")
    print(f"probing type: {type(inner)}")

    names = sorted(n for n in dir(inner)
                   if "quote" in n.lower() or "price" in n.lower())
    if not names:
        print("\nno quote/price attribute names. Full public surface:")
        allpub = sorted(n for n in dir(inner) if not n.startswith("_"))
        for i in range(0, len(allpub), 6):
            print("   " + "  ".join(f"{n:<22}" for n in allpub[i:i + 6]))
        print("\nIf a quote method is in that list under another name, say which.")
        return 1
    print(f"quote/price-ish attributes: {names}\n")

    # Try the plausible shapes, in order. One comma-joined string is how
    # `types` had to be passed for transactions, so it is the first guess.
    attempts = [
        ("quotes(comma string)", lambda m: m(",".join(SYMBOLS))),
        ("quotes(list)",         lambda m: m(SYMBOLS)),
        ("quotes(symbols=str)",  lambda m: m(symbols=",".join(SYMBOLS))),
        ("quotes(symbols=list)", lambda m: m(symbols=SYMBOLS)),
    ]

    for name in names:
        meth = getattr(inner, name, None)
        if not callable(meth):
            continue
        for label, call in attempts:
            try:
                resp = call(meth)
            except Exception as e:
                print(f"  {name}: {label:<22} -> {type(e).__name__}: {str(e)[:90]}")
                continue

            body = resp
            if hasattr(resp, "json"):
                print(f"  {name}: {label:<22} -> HTTP {getattr(resp, 'status_code', '?')}")
                try:
                    body = resp.json()
                except Exception as e:
                    print(f"      body not JSON: {e}")
                    continue
            else:
                print(f"  {name}: {label:<22} -> {type(resp).__name__}")

            if isinstance(body, dict) and body:
                print(f"\n  ✅ WORKS: {name} with {label}")
                print(f"  top-level keys: {list(body)[:8]}")
                k = next(iter(body))
                print(f"\n  sample entry for {k!r}:")
                print("  " + json.dumps(body[k], indent=2)[:900].replace("\n", "\n  "))
                print("\n  Candidate price fields to look for: lastPrice, mark, "
                      "regularMarketLastPrice, closePrice, quoteTime.")
                return 0
            print(f"      unexpected body: {type(body).__name__} {str(body)[:120]}")

    print("\n❌ Nothing worked. Paste the attribute list above and we go from there.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
