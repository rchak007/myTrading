#!/usr/bin/env python3
"""
probe_types_param.py
====================
Does omitting `types` return transactions our filtered fetch never sees?

    ../.venv/bin/python probe_types_param.py

Read-only. Fetches one account over one narrow window, twice.

WHY
    Schwab's UI shows an AEHR transfer pair the API never gives us:

        2023-09-05  Journaled Shares   TDA TRAN - TRANSFER OF SECURITY OR
                                       OPTION OUT (AEHR)            -109
        2023-09-05  Internal Transfer  AEHR TEST SYSTEMS            +109

    A full refetch from 2019 returned byte-identical data, so this is not a
    dropped chunk and not a watermark gap — Schwab is not returning the rows
    for the request we make.

    We request an explicit `types` list because passing all of them once 400'd.
    txn_cache's own docstring records five names the current API rejects, among
    them SECURITY_TRANSFER — which is precisely what a "TRANSFER OF SECURITY OR
    OPTION OUT" would be filed under. Since we cannot name it, naming anything
    at all excludes it.

    `types` is optional. This compares the same window with and without it.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import txn_cache                                       # noqa: E402
from schwab_client import make_client                  # noqa: E402

# The window and account holding AEHR's missing -109 pair (A9D5532F = ...171).
WINDOW = (date(2023, 9, 1), date(2023, 9, 30))
SYMBOL = "AEHR"


def _fetch(client, acct_hash, start, end, types_param):
    resp = client.transactions(
        acct_hash,
        start.strftime("%Y-%m-%dT00:00:00.000Z"),
        end.strftime("%Y-%m-%dT23:59:59.999Z"),
        types_param,
    )
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}: {getattr(resp, 'text', '')[:200]}"
    body = resp.json() or []
    if isinstance(body, dict):
        for k in ("transactions", "items"):
            if isinstance(body.get(k), list):
                body = body[k]
                break
    return (body if isinstance(body, list) else []), None


def main() -> int:
    client = make_client()
    accounts = txn_cache.get_linked_accounts(client)

    start, end = WINDOW
    print(f"window: {start} .. {end}\n")

    for acc in accounts:
        h = acc["hash"]
        with_types, err1 = _fetch(client, h, start, end, txn_cache.TYPES_PARAM)
        # The whole point: no types at all.
        without, err2 = _fetch(client, h, start, end, None)

        n_with = "ERR" if with_types is None else len(with_types)
        n_without = "ERR" if without is None else len(without)
        print(f"=== {h[:8]}  with types={n_with}  without types={n_without}")
        if err1:
            print(f"    with types -> {err1}")
        if err2:
            print(f"    WITHOUT types -> {err2}")

        if not isinstance(with_types, list) or not isinstance(without, list):
            print()
            continue

        seen = {t.get("activityId") for t in with_types}
        extra = [t for t in without if t.get("activityId") not in seen]
        if extra:
            print(f"    *** {len(extra)} transaction(s) ONLY visible without `types` ***")
            for t in extra[:10]:
                print(f"      type={t.get('type')!r}  {str(t.get('description'))[:64]}")
                for it in (t.get("transferItems") or []):
                    inst = it.get("instrument") or {}
                    if inst.get("symbol"):
                        print(f"          {inst.get('symbol')} "
                              f"assetType={inst.get('assetType')} "
                              f"amount={it.get('amount')}")
        else:
            print("    no difference")

        hits = [t for t in without if SYMBOL in json.dumps(t).upper()]
        if hits:
            print(f"    {SYMBOL} rows in the unfiltered result: {len(hits)}")
            for t in hits:
                print(f"      type={t.get('type')!r}  {str(t.get('description'))[:64]}")
        print()

    print("If the unfiltered call returns rows the filtered one misses, the fix is")
    print("to stop sending `types` at all and filter locally instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
