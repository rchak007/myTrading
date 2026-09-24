#!/usr/bin/env python3
"""
schwab_quotes.py
================
Real-time prices from Schwab, extended hours included.

One place, because two callers need the same answer and must not drift:
    jobStocksSignals.py  -> the "Current Price" column in stocks_signals.csv
    orders_sheet_prices.py -> Live_Price on the Dashboard

WHY NOT YAHOO
    data/stocks.fetch_current_price() calls Yahoo's v7 endpoint and reads
    `regularMarketPrice`, which is the REGULAR SESSION price — so after the
    close it returns the close, and "Current Price" equalled "Last Close" all
    evening. Yahoo does expose postMarketPrice/preMarketPrice, but the bigger
    problem is that it is one HTTP request per ticker: 133 of them, sequential.

    Schwab answers all 133 in a single call, is the broker of record, and marks
    the payload `realtime: true`.

MEASURED 2026-09-23 (probe_quotes_api.py)
    The method is `quotes` on the SCHWABDEV client — the project's SchwabClient
    wrapper forwards only fetch_positions, so reach it with get_client(). It
    takes one COMMA-JOINED STRING, the same shape `types` needs for
    transactions. The response is keyed by symbol.
"""
from __future__ import annotations

# Schwab rejects an over-long query string; chunk rather than discover the
# limit the hard way in production.
CHUNK = 250


def _num(v):
    try:
        if v is None or v == "":
            return None
        return float(str(v).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def extract_price(entry: dict) -> tuple[float | None, float | None]:
    """(price, day_percent) from one Schwab quote entry.

    Takes whichever block traded MOST RECENTLY rather than a fixed order.
    Preferring `quote` is wrong after hours — at 23:54 PT it still held AMD's
    16:00 close of 614.61 while `extended` had the live overnight 608.00.
    Preferring `extended` is equally wrong during the session, when it holds a
    stale pre-market print. The timestamp is the only thing that actually
    answers "which of these is the latest price".
    """
    q = entry.get("quote") or {}
    ext = entry.get("extended") or {}
    reg = entry.get("regular") or {}

    def _stamp(src) -> float:
        return max(_num(src.get("tradeTime")) or 0, _num(src.get("quoteTime")) or 0)

    def _last(src):
        for key in ("lastPrice", "mark"):
            v = _num(src.get(key))
            if v and v > 0:
                return v
        return None

    candidates = [(_stamp(ext), _last(ext)), (_stamp(q), _last(q))]
    candidates = [(t, p) for t, p in candidates if p]
    price = max(candidates, key=lambda c: c[0])[1] if candidates else None

    if price is None:                       # nothing live; fall back to closes
        for src, key in ((reg, "regularMarketLastPrice"), (q, "closePrice")):
            v = _num(src.get(key))
            if v and v > 0:
                price = v
                break

    # Percent comes from the same block as the price, so an overnight price is
    # never paired with the regular session's move.
    pct = None
    from_ext = bool(candidates) and price == _last(ext) and _stamp(ext) >= _stamp(q)
    order = ((ext, "netPercentChange"), (q, "netPercentChange"),
             (reg, "regularMarketPercentChange")) if from_ext else \
            ((q, "netPercentChange"), (reg, "regularMarketPercentChange"),
             (ext, "netPercentChange"))
    for src, key in order:
        v = _num(src.get(key))
        if v is not None:
            pct = v
            break
    return price, pct


def fetch_quotes(client_wrapper, symbols, log=print) -> dict:
    """Raw quote entries keyed by symbol. Never raises — an empty dict means
    'use whatever you had before', which keeps a quote outage from taking down
    a nine-minute signals run."""
    syms = [str(s).strip().upper() for s in symbols if str(s).strip()]
    if not syms:
        return {}

    inner = client_wrapper
    getter = getattr(client_wrapper, "get_client", None)
    if callable(getter):
        try:
            inner = getter() or client_wrapper
        except Exception as e:
            log(f"⚠️  quotes: could not get schwabdev client: {e}")
            return {}

    meth = getattr(inner, "quotes", None)
    if not callable(meth):
        log("⚠️  this schwabdev exposes no quotes() — falling back")
        return {}

    out: dict = {}
    for i in range(0, len(syms), CHUNK):
        batch = syms[i:i + CHUNK]
        try:
            resp = meth(",".join(batch))
            body = resp.json() if hasattr(resp, "json") else resp
            if isinstance(body, dict):
                out.update({k: v for k, v in body.items() if isinstance(v, dict)})
        except Exception as e:
            log(f"⚠️  quotes batch {i // CHUNK + 1} failed: {type(e).__name__}: {e}")
    return out


def price_map(client_wrapper, symbols, log=print) -> dict:
    """{SYMBOL: price} for whatever came back. Missing symbols are absent, not
    zero, so a caller can tell 'no quote' from 'priced at nothing'."""
    quotes = fetch_quotes(client_wrapper, symbols, log=log)
    out = {}
    for sym, entry in quotes.items():
        price, _ = extract_price(entry)
        if price is not None:
            out[sym.upper()] = price
    if out:
        log(f"Quotes: {len(out)}/{len(list(symbols))} symbol(s) priced live")
    return out
