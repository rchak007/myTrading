"""Bittensor subnet fundamentals from taostats, for the CRYPTO tab's SUBNET
section.

WHY THIS EXISTS
---------------
The CRYPTO tab already carries INDICATORS and RECOMMENDED for the subnet
alphas, all of it computed off Yahoo's `SN<netuid>-USD` daily bars. That is
price. It says nothing about the thing underneath the price.

A stock block answers that with FINANCIALS, off EDGAR. There is no 10-K for a
subnet, so FINANCIALS is deliberately switched off on this tab. This module is
its replacement: the on-chain figures that stand in for fundamentals when the
asset is an AMM-priced claim on an emission stream.

WHAT IT ACTUALLY TELLS YOU, AND WHAT IT CANNOT
----------------------------------------------
It tells you, from the chain:

  * the alpha price in TAO, and what the pool is worth
  * how much TAO the subnet is being paid per day, and its share of the
    TAO that all subnets together are being paid
  * how much alpha exists - the dilution you are swimming against
  * how many miners and validators are actually registered

It CANNOT tell you revenue. No subnet reports revenue on chain, and no
endpoint here carries it. Every revenue number in the research notes
(Lium's $2.7M billed, Targon's $10.4M) is self-reported off-chain and some of
it is stale by a year. So there is NO VERDICT in this section and no
revenue-over-emissions ratio computed for you. The section is figures, not a
judgement - deliberately, because the one judgement that matters here rests on
a number this module has no honest source for.

RATE LIMITS SHAPED THE DESIGN
-----------------------------
The free tier 429s after roughly five rapid calls. So:

  * TWO HTTP CALLS per refresh, both whole-list - every subnet in one
    response, not one call per ticker. 129 subnets came back in each.
  * refreshed at most once every TTL hours (default 20), cached to disk
  * every ticker on the tab reads the same cached snapshot

That also means the cache file accumulates daily snapshots, which is where
measured dilution comes from - see `_issuance()`. Nobody has to be told the
alpha issuance rate; two snapshots a day apart show it.

THE KEY IS NEVER WRITTEN BY THIS CODE
-------------------------------------
`_api_key()` READS a key from the environment or from a file that already
exists. It does not create one, copy one, or write one anywhere, and there is
no verb that does - which is the same rule the market-tracker key follows.
Without a key this module returns empty from every public function and the
SUBNET section simply does not render. Nothing else on the tab notices.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = "https://api.taostats.io"

# Yahoo's symbol -> netuid. Yahoo carries the subnet alphas via CoinMarketCap
# as SN<netuid>-USD, which makes most of this table mechanical. The two that
# are not:
#   VANTA-USD  Yahoo lists SN8 under its name, not its number. SN8-USD is
#              EMPTY. Confirmed against the chain: netuid 8 is "Vanta".
#   TAO22974   TAO itself, which is not a subnet at all. It maps to None and
#              gets the network-wide block instead - see `network_rows()`.
TICKER_NETUID = {
    "SN51-USD": 51,     # lium.io      GPU rental
    "SN4-USD": 4,       # Targon       confidential compute
    "SN64-USD": 64,     # Chutes       serverless inference
    "SN62-USD": 62,     # Ridges       AI software engineering
    "VANTA-USD": 8,     # Vanta        (ex-Prop Trading)
    "TAO22974-USD": None,
}

# Network issuance ceiling, TAO per day: 7200 blocks x 0.5 TAO after the
# halving. The interesting fact on the tab is how far BELOW this the chain
# actually runs - see `network_rows()`.
MAX_TAO_PER_DAY = 3600.0
BLOCKS_PER_DAY = 7200.0

STATE_DIR = Path(os.environ.get(
    "MYTRADING_STATE_DIR", Path.home() / ".local" / "state" / "myTrading"))
CACHE = Path(os.getenv("TAOSTATS_CACHE", STATE_DIR / "taostats_cache.json"))
# Where the key lives when it is not in the environment. remote_ops.py's
# `taostats_key` verb writes exactly this file and nothing else, and imports
# this constant rather than restating the path — a second copy of it is how
# the verb ends up writing somewhere the reader never looks.
KEY_FILE = Path(os.getenv(
    "TAOSTATS_KEY_FILE",
    Path.home() / ".config" / "myTrading" / "taostats.env")).expanduser()
TTL_HOURS = float(os.getenv("TAOSTATS_TTL_HOURS", "20"))
# How many daily snapshots to keep. 40 is enough to measure issuance over a
# month without the file growing without bound.
KEEP_SNAPSHOTS = 40


# ---------------------------------------------------------------- the key

def _api_key() -> str | None:
    """The taostats key, READ from wherever it already lives. Never written.

    Search order, first hit wins:

      1. TAOSTATS_API_KEY in the environment (which is how Pi 1's .env
         delivers it, since the job loads .env before importing anything)
      2. KEY_FILE - what the `taostats_key` ops verb writes. Either a bare
         `tao-...` line or a KEY=VALUE line.
      3. a short list of files that plausibly already hold it

    Step 3 exists because of a rule Chakravarti set for the market-tracker
    secret: *"have the other project read
    /home/chakravarti/agents/market-tracker/.env directly rather than copying
    the secret into its own file"*. The same rule applies here. One key, one
    place, read in situ.
    """
    v = os.getenv("TAOSTATS_API_KEY")
    if v and v.strip():
        return v.strip()

    home = Path.home()
    cands = [KEY_FILE, home / "agents" / "market-tracker" / ".env"]
    if (home / "agents").is_dir():
        cands += sorted((home / "agents").glob("*/.env"))

    for p in cands:
        try:
            if not p.is_file():
                continue
            for line in p.read_text(errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, _, val = line.partition("=")
                    if k.strip() == "TAOSTATS_API_KEY":
                        return val.strip().strip("'\"") or None
                elif line.startswith("tao-"):
                    # A bare-key file. Taostats keys carry a "tao-" prefix,
                    # which is what makes this safe to accept without a name:
                    # nothing else in a config file looks like one.
                    return line
        except Exception:                                   # noqa: BLE001
            continue
    return None


def have_key() -> bool:
    return _api_key() is not None


# ---------------------------------------------------------------- fetching

def _get(path: str, params: dict) -> list:
    """One GET, returning the `data` list. Raises on anything unexpected.

    urllib is NOT used here. Cloudflare answers it with error 1010 (access
    denied on client fingerprint) while requests goes straight through - half
    an hour went into that, so it is written down.
    """
    import requests
    key = _api_key()
    if not key:
        raise RuntimeError("no taostats API key")
    r = requests.get(BASE + path, params=params, timeout=60, headers={
        "Authorization": key, "accept": "application/json"})
    if r.status_code == 429:
        raise RuntimeError("taostats rate limit (429) - free tier allows "
                           "only a few calls a minute")
    r.raise_for_status()
    return r.json().get("data") or []


def _fetch() -> dict:
    """Both whole-list calls, folded into one netuid-keyed snapshot.

    Two calls, 129 subnets each. `limit=200` is deliberate: the default page
    is 50, and paging would multiply the call count by three for no reason.
    """
    pools = _get("/api/dtao/pool/latest/v1", {"limit": 200})
    subs = _get("/api/subnet/latest/v1", {"limit": 200})
    by_uid: dict[str, dict] = {}
    for row in pools:
        uid = row.get("netuid")
        if uid is None:
            continue
        # seven_day_prices is ~42 nested objects per subnet and the tab draws
        # its history from Yahoo anyway. Dropped before it is ever cached.
        row.pop("seven_day_prices", None)
        by_uid[str(uid)] = {"pool": row}
    for row in subs:
        uid = row.get("netuid")
        if uid is None:
            continue
        by_uid.setdefault(str(uid), {})["subnet"] = row
    return {"fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "subnets": by_uid}


def _load_cache() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except Exception:                                       # noqa: BLE001
        return {}


def _save_cache(c: dict) -> None:
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_suffix(".tmp")
        tmp.write_text(json.dumps(c))
        tmp.replace(CACHE)
    except Exception as e:                                  # noqa: BLE001
        print(f"⚠️  taostats cache not written: {e}")


def _stale(c: dict) -> bool:
    try:
        when = datetime.fromisoformat(c["fetched"])
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - when > timedelta(hours=TTL_HOURS)
    except Exception:                                       # noqa: BLE001
        return True


_MEM: dict | None = None            # per-process, so 12 blocks read disk once


def snapshot(force: bool = False, log=print) -> dict:
    """The current snapshot: cached, refreshed at most every TTL_HOURS.

    Returns {} when there is no key or the call fails - and on failure it
    returns the STALE CACHE rather than nothing, because yesterday's alpha
    supply is a far better answer than a blank section. The section says how
    old it is, so stale data cannot pass itself off as live.
    """
    global _MEM
    if _MEM is not None and not force:
        return _MEM
    c = _load_cache()
    if force or _stale(c):
        if have_key():
            try:
                fresh = _fetch()
                fresh["history"] = _roll(c.get("history") or [], fresh)
                _save_cache(fresh)
                c = fresh
            except Exception as e:                          # noqa: BLE001
                log(f"⚠️  taostats refresh failed ({type(e).__name__}: {e})"
                    + (" - using cached snapshot" if c else ""))
        elif not c:
            log("taostats: no API key (set TAOSTATS_API_KEY or "
                "TAOSTATS_KEY_FILE) - SUBNET section skipped")
    _MEM = c
    return c


def _roll(history: list, fresh: dict) -> list:
    """Append today's alpha supply per subnet, one entry per calendar day.

    This is the whole measured-dilution mechanism, and it costs nothing: the
    supply figure is already in the response, so keeping it turns a snapshot
    into a series. Same-day refreshes overwrite rather than stack.
    """
    day = fresh["fetched"][:10]
    entry = {"date": day, "alpha": {}, "tao_day": {}}
    for uid, blob in fresh["subnets"].items():
        pool, sub = blob.get("pool") or {}, blob.get("subnet") or {}
        a = _rao(pool.get("total_alpha"))
        if a is not None:
            entry["alpha"][uid] = a
        e = _rao(sub.get("emission"))
        if e is not None:
            entry["tao_day"][uid] = e * BLOCKS_PER_DAY
    out = [h for h in history if h.get("date") != day] + [entry]
    return out[-KEEP_SNAPSHOTS:]


# ---------------------------------------------------------------- numbers

def _f(v):
    """float or None. Taostats returns most numbers as decimal STRINGS."""
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _rao(v):
    """rao -> TAO (or alpha). Everything on chain is 1e-9 units."""
    f = _f(v)
    return None if f is None else f / 1e9


def _money(x, dp=0):
    return "" if x is None else f"${x:,.{dp}f}"


def _qty(x, dp=0):
    return "" if x is None else f"{x:,.{dp}f}"


def _pct(x, dp=1):
    return "" if x is None else f"{x:+.{dp}f}%"


def _issuance(uid: str, cache: dict) -> tuple[float | None, int | None]:
    """(alpha per day, days measured) for one subnet, from cached snapshots.

    MEASURED, not assumed. The nominal rate is often quoted as a flat 7,200
    alpha/day per subnet, but Taoflow changed what actually gets emitted and
    the honest way to know is to watch the supply move. Needs two snapshots at
    least a day apart; returns (None, None) until the cache has them, and the
    section shows a dash rather than a guess.
    """
    hist = [h for h in (cache.get("history") or []) if uid in (h.get("alpha") or {})]
    if len(hist) < 2:
        return None, None
    first, last = hist[0], hist[-1]
    try:
        d0 = datetime.strptime(first["date"], "%Y-%m-%d")
        d1 = datetime.strptime(last["date"], "%Y-%m-%d")
    except Exception:                                       # noqa: BLE001
        return None, None
    days = (d1 - d0).days
    if days < 1:
        return None, None
    delta = last["alpha"][uid] - first["alpha"][uid]
    return delta / days, days


# ---------------------------------------------------------------- rendering

def netuid_for(ticker) -> int | None:
    """netuid for a Yahoo symbol, or None. KeyError-free lookup for callers
    that want to know whether a ticker is a subnet at all."""
    return TICKER_NETUID.get(str(ticker).upper())


def is_subnet_ticker(ticker) -> bool:
    return str(ticker).upper() in TICKER_NETUID


def _age_note(cache: dict) -> str:
    try:
        when = datetime.fromisoformat(cache["fetched"])
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        hrs = (datetime.now(timezone.utc) - when).total_seconds() / 3600
        stamp = when.astimezone().strftime("%Y-%m-%d %H:%M")
        if hrs < 1:
            return f"on-chain, {stamp}"
        return f"on-chain, {stamp} ({hrs:.0f}h old)"
    except Exception:                                       # noqa: BLE001
        return "on-chain"


def subnet_rows(ticker, tao_usd=None, log=print) -> tuple[list, str]:
    """(groups, note) for the SUBNET section, shaped like oracle_value's.

    groups is [(label, names, values)] - the same shape _ind_rows and
    _oracle_rows already render, so orders_sheet needs no new painter.

    ([], "") for anything that is not a subnet alpha, which is how the stock
    Dashboard gets this section for free without a single HTTP call: the
    lookup fails and nothing happens.
    """
    name = str(ticker).upper()
    if name not in TICKER_NETUID:
        return [], ""
    cache = snapshot(log=log)
    if not cache.get("subnets"):
        return [], ""
    uid = TICKER_NETUID[name]
    if uid is None:
        return network_rows(cache, tao_usd), _age_note(cache)

    blob = cache["subnets"].get(str(uid)) or {}
    pool, sub = blob.get("pool") or {}, blob.get("subnet") or {}
    if not pool:
        return [], ""

    px_tao = _f(pool.get("price"))
    px_usd = px_tao * tao_usd if (px_tao and tao_usd) else None
    mcap_tao = _rao(pool.get("market_cap"))
    mcap_usd = mcap_tao * tao_usd if (mcap_tao and tao_usd) else None
    supply = _rao(pool.get("total_alpha"))
    staked = _rao(pool.get("alpha_staked"))
    in_pool = _rao(pool.get("alpha_in_pool"))
    pool_tao = _rao(pool.get("total_tao"))

    # TAO/day into this subnet's pool, and its share of what ALL subnets are
    # being paid. Two different denominators, both worth showing:
    #   Share_%  of the TAO actually flowing to subnets - this is the number
    #            that moves when Taoflow reallocates
    #   Of_Max_% of the 3,600/day ceiling - how much of the network's own
    #            issuance budget this subnet commands
    tao_day = (_rao(sub.get("emission")) or 0) * BLOCKS_PER_DAY \
        if sub.get("emission") is not None else None
    total_day = _total_tao_day(cache)
    share = (100 * tao_day / total_day) if (tao_day and total_day) else None
    of_max = (100 * tao_day / MAX_TAO_PER_DAY) if tao_day is not None else None

    issue, days = _issuance(str(uid), cache)
    dilute = (100 * issue * 365 / supply) if (issue and supply) else None

    groups = [
        ("SUBNET", ["Netuid", "Name", "Price_TAO", "Price_USD", "Day_%",
                    "Week_%", "Month_%", "Mcap_TAO", "Mcap_USD"],
         [uid, pool.get("name") or "", f"{px_tao:.6f}" if px_tao else "",
          f"${px_usd:,.2f}" if px_usd else "",
          _pct(_f(pool.get("price_change_1_day"))),
          _pct(_f(pool.get("price_change_1_week"))),
          _pct(_f(pool.get("price_change_1_month"))),
          _qty(mcap_tao), _money(mcap_usd)]),
        # EMISSION is the only income statement a subnet has. Everything in
        # this row is what the chain PAYS the subnet; none of it is revenue
        # the subnet earned from a customer.
        ("EMISSION", ["TAO_Day", "Share_%", "Of_Max_%", "Alpha_Day",
                      "Measured", "Dilution_Yr_%", "Recycled_24h",
                      "Reg_Cost_TAO", "Paying"],
         [_qty(tao_day, 1), "" if share is None else f"{share:.2f}%",
          "" if of_max is None else f"{of_max:.2f}%",
          _qty(issue, 0) if issue else "—",
          f"{days}d" if days else "—",
          "—" if dilute is None else f"{dilute:.1f}%",
          _qty(_rao(sub.get("recycled_24_hours")), 3),
          _qty(_rao(sub.get("neuron_registration_cost")), 4),
          "yes" if pool.get("subnet_emission_enabled") else "NO"]),
        # SUPPLY is the dilution you are swimming against, stated three ways
        # because they move differently: what exists, what is locked in stake,
        # and what is sitting in the AMM able to hit the bid tomorrow.
        ("SUPPLY", ["Alpha_Supply", "Staked", "In_Pool", "Pool_TAO",
                    "Root_Prop", "Liquidity_TAO"],
         [_qty(supply), _qty(staked), _qty(in_pool), _qty(pool_tao),
          "" if _f(pool.get("root_prop")) is None
          else f"{100 * _f(pool.get('root_prop')):.1f}%",
          _qty(_rao(pool.get("liquidity")))]),
        ("ACTIVITY", ["Miners", "Validators", "Max_UIDs", "Vol_24h_TAO",
                      "Buyers", "Sellers", "Buys", "Sells", "Sentiment",
                      "Registered"],
         [sub.get("active_miners", ""), sub.get("active_validators", ""),
          sub.get("max_neurons", ""), _qty(_rao(pool.get("tao_volume_24_hr"))),
          pool.get("buyers_24_hr", ""), pool.get("sellers_24_hr", ""),
          pool.get("buys_24_hr", ""), pool.get("sells_24_hr", ""),
          pool.get("fear_and_greed_sentiment") or "",
          str(sub.get("registration_timestamp") or "")[:10]]),
    ]
    return groups, _age_note(cache)


def _total_tao_day(cache: dict) -> float | None:
    """TAO per day flowing to ALL subnets, summed over the snapshot.

    Not a constant, and not 3,600. On 2026-11-15 the sum came to roughly
    1,027 TAO/day against a 3,600 ceiling, because Taoflow clips subnets with
    negative net TAO staking flow to zero emission and that TAO is simply not
    issued. Summed every time rather than hardcoded for exactly that reason.
    """
    tot = 0.0
    hit = False
    for blob in (cache.get("subnets") or {}).values():
        e = _rao((blob.get("subnet") or {}).get("emission"))
        if e:
            tot += e * BLOCKS_PER_DAY
            hit = True
    return tot if hit else None


def network_rows(cache: dict, tao_usd=None) -> list:
    """The network-wide block, for TAO itself rather than a subnet.

    TAO is not a subnet, so a SUBNET section about it would be a category
    error. What belongs here is the aggregate: how much of the issuance
    ceiling is actually being used, how the alpha complex compares with TAO's
    own cap, and which subnets are being paid at all.
    """
    subs = cache.get("subnets") or {}
    total_day = _total_tao_day(cache)
    paying = sum(1 for b in subs.values()
                 if (_rao((b.get("subnet") or {}).get("emission")) or 0) > 0)
    # Root (netuid 0) is TAO staked to root, not an alpha token - counted
    # separately so the alpha complex figure means what it says.
    alpha_mcap = 0.0
    for uid, b in subs.items():
        if uid == "0":
            continue
        m = _rao((b.get("pool") or {}).get("market_cap"))
        if m:
            alpha_mcap += m
    root_mcap = _rao(((subs.get("0") or {}).get("pool") or {}).get("market_cap"))

    ranked = sorted(
        ((uid, (b.get("pool") or {}).get("name") or uid,
          (_rao((b.get("subnet") or {}).get("emission")) or 0) * BLOCKS_PER_DAY)
         for uid, b in subs.items()), key=lambda t: t[2], reverse=True)
    top = ranked[0] if ranked else None

    used = (100 * total_day / MAX_TAO_PER_DAY) if total_day else None
    return [
        ("NETWORK", ["Subnets", "Paying_Emis", "Alpha_Mcap_TAO",
                     "Alpha_Mcap_USD", "Root_Stake_TAO", "TAO_USD"],
         [len(subs), paying, _qty(alpha_mcap),
          _money(alpha_mcap * tao_usd) if (alpha_mcap and tao_usd) else "",
          _qty(root_mcap), _money(tao_usd, 2) if tao_usd else ""]),
        # The headline fact of the whole tab. Nominal issuance is 3,600
        # TAO/day; what is actually reaching subnets is a fraction of it,
        # and the rest is never minted. Dilution is therefore far slower
        # than the nominal number implies - which cuts both ways, because
        # it is also the emission a subnet is competing for.
        ("ISSUANCE", ["TAO_Day_Issued", "Ceiling", "Used_%", "Clipped_%",
                      "Top_Subnet", "Top_TAO_Day", "Top_Share_%"],
         [_qty(total_day, 1), _qty(MAX_TAO_PER_DAY),
          "" if used is None else f"{used:.1f}%",
          "" if used is None else f"{100 - used:.1f}%",
          f"{top[1]} (SN{top[0]})" if top else "",
          _qty(top[2], 1) if top else "",
          f"{100 * top[2] / total_day:.1f}%" if (top and total_day) else ""]),
    ]


if __name__ == "__main__":                                  # pragma: no cover
    import sys
    force = "--force" in sys.argv
    snap = snapshot(force=force)
    if not snap:
        print("no snapshot (no key, or first call failed)")
        raise SystemExit(1)
    print(f"fetched {snap['fetched']}  subnets={len(snap['subnets'])}  "
          f"snapshots={len(snap.get('history') or [])}")
    tao = _f(os.getenv("TAO_USD")) or 190.80
    for t in TICKER_NETUID:
        groups, note = subnet_rows(t, tao_usd=tao)
        print(f"\n=== {t}   {note}")
        for label, names, vals in groups:
            print(f"  {label}")
            for n, v in zip(names, vals):
                print(f"    {n:<16} {v}")
