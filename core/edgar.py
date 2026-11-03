"""
core/edgar.py
=============
Annual revenue straight from SEC filings, so the OracleValue anchor is built
on ten-plus years instead of four.

WHY THIS EXISTS. The whole model rests on the stock's OWN mean price-to-sales
(see core/oracle.py). We were computing that mean from yfinance's income
statement, which returns only FOUR fiscal years. The label said "10-year mean
P/S"; the arithmetic was a four-year mean. That single fact explains every
calibration error we had:

    NVDA  the four years are 2023-2026 -- 34x, 43x, 29x, 22x. The AI bubble,
          trending DOWN. Averaging them gives 32x against a current 14.6x,
          so the model called a $205 stock worth $434.
    AAOI  four years, P/S scattering 1x to 6x. Mean of noise.
    SITM  four years, all post-IPO.
    TSLA  four years that miss the entire 2016-2021 growth era -- which is
          exactly why their 10-year column lands within 3% and ours did not.

Recency-weighting was tried first and rejected: it moves NVDA from 32.0 to
29.6, nowhere near enough. The sample was wrong, not the estimator. Do not
tune the weighting; fix the years.

WHAT EDGAR GIVES US, free and with no API key:

    NVDA  18 fiscal years (2008-2026)
    TSLA  17 fiscal years (2009-2025)
    MSFT  11 fiscal years
    AAOI  10 fiscal years
    SITM   8 fiscal years -- a real limit, not a data one; it IPO'd in 2019

TERMS OF USE. SEC asks for a descriptive User-Agent with a contact address
and no more than 10 requests/second. Both are honoured below. There is no
key, no account, and no quota beyond that.

CARE WITH THE RAW FACTS. companyconcept returns every XBRL fact ever filed
for a tag, including quarterly figures, segment breakdowns, and restatements
of the same year across several filings. Three filters make it an annual
series: form is 10-K, the period spans a full year, and for any one fiscal
year the most recently FILED value wins (so a restatement supersedes the
original).
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from datetime import date, datetime
from pathlib import Path

# SEC requires a contact address. This is Chakravarti's own, and goes only to
# the SEC in a User-Agent header, which is exactly what they ask for.
UA = os.getenv("EDGAR_UA",
               "myTrading research contact geniusact@keep-empowering.com")

CACHE = Path(os.getenv(
    "EDGAR_CACHE",
    Path(os.getenv("REMOTE_OPS_STATE",
                   Path.home() / ".local" / "state" / "myTrading"))
    / "edgar_cache"))

# Filings land quarterly at most, so a fortnight-old cache is still current
# and keeps the weekly run from hammering a public service for nothing.
MAX_AGE_DAYS = int(os.getenv("EDGAR_MAX_AGE_DAYS", "14"))

MIN_RATE_GAP = 0.12                 # ~8 req/s, inside SEC's limit of 10
_last_call = [0.0]

# Companies report revenue under different tags depending on era and
# industry. Tried in order; the first with a usable series wins.
REVENUE_TAGS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)

MIN_YEARS = 5                       # below this, fall back to yfinance


def annual_revenue(ticker: str, log=None) -> dict[int, float]:
    """{fiscal_year: revenue} from 10-K filings. {} when unavailable.

    Fail-soft on purpose: this is an upgrade to an input that already has a
    working fallback, so a dead network must cost us precision, never the
    whole weekly run.
    """
    ticker = ticker.upper().strip()
    cached = _read_cache(ticker)
    if cached is not None:
        return cached

    try:
        cik = cik_for(ticker)
        if not cik:
            _write_cache(ticker, {})
            return {}
        best = _merge(cik, REVENUE_TAGS, instant=False)
        if len(best) < MIN_YEARS:
            best = {}
        _write_cache(ticker, best)
        if log and best:
            yrs = sorted(best)
            log(f"   EDGAR {ticker}: {len(yrs)} fiscal years "
                f"{yrs[0]}..{yrs[-1]}")
        return best
    except Exception as e:                               # noqa: BLE001
        if log:
            log(f"   EDGAR {ticker}: {type(e).__name__} — using yfinance")
        return {}


def annual_series(ticker: str, tags, instant=False) -> dict[int, float]:
    """{fiscal_year: value} for the first of `tags` that the filer uses.

    `instant=True` for balance-sheet items, which XBRL reports as a point in
    time (one date, no start) rather than a period. Net income is a period;
    stockholders' equity is an instant. Asking for the wrong one silently
    returns nothing, which is how a moat model ends up with no evidence and
    rates everything NONE.
    """
    ticker = ticker.upper().strip()
    key = "_".join(t[:12] for t in tags) + ("_i" if instant else "")
    cached = _read_cache(ticker, key)
    if cached is not None:
        return cached
    try:
        cik = cik_for(ticker)
        if not cik:
            _write_cache(ticker, {}, key)
            return {}
        _write_cache(ticker, (merged := _merge(cik, tags, instant)), key)
        return merged
    except Exception:                                    # noqa: BLE001
        return {}


NET_INCOME_TAGS = ("NetIncomeLoss",
                   "ProfitLoss",
                   "NetIncomeLossAvailableToCommonStockholdersBasic")
EQUITY_TAGS = ("StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
GROSS_PROFIT_TAGS = ("GrossProfit",)


def cik_for(ticker: str) -> int | None:
    """Ticker -> CIK, from SEC's own mapping file (cached)."""
    m = _ticker_map()
    return m.get(ticker.upper().strip())


def _merge(cik: int, tags, instant: bool) -> dict[int, float]:
    """All tags combined into one series, earlier tags winning a tie.

    MERGED, NOT PICKED. Filers change tags as accounting standards change,
    so no single tag spans a company's history. Visa stopped reporting
    `StockholdersEquity` after 2011 and switched; taking whichever tag had
    the most years gave six ancient years of equity against net income from
    2020-2025 -- no overlap at all in the modern era, and a moat rated off
    four pre-2012 observations. Revenue has the same break at ASC 606 in
    2018, where `SalesRevenueNet` gives way to
    `RevenueFromContractWithCustomerExcludingAssessedTax`.

    Later tags only FILL YEARS the earlier ones lack, so the preferred tag
    still defines any year it covers and the series cannot mix two
    definitions for one year.
    """
    out: dict[int, float] = {}
    for tag in tags:
        for year, val in _concept(cik, tag, instant=instant).items():
            out.setdefault(year, val)
    return out


def _concept(cik: int, tag: str, instant=False) -> dict[int, float]:
    """One XBRL tag as an annual series, newest restatement winning."""
    url = (f"https://data.sec.gov/api/xbrl/companyconcept/"
           f"CIK{cik:010d}/us-gaap/{tag}.json")
    try:
        d = _get(url)
    except Exception:                                    # noqa: BLE001
        return {}                       # tag not used by this filer

    out: dict[int, tuple[float, str]] = {}
    for x in d.get("units", {}).get("USD", []):
        start, end = x.get("start"), x.get("end")
        if not end or x.get("form") != "10-K":
            continue
        try:
            b = date.fromisoformat(end)
            a = date.fromisoformat(start) if start else None
        except ValueError:
            continue
        if instant:
            # A balance-sheet item: one date, no period. Filings also carry
            # the PRIOR year's balance as a comparative, which is fine —
            # keyed by its own year, it simply agrees with that year's row.
            if a is not None:
                continue
        # A full fiscal year. Excludes quarters, and the 52/53-week
        # retail calendars still land inside this window.
        elif a is None or not 350 <= (b - a).days <= 380:
            continue
        val, filed = x.get("val"), x.get("filed", "")
        if val is None:
            continue
        # Same year filed twice means a restatement; the later one is the
        # company's current view of its own history.
        if b.year not in out or filed > out[b.year][1]:
            out[b.year] = (float(val), filed)
    return {y: v for y, (v, _) in out.items()}


def _ticker_map() -> dict[str, int]:
    path = CACHE / "company_tickers.json"
    if _fresh(path):
        try:
            return {k: int(v) for k, v in
                    json.loads(path.read_text()).items()}
        except Exception:                                # noqa: BLE001
            pass
    try:
        raw = _get("https://www.sec.gov/files/company_tickers.json")
    except Exception:                                    # noqa: BLE001
        return {}
    m = {}
    for row in (raw.values() if isinstance(raw, dict) else raw):
        try:
            m[str(row["ticker"]).upper()] = int(row["cik_str"])
        except (KeyError, TypeError, ValueError):
            continue
    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(m))
    except Exception:                                    # noqa: BLE001
        pass
    return m


def _get(url: str):
    """One rate-limited GET. SEC asks for <=10/s; we pace at ~8."""
    gap = time.monotonic() - _last_call[0]
    if gap < MIN_RATE_GAP:
        time.sleep(MIN_RATE_GAP - gap)
    _last_call[0] = time.monotonic()
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
    with urllib.request.urlopen(req, timeout=45) as r:
        data = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            import gzip
            data = gzip.decompress(data)
    return json.loads(data)


def _path(ticker: str, key: str = "revenue") -> Path:
    return CACHE / f"{ticker.upper()}_{key}.json"


def _read_cache(ticker: str, key: str = "revenue"):
    p = _path(ticker, key)
    if not _fresh(p):
        return None
    try:
        d = json.loads(p.read_text())
        return {int(k): float(v) for k, v in d.items()}
    except Exception:                                    # noqa: BLE001
        return None


def _write_cache(ticker: str, data: dict, key: str = "revenue") -> None:
    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        _path(ticker, key).write_text(json.dumps({str(k): v
                                                  for k, v in data.items()}))
    except Exception:                                    # noqa: BLE001
        pass                            # a cache we cannot write is not fatal


def _fresh(p: Path) -> bool:
    try:
        age = (datetime.now()
               - datetime.fromtimestamp(p.stat().st_mtime)).days
        return age <= MAX_AGE_DAYS
    except Exception:                                    # noqa: BLE001
        return False
