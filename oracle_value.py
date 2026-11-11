#!/usr/bin/env python3
"""
oracle_value.py
===============
Weekly OracleValue for every ticker — the slow half, cached like the Benter
targets and for the same reason.

    .venv/bin/python oracle_value.py --refresh     # fetch + compute + cache
    .venv/bin/python oracle_value.py               # show the cache
    .venv/bin/python oracle_value.py --one AAOI    # one name, verbose

WEEKLY, ON MONDAY. Chakravarti, 2026-11-01: *"lets do that every Monday 1st
run."* The inputs barely move — revenue, margins, cash flow and share count
change once a quarter — so recomputing hourly would cost 135 network calls to
learn nothing. Price is the only fast-moving term, and the sheet already has
it live, so the BAND is recomputed every cycle against a cached value.

OUR OWN RATINGS, NOT ANYONE ELSE'S. The moat comes from core/moat.py, which
reads ten years of SEC filings and judges durable excess returns on capital.
Nothing here is seeded from a published value.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from core.oracle import value as oracle_value          # noqa: E402

# In the repo, not the state dir — same transport as benter_targets.csv:
# Pi 2 computes, git carries it, Pi 1 reads. Versioned on purpose, so "what
# did the model think in November" stays answerable.
FILE = Path(os.getenv("ORACLE_FILE", HERE / "oracle_values.csv"))
COLS = ["Ticker", "Fetched", "Price_At_Fetch", "Oracle_Value", "Oracle_Low",
        "Oracle_High", "Oracle_Band", "Moat", "Oracle_Why", "Oracle_IQ",
        "Last_Earnings", "Next_Earnings",
        # The anchor's provenance, stored rather than parsed back out of
        # Oracle_Why: how many years the mean P/S rests on, and how much
        # those years agreed. Both decide how far to trust the value.
        "PS_Mean", "PS_Years", "PS_Spread",
        # FUNDAMENTALS RIDE ALONG. The weekly refresh already fetches
        # .info for every ticker, so these cost nothing extra here and
        # save 100+ API calls per sheet rebuild. Chakravarti, 2026-11-09:
        # "can you add PE ratio and PEG too. slowly i want to build some
        # more fundamental analysis" — so the column list is meant to grow.
        "PE", "Fwd_PE", "PEG", "EPS", "Net_Margin", "ROE", "Rev_Growth",
        "Debt_Equity", "Mkt_Cap_M", "Div_Yield",
        # THE FIVE-YEAR NORMALISED FIGURES, SHOWN not just used. The doc
        # told him to "compare against the five-year average margin in the
        # ORACLE section" — and that section never displayed it. He read
        # BASIS/Years 18 as the answer, which is years of SALES history for
        # the P/S anchor, an entirely different number. Advice that points
        # at something invisible is a bug in the advice.
        "Net_Margin_5y", "ROE_5y", "Margin_Shape", "Margin_Series"]

# THE HARDCODED MOAT TABLE IS GONE. It held four ratings read off
# StockOracle's site and fed them straight into our valuation, which made
# "our" moat theirs. Chakravarti, 2026-11-01: *"i am developing this so we
# make our own independent ratings and not hardcode from StockOracle"* — he
# supplied their screens so we could learn WHICH MEASURES MATTER and compute
# them ourselves, not to copy the outputs. See core/moat.py, which rates on
# ten years of SEC filings: median return on equity, how often it clears a
# cost-of-equity hurdle, and the level and steadiness of gross margin.

MAX_AGE_DAYS = int(os.getenv("ORACLE_MAX_AGE_DAYS", "14"))


def tickers() -> list[str]:
    """STOCK_TICKERS out of app.py by AST, never by import — app.py drags in
    schwabdev and Streamlit, which this job does not need and Pi 2 lacks."""
    import ast
    tree = ast.parse((HERE / "app.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "STOCK_TICKERS" for t in node.targets):
            return sorted({str(e.value) for e in node.value.elts
                           if isinstance(e, ast.Constant)})
    raise RuntimeError("no STOCK_TICKERS in app.py")


def load(log=print) -> dict:
    if not FILE.exists():
        return {}
    try:
        with FILE.open(newline="", encoding="utf-8") as fh:
            return {str(r["Ticker"]).upper(): r for r in csv.DictReader(fh)
                    if r.get("Ticker")}
    except Exception as e:                               # noqa: BLE001
        log(f"⚠️  could not read {FILE.name}: {e}")
        return {}


def five_year_margin(tk) -> float | None:
    """Mean NET MARGIN over the last five FISCAL years — StockOracle's
    "5-year Average", reproduced exactly on TSLA (10.50%).

    Fiscal years only; the TTM column is excluded, which is the whole point —
    TTM is the trough being normalized away.
    """
    try:
        fin = tk.income_stmt
        rev, ni = fin.loc["Total Revenue"], fin.loc["Net Income"]
        m = (ni / rev).dropna()
        m = m[:5]
        return float(m.mean()) if len(m) >= 3 else None
    except Exception:                                    # noqa: BLE001
        return None


def our_moat(sym: str):
    """Our own moat rating for one ticker, or None when we cannot judge.

    None is a real answer and must stay distinguishable from NONE: "we have
    too few filings to say" and "this company has no durable advantage" are
    different claims, and core/oracle.py treats a missing moat as neutral
    rather than as a penalty.
    """
    try:
        from core import moat
        rating, _ = moat.from_edgar(sym)
        return rating
    except Exception:                                    # noqa: BLE001
        return None


def five_year_roe(tk) -> float | None:
    """Mean RETURN ON EQUITY over the last five fiscal years.

    Same correction as five_year_margin, for the same reason. TSLA's page
    prints ROE TTM 3.80% beside a 5-year average of 19.61%, and their
    OracleIQ ranks profitability HIGH — which only makes sense against the
    average. yfinance's `returnOnEquity` is the trailing 3.80%, so feeding
    it straight in ranks a High company Medium.
    """
    try:
        fin, bal = tk.income_stmt, tk.balance_sheet
        ni = fin.loc["Net Income"]
        eq = next(bal.loc[k] for k in
                  ("Stockholders Equity", "Total Stockholder Equity",
                   "Common Stock Equity") if k in bal.index)
        r = (ni / eq).dropna()
        r = r[:5]
        return float(r.mean()) if len(r) >= 3 else None
    except Exception:                                    # noqa: BLE001
        return None


def revenue_cagr(tk) -> float | None:
    """Compound annual revenue growth across the available fiscal years.

    StockOracle ranks TSLA growth HIGH while its trailing year-over-year is
    flat-to-down. One year against a hard comparison is not a growth rate;
    the multi-year compound is. Same normalization lesson as the margin.
    """
    try:
        rev = tk.income_stmt.loc["Total Revenue"].dropna()
        if len(rev) < 3:
            return None
        newest, oldest = float(rev.iloc[0]), float(rev.iloc[-1])
        years = len(rev) - 1
        if oldest <= 0 or newest <= 0 or years < 2:
            return None
        return (newest / oldest) ** (1.0 / years) - 1.0
    except Exception:                                    # noqa: BLE001
        return None


def mean_ps(tk, shares) -> float | None:
    """The stock's OWN mean price-to-sales over the available fiscal years.

    THE KEY TO THE WHOLE THING. TSLA's Valuation Chart shows nine sub-models
    spanning $57.95 to $858.10, and OracleValue 320.80 sits at 91% of the
    Mean-Price-to-Sales model (352.76) — on MSFT it is 105% of it. Better
    still: TSLA's 10-year average P/S of 10.63 times revenue per share of
    29.25 gives $310.93 against a published $320.80, 3% out.

    So OracleValue is NOT a universal fair multiple. It is what THIS stock has
    historically been worth per dollar of sales, applied to today's sales.
    That is why SiTime earns 33x and Applied Optoelectronics earns 3.8x on
    identical 3% net margins — the market has always paid those multiples.

    A universal BASE_PE framework cannot reproduce that, which is why the
    first version of this module put TSLA at $22.67 against $320.80.

    REVENUE COMES FROM EDGAR, NOT YFINANCE. yfinance returns four fiscal
    years, so this was silently a four-year mean wearing a ten-year label —
    the single cause of every calibration error we had. On NVDA those four
    years were 34x, 43x, 29x, 22x (the AI bubble, trending down); their mean
    of 32x against a current 14.6x valued a $205 stock at $434. SEC filings
    give 18 years for NVDA and 17 for TSLA. See core/edgar.py.

    Returns (mean_ps, n_years, dispersion). Dispersion is the coefficient of
    variation of the yearly figures — a mean is only worth having if the
    thing averaged is stable, and this is how the caller knows. MSFT scores
    0.09 and we value it to within 0%; AAOI scores 0.50 and we do not.
    """
    try:
        import numpy as np
        from core import edgar

        px = tk.history(period="max", interval="1mo")["Close"]
        if px.empty or not shares:
            return None, 0, None

        rev = {y: v for y, v in
               edgar.annual_revenue(getattr(tk, "ticker", "")).items()}
        if len(rev) < edgar.MIN_YEARS:
            # Fall back to yfinance's four years rather than nothing. Fewer
            # years is a worse anchor, not an absent one — n_years travels
            # with the number so the caller can discount it.
            fin = tk.income_stmt
            rev = {ts.year: float(r)
                   for ts, r in fin.loc["Total Revenue"].dropna().items()
                   if r and r > 0}

        vals = []
        for year, r in sorted(rev.items()):
            if not r or r <= 0:
                continue
            yr = px[px.index.year == year]
            if len(yr):
                vals.append(float(yr.mean()) * shares / float(r))
        if len(vals) < 3:
            return None, 0, None

        # RECENCY-WEIGHTED, not a flat mean. Deep history is what makes the
        # anchor trustworthy, but a flat mean assumes a company reverts to
        # what it was worth a decade ago. MSFT's 11-year flat mean P/S is
        # 9.0x against 11.4x over the last four years, because the move to
        # cloud genuinely re-rated it — a flat mean reads that re-rating as
        # an error to be averaged away, and valued MSFT 15% low.
        #
        # TESTED AND LEFT OFF BY DEFAULT. On the P/S anchor alone, recency
        # weighting looked decisive — flat 36% mean error against 23% at a
        # five-year half-life. Inside the FULL model it is a wash: 24% flat
        # against 25% weighted, helping MSFT (-15% -> -9%) and SITM while
        # hurting TSLA (+7% -> -13%) and AAOI (-19% -> -27%). The other
        # anchors were already doing that work, so it double-counted.
        #
        # So it stays available and stays off. With four published anchors,
        # a knob that buys one point of mean error is a knob fitted to
        # noise, and flat is both simpler and more conservative where it
        # matters most: NVDA at $286 rather than $353 against a $205 price.
        # Revisit when there are a dozen published values to calibrate on.
        n = len(vals)
        hl = float(os.getenv("ORACLE_PS_HALFLIFE", "0") or 0) or float(n * 50)
        w = [0.5 ** ((n - 1 - k) / hl) for k in range(n)]
        m = float(sum(v * wi for v, wi in zip(vals, w)) / sum(w))
        # Dispersion stays unweighted: the question it answers is whether the
        # market ever agreed on a level, not what it thinks lately.
        cv = float(np.std(vals) / np.mean(vals)) if np.mean(vals) else None
        return m, n, cv
    except Exception:                                    # noqa: BLE001
        return None, 0, None


def compute_one(sym: str, info: dict | None = None, price=None):
    """One ticker -> Oracle. Returns None when yfinance gives us nothing."""
    import yfinance as yf
    i = info if info is not None else (yf.Ticker(sym).info or {})
    px = price or i.get("currentPrice") or i.get("previousClose")
    if not px:
        return None
    fpe = i.get("forwardPE")
    tk = yf.Ticker(sym)
    ps, yrs, cv = mean_ps(tk, i.get("sharesOutstanding"))
    return oracle_value(
        px,
        shares=i.get("sharesOutstanding"),
        revenue=i.get("totalRevenue"),
        net_income=i.get("netIncomeToCommon"),
        operating_cf=i.get("operatingCashflow"),
        free_cf=i.get("freeCashflow"),
        gross_margin=i.get("grossMargins"),
        net_margin=i.get("profitMargins"),
        roe=i.get("returnOnEquity"),
        market_cap=i.get("marketCap"),
        growth=i.get("revenueGrowth"),
        beta=i.get("beta"),
        total_debt=i.get("totalDebt"),
        total_cash=i.get("totalCash"),
        forward_eps=(px / fpe) if fpe and fpe > 0 else None,
        moat=our_moat(sym),
        normalized_margin=five_year_margin(tk),
        proj_growth=i.get("earningsGrowth") or i.get("revenueGrowth"),
        hist_ps=ps,
        hist_ps_years=yrs,
        hist_ps_dispersion=cv,
    )


def _fundamentals(i: dict) -> dict:
    """The plain ratios, straight from .info, rounded for the sheet.

    Blank where the figure does not exist rather than zero: an ETF has no
    PEG and a loss-maker has no meaningful P/E, and printing 0.00 would
    assert a number nobody computed. Same rule as everywhere else here.
    """
    def n(key, mult=1.0, places=2):
        v = i.get(key)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return ""
        if v != v:                       # NaN
            return ""
        return round(v * mult, places)

    cap = i.get("marketCap")
    return {
        "PE": n("trailingPE"),
        "Fwd_PE": n("forwardPE"),
        "PEG": n("trailingPegRatio"),
        "EPS": n("trailingEps"),
        "Net_Margin": n("profitMargins", 100.0, 1),
        "ROE": n("returnOnEquity", 100.0, 1),
        "Rev_Growth": n("revenueGrowth", 100.0, 1),
        "Debt_Equity": n("debtToEquity", 1.0, 1),
        "Mkt_Cap_M": round(float(cap) / 1e6, 0) if cap else "",
        "Div_Yield": n("dividendYield", 1.0, 2),
    }


def earnings_dates(sym: str, tk=None) -> tuple[str, str]:
    """(last reported, next expected) as ISO dates, or ("", "").

    Each ticker reports on its own schedule, which is the whole point: the
    value is a quarterly object, and the quarter turns over on a different
    day for every name.
    """
    import yfinance as yf
    try:
        tk = tk or yf.Ticker(sym)
        ed = tk.get_earnings_dates(limit=12)
        if ed is None or not len(ed):
            return "", ""
        today = date.today()
        past, future = [], []
        for ts, row in ed.iterrows():
            d = ts.date() if hasattr(ts, "date") else None
            if d is None:
                continue
            # Reported = an actual EPS exists. A date alone is only a
            # scheduled event, and scheduled dates move.
            got = row.get("Reported EPS")
            if d <= today and got == got and got is not None:
                past.append(d)
            elif d > today:
                future.append(d)
        return (max(past).isoformat() if past else "",
                min(future).isoformat() if future else "")
    except Exception:                                    # noqa: BLE001
        return "", ""


def reported_since_fetch(row) -> bool:
    """Did this ticker report after we last valued it?

    The trigger for a per-ticker refresh. Revenue per share, the normalized
    margin and the predictability record all move on earnings and at no
    other time, so this is when the number is genuinely new.
    """
    last, fetched = str(row.get("Last_Earnings", "")), str(row.get("Fetched", ""))
    if not last or not fetched:
        return False
    return last[:10] > fetched[:10]


def due_tickers(log=print) -> list[str]:
    """Which tickers to recompute on this run.

    Two triggers, deliberately different in scope:

      Monday (or a stale cache) -> everything. The cheap heartbeat.
      Any other day             -> only names that have reported since we
                                   last valued them, which is usually none
                                   and occasionally three.

    Chakravarti asked for both: 'weekly update due to price point of view'
    and 'get data after earnings ... per ticker, each will be different
    dates'. The band already tracks price on every run via for_sheet(); this
    is about the value underneath it.
    """
    cache = load(log=lambda *_: None)
    if due_today(log=log):
        return tickers()
    fresh = [t for t, r in cache.items() if reported_since_fetch(r)]
    if fresh:
        log(f"post-earnings refresh for {len(fresh)}: {', '.join(sorted(fresh))}")
    return sorted(fresh)


def iq_line(sym: str, info: dict, o) -> str:
    """The six-factor OracleIQ panel as one sheet-sized line.

    Fail-soft on purpose: the panel is context beside the value, never the
    value itself, so an earnings-history fetch that 500s must cost us the
    label and nothing else.
    """
    import yfinance as yf
    from core import oracle_iq
    try:
        tk = yf.Ticker(sym)
        try:
            ed = tk.get_earnings_dates(limit=24)
        except Exception:                                # noqa: BLE001
            ed = None
        return oracle_iq.summary(oracle_iq.panel(
            earnings=ed,
            moat=o.moat,
            band=o.band,
            net_margin=info.get("profitMargins"),
            normalized_margin=five_year_margin(tk),
            roe=five_year_roe(tk) or info.get("returnOnEquity"),
            gross_margin=info.get("grossMargins"),
            revenue_cagr=revenue_cagr(tk),
            revenue_growth=info.get("revenueGrowth"),
            earnings_growth=info.get("earningsGrowth"),
            total_debt=info.get("totalDebt"),
            total_cash=info.get("totalCash"),
            free_cf=info.get("freeCashflow"),
            operating_cf=info.get("operatingCashflow"),
            market_cap=info.get("marketCap"),
            ebitda=info.get("ebitda"),
            current_ratio=info.get("currentRatio"),
        ))
    except Exception:                                    # noqa: BLE001
        return ""


def due_today(log=print) -> bool:
    """Monday, or the cache is missing or stale. Chakravarti asked for
    'every Monday 1st run' — so the job is safe to schedule daily and will
    simply do nothing on the other six days.

    Also fires when the cache has gone past MAX_AGE_DAYS, so a Monday the Pi
    happened to be down does not leave the column blank all week.
    """
    cache = load(log=lambda *_: None)
    if not cache:
        log("no cache — refreshing regardless of weekday")
        return True
    ages = [a for a in (age_days(r) for r in cache.values()) if a is not None]
    if ages and min(ages) > MAX_AGE_DAYS:
        log(f"cache is {min(ages)}d old (> {MAX_AGE_DAYS}) — refreshing")
        return True
    if date.today().weekday() == 0:
        if ages and min(ages) == 0:
            log("already refreshed today")
            return False
        return True
    log(f"not Monday ({date.today():%A}) and cache is "
        f"{min(ages) if ages else '?'}d old — nothing to do")
    return False


def refresh(syms=None, log=print) -> int:
    """Recompute and REPLACE the cache. A failed ticker keeps its old row.

    `syms` may be a SUBSET — that is how the post-earnings path works. Rows
    for tickers not in the subset are carried through untouched, so a
    one-ticker refresh never silently empties the other 134.
    """
    import yfinance as yf
    syms = list(syms) if syms else tickers()
    prev, out, failed, refused = load(log=lambda *_: None), [], [], []
    today = date.today().isoformat()
    asked = {s.upper() for s in syms}
    carried = [r for t, r in prev.items() if t.upper() not in asked]

    for n, s in enumerate(syms, 1):
        try:
            i = yf.Ticker(s).info or {}
            why_refused = ""
            o = compute_one(s, i)
            if o is not None and o.value is None:
                # A DELIBERATE REFUSAL, NOT A FAILURE. The model looked at
                # this ticker and declined to value it — pre-revenue, or an
                # answer so far from the share price that an input must be
                # wrong.
                #
                # IT BLANKS THE VALUE, IT DOES NOT DELETE THE ROW. That was
                # the first version and it was wrong at a level I had not
                # thought about: FINANCIALS reads from this same cache, so
                # dropping the row took PE, PEG, the five-year margin and
                # the margin series with it. TSM and SNDK vanished from the
                # Dashboard entirely — two companies that report perfectly
                # well and simply cannot be PRICED by this method. Refusing
                # to assert a fair value is not a reason to withhold the
                # reported facts.
                #
                # The value stays empty and Oracle_Why says why, so nothing
                # the guard rejected can reappear: RCAT at 8,288,439x the
                # share price came back once already, restored from cache by
                # a retry, and must not again.
                refused.append(f"{s} ({_short(o.why)})")
                why_refused = o.why
                o = None            # fall through, writing a valueless row
            last_e, next_e = earnings_dates(s)
            ps, yrs, cv = mean_ps(yf.Ticker(s), i.get("sharesOutstanding"))
            out.append({
                "Ticker": s, "Fetched": today,
                "Price_At_Fetch": i.get("currentPrice") or i.get("previousClose") or "",
                "Oracle_Value": o.value if o else "",
                "Oracle_Low": o.low if o else "",
                "Oracle_High": o.high if o else "",
                "Oracle_Band": o.band if o else "",
                "Moat": (o.moat if o else "") or our_moat(s),
                "Oracle_Why": o.why if o else why_refused,
                "Oracle_IQ": iq_line(s, i, o) if o else "",
                "Last_Earnings": last_e, "Next_Earnings": next_e,
                "PS_Mean": round(ps, 2) if ps else "",
                "PS_Years": yrs or "",
                "PS_Spread": round(cv, 3) if cv is not None else "",
                **_fundamentals(i),
                "Net_Margin_5y": (round(m5 * 100, 1)
                                  if (m5 := five_year_margin(yf.Ticker(s)))
                                  is not None else ""),
                "ROE_5y": (round(r5 * 100, 1)
                           if (r5 := five_year_roe(yf.Ticker(s)))
                           is not None else ""),
                **dict(zip(("Margin_Shape", "Margin_Series"),
                           margin_trend(s))),
            })
        except Exception as e:                           # noqa: BLE001
            # A TRANSIENT failure — network, a yfinance hiccup. Here, and
            # only here, yesterday's row is better than a blank.
            failed.append(f"{s} ({type(e).__name__})")
            if s in prev:
                out.append(prev[s])
        if n % 25 == 0:
            log(f"   …{n}/{len(syms)}")

    out.extend(carried)
    out.sort(key=lambda r: str(r.get("Ticker", "")))

    try:
        FILE.parent.mkdir(parents=True, exist_ok=True)
        with FILE.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(out)
    except Exception as e:                               # noqa: BLE001
        log(f"⛔ could not write {FILE}: {e}")
        return 0

    fresh = sum(1 for r in out if str(r.get("Fetched", "")).startswith(today))
    log(f"{len(out)} row(s) cached — {fresh} fresh, {len(out) - fresh} kept")
    if refused:
        log(f"🚫 declined to value {len(refused)} (value blank, "
            f"fundamentals kept): "
            + ", ".join(refused[:8]) + (" …" if len(refused) > 8 else ""))
    if failed:
        log(f"⚠️  fetch failed for {len(failed)} (old row kept): "
            + ", ".join(failed[:8]) + (" …" if len(failed) > 8 else ""))
    return len(out)


def _short(why, n=46):
    why = str(why or "").replace("no honest value — ", "")
    return why[:n] + ("…" if len(why) > n else "")


def age_days(row) -> int | None:
    from datetime import datetime
    try:
        return (date.today() - datetime.strptime(
            str(row["Fetched"])[:10], "%Y-%m-%d").date()).days
    except Exception:                                    # noqa: BLE001
        return None


def for_sheet(ticker, price=None, cache=None) -> tuple[str, str]:
    """(cell_text, band) for the Dashboard RECOMMENDED row.

    The VALUE is cached weekly; the BAND is recomputed here against whatever
    price is passed in, so a stock that falls 20% during the week shows as
    cheaper the same day rather than waiting for Monday.
    """
    from core.oracle import BANDS
    cache = load(log=lambda *_: None) if cache is None else cache
    r = cache.get(str(ticker).upper())
    if not r:
        return "", ""
    a = age_days(r)
    if a is None or a > MAX_AGE_DAYS:
        return "", "STALE"
    try:
        val = float(r["Oracle_Value"])
    except (TypeError, ValueError):
        return "", ""
    # THE POINT VALUE IS THE HEADLINE, the range is context beside it.
    # Printing "$203–$536" in the OracleValue cell made the estimate look
    # like it said nothing — Chakravarti, seeing exactly that: "oracle
    # values are too wide man". The spread is real and still shown, but one
    # number is what a reader compares to the price.
    text = f"${val:,.2f}"
    band = r.get("Oracle_Band", "")
    try:
        if price and float(price) > 0 and val > 0:
            band = next(b for lim, b in BANDS if float(price) / val <= lim)
    except Exception:                                    # noqa: BLE001
        pass
    return text, band


def panel_for_sheet(ticker, price=None, cache=None) -> list[tuple[str, list]]:
    """The ORACLE section for one Dashboard block, as (label, cells) pairs.

    Three groups, each a name row and a value row, matching how INDICATORS
    reads. Returns [] when there is nothing cached — an empty section is
    better than a section of dashes.

    The split is deliberate. VALUE is the answer, QUALITY is the OracleIQ
    panel saying how the company scores, and BASIS is how much to trust the
    answer: a mean P/S over 18 years that the market agreed on is a
    different object from one over 4 years that scattered 50%, and the
    number alone cannot tell you which you are looking at.
    """
    cache = load(log=lambda *_: None) if cache is None else cache
    r = cache.get(str(ticker).upper())
    if not r:
        return []
    text, band = for_sheet(ticker, price, cache)
    if not text:
        return []

    gap = ""
    try:
        val = float(r["Oracle_Value"])
        if price and float(price) > 0 and val > 0:
            gap = f"{(val / float(price) - 1) * 100:+.1f}%"
    except (TypeError, ValueError, KeyError):
        pass

    rng = ""
    try:
        lo, hi = float(r["Oracle_Low"]), float(r["Oracle_High"])
        rng = f"${lo:,.0f}–${hi:,.0f}"
    except (TypeError, ValueError, KeyError):
        pass

    out = [("VALUE",
            ["OracleValue", "Band", "Upside_to_Fair", "Range", "Moat", "As_Of"],
            [text, band, gap, rng, r.get("Moat", ""),
             str(r.get("Fetched", ""))[:10]])]

    iq = str(r.get("Oracle_IQ", ""))
    if iq:
        names, vals = [], []
        for part in iq.split(" · "):
            k, _, v = part.partition(" ")
            names.append(k)
            vals.append(v)
        out.append(("QUALITY", names, vals))

    yrs, spread = r.get("PS_Years", ""), r.get("PS_Spread", "")
    # The caveat is spelled out rather than left to the reader. A thin or
    # scattered history is exactly when this signal should be discounted,
    # and that is not obvious from a dollar figure.
    # Thin history is the caveat that earned its place: across the four
    # published anchors, error falls monotonically with years (7y -> 54%,
    # 16y -> 6%). Wide spread is shown as a number but NOT called a warning —
    # it fired on two thirds of the book and does not track error at all.
    caveat = ""
    try:
        if yrs and int(yrs) < 8:
            caveat = f"thin history — {int(yrs)}y only, treat as weak"
        elif spread and float(spread) > 0.90:
            caveat = "multiple has ranged widely"
    except (TypeError, ValueError):
        pass
    out.append(("BASIS",
                ["Mean_P/S", "Years", "P/S_Spread", "Next_Earnings", "Caveat"],
                [f"{r['PS_Mean']}x" if r.get("PS_Mean") else "",
                 yrs,
                 f"{float(spread):.0%}" if spread not in ("", None) else "",
                 str(r.get("Next_Earnings", ""))[:10],
                 caveat]))
    return out


FIN_GROUPS = [
    ("VALUATION", [("PE", "PE"), ("Fwd_PE", "Fwd_PE"), ("PEG", "PEG"),
                   ("EPS", "EPS"), ("Mkt_Cap_M", "Mkt_Cap_M")]),
    # TTM AND THE FIVE-YEAR AVERAGE SIDE BY SIDE. The gap between them is
    # the whole cyclical story: MU's trailing net margin is 63.8% and its
    # five-year mean is a fraction of that. Printing only the trailing
    # figure is how a peak gets mistaken for a level.
    ("QUALITY", [("Net_Margin_%", "Net_Margin"),
                 ("Margin_5y_%", "Net_Margin_5y"),
                 ("ROE_%", "ROE"), ("ROE_5y_%", "ROE_5y"),
                 ("Rev_Growth_%", "Rev_Growth"),
                 ("Debt/Equity", "Debt_Equity"), ("Div_Yield_%", "Div_Yield")]),
    ("TREND", [("Margin_Shape", "Margin_Shape"),
               ("Margin_By_Year_%", "Margin_Series")]),
]


def margin_trend(sym: str, years: int = 4) -> tuple[str, str]:
    """(shape, series) for net margin over the last few fiscal years.

    ADDS INFORMATION, DOES NOT PICK A SIDE — which is why it stays out of
    the verdict. Chakravarti, 2026-11-09: *"i don't agree that MU is cyclic
    anymore ... now with AI all compute is reserved in future years."* He
    may well be right, and the five-year average cannot tell him either
    way: by construction it lags a genuine regime change by years, which is
    the price of not being fooled by a peak.

    What IS measurable is the SHAPE. A business stepping up reads nothing
    like one oscillating, and the series says which without anyone having
    to assert a thesis:

        MU   -37 -> 3 -> 23 -> 64    RISING 4y
        a true cycle would alternate instead

    Four consecutive years of improvement is not proof the cycle is dead.
    It is evidence, shown plainly, next to the five-year mean that argues
    the other way. He reads both and decides.
    """
    try:
        from core import edgar
        rev = edgar.annual_revenue(sym)
        ni = edgar.annual_series(sym, edgar.NET_INCOME_TAGS)
        yrs = [y for y in sorted(set(rev) & set(ni)) if rev.get(y)][-years:]
        m = [(y, ni[y] / rev[y]) for y in yrs]
        if len(m) < 3:
            return "", ""
        vals = [v for _, v in m]
        series = " → ".join(f"{v * 100:.0f}" for v in vals)
        # A MOVE SMALLER THAN THE DISPLAY IS NOT A MOVE. Comparing raw
        # floats made NVDA's "56 → 56" a turn, because underneath it is
        # 56.4 → 55.9 — and a shape that contradicts the series printed
        # beside it is worse than no shape at all. One percentage point is
        # the resolution the reader actually sees.
        TOL = 0.01
        ups = sum(1 for a, b in zip(vals, vals[1:]) if b - a > TOL)
        downs = sum(1 for a, b in zip(vals, vals[1:]) if a - b > TOL)
        spread = max(vals) - min(vals)
        # A FLAT YEAR MUST NOT BREAK MONOTONICITY. Requiring every step to
        # move strictly called NVDA (16 -> 49 -> 56 -> 56) cyclical because
        # of one tie at the end, and TSLA (15 -> 15 -> 7 -> 4) cyclical
        # because of one at the start — when one is plainly rising and the
        # other plainly falling. What matters is whether the series ever
        # turns, not whether every step moves.
        if ups and not downs:
            shape = f"RISING {len(vals)}y"
        elif downs and not ups:
            shape = f"FALLING {len(vals)}y"
        elif spread < 0.05:
            shape = "STEADY"
        else:
            shape = "CYCLICAL"
        return shape, series
    except Exception:                                    # noqa: BLE001
        return "", ""


def verdict(ticker, cache=None) -> tuple[str, str]:
    """('GOOD'|'BAD'|'NEUTRAL', why) for the FINANCIALS block as a whole.

    His ask, 2026-11-09: green when the financials look really good, red
    when bad, blue when neutral. A single colour is a strong claim, so the
    rule is deliberately plain and the reasons travel with it — a verdict
    you cannot interrogate is just a mood.

    Scored on durable things, not on the multiple: a cheap P/E is not a
    good business, and the cyclical trap is exactly a low P/E on a peak.
    """
    cache = load(log=lambda *_: None) if cache is None else cache
    r = (cache or {}).get(str(ticker).upper())
    if not r:
        return "", ""

    def f(k):
        try:
            return float(str(r.get(k, "")).strip())
        except (TypeError, ValueError):
            return None

    good, bad = [], []
    # Prefer the normalised margin; fall back to trailing when absent.
    m = f("Net_Margin_5y")
    m = m if m is not None else f("Net_Margin")
    if m is not None:
        (good if m >= 12 else bad if m < 0 else []).append(
            f"margin {m:.0f}%")
    roe = f("ROE_5y")
    roe = roe if roe is not None else f("ROE")
    if roe is not None:
        (good if roe >= 15 else bad if roe < 0 else []).append(
            f"ROE {roe:.0f}%")
    g = f("Rev_Growth")
    if g is not None:
        (good if g >= 10 else bad if g < 0 else []).append(
            f"growth {g:+.0f}%")
    de = f("Debt_Equity")
    if de is not None:
        (good if de < 80 else bad if de > 200 else []).append(
            f"D/E {de:.0f}")
    peg = f("PEG")
    if peg is not None and 0 < peg < 1:
        good.append(f"PEG {peg:.2f}")
    fpe, pe = f("Fwd_PE"), f("PE")
    if fpe and pe and fpe < pe:
        good.append("earnings rising")

    if not good and not bad:
        return "", ""
    if len(bad) >= 2 or (bad and not good):
        return "BAD", " · ".join(bad)
    if len(good) >= 3 and not bad:
        return "GOOD", " · ".join(good)
    return "NEUTRAL", " · ".join(good + [f"but {b}" for b in bad])


def financial_rows(ticker, cache=None) -> list[tuple[str, list, list]]:
    """The FINANCIALS section for one block, as (label, names, values).

    Read from the weekly cache, not from the network. The sheet rebuilds
    every half hour and there are 135 tickers; fetching .info per block
    would add thousands of calls a day to learn figures that change once a
    quarter. The cost is that these are as of `Fetched`, which the ORACLE
    section directly above already states.

    A field with no value is DROPPED, not shown blank — an ETF has no PEG
    and a loss-maker has no meaningful P/E, and a rail of empties reads as
    broken rather than as inapplicable.
    """
    cache = load(log=lambda *_: None) if cache is None else cache
    r = cache.get(str(ticker).upper())
    if not r:
        return []
    out = []
    for label, fields in FIN_GROUPS:
        pairs = [(nice, r.get(col)) for nice, col in fields
                 if str(r.get(col, "")).strip() not in ("", "nan", "None")]
        if pairs:
            out.append((label, [n for n, _ in pairs], [v for _, v in pairs]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Weekly conservative fair value")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--weekly", action="store_true",
                    help="refresh only if due (Monday, or cache stale). "
                         "Safe to run daily from cron.")
    ap.add_argument("--one", metavar="TICKER")
    a = ap.parse_args()

    if a.one:
        o = compute_one(a.one.upper())
        if not o or o.value is None:
            print(f"{a.one}: {o.why if o else 'no data'}")
            return 1
        print(f"{a.one.upper()}  OracleValue {o.as_text()}  ({o.band})")
        print(f"   point ${o.value:,.2f} · moat {o.moat} · gap {o.gap_pct:+.1f}%")
        print(f"   {o.why}")
        return 0

    if a.weekly:
        # Monday -> all; any other day -> only names that have reported
        # since we last valued them. Usually an empty list and a no-op.
        due = due_tickers()
        if not due:
            return 0
        print(f"computing OracleValue for {len(due)} → {FILE}")
        return 0 if refresh(due) else 1

    if a.refresh:
        print(f"computing OracleValue → {FILE}")
        return 0 if refresh() else 1

    cache = load()
    if not cache:
        print(f"nothing cached at {FILE} — run with --refresh")
        return 1
    ages = [x for x in (age_days(r) for r in cache.values()) if x is not None]
    print(f"{len(cache)} cached, {min(ages)}-{max(ages)} days old\n")
    print(f"{'':7}{'value':>12}{'band':>13}  moat")
    for t, r in sorted(cache.items()):
        print(f"{t:<7}{r['Oracle_Value']:>12}{r['Oracle_Band']:>13}  {r['Moat']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
