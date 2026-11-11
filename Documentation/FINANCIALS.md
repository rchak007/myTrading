# FINANCIALS — what we retrieve, and how to read it

The FINANCIALS section sits under ORACLE in every Dashboard block. ORACLE is
a **model's opinion**; FINANCIALS is **what the company reported**. Keeping
them apart matters, because when one of them looks wrong you need to know
whether you are arguing with a company or with us.

---

## 1. What is retrieved, and when

Everything comes from `yfinance`'s `.info` and from SEC filings, and is
cached in `oracle_values.csv` by `oracle_value.py`.

**It refreshes weekly on Monday, plus the day after a ticker reports.** Not
every cycle. The sheet rebuilds every half hour against 139 tickers; fetching
fundamentals each time would add thousands of API calls a day to learn
figures that change once a quarter. The `As_Of` date in the ORACLE row
directly above tells you how old they are.

### VALUATION row

| Field | What it is | Source |
|---|---|---|
| `PE` | Price ÷ trailing 12-month earnings | `.info trailingPE` |
| `Fwd_PE` | Price ÷ next year's estimated earnings | `.info forwardPE` |
| `PEG` | P/E ÷ growth rate | `.info trailingPegRatio` |
| `EPS` | Trailing 12-month earnings per share | `.info trailingEps` |
| `Mkt_Cap_M` | Market capitalisation, $ millions | `.info marketCap` |

### QUALITY row

| Field | What it is | Source |
|---|---|---|
| `Net_Margin_%` | Net income ÷ revenue, trailing | **derived** — see §4 |
| `Margin_5y_%` | The same, averaged over 5 fiscal years | income statement |
| `ROE_%` | Return on equity, trailing | `.info returnOnEquity` |
| `ROE_5y_%` | The same, over 5 fiscal years | income + balance sheet |
| `Rev_Growth_%` | Revenue growth, latest year on year | `.info revenueGrowth` |
| `Debt/Equity` | Total debt ÷ equity, as a percentage | `.info debtToEquity` |
| `Div_Yield_%` | Dividend yield | `.info dividendYield` |

### TREND row

| Field | What it is | Source |
|---|---|---|
| `Margin_Shape` | RISING / FALLING / CYCLICAL / STEADY | computed from SEC filings |
| `Margin_By_Year_%` | The actual margins, oldest to newest | SEC filings |

**A blank is never a zero.** An ETF has no PEG, a loss-making company has no
meaningful P/E, and a margin on near-zero revenue is noise. Those fields are
left empty rather than filled with `0.00`, which would read as "breakeven"
instead of "unknown".

---

## 2. The one idea that matters most: trailing vs normalised

Three rows carry the same figure twice — trailing and five-year. **The gap
between them is usually the most informative thing in the section.**

```
MU    Net_Margin_%  63.8     Margin_5y_%  13.1
      ROE_%         88.3     ROE_5y_%     16.4
```

A 63.8% net margin for a memory manufacturer is a cycle peak, not a level.
Valuing the trailing figure is how a peak gets mistaken for normal — and the
cheap-looking P/E that comes with it is the trap, not the opportunity.

Compare:

```
MSFT  Net_Margin_%  41.2     Margin_5y_%  36.6     gap of 5 points
MU    Net_Margin_%  63.8     Margin_5y_%  13.1     gap of 51 points
```

Microsoft earns roughly what it has always earned. Micron does not. **Both
may be good investments; only one of them has a margin you can extrapolate.**

The valuation model and the FINANCIALS verdict both use the **five-year**
figures for exactly this reason.

---

## 3. How the verdict is scored

The FINANCIALS label is coloured green, red or blue, with its reasons printed
beside it. The rule is deliberately plain, because a verdict you cannot
interrogate is just a mood.

| Test | Good | Bad |
|---|---|---|
| Net margin (5-year preferred) | ≥ 12% | < 0% |
| Return on equity (5-year preferred) | ≥ 15% | < 0% |
| Revenue growth | ≥ 10% | < 0% |
| Debt / equity | < 80 | > 200 |
| PEG | between 0 and 1 | — |
| Forward P/E below trailing P/E | yes | — |

- **GREEN (GOOD)** — three or more positives and nothing negative
- **RED (BAD)** — two negatives, or negatives with nothing positive
- **BLUE (NEUTRAL)** — anything mixed; the reasons end with `but …`

**It is scored on durable things, never on how cheap the multiple is.** A low
P/E is the cyclical trap, not evidence of quality, so it earns no points.
That is why MU scores green on growth, debt and PEG rather than on its 63.8%
peak margin.

**PEG below 1 is highlighted orange** in the VALUATION row — the single most
quoted "growth is cheap" marker, and the one most worth checking rather than
trusting.

### Worked examples from the book

```
MSFT  GOOD     margin 37% · ROE 32% · growth +19% · D/E 28 · earnings rising
NVDA  GOOD     margin 44% · ROE 64% · growth +106% · D/E 17 · PEG 0.22
TSLA  NEUTRAL  ROE 17% · D/E 18 · earnings rising · but growth -0%
AAOI  BAD      margin -35% · ROE -37%
SNDK  NEUTRAL  ROE 16% · growth +378% · D/E 1 · but margin -3%
```

TSLA is the instructive one: it has a defensible balance sheet and decent
normalised returns, and revenue stopped growing. One negative against three
positives is NEUTRAL, not BAD — and the `but` tells you which to watch.

---

## 4. What we do NOT trust, and why

**`profitMargins` is wrong often enough to be derived instead.** yfinance
returns a flat `0.0` for all four ROBOTICS names — ARBE lost $41M on $1.9M of
revenue and the field says zero. A `0.0` on a company bleeding tens of
millions reads as *breakeven*. Net margin is therefore computed from net
income ÷ revenue whenever the reported field is empty or is zero against a
material loss.

**Margins beyond ±1000% are suppressed.** Lightwave Logic loses $24M on $0.3M
of sales; the arithmetic gives −7,900% and the limit gives `-inf`. Seven
tickers carried a literal `-inf` before this bound existed, and non-finite
floats are not merely ugly — the Sheets API rejects them and can blank an
entire write.

**ROE survives where margin does not**, because equity is a sane denominator
where near-zero revenue is not. That is why ARBE shows a blank margin and a
real −83% ROE.

**A single year-over-year growth figure is weak.** It can be an easy
comparison, an acquisition, or a recovery from a collapse. Prefer the
compounded figure in the ORACLE section.

---

## 5. The traps, in the order they bite

1. **The cyclical P/E trap.** Cyclicals look cheapest at the TOP of the
   cycle, because the E in P/E is at a peak that will not hold. Always read
   `Margin_5y_%` beside `Net_Margin_%`.
2. **A blank P/E means it lost money.** Not an error. Value it on sales —
   which is exactly why OracleValue leans on price-to-sales.
3. **High margin, low ROE** → leverage is flattering the return. Check
   `Debt/Equity`.
4. **Low margin, high ROE** → a thin-margin business turning capital fast.
   Normal for retail and distribution; judge against its own history.
5. **PEG under 1** → growth is covering the multiple. Breaks down completely
   when growth is erratic or negative, which is most of the time on a
   cyclical.
6. **Growth that is not growth.** One huge year is usually a comparison, not
   a trend. `Margin_Shape` tells you whether the business is stepping up or
   oscillating.
7. **Debt that matters.** Leverage is fine while business is good and is what
   kills companies when it is not. Pair it with free cash flow — debt
   serviced from cash flow is a different animal from debt serviced by
   refinancing.

---

## 6. Is it still cyclical? — the honest limit

`Margin_Shape` and `Margin_By_Year_%` exist because the five-year average
**cannot answer this**, and it is worth being clear about why.

```
MU    RISING 4y    -38 → 3 → 23 → 64
AVGO  CYCLICAL      35 → 39 → 11 → 36
INTC  CYCLICAL      13 → 3 → -35 → -1
TSLA  FALLING 4y    15 → 15 → 7 → 4
```

Four straight years of improvement is not what a cycle looks like. It is also
not proof the cycle is dead — supply response, not demand, is what has always
ended memory cycles, and competitors are adding capacity now.

**By construction the five-year mean LAGS a genuine regime change by years.**
That is the price of not being fooled by a peak, and it cuts both ways. So
the shape is shown as *evidence beside the mean that argues the other way*,
and is deliberately kept **out of the verdict**. You weigh it; the model does
not adopt a thesis.

---

## 7. Where to read it

- **Dashboard** — FINANCIALS under ORACLE in each held ticker's block
- **REST-DASHBOARD** — the same for every other ticker in `app.py`
- **DASHBOARD-DOC** — generated field reference, including this colour key
- `oracle_values.csv` — the raw cache, one row per ticker

**The order worth reading a block in:** protection first (`Has_Stop` — nothing
else matters if a gap can take the position out), then RATING, then TREND and
STRUCTURE, then ORACLE *with its BASIS line before the number*, then
FINANCIALS with the traps above, and only then the RECOMMENDED levels.
