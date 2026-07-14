# Schwab Positional P&L — schwabAPI/

Household-level FIFO P&L across every Schwab account, for every stock you have
ever traded (open **and** closed). Built from the raw transaction history, not
from the positions API.

## Files

| File | Role |
|---|---|
| `schwab_client.py` | The ONE Schwab connection. Points at `../tokens.json`. |
| `txn_cache.py` | Schwab → `data/transactions/<hash>.csv`. Chunked, de-duped, incremental. |
| `txn_parser.py` | raw JSON → normalized ledger (one row per security leg). |
| `pl_engine.py` | FIFO lots → realized / unrealized / dividends per ticker. |
| `build_pl_report.py` | Entry point. Writes the report. |
| `corporate_actions.json` | Ticker rename / merger map. |
| `manual_basis.csv` | Cost basis for ACATS transfer-ins (Schwab has none). |

## Token — one file, in the parent folder

`schwab_client.py` uses **`~/github/myTrading/tokens.json`** — the same file
`app.py` already uses via `data/schwab/schwab_helper.py`. `.env` is read from
`~/github/myTrading/.env` (with a local fallback).

The old `portfolio_snapshot.make_client()` called
`schwabdev.Client(app_key, app_secret, callback_url)` with no `tokens_file`,
so schwabdev defaulted to `tokens.json` **in the current working directory** —
i.e. it silently created a *second* token file inside `schwabAPI/`. With a
~7-day re-auth cycle, whichever copy you don't refresh dies. That's now fixed;
`stray_token_check()` warns if a second one ever reappears.

**On the Pi, delete `schwabAPI/tokens.json` if it exists.**

## Cold start on the Pi

```bash
cd ~/github/myTrading/schwabAPI
rm -f tokens.json                      # never a second token file
rm -rf data/transactions               # start clean — see "WSL cache" below
python build_pl_report.py              # pulls full history, writes the report
```

Full history is pulled in **1-year chunks** from `SCHWAB_START_DATE`
(default `2019-01-01`), shrinking the window automatically on timeout.
Roughly 7 chunks × 6 accounts on the first run.

Subsequent runs are incremental: they re-fetch the last **10 days** (Schwab
back-dates corrections) and merge on `activityId`.

Output lands in `~/github/jobMyTrading/outputs/portfolio/`:

| File | Contents |
|---|---|
| `ticker_pl_summary.csv` | One row per ticker ever traded. Open + closed. |
| `ticker_txns.csv` | Every transaction: date, **account**, action, qty, price, value, realized P&L, running position, running basis. |
| `open_lots.csv` | The FIFO lots still open. |
| `anomalies.csv` | **Read this.** Anything the engine could not explain. |
| `portfolio_pl.xlsx` | All four as tabs. |

`build_pl_report.py` does **not** git push — `gitpush.py` stays the sole git writer.

## Three bugs in the old cache (fixed)

1. **`transactionId` doesn't exist.** Schwab returns `activityId`. The old
   normalizer wrote `t.get("transactionId")`, so every id was `NaN` — and
   `drop_duplicates(subset=["transactionId"])` referenced a column that wasn't
   even there. De-dupe never ran.
2. **`amount` doesn't exist.** Schwab returns `netAmount`. That column was `NaN` too.
3. **Type-scoped cache, type-agnostic state file.** `build_deposit_withdrawal_df`
   fetched only cash types, then wrote `last_date` to `state.json`. Any later
   run wanting TRADE rows saw the range as "already cached" and skipped it —
   permanently. The cache now always pulls **every** type and filters downstream.

Because of (3), **the WSL cache is not trustworthy.** Cold start clean on the Pi.

## Corporate actions

Netted per symbol per day, then applied basis-preserving:

- **Forward split** — `NVDA +21`, `TSLA +14`, `ARKB +2582`: share count up, total basis unchanged.
- **Reverse split** — `CALA -900 / +45`, `ETHU -180 / +9`: share count down, basis unchanged.
- **Rename / merger** — `STPK→STEM`, `CYFRF→STKE`, `BRPHD→GLXY`: remapped via
  `corporate_actions.json`, basis carries across.
- **Re-registration** — `NPPTF -263/+263`, `GLXY -50/+50/+50/-50`: nets to zero, no-op.
  (Applied one at a time these would zero the position and destroy the basis;
  daily netting is what prevents that.)
- **Internal account journals** — free at household level, they cancel out.
- **Option expiry** — closes the lot at 0, realizing the full premium.
- **Sold-to-open calls** — `positionEffect` is `OPENING` with `qty < 0`, so the
  engine keys off the **sign of qty**, and carries short lots properly.

Anything it can't explain → `anomalies.csv`. Add the mapping and re-run.

## The one thing code cannot fix: transferred-in cost basis

25 tickers came into Schwab via *"Transfer of Security or Option In"* (ACATS
from your old broker) with **`cost: 0.0`**. Schwab genuinely does not have your
basis for them, which is why the old `per_ticker_pl.csv` showed `ASML` and
`GOOG` at `cost_basis = 0`.

Affected: `TSLA, COIN, GOOGL, TTD, CRSP, ASML, META, DDOG, MRVL, ARKK, ARKG, MU,
VYGVQ, CALA, IZRL, MDB, MGNI, SNOW, SHOP, TWLO, SNAP, ROKU, FUBO, 512807108, 25058X105`

Their P&L is **overstated** until you fill in `manual_basis.csv` from your old
broker's statements:

```csv
symbol,cost_per_share
SNAP,52.10
MRVL,44.30
```

Anything you can't find, leave out — it'll stay flagged `UNKNOWN_BASIS` in the
summary so you always know which numbers to distrust.

## `"System transfer"` — read before you validate

17 transactions are typed `TRADE`, described `"System transfer"`, with
`netAmount = 0` and a **positive** `cost` (the carried basis, not a cash flow).
These moved positions *into* accounts `67024171` (+2,337 sh) and `15238922`
(+45 sh) during an account consolidation. **Schwab does not report the outgoing
leg.** The engine treats them as transfers-in with known basis.

On the WSL cache this leaves share counts that disagree with what Schwab says
you hold (e.g. FIFO `TSLA = 275` vs positions `TSLA = 218`). That's why
`ticker_pl_summary.csv` carries a **`qty_mismatch`** column: `open_qty` (our
FIFO) minus `schwab_qty` (live positions API). **After the clean Pi cold start,
`qty_mismatch` should be 0 for every ticker.** Any ticker where it isn't has a
history gap, and its P&L is wrong. That column is your validation gate — check
it before you trust a single number.

## Next (deferred, as agreed)

- cron: daily `build_pl_report.py`, pre-open and post-close
- Streamlit tab reading `outputs/portfolio/`