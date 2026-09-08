# Historical P&L — per-ticker, all Schwab accounts

Answers one question: **for every stock I have ever traded, what did I actually make or lose?**

Realized P&L (FIFO), unrealized P&L, dividends and fees, per ticker, aggregated
across **all** linked Schwab accounts — and broken out per account.

> **This runs on Pi 1 only.** Pi 1 owns `tokens.json`. Pi 2 is for editing code.
> Everything below assumes an SSH session on Pi 1.

---

## 1. State of play

| | |
|---|---|
| Pipeline | **Complete and working** — 5 modules, committed 2026-07-13 (`627f0f5`) |
| Has it run? | **Yes, at least once, on Pi 1.** `corporate_actions.json` is seeded with mergers found in the real history |
| Cache on Pi 2 | None — and there never will be. Wrong box. |
| Wired to cron? | **No.** Standalone, run by hand |
| Left half-done | `manual_basis.csv` is still empty, and 7 CUSIPs in `corporate_actions.json` are unconfirmed. See §7 |

---

## 2. How it works

```
schwab_client.py     credentials, shared tokens.json, cache/report paths
      │
      ▼
txn_cache.py         fetch transactions → cache to disk   ← the incremental part
      │
      ▼
txn_parser.py        raw Schwab JSON → flat ledger, one row per security leg
      │
      ▼
pl_engine.py         FIFO lot tracking → realized P&L, open lots, anomalies
      │
      ▼
build_pl_report.py   + live prices → CSVs and XLSX          ← THE ENTRY POINT
```

### 2.1 Download once, then only the delta

This is the behaviour you asked for, and it is already implemented.

**Cold start** (no cache): pulls everything from `SCHWAB_START_DATE`, default
**2019-01-01**, to today, for every linked account. Walks the range in 365-day
chunks (the Schwab maximum), halving the window on timeout rather than giving
up, with exponential backoff on errors.

**The watermark**: after each account syncs, `<account_hash>.state.json` records

```json
{ "last_date": "2026-09-05", "count": 4812 }
```

**Incremental runs** resume at **`last_date − 10 days`**, not `last_date + 1`.
That overlap is deliberate: Schwab back-dates and corrects already-settled
activity, so resuming exactly where you stopped silently loses those
corrections. Re-fetched rows are de-duplicated on `activityId`, **keeping the
last copy**, so a correction overwrites the original.

Cache layout, one pair per account, in `schwabAPI/data/transactions/`:

```
<hash>.csv          full raw dump, one row per transaction (raw_json preserved)
<hash>.state.json   {"last_date": "YYYY-MM-DD", "count": N}
```

Both are gitignored (`schwabAPI/.gitignore:49`). Transaction history never
enters the repo.

### 2.2 P&L method

**Household-level FIFO.** All accounts collapse into one ledger per symbol, so
shares journalled between your own Schwab accounts cancel out instead of looking
like a sale and a repurchase. Every transaction row still records which account
it happened in.

Handled explicitly:

- **Long and short** positions (a sold-to-open covered call has `qty < 0`)
- **Option expiry** — closes the lot at zero, realizing the full premium
- **Splits** — share count changes, **total basis preserved**
- **Reverse splits and re-registrations** — Schwab books a re-registration as
  `-50 / +50 / +50 / -50` across four transactions on one day. Applied one at a
  time these momentarily drive the position to zero and destroy the cost basis,
  so they are **netted per day** before being applied
- **Mergers / ticker changes** — via `symbol_map` in `corporate_actions.json`
- **ACATS transfers in** — Schwab reports cost `0`. Uses `manual_basis.csv` if
  you supply a price, otherwise flags the ticker `UNKNOWN_BASIS`

```
Total P&L = realized + unrealized + dividends
Return %  = total P&L / open cost basis × 100
```

---

## 3. Files

**Current — these are the pipeline:**

| File | Role |
|---|---|
| `build_pl_report.py` | **Entry point.** Run this. |
| `schwab_client.py` | Credentials, paths, `COLD_START`, `OVERLAP_DAYS` |
| `txn_cache.py` | Fetch + cache + watermark + de-dupe |
| `txn_parser.py` | Raw JSON → ledger |
| `pl_engine.py` | FIFO lots, realized P&L, anomalies |
| `manage_cache.py` | Cache inspection and surgery |
| `corporate_actions.json` | Ticker remaps (merger/rename) |
| `manual_basis.csv` | Your cost basis for ACATS transfer-ins |
| `portfolio_snapshot.py` | Positions snapshot (separate tool) |

**Superseded — do not use, kept for reference:**
`analytics_core.py`, `analytics_core_enhanced.py`, `historical_pl_report.py`,
`OLD-historical_pl_report_fixed.py`, `account_pl.py`,
`2-3-26-analytics_core copy.py`, `1-2-26-portfolio_snapshot copy.py`.

> An earlier version of this README told you to run
> `historical_pl_report_enhanced.py`. **That file does not exist.** It was
> superseded by `build_pl_report.py`. If you are reading an old copy, stop.

---

## 4. Running it on Pi 1

Every command below must be run **from inside `schwabAPI/`** — the modules use
plain imports (`import txn_cache`), and `manage_cache.py` looks for the cache at
the *relative* path `data/transactions`. Run it from anywhere else and it will
tell you the cache does not exist even when it is perfectly healthy.

```bash
cd ~/github/myTrading/schwabAPI
PY=~/github/myTrading/.venv/bin/python     # fall back to python3 if absent
```

### Prerequisites

1. `~/github/myTrading/.env` has `app_key`, `app_secret`, `callback_url`
2. `~/github/myTrading/tokens.json` exists and is **fresh** — Schwab refresh
   tokens die after 7 days. Re-auth from your phone with the `auth_url` /
   `auth_code` verbs (see `Documentation/remoteOpsGuide-9-7-26.md`)
3. There must be exactly **one** `tokens.json`, in the repo root.
   `schwabAPI/` must never grow its own — the code warns loudly if it does

### Check the cache before anything else

```bash
$PY manage_cache.py list
```

Shows each account's row count, file size, and `last_date`. This is your
"is the cache good?" check. What you want to see: one `.csv` **and** one
`.state.json` per account, `last_date` within a few days of today, counts in the
thousands. See §6 for reading the output.

### First run / full rebuild

```bash
$PY build_pl_report.py --out ~/pl_out
```

Cold start pulls ~7 years across every account. Expect it to take a while and
log every chunk. It is resumable — if it dies, run it again and it picks up from
the watermark.

### Routine run (the normal case)

```bash
$PY build_pl_report.py --out ~/pl_out
```

Same command. It fetches only `last_date − 10 days` → today per account.

### Useful flags

| Flag | Effect |
|---|---|
| `--offline` | Rebuild reports from cache only. **No Schwab call** — works with a dead token |
| `--no-prices` | Skip live quotes; unrealized P&L left blank rather than faked as zero |
| `--out DIR` | Output directory. **Use it** — the default is `~/github/jobMyTrading/outputs/portfolio` |

`--offline` is the one to reach for when you just changed `manual_basis.csv` or
`corporate_actions.json` and want to see the effect. No API, no rate limits,
instant.

---

## 5. Cache management

```bash
$PY manage_cache.py list                              # status of every account
$PY manage_cache.py clear <account_hash>              # force one account to cold start
$PY manage_cache.py clear-all                         # nuke everything (prompts)
$PY manage_cache.py reset-date <hash> 2026-01-01      # re-pull from a date
```

`reset-date` is the surgical tool: it rewinds one account's watermark so the
next run re-fetches from that date and merges, without discarding what you have.
Prefer it over `clear-all`, which throws away a seven-year download.

---

## 6. Reading the output

Four CSVs plus `portfolio_pl.xlsx` (same content as tabs) land in `--out`.

### `ticker_pl_summary.csv` — the answer to "how am I doing"

One row per ticker **ever traded**, open and closed. Sorted by total P&L.

| Column | Meaning |
|---|---|
| `status` | `OPEN` or `CLOSED` |
| `accounts` | Which accounts held it |
| `open_qty` / `schwab_qty` | Our FIFO share count vs what Schwab says you hold |
| `qty_mismatch` | Difference. **Should be 0.** Anything else means the engine lost track |
| `avg_cost`, `open_cost_basis` | Cost basis of what is still open |
| `realized_pl` | Locked in from closed lots |
| `unrealized_pl` | Market value − open cost basis |
| `dividends`, `fees` | Cash in, cash out |
| `total_pl`, `total_pl_pct` | realized + unrealized + dividends |
| `basis_flag` | `OK`, `UNKNOWN_BASIS`, or `DIVIDEND_ONLY` |

The run also prints the top 15 and household totals to the terminal.

### `anomalies.csv` — **read this one**

Everything the engine could not explain. An empty file means a clean run. Issues
it raises: `transfer_in_no_basis`, `shares_from_nothing`, `bad_split_adjustment`,
`unbalanced_internal_journal`, `transfer_out`.

### `ticker_txns.csv`, `open_lots.csv`

Every transaction with running position, cost basis and per-row realized P&L —
this is where you go to audit a number you distrust. And the FIFO lots still
open, with per-lot cost and whether the basis is known.

---

## 7. The triage loop — where this was left

The engine is honest about what it cannot know, and two files are how you teach
it. **This is the unfinished work.**

### `manual_basis.csv` — currently empty

Shares transferred in from another broker arrive with cost `0`, so their P&L is
overstated by the entire purchase price. The run prints the affected tickers:

```
!! N tickers have NO cost basis from Schwab (transferred in from another broker).
   Their P&L is understated. Fill in manual_basis.csv: ABC, XYZ
```

Look each up in the **old broker's** statement, then add:

```csv
symbol,cost_per_share
ABC,142.50
```

Re-run with `--offline` to see it applied.

### `corporate_actions.json` — partly done

`symbol_map` already carries `STPK→STEM`, `CYFRF→STKE`, `BRPHD→GLXY`, found in
your real history. Still open: **7 unconfirmed CUSIPs** in
`unmapped_cusips_to_confirm`, one flagged as *likely* LRCX. Confirm before
mapping — a wrong map silently merges two tickers' P&L.

### The loop

```
run → read anomalies.csv → add a mapping or a basis → run --offline → repeat
```

Stop when `anomalies.csv` is empty and every `qty_mismatch` is `0`.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `manage_cache.py` says cache doesn't exist | You are not in `schwabAPI/`. It uses a relative path. `cd` there |
| `No token file at .../tokens.json` | Re-auth on Pi 1. Never create a second token file in `schwabAPI/` |
| `Missing app_key... in .env` | `.env` is read from the repo root, with a local fallback |
| `WARNING: stray token file` | A second `tokens.json` appeared in `schwabAPI/`. Delete it — two copies drift and the 7-day re-auth breaks one |
| HTTP 429 | Wait 15–30 min. The cache means you resume, not restart |
| `No transactions. Nothing to do.` | Cold start found nothing — check `SCHWAB_START_DATE` and that accounts are linked |
| Timeouts on cold start | Already handled: the chunk halves down to 15 days before failing |
| `qty_mismatch` non-zero | A split or corporate action was misread. Check `anomalies.csv` and `ticker_txns.csv` for that symbol |
| P&L looks too good | Almost always `UNKNOWN_BASIS` — a transfer-in with cost 0. See §7 |
| Wrote to a directory you didn't expect | You omitted `--out`; default is `~/github/jobMyTrading/outputs/portfolio` |

---

## 9. Configuration

| Setting | Where | Default |
|---|---|---|
| `SCHWAB_START_DATE` | env | `2019-01-01` |
| `OVERLAP_DAYS` | `schwab_client.py:45` | `10` |
| `CACHE_DIR` | `schwab_client.py:28` | `schwabAPI/data/transactions` |
| `PORTFOLIO_OUT_DIR` | env | `~/github/jobMyTrading/outputs/portfolio` |
| `TOKENS_FILE` | `schwab_client.py:25` | `~/github/myTrading/tokens.json` |

This script never calls git. `gitpush.py` remains the only git writer.

---

## 10. Not handled

- **Options P&L is partial** — expiry is handled, assignment and exercise are not
- **No tax lots** — no short- vs long-term split, no wash sales. Not a tax tool
- **Verify against Schwab's own statements** before trusting any number
