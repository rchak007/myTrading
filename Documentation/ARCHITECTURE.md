# Architecture — modules, data flow, contracts

**The technical view.** Every program that runs, what it reads and writes, and
the rules that keep them from interfering. For the conceptual view see
[HOW-IT-WORKS.md](HOW-IT-WORKS.md); for schedules see
[OPERATIONS.md](OPERATIONS.md).

---

## 1. The injection contract

**A module owns only its own logic.** Everything shared is passed in:

```python
build_cash_table(client, orders_df, mask=True, log=log)
write_orders_sheet(client_wrapper=..., signals_df=..., quotes=..., log=log)
```

Paths, `log()`, the HTML renderer and the Schwab client all come from
`jobStocksSignals.py`. A module that resolved its own paths could not be tested
without the whole system, and two modules resolving them separately would drift.

**The exceptions are deliberate:** `orders_sheet.py` and `remote_ops.py` resolve
their own spreadsheet from the environment, because the caller has no business
knowing about Sheets; `cash_reserve.py` owns its state files, because they are
money history and must live outside the repo.

---

## 2. Data flow

```
   SCHWAB API
       │  positions · orders · quotes · price history · transactions
       ▼
  jobStocksSignals.py ─────────── the hub, every :15 and :50, ~9 min
       │
       ├─→ CSV/HTML  ──→ ~/github/jobMyTrading/ ──→ gitpush.py ──→ GitHub ──→ Streamlit
       │
       ├─→ orders_sheet.write_orders_sheet()  ──→  Positions · Cash · Dashboard · header
       │
       └─→ cash_reserve.apply_fills()         ──→  ~/.local/state/myTrading/reserve_ledger.csv

  order_engine.py ─── 5,30,45 past the hour ─── reads Orders tab, writes status
       │                                          places orders after 13:15 PT
       ▼
   SCHWAB API

  remote_ops.py ───── every 10 min ──────────── reads ops sheet, runs allowlisted verbs
```

**Nothing pushes to git except `gitpush.py`.** Every job runs `--no-push` and
only writes files. So "did it reach GitHub?" is never a question about the job
that produced the file.

---

## 3. Modules

### The hub

| file | role |
|---|---|
| `jobStocksSignals.py` | orchestrates everything on the `:15`/`:50` cycle. Owns paths, `log()`, `build_html_table()`, `get_schwab_client()`. ~9-10 min per run |
| `app.py` | `STOCK_TICKERS` and the curated fund lists. Hand-maintained; `ticker_audit.py` finds holdings missing from it |
| `core/`, `data/` | indicators, macro regime, signal construction, Schwab client helper |
| `core/recommend.py` | the four advisory Rec_* price levels. **Pure** — no I/O, no clock, testable on Pi 2. See [RECOMMENDED-LEVELS.md](RECOMMENDED-LEVELS.md) |

### Sheets

| file | role |
|---|---|
| `orders_sheet.py` | builds and writes Positions, Cash, Dashboard, header. Coverage flags, fenced markers, block layout. The per-block `ORDERS` list carries both resting Schwab orders and waiting sheet intents |
| `orders_sheet_init.py` | creates the four tabs from scratch. `--force` **wipes** them |
| `orders_sheet_prices.py` | `Live_Price` and `Day_%` only, every 5 min. Sheet-only: no files, no git |
| `remote_ops.py` | the ops-sheet poller. Allowlisted verbs, top-scan row model |
| `gsheet_notes.py` | reads notes from a sheet; degrades silently if gspread is absent |
| `chitra.py` | Chitra's account. Her tab is the whole management surface — positions, Merrill orders, and HUMAN-TYPED conditions that go green when the close crosses them. Fixed row layout; each section clears only its own range, never `ws.clear()`. Also the lavender `CHITRA` row on his Dashboard, which cannot reach his totals |

### Orders

| file | role |
|---|---|
| `order_intent.py` | validates and normalises one row. `normalize()`, `fingerprint()`, `describe()` |
| `order_exec_config.py` | every limit, in code the sheet cannot reach. Kill switch, `LIVE_TRADING`, allowlists, caps |
| `order_engine.py` | evaluates triggers, runs guards, cancels blockers, places orders |
| `schwab_quotes.py` | one definition of "current price", shared by the job and the price updater |

### Money

| file | role |
|---|---|
| `cash_reserve.py` | per-(account, ticker) reserves. Append-only ledger, `fence`/`seed`/`apply_fills`, the `available_to_buy` gate |
| `sell_guard.py` | flags held positions with no protective sell |
| `schwabAPI/build_pl_report.py` | historical P&L. Transaction cache → ledger → FIFO → CSV/XLSX |
| `schwabAPI/txn_cache.py` | per-account transaction cache with watermarks and a 10-day overlap |
| `schwabAPI/pl_engine.py` | FIFO lot accounting, splits, mergers, manual adjustments |

### Auth and health

| file | role |
|---|---|
| `schwab_auth.py` | token store in `~/.schwabdev/tokens.db`. `--status`, `--url`, `--code`. Takes the job lock when installing |
| `token_watch.py` | emails before the token dies. `--from-sheet` lets Pi 2 do it. Owns `send()` — the one mailer, including inline images |
| `reminders.py` | nags until struck out. **Two channels:** `open-items` daily, `options` three times a day while the market is open. One mailer, one set of Gmail gotchas |
| `market_calendar.py` | is the market open? NYSE holidays and half-days **derived from rules**, no table to expire and no dependency. Pure |
| `health_check.py` | cron lines, output freshness, token, trading state, P&L quality |
| `smoke_test.py` | asserts every cross-module function still exists, and that the few cross-module *signatures* still take the arguments their callers pass. **Run before pushing** |
| `test_recommend.py` | the recommendation formulas, every case a real trap from the live data |
| `test_coverage.py` | the Dashboard protection flags. Both bugs here said "covered" about something that was not |
| `test_market_calendar.py` | the derived calendar against the published NYSE one, including the years the observance rules surprise you |
| `test_chitra.py` | her tab: close-not-live triggers, and that an unjudgeable row claims no protection |

### Probes

`probe_order_api.py`, `probe_quotes_api.py`, `probe_price_history.py`,
`probe_price_today.py`, `schwabAPI/probe_schwab_api.py`.

**These exist because guessing this API has been wrong five times** —
`tokens_file`→`tokens_db`, `account_linked`→`linked_accounts`, `types` needing a
comma-joined string, `order_place`→`place_order`, and `price_history` silently
returning yesterday's bar without an explicit `endDate`. Each cost a production
failure. **Measure before writing against an endpoint.**

---

## 4. State

| path | what | rules |
|---|---|---|
| `~/.schwabdev/tokens.db` | Schwab tokens | SQLite. `tokens.json` is a dead fossil |
| `~/.local/state/myTrading/reserve_ledger.csv` | reserve history | **append-only.** Never rewritten or sorted; corrections are new rows |
| `~/.local/state/myTrading/reserves_config.csv` | which pairs are fenced | declarative intent, safe to rewrite |
| `~/.local/state/myTrading/order_ledger.csv` | every order decision | append-only, write-ahead before submit |
| `~/.local/state/myTrading/remote_ops_audit.log` | ops verbs | JSON per line, never rotated |
| `schwabAPI/data/transactions/` | transaction cache | per account, with a `.state.json` watermark |
| `~/github/jobMyTrading/` | published output | the only thing `gitpush.py` commits |
| `chitra_holdings.csv` | Chitra's positions | in the REPO. From a statement, not the API — only the price is live |
| `chitra_orders.csv` | her resting Merrill orders | in the REPO, transcribed from screenshots. Records its own `As_Of`, so "none" means checked |
| `chitra_reserves.csv` | her fencing and cash earmarks | in the REPO. A **note, not a ledger** — nothing watches fills or refuses an over-spend on that account |
| the `Chitra` tab, cols A–F | her conditions | **human-owned.** The only Chitra state the repo does not hold |

**Balances are always a fold over the ledger**, never a stored number.
`Balance_After` is advisory.

---

## 5. Locking

One shared lock, `/tmp/jobmytrading.lock`, exists so `gitpush.py` cannot commit
`jobMyTrading` while a job is mid-write — `git add -A` on a half-written CSV
commits a truncated file.

| flag | behaviour | used by |
|---|---|---|
| `-w 600` | wait up to 10 min | jobs that must not be skipped |
| `-n` | skip immediately | anything that runs again soon |

**Rules learned the hard way:**

- **Anything writing files into `jobMyTrading` takes it.** Sheet-only writers
  do not need it for git reasons — but `orders_sheet_prices.py` takes it anyway,
  because `jobStocksSignals` rebuilds the Dashboard with `clear()` + rewrite and
  a price write landing mid-rebuild is lost.
- **A manual run takes no lock.** Always wrap one:
  `flock -w 900 /tmp/jobmytrading.lock .venv/bin/python jobStocksSignals.py --no-push`
- **A lock wait must never outlive its own process.** `schwab_auth.install()`
  waits 180s under a cron `timeout 300`. Waiting 600s once got the poller killed
  mid-wait and stranded a row on `RUNNING` forever.

---

## 6. Conventions that matter

**Accounts are the last 3 digits, everywhere.** `acct_key()` is implemented
identically in `orders_sheet`, `cash_reserve` and `order_engine`. Full account
numbers never enter a file or a sheet.

**Timestamps are Pacific with `%Z`.** The job log once printed UTC while the
sheet used the host's timezone — two clocks, neither matching the market hours
or cron schedule they were read against.

**Coverage is per (ticker, account), blank on TOTAL rows.** There is no honest
aggregate; a roll-up would assert something about accounts it cannot see.

**The sheet is a mailbox, not a database.** Pi 1's local files are the truth.
Pi 1 writes *to* the sheet and never reads its own writes back as fact.

**Regenerated tabs paint absolutely.** `ws.clear()` removes values but **not
formatting**, so `_paint()` resets the range before colouring. Stale yellow once
landed on unrelated cells after the blocks were reordered by market value.

---

## 7. Failure modes this system has actually had

Each of these shipped, ran, and was wrong in a way nothing announced.

| what happened | why it was invisible |
|---|---|
| `price_history` returned **Friday's** bar | no error — just a close a day and $15 stale |
| `build_reserves_table` deleted by a wholesale edit | the step was wrapped in `try/except` and logged `(non-fatal)` for two days |
| One intent placed **three** orders | `SUBMITTED` is a live state; the trigger stayed true and re-fired each tick |
| A seed double-counted a sale | manual `seed` plus `apply_fills` crediting the same money |
| A row skipped as history | `stamp_row_ids` recycled a spent Row_ID; the skip logged nothing |
| A reverse-split ATH read as a high | FCEL's ATH field says 234,900 against a $16 price — a number, not an error |
| A config CSV never reached git (×3) | `.gitignore`'s `*.csv` ate it; `git add -A` reports nothing and the commit succeeds |
| A dip buy reported as a breakout | the intent's `Close_Is` was read and then dropped, so coverage guessed from the price |
| A filled order still claiming coverage | `SUBMITTED` is live to the engine, so a spent intent read as arranged protection |
| Eleven months of P&L missing | `_fetch_chunk` returned `[]` on a non-200, indistinguishable from "no transactions" |

**The common shape: a failure that looks like a normal result.** Hence the
recurring rules — fail closed, say why, never return an empty list where an
error belongs, and make staleness visible rather than silent.

---

## 8. Adding something

1. Write it on Pi 2, honouring the injection contract
2. `python3 -m py_compile <file>` then **`.venv/bin/python smoke_test.py`**
3. Test the pure logic locally — Pi 2 has pandas but no Schwab, no credentials
4. If it touches a Schwab endpoint you have not used, **write a probe first**
5. Commit, push, `git_pull` on Pi 1
6. Add to `health_check.py` if it is scheduled, and to `OPERATIONS.md` §1
7. Update `PROJECT_PLAN.md` — move finished items to §8 rather than deleting
