# Operations — what runs, when, and how to tell if it stopped

**Everything scheduled on Pi 1, in one place.** Pi 2 writes code and pushes;
Pi 1 pulls and runs it with the credentials.

Check the whole thing with one command:

```bash
cd ~/github/myTrading && .venv/bin/python health_check.py
```

Exit 0 = healthy, 1 = something stale or missing. `--quiet` prints only
problems, `--json` is machine readable.

---

## 1. The schedule

All times Pacific. `flock` prevents two jobs writing the same files at once.

| When | Job | Writes | Lock |
|---|---|---|---|
| `:00` hourly | `jobCryptoSignals.py` | `crypto_signals.csv` | `-w 600` waits |
| `:15`, `:50` Mon–Fri, 01–16h | `jobStocksSignals.py` | signals, orders, cash, reserves CSVs + **both sheet tabs** | `-w 600` waits |
| `*/5` | `gitpush.py` | pushes `jobMyTrading` to GitHub | `-n` skips |
| `*/15` | `gitpush.py bots` | pushes `botsMyTrading` | `-n` skips (bots lock) |
| `*/10` | `remote_ops.py` | ops sheet results | own lock, `-n` |
| `*/5` 01–16h Mon–Fri | `orders_sheet_prices.py` | `Live_Price`, `Day_%` | `-n` skips |
| `04:30` daily | `build_pl_report.py` | `outputs/portfolio/*` | `-w 600` waits |
| `05:00`, `17:00` Mon–Fri | `45_Signal.py` | 45° scan CSVs | scan unlocked, copy locked |
| `*/15` 06–14h Mon–Fri | `order_engine.py` | Orders tab validation; evaluates triggers after 13:15 PT | `-n` skips |

**On PI 2** (it holds the Gmail credentials and needs nothing from Schwab):

| When | Job | Does |
|---|---|---|
| `08:30`, `18:30` | `token_watch.py --from-sheet` | Schwab token expiry + Pi 1 liveness |
| `09:15` | `reminders.py` | nags about anything outstanding in `reminders.csv`, with `reminder_notes/` images inline at the end |

**Not yet scheduled:** `health_check.py` (see §5).

### Re-authorising is safe at any time

`auth_code` takes the shared `/tmp/jobmytrading.lock` before installing new
tokens, waits up to ten minutes for whatever is running, then swaps in about a
second.

This used to be a rule you had to hold in your head — installing tokens
invalidates the old refresh token, so a job that happened to refresh its access
token at that moment failed, which meant checking the clock against the stocks
cron before re-authorising. The lock puts that rule in the code instead.

If the lock cannot be taken within ten minutes it proceeds anyway and says so.
A token about to expire is worth more than a clean run of one job, and that job
will simply need re-running.

### The token warning runs on PI 2, not Pi 1

Neither machine has both halves: Pi 1 holds the Schwab credentials, Pi 2 holds
the Gmail ones (market-tracker lives there and nowhere else). Copying either
across would have created a second place to rotate a secret.

So nothing is copied. Pi 1 already writes the token state into the `Orders`
header every cycle, and Pi 2 can read that:

```
A2  SCHWAB TOKEN   OK — expires 2026-09-30 23:21:22 PDT (1.52 days)
A3  LAST POLL      2026-09-29 10:59:13 PDT
```

```cron
# on PI 2
30 8,18 * * * cd /home/chakravarti/github/myTrading && set -a && . ./.env && set +a && timeout 120 .venv/bin/python token_watch.py --from-sheet >> /home/chakravarti/.local/state/token_watch.log 2>&1
```

**It watches two things, and the second is free.** `LAST POLL` is rewritten
every cycle, so a header that has stopped moving means **Pi 1 itself is down** —
which is precisely the failure Pi 1 could never report. Over
`POLL_STALE_HOURS` (default 3) the email leads with it.

`token_watch.py` without `--from-sheet` still reads the local token store, for
running on Pi 1 by hand.

### Email

`token_watch.py` sends through Gmail SMTP using the **market-tracker
project's** credentials, read directly from
`/home/chakravarti/agents/market-tracker/.env` (`GMAIL_ADDRESS`,
`GMAIL_APP_PASSWORD`).

Deliberately not copied into this repo's `.env`: one place to rotate the
password, one place that can go stale. Recipients are this project's business
and come from `ALERT_TO` here, defaulting to both of Chakravarti's addresses.

The reference — including gotchas that cost real debugging time on that project
— is `/home/chakravarti/agents/market-tracker/EMAIL-SETUP.md`. The two that
shape the code here:

- **Strip spaces from the app password.** Google displays it in groups of four;
  the credential is the 16 unbroken characters, and sending it with spaces
  gives a `535` indistinguishable from a wrong password.
- **`.env` must win over `os.environ`.** A stale export in an interactive shell
  otherwise shadows the correct value and produces the same baffling `535`.

Expect periodic blocks: Gmail SMTP from a residential IP on a fixed schedule is
what anti-abuse systems look for. `send()` therefore never raises, and the
caller only records the warning as sent on success — so a refused send simply
retries next run rather than being lost.

### The order engine's two jobs in one schedule

It runs every 15 minutes and does different work depending on the clock:

- **Any time** — validates every row in the `Orders` tab and writes the verdict
  into column N. This is the feedback loop: a mis-typed row says so within
  minutes of being typed rather than failing silently at the close.
- **After 13:15 PT** — also evaluates triggers against the completed daily bar,
  and submits what fired.

`LIVE_TRADING` is off unless `ORDER_ENGINE_LIVE=1` is in `.env`, so the
schedule is safe to add before you are ready to trade from it.

### Why the locks differ

`-w 600` **waits** up to 10 minutes: used by jobs that must not be skipped.
`-n` **skips immediately**: used by anything that will simply run again soon —
missing one tick of a 5-minute job costs nothing.

**Running a job by hand takes no lock.** Cron will start its own copy on top of
yours, and both will write the same files. Always wrap a manual run:

```bash
flock -w 900 /tmp/jobmytrading.lock .venv/bin/python jobStocksSignals.py --no-push
```

---

## 2. Nothing pushes except `gitpush.py`

Every signal job runs `--no-push` and only writes files. `gitpush.py` is the
sole git writer, on its own `*/5` schedule, doing `git add -A` at the repo root.

So **"did it reach GitHub?" is never a question about the job that produced the
file.** If a file is on Pi 1's disk but not on GitHub, look at `gitpush.py`.

---

## 3. What writes to the Google Sheets

| Sheet | Written by | When |
|---|---|---|
| `myTrading-ORDERS-pi1` · Positions, Cash, Dashboard, header | `jobStocksSignals.py` step 4d | `:15`, `:50` |
| `myTrading-ORDERS-pi1` · `Live_Price`, `Day_%` | `orders_sheet_prices.py` | every 5 min |
| `myTrading-ops-pi1` · columns C–J | `remote_ops.py` | every 10 min |

Both sheets are edited by `mytrading-ops@…` (Editor). Pi 2 reads them through
`mytrading-reader@…` (Viewer). See `ordersSheetDesign` §12.

**The header block is the health check.** `LAST POLL` is rewritten every cycle
even when nothing changed, so a timestamp three days old means the poller is
dead — no failure detection needed.

---

## 4. Logs

| Log | Written by |
|---|---|
| `~/job_stocks.log` | stocks cron (stdout), copied into `jobMyTrading` |
| `~/github/jobMyTrading/job_stocks.log` | the job's own `log()` |
| `~/crypto_job.log` | crypto cron |
| `~/gitpush_cron.log` | the pusher |
| `~/.local/state/myTrading/remote_ops_cron.log` | ops poller |
| `~/.local/state/myTrading/remote_ops_audit.log` | ops audit, JSON per line, never rotated |
| `~/.local/state/myTrading/prices_cron.log` | price updater |
| `~/pl_report.log` | daily P&L |
| `~/45_signal_{morning,evening}.log` | 45° scans |

A manual run prints to the terminal **and** appends to the job's own log; the
cron redirect only captures the cron copy. So a `grep` of `job_stocks.log`
after a manual run may show nothing — that is expected.

---

## 5. Checking it still works

### `smoke_test.py` — before you trust an edit

```bash
.venv/bin/python smoke_test.py
```

Imports every module and asserts the functions **other modules call** are still
present. Run it after any edit and before pushing.

It exists because `py_compile` cannot see a deleted function. A wholesale edit
to `cash_reserve.py` removed `build_reserves_table` and `overcommit_warnings`
— both syntactically fine, both silently gone — and the reserves step failed
`(non-fatal)` for two days before anyone noticed the CSV had stopped changing.

### `test_recommend.py` — the recommendation formulas

```bash
.venv/bin/python test_recommend.py
```

Runs anywhere: no Schwab, no Sheets, no network. **Every case is a real trap
the live data contains**, not a hypothetical — FCEL's reverse-split ATH of
234,900, SOFI's Supertrend sitting above price, MSTX's negative MRC band,
TSLA's collapsed stop ladder. Run it with `smoke_test.py` after touching
`core/recommend.py`.

### `test_coverage.py` — the Dashboard protection flags

```bash
.venv/bin/python test_coverage.py
```

These flags decide whether a position reads as **protected**, and both bugs
they have had said "covered" about something that was not — the expensive
direction to be wrong in. Run it after touching `coverage_for`, `_classify` or
`read_intents`.

### A config file that vanishes from git

**`.gitignore` has eaten a config CSV three times now.** The `*.csv`
catch-all is broad on purpose — the pipeline writes a lot of CSVs — so any
file the *repo* owns needs an explicit negation placed **after** it:

```gitignore
*.csv
!/Data/*.csv
!/reminders.csv            # these must come AFTER the catch-all
!/chitra_holdings.csv      # or it wins, silently
```

The failure is invisible: `git add -A` reports nothing, the commit succeeds,
and the file simply is not there. `reminders.csv` ran untracked for a day
while its own docstring claimed it was versioned. **After adding any config
file, check it:**

```bash
git check-ignore -v <file>     # no output = tracked, which is what you want
```

### `health_check.py` — whether it is all still running

Checks six things, none of which require the jobs to cooperate:

1. **Crontab lines** — an active line exists for each of the seven jobs. A
   commented-out line reads as missing, which is what it is.
2. **Output freshness** — every job leaves a file; its age says whether the job
   ran. Tolerances are roughly two missed runs, so one hiccup is not an alarm
   but a stopped job is. Weekday-only outputs are not checked at weekends.
3. **Schwab token** — a hard 7-day cap that takes down everything Schwab, so it
   belongs beside the jobs it would kill.
4. **P&L reconciliation** — a report can run on schedule and still be wrong, so
   the share-count and cost-basis defect counts are surfaced here rather than
   only in a run log nobody reads.
5. **Trading state** — whether the kill switch is on, and whether
   `ORDER_ENGINE_LIVE` is set. Both are surprising in both directions: an
   engine you believe is armed but is not, and one you believe is idle but is
   live, are equally worth knowing.
6. **Orders placed today**, counted from the ledger. A number you did not
   expect is the first sign of something firing repeatedly.

Suggested cron once you are happy with it — mails only on failure:

```cron
0 7 * * * cd /home/rchak007/github/myTrading && .venv/bin/python health_check.py --quiet || true
```

**Staleness is the signal throughout.** A stale CSV looks exactly like a fresh
one and a cron that stopped firing produces no error anywhere, so age is the
only thing that reliably distinguishes them.

---

## 6. Runtime and the timing trap

`jobStocksSignals.py` takes **~9-10 minutes** and logs `⏱ Total runtime` on
every run, warning past 25 minutes.

That threshold matters: cron fires at `:15` and `:50`, so the narrow gap is
**25 minutes**, while the crontab's `timeout` allows **40**. A run that ever
exceeds the gap collides with the next, which waits on `flock -w 600` and then
dies **without running** — no error, just a skipped cycle. See
`PROJECT_PLAN.md` §7.

Crypto has the same shape: `:00` hourly with `timeout 2400`. A crypto run over
25 minutes would starve the `:15` stocks run the same way.

---

## 7. When something looks wrong

| Symptom | Look at |
|---|---|
| Sheet not updating | `LAST POLL` in the Orders header; then `job_stocks.log` |
| File on Pi 1 but not GitHub | `~/gitpush_cron.log` |
| Ops sheet row never runs | column C — a stamped row above is skipped, not a boundary; see `remoteOpsGuide` |
| Prices stale on the Dashboard | `prices_cron.log`; market hours only |
| Everything Schwab failing | token expiry — `token_status` in the ops sheet |
| P&L numbers look wrong | `anomalies.csv`, then `PROJECT_PLAN.md` §1 |

---

## 8. Pi 2 — the dev machine

No credentials, no crons, nothing scheduled. It has:

- `gspread` + `google-auth` + `pandas` in `.venv`
- a **Viewer** key at `~/.config/myTrading/gsheets-reader.json`
- read-only clones of `jobMyTrading` and `botsMyTrading` via deploy keys

So it can read the sheets and every published output, and change nothing. That
is deliberate: Pi 2 is where untested code runs.

Reached over **Tailscale**, not the LAN address.
