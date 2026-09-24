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

**Not yet scheduled:** `token_watch.py` (needs SMTP credentials),
`health_check.py` (see §5).

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

## 5. `health_check.py`

Checks four things, none of which require the jobs to cooperate:

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
