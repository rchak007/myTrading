# myTrading — Project Plan & Open Items

**Living document.** Unlike the dated guides in this folder, this one is edited
in place. Ask Claude Code "what's still open?" and it works from this file.

Three kinds of entry, kept deliberately separate because they need different
responses:

| | |
|---|---|
| 💡 **IDEA / PLAN** | Something to build. Needs design before code. |
| 🐞 **DEFECT** | Something built that is wrong. Blocks trusting output. |
| ❓ **OPEN QUESTION** | A decision only Chakravarti can make. Blocks work downstream. |

Finished items move to [§8 Done](#8-done) rather than being deleted — the
history of what was fixed is worth as much as the list of what is left.

Last updated: 2026-09-14

---

## 1. Historical P&L per equity

**Goal:** Know the real profit and loss on every stock ever traded, across all
six Schwab accounts, with incremental updates so it stays current.

**Status:** Pipeline works end to end. First full pull completed 2026-09-08 —
**5,816 transactions, 2020-03 → 2026-09, six accounts.** Numbers are NOT yet
trustworthy; see the defects.

### 🐞 DEFECT — 23 tickers where our share count exceeds Schwab's
The error is always in one direction: we hold *more* than reality, never less.
Thirteen tickers show Schwab at zero while the engine still has open lots
(SOFI 315, BITO 294, ASST 420, METV 175, TDOC 100, AI 100, SNOW 65, ROKU 50,
BYND 22, XYZ 14, `86800U104` 12, NFLX 8, `25058X105` 145). TSLA is off by
exactly 36.00, AMD by 90, NVDA by 33, AEHR by 109.

We are systematically missing **exits**. Prime suspect: the `SECURITY_TRANSFER`
type filter, dropped because the current API rejects the name — if ACATS
transfers-out are filed under it, every one was lost. Until this is closed the
cost basis of those positions is sitting in `open_cost_basis` instead of being
realized, so **realized P&L is understated**.

*Next step:* the action histogram from `ticker_txns.csv` — if `TRANSFER_OUT` is
absent while thirteen tickers went to zero, confirmed. Then find the new type
name.

### 🐞 DEFECT — 26 tickers with no cost basis (`UNKNOWN_BASIS`)
Transferred in from another broker, so Schwab reports cost 0 and their P&L is
overstated. Includes four raw CUSIPs: `512807108`, `862945102`, `25058X105`,
`86800U104`. Fix by filling `schwabAPI/manual_basis.csv` from old broker
statements. Affects TSLA, MRVL, MU, COIN, GOOGL, META, TTD, ASML and others.

### 🐞 DEFECT — 7 unconfirmed CUSIPs in `corporate_actions.json`
`unmapped_cusips_to_confirm`, one flagged as *likely* LRCX. A wrong mapping
silently merges two tickers' P&L, so confirm before adding.

### 💡 IDEA — Validate against statements
Pre-2022 was **TD Ameritrade**; those statements exist and a first check
"looked not bad." Needs far more testing before the numbers are believed.
Post-2022 Schwab history should be checked with `ticker_report.py <TICKER>`,
which lists every transaction with running position — the first date that
disagrees with Schwab's UI is the defect.

### 💡 IDEA — P&L dashboard
Views and analysis over the results: per-ticker, per-account, household.
**Time slices** are an explicit requirement — "what about the last 6 months?" —
so the engine needs to answer P&L over an arbitrary window, not just
all-time. Current output is all-time only; windowed realized P&L means
replaying FIFO between two dates.

### 💡 IDEA — schedule the incremental pull, and watch its health
Today `build_pl_report.py` is run by hand. It needs a cron on Pi 1 doing the
incremental fetch (`last_date − 10 days` → today per account), plus a health
check so a *silent* failure is noticed. Silence is the risk: a failed run
leaves the last good CSVs in place, so the report looks fine while going stale.
Health signals worth asserting: every account's `last_date` is recent, row
counts only grow, and `anomalies.csv` has not suddenly jumped.

Note the auth constraint — the Schwab **refresh token expires in ~7 days**, so
any schedule longer than that guarantees a dead run. Either the cron cadence
stays well inside the window, or re-auth is handled first (see §6).

### ❓ OPEN QUESTION — what cadence?
Daily after the close? Weekly? The trade-off is API load versus staleness.
Transactions settle and get back-dated by Schwab, which the 10-day overlap
already absorbs, so daily is not required for correctness — it is a question of
how fresh the dashboard should be.

### 💡 IDEA — publish P&L output so Pi 2 can analyse it
Pi 1 holds the credentials and does the fetching; Pi 2 has no Schwab access and
is the analysis box. The plumbing for this **already exists in the design**:
`schwab_client.REPORT_DIR` defaults to
`~/github/jobMyTrading/outputs/portfolio`, and `build_pl_report.py` deliberately
does not git push — `gitpush.py` remains the sole git writer. So dropping the
`--out` override makes the CSVs land where `gitpush.py` will carry them, and Pi
2 pulls read-only.

*To verify first:* whether `jobMyTrading` is actually cloned on Pi 1, and
whether `gitpush.py` already covers `outputs/portfolio`.

---

## 2. Seed money / capital reserves

**Goal:** Reserve a fixed dollar allocation per ticker. If MU is sold, the
remaining reserved dollars stay earmarked for buying MU back, rather than
being treated as free cash.

**Status:** Module **added 2026-09-14** (`cash_reserve.py`, spec in
`Documentation/CASH_RESERVE_HANDOFF.md`). Written but **never run and not wired
into anything**. Follows the house injection contract: no paths, no logger, no
HTML renderer, no Schwab client of its own.

State lives outside the repo at `~/.local/state/myTrading/`
(`reserve_ledger.csv`, `reserves_config.csv`), override with
`MYTRADING_STATE_DIR`. The ledger is append-only money history and must never
be committed, sorted, or de-duplicated in place — corrections are new `ADJUST`
rows, and `fold_balances()` is the only authority on a balance.

### ❓ OPEN QUESTION — `CASH_ONLY` or `TOTAL_CAPITAL` as the default?
Both policies are implemented; the handoff deliberately declines to pick.
- **`CASH_ONLY`** (current default) — the reserve is dry powder. Buys debit,
  sells credit. `Available_To_Buy = balance`. Deterministic, needs no price data.
- **`TOTAL_CAPITAL`** — `Available_To_Buy = Target_Capital − Position_Value`,
  capped by the reserve. Self-limiting as a position runs up, **but a sharp
  drawdown automatically re-opens buying capacity.** That may be exactly what
  is wanted, or exactly what is not.

Must be confirmed before the gate is wired to anything that can submit an order.

### 🐞 DEFECT — `close()` raises on a breached reserve
`close()` appends a `CLOSE` event of `-cur`. When the balance is already
negative (a BREACH), `-cur` is **positive**, and `append_events()` enforces
`CLOSE` ∈ `DEBIT_EVENTS` must be `<= 0` — so it raises `ValueError` and the
reserve cannot be closed. A breached reserve is exactly the one you most want
to close.

### 🐞 DEFECT — the order-exec design calls an API that does not exist
`orderExecutionDesign-9-7-26.md` §7 uses
`cash_reserve.committed_for_other_tickers(account)`. The module has no such
function. Its gate is `available_to_buy(account, ticker) -> {"allowed",
"available", "policy", "reason"}` — per-pair and fail-closed, rather than a
per-account subtraction. The module's shape is the better one; the design doc's
§7 formula needs rewriting to match, not the reverse.

### 🐞 DEFECT — `fetch_fills()` still probes the pre-upgrade schwabdev
It tries `transactions` / `account_transactions` / `transactions_all` across
several kwarg shapes because the signature "varies by version". **We now know
the signature exactly** (measured 2026-09-08, see `probe_schwab_api.py`):
`transactions(accountHash, startDate, endDate, types: str, symbol=None)`, and
accounts come from `linked_accounts()`, not `account_linked()`. Replace the
probing with the confirmed call. Note `types` must be a comma-joined **string**;
`"TRADE"` alone is valid.

Handoff §8 step 4 also asks to confirm `activityId` not `transactionId`, and
`netAmount` not `amount` — both **already confirmed** on 2026-09-08 while
fixing `txn_cache.py`. The module already reads them correctly.

### 💡 IDEA — wire it in (handoff §8 build order)
Unrun and unwired. In order: smoke-test offline with
`MYTRADING_STATE_DIR=/tmp/rtest` (needs no Schwab), add `OUT_RESERVES_CSV` /
`OUT_RESERVES_HTML` to `jobStocksSignals.py`, add the reporting step in its own
`try/except` so a reserve failure stays non-fatal, then fix `fetch_fills()`.
**Do not** gate live orders on it until §4's security model exists.

### 💡 IDEA — decide where sell proceeds go
A SELL currently credits back to the ticker's fence, keeping the sleeve intact.
The alternative is releasing to free account cash. Affects `sell_guard.py`
integration.

### 🐞 DEFECT — the ledger has no backup
`reserve_ledger.csv` is money history, lives outside the repo, and is therefore
backed up by nothing. Needs a real backup path before it matters.

---

## 3. Order-coverage check ("is every position protected?")

**Goal:** For every ticker held, confirm the right resting orders exist:
- a **SELL** below, if it moves against you (downside protection)
- a **TRIM** above, to take profits
- a **BUY**, where seed money is allocated for that ticker

**Status:** 💡 Idea, partially prototyped. `sell_guard.py` already implements
the downside half — it flags 🔴 `NO_SELL` for any held position with no
protective SELL leg priced below the current price, and correctly refuses to
count a sell LIMIT *above* the price (that's a profit target, not protection).

### 💡 IDEA — extend to trims and seed-money buys
Needs design. The trim side is the mirror of `sell_guard`; the buy side depends
on §2 being settled first, since "where I want to buy" is a reserve question.

---

## 4. Sheet-driven trade execution

**Goal:** Specify an intent in a sheet — "buy MU if it closes above X" — and
have the program place the order when the condition is met. Conditions Schwab
itself cannot express as a resting order.

**Status:** Designed in full, **not implemented**. See
`Documentation/orderExecutionDesign-9-7-26.md`. Nothing ships until that
document's §6 security model is built in full.

### ❓ OPEN QUESTION — the sheet must cover mixed order kinds
New requirement, 2026-09-08. Some orders go **straight to Schwab as a limit
order**; others are **close-condition triggers the program evaluates**. Both
must live in one sheet with columns that make the distinction unambiguous, and
cover every case. The current design assumes only the second kind. Needs a
column design pass before implementation.

### 🐞 DEFECT — §4.2 contradicts §6.3 on `Row_ID` collisions
§4.2 rejects collisions "in any state"; §6.3 skips terminal-state matches
silently and only voids live-state ones. §6.3 is correct — a filled row stays
in the sheet and is re-read every cycle, so §4.2's wording would flip every
historical row to `VOID`.

### 🐞 DEFECT — the daily evaluation window can lose a bar
Timer runs 06:00–13:30 PT; trigger evaluation only fires after 13:15, roughly
four cycles. If the price fetch fails through 13:30 the rows correctly stay
`ARMED`, but next morning the "after 13:15" condition is false, so **that day's
close is never evaluated** and a trigger that genuinely fired is skipped
permanently. Fix: evaluate *the most recent unevaluated completed bar*, not
*today's if it is after 13:15*.

### 🐞 DEFECT — `enable_trading` reopens the hole §3.3 closes
§3.3 argues that putting order capability in the general ops channel makes a
`remote_ops` compromise an order-placement compromise. §6.7 then puts
`enable_trading` in exactly that channel, so sheet write access clears the kill
switch. The directions are not symmetric: disabling is fail-safe, enabling is
not. Suggest `disable_trading` + `trading_status` via the sheet, SSH/local
presence required to re-enable. Also mechanical: `remote_ops` runs as the user
and cannot create `/etc/myTrading/TRADING_DISABLED` without a sudoers rule.

### ❓ OPEN QUESTION — carried from the design doc
- **`Qty` vs `Notional`** — support both, or force `Qty` in v1? `Notional`
  needs a share-rounding rule, and a decision for rounding to zero.
- **Which accounts** — 401k/PCRA accounts are custodial and often not linked to
  the developer app. Confirm which account hashes actually accept orders.
- **Unfilled DAY limit orders** — auto-rearm next session, or terminal and
  require a new row? Doc assumes terminal.
- **Notification** — does a submission need to reach the phone? The sheet alone
  is not a notification channel.
- **GTC vs `Expires_On`** — §2 forbids order modification and the only
  `SUBMITTED → CANCELLED` path is a human cancelling at Schwab, so `Expires_On`
  has no effect once submitted. A GTC order can rest past its expiry date.
  Probably acceptable, but should be stated explicitly.

---

## 5. Evaluate bot / job runs

**Goal:** Every day, scan the logs from the trading jobs and bots, and report
anything that failed — without Chakravarti having to go and look.

**Status:** 💡 Idea. Nothing built.

**Shape:** Pi 1 runs the jobs and already pushes to two repos. Pi 2 has no
credentials and no execution role, which makes it the right place to analyse:
it pulls **read-only** and reports. Same split as §1's publishing idea — Pi 1
produces, Pi 2 reads.

```
   Pi 1 (executes, has creds)          Pi 2 (analyses, read-only)
        │ push                                  │ pull
        ▼                                       ▼
   github.com/rchak007/jobMyTrading   ──────────┤
   github.com/rchak007/botsMyTrading  ──────────┘
```

### 💡 IDEA — read-only access for Pi 2 to both repos
`jobMyTrading` and `botsMyTrading`. **Read-only is a deliberate constraint, not
a convenience** — Pi 2 must not be able to write to the repos that drive
execution. Use a deploy key per repo (read-only checkbox) or a fine-grained PAT
scoped to contents:read on exactly those two. Pi 2's existing `github-agents`
SSH identity has *write* access to `myTrading`, so this needs separate
credentials rather than reusing that one.

### 💡 IDEA — daily log scan and failure report
Parse the logs, decide what "failing" means, report. First make it work by hand
on a real day's logs, *then* automate — the classifier is the hard part and is
best written against logs whose outcome is already known.

### 💡 IDEA — cron the pull-and-analyse loop
Only after the manual version works. Read-only `git pull` on Pi 2, then run the
analysis. Note Pi 2 pulls; it never pushes to these two.

### ❓ OPEN QUESTION — what counts as "failing", and where does the report go?
A non-zero exit? A traceback in the log? A job that did not run at all —
which is the one that produces *no* log line and so is easiest to miss?
And delivery: terminal on demand, the ops sheet, a file in the repo, or a push
notification? "Nothing ran today" is the failure mode most worth catching and
the hardest to detect from logs alone.

---

## 6. Research review cadence

**Goal:** Keep the coverage log and watchlist current, so bucket assignments
(accumulate / trade / avoid) reflect what the businesses are actually doing
rather than what they were doing months ago.

**Source:** `Documentation/WALL-STREET-LEVEL-STOCK-ANALYSIS.md` — prompt library
(Part A), portfolio framework and tier rules (Part C), analytical principles
(Part D), coverage log (Part E), review protocol (Part F), 29-name watchlist
(Part G).

### 🔁 RECURRING — run the full review MONTHLY

**Cadence override:** Part F of that document says *quarterly*. Chakravarti
changed this to **monthly** on 2026-09-14. This section is authoritative.

- **Scope:** the 29-name watchlist (Part G) plus the coverage log (Part E)
- **Per name:** the 8-point checklist in Part F — price/ATH with as-of date,
  latest quarter, guidance shape, **estimate revision direction**, share-count
  change, forward P/E and PEG, any Tier 1 invalidation event, bucket confirmation
- **Portfolio level:** hyperscaler capex guidance (the stated leading indicator),
  DRAM contract pricing for the memory bucket, Fed path, theme exposure vs dry
  powder, any position past its dollar cap
- **Prompts:** #1 + #6 + #10 per name; #4 ahead of any earnings inside the window
- **Priority backlog:** 15 names have never been verified against current
  fundamentals — ALAB, SIMO, SITM, COHR, LITE, GLW, NET, CEG, VST, NEE, GEV,
  VRT, BE, AMD, MRVL. Work these first.

**Next due: October 2026.**

| Run | Date completed | Notes |
|---|---|---|
| Sep 2026 | 2026-09-14 | Baseline — Part E coverage log as written |
| Oct 2026 | — | |

### 💡 IDEA — automate the reminder, and make it persist until acknowledged
Chakravarti's requirement: the reminder should keep nagging **until he confirms
it is done**, not fire once and vanish. A plain cron that prints into a log is
exactly the silent-failure shape already flagged in §1 and §5 — a reminder
nobody sees is not a reminder.

So it needs somewhere to record acknowledgement and something that re-raises
while that is unset. Options not yet chosen: a row in the ops sheet, a state
file on Pi 2 plus a re-raise on each session, or a scheduled agent. Decide when
the ops-sheet work settles.

Until then this is manual: ask "what's open?" and this section reports it.

---

## 7. Infrastructure defects (cross-cutting)

### 🐞 DEFECT — `auth_url` / `auth_code` verbs fail from the sheet
`remote_ops.py` has no `dotenv` import, and `schwab_auth.py` reads credentials
only from the environment. So `authorize_url()` and `install()` raise
`missing in .env` unless the service environment supplies them. Worse,
`auth_url` is **not** wrapped in `try/except` in `run_pyverb`, so the exception
escapes `run_verb` into `main()`, crashes the poll cycle *after* the row was
marked `RUNNING`, and strands that row forever (non-empty Status = "already
handled"). `token_status` is unaffected — `status()` reads only the token file.
*Fix:* load `.env` in `run_pyverb` and wrap the whole body.

### 🐞 DEFECT — `schwab_auth.py --status` is misleading
It computes expiry from a `refresh_token_issued` stamp in
`~/github/myTrading/tokens.json` against a hardcoded 7-day TTL. But current
schwabdev stores tokens in **`~/.schwabdev/tokens.db`**, and `tokens.json` is a
dead artifact from May. Its date says nothing about whether auth works — this
caused a wrong "your token expired" diagnosis on 2026-09-08. Either point
`schwab_auth.py` at the DB or stop trusting its verdict.

### 🐞 DEFECT (minor) — auth code leaks into the audit log on a typo
A mistyped verb (`auth_cod`) with a real redirect URL in column B is rejected,
and `audit(event="rejected", args=arg)` logs the URL in plaintext, because the
typo is not in `PY_VERBS` and so escapes `safe_arg()`. Codes are single-use and
short-lived, so this is minor.

### ❓ OPEN QUESTION — is the ops sheet shared as Editor or Viewer?
`remote_ops.py` requests full read/write scope and is "the only writer" — it
writes `RUNNING` into column C, then C..J on completion. But the **Drive share
is the real boundary**; the OAuth scope is client-side only. If
`myTrading-ops-pi1` is shared with `mytrading-ops@...gserviceaccount.com` as
**Viewer**, every write-back fails, `write_back()` retries three times, logs
`writeback_failed` to the audit log, and returns quietly — the sheet row just
stays blank. Never verified, because the ops channel has never run successfully.
Check Manage access before debugging anything else there.

### 💡 IDEA — `ls jobs` / `ls bots` verbs probably fail on Pi 1
`JOBS_REPO` and `BOTS_REPO` default to `~/github/jobMyTrading` and
`~/github/botsMyTrading`. Confirm Pi 1's layout or repoint the `DIRS` dict.

---

## 8. Done

Kept for history — what was fixed, and when.

**2026-09-14 — RSI was not RSI**
- `core/indicators.compute_rsi` averaged gains/losses with a SIMPLE moving
  average rather than Wilder's RMA. That is a different indicator (Cutler's
  RSI) and reads far too low after a decline. CRDO showed **20.9** in
  `beth_funds.csv` against **~30.8 on both TradingView and Yahoo daily** —
  measured 11–17 points of divergence on a comparable series.
- The same file's `compute_most_rsi()` had always used the correct RMA, so
  `RSI` and `MOST_RSI` in one CSV were computed by two different definitions.
- Affects `signal_combined` (BUY requires RSI > 50) and therefore the
  `Combined Signal` / `Full Combined` columns. Does **not** affect
  `SIGNAL-Super-MOST-ADXR`, which uses only Supertrend/MOST/ADXR.
- **Deliberately NOT changed:** `backtest_strategies.py:454-455` keeps the old
  simple-mean formula in `run_backtest_mean_reversion_rsi`. Chakravarti's call,
  2026-09-14 — that strategy has been tuned against it and changing the formula
  would move every historical result. So backtest and live RSI now use
  different definitions **on purpose**; do not "fix" it without asking.

**2026-09-08 — Schwab P&L pipeline made to run**
- Fixed `schwab_client.py` for current schwabdev: tokens live in
  `~/.schwabdev/tokens.db`, not `tokens.json`; the `tokens_file` kwarg no
  longer exists (`9a959c8`)
- Fixed `txn_cache.py` for the current API: `account_linked()` →
  `linked_accounts()`; `types` is a comma-joined **string**, not a list; five
  type names now rejected as invalid enums (`4431c0b`)
- Added `probe_schwab_api.py` — measures the max date window (364 days) and
  which `types` values are valid, instead of guessing (`5c371b8`)
- Added `ticker_report.py` — per-ticker transaction listing with running
  position, for validating against Schwab (`49c35f1`)
- Rewrote `schwabAPI/README.MD`, which was two generations stale and pointed at
  a file that does not exist (`2b4222b`)
- **First successful full history pull: 5,816 transactions, six accounts**

**2026-09-07 — Schwab re-auth from the phone**
- Added `token_status` / `auth_url` / `auth_code` in-process verbs to
  `remote_ops.py`, with the pasted code burned from the sheet cell after use
  (`c5e714c`)
- Redacted free-form arguments from the audit log by rule rather than by verb
  name, so future `PY_VERBS` entries are covered automatically — the code was
  previously written to the audit log in plaintext, twice, despite being
  scrubbed from the sheet (`c5e714c`)
- Fixed `--dry-run`, which reported the new verbs as `WOULD REJECT` (`c5e714c`)
- Wrote `Documentation/remoteOpsGuide-9-7-26.md` (`e7b5114`)
- Recorded `Documentation/orderExecutionDesign-9-7-26.md` (`ad8f5f6`)
- Untracked `tokens.json.bak`: `.gitignore` had covered it since line 66, but
  gitignore has no effect on an already-tracked file, and `schwab_auth.py`
  rotates a live token into that exact path on every re-auth (`b7febe8`)
