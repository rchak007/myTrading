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

Finished items move to [§6 Done](#6-done) rather than being deleted — the
history of what was fixed is worth as much as the list of what is left.

Last updated: 2026-09-08

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

---

## 2. Seed money / capital reserves

**Goal:** Reserve a fixed dollar allocation per ticker. If MU is sold, the
remaining reserved dollars stay earmarked for buying MU back, rather than
being treated as free cash.

**Status:** 💡 Idea only. Nothing built.

### ❓ OPEN QUESTION — what does a reserve actually mean?
Two readings, and the guard logic differs materially:
- **cash committed to a ticker** — money set aside, spent on purchase
- **total capital allocated to a ticker** — a target position size, needing a
  `Target_Capital` column in a config file

This blocks §7 of the order-execution design and any `cash_reserve.py`.

### 🐞 DEFECT — `cash_reserve.py` is referenced but does not exist
`orderExecutionDesign-9-7-26.md` §3.4 and §7 call
`cash_reserve.committed_for_other_tickers(account)` as though the module
exists. It does not, nor do `reserves_config.csv` or `reserve_ledger.csv`. The
"reserve" wording inside `stocks_cash.py` is a *different* concept — cash tied
up by open BUY orders, already netted out by `Cash_After_Open_Orders`.

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

## 5. Infrastructure defects (cross-cutting)

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

### 💡 IDEA — `ls jobs` / `ls bots` verbs probably fail on Pi 1
`JOBS_REPO` and `BOTS_REPO` default to `~/github/jobMyTrading` and
`~/github/botsMyTrading`. Confirm Pi 1's layout or repoint the `DIRS` dict.

---

## 6. Done

Kept for history — what was fixed, and when.

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
