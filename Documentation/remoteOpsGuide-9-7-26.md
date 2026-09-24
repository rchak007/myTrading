# Remote Ops Guide — Pi 1 command channel + Schwab re-auth

`remote_ops.py` lets you drive Pi 1 from your phone through a Google Sheet.
You type a **verb** into a row; Pi 1 polls the sheet on a cron, runs only
verbs that appear in its allowlist, and writes the result back into the same
row so you can read it without SSH.

As of the `auth_code` patch it also carries the three Schwab OAuth verbs, so
the 7-day refresh-token renewal can be done entirely from a phone.

- **Script:** `remote_ops.py` (repo root)
- **Auth helper:** `schwab_auth.py` — called in-process, never shelled out to
- **Sheet:** `myTrading-ops-pi1`, tab `ops`
- **Runs on:** Pi 1

---

## 1. Security model

This is the part to preserve when editing. Everything else is detail.

- The sheet supplies a **verb name, never a command string**.
- Every subprocess is an **argv list**. `shell=True` appears nowhere.
- Subprocess verb arguments are **keys into fixed dicts** (`LOGS`, `DIRS`),
  never paths — so `../../.env` cannot be smuggled through column B.
- Anything that moves funds or restarts the bot service is deliberately
  **not** in the allowlist. Keep it that way.

### The one exception

`auth_code` takes a free-form string (the pasted redirect URL), which breaks
the keys-only rule. It is safe **only** because that string never reaches an
argv or a shell: it is parsed by `urlparse`, validated against a character
class, and the extracted code goes into a `requests` POST body. This is why
the auth verbs run in-process rather than through `subprocess`.

> **Do not add a free-string verb that shells out.**

### Argument redaction

`safe_arg()` masks arguments in the audit log and `--dry-run` output. In-process
verbs are the only ones taking a free-form argument, so **for a `PY_VERBS` entry
the default is redact** — a verb must opt out explicitly. Dict-key arguments
(`stocks`, `repo`) log verbatim, because they are safe and useful.

`PY_VERBS` maps `verb -> (needs_arg, sensitive)`. These were a single boolean
until `seed` arrived, which needs an argument whose value is *not* secret and
which moves money — so it must be logged, not hidden. A half-written entry (a
bare `True`, a 1-tuple) is treated as **sensitive**: redacting something
harmless costs an unhelpful log line, while the reverse writes a credential
into a file we keep forever.

Known residual: a **typo'd** verb (`auth_cod`) with a real URL in column B is
rejected, and the rejection audit line logs the URL in plaintext. The code is
single-use and short-lived, so this is minor, but worth knowing.

---

## 2. How the loop works

Each cron run:

1. Opens the sheet (`GSHEET_OPS_ID`, tab `GSHEET_OPS_TAB`).
2. Examines the top `SCAN_WINDOW = 25` rows from row 2 down.
   - Non-empty Status → already handled, skip and keep scanning.
   - Blank column A → skip, keep scanning.
   - Verb present but its required argument missing → skip unstamped, so it
     runs once you finish typing.
3. For a live row: **claims** it by writing `RUNNING` into C *before* running
   anything. If that write fails, the verb does **not** run and the batch
   stops — see below.
4. Runs the verb, writes C..J back in a single range call.

**No state is stored.** What has run is read off the sheet itself every poll —
column C is the record — so inserting rows at the top is safe and nothing can
point at a stale row number.

**Batch cap:** `MAX_ROWS_PER_RUN = 5` executions per poll. A pasted wall of
rows cannot stampede the box; the remainder runs on the next tick — which only
works because stamped rows are skipped rather than treated as a boundary. An
earlier version stopped at the first stamped row, so queueing six commands ran
five and abandoned the sixth forever.

**Run-once guarantee.** A non-empty Status is the *only* record that a row was
handled, so the claim write is load-bearing: `write_back()` returns whether it
landed, and an unclaimable row is skipped entirely rather than run blind. For
`seed` a re-run would fence the money twice.

**Output:** the full text is spilled to `remote_ops_out/` on disk; column I
keeps the last `SHEET_OUTPUT_MAX = 1500` chars. The **tail** is kept, not the
head — for logs and git output that is the useful end.

**Write-back retries:** 3 attempts with exponential backoff for transient
429/500s, then an audited `writeback_failed` and give up.

---

## 3. Sheet layout

Row 1 is the header (written by `--init`, or pasted).

| Col | Field | Filled by |
|-----|------------|-----------|
| A | Verb | **you** |
| B | Args | **you** (only if the verb needs one) |
| C | Status | Pi 1 — `RUNNING` / `OK` / `FAIL` / `REJECTED` |
| D | QueuedAt | Pi 1 |
| E | StartedAt | Pi 1 |
| F | FinishedAt | Pi 1 |
| G | Secs | Pi 1 |
| H | Exit | Pi 1 |
| I | Output | Pi 1 (tail, truncated to 1500 chars) |
| J | Host | Pi 1 |

### Row rules — top-scan, changed 2026-09-22

**New commands go at the TOP, directly under the header.** Insert rows above
the existing ones; history sinks down the sheet.

The poller examines the **top 25 rows** (`SCAN_WINDOW`) every poll. A row that
already has a Status is **skipped**, never a stopping point — so a completed
row can never hide a pending one below it. Up to `MAX_ROWS_PER_RUN = 5`
unstamped rows execute per poll; the rest run on the next one.

- **Never clear a Status.** A cleared Status makes the row look pending, and it
  runs again. For `seed` that fences the money twice.
- **Never delete a row** — archive by copying to another tab.
- Leave **C..J empty** on rows you add.
- **Blank column A is a spacer** — skipped, and the scan keeps going. So you
  can insert three blank rows and fill them in any order.
- A verb whose **required argument is still empty is skipped**, unstamped, and
  runs on a later poll once you finish typing. Rows below it still run, so a
  row you are mid-way through typing cannot hold up the queue — the trade-off
  is that a later row may run first.

> **Where the window runs out.** Rows 2–26 are examined. Since new work goes on
> top, a pending row only falls outside that if you put a command below 25 rows
> of history. Raise `SCAN_WINDOW` if you ever need a deeper queue.

#### Why this replaced the cursor

The old design stored the last-processed row *number* on Pi 1 and scanned
downward from it. Inserting a row at the top shifted every row below it, so the
stored number pointed at the wrong row — and new commands above it were never
scanned at all. Silently. The cursor also had a cold-start rule that once
swallowed a pre-typed `git_pull`.

Nothing is stored now; column C on the sheet is the only record of what ran.

**What replaced the cursor's safety.** The cursor guaranteed a row ran once.
Now that guarantee comes from the row being **claimed** — `Status = RUNNING` is
written *before* the verb runs, and `write_back()` returns whether that write
landed. **If the claim fails, the verb does not run** and the batch stops. This
matters for `seed`: a re-run would fence the money twice. Previously
`write_back` gave up silently after three attempts and the verb ran anyway.

### The template row

A template row with its Status pre-filled is simply skipped, like any other
completed row, so it is harmless wherever it sits.

Better still, keep usage notes on a separate `Documentation` tab. The poller
opens only the tab named by `GSHEET_OPS_TAB` (default `ops`), so **any other
tab is invisible to it** and safe for notes, examples and usage text.

---

## 4. Verb reference

### Subprocess verbs (`VERBS`)

| Verb | Column B | Timeout | Does |
|------|----------|---------|------|
| `ping` | — | 20s | `date -Is` + "alive on \<host\>" |
| `status` | — | 60s | service ActiveState/SubState/NRestarts, uptime, `df -h /` |
| `uptime` | — | 30s | `uptime` + `free -h` |
| `disk` | — | 30s | `df -h` |
| `git_pull` | — | 240s | `pull --rebase --autostash`, then last 5 commits |
| `git_status` | — | 60s | `status --short --branch` |
| `git_log` | — | 60s | last 15 commits, one line each |
| `cron_check` | — | 30s | `crontab -l` |
| `svc_log` | — | 60s | last 80 `journalctl` lines for the bot service |
| `ls` | optional | 30s | `ls -la` of a **dir key** (default `jobs`) |
| `tail_log` | **required** | 60s | last 120 lines of a **log key** |

**Log keys** (`LOGS`): `stocks`, `crypto`, `45`, `ops`
`ops` is the remote-ops audit log itself — useful for debugging the channel
from the channel.

**Dir keys** (`DIRS`): `repo`, `jobs`, `bots`

Add keys to these dicts in code, **never** to the sheet.

### In-process verbs (`PY_VERBS`)

These bypass `subprocess` entirely and call `schwab_auth` / `cash_reserve`
directly.

| Verb | Column B | Does |
|------|----------|------|
| `token_status` | — | Schwab token state as JSON — state, hours/days left, expiry |
| `auth_url` | — | The tappable re-authorization link, plus instructions |
| `auth_code` | **required** | Exchanges the pasted redirect URL for new tokens |
| `reserves` | — | The reserves table — what is fenced, deployed, available |
| `seed` | **required** | Fences cash to a ticker. See below |

Token states: `OK`, `RENEW_NOW` (past day 6 of 7), `EXPIRED`, `MISSING`
(no token file), `UNKNOWN` (file exists but no issue stamp found).

#### Failure handling

`run_pyverb` never raises. Any exception from an in-process verb becomes exit 1
with the message in column I, and the row reads `FAIL`.

This matters because the row is stamped `RUNNING` *before* the verb runs. An
escaping exception used to kill the whole poll cycle and leave that row looking
handled — stuck on `RUNNING`, skipped forever, with nothing visible anywhere
except a poller that quietly died every 10 minutes. `auth_url` was the worst
case: it is what you reach for when the token has expired, which is exactly
when everything else is failing too.

It also calls `load_dotenv()` itself, so running `remote_ops.py` by hand
without sourcing `.env` first still works.

#### `seed` — fencing cash without SSH

```
seed    ACCT TICKER AMOUNT [POLICY] [TARGET]
```

| Example in column B | Meaning |
|---|---|
| `885 NOC 5000` | fence $5,000 of account 885 to NOC, `CASH_ONLY` |
| `171 MU 7606.68 TOTAL_CAPITAL 7968.10` | fence $7,606.68, and stop buying once that account's MU position reaches $7,968.10 |

`POLICY` defaults to `CASH_ONLY`. `TOTAL_CAPITAL` **requires** `TARGET` —
without it the reserve can never bind, so it is rejected rather than accepted
and silently ignored. `$` and thousands commas are tolerated.

**Guard rail.** Amounts above `REMOTE_OPS_SEED_MAX` (default `$100,000`) are
rejected with a message rather than applied. A stray digit on a phone keyboard
is the realistic failure mode here; a genuinely larger seed goes through the
CLI on Pi 1.

**The ledger stays authoritative.** This calls the same `cash_reserve.seed()`
the CLI does, recording `source=ops_sheet`. The sheet carries intent, never a
balance. There is no un-seed verb on purpose — reversing money is a
`withdraw`/`close` on Pi 1, where you can see the ledger first.

Seeding a pair that is already fenced **adds** to its balance and overwrites
its policy/target, the same as the CLI.

### Exit codes in column H

| Code | Meaning |
|------|---------|
| 0 | success |
| 1 | `auth_code` exchange raised — column I has the exception |
| 2 | bad argument, unknown key, or missing required arg |
| 124 | timed out |
| 127 | binary not found |
| other | the first non-zero exit from the verb's command list |

---

## 4b. The cron entry

```cron
# Remote ops channel — polls the ops sheet every 10min. See Documentation/remoteOpsGuide-9-7-26.md §4b
*/10 * * * * cd /home/rchak007/github/myTrading && set -a && . ./.env && set +a && flock -n /tmp/remote_ops.lock timeout 300 .venv/bin/python remote_ops.py >> /home/rchak007/.local/state/myTrading/remote_ops_cron.log 2>&1
```

Why each piece is there:

| | |
|---|---|
| `set -a && . ./.env && set +a` | **The one that is not optional.** `remote_ops.py` has no `dotenv` import — it reads `GSHEET_OPS_ID` and `REMOTE_OPS_CREDS` from the environment. Without this the job exits instantly with *"GSHEET_OPS_ID is not set"* and nothing ever runs. Do not "tidy" it away. |
| `flock -n /tmp/remote_ops.lock` | skip this cycle if the previous one is still going. Its own lock, not `jobmytrading.lock`, so ops commands do not queue behind a long signals run |
| `timeout 300` | kill a wedged verb instead of letting runs pile up |
| `>> ...cron.log 2>&1` | nothing rotates this; check its size occasionally |

`MAX_ROWS_PER_RUN = 5` inside the script already caps how much one cycle can do.

**Before trusting it, test under a bare environment** — cron has almost none,
and that is where a line like this fails:

```bash
env -i PATH=/usr/bin:/bin HOME=/home/rchak007 /bin/sh -c 'cd /home/rchak007/github/myTrading && set -a && . ./.env && set +a && flock -n /tmp/remote_ops.lock timeout 300 .venv/bin/python remote_ops.py'
```

Silence means success — an idle run prints nothing. Confirm with the audit log.

**Health check:** an `idle` event lands in `remote_ops_audit.log` every 10
minutes. Those ticks are the heartbeat; if they stop, the poller is dead.

⚠️ **Do not use `auth_url` / `auth_code` from the sheet until the `.env` defect
is fixed** (`PROJECT_PLAN.md` §4). They work in a shell where `.env` has been
sourced, but under cron `creds()` raises and `auth_url` is not wrapped in a
try/except — so it crashes the cycle *after* the row is marked `RUNNING`,
stranding it. `token_status` is unaffected.

---

## 5. Command-line flags

```bash
.venv/bin/python remote_ops.py [flag]
```

| Flag | Effect |
|------|--------|
| *(none)* | normal poll — the cron entry point |
| `--verbs` | print the allowlist and exit (no sheet access needed) |
| `--dry-run` | show what would run; execute nothing, write nothing |
| `--init` | write the header row to A1:J1 and exit |
| `--nudge` | append a re-auth prompt row if the refresh token is aging out |

`--verbs` is the quickest sanity check that a code change parsed — it needs no
credentials and no network.

---

## 6. Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `GSHEET_OPS_ID` | *(required)* | spreadsheet key |
| `GSHEET_OPS_TAB` | `ops` | worksheet name |
| `REMOTE_OPS_CREDS` | falls back to `GSHEET_CREDS`, then `/etc/myTrading/gsheets.json` | service-account JSON |
| `MYTRADING_REPO` | `~/github/myTrading` | repo for git verbs and log paths |
| `JOBS_REPO` | `~/github/jobMyTrading` | `ls jobs` |
| `BOTS_REPO` | `~/github/botsMyTrading` | `ls bots` |
| `REMOTE_OPS_STATE` | `~/.local/state/myTrading` | audit log, spilled output |
| `BOT_SERVICE` | `mytrading-bot.service` | target of `status` and `svc_log` |
| `SCHWAB_TOKENS` | see `schwab_auth.py` | token file location |

Scopes: this module uses full `spreadsheets` read/write and is **the only
writer**. `gsheet_notes.py` uses `spreadsheets.readonly` and must keep doing so.

### Schwab credentials

`schwab_auth.py` reads the same lowercase names
`schwabAPI/portfolio_snapshot.py` uses. In `~/github/myTrading/.env`:

```
app_key=...
app_secret=...
callback_url=https://127.0.0.1:8182
```

`callback_url` must match the registered app **exactly**, port included.

---

## 7. Schwab re-auth from your phone

Schwab refresh tokens are hard-capped at **7 days**. `RENEW_AT` is 6 days,
giving one day of slack.

### Verify on Pi 1 first — before touching the sheet

```bash
cd ~/github/myTrading
set -a; . ./.env; set +a
.venv/bin/python schwab_auth.py --status
.venv/bin/python schwab_auth.py --url
```

`schwab_auth.py` also takes `--code PASTED` (redirect URL or bare code) — the
local equivalent of an `auth_code` row, for when you are already on the box.

### Phone workflow

1. Add a row: `token_status` → read days left in column I.
2. Add a row: `auth_url` → tap the link that comes back in column I.
3. Sign in, approve. **The redirect to 127.0.0.1 will fail to load — that is
   expected.**
4. Tap the address bar, **Select All**, copy. You need the whole thing,
   `code=` and all.
   *Mobile Safari and Chrome both keep the failed URL in the address bar.
   Chrome hides the query string until you tap in — tap, then select all.*
5. Add a row: `auth_code` in column A, the pasted URL in column B.
6. Column I comes back with the validated expiry date.

Column B is overwritten with `(consumed)` after the exchange. The code is
single-use and short-lived anyway, but leaving it in a spreadsheet cell is
pointless risk.

### What protects you

- New tokens are proven against `trader/v1/accounts/accountNumbers` **before**
  the live file is touched. A bad exchange leaves the working `tokens.json`
  exactly as it was.
- The previous file is retained as `tokens.json.bak`; the new file is written
  `0600` via atomic rename.
- The existing JSON shape is preserved leaf-by-leaf, so a schwabdev version
  change cannot leave you with a structurally valid but unreadable token file.
- The authorization code is worthless without `app_key` + `app_secret`, which
  never leave Pi 1.

---

## 8. The day-6 nudge

`nudge(ws)` has Pi 1 ask **you** for a re-auth, with the tappable link already
waiting in the sheet. It appends a row only when the state is `RENEW_NOW`,
`EXPIRED`, `MISSING` or `UNKNOWN`; on `OK` it does nothing and returns 0.

Column A is left **blank** on the appended row, so the scan walks past it
without trying to execute it — the row is a notification, not a command.

Suggested cron on Pi 1:

```cron
30 7 * * *  cd ~/github/myTrading && .venv/bin/python remote_ops.py --nudge
```

---

## 9. Files on disk (Pi 1)

```
~/.local/state/myTrading/
├── remote_ops_audit.log      one JSON object per line, append-only, not rotated
└── remote_ops_out/           full spilled output, <stamp>_r<row>_<verb>.txt
```

Audit events: `start`, `done`, `awaiting_arg`, `claim_failed`,
`rejected`, `batch_cap`, `idle`, `writeback_failed`, `redact_failed`, `nudge`.

The audit log is never rotated by us. Read it from your phone with a
`tail_log` / `ops` row.

---

## 10. Troubleshooting

| Symptom | Cause |
|---------|-------|
| Nothing runs, no rows change | Every row in the top 25 already has a Status, or the pending row sits below the window. Check column C is genuinely empty on the row you added. |
| A row is skipped forever | Column C is non-empty. Pi 1 treats any Status as "already handled". |
| A row ran twice | Its Status was **cleared**, making it look pending again. Never clear a Status. |
| `REJECTED` in column C | Verb is not in either allowlist. Column I lists the valid ones. |
| Exit 2 on `tail_log` / `ls` | Bad key. Column I lists the valid keys. |
| `auth_code` returns exit 1 | Code expired or already used, or the URL was truncated. Re-run `auth_url` and copy the **whole** address bar. |
| `GSHEET_OPS_ID is not set` | Env missing from the systemd unit or `.env`. |
| Only 5 rows ran | `MAX_ROWS_PER_RUN`. The rest run on the next tick. |
