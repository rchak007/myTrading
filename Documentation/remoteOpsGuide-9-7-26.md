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

`safe_arg()` masks arguments in the audit log and `--dry-run` output. It keys
off the rule — *in-process verbs are the only ones taking a free-form
argument* — rather than off the name `auth_code`, so any future `PY_VERBS`
entry with `needs_arg=True` is covered the day it is added. Dict-key
arguments (`stocks`, `repo`) still log verbatim, because they are safe and
useful.

Known residual: a **typo'd** verb (`auth_cod`) with a real URL in column B is
rejected, and the rejection audit line logs the URL in plaintext. The code is
single-use and short-lived, so this is minor, but worth knowing.

---

## 2. How the loop works

Each cron run:

1. Opens the sheet (`GSHEET_OPS_ID`, tab `GSHEET_OPS_TAB`).
2. Reads the **cursor** from `~/.local/state/myTrading/remote_ops.state` —
   the last row already dealt with.
3. Walks rows `cursor+1` → bottom of sheet.
   - Blank column A (spacer row) → skip, advance cursor.
   - Non-empty column C (already handled) → skip, advance cursor.
4. For a live row: writes `RUNNING` into C **first**, so a slow verb shows up
   on your phone and a crash mid-flight leaves a visible marker instead of a
   silently re-runnable blank row.
5. Runs the verb, writes C..J back in a single range call.
6. Advances the cursor, writes it atomically via a `.tmp` + rename.

**Batch cap:** `MAX_ROWS_PER_RUN = 5`. A pasted wall of rows cannot stampede
the box; the cursor is *not* advanced past the cap, so the remainder runs on
the next tick.

**Cold start:** with no state file, the first run **adopts the bottom row and
executes nothing**. That stops a fresh install from replaying months of
history. `--catchup` overrides it once (sets cursor to row 1).

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

### Row rules

- **Never delete a row.** Rows are append-only. Deleting one shifts every row
  number below it and the cursor silently skips work. Archive by copying to
  another tab.
- Leave **C..J empty** on rows you add. A non-empty Status means "already
  handled" and the row is skipped forever.
- Blank column A is a spacer — the cursor walks straight past it. This is what
  makes the `--nudge` row inert.

### Current seed (rows 1–2, real work starts row 3)

Row 2 is a template that never executes, protected two ways: column C is
pre-filled (so the row is skipped), and cold start adopts the bottom row
anyway. **Do not clear C2.**

After pasting the seed, pin the cursor explicitly rather than trusting cold
start:

```bash
.venv/bin/python remote_ops.py --reset-cursor 2
```

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

These bypass `subprocess` entirely and call `schwab_auth` directly.

| Verb | Column B | Does |
|------|----------|------|
| `token_status` | — | Schwab token state as JSON — state, hours/days left, expiry |
| `auth_url` | — | The tappable re-authorization link, plus instructions |
| `auth_code` | **required** | Exchanges the pasted redirect URL for new tokens |

Token states: `OK`, `RENEW_NOW` (past day 6 of 7), `EXPIRED`, `MISSING`
(no token file), `UNKNOWN` (file exists but no issue stamp found).

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

## 5. Command-line flags

```bash
.venv/bin/python remote_ops.py [flag]
```

| Flag | Effect |
|------|--------|
| *(none)* | normal poll — the cron entry point |
| `--verbs` | print the allowlist and exit (no sheet access needed) |
| `--dry-run` | show what would run; execute nothing, write nothing |
| `--catchup` | on cold start, actually run the pending backlog |
| `--init` | write the header row to A1:J1 and exit |
| `--nudge` | append a re-auth prompt row if the refresh token is aging out |
| `--reset-cursor ROW` | force the cursor; rows at or below ROW are ignored |

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
| `REMOTE_OPS_STATE` | `~/.local/state/myTrading` | cursor, audit log, spilled output |
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

Column A is left **blank** on the appended row, so the cursor walks past it
without trying to execute it — the row is a notification, not a command.

Suggested cron on Pi 1:

```cron
30 7 * * *  cd ~/github/myTrading && .venv/bin/python remote_ops.py --nudge
```

---

## 9. Files on disk (Pi 1)

```
~/.local/state/myTrading/
├── remote_ops.state          cursor — one integer, atomically replaced
├── remote_ops_audit.log      one JSON object per line, append-only, not rotated
└── remote_ops_out/           full spilled output, <stamp>_r<row>_<verb>.txt
```

Audit events: `cold_start_adopt`, `cursor_reset`, `start`, `done`,
`rejected`, `batch_cap`, `idle`, `writeback_failed`, `redact_failed`, `nudge`.

The audit log is never rotated by us. Read it from your phone with a
`tail_log` / `ops` row.

---

## 10. Troubleshooting

| Symptom | Cause |
|---------|-------|
| Nothing runs, no rows change | Cold start adopted the bottom row. Add a *new* row below, or `--reset-cursor`. |
| A row is skipped forever | Column C is non-empty. Pi 1 treats any Status as "already handled". |
| Rows run out of order / get skipped | A row was **deleted**. Row numbers shifted under the cursor. Never delete. |
| `REJECTED` in column C | Verb is not in either allowlist. Column I lists the valid ones. |
| Exit 2 on `tail_log` / `ls` | Bad key. Column I lists the valid keys. |
| `auth_code` returns exit 1 | Code expired or already used, or the URL was truncated. Re-run `auth_url` and copy the **whole** address bar. |
| `GSHEET_OPS_ID is not set` | Env missing from the systemd unit or `.env`. |
| Only 5 rows ran | `MAX_ROWS_PER_RUN`. The rest run on the next tick. |
