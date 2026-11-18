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

#### Fixed 2026-11-16 — a REJECTED row used to log its argument (`e21f4a0`)

`safe_arg()` decides what to redact by looking the verb up in `PY_VERBS`. An
**unknown** verb is not in that table, so it fell through to "not sensitive"
and the argument went into the append-only audit log verbatim.

The realistic path is a typo. `auth_cod` with a real redirect URL in column B
is rejected — and the rejection line wrote the whole URL, authorization code
included, into a file nothing rotates. A single-use short-lived code, but
written permanently to disk, which is the part that matters.

The REJECTED branch now logs `(N chars, not logged)` instead of the value. It
never needed the argument: the row is being refused *because the verb is not
real*, so nothing downstream can use it.

**The general shape is worth remembering.** A redaction policy keyed on a
lookup table is only as good as its behaviour on a miss, and the safe default
on a miss is always "redact". This one defaulted the other way, and the only
code path that reached it was the error path — the one least exercised and
least watched.

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
| `locks` | — | 30s | who holds each lock file right now. Reach for this when a job has gone quiet — `flock -n` exits **silently** by design, so a stale lock makes every later run vanish without a word |
| `pip_upgrade` | — | 300s | upgrade the venv's packages. The one subprocess verb that changes the machine |

**Log keys** (`LOGS`): `stocks`, `crypto`, `45`, `ops`
`ops` is the remote-ops audit log itself — useful for debugging the channel
from the channel.

**Dir keys** (`DIRS`): `repo`, `jobs`, `bots`

Add keys to these dicts in code, **never** to the sheet.

### In-process verbs (`PY_VERBS`)

These bypass `subprocess` entirely and call `schwab_auth` / `cash_reserve`
directly.

Grouped by what they touch. **Nothing in this table places or cancels an
order** — that is the engine's job, and the ops channel deliberately cannot
do it. The verbs that move money move an *earmark*; the verbs that advise
print text.

**Auth and health**

| Verb | Column B | Does |
|------|----------|------|
| `token_status` | — | Schwab token state as JSON — state, hours/days left, expiry |
| `auth_url` | — | The tappable re-authorization link, plus instructions |
| `auth_code` | **required** | Exchanges the pasted redirect URL for new tokens. Takes the job lock first, so it is safe at any time — see below. **Redacted** in the audit log |
| `taostats_key` | **required** | Installs the taostats API key for the CRYPTO tab's SUBNET section. Writes one file, mode 0600, and verifies it with one live call. **Redacted** |
| `run_signals` | — | Rebuild every sheet tab now instead of waiting for `:15` / `:50`. Launches detached and returns at once |

**Cash, reserves and parking**

| Verb | Column B | Does |
|------|----------|------|
| `reserves` | — | The reserves table — what is fenced, deployed, available |
| `seed` | **required** | Fences cash to a ticker. See below |
| `fence` | **required** | Marks a (ticker, account) pair as off-limits to the engine |
| `withdraw` | **required** | Returns some fenced cash to the free pool. See below |
| `close` | **required** | Zeroes a reserve entirely and deactivates its config row |
| `borrow` / `return_borrow` | **required** | Records cash borrowed from another account, and its repayment |
| `owe` / `paid_back` | **required** | The same for money owed to a person |
| `park` / `unpark` | **required** | Records reserve cash parked in a vehicle (a QQQ share, say) and its release |
| `parked` | — | Everything currently parked, by owner and vehicle |

**Orders**

| Verb | Column B | Does |
|------|----------|------|
| `reconcile` | — | Ask Schwab what became of every `SUBMITTED` order and record it — `FILLED` / `CANCELLED` / `REJECTED` / `EXPIRED`, with filled quantity and average price. A chained row arms once its parent reads `FILLED` |
| `apply_fills` | — | Fold recorded fills back into the reserve ledger |
| `tidy_orders` | — | Normalise the Orders tab in place |
| `orders_columns` | — | Report which column each Orders header currently sits in |

**Options** — see §4c

| Verb | Column B | Does |
|------|----------|------|
| `options` | — | Rebuild the OPTIONS tab now |
| `cc` | **required** | What covered call to write on these tickers today |
| `csp` | **required** | The same for a cash-secured put |

**Notes, attention and due dates**

| Verb | Column B | Does |
|------|----------|------|
| `note` / `note_replace` / `note_delete` | **required** | Append, replace or remove a ticker note. Append-only underneath — a delete is a new row, never an erasure |
| `notes` | — | Every current note |
| `attend` | **required** | Stamp a ticker as looked at today |
| `attend_list` | — | When each ticker was last attended |
| `due` | **required** | Set a due date. **The date goes LAST**, so one date updates any number of tickers: `due NVDA AMD MU 12/01`. Accepts `2026-12-01`, `12/01` (the *next* one), `30d`, `4w`, `3m`, `today`, `tomorrow`, `eom`, and `-` to clear. All-or-nothing: a typo in the fourth symbol leaves the first three undated |
| `due_list` | — | Every due date, without opening the DUE tab |

**Documentation**

| Verb | Column B | Does |
|------|----------|------|
| `verbs_doc` | — | Write the `VERBS-howToUse` tab from the live allowlists |
| `dash_doc` | — | Write the Dashboard's own how-to-read tab |
| `orders_doc` | — | The same for the Orders tab |

> This table is maintained by hand and the allowlists are not. `verbs_doc`
> generates its tab **from the code**, so if the two ever disagree, run it
> and believe the tab.

Token states: `OK`, `RENEW_NOW` (past day 6 of 7), `EXPIRED`, `MISSING`
(no token file), `UNKNOWN` (file exists but no issue stamp found).

#### Failure handling

`run_pyverb` never raises. Any exception from an in-process verb becomes exit 1
with the message in column I, and the row reads `FAIL`.

This matters because the row is stamped `RUNNING` *before* the verb runs. An
escaping exception used to kill the whole poll cycle and leave that row looking
handled — stuck on `RUNNING`, skipped forever, with nothing visible anywhere
except a poller that quietly died every cycle. `auth_url` was the worst
case: it is what you reach for when the token has expired, which is exactly
when everything else is failing too.

It also calls `load_dotenv()` itself, so running `remote_ops.py` by hand
without sourcing `.env` first still works.

#### `withdraw` / `close` — releasing a reserve

```
withdraw  ACCT TICKER AMOUNT      e.g.  withdraw  171 GOOG 5000
close     ACCT TICKER             e.g.  close     171 SPY
```

`withdraw` returns that much fenced cash to the account's free pool.
`close` zeroes the reserve entirely and deactivates the config row; the ledger
history is kept either way, because corrections are new rows and nothing is
ever rewritten.

**The position is untouched by both.** These move the *earmark*, not the
shares.

**Added 2026-10-08 because the channel could only ever ADD.** Account 171 was
over-fenced by $6,288 — $36,136 reserved against $29,848 of cash — and there
was no way to release any of it without SSH, which is exactly what the ops
sheet exists to avoid. `withdraw` was in `cash_reserve.py`'s CLI the whole
time and had simply never been exposed.

Both only ever REDUCE a reserve. Neither moves money between accounts, and
neither can place an order.

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
*/1 * * * * cd /home/rchak007/github/myTrading && set -a && . ./.env && set +a && flock -n /tmp/remote_ops.lock timeout 300 .venv/bin/python remote_ops.py >> /home/rchak007/.local/state/myTrading/remote_ops_cron.log 2>&1
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

## 4c. The options verbs

Added 2026-11-17. Chakravarti: *"give me a verb that i can say hey for MSTR
tell me covered call what is best way to do today?"*

```
cc   TICKER[@basis][xN] ...   e.g.  cc  MSTR
csp  TICKER[@basis][xN] ...   e.g.  csp NVDA MRVL
options                       rebuild the OPTIONS tab now
```

**Recommendation only.** Neither verb can place an order; there is no code
path from here to a submission. Read the answer, decide, enter it at your
broker yourself.

### A ticker Schwab cannot see still gets an answer

Corrected 2026-11-17. Chakravarti: *"Charles Schwab I don't have GOOG or
CBRS .. but when I ask recommendations I need it"*.

The first version refused with "you have 0 shares". That is this pipeline
reporting **its own blind spot** as though it were a fact about the
portfolio — the shares are at SoFi, and the premium, the greeks and the
right strike do not depend on where they are custodied. Only the *cover*
does, and that is his to confirm.

So the ladder is printed either way, sizing assumes one contract, and the
assumption is stated:

> ⚠  Schwab shows 0 shares of GOOG, so it cannot check your cover. Sizing
> assumes 1 contract = 100 shares. The strike and premium are right either
> way; the share count is yours to confirm.

**Every covered-call line states the cover it needs**, in the order line
itself:

```
→ TODAY: SELL 1 GOOG 2026-11-28 $360.00C — LIMIT $2.35 ($235)  — needs 100 shares
```

Added 2026-11-18. A demo printed `0 sh` and `SELL 2 GOOG` four lines apart
and left the reader to reconcile them — Chakravarti: *"I don't have 200
shares of Google lol. Only 100 in SoFi"*. A contract count is abstract;
"needs 200 shares" is something you can check against your own account in
one second, so it belongs where the order is read and not in a note above
it. Puts say `needs $12,000 cash` for the same reason.

Where Schwab *can* see the shares, `xN` is capped at what you actually hold.
Where it cannot, nothing can cap it — so the implied share count is printed
instead, and the warning says the size came from you.

Two optional suffixes supply what Schwab cannot:

| Written | Means |
|---|---|
| `cc GOOG` | one contract, no basis, no assignment maths |
| `cc GOOG@175` | cost basis $175 — brings back what assignment pays |
| `cc GOOGx2` | size it at two contracts |
| `cc GOOG@175x2` | both |

A typed `@basis` **wins over Schwab's**. If you are telling the verb what
you paid, you know something it does not.

### What `cc` answers with

```
MSTR  $96.50  ·  IV 58% (rank 61%)  ·  100 sh  ·  basis $118.00

COVERED CALL LADDER — closest strike to Δ0.15 at each expiry,
never below your $118.00 basis

   EXPIRY       DTE     STRIKE   DELTA      MID    PER CT    %/YR       OI
   2026-11-21     4    $142.00    0.14    $0.80       $80     56%    2,000
   2026-11-28    11    $145.00    0.16    $1.85      $185     47%    2,000
   2026-12-19    32    $155.00    0.16    $4.40      $440     38%    9,000

→ TODAY: SELL 1 MSTR 2026-11-28 $145.00C — LIMIT $1.85 ($185)
   Δ0.16  ·  11 days  ·  47%/yr  ·  +10.7% from spot
   Bid $1.80 / ask $1.90 — work the mid, do not hit the bid.

   If assigned: you sell 100 shares at $145.00 — $2,700 on the shares
   (+23% on basis) plus $185 premium.
```

One row per expiry, not one row per strike, because the question being asked
is *which date* — and the **%/YR column is the whole point**. The $4.40 on
the 32-day is the biggest number on the board and the worst trade on it.
Premium per day of capital tied up is the only way two expiries compare.

### Two rules that are hard limits, not preferences

**Nothing in the money is ever suggested to open.** That is a decision to be
assigned, dressed up as income, and it is a hard veto.

**Below cost basis is a WARNING, not a veto — corrected 2026-11-17.** The
first version refused outright. Chakravarti: *"this means MSTR i can never
do covered call since i am holding this when its down"*. He was right, and
the rule was worst exactly where it mattered: a position under water is when
the income is most wanted, and the floor silenced the whole board there.

Every strike offered is still **out of the money** — above spot. It is only
below what he *paid*. Those are different claims and conflating them cost
him the trade. So the ladder marks below-basis rungs with `⚠`, the pick
prefers a clear strike when one exists in his window, and when none does it
recommends the best out-of-the-money rung anyway with the arithmetic spelled
out:

```
   ⚠  This strike is $13 UNDER your $118 basis. It is still out of the
      money — you only get assigned if the stock rallies 9% from here, and
      at that point you are down $1,300 against basis instead of today's
      larger unrealised loss, plus $145 premium. Net −$1,155. Your call,
      not the tab's.
      Clear of basis: 2026-12-19 $120.00C at $210 (32d, 25%/yr) — longer
      than your usual 7-14 days, but assignment there is a gain.
```

That last line matters as much as the warning. "Take the loss or earn
nothing" is a false choice when a clear strike exists one expiry further
out, and the old behaviour hid it.

### The defaults are his, not ours

Δ0.15, 7–14 days, close at 85% of premium captured, roll in gamma week —
the programme stated on 2026-11-17, and the reason the 4-day at 56%/yr loses
to the 11-day at 47% in the example above. Override with
`OPTIONS_TARGET_DELTA`, `OPTIONS_DTE_LO`, `OPTIONS_DTE_HI`.

### It places nothing

Both verbs are read-only by construction: fetch a chain, print a ladder. An
ops verb that both advised and acted would be one typo away from writing
calls against shares meant to be held, so sizing and timing stay manual.

### The OPTIONS tab

`options` rebuilds it; `run_signals` also does, as the last tab written.

**Columns A–G are yours.** `Ticker · Type · Strike · Premium · Expiry · Qty ·
Acct` — type SoFi positions and anything else no API reaches here. Column H
onward is regenerated every cycle: DTE, mark, P/L, kept %, delta, gamma,
theta/day, IV, IV rank, distance to strike, verdict, action, roll candidate
and its net credit.

Leave strike and expiry **blank** and the row becomes a question — the tab
suggests a contract instead of grading one.

**Do not type Schwab positions.** They are already reported and already
parsed; a typed copy drifts and gets counted twice. If you type one anyway,
the merge drops it and says so in the log.

**Why it cannot eat your typed rows.** The tab is both input and output,
which is the one genuinely dangerous thing about it. Typed rows are mirrored
to an append-only ledger (`options_manual.csv`) on every successful read, and
the writer **refuses to write at all** if the read failed. Worst case is a
stale tab, never a blank one.

**A hand-typed short call is never reported NAKED.** Schwab cannot see SoFi
shares, and crying wolf on those rows would train the flag to be ignored on
the Schwab rows where it means something real.

### Where the data comes from

One Schwab `/marketdata/v1/chains` call per underlying, cached 15 minutes.
The contract you are short and every contract you could roll into arrive in
the same response, so the roll ladder costs nothing extra. Greeks are
Schwab's own — nothing here prices an option.

On failure it serves the stale cache and prints its age rather than blanking
the row.

> **Unverified as of 2026-11-17.** This is the first Schwab *market-data*
> call this repo has ever made; everything before it was accounts, positions,
> orders and transactions. Method name, entitlement and IV units are all
> unconfirmed against the live API. Run `options` once on Pi 1 and the log
> names which of the three failed. See PROJECT_PLAN §10.

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
| `OPTIONS_TARGET_DELTA` | `0.15` | delta the `cc` / `csp` ladder aims at |
| `OPTIONS_DTE_LO` / `_HI` | `7` / `14` | the expiry window a recommendation must fall in |
| `OPTIONS_LADDER_DTE` | `60` | how far out the ladder *shows*, as against recommends |
| `OPTION_CHAIN_TTL_MIN` | `15` | chain cache lifetime; stops three reminder runs becoming three times the calls |
| `OPTIONS_WATCH` | *(empty)* | extra underlyings to pull a chain for, comma separated |
| `GSHEET_OPTIONS_TAB` | `OPTIONS` | worksheet name |
| `OPTIONS_MANUAL_PATH` | `$MYTRADING_STATE_DIR/options_manual.csv` | the mirror of your hand-typed rows |

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
| `cc` / `options` says *"no option chain available"* | The chain call failed. Column I carries the reason: a method-name miss, a 403 (market-data entitlement not granted on the developer app), or a throttle. See `remoteOpsGuide` §4c and PROJECT_PLAN §10. |
| `cc` recommends nothing at all | Every liquid strike sits below your cost basis, and suggesting one is refused rather than risking assignment at a loss. The message says so outright. |
| OPTIONS tab lost a hand-typed row | It should not be possible — the writer refuses to write when the read failed, and typed rows mirror to `options_manual.csv` on every read. Check that ledger; the most recent batch is the last good state. |
| OPTIONS tab shows a position twice | A Schwab position was also typed by hand. Delete the typed copy — the merge normally drops it and logs that it did. |
| `IV_Rank` is blank everywhere | Fewer than 20 daily readings exist for that ticker. It fills in on its own; a rank off four readings would be noise with a `%` on it. |
