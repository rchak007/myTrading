# Schwab Order Execution Sheet — Design Document

**Status:** Design complete, not implemented
**Target implementer:** Claude Code on Pi 2 (`~/github/myTrading`), deployed to Pi 1
**Prerequisite:** none of this ships until Section 6 (security model) is implemented in full

---

## 1. Purpose

Let Chakravarti specify, in a Google Sheet, an intent of the form:

> "Buy 30 shares of AVGO in account `...431` if it closes above 245.00, good until Oct 31."

…and have the system place that order with Charles Schwab automatically when the
condition is met, with hard guardrails, a full audit trail, and a kill switch.

This closes the last manual gap in `myTrading`: signals are generated
automatically, positions and cash are reported automatically, but every order is
still placed by hand in the Schwab UI.

---

## 2. Non-goals

Explicitly out of scope for v1. Do not build these.

- **No signal-driven auto-trading.** The engine never invents an order from
  `stocks_signals.csv`. A human writes every row. The signals stack informs the
  human; it does not feed the engine.
- **No options, no margin, no short selling.** Equity `BUY` and `SELL` only, cash
  account semantics.
- **No order modification.** An armed row is immutable. To change it: cancel and
  write a new row.
- **No stop-loss / bracket / OCO orders.** Single-leg only.
- **No crypto.** Crypto execution stays in the existing on-chain path.
- **No intrabar triggering in v1.** See Section 5.

---

## 3. Architecture placement

### 3.1 Where the engine runs

The engine runs on **Pi 1** (air-gapped execution environment). This is the only
machine that holds `tokens.json` and it must stay that way.

```
                    Google Sheet  ("Orders" tab)
                    ┌──────────────────────────┐
   Chakravarti ───► │ intent rows + token      │ ◄─── engine writes status back
   (writes intent)  └──────────────────────────┘      (Pi 1, outbound only)
                                 ▲
                                 │ poll every N min (outbound HTTPS only)
                                 │
                         ┌───────┴────────┐
                         │     Pi 1       │  tokens.json, HMAC secret,
                         │  order_exec.py │  order_ledger.csv (source of truth)
                         └───────┬────────┘
                                 │ Schwab API (place order, poll status)
                                 ▼
                          Charles Schwab
```

Pi 1 keeps its existing posture: **outbound polling only, no inbound ports, no
SSH exposure, no git write credentials.** The sheet is a mailbox, not a control
plane.

### 3.2 The sheet is untrusted input

This is the single most important design decision in this document.

Anyone who gains write access to the Google Sheet — a leaked service-account
key, a mis-shared link, a compromised Google account — must **not** be able to
cause an order to be placed. Therefore:

- The sheet carries **intent**, not **authority**.
- Authority comes from a per-row HMAC token that can only be generated on a
  machine holding a secret Pi 1 also holds (Section 6.2).
- The **local append-only ledger on Pi 1 is the source of truth** for state, not
  the sheet. The sheet's `Status` column is a *mirror written by the engine*, and
  the engine never reads it back as fact.

Treat every field the engine reads from the sheet as hostile until the token
verifies.

### 3.3 Relationship to `remote_ops.py`

`remote_ops.py` already implements the Sheets-polling command-channel pattern:
allowlisted verbs, no `shell=True`, state cursor for replay prevention,
per-run audit log, systemd timer + service under `flock`.

**Reuse the pattern, do not reuse the channel.** The order engine gets its own
tab, its own service account, its own systemd unit, and its own state cursor.
Mixing order placement into the general ops channel would mean any `remote_ops`
verb compromise becomes an order-placement compromise.

### 3.4 Module boundaries (existing house rules apply)

Per `myTrading` architecture rules, the new module owns *only* order-intent logic.
Everything else is injected by the caller:

| Concern | Comes from |
|---|---|
| paths / `JOB_DIR` / `MYTRADING_DIR` | `jobStocksSignals` |
| `log()` | `jobStocksSignals.log` |
| Schwab client | `jobStocksSignals.get_schwab_client()` |
| raw `schwabdev` client unwrap | `stocks_orders._raw_client()` — one unwrapper, reused |
| open-order reconciliation | `stocks_orders.build_orders_table()` |
| cash availability | `stocks_cash.build_cash_table()` + `cash_reserve.py` |
| HTML rendering | `jobStocksSignals.build_html_table` |

Do not redefine any of the above. Do not import `schwabdev` directly.

---

## 4. The sheet

### 4.1 Tab layout

New spreadsheet (**not** the existing trade-journal file), tab named `Orders`.
A second tab `AuditLog` is append-only, written by the engine.

Columns fall into two regions. **Human region** is written by Chakravarti and is
covered by the HMAC. **Engine region** is written only by the engine; edits there
are ignored.

#### Human region (covered by the token)

| Column | Type | Notes |
|---|---|---|
| `Row_ID` | string | Unique, human-assigned, e.g. `2026-10-14-AVGO-01`. Never reused. |
| `Account` | string | Last 3 digits only, e.g. `431`. Must be in the account allowlist. |
| `Ticker` | string | Uppercase. Must be in the ticker allowlist. |
| `Side` | enum | `BUY` \| `SELL` |
| `Trigger_Type` | enum | `CLOSE_ABOVE` \| `CLOSE_BELOW` |
| `Trigger_Price` | float | > 0 |
| `Bar` | enum | `DAILY` (v1 only — see 5.2) |
| `Qty` | int | Shares. Mutually exclusive with `Notional`. |
| `Notional` | float | USD. Engine converts to whole shares at submit time. |
| `Order_Type` | enum | `LIMIT` \| `MARKET` (MARKET disabled by default, see 6.6) |
| `Limit_Offset_Pct` | float | For `LIMIT`: how far through the trigger to place the limit. e.g. `0.5` on a BUY → limit = trigger × 1.005 |
| `TIF` | enum | `DAY` \| `GTC` |
| `Expires_On` | date | `YYYY-MM-DD`. Row auto-expires at this date's close. Max 90 days out. |
| `Confirm_Token` | string | HMAC, pasted from `arm_order.py`. See 6.2. |

#### Engine region (written by engine, read-only to humans)

| Column | Notes |
|---|---|
| `Status` | Mirror of ledger state. See Section 5. |
| `Engine_Note` | Last reason string — why rejected, why skipped, why not triggered. |
| `Last_Checked` | ISO timestamp, PST. |
| `Triggered_At` | Timestamp of the bar close that satisfied the condition. |
| `Trigger_Close` | The actual closing price that fired it. |
| `Submitted_At` | Timestamp the Schwab call returned success. |
| `Schwab_Order_ID` | For joining to `stocks_orders.csv`. |
| `Filled_Qty` / `Fill_Price` | From reconciliation. |
| `Terminal_At` | Timestamp of entry into a terminal state. |

### 4.2 Row lifecycle rules

- A row whose human-region fields change after arming will **fail token
  verification** and go to `VOID` with a note. This is the intended way to cancel:
  edit any intent cell and the row dies at the next poll.
- `Row_ID` collisions with any ledger entry, in any state, are rejected
  (`DUPLICATE_ROW_ID`). This is the replay defence.
- Blank rows and rows with `Status` already terminal are skipped without work.

---

## 5. State machine

```
                    ┌──────────────────────────────────────────┐
                    │                                          ▼
  (new row) ──► ARMED ──► TRIGGERED ──► SUBMITTED ──► FILLED
                 │  │         │              │    └──► PARTIAL ──► FILLED
                 │  │         │              │    └──► REJECTED
                 │  │         │              └───────► CANCELLED
                 │  │         └──► BLOCKED ──► (retry next cycle, or EXPIRED)
                 │  └──► VOID          (token mismatch / validation failure)
                 └─────► EXPIRED       (past Expires_On, never triggered)
```

### 5.1 Transitions

| From | To | Condition |
|---|---|---|
| — | `ARMED` | New `Row_ID`, token verifies, all validation passes |
| — | `VOID` | Token fails, allowlist fails, schema fails, duplicate `Row_ID` |
| `ARMED` | `TRIGGERED` | Completed bar's **close** satisfies `Trigger_Type` vs `Trigger_Price` |
| `ARMED` | `EXPIRED` | Now > `Expires_On` close |
| `ARMED` | `VOID` | Human region changed (token no longer verifies) |
| `TRIGGERED` | `BLOCKED` | A pre-submit guard failed (cash, cap, kill switch, market closed) |
| `TRIGGERED` | `SUBMITTED` | Schwab accepted the order; `Schwab_Order_ID` captured |
| `BLOCKED` | `TRIGGERED` | Guard cleared on a later cycle (retry) |
| `BLOCKED` | `EXPIRED` | Still blocked past `Expires_On` |
| `SUBMITTED` | `FILLED` / `PARTIAL` / `REJECTED` / `CANCELLED` | Reconciliation against Schwab |

Terminal: `FILLED`, `REJECTED`, `CANCELLED`, `EXPIRED`, `VOID`.
A terminal row is never re-evaluated. Ever.

### 5.2 Trigger evaluation — bar-boundary close confirmation

**The engine never triggers on an intrabar price touch.** A trigger fires only on
the close of a completed bar. Rationale: intrabar wicks produce false triggers,
and the whole HHLL/pivot framework in this system is already close-based
(wicks never break levels). The execution layer must not contradict the analysis
layer.

v1 supports `Bar = DAILY` only.

- Evaluation runs once per trading day, after the daily bar is final.
- "Final" means: **after 13:15 PT** (15 minutes past the 13:00 PT US equity close)
  to allow the data source to settle. Do not evaluate before this.
- A daily-close trigger therefore submits **the next trading morning**, not the
  same day. This is by design and must be stated in the runbook — Chakravarti
  should not be surprised by it.
- Price source for the bar close: **Schwab price history API** (the token is
  already on Pi 1). Do not use yfinance here — the T-Mobile Inseego hotspot
  blocks Cloudflare and yfinance goes down with it. A failed price fetch must
  leave the row `ARMED`, never advance it.
- Cross-check: if the Schwab close and the last `stocks_signals.csv` close for
  that ticker differ by more than 2%, do **not** trigger; log
  `PRICE_DISAGREEMENT` and leave `ARMED`.

`HOURLY` and `INTRADAY_TOUCH` are deliberately deferred. Add them only after
v1 has run clean for a full quarter.

---

## 6. Security model

This section is the gate. Nothing order-capable is wired up until every control
below exists and has a test.

### 6.1 Threat model

| Threat | Control |
|---|---|
| Sheet write access is compromised | HMAC token (6.2) — attacker cannot forge a valid row |
| Order-sheet service account key leaks | Key is scoped to one file; and it alone is insufficient (6.2) |
| Replay of an old valid row | `Row_ID` uniqueness against the ledger + state cursor (6.3) |
| Engine bug loops and submits repeatedly | Idempotency key + per-day submission cap + `flock` (6.4, 6.5) |
| Fat-finger: 3000 shares instead of 300 | Notional caps, per-order and per-day (6.5) |
| Something is going wrong, need to stop now | Kill switch file (6.7) |
| Schwab token expiry mid-flight | Fail closed; row stays `TRIGGERED`, retries next cycle |

### 6.2 Per-row confirmation token (the core control)

A secret `ORDER_HMAC_SECRET` (32 random bytes) lives in
`/etc/myTrading/order_hmac.key`, mode `0400`, root-owned, **on Pi 1 and on the
Dell only**. It is never in git, never in the sheet, never in Sheets secrets,
never in Streamlit secrets.

Token generation, on the Dell, via a new CLI `arm_order.py`:

```
$ python3 arm_order.py --row-id 2026-10-14-AVGO-01 --account 431 \
      --ticker AVGO --side BUY --trigger-type CLOSE_ABOVE --trigger-price 245.00 \
      --bar DAILY --qty 30 --order-type LIMIT --limit-offset-pct 0.5 \
      --tif DAY --expires-on 2026-10-31

  Canonical:  2026-10-14-AVGO-01|431|AVGO|BUY|CLOSE_ABOVE|245.00|DAILY|30||LIMIT|0.5|DAY|2026-10-31
  Confirm_Token:  9f3a7c21e8b0

  Paste this row into the Orders tab. Any edit to an intent cell invalidates it.
```

Token = first 12 hex chars of `HMAC-SHA256(secret, canonical_string)`.

Canonical string rules — the engine must build this **identically** or every row
fails:

- Pipe-delimited, fields in the fixed order shown above.
- All strings uppercased and stripped, except `Row_ID` and `Expires_On`.
- Floats formatted to exactly 2 decimal places.
- Empty optional fields render as an empty segment (note the `||` above where
  `Notional` is unset).

Write this canonicalizer **once**, in the shared module, and import it into both
`arm_order.py` and the engine. Two implementations will drift and every row will
`VOID`. A unit test must assert round-trip agreement.

### 6.3 Replay prevention

- Ledger holds every `Row_ID` ever seen, including terminal ones.
- Any sheet row whose `Row_ID` matches a ledger entry in a **terminal** state is
  skipped silently (it is just a historical row still sitting in the sheet).
- Any sheet row whose `Row_ID` matches a ledger entry in a **live** state
  (`ARMED`/`TRIGGERED`/`BLOCKED`/`SUBMITTED`) but whose token differs from the
  ledger's recorded token → `VOID`, note `INTENT_MUTATED`.
- Reuse of a `Row_ID` for a genuinely new intent is not supported. Use a new ID.

### 6.4 Idempotency

Every submission carries a deterministic idempotency key:
`sha256(Row_ID + Confirm_Token)[:16]`, recorded in the ledger **before** the
Schwab call is made (write-ahead).

On startup, any ledger entry with `submit_attempted` but no `Schwab_Order_ID`
is a **crash-during-submit**. The engine must not retry blindly. It must:

1. Fetch open + recent orders via `stocks_orders.build_orders_table(..., open_only=False, days_back=2)`.
2. Look for a matching ticker/side/qty/account entered within the attempt window.
3. If found → adopt that `Schwab_Order_ID`, move to `SUBMITTED`.
4. If not found → move to `BLOCKED` with `MANUAL_REVIEW_REQUIRED` and **stop
   touching that row**. A human resolves it. Never auto-retry a submit whose
   outcome is unknown.

### 6.5 Caps

Hard-coded defaults in `order_exec_config.py`, all overridable only by editing
code on the Pi (not by the sheet):

```python
MAX_NOTIONAL_PER_ORDER   = 2_500.00     # start at 100.00 for the first live weeks
MAX_NOTIONAL_PER_DAY     = 5_000.00     # aggregate across all rows
MAX_SUBMISSIONS_PER_DAY  = 3
MAX_ARMED_ROWS           = 25
TICKER_ALLOWLIST         = {...}        # subset of STOCK_TICKERS, hand-curated
ACCOUNT_ALLOWLIST        = {"431", "482"}
MAX_EXPIRY_DAYS          = 90
ALLOW_MARKET_ORDERS      = False
```

A cap breach → `BLOCKED`, not `VOID`. Caps reset at midnight PT and the daily
counters live in the ledger, not in memory.

### 6.6 Order type

`MARKET` is disabled by default. A market order on a thin open after an
overnight gap is exactly how a "buy above 245" turns into a fill at 261.

`LIMIT` construction:
- BUY: `limit = Trigger_Price × (1 + Limit_Offset_Pct/100)`, rounded to 2dp
- SELL: `limit = Trigger_Price × (1 - Limit_Offset_Pct/100)`
- Sanity gate: if the current quote is already more than 5% through the limit in
  the adverse direction, → `BLOCKED`, note `GAPPED_THROUGH`. The setup that was
  intended no longer exists; a human should look at it.

### 6.7 Kill switch

Presence of `/etc/myTrading/TRADING_DISABLED` (any content) blocks **all**
submissions immediately. Triggered rows go to `BLOCKED` with `KILL_SWITCH`, and
stay recoverable — removing the file resumes them on the next cycle.

Add `disable_trading` / `enable_trading` / `trading_status` as allowlisted verbs
in `remote_ops.py`, so the switch can be thrown from the phone via the existing
ops channel without SSH.

Additionally, the engine hard-fails closed if:
- `tokens.json` is missing, unreadable, or the Schwab client fails to build
- The ledger file is missing or unparseable
- The HMAC key file is missing or has wrong permissions
- The system clock is more than 5 minutes off NTP

### 6.8 Credentials

A **new, dedicated service account** for the orders sheet. Do not reuse
`/etc/myTrading/gsheets.json` (that one is the read-only journal identity and is
also present in Streamlit Cloud secrets — an entirely different trust boundary).

- Key at `/etc/myTrading/gsheets_orders.json`, mode `0400`.
- Shared as **Editor on exactly one spreadsheet** via Drive ACL. Server-side ACL
  is the real boundary; the OAuth scope is client-side only and is not a security
  control.
- This account must **never** appear in Streamlit secrets or in any repo.
- The dashboard does not read this sheet. If order status needs to surface in
  Streamlit, the engine writes `orders_exec.csv` into `jobMyTrading` and
  `gitpush.py` carries it, same as every other output.

---

## 7. Cash integration

Before submitting a BUY, the engine must confirm the money is actually there and
not already spoken for.

```
available = stocks_cash.Cash_After_Open_Orders   (for that account)
          - cash_reserve.committed_for_other_tickers(account)
```

- `Cash_After_Open_Orders` already nets out open BUY legs, so working orders are
  not double-counted.
- `cash_reserve.py` fencing is consulted per `(Account, Ticker)`. A submission
  that would exceed the ticker's reserve → `BLOCKED`, note `RESERVE_EXCEEDED`.
- On a successful submit, append a row to `reserve_ledger.csv` so the commitment
  is visible to the rest of the system. Follow the existing append-only
  convention; do not rewrite history in that file.
- **Open decision, must be resolved before coding this section:** whether
  `cash_reserve` means "cash committed to a ticker" or "total capital allocated
  to a ticker". The latter needs a `Target_Capital` column in
  `reserves_config.csv`. The engine's guard logic differs materially between the
  two. Ask before implementing.

SELL orders: verify the position exists and `Qty ≤ held shares` from the holdings
path in `jobStocksSignals.fetch_schwab_holdings()`. Never sell more than held —
that is a short, and shorts are out of scope.

---

## 8. Files and state

| Path | Owner | Purpose |
|---|---|---|
| `~/github/myTrading/order_exec.py` | new | The engine. Intent parsing, state machine, submission. |
| `~/github/myTrading/order_exec_config.py` | new | Caps, allowlists, feature flags. |
| `~/github/myTrading/order_canonical.py` | new | Canonical-string + HMAC. Shared by engine and CLI. |
| `~/github/myTrading/arm_order.py` | new | CLI token generator (runs on Dell). |
| `~/.local/state/myTrading/order_ledger.csv` | engine | **Source of truth.** Append-only. |
| `~/.local/state/myTrading/order_cursor.json` | engine | Poll cursor + daily counters. |
| `/etc/myTrading/order_hmac.key` | manual | 32 random bytes, `0400`. |
| `/etc/myTrading/gsheets_orders.json` | manual | Dedicated service account. |
| `/etc/myTrading/TRADING_DISABLED` | manual/ops | Kill switch. |
| `~/github/jobMyTrading/orders_exec.csv` | engine | Dashboard mirror, pushed by `gitpush.py`. |
| `~/.local/state/myTrading/order_audit.log` | engine | One JSON line per decision. |

### 8.1 Ledger schema (append-only, never mutated)

```
ts_utc, row_id, event, from_state, to_state, confirm_token, idem_key,
account, ticker, side, qty, notional, trigger_price, trigger_close,
limit_price, schwab_order_id, filled_qty, fill_price, note
```

Current state of a row = the `to_state` of its most recent event. Never edit a
prior line. If you find yourself wanting to, the design is wrong.

### 8.2 Audit log

One JSON object per line, per row, per cycle, including "nothing happened"
decisions. When a trade goes wrong at 3am the audit log is the only thing that
explains why. Log the inputs (close price, cash available, caps consumed) not
just the outcome.

---

## 9. Scheduling

Own systemd timer + service pair on Pi 1, separate from `remote_ops`:

- `order-exec.timer` — every 5 minutes, `06:00–13:30 PT`, weekdays only.
- Service runs under `flock /tmp/jobmytrading.lock -w 300` (shared lock, blocking
  with timeout — this job must not silently skip like `gitpush.py` does).
- `timeout 600` placed **inside** `sh -c` so the signal reaches Python.
- `RandomizedDelaySec=30` to avoid landing exactly on `:15`/`:50` alongside
  `jobStocksSignals.py`.

Per-cycle work:

1. Preflight: kill switch, clock, token, ledger, HMAC key → abort cleanly if any fail.
2. Read sheet → validate + token-verify → arm new rows.
3. Reconcile `SUBMITTED` rows against Schwab.
4. Expire stale rows.
5. If after 13:15 PT and not yet evaluated today: evaluate daily-close triggers.
6. Submit `TRIGGERED` rows that pass all guards.
7. Write status back to sheet, write `orders_exec.csv`, flush audit log.

Steps 2–4 are safe every cycle. Step 5 runs at most once per trading day
(guarded by `order_cursor.json`). Step 6 is where money moves.

---

## 10. Rollout phases

Do not skip phases. Do not compress them.

| Phase | Gate | Duration |
|---|---|---|
| **0 — Dry run** | `DRY_RUN=True`. Full pipeline, but instead of calling Schwab, log `WOULD_SUBMIT` with the exact payload. No Schwab write calls exist in the code path yet. | Until 10+ synthetic rows have gone ARMED→TRIGGERED→WOULD_SUBMIT correctly |
| **1 — Shadow** | Still dry-run, but on real intents Chakravarti was going to place manually. Compare `WOULD_SUBMIT` payload to what he actually placed by hand. | 2 weeks / 5+ real comparisons |
| **2 — Live, tiny** | `DRY_RUN=False`, `MAX_NOTIONAL_PER_ORDER=100`, `MAX_SUBMISSIONS_PER_DAY=1`, allowlist of 3 liquid tickers. | 1 month clean |
| **3 — Live, normal** | Raise caps to Section 6.5 defaults. | — |

Phase 0 acceptance is the gate for writing any code that calls a Schwab order
endpoint. Until Phase 0 passes, the module should not import an order-placement
function at all.

---

## 11. Test plan

Unit (no network):
- Canonicalizer round-trip: `arm_order.py` output verifies in `order_exec.py` for
  every field permutation, including empty `Notional`, empty `Limit_Offset_Pct`.
- Token tamper: mutate each intent field in turn → all must `VOID`.
- State machine: every transition in 5.1, and assert every *illegal* transition
  raises.
- Duplicate `Row_ID` in terminal state → skipped; in live state with new token → `VOID`.
- Caps: order at cap, at cap+0.01, daily aggregate crossing mid-day.
- Expiry boundary: `Expires_On` = today, before and after 13:15 PT.
- Kill switch present → no submission path reachable.

Integration (Schwab sandbox / read-only calls):
- Price fetch failure leaves row `ARMED`, does not advance.
- `PRICE_DISAGREEMENT` gate fires on a 3% divergence.
- Reconciliation correctly joins `Schwab_Order_ID` back from
  `build_orders_table()`.
- Crash-during-submit recovery (6.4): kill the process between write-ahead and
  the Schwab call, restart, assert it lands in `SUBMITTED` (if the order exists)
  or `MANUAL_REVIEW_REQUIRED` (if it does not) — never a second submit.

---

## 12. Open decisions — resolve with Chakravarti before coding

1. **`cash_reserve` framing** — "cash committed to a ticker" vs "total capital
   allocated to a ticker". Blocks Section 7. Needs `Target_Capital` in
   `reserves_config.csv` if the latter.
2. **`Qty` vs `Notional`** — support both, or force `Qty` only in v1? `Notional`
   needs a share-rounding rule (floor, and what to do when it rounds to 0).
3. **Which accounts** — the 401k/PCRA accounts are custodial and often not linked
   to the Schwab developer app. Confirm which account hashes actually accept
   order placement before designing around them.
4. **Unfilled DAY limit orders** — auto-rearm the row for the next session, or
   let it go terminal and require a new row? Default assumption in this doc:
   terminal (`CANCELLED`), requires a new row.
5. **Notification** — does a submission need to reach the phone? If yes, that is a
   separate outbound channel (the sheet alone is not a notification).

---

## 13. Implementation order

1. `order_canonical.py` + unit tests (no Schwab, no Sheets)
2. `arm_order.py` CLI
3. Sheet + dedicated service account + ACL; read path only
4. Ledger + state machine + validation, `DRY_RUN=True`
5. Trigger evaluation (Schwab price history)
6. Guards: caps, allowlists, kill switch, cash
7. Reconciliation against `stocks_orders.build_orders_table()`
8. `orders_exec.csv` output + `gitpush.py` wiring
9. systemd timer/service + `flock`
10. `remote_ops.py` kill-switch verbs
11. **Only now:** the Schwab submit call, behind `DRY_RUN`

Work one step at a time. Confirm each step's output before moving on.
