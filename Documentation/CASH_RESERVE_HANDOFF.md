# CASH_RESERVE_HANDOFF.md

Handoff spec for `cash_reserve.py` — per-(Account, Ticker) cash fencing in `myTrading`.

Written for Claude Code. Read this before touching the module.

---

## 1. What this solves

Schwab gives you cash per **account**. Signals fire per **ticker**. Nothing in
between says *"this $5,000 belongs to NVDA and nothing else may spend it."*

Without that, one enthusiastic BUY signal drains the account and starves every
other position in it. `cash_reserve.py` is the fence: you **seed** a ticker with
a slice of an account's cash, and from then on that ticker can only spend what
was earmarked for it.

`stocks_cash.py` answers *"how much cash does account ...431 have?"*
`cash_reserve.py` answers *"how much of it is NVDA allowed to touch?"*
They are complementary, not overlapping. Do not merge them.

---

## 2. Where it sits

```
jobStocksSignals.py          orchestrator — owns paths, log(), build_html_table(),
  │                          get_schwab_client(). Injects them downstream.
  ├── stocks.py              signals + holdings  -> stocks_signals.csv
  ├── stocks_orders.py       open orders         -> stocks_orders.csv
  ├── stocks_cash.py         per-account cash    -> cash.csv
  └── cash_reserve.py        per-ticker fencing  -> reserves.csv     <-- this
```

Follows the same injection contract as `stocks_cash.py`. The module defines
**no** paths, **no** logger, **no** HTML renderer, **no** Schwab client. All
injected by the caller. Do not add local copies of any of those — that rule has
been broken before and it is the single most important convention in this repo.

The one exception is **state file paths**. `reserve_ledger.csv` and
`reserves_config.csv` are machine-local money records under
`~/.local/state/myTrading/` (override via `MYTRADING_STATE_DIR`). They are
deliberately **not** in the repo and must never be committed. Add to
`.gitignore` if any path in the repo ever points at them.

---

## 3. Data model

### `reserves_config.csv` — declarative intent, safe to rewrite

| Column | Meaning |
|---|---|
| `Account` | last 3 digits only (`431`) — never a full account number |
| `Ticker` | uppercase |
| `Policy` | `CASH_ONLY` or `TOTAL_CAPITAL` (see §4) |
| `Seed_Cash` | cumulative cash fenced to this pair |
| `Target_Capital` | total capital ceiling — only meaningful for `TOTAL_CAPITAL` |
| `Max_Order_Pct` | optional per-order cap, % of available. Blank = no cap |
| `Effective_From` | PST timestamp. Fills before this are skipped, not debited |
| `Active` | `Y`/`N` |
| `Notes` | free text |

### `reserve_ledger.csv` — append-only history, NEVER rewritten

| Column | Meaning |
|---|---|
| `Timestamp_PST` | write time |
| `Event_ID` | `uuid4[:12]`, row identity |
| `Account`, `Ticker` | keys |
| `Event` | `SEED` `TOPUP` `WITHDRAW` `CLOSE` `BUY_FILL` `SELL_FILL` `SKIP_BUY` `SKIP_SELL` `ADJUST` |
| `Amount` | **signed**: credit > 0, debit < 0, skip == 0 |
| `Balance_After` | advisory snapshot — `fold_balances()` is authoritative |
| `Ref` | idempotency key. Schwab `activityId` for fills, `op:<uuid>` for manual ops |
| `Source` | `cli` / `job` / `sheet` / `manual` |
| `Reason` | free text |

**Rules that are not negotiable:**

1. Never rewrite, sort, or de-duplicate the ledger in place. Corrections are
   new `ADJUST` rows.
2. Balances are always `fold_balances()` — a sum over the ledger. `Balance_After`
   exists only so a human can read the file; `--reconcile` flags drift.
3. Appends go through `append_events()`, which takes an exclusive `flock` and
   `fsync`s. Nothing else writes this file.

---

## 4. OPEN DECISION — `Target_Capital` framing

This was left unresolved and Claude Code should **not** silently pick one.

- **`CASH_ONLY`** — `Seed_Cash` is *cash committed to the ticker*. The reserve
  balance is dry powder. Buys debit it, sells credit it back.
  `Available_To_Buy = reserve balance`.
  Simple, needs no price data, and the ledger alone is sufficient state.

- **`TOTAL_CAPITAL`** — `Target_Capital` is *total capital allocated to the
  ticker*, cash plus the market value of the open position.
  `Available_To_Buy = Target_Capital − Position_Value`, capped by the reserve.
  Naturally self-limiting (a position that runs up stops attracting new money),
  but it depends on live marks, so the gate's answer moves with the market.

Both are implemented. `CASH_ONLY` is the default because it is deterministic.
Confirm the intended default before wiring the gate into live order submission —
under `TOTAL_CAPITAL` a sharp drawdown *re-opens* buying capacity automatically,
which may or may not be what's wanted.

---

## 5. The SKIP rows — why they exist

This is the subtle part; do not "simplify" it away.

Every fill-derived event carries `Ref = activityId`. Before applying fills,
`seen_refs()` collects every `Ref` already in the ledger **including SKIP rows**,
and anything already seen is dropped.

When a fill arrives that *cannot* be applied — ticker not fenced, reserve
inactive, or the fill predates `Effective_From` — it is written as `SKIP_BUY` /
`SKIP_SELL` with `Amount = 0` and the same `Ref`. That solves two problems at once:

1. **Retro-debiting.** Seed NVDA today; last month's NVDA buys must not
   suddenly debit the new reserve. They get SKIP'd once and are permanently
   settled.
2. **Repeating nudges.** Without a persisted marker, every run re-discovers the
   same unfenced fill and warns about it again, every 35 minutes, forever.

A SKIP row means *"seen, deliberately not applied."* It is not an error state.

---

## 6. API surface

```python
from cash_reserve import (
    seed, topup, withdraw, close,          # management
    available_to_buy,                      # the gate
    apply_fills, fetch_fills,              # reconciliation
    build_reserves_table, write_reserve_outputs,   # reporting
    fold_balances, balance_of, reconcile,  # introspection
)
```

The gate, fail-closed by design:

```python
res = available_to_buy("431", "NVDA", positions_df=df)
# {"allowed": bool, "available": float, "policy": str, "reason": str}
```

An unfenced pair returns `allowed=False`. A ticker with no reserve row has no
permission to spend. Keep it that way — do not add a "default allow" fallback.

---

## 7. CLI

```bash
python3 cash_reserve.py --seed     --account 431 --ticker NVDA --amount 5000
python3 cash_reserve.py --seed     --account 431 --ticker AMD  --amount 4000 \
                                   --policy TOTAL_CAPITAL --target 4000
python3 cash_reserve.py --topup    --account 431 --ticker NVDA --amount 1000
python3 cash_reserve.py --withdraw --account 431 --ticker NVDA --amount 500
python3 cash_reserve.py --close    --account 431 --ticker NVDA
python3 cash_reserve.py --check    --account 431 --ticker NVDA --amount 2000  # exit 0/1
python3 cash_reserve.py --apply-fills --days 7 [--dry-run]
python3 cash_reserve.py --reconcile
python3 cash_reserve.py --list
python3 cash_reserve.py --write
```

`--check` exits 0 when allowed, 1 when blocked, so it can gate a shell step.

---

## 8. Build order for Claude Code

Do these one at a time, verifying each before moving on.

1. **Drop the module in and smoke-test it offline.**
   `MYTRADING_STATE_DIR=/tmp/rtest python3 cash_reserve.py --seed --account 431 --ticker TEST --amount 1000`
   then `--list`, `--check`, `--withdraw`, `--reconcile`. No Schwab needed for any of these.

2. **Add the output paths to `jobStocksSignals.py`**, next to the existing
   `OUT_CASH_CSV` / `OUT_CASH_HTML` definitions:
   ```python
   OUT_RESERVES_CSV  = JOB_DIR / "reserves.csv"
   OUT_RESERVES_HTML = JOB_DIR / "reserves.html"
   ```
   The module falls back to `JOB_DIR / "reserves.csv"` if they're absent, but
   define them properly — paths belong in the orchestrator.

3. **Wire the reporting step** into `jobStocksSignals.py` immediately after the
   existing `# 4c. Cash & cash investments` block, in its own
   `try/except` so a reserve failure is non-fatal, exactly like the cash step:
   ```python
   try:
       from cash_reserve import build_reserves_table, write_reserve_outputs
       df_res = build_reserves_table(positions_df=df, cash_df=df_cash, log=log)
       write_reserve_outputs(df_res, updated_pst,
                             out_csv=OUT_RESERVES_CSV, out_html=OUT_RESERVES_HTML,
                             html_builder=build_html_table, log=log)
   except Exception as e:
       log(f"⚠️  Reserve step failed (non-fatal): {e}")
   ```
   Note `df_cash` must still be in scope — if the cash block scopes it inside
   its own `try`, hoist it to `df_cash = None` above both.

4. **Verify `fetch_fills()` against the live client.** This is the one function
   written defensively rather than from confirmed behaviour. It probes
   `transactions` / `account_transactions` / `transactions_all` and several
   kwarg shapes because the schwabdev method signature varies by version.
   Run `--apply-fills --days 7 --dry-run` first and read the output. Confirm:
   - the field is `activityId`, **not** `transactionId`
   - the field is `netAmount`, **not** `amount`
   - leg container is `transferItems` vs `transactionItems`
   Once confirmed, delete the probing branches and hard-code the real one. Leave
   a comment recording which version it was verified against.

5. **Only then** consider gating live orders on `available_to_buy()`. That is
   downstream of the Schwab order-execution work, which still needs its own
   security model. Do not wire the gate into anything that can submit an order
   until that exists.

---

## 9. Related work this unlocks or depends on

| Item | Relationship |
|---|---|
| **Fills feed** (step 4 above) | **Blocking.** Without reliable fills the ledger only records manual ops; reserves drift from reality the first time an order fills. Highest priority. |
| **Schwab order-execution sheet** | `available_to_buy()` is the cash-reserve integration that design already called for. State machine `ARMED→TRIGGERED→SUBMITTED→FILLED/REJECTED` should consult the gate at `TRIGGERED`. Still needs its full security model first. |
| **`dashboard.py` panel** | Add a Reserves table reading `reserves.csv`. Remember: `dashboard.py` and `gsheet_notes.py` are always updated together and deploy to **`jobMyTrading`**, not `myTrading` — Streamlit Cloud reads from `jobMyTrading`. |
| **`sell_guard.py`** | A SELL credits the reserve. Decide whether proceeds return to the ticker's fence (keeps the sleeve intact) or to free account cash (releases capital). Currently: back to the fence. |
| **Google Sheet seeding** | A `Reserves` tab so seeding doesn't require SSH. Goes through `remote_ops.py` as new allowlisted verbs (`reserve_seed`, `reserve_topup`, `reserve_withdraw`) — never `shell=True`, and honour the existing state cursor for replay prevention. |
| **Cron** | Reporting rides inside `jobStocksSignals.py` at `:15`/`:50`, so no new crontab entry and no new lock. If `--apply-fills` ever runs standalone, it must join `flock /tmp/jobmytrading.lock -w 600`. |
| **Backup** | `reserve_ledger.csv` is money history and is gitignored, so nothing backs it up today. Needs a real backup path before the file matters. |
| **Kill switch** | Share the sentinel file the order-execution design specifies — when present, `available_to_buy()` should return `allowed=False` for everything. Not implemented yet; add it when the sentinel exists. |

---

## 10. Traps

- **Account keys are last-3 everywhere.** `acct_key()` normalises `...431`,
  `431`, and the full number to `431`. Never persist a full account number.
- **Pi 2 writes code, Pi 1 executes.** Pi 1 picks up changes only when
  `deploy/RELEASE` changes. Bumping it is a separate step from pushing.
- **`flock -n` on `gitpush.py`** means it silently skips while a signal job holds
  the lock. Normal. Not a reserve bug.
- **`timeout` must go inside `sh -c`**, not wrapping it, or the signal never
  reaches Python.
- **Don't let `Balance_After` become the source of truth.** It is advisory.
  Anything that reads it instead of `fold_balances()` is a bug.
- **Negative reserve = BREACH, logged, and still ledgered.** The ledger records
  what happened, not what should have. Fix breaches with `ADJUST` rows, never by
  editing history.
