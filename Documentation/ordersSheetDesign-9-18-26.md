# Orders Sheet — Design

**Status:** Design in progress, nothing implemented
**Sheet:** `myTrading-ORDERS-pi1` (separate from `myTrading-ops-pi1`)
**Runs on:** Pi 1
**Relationship to `orderExecutionDesign-9-7-26.md`:** that document specifies a
hardened, HMAC-gated execution engine. This one specifies the *sheet* — the
layout, the ownership rules and the reporting that comes first. Execution is
phase 3 here and inherits that document's security model.

---

## 1. What this is for

One screen, readable from a phone, that answers:

- what do I own, and is each position protected?
- what orders are live, where, and what happened to them?
- how much cash is actually free after seed money is reserved?
- **is the system even working right now?**

---

## 2. Three principles

**The sheet is a mailbox, not a database.** Pi 1's local files are the truth —
`cash_reserve.py`'s ledger for money, Schwab for positions and orders. Pi 1
writes *to* the sheet and never reads its own writes back as fact. Otherwise a
stray edit makes it re-place an order.

**Every column has exactly one owner.** Nothing straddles. See §5.

**Reporting before execution.** Phase 1 places no orders at all and still
delivers most of the value. See §10.

---

## 3. Tabs

| Tab | Written by | Contents |
|---|---|---|
| `Orders` | you (intent) + Pi 1 (status) | the working sheet |
| `Positions` | **Pi 1 only** | holdings from Schwab, with coverage flags |
| `Cash` | **Pi 1 only** | cash, seed reserved, free to deploy |
| `History` | **Pi 1 only, append-only** | completed rows |

---

## 4. Header block — rows 1-6 of `Orders`

Written by Pi 1 **every cycle**, even when nothing else changes. That is the
point: the header's own staleness is the health check. A `LAST POLL` three days
old tells you the poller is dead, and nothing has to detect that failure.

```
A1  ✅ SYSTEM OK        |  🔴 ACTION REQUIRED — SCHWAB TOKEN EXPIRED
A2  SCHWAB TOKEN        |  OK — expires 2026-09-23 14:30 PT (5.2 days)
A3  LAST POLL           |  2026-09-18 14:05:12 PT
A4  TRADING             |  ENABLED  /  DISABLED (kill switch)
A5  FREE TO DEPLOY      |  $12,480 across 3 accounts
A6  ALERTS              |  3 holdings with no SELL-STOPLOSS
```

Set conditional formatting once so A1 goes red whenever it is not `OK`. The
alarm should be visible from a phone without reading anything.

Row 7 blank, row 8 column headers, **data starts row 9**. Whatever reads this
sheet needs a `DATA_START_ROW = 9` constant rather than assuming row 2.

**Prerequisite:** the token expiry must come from `~/.schwabdev/tokens.db`.
`schwab_auth.py --status` currently reads the dead legacy
`~/github/myTrading/tokens.json` and reports a misleading answer — see
`PROJECT_PLAN.md` §7. That defect now blocks the header.

---

## 5. `Orders` columns

### You write — A through K

| Col | Name | Notes |
|---|---|---|
| A | `Row_ID` | **Leave blank.** Pi 1 stamps it once (`2026-09-18-MU-01`), then it is the row's permanent identity |
| B | `Date` | when you added the intent |
| C | `Acct` | **last 3 digits** — see §6 |
| D | `Ticker` | uppercase |
| E | `Action` | `SELL-TRIM` \| `SELL-STOPLOSS` \| `BUY-DIP` \| `BUY-BREAKOUT` \| blank = nothing to do |
| F | `Trigger_Price` | the condition price — not necessarily the fill price |
| G | `Limit_Price` | optional. What you will actually accept. Matters on a stop, where a gap can fill far below the trigger |
| H | `Qty` | |
| I | `Qty_Unit` | `SHARES` \| `PCT_POS` \| `USD` — so "trim 25%" needs no arithmetic from you |
| J | `After_Close` | `Y` = Pi 1 watches the daily close; `N` = resting order at Schwab. See §7 |
| K | `Expires_On` | stops a BUY-DIP firing eight months later in a different world |

### Pi 1 writes — L through V. Never type in these.

| Col | Name | Notes |
|---|---|---|
| L | `Venue` | `SCHWAB_RESTING` \| `PI_WATCHING` \| `SCHWAB_DIRECT` |
| M | `Status` | `NEW → ARMED → PLACED → TRIGGERED → FILLED`, or `PARTIAL` / `CANCELLED` / `EXPIRED` / `REJECTED` / `BLOCKED` |
| N | `Status_Date` | |
| O | `Validation` | **computed.** "SELL-TRIM priced below current price" is an error Pi 1 should catch, not prose you type |
| P | `Current_Price` | context — is the trigger near? |
| Q | `Schwab_Order_ID` | joins to `stocks_orders.csv` |
| R | `Filled_Qty` | |
| S | `Fill_Price` | |
| T | `Seed_Left` | mirrored from `cash_reserve.py`. **Never editable** |
| U | `Engine_Note` | why blocked, rejected or skipped |
| V | `Last_Checked` | proves the poller saw this row |

`W` onward is free-text comments, yours.

---

## 6. Account digits — last 3, decided 2026-09-18

The sheet currently mixes `1771` (4 digits) with `922` and `885` (3). The
codebase keys on the **last 3** everywhere — `cash_reserve.acct_key()` does
`s[-3:]`, `stocks_cash._mask()` the same, and `orderExecutionDesign` states the
rule as never persisting a full account number.

**Decision: the sheet matches the code. Last 3.** Changing the sheet is free;
changing `acct_key()` would touch the reserve ledger's existing keys.

---

## 7. The split that matters — resting vs watched

`After_Close` is not a preference, it selects the mechanism:

| | `N` — resting at Schwab | `Y` — Pi 1 watches |
|---|---|---|
| Who holds it | Schwab | Pi 1 |
| Survives Pi 1 dying | **yes** | no |
| Can express | price touched | *closed* above/below |

**SELL-STOPLOSS should always be `N`.** It is the order whose entire job is
protecting you, and it must not depend on a Raspberry Pi being alive. Only use
`Y` where Schwab genuinely cannot express the condition.

---

## 8. Orders placed directly at Schwab

**Both paths allowed.** Requiring every order to go through the sheet means Pi 1
being down stops you trading — a worse failure than the confusion it avoids.

`stocks_orders.build_orders_table()` already reads open Schwab orders. Each
cycle, Pi 1 reconciles: any open order at Schwab with no matching sheet row is
appended with `Venue = SCHWAB_DIRECT`. Place orders however you like; the sheet
always reflects reality.

---

## 9. Seed money, history, backup

**Seed money.** `cash_reserve.py` owns it — append-only ledger, balances always
a fold over it. The sheet **mirrors** `Seed_Left`; you never type in that
column. Money state that lives only in a spreadsheet is one accidental edit from
being wrong and impossible to audit afterwards.

`Cash` tab: `Free_To_Deploy = Cash_After_Open_Orders − Seed_Reserved`.

**History.** Do **not** move completed rows down the sheet — moving a row
changes every row number below it and breaks anything tracking position. Pi 1
*copies* completed rows to `History` and marks the `Orders` row terminal.
`Row_ID` makes position irrelevant.

**Backup.** Two layers. Google Sheets' own version history covers accidental
edits. For the programmatic one, Pi 1 dumps all tabs to a dated CSV daily and
commits to `jobMyTrading` — git-versioned, diffable, and that pipeline already
exists.

---

## 10. Phasing — build reporting first

**Phase 1 — Pi 1 places nothing.** Writes `Positions`, `Cash`, `Seed_Left`, the
header block, reconciles Schwab orders into the sheet, flags coverage gaps.
Worst case it writes a wrong number in a cell. This alone gives one screen
showing every holding, its gaps, seed reserved and free cash.

**Phase 2 — resting orders.** Pi 1 places `After_Close = N` rows at Schwab.

**Phase 3 — watched triggers.** Pi 1 evaluates daily closes and places on
trigger. Inherits the full security model of `orderExecutionDesign-9-7-26.md`.

Do not compress these. Phase 1 is genuinely useful and carries no execution
risk.

---

## 11. Coverage check

For every holding, the four actions are expected: `SELL-TRIM`,
`SELL-STOPLOSS`, `BUY-DIP`, and `BUY-BREAKOUT` where seed is allocated. Pi 1
generates skeleton rows with blank prices and flags what is missing —
*"you hold AEHR with no SELL-STOPLOSS."* **Pi 1 never invents a price.**

This is `PROJECT_PLAN.md` §3 surfacing in the sheet, and `sell_guard.py`
already implements the stop-loss half.

---

## 11b. `Positions` — grain and totals

**Decided 2026-09-18.** One row per **(ticker, account)**, plus a `TOTAL` row
after a ticker **only when it is held in more than one account**. A
single-account holding gets no total row — it would just be the same numbers
twice.

```
Ticker  Acct   Qty  Avg_Cost  Market_Value  Has_Stop  Has_Trim  ...
AAPL    431     50    182.40      9,875.00       Y         N
AAPL    482     30    201.10      5,925.00       N         N      <- unprotected
AAPL    TOTAL   80    189.43     15,800.00       -         -
MSFT    431     10    405.27      4,940.00       Y         Y      <- no total row
```

Rules for the `TOTAL` row:

- `Acct` reads `TOTAL`; sorts last within its ticker group
- `Qty`, `Market_Value`, `Unrealized_PL`, `Seed_Reserved` are sums
- **`Avg_Cost` is quantity-weighted**, not a mean of the per-account averages.
  `Σ(qty × avg_cost) / Σ(qty)`. A plain average is wrong whenever the accounts
  hold different sizes, and wrong in a way that looks plausible
- the four `Has_*` flags are **blank** — see below

### Coverage is per account, and must stay that way

`Has_Stop` and its siblings are evaluated **per (ticker, account)**, never
rolled up. This is not a presentation choice.

A stop-loss resting in account `...431` protects only the shares in `...431`.
If AAPL is held in two accounts with a stop in one of them, a rolled-up
`Has_Stop = Y` would report the position as protected while half of it is
naked. The aggregate answer is not merely less precise — it is false, and false
in the direction that gets expensive.

Hence the blank on `TOTAL` rows: there is no honest aggregate value to put
there.

---

## 12. Credentials — one service account for both sheets

**Decided 2026-09-18.** A single identity manages both sheets:

```
mytrading-ops@mytrading-sheets.iam.gserviceaccount.com
```

| Sheet | Access | Why write access |
|---|---|---|
| `myTrading-ops-pi1` | **Editor** | Pi 1 writes verb results back into columns C..J |
| `myTrading-ORDERS-pi1` | **Editor** | Pi 1 writes status, header block, Positions, Cash, History |

Key on Pi 1 at `/etc/myTrading/gsheets-ops.json`, mode `600`, owned by
`rchak007` — readable by the process, by nobody else on the box. Pointed at by
`REMOTE_OPS_CREDS` in `.env`.

Both are **Editor, not Viewer**, and that is required rather than convenient:
the Drive share is the real boundary, and the OAuth scope is client-side only.
With Viewer, `remote_ops` would run a verb correctly, then fail every write-back
— three retries, an audited `writeback_failed`, and a row that silently stays
blank.

### Deviating from `orderExecutionDesign` §6.8, deliberately

That document specifies a **separate, dedicated** service account for the orders
sheet. Chakravarti has decided against it, now and later.

The reasoning that makes this defensible: both keys would live on Pi 1 anyway,
so a Pi 1 compromise reaches both sheets regardless. A second identity only
helps against a key leaking *independently* of the box — committed by accident,
copied to a laptop, pasted somewhere.

What it costs, stated plainly so it is not rediscovered later as a surprise:
**if this one key leaks, it can write to both the ops channel and the orders
sheet.** Once Pi 1 can place orders (phase 2), that means a single leaked key
reaches order placement. The compensating controls then have to come from
elsewhere — the caps, allowlists and kill switch in `orderExecutionDesign` §6.5
and §6.7 — rather than from credential separation.

Do not re-raise this as a defect. It is a decision.

---

## 13. Open decisions

1. **`PCT_POS` semantics** — is "trim 25%" a percentage of the current position
   or of the original entry? They diverge after the first trim.
2. **Expiry default** — if `Expires_On` is blank, does the row live forever or
   default to 90 days as the execution design caps it?
3. **Partial fills** — does a partially filled row stay live for the remainder,
   or go terminal?
4. **Multiple accounts, same ticker** — one row per (account, ticker, action),
   or one row per ticker spanning accounts?
5. **Who cancels?** If you delete a row that is already `SCHWAB_RESTING`, does
   Pi 1 cancel the order at Schwab, or leave it and warn?
