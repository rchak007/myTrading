# Orders Sheet — Design

**Status:** **Phase 1 live since 2026-09-19.** `Positions`, `Cash`, `Dashboard`
and the header block are written every cycle by `jobStocksSignals.py` step 4d.
Pi 1 still places **no orders** — phases 2 and 3 are not built. See §10.
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
| `Dashboard` | **Pi 1 only, regenerated** | per-ticker blocks: positions + live orders together. See §11c |

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

**Prerequisite — cleared 2026-09-18 in `e9a866a`.** The token expiry must come
from `~/.schwabdev/tokens.db`; `schwab_auth.py` had been reading (and writing)
the dead legacy `tokens.json`, which current schwabdev ignores. No longer
blocking. See `PROJECT_PLAN.md` §7.

### Timestamps are Pacific — changed 2026-09-19

`LAST POLL`, the `Dashboard` header and the `jobStocksSignals.py` log all print
Pacific time with `%Z`, so they read `PDT` or `PST` as the season dictates.

Previously the job log printed UTC while the sheet used the Pi's *own* timezone
— two different clocks in one system, neither matching the market hours or the
cron schedule you read them against. The sheet side is now pinned explicitly
rather than inherited from the host, because the Pi's clock may itself be UTC.

`orders_sheet._now()` falls back to local time if the tz database is missing,
rather than failing the write — a wrong-looking timestamp beats a lost refresh.

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

**Live since 2026-09-22.** `orders_sheet.load_reserves()` folds
`cash_reserve.fold_balances()` into `{(acct, ticker): amount}`. `Seed_Reserved`
appears per (ticker, account) on `Positions` and the `Dashboard`, and summed per
account on `Cash`. A pair folding to zero — closed, or fully deployed — is
dropped, so the column reads blank rather than `$0.00` beside a ticker with no
reserve.

Fail-soft by design: if `cash_reserve` cannot be imported or the ledger is
missing, the column is left blank and the sheet still writes. Positions and
coverage flags are the part you cannot get anywhere else.

**A reserve is an earmark, not a separate pot.** The money stays ordinary cash
in the account; nothing stops it being spent elsewhere. `Seed_Reserved` only
subtracts it from `Free_To_Deploy` so you can see it is spoken for.

**Over-fencing is visible, not prevented.** Nothing validates a seed against the
account's real balance (PROJECT_PLAN §2), so reserving more than you hold is
accepted. It then shows as a negative `Free_To_Deploy`, a log warning naming the
account and shortfall, and `OVER-FENCED: <acct>` in the header `ALERTS` row.

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

## 11c. `Dashboard` — one block per ticker

**Added 2026-09-19.** `Positions` and `Orders` answer "show me everything of one
kind". The `Dashboard` answers the question actually asked while trading:
*for this one ticker, what do I hold and what is resting against it?*

### Layout

One block per ticker, alphabetical. Column A carries only the ticker label, so
the tables underneath start in column B and line up across every block.

```
AEHR │ POSITIONS                                      ← cyan, bold
     │ Ticker  Acct  Qty  Avg_Cost  Market_Value  Unrealized_PL  Has_Stop …
     │ AEHR    171    26     80.98       2430.74         127.49  N
     │
     │ ORDERS                                         ← yellow, bold
     │ Acct  Side  Type  Qty  Limit_Price  Stop_Price  Status  Entered  Order_ID
     │ — no open orders —
```

A ticker split across accounts gets its `TOTAL` row inside the block, same rule
as §11b — and the `Has_*` flags stay blank on it, same reason.

Row 1 is a header carrying the tab's **own** update time, so a stale Dashboard
is visible without cross-checking `LAST POLL` in `Orders`.

### Live prices — added 2026-09-23

`Live_Price` (col F) and `Day_%` (col G) sit beside `Avg_Cost`, so paid / now /
today's move reads left to right. **`jobStocksSignals.py` writes them blank**;
`orders_sheet_prices.py` fills them on its own faster schedule.

**Why a separate script.** The full job takes ~9 minutes and runs at `:15` and
`:50`, so its prices are already ~9 minutes old on arrival and up to 35 minutes
old before the next write. Fine for signals, useless for "is it at my trigger
right now". The price updater makes one Schwab `quotes` call and writes two
columns — seconds, so it can run every 3 minutes.

**Schwab `quotes`, not `Market_Value / Qty`.** The division is tempting and
needs no new API, but the quotes payload carries `realtime: true` and a separate
`extended` block for pre/post market. A derived figure gives neither. Measured
with `probe_quotes_api.py`: the method is `quotes` on the **schwabdev** client
(the wrapper forwards only `fetch_positions`, reach it via `get_client()`), and
it takes a single **comma-joined string** — the same shape `types` needed for
transactions.

**It touches nothing else.** Sheet-only: no files, no git, so `gitpush.py` never
sees it. It runs under `flock -n` on the *same* lock as `jobStocksSignals.py`,
so it skips while the big job rebuilds the tab rather than writing into a
half-built layout.

**Row positions are re-read every run**, never cached, and matched by ticker —
the full job regenerates the tab wholesale, so every row moves.

**Sections are tracked, not inferred.** An ORDERS row holds the account in
column B and the side in C, so `171 | SELL` reads exactly like a ticker with an
account beside it. The scanner follows `POSITIONS` / `ORDERS` markers and only
collects inside a positions block; without that it wrote prices into the orders
table.

### Ticker labels link to TradingView — added 2026-09-23

Each block's ticker in column A is written as
`=HYPERLINK("https://www.tradingview.com/chart/<layout>/?symbol=<TICKER>", "<TICKER>")`,
opening Chakravarti's saved layout (`ajSFidjP`, override with
`TRADINGVIEW_CHART`) which already carries his indicators. Same browser profile,
so it opens signed in.

**No exchange prefix.** `NASDAQ:AEHR` is the obvious form and is wrong here:
these holdings span NASDAQ, NYSE and NYSE Arca (IBIT, ARKB, HODL), so a
hardcoded prefix breaks every ticker not on that exchange. A bare symbol lets
TradingView resolve the primary listing.

Written by `_link_tickers()` as a **separate** batch from the table, because a
formula needs `USER_ENTERED` while the data wants `RAW` — sending the whole
table as `USER_ENTERED` would let Sheets reinterpret values it has no business
touching, such as an order's `Entered` timestamp becoming a date. Best-effort
like `_paint()`: a link is a convenience, the data is not.

### Read-only by construction

Nothing a human types lives here. That is what lets it be regenerated wholesale
every cycle with `ws.clear()` and no merge step — and therefore what makes it
impossible for a refresh to eat a typed intent.

A writable block layout was considered and rejected: blocks break sorting and
filtering, every inserted row shifts the ones below it, and reconciling typed
values back out of a regenerated layout needs exactly the merge step whose
absence makes this safe. Intent goes in `Orders`, which is flat and row-stable.

### The `ORDERS` rows are Schwab's, not the `Orders` tab's

They come from the live Schwab order list, not from what you typed. What
matters for protection is what is actually resting at the broker — and per §8
you place orders directly too, so the `Orders` tab is not a complete picture.
This is the same reasoning that drives the `Has_*` flags, and it means the two
always agree: `— no open orders —` under a ticker is exactly why its `Has_Stop`
says `N`.

### Formatting never costs data

`_paint()` is wrapped in its own `try/except`. `gspread`'s `format()` signature
varies across versions; losing the colours to a library difference is a nuisance,
losing the write is not. On failure it logs
`⚠️ dashboard formatting skipped (data is fine)` and the data stands.

### Defect fixed 2026-09-19 — one NaN blanked the whole tab

The first run produced an empty tab and
`Out of range float values are not JSON compliant: nan`. Google rejects the
**entire batch** on a single non-compliant float. Order legs are full of them —
a stop order has no `Limit_Price`, a limit order has no `Stop_Price`. `_put()`
had always scrubbed `NaN`; the dashboard path was the one place it was missed.
Both now share `_scrub()`.

Worth remembering as a class: **any** unexpected `NaN` anywhere in a payload
blanks the whole write, and the symptom is an empty tab rather than a partial
one. That error line is the first place to look.

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

### A second, read-only identity for Pi 2 — added 2026-09-19

```
mytrading-reader@mytrading-sheets.iam.gserviceaccount.com     Viewer on both sheets
```

Purpose: let Claude, running on **Pi 2**, read the `Dashboard`, `Positions` and
`Cash` tabs directly — so a question like "which holdings have no stop?" is
answered from live data rather than from pasted screenshots. See PROJECT_PLAN §3.

| | Pi 1 | Pi 2 |
|---|---|---|
| Identity | `mytrading-ops@…` | `mytrading-reader@…` |
| Role | Editor | **Viewer** |
| Key | `/etc/myTrading/gsheets-ops.json` | `~/.config/myTrading/gsheets-reader.json` |
| Env var | `REMOTE_OPS_CREDS` | `GSHEET_READER_CREDS` |

This does place a credential on Pi 2, which the Pi1/Pi2 split otherwise avoids.
Accepted because a Viewer key cannot write, cannot trade, and cannot reach the
token store — and because the alternative was reading the system through
screenshots. **The role must stay Viewer.** Pi 2 is where untested code runs; an
Editor key there would let a half-written script write to the live trading sheet.

Pi 2 setup (Debian PEP 668 blocks `pip --user`, hence the venv):

```bash
mkdir -p ~/.config/myTrading && chmod 700 ~/.config/myTrading
chmod 600 ~/.config/myTrading/gsheets-reader.json
python3 -m venv .venv && .venv/bin/pip install gspread google-auth
```

Sheet IDs live in Pi 2's own `.env` — gitignored, and deliberately **not** a
copy of Pi 1's: it carries no Schwab credentials and no editor key.

**Note the exact address.** It is `mytrading-reader@…`, not `reader@…`. Google's
share dialog accepts a nonexistent principal without complaint, so a wrong
address fails later as an opaque `PermissionError` rather than at the point of
the mistake.

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
