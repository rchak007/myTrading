# How it works

**The non-technical view.** What this system is for, what it does each day, and
where to look when you want something. For code and file layout see
[ARCHITECTURE.md](ARCHITECTURE.md); for schedules and logs see
[OPERATIONS.md](OPERATIONS.md).

---

## 1. What it is for

Chakravarti trades a portfolio of ~130 tickers across six Schwab accounts. The
system exists to answer four questions without a spreadsheet full of manual
work, and to act on two of them:

| | |
|---|---|
| **What do I own, and is it protected?** | every position, with a flag for whether a stop and a profit target actually exist |
| **What is this worth, and what did I make?** | live prices, and realised P&L back to 2020 |
| **How much can I actually spend?** | cash, minus money earmarked for specific tickers |
| **Act when a condition is met** | place an order when a stock *closes* through a price |

The last one is the part Schwab cannot do itself, which is why it exists here.

---

## 2. Two machines, one job each

```
   PI 2  (dev)                         PI 1  (production)
   ──────────                          ──────────────────
   writes code            git push →   pulls and runs it
   no Schwab credentials               holds every credential
   reads the sheets (Viewer)           writes the sheets (Editor)
   sends the warning emails            does the trading
```

The split is deliberate. Pi 2 is where untested code runs, so it holds nothing
that could place an order. Pi 1 has the credentials but never edits code
directly — everything arrives through git, which means everything is reviewable
and revertible.

Neither machine has everything, and that is the point. The one thing each needs
from the other travels through a Google Sheet rather than a copied secret.

---

## 3. The daily rhythm

Times are Pacific. Full schedule in [OPERATIONS.md](OPERATIONS.md) §1.

**Overnight — 04:30**
The P&L job pulls the last week of Schwab transactions and rebuilds realised
profit and loss for every ticker ever traded.

**Through the day — every 5 to 15 minutes**
Signals are recomputed, live prices refresh on the Dashboard, and every row you
have typed into the Orders tab is re-validated. If a row cannot work — you typed
a quantity larger than you hold, or a ticker that does not exist — the sheet
tells you within minutes rather than at the close.

**After the close — 13:30**
The order engine reads the day's completed daily bar. Any row whose condition is
met becomes a real order at Schwab.

**Twice daily — 08:30, 18:30**
Pi 2 checks whether the Schwab token is near expiry, and whether Pi 1 has gone
silent. Either one gets an email.

---

## 4. The two Google Sheets

### `myTrading-ORDERS-pi1` — what you own and what you want to happen

| tab | what it is |
|---|---|
| **Dashboard** | the main screen. One block per ticker, biggest first: what you hold in each account, live price, and whether it is protected |
| **Orders** | where you type intents. Pi 1 reads them and writes back what it made of each |
| **Positions** | the same holdings as a flat table |
| **Cash** | cash per account, less money reserved for specific tickers |
| **History** | completed rows |
| **Chitra** | your sister's account, which you manage. Read-only, rendered from `chitra_holdings.csv` in the repo — see §5b |

The **header block** at the top of `Orders` is rewritten every cycle. Its own
staleness is the health check: a `LAST POLL` three days old means Pi 1 is dead,
and nothing has to detect that.

### `myTrading-ops-pi1` — driving Pi 1 from a phone

Type a verb in column A, and Pi 1 runs it within ten minutes and writes the
result back. `git_pull`, `token_status`, `seed`, `fence`, `reserves`,
`auth_url`, `auth_code`, and a dozen read-only diagnostics.

It is an allowlist, not a shell: the sheet supplies a **verb name**, never a
command. See [remoteOpsGuide](remoteOpsGuide-9-7-26.md).

---

## 5. Protection: the `Has_Stop` / `Has_Trim` flags

For every position, the Dashboard says whether a stop (protection below) and a
trim (profit target above) actually exist.

| | meaning |
|---|---|
| **Y** | a live order resting at Schwab. Fires the moment price touches it, and works with every machine here switched off |
| **P** | an intent in the Orders tab. Fires only on a completed daily **close**, and only if Pi 1 is alive |
| **N** | nothing. Painted **yellow**, because an unprotected holding should be impossible to scroll past |

**The distinction between Y and P matters.** A `P` stop does not protect against
an intraday collapse — that is the trade you accept by saying "closes below"
instead of "touches".

**Coverage is per (ticker, account), never rolled up.** A stop in one account
protects only the shares in that account. Rolling it up would report a position
as protected while half of it is naked — not less precise, *false*, and false in
the expensive direction.

---

## 5a. Recommended levels: where an order *would* go

The flags above say whether an order **exists**. Under them, each block carries
three lines saying where one **would go** — four prices derived from the
indicators the signals job already computes.

```
RECOMMENDED                                     Rec_Stop  Rec_Trim  Rec_Dip
① ⅓ out · ⅓ trim · first bid · ATR 11.27 (3.2%)   339.01    413.18   254.33
② another ⅓ out (close-confirmed) · ⅓ trim        327.75    446.67      —
③ ALL OUT — hard stop, rests at Schwab · final ⅓  316.47    480.16      —
```

**Read the rows top to bottom as a scale-out: a third, a third, then the rest.**

**Two stops, two mechanisms, and the difference is the point.** ① and ② are
*close-confirmed* — they belong in the Orders tab as "closes below", so a wick
cannot take you out at the worst price of the day. ③ is a *resting Schwab
stop*: a touch trigger, further out, and a **full exit**. Earnings is the one
event that defeats ① and ②, because the gap happens before any close can
confirm anything — which is why held names with earnings inside the window get
a banner at the top of the Dashboard.

**Trim is three levels because one forces an all-or-nothing decision**, and you
will always feel you sold too early.

**The dip is either an add or a re-entry**, labelled as such. An `add:` sits
above the stop — you still hold. A `re:` sits below the whole stop ladder —
you were stopped out, and this is the way back in.

**They place nothing.** These are numbers to read. Typing one into the Orders
tab is a deliberate act, and the levels are *levels* — they carry no position
size, which is still the missing half of the risk decision.

Full derivation, including the three traps in the data that any obvious
formula falls into: [RECOMMENDED-LEVELS.md](RECOMMENDED-LEVELS.md).

---

## 5b. Chitra's account

Chakravarti manages his sister's IRA. It appears in two places:

- **the `Chitra` tab** — her whole account, priced live
- **a lavender `CHITRA` row** inside any Dashboard block whose ticker she
  shares, so a move on a name you both hold is visible where you make it

**Her shares never touch your numbers.** The row is added when the Dashboard is
drawn, *after* your per-ticker TOTAL is worked out, so it cannot reach your
totals, your coverage flags, your reserves, or the order engine. The coverage
columns on her row are blank rather than `N` — there is no Schwab connection to
her account, so "no stop" is not something this system can honestly claim.

**To update it: send the statement.** The holdings live in
`chitra_holdings.csv` in the repo, not in the sheet — anything typed into the
tab is overwritten on the next cycle. The file header records what the
statement totalled, so if the tab's TOTAL stops matching it, a row was
mistyped.

---

## 6. Orders: what belongs in the sheet, and what does not

```
In the sheet:  "sell BE if it CLOSES below 214"      Schwab cannot express this
At Schwab:     "sell BE at 350"                      an ordinary limit order
```

A limit order fills the instant price *touches* the level, wick included. A
close trigger waits for the bar to finish, so a spike that reverses does not
fire it. That difference is the entire reason the sheet exists — anything Schwab
can already do belongs at Schwab, where it works whether or not a Raspberry Pi
is awake.

**A row is ten cells:**

```
Date  Acct  Ticker  Side  Close_Is  Trigger_Price  Limit_Price  Qty  Expires_On  Notes
      171   BE      SELL  BELOW     214                         23              trend break
```

- Leave **Row_ID blank** — Pi 1 stamps it, and that is what makes adding new
  rows at the top safe.
- Leave **Limit_Price blank** — Pi 1 derives one from the live bid/ask when it
  places. A price typed days ago is stale by the time the trigger fires.

**Every order is GTC LIMIT.** To exit regardless of price, set the limit *through*
the market rather than reaching for a market order: it fills immediately at the
best available price *and* puts a floor under a bad fill.

**It submits the NEXT session.** The daily bar has to finish before it can be
read, and by then the market is shut.

**A Row_ID is placed once.** To order again, use a new row.

---

## 7. Seed money: keeping capital with a ticker

Sell a stock and the cash disappears into the account. Seed money keeps it
earmarked.

```
fence  171 BE        ← register a holding you already own, no cash moves
```

From then on, **every sale of BE in account 171 credits its reserve** — Schwab
is monitored directly, so a sale you place by hand counts too. Buying it back
debits. `Free_To_Deploy` stops counting that money as available, so you can see
it is spoken for.

**It is an earmark, not a separate pot.** Nothing stops you spending it
elsewhere; the system makes the commitment visible, the discipline is yours.

`seed` is for money that did *not* come from selling that ticker. Using `seed`
for sale proceeds double-counts them, because the fill credits the same money.

Detail in [CASH_RESERVE_HANDOFF.md](CASH_RESERVE_HANDOFF.md).

---

## 8. What protects you

No signing key, by choice: requiring a laptop to sign each intent defeats a
phone-edited sheet, and sheet access can cause trades but cannot move money out
of the account. What stands in its place:

1. **`ORDER_ENGINE_LIVE`** — unset means validate and preview, never place
2. **The kill switch** — `/etc/myTrading/TRADING_DISABLED` blocks everything
3. **The account's own limits** — a sell cannot exceed shares held, a buy cannot
   exceed free cash. These scale with the portfolio, where a fixed dollar cap
   would block legitimate trades
4. **An account allowlist**, failing closed when unset
5. **Validation written back every cycle**, so a bad row says so in minutes
6. **A submissions-per-day cap** — the one guard against a runaway loop, which
   no position check can see
7. **Write-ahead to a ledger** before any order, so a crash mid-submit is
   detectable rather than silently repeatable

**The realistic failure is a mis-typed cell, not a break-in.** Guards 3 and 5
exist for that.

---

## 9. When something looks wrong

| symptom | look at |
|---|---|
| Sheet not updating | `LAST POLL` in the Orders header |
| A row does nothing | its `Validation` cell — it says why |
| Everything Schwab failing | token expiry — `token_status` in the ops sheet |
| P&L looks wrong | `PROJECT_PLAN.md` §1 — parts are known-approximate |
| Is it all still running? | `health_check.py` |

---

## 10. What is deliberately not automated

- **Deciding what to trade.** The system reports and executes; it does not choose.
- **Cancelling your own orders**, except one case: a resting sell that would
  starve a triggered exit of shares.
- **Reconciling a filled order back into the sheet** — a known gap, `PROJECT_PLAN` §4.
- **Placing anything from Pi 2.** By design.
