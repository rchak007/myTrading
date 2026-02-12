# Historical P&L Tracking System for Schwab Accounts

## Overview

This system calculates **per-ticker profit & loss** across all your Schwab brokerage accounts, including:

- ✅ **Realized P&L** - Gains/losses from closed positions (sells)
- ✅ **Unrealized P&L** - Current market value vs. cost basis
- ✅ **Dividend Income** - Total dividends received per ticker
- ✅ **Grand Total P&L** - Aggregated across all accounts and tickers

### Key Features

🚀 **Smart Caching** - Historical transactions are cached locally. Only NEW transactions are fetched on subsequent runs, avoiding API rate limits.

📊 **Per-Ticker Analysis** - Know exactly which stocks/ETFs are your winners and losers.

🏦 **Multi-Account Support** - Aggregates across all your Schwab accounts automatically.

💾 **Exports to CSV** - Detailed reports saved for further analysis.

---

## Installation

### 1. Copy the enhanced files to your schwabAPI directory

```bash
# Navigate to your schwabAPI directory
cd /path/to/myTrading/schwabAPI

# Copy these files (Claude will provide them):
# - analytics_core_enhanced.py
# - historical_pl_report_enhanced.py
# - manage_cache.py
```

### 2. Verify dependencies

Your existing setup should already have:
- `schwabdev` (Schwab API client)
- `pandas`
- `python-dotenv`

If not:
```bash
pip install schwabdev pandas python-dotenv
```

### 3. Verify .env file

Ensure your `.env` file contains:
```
app_key=YOUR_SCHWAB_APP_KEY
app_secret=YOUR_SCHWAB_APP_SECRET
callback_url=https://127.0.0.1
```

---

## Quick Start

### First Run (Historical Data Collection)

```bash
python historical_pl_report_enhanced.py
```

**What happens:**
1. Fetches ALL transactions from 2022-01-01 to today
2. Caches them to `data/transactions/`
3. Calculates per-ticker P&L
4. Saves reports to CSV files

**Expected output:**
```
================================================================================
COMPREHENSIVE HISTORICAL P&L REPORT (Per-Ticker Analysis)
================================================================================

Analyzing transactions from 2022-01-01 through today...
Using smart caching - only new transactions will be fetched.

[INFO] Found 3 linked accounts
[INFO] Processing account 1234567...
[INFO] Fetching new transactions for <hash> from 2022-01-01 to 2025-02-02
[INFO] Fetched 150 tx for hash=<hash> from 2022-01-01 to 2022-03-01
...

================================================================================
GRAND TOTAL (All Accounts, All Tickers)
================================================================================
Total Cost Basis:      $125,450.00
Total Market Value:    $142,380.00
Total Unrealized P&L:  $12,230.00
Total Realized P&L:    $3,200.00
Total Dividends:       $1,500.00
--------------------------------------------------------------------------------
GRAND TOTAL P&L:       $16,930.00
GRAND TOTAL RETURN:    13.49%
================================================================================

PER-TICKER P&L (Aggregated Across All Accounts)
================================================================================
Symbol    Qty   Avg Price  Cost Basis  Market Value  Unrealized P&L  ...
AAPL    100.00     145.23    14,523.00     18,250.00       3,727.00  ...
SPY      50.00     412.50    20,625.00     23,100.00       2,475.00  ...
TSLA     25.00     198.40     4,960.00      6,125.00       1,165.00  ...
...
```

### Subsequent Runs (Incremental Updates)

```bash
python historical_pl_report_enhanced.py
```

**What happens:**
1. Checks cached data
2. Only fetches NEW transactions since last run
3. Updates P&L calculations
4. Saves updated reports

**Much faster!** Only queries for new data.

---

## Configuration

### Change Historical Start Date

Edit `historical_pl_report_enhanced.py`:

```python
# Line ~40
start_date = "2020-01-01"  # Change this to your desired start date
```

### Change Transaction Types

Edit `analytics_core_enhanced.py` if you want to include/exclude specific transaction types:

```python
# Line ~92-118
types_list = [
    "TRADE",                    # Buy/sell transactions
    "DIVIDEND_OR_INTEREST",     # Dividend payments
    "CORPORATE_ACTION",         # Stock splits, mergers, etc.
    # ... add/remove as needed
]
```

---

## Cache Management

### View Cache Status

```bash
python manage_cache.py list
```

Output:
```
================================================================================
CACHED TRANSACTION FILES
================================================================================

Found 3 transaction cache files:

  Account Hash: 4BAEBA8E79943...
  Cache File:   4BAEBA8E79943....csv
  Size:         245.3 KB
  Last Date:    2025-02-02

  ...
```

### Clear All Cache (Start Fresh)

```bash
python manage_cache.py clear-all
```

⚠️ **WARNING:** This deletes ALL cached transactions. You'll need to re-fetch historical data on next run.

### Clear Cache for Specific Account

```bash
python manage_cache.py clear <account_hash>
```

### Interactive Mode

```bash
python manage_cache.py
```

Provides a menu-driven interface for cache management.

---

## Output Files

All reports are saved to:

### 1. `data/per_ticker_pl.csv`

Per-ticker P&L aggregated across all accounts:

| symbol | qty_current | avg_price | cost_basis | market_value | unrealized_pl | realized_pl | dividends | total_pl | total_pl_pct |
|--------|------------|-----------|------------|--------------|---------------|-------------|-----------|----------|--------------|
| AAPL   | 100.00     | 145.23    | 14,523.00  | 18,250.00    | 3,727.00      | 500.00      | 120.00    | 4,347.00 | 29.93        |

### 2. `data/per_account_ticker_pl.csv`

Per-account, per-ticker breakdown:

| account_number | symbol | qty_current | cost_basis | market_value | total_pl |
|----------------|--------|------------|------------|--------------|----------|
| 1234567        | AAPL   | 50.00      | 7,261.50   | 9,125.00     | 1,863.50 |
| 1234567        | SPY    | 30.00      | 12,375.00  | 13,860.00    | 1,485.00 |

### 3. `data/pl_summary.json`

Summary statistics in JSON format:

```json
{
  "total_cost_basis": 125450.0,
  "total_market_value": 142380.0,
  "total_unrealized_pl": 12230.0,
  "total_realized_pl": 3200.0,
  "total_dividends": 1500.0,
  "grand_total_pl": 16930.0,
  "grand_total_pl_pct": 13.49
}
```

---

## Understanding the Calculations

### Cost Basis Tracking (FIFO)

The system uses **FIFO (First In, First Out)** for calculating realized gains:

**Example:**
```
1. Buy 100 AAPL @ $150 = $15,000 cost basis
2. Buy 50 AAPL @ $160  = $8,000 cost basis
3. Sell 80 AAPL @ $170

Realized P&L calculation:
- Sell 80 shares (FIFO: first 80 from the 100 bought at $150)
- Realized: (170 - 150) * 80 = $1,600
- Remaining position: 20 @ $150 + 50 @ $160
```

### Unrealized P&L

```
Current Market Value - Cost Basis of Holdings
```

### Total P&L

```
Total P&L = Realized P&L + Unrealized P&L + Dividends
```

### Return %

```
Return % = (Total P&L / Cost Basis) * 100
```

---

## Troubleshooting

### Issue: "No trading data found"

**Causes:**
1. No trades in the date range
2. Cache is corrupted
3. API authentication failed

**Solutions:**
```bash
# Clear cache and try again
python manage_cache.py clear-all
python historical_pl_report_enhanced.py

# Check .env file has correct credentials
cat .env
```

### Issue: "HTTP 429 - Rate Limit"

**Solution:** The caching system should prevent this, but if it happens:

```bash
# Wait 15-30 minutes, then run again
# The cache will pick up where it left off
python historical_pl_report_enhanced.py
```

### Issue: Missing transactions

**Solution:** Force re-fetch from specific date:

```python
# In historical_pl_report_enhanced.py, change start_date
start_date = "2021-01-01"  # Earlier date

# Then clear cache and re-run
python manage_cache.py clear-all
python historical_pl_report_enhanced.py
```

### Issue: Incorrect realized P&L

**Causes:**
1. Corporate actions (splits, mergers) not handled
2. Short sales mixed with long positions

**Solutions:**
1. Check `trade_history` in the tracker for debugging
2. Manually verify against Schwab statements
3. Corporate actions may need manual adjustment

---

## Advanced Usage

### Export to Excel for Further Analysis

```python
import pandas as pd

# Read the CSV files
per_ticker = pd.read_csv("data/per_ticker_pl.csv")
per_account = pd.read_csv("data/per_account_ticker_pl.csv")

# Export to Excel with multiple sheets
with pd.ExcelWriter("portfolio_pl_analysis.xlsx") as writer:
    per_ticker.to_excel(writer, sheet_name="By Ticker", index=False)
    per_account.to_excel(writer, sheet_name="By Account", index=False)
```

### Filter for Specific Ticker

```python
from analytics_core_enhanced import calculate_comprehensive_pl
from portfolio_snapshot import make_client

client = make_client()
per_ticker_df, _, _ = calculate_comprehensive_pl(client, start_date="2022-01-01")

# Filter for AAPL only
aapl_data = per_ticker_df[per_ticker_df["symbol"] == "AAPL"]
print(aapl_data.to_string())
```

### Calculate Tax-Loss Harvesting Opportunities

```python
# Find positions with unrealized losses
per_ticker_df = pd.read_csv("data/per_ticker_pl.csv")
losses = per_ticker_df[per_ticker_df["unrealized_pl"] < -1000]  # >$1k loss
print("Tax-loss harvesting opportunities:")
print(losses[["symbol", "unrealized_pl", "market_value"]])
```

---

## Architecture Overview

```
historical_pl_report_enhanced.py
    │
    ├─→ analytics_core_enhanced.py
    │   ├─→ get_linked_accounts()
    │   ├─→ get_transactions_cached()  [smart caching]
    │   ├─→ build_per_ticker_pl()      [FIFO tracking]
    │   └─→ calculate_comprehensive_pl()
    │
    └─→ portfolio_snapshot.py
        ├─→ make_client()
        └─→ get_positions_all_accounts()

Data Flow:
1. Fetch account list
2. For each account:
   - Check cache for existing transactions
   - Fetch only NEW transactions from API
   - Update cache
3. Process transactions:
   - Track buys/sells (FIFO)
   - Record dividends
   - Calculate realized P&L
4. Get current positions from API
5. Calculate unrealized P&L
6. Aggregate results
```

---

## Future Enhancements

Potential additions:
- [ ] Support for options trading P&L
- [ ] Tax reporting (short-term vs long-term gains)
- [ ] Performance attribution (sector/asset class)
- [ ] Benchmark comparison (S&P 500, etc.)
- [ ] Web dashboard (Streamlit app)

---

## Support

For issues or questions:
1. Check cached files: `python manage_cache.py list`
2. Review log output for errors
3. Verify API credentials in `.env`
4. Check Schwab API status: https://developer.schwab.com/

---

## License

This is an extension of your existing Schwab trading system. Use at your own risk. Verify all P&L calculations against official Schwab statements.