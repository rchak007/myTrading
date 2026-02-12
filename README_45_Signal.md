# 45_Signal.py - Enhanced Stock Scanner

## Overview

`45_Signal.py` combines two powerful analysis systems:
1. **45° Trend Analysis**: Identifies stocks with ideal ~100% annual growth trends
2. **SIGNAL-Super-MOST-ADXR**: Real-time trading signals from Supertrend, MOST, and ADXR indicators

This creates a comprehensive view: which stocks have high-quality trends AND actionable trading signals.

## What Makes This Different from 45.py

### Original 45.py (Google Colab)
- Pure 45° trend analysis
- Designed for Colab with Streamlit UI
- Focuses on trend quality scoring only

### New 45_Signal.py (Standalone Python)
- ✅ Works as a standalone Python program (no Colab needed)
- ✅ Integrates with your myTrading system
- ✅ Adds SIGNAL-Super-MOST-ADXR column
- ✅ Uses existing indicator infrastructure
- ✅ Same folder structure as app.py

## Installation & Requirements

### Prerequisites
Your myTrading repo already has most dependencies. Just ensure:

```bash
pip install yfinance pandas numpy scipy
```

### File Location
Place `45_Signal.py` in:
```
/home/rchak007/github/myTrading/45_Signal.py
```
(Same directory as `app.py`)

## Usage

### Basic Run
```bash
cd /home/rchak007/github/myTrading
python 45_Signal.py
```

### Output Files
Creates 3 CSV files in `outputs/`:

1. **45_signal_full.csv**
   - All tickers that passed volume filter
   - Sorted by Score (highest first)

2. **45_signal_top.csv** 
   - Only tickers with Score >= 70
   - Your strongest trend candidates

3. **45_signal_mylist.csv**
   - Your watchlist tickers (from MYLIST)
   - Regardless of score

## Understanding the Output

### Key Columns

| Column | Description | Good Values |
|--------|-------------|-------------|
| **Score** | 45° trend quality (0-100) | 70+ excellent, 50-69 good |
| **SIGNAL-Super-MOST-ADXR** | Combined trading signal | BUY, HOLD, EXIT |
| **Earnings_Alert** | Earnings within 7 days | Empty = safe |
| **Price** | Current price | - |
| **Supertrend_Signal** | Supertrend indicator | BUY/SELL |
| **MOST_Signal** | MOST indicator | BUY/SELL |
| **ADXR_State** | Trend strength | STRONG/FLAT |
| **RS_63d_vs_SPY_%** | Outperformance vs SPY | Positive = beating market |
| **Dist_to_21EMA_%** | Distance to 21-day EMA | -1 to +3% = buy zone |
| **Buy_Hint** | Entry point suggestions | "21-EMA ready" etc. |

### Signal Combinations

The program shows you both:
- **Trend Quality (Score)**: Is this a high-quality uptrend?
- **Trading Signal**: Should I buy/hold/sell RIGHT NOW?

**Ideal Setup**:
```
Score: 75+
SIGNAL-Super-MOST-ADXR: BUY
Earnings_Alert: (blank)
Dist_to_21EMA_%: 0.5%
Buy_Hint: "21-EMA ready"
```

## Configuration

Edit these variables at the top of `45_Signal.py`:

```python
# Filtering
EARNINGS_WINDOW_DAYS = 7        # Earnings proximity alert
MIN_DOLLAR_VOLUME = 20_000_000  # Liquidity filter
PASS_SCORE = 70                 # "Top" list threshold

# Universe
UNIVERSE_SOURCES = ("SPX", "NDX")  # S&P 500 + Nasdaq-100

# Custom tickers to always include
MORE_TICKERS = [
    "SOFI", "OKTA", "MSTR", "BMNR", "SSK"
]

# Your permanent watchlist
MYLIST = [
    "TSLA", "MSTR", "NVDA", "GOOG", "PLTR", ...
]
```

## Example Output

```
[1/650] Processing AAPL...
  📈 AAPL: Score=65, Signal=BUY, Price=$180.50

[2/650] Processing TSLA...
  📈 TSLA: Score=82, Signal=BUY, Price=$245.20 🔴 EARNINGS!

[3/650] Processing BMNR...
  ⏭️  BMNR: Failed volume filter ($5,234,567 < $20,000,000)

================================================================
TOP SCORERS (PREVIEW)
================================================================
Ticker  Score  SIGNAL-Super-MOST-ADXR  Earnings_Alert    Price  RS_63d_vs_SPY_%  Buy_Hint
MNST       90                     BUY                  110.50             15.2  21-EMA ready
TSM        85                     BUY                  165.30             12.8               
NVDA       82                    HOLD                  920.45             18.5               
TSLA       82                     BUY  🔴 EARNINGS SOON  245.20             22.1  50-SMA ready
```

## How It Works Under the Hood

### Data Flow
1. **Fetch Universe**: Download S&P 500 + Nasdaq-100 + your tickers
2. **Volume Filter**: Remove illiquid stocks
3. **45° Analysis**: Calculate trend quality score (uses `stock_scoring.py`)
4. **Signal Analysis**: Calculate trading signals (uses `core.indicators`, `core.signals`)
5. **Combine Results**: Merge both analyses
6. **Output**: Save 3 CSV files + preview

### Integration Points
The program imports from your existing modules:

```python
from core.utils import _fix_yf_cols
from core.indicators import apply_indicators
from core.signals import signal_super_most_adxr
from data.stock_scoring import calculate_45_degree_score, get_earnings_alert
```

This ensures consistency with your Streamlit app.

## Common Use Cases

### Case 1: Find New Trade Ideas
```bash
python 45_Signal.py
# Review: outputs/45_signal_top.csv
# Look for: Score 70+, SIGNAL=BUY, no earnings alert
```

### Case 2: Check Your Watchlist
```bash
python 45_Signal.py
# Review: outputs/45_signal_mylist.csv
# See how your holdings score + current signals
```

### Case 3: Daily Quick Scan
```bash
# Add to crontab for daily 9am scan:
0 9 * * * cd /home/rchak007/github/myTrading && python 45_Signal.py
```

## Troubleshooting

### "No module named 'core'"
**Solution**: Run from myTrading directory:
```bash
cd /home/rchak007/github/myTrading
python 45_Signal.py
```

### "Failed to load S&P 500"
**Cause**: Wikipedia/website blocking
**Solution**: Check internet connection, or edit universe sources

### Long Runtime
**Normal**: Scanning 600+ tickers takes 10-15 minutes
**Speed up**: Reduce universe or add to MORE_TICKERS only:
```python
UNIVERSE_SOURCES = ()  # Skip SPX/NDX
MORE_TICKERS = ["AAPL", "TSLA", "NVDA"]  # Just scan these
```

## Comparison Table

| Feature | 45.py (Original) | 45_Signal.py (New) |
|---------|------------------|-------------------|
| Platform | Google Colab | Standalone Python |
| UI | Streamlit (ngrok) | CSV output |
| Scoring | ✅ 45° trend score | ✅ 45° trend score |
| Trading Signals | ❌ No | ✅ SIGNAL-Super-MOST-ADXR |
| Integration | Standalone | Uses myTrading modules |
| Automation | Manual Colab run | Can be automated |
| Output | CSV + UI | CSV files |

## Next Steps

1. **First Run**: Test with small universe
   ```python
   UNIVERSE_SOURCES = ()
   MORE_TICKERS = ["AAPL", "TSLA", "NVDA"]
   ```

2. **Review Output**: Check `outputs/45_signal_full.csv`

3. **Customize**: Adjust MYLIST to your holdings

4. **Automate**: Add to cron for daily scans

5. **Integrate**: Use output in your trading workflow

## Tips

- **High Score + BUY Signal + No Earnings** = Strong entry candidate
- **Score dropped 20+ points** = Trend degrading, consider exit
- **BUY_HINT present** = Price near key support, good entry
- **Earnings Alert** = Wait until after earnings to enter
- **LOW Score but BUY Signal** = Short-term trade, not long-term hold

---

**Questions?** Check the original modules:
- Scoring logic: `data/stock_scoring.py`
- Signal logic: `core/signals.py`
- Indicator logic: `core/indicators.py`