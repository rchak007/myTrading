# Quick Start Guide - Streamlit 45° Signal Scanner

## Installation

### 1. Ensure you have Streamlit installed
```bash
pip install streamlit
```

### 2. Place the file in your myTrading directory
```bash
# File should be at:
/home/rchak007/github/myTrading/app_45_signal.py
```

## Running the App

### Start the Streamlit server
```bash
cd /home/rchak007/github/myTrading
streamlit run app_45_signal.py
```

The app will open in your browser automatically at `http://localhost:8501`

---

## How to Use

### 1️⃣ Configure Settings (Sidebar)

**Scan Period**: How much historical data to analyze
- 360d = ~1 year
- 420d = ~1.5 years (default, recommended)
- 540d = ~2 years

**Min Dollar Volume**: Liquidity filter
- Default: $20M (good balance)
- Increase for large caps only
- Decrease to include small caps

**Earnings Window**: Alert if earnings within N days
- Default: 7 days

**Pass Score**: Threshold for "top scorers"
- Default: 70 (high quality only)
- Lower to 60 for more results

### 2️⃣ Add Your Tickers

In the sidebar "Tickers to Scan" box:
```
TSLA
NVDA
AAPL
MSFT
...
```

Or comma-separated:
```
TSLA, NVDA, AAPL, MSFT
```

**Watchlist** (tracked separately):
- Your holdings you want to monitor

### 3️⃣ Run the Scan

1. Go to **"🔍 Scan"** tab
2. Click **"🚀 Run Scan"**
3. Wait for progress bar to complete
4. View results in preview

### 4️⃣ Review Results

**📊 Full Tab**
- All scanned tickers
- Filter by score and signal
- Download filtered results

**⭐ Top Tab**
- Only high scorers (>= threshold)
- Best setups highlighted

**📋 Watch Tab**
- Your watchlist specifically
- Action recommendations shown

**📖 Guide Tab**
- Quick reference
- Column meanings
- Ideal setups

---

## Features

### ✅ Real-time Scanning
- Progress bar shows status
- Live ticker-by-ticker updates
- Results saved to CSV automatically

### 🎨 Color Coding
- 🟢 Green = BUY signals, high scores
- 🟡 Yellow = HOLD signals, medium scores
- 🔴 Red = EXIT signals, low scores

### 📊 Interactive Filtering
- Slider for minimum score
- Multi-select for signals
- Dynamic result count

### 💾 Auto-save Results
Files saved to `outputs/`:
- `45_signal_full.csv` - All results
- `45_signal_top.csv` - Top scorers
- `45_signal_mylist.csv` - Your watchlist

### 📥 Download Capability
- Download filtered results
- CSV format for Excel/Sheets
- One-click export

---

## Example Workflow

### Morning Routine
1. **Start app**: `streamlit run app_45_signal.py`
2. **Scan watchlist** (10 tickers, ~30 seconds)
3. **Check Watch tab** for signals on holdings
4. **Review Top tab** for new opportunities

### Weekend Deep Dive
1. **Add 50+ tickers** to scan list
2. **Run full scan** (~5 minutes)
3. **Filter**: Score >= 75, Signal = BUY
4. **Download results** for deeper analysis

### New Stock Research
1. **Add candidate tickers** to scan
2. **Compare scores** side-by-side
3. **Check earnings alerts**
4. **Verify signals align**

---

## Tips & Tricks

### ⚡ Fast Scans
- Keep ticker list under 20 for quick results
- Use watchlist feature for daily checks

### 🎯 Find Best Setups
In "Full" tab, filter for:
- Min Score: 75
- Signals: BUY only
- Then sort by RS_63d_vs_SPY_%

### 🔍 Monitor Holdings
Add all holdings to watchlist
- Quick check if any EXIT signals
- Track score changes over time

### 📊 Sector Comparison
Scan all semis, all tech, etc.
- Compare scores within sector
- Find relative strength leaders

### ⏰ Automated Daily Scan
Create a script that runs scan + sends results:
```bash
#!/bin/bash
cd /home/rchak007/github/myTrading
python 45_Signal.py  # Command-line version
# Then email/slack the CSVs
```

---

## Keyboard Shortcuts

While in Streamlit:
- **R** = Rerun app (refresh data)
- **C** = Clear cache
- **Cmd/Ctrl + K** = Open command palette

---

## Troubleshooting

### App won't start
```bash
# Check Streamlit is installed
pip install streamlit --upgrade

# Check you're in right directory
pwd  # Should show: /home/rchak007/github/myTrading

# Try explicit path
streamlit run /home/rchak007/github/myTrading/app_45_signal.py
```

### "Module not found" errors
```bash
# Make sure you're in myTrading directory
cd /home/rchak007/github/myTrading

# Check imports are available
python -c "from core.signals import signal_super_most_adxr"
```

### Slow scanning
- Reduce number of tickers
- Close other browser tabs
- Check internet connection (yfinance downloads)

### Results look wrong
- Clear cache: Press **C** in app
- Rerun scan fresh
- Check CSV files in outputs/ manually

---

## Advanced Usage

### Custom Indicator Parameters
Edit at top of `app_45_signal.py`:
```python
ATR_PERIOD = 10        # Change to 14
ATR_MULTIPLIER = 3.0   # Change to 2.5
PASS_SCORE = 70        # Or make it a sidebar input
```

### Add New Columns
In `scan_ticker()` function, add to return dict:
```python
return {
    # ... existing columns ...
    "MyCustomMetric": my_value,
}
```

### Change Color Thresholds
In `style_dataframe()`:
```python
def color_score(val):
    if val >= 85:  # Changed from 80
        return "background-color: #90EE90"
```

### Multi-page App
Create additional pages:
```bash
app_45_signal.py          # Main page
pages/
  1_📊_Analysis.py        # Additional page
  2_📈_Charts.py          # Another page
```

Streamlit auto-detects pages/ folder!

---

## Comparison: Streamlit vs Command-line

| Feature | Streamlit (app_45_signal.py) | CLI (45_Signal.py) |
|---------|------------------------------|-------------------|
| Interface | Web browser | Terminal |
| Interactivity | ✅ Filters, sorting | ❌ Static CSVs |
| Speed | Slower (UI rendering) | Faster (no UI) |
| Convenience | ✅ Visual, intuitive | Text-based |
| Automation | Hard to automate | ✅ Easy (cron jobs) |
| Best for | Manual analysis | Automated scans |

**Recommendation**: 
- Use **Streamlit** for interactive exploration
- Use **CLI** for automated daily scans

---

## Screenshots Walkthrough

### Main Interface
```
┌─────────────────────────────────────────────┐
│  Sidebar                │  Main Content     │
│  ┌───────────────┐     │  ┌──────────────┐ │
│  │ Configuration │     │  │ Scan Results │ │
│  │ - Period      │     │  │ - Top 10     │ │
│  │ - Filters     │     │  │ - Metrics    │ │
│  ├───────────────┤     │  └──────────────┘ │
│  │ Tickers       │     │                    │
│  │ - Input box   │     │  [Tab Navigation]  │
│  │ - Watchlist   │     │  Scan|Full|Top...  │
│  └───────────────┘     │                    │
└─────────────────────────────────────────────┘
```

### After Scanning
Results appear in tabs with color-coded signals and scores.

---

## Next Steps

1. **Try it**: Run a small scan (5-10 tickers)
2. **Explore tabs**: See different views of data
3. **Filter results**: Play with sliders/dropdowns
4. **Customize**: Edit ticker lists in sidebar
5. **Integrate**: Use alongside your existing app.py

---

**Pro Tip**: Keep both `app.py` (your main app) and `app_45_signal.py` (this scanner) running in different browser tabs. Switch between them for comprehensive analysis!

---

## Questions?

- Check the **Guide tab** in the app itself
- Review `README_45_Signal.md` for concepts
- See `TRADING_GUIDE.md` for decision framework
- Read `CONFIG_EXAMPLES.md` for usage patterns