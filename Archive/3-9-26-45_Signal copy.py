#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
45_Signal.py - 45° Uptrend Finder with SIGNAL-Super-MOST-ADXR Integration

Command-line version that uses shared scanner_45 module.

Usage:
    python 45_Signal.py
    python 45_Signal.py
    python 45_Signal.py --pass-score 80

Outputs:
    - outputs/45_signal_full.csv    : All tickers that passed filters
    - outputs/45_signal_top.csv     : Tickers with Score >= pass_score
    - outputs/45_signal_mylist.csv  : Your MYLIST watchlist tickers
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Import shared scanner functions
from data.scanner_45 import (
    build_universe,
    scan_universe,
    DEFAULT_CONFIG,
)

# -----------------------
# STATIC CONFIGURATION
# (overridable via CLI args)
# -----------------------
UNIVERSE_SOURCES = ("SPX", "NDX")

# Additional tickers beyond SPX/NDX
MORE_TICKERS = [
    "SOFI", "OKTA", "MSTR", "BMNR", "SSK"
]

# Your permanent watchlist
MYLIST = [
    "TSLA", "MSTR", "NVDA", "GOOG", "PLTR", "BMNR", "BE", "MU", "IREN", "CRDO",
    "AVGO", "TSM", "SOFI", "AMD", "APP", "COIN", "LEU", "HOOD", "OKLO", "CRVW",
    "MP", "INOD", "CFG", "APLD", "AAOI", "CORZ", "UPXI", "STKE"
]
MYLIST_SET = set(MYLIST)


# -----------------------
# HELPER
# -----------------------
def safe_cols(df, wanted: list) -> list:
    """Return only columns that exist in df — prevents KeyError on old CSVs."""
    return [c for c in wanted if c in df.columns]


# -----------------------
# MAIN EXECUTION
# -----------------------
def main():

    # --- CLI arguments ---
    parser = argparse.ArgumentParser(
        description="45 degree Signal Scanner - Find high-quality uptrend stocks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 45_Signal.py                  # defaults: pass score 70
  python 45_Signal.py --pass-score 80  # stricter pass score
        """
    )
    parser.add_argument(
        "--pass-score",
        type=int,
        default=70,
        help="Minimum Score_Weighted for 'top' list (default: 70)"
    )
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config.update({
        "earnings_window_days": 7,
        "min_dollar_volume": 20_000_000,
        "scan_period": "420d",
    })
    pass_score = args.pass_score

    print("\n" + "=" * 80)
    print("45 DEGREE UPTREND FINDER WITH SIGNAL-Super-MOST-ADXR")
    print("=" * 80)
    print(f"  Scores     : 30d / 60d / 90d / 120d / Weighted (50-30-15-5%)")
    print(f"  Pass Score : {pass_score} (based on Score_Weighted)")
    print("=" * 80 + "\n")

    # --- Universe ---
    print("Building universe from S&P 500 and Nasdaq-100...")
    universe = build_universe(UNIVERSE_SOURCES, verbose=True)
    print(f"Loaded {len(universe)} tickers from index lists")

    print(f"Adding {len(MORE_TICKERS)} extra tickers: {MORE_TICKERS}")
    for t in MORE_TICKERS:
        if t not in universe:
            universe.append(t)
    universe = sorted(set(universe))
    print(f"Total universe size: {len(universe)} tickers\n")

    # --- Scan ---
    print("=" * 80)
    print(f"SCANNING {len(universe)} TICKERS")
    print("=" * 80)
    print("Legend: passed=OK  failed=no data  skipped=volume too low\n")

    df_all = scan_universe(universe, config=config, verbose=True)

    print("\n" + "=" * 80)
    print(f"SCAN COMPLETE: {len(df_all)} tickers passed all filters")
    print("=" * 80 + "\n")

    if df_all.empty:
        print("No tickers passed all filters. Exiting.")
        return

    # --- Save files ---
    outputs_dir = APP_DIR / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    print(f"Saving output files to {outputs_dir}/")

    full_path = outputs_dir / "45_signal_full.csv"
    df_all.to_csv(full_path, index=False)
    print(f"   Saved: {full_path.name}  ({len(df_all)} tickers)")

    # Sort by Score_Weighted for top list
    df_sorted = df_all.sort_values("Score_Weighted", ascending=False) if "Score_Weighted" in df_all.columns else df_all
    top_df = df_sorted[df_sorted["Score_Weighted"] >= pass_score].copy() if "Score_Weighted" in df_sorted.columns else df_sorted.head(0)
    top_path = outputs_dir / "45_signal_top.csv"
    top_df.to_csv(top_path, index=False)
    print(f"   Saved: {top_path.name}  ({len(top_df)} tickers, Score_Weighted >= {pass_score})")

    mylist_df = df_all[df_all["Ticker"].isin(MYLIST_SET)].copy()
    mylist_path = outputs_dir / "45_signal_mylist.csv"
    mylist_df.to_csv(mylist_path, index=False)
    print(f"   Saved: {mylist_path.name}  ({len(mylist_df)} tickers)")

    # --- Previews ---
    preview_cols = ["Ticker", "Score_Weighted", "Score_30", "Score_60", "Score_90", "Score_120",
                    "SIGNAL-Super-MOST-ADXR", "Earnings_Alert",
                    "Market_Cap", "Price", "RS_63d_vs_SPY_%", "Buy_Hint"]

    print("\n" + "=" * 80)
    print(f"TOP SCORERS  (Score_Weighted >= {pass_score}, top 20)")
    print("=" * 80)
    if len(top_df) > 0:
        print(top_df[safe_cols(top_df, preview_cols)].head(20).to_string(index=False))
    else:
        print(f"  (No tickers scored >= {pass_score})")

    print("\n" + "=" * 80)
    print("MYLIST WATCHLIST")
    print("=" * 80)
    if not mylist_df.empty:
        print(mylist_df[safe_cols(mylist_df, preview_cols)].to_string(index=False))
    else:
        print("  (No MyList tickers passed filters)")

    print("\n" + "=" * 80)
    print("ALL DONE - files saved to outputs/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()