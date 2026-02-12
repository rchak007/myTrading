"""
data/scanner_45.py - Shared 45° Signal Scanner Functions

This module contains all the scanning logic used by both:
- 45_Signal.py (CLI version)
- app_45_signal.py (Streamlit version)

By centralizing the code here, we ensure that any modifications
only need to be made in ONE place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from core.utils import _fix_yf_cols
from core.indicators import apply_indicators
from core.signals import signal_super_most_adxr
from data.stock_scoring import calculate_45_degree_score, get_earnings_alert


# -----------------------
# CONFIGURATION DEFAULTS
# -----------------------
DEFAULT_CONFIG = {
    "atr_period": 10,
    "atr_multiplier": 3.0,
    "rsi_period": 14,
    "vol_lookback": 20,
    "adxr_len": 14,
    "adxr_lenx": 14,
    "adxr_low_threshold": 20.0,
    "adxr_flat_eps": 1e-6,
    "scan_period": "420d",
    "scan_interval": "1d",
    "earnings_window_days": 7,
    "min_dollar_volume": 20_000_000,
}


# -----------------------
# HELPER FUNCTIONS
# -----------------------
def avg_dollar_vol(df: pd.DataFrame, days: int = 30) -> float:
    """
    Calculate average dollar volume over last N sessions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV dataframe with Close and Volume columns
    days : int
        Number of days to average over
        
    Returns:
    --------
    float : Average (Close * Volume) or np.nan if insufficient data
    """
    if "Volume" not in df.columns:
        return np.nan
    c = df["Close"].tail(days)
    v = df["Volume"].tail(days)
    if len(c.dropna()) < days or len(v.dropna()) < days:
        return np.nan
    dv = (c * v).mean()
    return float(dv.iloc[0] if hasattr(dv, 'iloc') else dv)


# -----------------------
# UNIVERSE FETCHING
# -----------------------
def get_spx_tickers(verbose: bool = False) -> list[str]:
    """
    Get S&P 500 tickers from Wikipedia.
    
    Parameters:
    -----------
    verbose : bool
        If True, print error messages
    
    Returns:
    --------
    list[str] : List of ticker symbols, or empty list if fetch fails
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        hdrs = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        }
        tbls = pd.read_html(url, storage_options=hdrs)
        df = tbls[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        if verbose:
            print(f"✅ Fetched {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        if verbose:
            print(f"❌ Failed to fetch S&P 500: {e}")
        return []


def get_ndx_tickers(verbose: bool = False) -> list[str]:
    """
    Get Nasdaq-100 tickers from Wikipedia.
    
    Parameters:
    -----------
    verbose : bool
        If True, print error messages
    
    Returns:
    --------
    list[str] : List of ticker symbols, or empty list if fetch fails
    """
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        hdrs = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        }
        tbls = pd.read_html(url, storage_options=hdrs)
        for idx, tbl in enumerate(tbls):
            if 'Ticker' in tbl.columns:
                tickers = tbl["Ticker"].str.replace(".", "-", regex=False).tolist()
                if verbose:
                    print(f"✅ Fetched {len(tickers)} Nasdaq-100 tickers")
                return tickers
            elif 'Symbol' in tbl.columns:
                tickers = tbl["Symbol"].str.replace(".", "-", regex=False).tolist()
                if verbose:
                    print(f"✅ Fetched {len(tickers)} Nasdaq-100 tickers")
                return tickers
    except Exception as e:
        if verbose:
            print(f"❌ Failed to fetch Nasdaq-100: {e}")
        return []
    return []


def build_universe(sources: tuple[str, ...] = ("SPX", "NDX"), verbose: bool = False) -> list[str]:
    """
    Combine tickers from chosen sources, removing duplicates.
    
    Parameters:
    -----------
    sources : tuple[str, ...]
        Which universes to include ("SPX", "NDX")
    verbose : bool
        If True, print status messages
        
    Returns:
    --------
    list[str] : Sorted unique list of tickers
    """
    tickers = []
    if "SPX" in sources:
        spx = get_spx_tickers(verbose=verbose)
        tickers.extend(spx)
    if "NDX" in sources:
        ndx = get_ndx_tickers(verbose=verbose)
        tickers.extend(ndx)
    
    unique_tickers = sorted(set(tickers))
    if verbose and unique_tickers:
        print(f"Total unique tickers: {len(unique_tickers)}")
    
    return unique_tickers


# -----------------------
# CORE SCANNING FUNCTION
# -----------------------
def scan_ticker(
    ticker: str,
    spy_close: pd.Series | None = None,
    config: dict | None = None,
    verbose: bool = True
) -> dict | None:
    """
    Scan a single ticker for 45° trend quality + trading signals.
    
    This is the SINGLE SOURCE OF TRUTH for scanning logic.
    Both CLI and Streamlit versions call this function.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    spy_close : pd.Series, optional
        SPY close prices for relative strength calculation
    config : dict, optional
        Configuration parameters (uses DEFAULT_CONFIG if None)
    verbose : bool
        If True, print progress messages (for CLI)
        
    Returns:
    --------
    dict : Scan results with all metrics, or None if ticker fails filters
    """
    # Use default config if not provided
    if config is None:
        config = DEFAULT_CONFIG
    
    # Extract config values
    scan_period = config.get("scan_period", DEFAULT_CONFIG["scan_period"])
    scan_interval = config.get("scan_interval", DEFAULT_CONFIG["scan_interval"])
    min_dollar_vol = config.get("min_dollar_volume", DEFAULT_CONFIG["min_dollar_volume"])
    earnings_window = config.get("earnings_window_days", DEFAULT_CONFIG["earnings_window_days"])
    
    atr_period = config.get("atr_period", DEFAULT_CONFIG["atr_period"])
    atr_multiplier = config.get("atr_multiplier", DEFAULT_CONFIG["atr_multiplier"])
    rsi_period = config.get("rsi_period", DEFAULT_CONFIG["rsi_period"])
    vol_lookback = config.get("vol_lookback", DEFAULT_CONFIG["vol_lookback"])
    adxr_len = config.get("adxr_len", DEFAULT_CONFIG["adxr_len"])
    adxr_lenx = config.get("adxr_lenx", DEFAULT_CONFIG["adxr_lenx"])
    adxr_low_threshold = config.get("adxr_low_threshold", DEFAULT_CONFIG["adxr_low_threshold"])
    adxr_flat_eps = config.get("adxr_flat_eps", DEFAULT_CONFIG["adxr_flat_eps"])
    
    # Download data
    try:
        df = yf.download(ticker, period=scan_period, interval=scan_interval,
                        progress=False, auto_adjust=False)
        if df.empty:
            if verbose:
                print(f"  ❌ {ticker}: No data returned")
            return None

        # Fix yfinance column issues
        df = _fix_yf_cols(df)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        
        if len(df) < 120:
            if verbose:
                print(f"  ❌ {ticker}: Insufficient data (only {len(df)} bars)")
            return None
            
    except Exception as e:
        if verbose:
            print(f"  ❌ {ticker}: Download failed - {e}")
        return None

    # Volume filter
    dv = avg_dollar_vol(df, days=30)
    if pd.isna(dv) or dv < min_dollar_vol:
        if verbose:
            print(f"  ⏭️  {ticker}: Failed volume filter (${dv:,.0f} < ${min_dollar_vol:,.0f})")
        return None

    # Calculate 45° trend score
    try:
        base = df[["High", "Low", "Close", "Volume"]].copy()
        score_data = calculate_45_degree_score(base, spy_close)
        score = score_data["score"]
        earnings_alert = get_earnings_alert(ticker, earnings_window)
    except Exception as e:
        if verbose:
            print(f"  ⚠️  {ticker}: Score calculation failed - {e}")
        score = 0
        score_data = {}
        earnings_alert = ""

    # Calculate trading signals
    try:
        df_indicators = apply_indicators(
            df,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier,
            rsi_period=rsi_period,
            vol_lookback=vol_lookback,
            adxr_len=adxr_len,
            adxr_lenx=adxr_lenx,
            adxr_low_threshold=adxr_low_threshold,
            adxr_flat_eps=adxr_flat_eps,
        )
        
        last = df_indicators.iloc[-1]
        
        # Get individual signals
        st_sig = str(last.get("Supertrend_Signal", "SELL"))
        most_sig = str(last.get("MOST_Signal", "SELL"))
        adxr_state = str(last.get("ADXR_State", "FLAT"))
        
        # Calculate combined signal
        signal = signal_super_most_adxr(st_sig, most_sig, adxr_state)
        
    except Exception as e:
        if verbose:
            print(f"  ⚠️  {ticker}: Signal calculation failed - {e}")
        signal = "N/A"
        st_sig = "N/A"
        most_sig = "N/A"
        adxr_state = "N/A"

    if verbose:
        print(f"  📈 {ticker}: Score={score}, Signal={signal}, Price=${df['Close'].iloc[-1]:.2f}"
              f"{' 🔴 EARNINGS!' if earnings_alert else ''}")

    # Build result dictionary - SINGLE SOURCE OF TRUTH for output format
    return {
        "Ticker": ticker,
        "Score": int(score),
        "SIGNAL-Super-MOST-ADXR": signal,
        "Earnings_Alert": earnings_alert,
        "Price": float(df["Close"].iloc[-1]),
        "Supertrend_Signal": st_sig,
        "MOST_Signal": most_sig,
        "ADXR_State": adxr_state,
        "Slope120": score_data.get("slope120", np.nan),
        "R²_120": score_data.get("r2_120", np.nan),
        "Slope60": score_data.get("slope60", np.nan),
        "R²_60": score_data.get("r2_60", np.nan),
        "Angle°": score_data.get("angle_deg", np.nan),
        "Ret_63d_%": score_data.get("ret_63d", np.nan),
        "RS_63d_vs_SPY_%": score_data.get("rs_63d_vs_spy", np.nan),
        "MaxDD_120d_%": score_data.get("max_dd_120d", np.nan),
        "%Above21_60d": score_data.get("perc_above_21", np.nan),
        "%Above50_60d": score_data.get("perc_above_50", np.nan),
        "Dist_to_21EMA_%": score_data.get("dist_to_21ema", np.nan),
        "Dist_to_50SMA_%": score_data.get("dist_to_50sma", np.nan),
        "Dist_to_200SMA_%": score_data.get("dist_to_200sma", np.nan),
        "AvgDollarVol30d": float(dv),
        "Buy_Hint": " & ".join(score_data.get("buy_hints", [])),
    }


def fetch_spy_data(scan_period: str = "420d") -> pd.Series | None:
    """
    Fetch SPY close prices for relative strength calculations.
    
    Parameters:
    -----------
    scan_period : str
        Period to fetch (e.g., "420d")
        
    Returns:
    --------
    pd.Series : SPY close prices, or None if fetch fails
    """
    try:
        spy = yf.download("SPY", period=scan_period, interval="1d",
                         progress=False, auto_adjust=False)
        spy = _fix_yf_cols(spy)
        return spy["Close"]
    except Exception:
        return None


def scan_universe(
    tickers: list[str],
    config: dict | None = None,
    verbose: bool = True,
    progress_callback=None
) -> pd.DataFrame:
    """
    Scan multiple tickers and return results as DataFrame.
    
    Parameters:
    -----------
    tickers : list[str]
        List of ticker symbols to scan
    config : dict, optional
        Configuration parameters
    verbose : bool
        Print progress messages
    progress_callback : callable, optional
        Function to call with progress updates (i, total)
        Useful for Streamlit progress bars
        
    Returns:
    --------
    pd.DataFrame : Scan results sorted by Score descending
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    scan_period = config.get("scan_period", DEFAULT_CONFIG["scan_period"])
    
    # Fetch SPY once for all tickers
    if verbose:
        print("🔍 Fetching SPY data for relative strength...")
    
    spy_close = fetch_spy_data(scan_period)
    
    if spy_close is not None and verbose:
        print(f"✅ SPY data loaded: {len(spy_close)} bars\n")
    elif verbose:
        print("⚠️  SPY data unavailable, relative strength will be N/A\n")
    
    # Scan all tickers
    results = []
    total = len(tickers)
    
    for i, ticker in enumerate(tickers, 1):
        if verbose:
            print(f"[{i}/{total}] Processing {ticker}...")
        
        # Call progress callback if provided (for Streamlit)
        if progress_callback:
            progress_callback(i, total)
        
        res = scan_ticker(ticker, spy_close, config, verbose)
        if res:
            results.append(res)
        
        if verbose:
            print()  # Blank line
    
    if not results:
        return pd.DataFrame()
    
    # Create and sort DataFrame
    df = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
    return df