# data/stock_scoring.py
"""
Stock scoring system based on 45° uptrend analysis.
Extracts scoring logic and earnings detection from the 45.py notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
import datetime as dt


# -----------------------
# Configuration
# -----------------------
EARNINGS_WINDOW_DAYS = 7  # Flag earnings within this many days


# -----------------------
# Helper Functions
# -----------------------
def ema(s: pd.Series, n: int) -> pd.Series:
    """Exponential Moving Average"""
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    """Simple Moving Average"""
    return s.rolling(n, min_periods=n).mean()


def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    """Calculate True Range"""
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    """Average True Range"""
    return true_range(h, l, c).rolling(n, min_periods=n).mean()


def regression_stats(close: pd.Series, window: int = 120, log: bool = True) -> tuple[float, float]:
    """
    Calculate linear regression slope and R² over a window.
    Returns (slope, r_squared)
    """
    y = close.tail(window)
    if len(y.dropna()) < window:
        return (np.nan, np.nan)
    
    if log:
        y = np.log(y)
    
    x = np.arange(len(y))
    r = linregress(x, y)
    return (float(r.slope), float(r.rvalue ** 2))


def returns_over_period(close: pd.Series, days: int = 63) -> float:
    """Calculate percentage returns over a period"""
    if len(close.dropna()) < days + 1:
        return np.nan
    return float((close.iloc[-1] / close.iloc[-days - 1] - 1) * 100.0)


def percent_above(close: pd.Series, ref: pd.Series, window: int = 60) -> float:
    """Percentage of time price was above reference over window"""
    z = close.tail(window)
    r = ref.reindex_like(close).tail(window)
    if len(z.dropna()) < window or len(r.dropna()) < window:
        return np.nan
    return float((z > r).mean() * 100.0)


def max_dd_from_high(close: pd.Series, window: int = 120) -> float:
    """Maximum drawdown from high over window (returns negative %)"""
    y = close.tail(window)
    if len(y.dropna()) < window:
        return np.nan
    rm = y.cummax()
    return float((y / rm - 1).min() * 100.0)


# -----------------------
# Earnings Detection
# -----------------------


def earnings_within_window(ticker: str, days: int = 7) -> bool:
    """
    Check if ticker has earnings within N calendar days from today.
    Returns True if earnings are within the window, False otherwise.
    
    Note: ETFs and funds don't have earnings, so this will return False for them.
    This is expected behavior, not an error.
    """
    import warnings
    
    # Method 1: get_earnings_dates (most reliable for future earnings)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            tk = yf.Ticker(ticker)
            ed = tk.get_earnings_dates(limit=10)  # ✅ Changed from limit=1
            
            if ed is not None and len(ed) > 0:
                # ✅ KEY FIX: Loop through dates to find next FUTURE earnings
                today = dt.date.today()
                
                for idx, row in ed.iterrows():
                    earnings_date = pd.to_datetime(idx).date()
                    days_until = (earnings_date - today).days
                    
                    # Only consider future dates
                    if days_until >= 0:
                        # Found next future earnings - check if within window
                        if days_until <= days:
                            return True
                        else:
                            # Next earnings is beyond our window
                            return False
                
                # All dates were historical
                return False
                
    except Exception:
        pass
    
    # Method 2: calendar (fallback)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            cal = yf.Ticker(ticker).calendar
            
            # Handle dict format (newer yfinance versions)
            if isinstance(cal, dict):
                for key in ['Earnings Date', 'earningsDate', 'Earnings_Date']:
                    if key in cal:
                        earnings_val = cal[key]
                        
                        # Handle list/tuple format
                        if isinstance(earnings_val, (list, tuple)) and len(earnings_val) > 0:
                            nxt = pd.to_datetime(earnings_val[0]).date()
                        else:
                            nxt = pd.to_datetime(earnings_val).date()
                        
                        delta = (nxt - dt.date.today()).days
                        return (delta >= 0) and (delta <= days)
            
            # Handle DataFrame format (older yfinance versions)
            elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                raw = cal.loc["Earnings Date"].values
                if len(raw) > 0:
                    nxt = pd.to_datetime(raw[0]).date()
                    delta = (nxt - dt.date.today()).days
                    return (delta >= 0) and (delta <= days)
                    
    except Exception:
        pass
    
    return False  # Unknown = let it pass (ETFs will reach here)

# -----------------------
# Score Calculation
# -----------------------
def calculate_45_degree_score(
    df: pd.DataFrame,
    spy_close: pd.Series | None = None,
) -> dict:
    """
    Calculate 45° uptrend score for a stock based on various metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV dataframe with columns: High, Low, Close, Volume
    spy_close : pd.Series, optional
        SPY close prices for relative strength calculation
    
    Returns:
    --------
    dict with keys:
        - score: int (0-100)
        - earnings_alert: str ("🔴 EARNINGS SOON" or "")
        - slope120: float
        - r2_120: float
        - slope60: float
        - r2_60: float
        - angle_deg: float
        - ret_63d: float
        - rs_63d_vs_spy: float
        - max_dd_120d: float
        - perc_above_21: float
        - perc_above_50: float
        - dist_to_21ema: float
        - dist_to_50sma: float
        - dist_to_200sma: float
        - buy_hints: list[str]
    """
    if df is None or df.empty or len(df) < 120:
        return _empty_score_dict()
    
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    
    # Calculate moving averages
    ema21 = ema(c, 21)
    sma50 = sma(c, 50)
    sma200 = sma(c, 200)
    
    # Distance to MAs
    dist21 = (c.iloc[-1] / ema21.iloc[-1] - 1) * 100 if pd.notna(ema21.iloc[-1]) else np.nan
    dist50 = (c.iloc[-1] / sma50.iloc[-1] - 1) * 100 if pd.notna(sma50.iloc[-1]) else np.nan
    dist200 = (c.iloc[-1] / sma200.iloc[-1] - 1) * 100 if pd.notna(sma200.iloc[-1]) else np.nan
    
    # Regression stats
    slope120, r2_120 = regression_stats(c, 60, log=True)    # changing 120 and 60 to 60 and 30 for these 2 lines -- 
    slope60, r2_60 = regression_stats(c, 30, log=True)
    
    # Returns
    ret3m = returns_over_period(c, 63)
    
    # Relative strength vs SPY
    if spy_close is not None:
        spy_ret3m = returns_over_period(spy_close, 63)
        rs_3m = ret3m - spy_ret3m if pd.notna(ret3m) and pd.notna(spy_ret3m) else np.nan
    else:
        rs_3m = np.nan
    
    # Percent above metrics
    perc_above_21 = percent_above(c, ema21, 60)
    perc_above_50 = percent_above(c, sma50, 60)
    
    # Drawdown
    mdd120 = max_dd_from_high(c, 120)
    
    # --- 45° TREND SCORING LOGIC ---
    score = 0
    angle_deg = None
    
    # (A) Slope check for 45° angle
    if pd.notna(slope120):
        # slope in log-space => daily log return
        # annualized ≈ 252 * slope
        # 45° means 100% per year => daily ~0.00274
        radians = np.arctan(slope120)
        angle_deg = np.degrees(radians)
        
        # "perfect 45°" => we want slope ~ 0.00274
        # let's say within ±0.00137 is "perfect"
        diff = abs(slope120 - 0.00274)
        if diff < 0.00137:
            score += 30
        elif diff < 0.00274:
            score += 15
    
    # (B) Consistency (R²)
    if pd.notna(r2_120):
        if r2_120 >= 0.85:
            score += 20
        elif r2_120 >= 0.70:
            score += 10
    
    # (C) Relative Strength
    if pd.notna(rs_3m):
        if rs_3m > 10:
            score += 15
        elif rs_3m > 0:
            score += 10
    
    # (D) Momentum: 60-day slope + R²
    if pd.notna(slope60):
        if slope60 > 0:
            score += 5
            if pd.notna(r2_60) and r2_60 > 0.75:
                score += 5
    
    # (E) % above 21/50
    if pd.notna(perc_above_21) and perc_above_21 > 60:
        score += 5
    if pd.notna(perc_above_50) and perc_above_50 > 50:
        score += 5
    
    # (F) Minimal drawdown
    if pd.notna(mdd120) and mdd120 > -10:
        score += 5
    
    # Buy hints
    buy_hints = []
    if pd.notna(dist21) and -1 <= dist21 <= 3:
        buy_hints.append("21-EMA ready")
    if pd.notna(dist50) and -3 <= dist50 <= 1:
        buy_hints.append("50-SMA ready")
    
    return {
        "score": int(score),
        "slope120": float(slope120) if pd.notna(slope120) else np.nan,
        "r2_120": float(r2_120) if pd.notna(r2_120) else np.nan,
        "slope60": float(slope60) if pd.notna(slope60) else np.nan,
        "r2_60": float(r2_60) if pd.notna(r2_60) else np.nan,
        "angle_deg": float(angle_deg) if pd.notna(angle_deg) else np.nan,
        "ret_63d": float(ret3m) if pd.notna(ret3m) else np.nan,
        "rs_63d_vs_spy": float(rs_3m) if pd.notna(rs_3m) else np.nan,
        "max_dd_120d": float(mdd120) if pd.notna(mdd120) else np.nan,
        "perc_above_21": float(perc_above_21) if pd.notna(perc_above_21) else np.nan,
        "perc_above_50": float(perc_above_50) if pd.notna(perc_above_50) else np.nan,
        "dist_to_21ema": float(dist21) if pd.notna(dist21) else np.nan,
        "dist_to_50sma": float(dist50) if pd.notna(dist50) else np.nan,
        "dist_to_200sma": float(dist200) if pd.notna(dist200) else np.nan,
        "buy_hints": buy_hints,
    }


def _empty_score_dict() -> dict:
    """Return empty score dictionary with NaN values"""
    return {
        "score": 0,
        "slope120": np.nan,
        "r2_120": np.nan,
        "slope60": np.nan,
        "r2_60": np.nan,
        "angle_deg": np.nan,
        "ret_63d": np.nan,
        "rs_63d_vs_spy": np.nan,
        "max_dd_120d": np.nan,
        "perc_above_21": np.nan,
        "perc_above_50": np.nan,
        "dist_to_21ema": np.nan,
        "dist_to_50sma": np.nan,
        "dist_to_200sma": np.nan,
        "buy_hints": [],
    }


def get_earnings_alert(ticker: str, days: int = EARNINGS_WINDOW_DAYS) -> str:
    """
    Get earnings alert string for a ticker.
    Returns "🔴 EARNINGS SOON" if earnings within window, "" otherwise.
    """
    has_earnings = earnings_within_window(ticker, days)
    return "🔴 EARNINGS SOON" if has_earnings else ""