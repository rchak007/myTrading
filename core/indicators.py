# core/indicators.py
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    if close is None or len(close) < period + 1:
        return pd.Series(np.nan, index=close.index if close is not None else None)

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_supertrend(
    df: pd.DataFrame,
    atr_period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Adds:
      - Supertrend (line)
      - Supertrend_Signal (BUY/SELL)
    """
    out = df.copy()

    if out.empty or len(out) < atr_period + 2:
        out["Supertrend"] = np.nan
        out["Supertrend_Signal"] = "SELL"
        return out

    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    close = out["Close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    # Wilder-style ATR smoothing like you used (EMA with alpha=1/period)
    atr = tr.ewm(alpha=1.0 / atr_period, adjust=False, min_periods=atr_period).mean()

    src = (high + low) / 2.0
    basic_up = src - multiplier * atr
    basic_dn = src + multiplier * atr

    up = pd.Series(np.nan, index=out.index)
    dn = pd.Series(np.nan, index=out.index)
    up.iloc[0] = basic_up.iloc[0]
    dn.iloc[0] = basic_dn.iloc[0]

    for i in range(1, len(out)):
        up1 = up.iloc[i - 1]
        dn1 = dn.iloc[i - 1]
        up.iloc[i] = max(basic_up.iloc[i], up1) if close.iloc[i - 1] > up1 else basic_up.iloc[i]
        dn.iloc[i] = min(basic_dn.iloc[i], dn1) if close.iloc[i - 1] < dn1 else basic_dn.iloc[i]

    trend = pd.Series(1, index=out.index)
    for i in range(1, len(out)):
        up1 = up.iloc[i - 1]
        dn1 = dn.iloc[i - 1]
        if trend.iloc[i - 1] == -1 and close.iloc[i] > dn1:
            trend.iloc[i] = 1
        elif trend.iloc[i - 1] == 1 and close.iloc[i] < up1:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]

    out["Supertrend"] = np.where(trend == 1, up, dn)
    out["Supertrend_Signal"] = np.where(trend == 1, "BUY", "SELL")
    return out


def compute_adxr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
    length_x: int = 14,
) -> pd.Series:
    """
    Matches the Pine logic you pasted:
      SmoothedTR = prev - prev/len + TR
      DI+, DI-, DX, ADX = SMA(DX,len), ADXR=(ADX + ADX[lenX])/2
    """
    h = high.astype(float)
    l = low.astype(float)
    c = close.astype(float)

    prev_h = h.shift(1)
    prev_l = l.shift(1)
    prev_c = c.shift(1)

    true_range = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)

    up_move = h - prev_h
    down_move = prev_l - l

    dmp = np.where(up_move > down_move, np.maximum(up_move, 0.0), 0.0)
    dmm = np.where(down_move > up_move, np.maximum(down_move, 0.0), 0.0)
    dmp = pd.Series(dmp, index=h.index)
    dmm = pd.Series(dmm, index=h.index)

    sm_tr = pd.Series(np.nan, index=h.index)
    sm_p = pd.Series(np.nan, index=h.index)
    sm_m = pd.Series(np.nan, index=h.index)

    sm_tr_prev = 0.0
    sm_p_prev = 0.0
    sm_m_prev = 0.0

    for i in range(len(h)):
        tr_i = float(true_range.iloc[i]) if pd.notna(true_range.iloc[i]) else 0.0
        p_i = float(dmp.iloc[i]) if pd.notna(dmp.iloc[i]) else 0.0
        m_i = float(dmm.iloc[i]) if pd.notna(dmm.iloc[i]) else 0.0

        sm_tr_i = sm_tr_prev - (sm_tr_prev / length) + tr_i
        sm_p_i = sm_p_prev - (sm_p_prev / length) + p_i
        sm_m_i = sm_m_prev - (sm_m_prev / length) + m_i

        sm_tr.iloc[i] = sm_tr_i
        sm_p.iloc[i] = sm_p_i
        sm_m.iloc[i] = sm_m_i

        sm_tr_prev = sm_tr_i
        sm_p_prev = sm_p_i
        sm_m_prev = sm_m_i

    di_plus = (sm_p / sm_tr) * 100.0
    di_minus = (sm_m / sm_tr) * 100.0

    denom = (di_plus + di_minus).replace(0, np.nan)
    dx = ((di_plus - di_minus).abs() / denom) * 100.0

    adx = dx.rolling(window=length, min_periods=length).mean()
    adxr = (adx + adx.shift(length_x)) / 2.0
    return adxr


def classify_adxr_state(
    adxr: pd.Series,
    low_threshold: float = 20.0,   # logical default (Wilder “trend strength” style)
    eps: float = 1e-6,
) -> pd.Series:
    slope = adxr - adxr.shift(1)
    state = np.where(slope > eps, "RISING", np.where(slope < -eps, "FALLING", "FLAT"))
    low_flat = (adxr < low_threshold) & (np.abs(slope.fillna(0.0)) <= eps)
    state = np.where(low_flat, "LOW_FLAT", state)
    return pd.Series(state, index=adxr.index)


def adxr_signal_from_state(state: str) -> str:
    if state == "RISING":
        return "TREND_OK"
    if state == "LOW_FLAT":
        return "STANDDOWN"
    return "WEAK"


def compute_most_rsi(close: pd.Series) -> pd.DataFrame:
    """
    Adds MOST_RSI_MA (yellow) and MOST_RSI_Line (brown) + MOST_RSI_Signal
    This is the exact logic you were matching (the working one).
    """
    close = close.astype(float)

    rsi_len = 14
    chg = close.diff()
    up_c = chg.clip(lower=0.0)
    down_c = (-chg).clip(lower=0.0)

    up_rma = up_c.ewm(alpha=1.0 / rsi_len, adjust=False, min_periods=rsi_len).mean()
    down_rma = down_c.ewm(alpha=1.0 / rsi_len, adjust=False, min_periods=rsi_len).mean()

    rs = up_rma / down_rma
    rsi_tv = np.where(
        down_rma == 0,
        100.0,
        np.where(up_rma == 0, 0.0, 100.0 - (100.0 / (1.0 + rs))),
    )
    rsi_series = pd.Series(rsi_tv, index=close.index)

    # CMO-like adaptiveness (your working version)
    cmo_period = 9
    delta = (rsi_series - rsi_series.shift(1)).fillna(0.0)
    vud1 = delta.clip(lower=0.0)
    vdd1 = (-delta).clip(lower=0.0)

    vUD = vud1.rolling(window=cmo_period, min_periods=1).sum()
    vDD = vdd1.rolling(window=cmo_period, min_periods=1).sum()

    denom = (vUD + vDD).replace(0, np.nan)
    vCMO = ((vUD - vDD) / denom).fillna(0.0).abs()

    ma_length = 5
    valpha = 2.0 / (ma_length + 1.0)

    exMov = pd.Series(np.nan, index=close.index)
    var_prev = 0.0
    for i in range(len(close)):
        src_i = rsi_series.iloc[i]
        src_i = 0.0 if pd.isna(src_i) else float(src_i)
        a_i = float(valpha * vCMO.iloc[i])
        var_i = (a_i * src_i) + ((1.0 - a_i) * var_prev)
        exMov.iloc[i] = var_i
        var_prev = var_i

    percent = 9.0
    fark = exMov * percent * 0.01
    longStop = exMov - fark
    shortStop = exMov + fark

    longStop_adj = pd.Series(np.nan, index=close.index)
    shortStop_adj = pd.Series(np.nan, index=close.index)

    for i in range(len(close)):
        if i == 0:
            longStop_adj.iloc[i] = longStop.iloc[i]
            shortStop_adj.iloc[i] = shortStop.iloc[i]
            continue
        ls_prev = longStop_adj.iloc[i - 1]
        ss_prev = shortStop_adj.iloc[i - 1]
        longStop_adj.iloc[i] = max(longStop.iloc[i], ls_prev) if exMov.iloc[i] > ls_prev else longStop.iloc[i]
        shortStop_adj.iloc[i] = min(shortStop.iloc[i], ss_prev) if exMov.iloc[i] < ss_prev else shortStop.iloc[i]

    dir_series = pd.Series(1, index=close.index, dtype=int)
    for i in range(1, len(close)):
        prev_dir = dir_series.iloc[i - 1]
        ls_prev = longStop_adj.iloc[i - 1]
        ss_prev = shortStop_adj.iloc[i - 1]
        cur_ex = exMov.iloc[i]
        if prev_dir == -1 and cur_ex > ss_prev:
            dir_series.iloc[i] = 1
        elif prev_dir == 1 and cur_ex < ls_prev:
            dir_series.iloc[i] = -1
        else:
            dir_series.iloc[i] = prev_dir

    most_line = np.where(dir_series.values == 1, longStop_adj.values, shortStop_adj.values)
    most_signal = np.where(exMov > most_line, "BUY", "SELL")

    return pd.DataFrame(
        {
            "MOST_MA": exMov,
            "MOST_Line": most_line,
            "MOST_Signal": most_signal,
        },
        index=close.index,
    )


def apply_indicators(
    df: pd.DataFrame,
    atr_period: int = 10,
    atr_multiplier: float = 3.0,
    rsi_period: int = 14,
    vol_lookback: int = 20,
    adxr_len: int = 14,
    adxr_lenx: int = 14,
    adxr_low_threshold: float = 20.0,
    adxr_flat_eps: float = 1e-6,
) -> pd.DataFrame:
    """
    Expected df columns: High, Low, Close, Volume
    Adds:
      Supertrend, Supertrend_Signal
      RSI
      Avg_Volume
      MOST MA/Line/Signal
      ADXR + State + Signal
    """
    out = df.copy()

    out = compute_supertrend(out, atr_period=atr_period, multiplier=atr_multiplier)

    out["RSI"] = compute_rsi(out["Close"], period=rsi_period)
    out["Avg_Volume"] = out["Volume"].rolling(window=vol_lookback, min_periods=vol_lookback).mean()

    most = compute_most_rsi(out["Close"])
    out["MOST_MA"] = most["MOST_MA"]
    out["MOST_Line"] = most["MOST_Line"]
    out["MOST_Signal"] = most["MOST_Signal"]

    out["ADXR"] = compute_adxr(out["High"], out["Low"], out["Close"], length=adxr_len, length_x=adxr_lenx)
    out["ADXR_State"] = classify_adxr_state(out["ADXR"], low_threshold=adxr_low_threshold, eps=adxr_flat_eps)
    out["ADXR_Signal"] = out["ADXR_State"].map(adxr_signal_from_state)

    return out

