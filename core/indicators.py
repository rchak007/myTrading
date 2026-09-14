# core/indicators.py
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder's RSI -- what TradingView's ta.rsi(), Yahoo, StockCharts, TA-Lib and
    everything else mean by "RSI(14)".

    This previously averaged gains and losses with .rolling().mean(), i.e. a
    SIMPLE moving average. That is a different indicator (Cutler's RSI), and it
    reads far too low after a drop: a plain mean gives the newest bar full
    weight and then drops it entirely `period` bars later, while Wilder's RMA
    weights it 1/period and carries the rest forward. Measured 11-17 points of
    divergence on a CRDO-shaped decline -- CRDO on 2026-09-14 read 20.9 here
    against ~30.8 on both TradingView and Yahoo daily.

    compute_most_rsi() below always used the correct RMA, so the two RSIs in
    this one file disagreed with each other.
    """
    if close is None or len(close) < period + 1:
        return pd.Series(np.nan, index=close.index if close is not None else None)

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # avg_loss == 0 -> rs is inf -> 100, which is correct. But a dead-flat
    # window gives 0/0 -> NaN; report that as neutral rather than missing.
    return rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)


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


# ═══════════════════════════════════════════════════════════════════
# Mean Reversion Channel (fareid's MRI Variant)
# ═══════════════════════════════════════════════════════════════════
# Port of: https://www.tradingview.com/script/... by ©fareidzulkifli
# License: MPL 2.0 — credit retained.
#
# The indicator builds a channel around a SuperSmoother filter (Ehlers)
# applied to HLC3, with bands set at ±π * mult * smoothed-True-Range.
# Five levels: R2 / R1 / Mean / S1 / S2.
#
# Defaults match the Pine source exactly:
#   length=200, innermult=1.0, outermult=2.415, source=hlc3
#   inner band offset = π * 1.0    ≈ 3.14159
#   outer band offset = π * 2.415  ≈ 7.58605
# ═══════════════════════════════════════════════════════════════════

_MRC_PI = np.pi


def _supersmoother(src: pd.Series, length: int) -> pd.Series:
    """
    Ehlers' 2-pole SuperSmoother filter.

    Recursive formula:
        a1 = exp(-sqrt(2) * π / length)
        b1 = 2 * a1 * cos(sqrt(2) * π / length)
        c3 = -a1**2
        c2 = b1
        c1 = 1 - c2 - c3
        ss[i] = c1 * src[i] + c2 * ss[i-1] + c3 * ss[i-2]

    For the first two bars Pine seeds with the source value (nz fallback),
    so we mirror that behaviour: ss[0] = src[0], ss[1] = src[1].
    """
    s = src.astype(float).values
    n = len(s)
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return pd.Series(out, index=src.index)

    a1 = np.exp(-np.sqrt(2.0) * _MRC_PI / length)
    b1 = 2.0 * a1 * np.cos(np.sqrt(2.0) * _MRC_PI / length)
    c3 = -(a1 ** 2)
    c2 = b1
    c1 = 1.0 - c2 - c3

    # Seed: Pine's nz(_src[1]) / nz(_src[2]) effectively makes the first two
    # outputs equal to source when no prior output exists.
    out[0] = s[0] if not np.isnan(s[0]) else 0.0
    if n > 1:
        out[1] = s[1] if not np.isnan(s[1]) else out[0]

    for i in range(2, n):
        src_i = s[i] if not np.isnan(s[i]) else 0.0
        out[i] = c1 * src_i + c2 * out[i - 1] + c3 * out[i - 2]

    return pd.Series(out, index=src.index)


def compute_mrc_bands(
    df: pd.DataFrame,
    length: int = 200,
    inner_mult: float = 1.0,
    outer_mult: float = 2.415,
) -> pd.DataFrame:
    """
    Mean Reversion Channel — 5 levels (R2, R1, Mean, S1, S2).

    Inputs:
        df: DataFrame with columns High, Low, Close
    Returns DataFrame with columns:
        MRC_Mean, MRC_R1, MRC_S1, MRC_R2, MRC_S2

    The mean is SuperSmoother(HLC3, length).
    The "range" component is SuperSmoother(TrueRange, length).
    Inner bands = mean ± range * π * inner_mult
    Outer bands = mean ± range * π * outer_mult
    """
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    n = len(df)
    if n < 3:
        nan = pd.Series(np.nan, index=df.index)
        return pd.DataFrame({
            "MRC_Mean": nan, "MRC_R1": nan, "MRC_S1": nan,
            "MRC_R2": nan, "MRC_S2": nan,
        })

    hlc3 = (high + low + close) / 3.0

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    # First TR has no prev_close — Pine treats it as high-low; pandas concat
    # already yields high-low for that row since the shifted columns are NaN
    # and .max ignores NaN. Belt-and-braces:
    tr.iloc[0] = float(high.iloc[0] - low.iloc[0])

    mean_line = _supersmoother(hlc3, length)
    range_line = _supersmoother(tr, length)

    inner_offset = range_line * (_MRC_PI * inner_mult)
    outer_offset = range_line * (_MRC_PI * outer_mult)

    r1 = mean_line + inner_offset
    s1 = mean_line - inner_offset
    r2 = mean_line + outer_offset
    s2 = mean_line - outer_offset

    # Until SuperSmoother has had `length` bars to stabilise, the values are
    # not meaningful — mask the warmup region to NaN so downstream code
    # doesn't act on garbage early bars.
    if n > length:
        warmup_mask = np.arange(n) < length
        for s in (mean_line, r1, s1, r2, s2):
            s.iloc[warmup_mask] = np.nan

    return pd.DataFrame({
        "MRC_Mean": mean_line,
        "MRC_R1": r1,
        "MRC_S1": s1,
        "MRC_R2": r2,
        "MRC_S2": s2,
    }, index=df.index)


def classify_mrc_zone(
    close: pd.Series,
    mrc: pd.DataFrame,
) -> pd.Series:
    """
    Map close + MRC bands → zone label (close-based, screener-friendly).

    Zone definitions (mirrors Pine `condition`, but uses Close instead of
    High/Low so intraday wicks that didn't hold are filtered out):

        Strong_OB    : Close >= R2
        OB           : R1 <= Close < R2
        Above_Mean   : Mean < Close < R1
        Near_Mean    : Close ≈ Mean (within 1 inner band-half — i.e. very tight)
        Below_Mean   : S1 < Close < Mean
        OS           : S2 < Close <= S1
        Strong_OS    : Close <= S2
        N/A          : any band is NaN (warmup or insufficient data)
    """
    out = pd.Series("N/A", index=close.index, dtype=object)

    c = close.astype(float)
    mean = mrc["MRC_Mean"]
    r1 = mrc["MRC_R1"]
    s1 = mrc["MRC_S1"]
    r2 = mrc["MRC_R2"]
    s2 = mrc["MRC_S2"]

    valid = mean.notna() & r1.notna() & s1.notna() & r2.notna() & s2.notna()

    # Half the inner-band width = "near mean" tolerance
    near_tol = (r1 - mean) * 0.5

    out = out.where(~valid, "Above_Mean")  # default for valid rows; refined below

    out[valid & (c >= r2)] = "Strong_OB"
    out[valid & (c < r2) & (c >= r1)] = "OB"
    out[valid & (c < r1) & (c > mean + near_tol)] = "Above_Mean"
    out[valid & (c >= mean - near_tol) & (c <= mean + near_tol)] = "Near_Mean"
    out[valid & (c < mean - near_tol) & (c > s1)] = "Below_Mean"
    out[valid & (c <= s1) & (c > s2)] = "OS"
    out[valid & (c <= s2)] = "Strong_OS"

    return out


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
    mrc_length: int = 200,
    mrc_inner_mult: float = 1.0,
    mrc_outer_mult: float = 2.415,
) -> pd.DataFrame:
    """
    Expected df columns: High, Low, Close, Volume
    Adds:
      Supertrend, Supertrend_Signal
      RSI
      Avg_Volume
      MOST MA/Line/Signal
      ADXR + State + Signal
      MRC_Mean, MRC_R1, MRC_S1, MRC_R2, MRC_S2 (Mean Reversion Channel)
      MRC_Dist_Pct  : (Close - Mean) / Mean * 100
      MRC_Zone      : Strong_OB / OB / Above_Mean / Near_Mean / Below_Mean / OS / Strong_OS
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

    # Mean Reversion Channel
    mrc = compute_mrc_bands(
        out,
        length=mrc_length,
        inner_mult=mrc_inner_mult,
        outer_mult=mrc_outer_mult,
    )
    out["MRC_Mean"] = mrc["MRC_Mean"]
    out["MRC_R1"]   = mrc["MRC_R1"]
    out["MRC_S1"]   = mrc["MRC_S1"]
    out["MRC_R2"]   = mrc["MRC_R2"]
    out["MRC_S2"]   = mrc["MRC_S2"]

    # Derived: % distance from mean (positive = above, negative = below)
    out["MRC_Dist_Pct"] = ((out["Close"] - out["MRC_Mean"]) / out["MRC_Mean"]) * 100.0

    # Derived: zone label (close-based)
    out["MRC_Zone"] = classify_mrc_zone(out["Close"], mrc)

    return out