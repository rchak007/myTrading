# core/macro.py
"""
Macro regime computation for stock trading.

Computes BULL / NEUTRAL / BEAR from VIX + SPY vs 200MA + market breadth,
then combines with per-stock SIGNAL-Super-MOST-ADXR to produce a
Regime_Action string.

Pure functions — no side effects, no Streamlit, no yfinance calls.
Importable from app.py, jobStocksSignals.py, or anywhere.

Usage:
    from core.macro import compute_macro_regime, compute_regime_action
"""

from __future__ import annotations

import math
from core.config import MACRO_THRESHOLDS


# ═══════════════════════════════════════════════════════════════════
# Macro Regime Computation
# ═══════════════════════════════════════════════════════════════════

def compute_macro_regime(
    vix: float,
    spy_above_200ma: bool,
    breadth_pct: float,
    override: str = "AUTO",
) -> dict:
    """
    Compute the stock market macro regime from three filters.

    Parameters
    ----------
    vix : float
        Current VIX level (e.g. 29.81). NaN → treated as unknown/cautious.
    spy_above_200ma : bool
        True if SPY close > SPY 200-day MA.
    breadth_pct : float
        % of S&P 500 stocks above their 200MA (0-100). NaN → cautious.
    override : str
        "AUTO" (compute from data), or "BULL"/"NEUTRAL"/"BEAR" to force.

    Returns
    -------
    dict with keys:
        regime      : str   — "BULL", "NEUTRAL", or "BEAR"
        risk_pct    : float — suggested risk % per trade (2.0, 1.0, 0.5, or 0.0)
        reason      : str   — human-readable explanation
        overridden  : bool  — True if override was applied
        vix         : float — echo back for logging
        spy_above   : bool
        breadth_pct : float
    """
    result = {
        "vix": vix,
        "spy_above": spy_above_200ma,
        "breadth_pct": breadth_pct,
        "overridden": False,
    }

    # ── Override path ──
    if override and override.upper() in ("BULL", "NEUTRAL", "BEAR"):
        result["regime"] = override.upper()
        result["overridden"] = True
        result["reason"] = f"Manual override → {override.upper()}"
        result["risk_pct"] = {"BULL": 2.0, "NEUTRAL": 1.0, "BEAR": 0.0}[result["regime"]]
        return result

    # ── Computed path ──
    vix_low    = MACRO_THRESHOLDS["vix_low"]
    vix_high   = MACRO_THRESHOLDS["vix_high"]
    bh         = MACRO_THRESHOLDS["breadth_healthy"]
    bw         = MACRO_THRESHOLDS["breadth_weak"]

    vix_safe   = (not math.isnan(vix)) and vix < vix_low
    vix_danger = math.isnan(vix) or vix >= vix_high
    vix_mid    = (not math.isnan(vix)) and vix_low <= vix < vix_high

    breadth_ok   = (not math.isnan(breadth_pct)) and breadth_pct > bh
    breadth_weak = (not math.isnan(breadth_pct)) and bw <= breadth_pct <= bh
    breadth_poor = math.isnan(breadth_pct) or breadth_pct < bw

    # ── BEAR: any hard stop triggers ──
    #   VIX > 30  OR  SPY < 200MA  →  sit out
    if vix_danger or not spy_above_200ma:
        reasons = []
        if vix_danger:
            reasons.append(f"VIX={vix:.1f} ≥ {vix_high}")
        if not spy_above_200ma:
            reasons.append("SPY < 200MA")
        if breadth_poor:
            reasons.append(f"Breadth={breadth_pct:.0f}% < {bw}%")
        result["regime"] = "BEAR"
        result["risk_pct"] = 0.0
        result["reason"] = "BEAR — " + " + ".join(reasons) + " → Sit out"
        return result

    # ── BULL: all three green ──
    #   VIX < 20  AND  SPY > 200MA  AND  breadth > 60%
    if vix_safe and spy_above_200ma and breadth_ok:
        result["regime"] = "BULL"
        result["risk_pct"] = 2.0
        result["reason"] = (
            f"BULL — VIX={vix:.1f} < {vix_low}, SPY > 200MA, "
            f"Breadth={breadth_pct:.0f}% > {bh}% → Risk 2%/trade"
        )
        return result

    # ── NEUTRAL: everything else ──
    #   VIX < 20 + breadth weakening:  risk 1%
    #   VIX 20-30 + either warning:   risk 0.5%
    reasons = []
    if vix_mid:
        reasons.append(f"VIX={vix:.1f} ({vix_low}–{vix_high})")
    if breadth_weak:
        reasons.append(f"Breadth={breadth_pct:.0f}% ({bw}–{bh}%)")
    if breadth_poor:
        reasons.append(f"Breadth={breadth_pct:.0f}% < {bw}%")
    if not reasons:
        reasons.append("Mixed signals")

    # Sub-tier: VIX safe + only breadth weak → 1%; otherwise → 0.5%
    if vix_safe and breadth_weak:
        risk = 1.0
    else:
        risk = 0.5

    result["regime"] = "NEUTRAL"
    result["risk_pct"] = risk
    result["reason"] = "NEUTRAL — " + " + ".join(reasons) + f" → Risk {risk}%/trade"
    return result


# ═══════════════════════════════════════════════════════════════════
# Regime-Aware Action (combines per-stock signal + macro regime)
# ═══════════════════════════════════════════════════════════════════

def compute_regime_action(
    signal: str,
    macro_regime: str,
    score_weighted: float = 0.0,
) -> str:
    """
    Combine a stock's SIGNAL-Super-MOST-ADXR with the macro regime
    to produce an actionable recommendation.

    Parameters
    ----------
    signal : str
        One of "BUY", "HOLD", "EXIT", "STANDDOWN"
    macro_regime : str
        "BULL", "NEUTRAL", or "BEAR"
    score_weighted : float
        45° trend score (0-100) — used for BULL regime upgrades

    Returns
    -------
    str — action label, e.g. "BUY (full size)", "HOLD → consider", "SIT OUT"
    """
    sig = signal.upper().strip()
    regime = macro_regime.upper().strip()

    # ── BEAR regime: everything is cautious ──
    if regime == "BEAR":
        if sig == "BUY":
            return "⚠️ BUY (sit out — bear)"
        if sig == "HOLD":
            return "⚠️ HOLD (reduce / exit)"
        if sig == "EXIT":
            return "🔴 EXIT"
        if sig == "STANDDOWN":
            return "🔴 STANDDOWN"
        return "🔴 SIT OUT"

    # ── NEUTRAL regime: standard conservative ──
    if regime == "NEUTRAL":
        if sig == "BUY":
            return "🟡 BUY (half size)"
        if sig == "HOLD":
            return "🟡 HOLD"
        if sig == "EXIT":
            return "🔴 EXIT"
        if sig == "STANDDOWN":
            return "🔴 STANDDOWN"
        return "🟡 WAIT"

    # ── BULL regime: more aggressive ──
    if regime == "BULL":
        if sig == "BUY":
            return "🟢 BUY (full size)"
        if sig == "HOLD":
            # In bull, HOLD with high score → actionable
            if score_weighted >= 50:
                return "🟢 HOLD → consider (score≥50)"
            return "🟡 HOLD"
        if sig == "EXIT":
            return "🔴 EXIT"
        if sig == "STANDDOWN":
            return "🟡 STANDDOWN (watch)"
        return "🟡 WAIT"

    # Fallback
    return sig  