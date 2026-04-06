# ═══════════════════════════════════════════════════════════════════
# INTEGRATION GUIDE — Macro Regime + Centralized Config
# ═══════════════════════════════════════════════════════════════════
#
# 3 new files created:
#   core/config.py  → single source of truth for indicator params + macro override
#   core/macro.py   → compute_macro_regime() + compute_regime_action()
#   data/stocks.py  → add_macro_regime_columns() added (existing funcs unchanged)
#
# Below are the EXACT changes needed in app.py and jobStocksSignals.py.
#
# ═══════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────
# CHANGE 1: app.py — Replace hardcoded params with core.config import
# ───────────────────────────────────────────────────────────────────
#
# REMOVE these lines (app.py lines 76-86):
#
#   ATR_PERIOD = 10
#   ATR_MULTIPLIER = 3.0
#   RSI_PERIOD = 14
#   VOL_LOOKBACK = 20
#   VOL_MULTIPLIER = 1.2
#   RSI_BUY_THRESHOLD = 50.0
#   ADXR_LEN = 14
#   ADXR_LENX = 14
#   ADXR_LOW_THRESHOLD = 20.0
#   ADXR_FLAT_EPS = 1e-6
#
# REPLACE with:
#
#   from core.config import INDICATOR_PARAMS, MACRO_REGIME_OVERRIDE
#   from core.macro import compute_macro_regime
#   from data.stocks import add_macro_regime_columns
#
#   # Unpack for backward compatibility with existing code
#   ATR_PERIOD        = INDICATOR_PARAMS["atr_period"]
#   ATR_MULTIPLIER    = INDICATOR_PARAMS["atr_multiplier"]
#   RSI_PERIOD        = INDICATOR_PARAMS["rsi_period"]
#   VOL_LOOKBACK      = INDICATOR_PARAMS["vol_lookback"]
#   VOL_MULTIPLIER    = INDICATOR_PARAMS["vol_multiplier"]
#   RSI_BUY_THRESHOLD = INDICATOR_PARAMS["rsi_buy_threshold"]
#   ADXR_LEN          = INDICATOR_PARAMS["adxr_len"]
#   ADXR_LENX         = INDICATOR_PARAMS["adxr_lenx"]
#   ADXR_LOW_THRESHOLD = INDICATOR_PARAMS["adxr_low_threshold"]
#   ADXR_FLAT_EPS     = INDICATOR_PARAMS["adxr_flat_eps"]


# ───────────────────────────────────────────────────────────────────
# CHANGE 2: app.py — Add macro regime override to sidebar
# ───────────────────────────────────────────────────────────────────
#
# In the sidebar section (after the Schwab token panel), ADD:
#
#   with st.sidebar:
#       st.subheader("🎛️ Macro Regime Override")
#       regime_options = ["AUTO", "BULL", "NEUTRAL", "BEAR"]
#       # Default from .env or "AUTO"
#       default_idx = regime_options.index(MACRO_REGIME_OVERRIDE) \
#           if MACRO_REGIME_OVERRIDE in regime_options else 0
#       macro_override = st.selectbox(
#           "Override macro regime:",
#           regime_options,
#           index=default_idx,
#           help="AUTO = computed from VIX+SPY+breadth. "
#                "Set to BULL/NEUTRAL/BEAR to force a regime for testing."
#       )
#       if macro_override != "AUTO":
#           st.warning(f"⚠️ Macro regime OVERRIDDEN → {macro_override}")


# ───────────────────────────────────────────────────────────────────
# CHANGE 3: app.py — Compute macro regime in the Stock Market Context
# ───────────────────────────────────────────────────────────────────
#
# After the existing breadth computation (where you have vix, spy_close,
# spy_ma200, spy_status, and breadth already computed), ADD:
#
#       # ── Compute macro regime ──
#       macro_info = compute_macro_regime(
#           vix=vix,
#           spy_above_200ma=(spy_status == "ABOVE"),
#           breadth_pct=breadth["pct"],
#           override=macro_override,   # from sidebar selectbox
#       )
#       macro_regime = macro_info["regime"]
#
#       # Display regime
#       regime_colors = {"BULL": "🟢", "NEUTRAL": "🟡", "BEAR": "🔴"}
#       st.markdown(
#           f"**Macro Regime: {regime_colors.get(macro_regime, '')} {macro_regime}** "
#           f"(Risk {macro_info['risk_pct']}%/trade)"
#       )
#       st.caption(macro_info["reason"])
#       if macro_info["overridden"]:
#           st.warning("Regime is manually overridden — not computed from market data")


# ───────────────────────────────────────────────────────────────────
# CHANGE 4: app.py — Add regime columns to stock DataFrame
# ───────────────────────────────────────────────────────────────────
#
# After building df_stocks (either from CSV or fresh compute), ADD
# before the st.dataframe() call:
#
#       # Add macro regime + action columns
#       df_stocks = add_macro_regime_columns(df_stocks, macro_regime)
#


# ───────────────────────────────────────────────────────────────────
# CHANGE 5: jobStocksSignals.py — use core.config + compute regime
# ───────────────────────────────────────────────────────────────────
#
# In jobStocksSignals.py, replace any hardcoded params with:
#
#   from core.config import INDICATOR_PARAMS, MACRO_REGIME_OVERRIDE
#   from core.macro import compute_macro_regime
#   from data.breadth import fetch_vix_value, fetch_spy_vs_200ma, breadth_proxy_from_spy
#   from data.stocks import build_stocks_signals_table, add_macro_regime_columns
#
#   # Build signals table using centralized params
#   df = build_stocks_signals_table(
#       tickers,
#       **INDICATOR_PARAMS,           # ← unpack all params from config
#   )
#
#   # Compute macro regime (uses .env override if set, else computes)
#   vix = fetch_vix_value()
#   spy_close, spy_ma200, spy_status = fetch_spy_vs_200ma()
#   breadth = breadth_proxy_from_spy(spy_close, spy_ma200)
#
#   macro_info = compute_macro_regime(
#       vix=vix,
#       spy_above_200ma=(spy_status == "ABOVE"),
#       breadth_pct=breadth["pct"],
#       override=MACRO_REGIME_OVERRIDE,   # from .env or "AUTO"
#   )
#
#   # Add regime columns
#   df = add_macro_regime_columns(df, macro_info["regime"])
#
#   # Save CSV (now includes Macro_Regime + Regime_Action columns)
#   df.to_csv(output_path, index=False)


# ───────────────────────────────────────────────────────────────────
# CHANGE 6: bot.py — OPTIONAL (no changes needed now)
# ───────────────────────────────────────────────────────────────────
#
# bot.py already reads from .env via os.getenv() with the SAME default
# values as core/config.py uses. So bot.py will continue to work as-is.
#
# If you WANT bot.py to also use core.config, replace its parameter
# block with:
#
#   from core.config import INDICATOR_PARAMS
#   ATR_PERIOD   = INDICATOR_PARAMS["atr_period"]
#   ATR_MULT     = INDICATOR_PARAMS["atr_multiplier"]
#   ...etc
#
# But this is optional — the .env approach already works.


# ───────────────────────────────────────────────────────────────────
# .env additions (optional — all have sane defaults)
# ───────────────────────────────────────────────────────────────────
#
# # Macro regime override (AUTO = compute from VIX+SPY+breadth)
# MACRO_REGIME_OVERRIDE=AUTO
#
# # Macro thresholds (defaults shown — only override if needed)
# MACRO_VIX_LOW=20
# MACRO_VIX_HIGH=30
# MACRO_BREADTH_HEALTHY=60
# MACRO_BREADTH_WEAK=50
#
# # Indicator params (these are already the defaults in core/config.py)
# ATR_PERIOD=10
# ATR_MULT=3.0
# RSI_PERIOD=14
# VOL_LOOKBACK=20
# VOL_MULTIPLIER=1.2
# RSI_BUY_THRESHOLD=50.0
# ADXR_LEN=14
# ADXR_LENX=14
# ADXR_LOW=20.0
# ADXR_EPS=1e-6