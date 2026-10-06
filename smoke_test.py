"""Import-surface smoke test. Catches a function deleted by a bad edit."""
import importlib, sys
sys.path.insert(0, ".")

EXPECTED = {
    "cash_reserve": ["build_reserves_table", "overcommit_warnings", "fetch_fills",
                     "apply_fills", "fence", "seed", "topup", "withdraw", "close",
                     "available_to_buy", "position_value", "reconcile",
                     "write_reserve_outputs", "fold_balances", "read_config"],
    "orders_sheet": ["write_orders_sheet", "build_positions_table", "coverage_for",
                     "fetch_positions_detailed", "load_reserves", "load_fenced",
                     "parse_option",
                     "read_intents", "build_dashboard", "build_cash_rows"],
    "order_engine": ["main", "preflight", "check_guards", "daily_close", "submit",
                     "preview", "resolve_limit", "build_order_json",
                     "stamp_row_ids", "account_state", "triggered"],
    "order_intent": ["normalize", "fingerprint", "describe", "idempotency_key"],
    "schwab_quotes": ["fetch_quotes", "extract_price", "price_map", "bid_ask",
                      "marketable_limit"],
    "remote_ops": ["run_verb", "run_pyverb", "safe_arg", "write_back"],
    "token_watch": ["send", "load_mail_env", "read_from_sheet", "compose",
                    "mail_env_path", "recently_sent", "mark_sent",
                    "poll_is_stale", "expected_last_poll"],
    "reminders": ["read_rows", "write_rows", "due", "compose", "notes_images",
                  "use_channel", "CHANNELS", "weekday_of", "WEEKDAY_NAME"],
    "market_calendar": ["is_open", "is_trading_day", "holidays", "early_closes",
                        "close_time", "easter", "describe"],
    "chitra": ["load", "load_orders", "load_reserves", "cash_position", "meta", "write_tab", "dashboard_row",
               "read_conditions", "evaluate", "coverage", "build_positions",
               "build_orders", "build_condition_status",
               "ACCT_LABEL", "POS_COLS", "ORD_COLS", "CON_COLS", "MET"],
    "core.recommend": ["recommend", "recommend_row", "attach", "tick_round",
                       "Rec", "earnings_soon", "TIER1", "TIER2", "TIER3",
                       "ALL_LEVELS", "FIELDS"],
}

# Signatures another module actually calls through. A name that still exists
# with the keyword removed passes the hasattr check above and fails at send
# time — which for a reminder means silence, the one failure nobody notices.
SIGNATURES = {("token_watch", "send"): ["subject", "body", "images"],
              ("orders_sheet", "build_dashboard"): ["signals_df", "fenced",
                                                    "chitra_rows", "options"],
              ("orders_sheet", "coverage_for"): ["held_qty", "options"],
              ("core.recommend", "recommend"): ["price", "atr", "fenced"],
              ("orders_sheet", "_classify"): ["side", "px", "price",
                                              "direction"]}

bad = 0
for mod, names in EXPECTED.items():
    try:
        m = importlib.import_module(mod)
    except Exception as e:
        print(f"  {mod:<16} IMPORT FAILED: {type(e).__name__}: {e}")
        bad += 1
        continue
    missing = [n for n in names if not hasattr(m, n)]
    print(f"  {mod:<16} {'OK' if not missing else 'MISSING: ' + ', '.join(missing)}")
    bad += len(missing)

import inspect
for (mod, fn), params in SIGNATURES.items():
    try:
        sig = inspect.signature(getattr(importlib.import_module(mod), fn))
    except Exception as e:
        print(f"  {mod}.{fn:<10} SIGNATURE UNREADABLE: {e}")
        bad += 1
        continue
    gone = [p for p in params if p not in sig.parameters]
    print(f"  {mod + '.' + fn:<16} {'OK' if not gone else 'LOST PARAM: ' + ', '.join(gone)}")
    bad += len(gone)

print(f"\n{'all present' if not bad else str(bad) + ' problem(s)'}")
raise SystemExit(1 if bad else 0)
