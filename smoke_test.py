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
                     "read_intents", "build_dashboard", "build_cash_rows"],
    "order_engine": ["main", "preflight", "check_guards", "daily_close", "submit",
                     "preview", "resolve_limit", "build_order_json",
                     "stamp_row_ids", "account_state", "triggered"],
    "order_intent": ["normalize", "fingerprint", "describe", "idempotency_key"],
    "schwab_quotes": ["fetch_quotes", "extract_price", "price_map", "bid_ask",
                      "marketable_limit"],
    "remote_ops": ["run_verb", "run_pyverb", "safe_arg", "write_back"],
}

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

print(f"\n{'all present' if not bad else str(bad) + ' problem(s)'}")
raise SystemExit(1 if bad else 0)
