import os
import logging

from dotenv import load_dotenv
import schwabdev
from pathlib import Path
from data.schwab.token_sync import sync_db_to_local, sync_local_to_db


APP_DIR = Path(__file__).resolve().parents[1]  # points to repo root
TOKEN_PATH = APP_DIR / "tokens.json"
USER_ID = "main"

def get_env(name: str, fallback: str | None = None) -> str:
    """Try lowercase first (Schwabdev docs), then uppercase fallback."""
    val = os.getenv(name)
    if not val and fallback:
        val = os.getenv(fallback)
    return val or ""


def main():
    load_dotenv()  # loads .env from current folder

    # Read keys (supports both lower + UPPER from earlier)
    app_key = get_env("app_key", "SCHWAB_APP_KEY")
    app_secret = get_env("app_secret", "SCHWAB_APP_SECRET")
    callback_url = get_env("callback_url", "SCHWAB_CALLBACK_URL")

    if not app_key or not app_secret or not callback_url:
        raise RuntimeError(
            "Missing app_key / app_secret / callback_url in .env "
            "(use app_key, app_secret, callback_url or SCHWAB_APP_KEY, SCHWAB_APP_SECRET, SCHWAB_CALLBACK_URL)."
        )

    logging.basicConfig(level=logging.INFO)

    # This is exactly how the official example creates the client
    # docs/examples/api_demo.py on the repo :contentReference[oaicite:1]{index=1}
    sync_db_to_local(USER_ID, TOKEN_PATH)
    client = schwabdev.Client(app_key, app_secret, callback_url)
    sync_local_to_db(USER_ID, TOKEN_PATH)

    print("\n✅ Connected to Schwabdev client, fetching linked accounts…")

    # 1) Get linked accounts + hashes
    linked_accounts = client.account_linked().json()
    print(f"\nFound {len(linked_accounts)} linked accounts:")
    for acct in linked_accounts:
        masked = acct.get("accountNumberMasked")
        acct_type = acct.get("accountType")
        hash_val = acct.get("hashValue")
        print(f"  - {masked} | type={acct_type} | hash={hash_val}")

    # 2) Get details (balances) for all accounts
    print("\nFetching account details (with balances)…")
    details = client.account_details_all().json()

    rows = []
    total_equity = 0.0

    for acct in details:
        # Schwab's Accounts & Trading API typically nests this under 'securitiesAccount' :contentReference[oaicite:2]{index=2}
        sa = acct.get("securitiesAccount", {})
        account_number = sa.get("accountNumber")

        balances = sa.get("currentBalances", {}) or {}
        # prefer 'equity', fall back to 'liquidationValue' if needed
        equity = balances.get("equity", balances.get("liquidationValue", 0.0))
        cash = (
            balances.get("cashBalance")
            or balances.get("cashAvailableForTrading")
            or 0.0
        )

        try:
            eq_float = float(equity or 0.0)
        except (TypeError, ValueError):
            eq_float = 0.0

        total_equity += eq_float

        rows.append(
            {
                "accountNumber": account_number,
                "equity": eq_float,
                "cash": float(cash or 0.0),
            }
        )

    print("\n=== Account Equity Snapshot ===")
    for row in rows:
        print(
            f"Account {row['accountNumber']}: "
            f"Equity = ${row['equity']:.2f}, Cash = ${row['cash']:.2f}"
        )

    print(f"\n💰 Total equity across ALL accounts: ${total_equity:.2f}\n")


if __name__ == "__main__":
    print("=== Schwabdev auth + accounts test ===")
    print("If this is your first run, a browser window will open for Schwab login/authorization.")
    main()
