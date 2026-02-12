#!/usr/bin/env python3
"""
Simple Schwab Authentication
=============================
No database sync - just local tokens.json file.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import schwabdev

# Setup
TOKEN_PATH = Path("tokens.json")

def get_env(name: str, fallback: str | None = None) -> str:
    """Try lowercase first, then uppercase fallback."""
    val = os.getenv(name)
    if not val and fallback:
        val = os.getenv(fallback)
    return val or ""

def main():
    load_dotenv()
    
    # Read keys
    app_key = get_env("app_key", "SCHWAB_APP_KEY")
    app_secret = get_env("app_secret", "SCHWAB_APP_SECRET")
    callback_url = get_env("callback_url", "SCHWAB_CALLBACK_URL")
    
    if not app_key or not app_secret or not callback_url:
        raise RuntimeError(
            "Missing credentials in .env\n"
            "Need: app_key, app_secret, callback_url\n"
            "(or SCHWAB_APP_KEY, SCHWAB_APP_SECRET, SCHWAB_CALLBACK_URL)"
        )
    
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("SCHWAB AUTHENTICATION")
    print("="*80)
    
    if not TOKEN_PATH.exists():
        print("\n🔑 No token found - will open browser for login...")
    else:
        print("\n🔑 Found existing token - attempting to use it...")
    
    # Create client (will auto-refresh or re-auth if needed)
    client = schwabdev.Client(app_key, app_secret, callback_url)
    
    print("\n✅ Connected to Schwab!\n")
    
    # Test: Get linked accounts
    print("Fetching linked accounts...")
    linked_accounts = client.account_linked().json()
    print(f"\nFound {len(linked_accounts)} linked accounts:")
    
    for acct in linked_accounts:
        masked = acct.get("accountNumberMasked", "???")
        acct_type = acct.get("accountType", "???")
        hash_val = acct.get("hashValue", "???")[:16]
        print(f"  • {masked} | {acct_type} | hash={hash_val}...")
    
    # Test: Get account details with balances
    print("\nFetching account balances...")
    details = client.account_details_all().json()
    
    total_equity = 0.0
    for acct in details:
        sa = acct.get("securitiesAccount", {})
        account_number = sa.get("accountNumber", "???")
        balances = sa.get("currentBalances", {}) or {}
        equity = float(balances.get("equity", 0) or 0)
        cash = float(balances.get("cashBalance", 0) or 0)
        
        total_equity += equity
        print(f"  • Account {account_number}: Equity=${equity:,.2f}, Cash=${cash:,.2f}")
    
    print(f"\n💰 Total equity across all accounts: ${total_equity:,.2f}")
    print(f"\n✅ Token saved to: {TOKEN_PATH}")
    print("✅ You can now run your trading scripts!\n")

if __name__ == "__main__":
    main()