"""
Diagnostic: Find Valid Transaction Date Ranges
===============================================

Tests different start dates to find when your accounts actually have data.
"""

from datetime import date, timedelta
from portfolio_snapshot import make_client
from analytics_core_enhanced import get_linked_accounts, fetch_all_transactions

def test_date_ranges(client, account_hash, account_number):
    """Test progressively more recent dates to find valid range"""
    
    print(f"\nTesting account {account_number} (hash: {account_hash[:16]}...)")
    
    # Test dates (progressively more recent)
    test_dates = [
        date(2020, 1, 1),
        date(2021, 1, 1),
        date(2022, 1, 1),
        date(2023, 1, 1),
        date(2024, 1, 1),
        date(2025, 1, 1),
        date.today() - timedelta(days=90),  # 3 months ago
        date.today() - timedelta(days=30),  # 1 month ago
    ]
    
    valid_start = None
    
    for test_date in test_dates:
        end_date = min(test_date + timedelta(days=7), date.today())
        
        print(f"  Testing {test_date} to {end_date}...", end="")
        
        try:
            txs = fetch_all_transactions(client, account_hash, test_date, end_date)
            
            if txs:
                print(f" ✓ FOUND {len(txs)} transactions!")
                if valid_start is None:
                    valid_start = test_date
            else:
                print(f" - No transactions (but no error)")
        
        except Exception as e:
            print(f" ✗ ERROR: {str(e)[:50]}")
    
    if valid_start:
        print(f"\n  → Recommended start date: {valid_start}")
    else:
        print(f"\n  → No valid date range found - account may be empty")
    
    return valid_start


def main():
    print("=" * 80)
    print("DIAGNOSING TRANSACTION DATE RANGES")
    print("=" * 80)
    
    client = make_client()
    accounts = get_linked_accounts(client)
    
    print(f"\nFound {len(accounts)} accounts")
    print("\nTesting each account to find valid transaction date ranges...")
    
    results = {}
    
    for acct in accounts:
        acc_num = acct["account_number"]
        acc_hash = acct["hash"]
        
        valid_start = test_date_ranges(client, acc_hash, acc_num)
        results[acc_num] = valid_start
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    earliest_date = None
    
    for acc_num, start_date in results.items():
        if start_date:
            print(f"Account {acc_num}: Data available from {start_date}")
            if earliest_date is None or start_date < earliest_date:
                earliest_date = start_date
        else:
            print(f"Account {acc_num}: No transaction data found")
    
    if earliest_date:
        print(f"\n✓ RECOMMENDED: Use start_date = '{earliest_date}' in your reports")
        print(f"\nUpdate historical_pl_report_enhanced.py line ~40:")
        print(f'  start_date = "{earliest_date}"')
    else:
        print("\n⚠ WARNING: No transaction data found in any account!")
        print("  Possible reasons:")
        print("  1. Accounts are new with no trading history yet")
        print("  2. API permissions don't include historical transactions")
        print("  3. Schwab API limitations on date ranges")


if __name__ == "__main__":
    main()