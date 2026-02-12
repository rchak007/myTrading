#!/usr/bin/env python3
"""
Cache Management Tool for Schwab Transactions
==============================================

Manages both:
- {hash}.csv files (transaction data)
- {hash}.state.json files (metadata: last_date)
"""

import sys
import json
from pathlib import Path

CACHE_DIR = Path("data/transactions")

def list_cache_files():
    """List all cached accounts with their metadata"""
    if not CACHE_DIR.exists():
        print("❌ Cache directory doesn't exist")
        return
    
    csv_files = sorted(CACHE_DIR.glob("*.csv"))
    state_files = sorted(CACHE_DIR.glob("*.state.json"))
    
    if not csv_files and not state_files:
        print("📭 No cache files found")
        return
    
    print("\n" + "="*80)
    print("CACHED ACCOUNTS")
    print("="*80)
    
    # Show CSV files (transaction data)
    if csv_files:
        print("\n📊 Transaction Data (.csv):")
        for f in csv_files:
            size_kb = f.stat().st_size / 1024
            lines = sum(1 for _ in open(f)) - 1  # subtract header
            print(f"  • {f.stem[:16]}... | {size_kb:,.1f} KB | {lines:,} transactions")
    
    # Show state files (metadata)
    if state_files:
        print("\n📅 Metadata (.state.json):")
        for f in state_files:
            try:
                data = json.loads(f.read_text())
                last_date = data.get("last_date", "unknown")
                print(f"  • {f.stem[:16]}... | last_date: {last_date}")
            except:
                print(f"  • {f.stem[:16]}... | ⚠️  corrupted")
    
    print()

def clear_cache_for_account(account_hash):
    """Delete cache files for a specific account"""
    csv_file = CACHE_DIR / f"{account_hash}.csv"
    state_file = CACHE_DIR / f"{account_hash}.state.json"
    
    deleted = []
    if csv_file.exists():
        csv_file.unlink()
        deleted.append("CSV")
    if state_file.exists():
        state_file.unlink()
        deleted.append("state.json")
    
    if deleted:
        print(f"✓ Deleted {account_hash[:16]}... ({', '.join(deleted)})")
    else:
        print(f"❌ No cache found for {account_hash}")

def clear_all_cache():
    """Delete all cache files"""
    if not CACHE_DIR.exists():
        print("❌ Cache directory doesn't exist")
        return
    
    csv_files = list(CACHE_DIR.glob("*.csv"))
    state_files = list(CACHE_DIR.glob("*.state.json"))
    total = len(csv_files) + len(state_files)
    
    if total == 0:
        print("📭 No cache files to delete")
        return
    
    # Confirm
    response = input(f"\n⚠️  WARNING: This will delete {total} files. Continue? (yes/no): ")
    if response.lower() != "yes":
        print("❌ Cancelled")
        return
    
    # Delete
    for f in csv_files:
        f.unlink()
        print(f"✓ Deleted: {f.name}")
    
    for f in state_files:
        f.unlink()
        print(f"✓ Deleted: {f.name}")
    
    print(f"\n✓ Cleared all cache ({total} files deleted)")

def reset_cache_date(account_hash, new_date):
    """Update the last_date in state file"""
    state_file = CACHE_DIR / f"{account_hash}.state.json"
    
    if not state_file.exists():
        print(f"❌ No state file for {account_hash}")
        return
    
    try:
        data = json.loads(state_file.read_text())
        old_date = data.get("last_date", "none")
        data["last_date"] = new_date
        state_file.write_text(json.dumps(data, indent=2))
        print(f"✓ Updated {account_hash[:16]}...")
        print(f"  Old: {old_date}")
        print(f"  New: {new_date}")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage:")
        print("  python manage_cache.py list")
        print("  python manage_cache.py clear <account_hash>")
        print("  python manage_cache.py clear-all")
        print("  python manage_cache.py reset-date <account_hash> <YYYY-MM-DD>")
        return
    
    cmd = sys.argv[1]
    
    if cmd == "list":
        list_cache_files()
    
    elif cmd == "clear":
        if len(sys.argv) < 3:
            print("❌ Usage: manage_cache.py clear <account_hash>")
        else:
            clear_cache_for_account(sys.argv[2])
    
    elif cmd == "clear-all":
        clear_all_cache()
    
    elif cmd == "reset-date":
        if len(sys.argv) < 4:
            print("❌ Usage: manage_cache.py reset-date <account_hash> <YYYY-MM-DD>")
        else:
            reset_cache_date(sys.argv[2], sys.argv[3])
    
    else:
        print(f"❌ Unknown command: {cmd}")

if __name__ == "__main__":
    main()