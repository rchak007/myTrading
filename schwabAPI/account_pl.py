#!/usr/bin/env python3
"""
Account P&L Analyzer
====================
Reads a single account's CSV cache file and lists all trades.

Usage:
    python account_pl.py <account_hash>.csv
    python account_pl.py 0422CE2E96848FDF660FC4C76F25CC63FD00526DE42E04C27E5D70E493176FD3.csv
"""

import sys
import json
import pandas as pd
from pathlib import Path


def extract_trades_from_csv(csv_path):
    """
    Read CSV and extract all TRADE transactions.
    
    Returns:
        DataFrame with columns: date, symbol, positionEffect, qty, price, cost, amount
    """
    
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print("DF count = ", len(df))
    # print("Df head = ", df.head())
    # print("Df tail = ", df.tail())
    
    # Filter to TRADE type only
    trades_df = df[df['type'] == 'TRADE'].copy()
    # print("trade DF count = ", len(trades_df))
    
    print(f"\nFound {len(trades_df)} TRADE transactions\n")
    
    rows = []
    
    for _, row in trades_df.iterrows():
        if pd.isna(row['raw_json']):
            continue
        
        try:
            tx = json.loads(row['raw_json'])
        except:
            print(f"⚠️  Failed to parse JSON for transaction on {row['date']}")
            continue
        
        # Extract fields
        symbol = None
        qty = 0.0
        price = 0.0
        cost = 0.0
        amount = 0.0
        position_effect = ""
        
        # Check both transaction structures
        if 'transactionItem' in tx:
            item = tx['transactionItem']
            instrument = item.get('instrument', {})
            symbol = instrument.get('symbol')
            qty = float(item.get('amount', 0) or 0)
            price = float(item.get('price', 0) or 0)
            cost = float(item.get('cost', 0) or 0)
            amount = abs(qty) * price
            position_effect = item.get('instruction', '') or item.get('positionEffect', '')
        
        elif 'transferItems' in tx:
            items = tx.get('transferItems', [])
            if items and len(items) > 0:
                item = items[0]
                instrument = item.get('instrument', {})
                symbol = instrument.get('symbol')
                qty = float(item.get('amount', 0) or 0)
                price = float(item.get('price', 0) or 0)
                cost = float(item.get('cost', 0) or 0)
                amount = abs(qty) * price
                position_effect = item.get('positionEffect', '') or item.get('instruction', '')
        
        if not symbol:
            print(f"⚠️  No symbol found for transaction on {row['date']}")
            continue
        
        # Skip CURRENCY_USD
        if symbol == 'CURRENCY_USD':
            continue
        
        rows.append({
            'date': row['date'],
            'symbol': symbol,
            'positionEffect': position_effect,
            'qty': qty,
            'price': price,
            'cost': cost,
            'amount': amount,
        })
    
    if not rows:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(rows)
    
    # Sort by symbol then date
    result_df = result_df.sort_values(['symbol', 'date'])
    
    return result_df


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: Please provide CSV file path")
        print("Example: python account_pl.py data/transactions/0422CE2E96848FDF*.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    print("=" * 120)
    print("ACCOUNT TRADE LISTING")
    print("=" * 120)
    print(f"\nReading: {csv_file}")
    
    # Extract trades
    trades_df = extract_trades_from_csv(csv_file)
    
    if trades_df.empty:
        print("\n❌ No trades found!\n")
        return
    
    # Display all trades - NO LIMITS
    pd.set_option('display.max_rows', 10000)  # Show up to 10k rows
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.float_format', lambda x: f'{x:.2f}')
    
    print("\n" + "=" * 120)
    print("ALL TRADES (sorted by symbol, then date)")
    print("=" * 120 + "\n")
    
    print(trades_df.to_string(index=False))
    
    print(f"\n\nTotal trades: {len(trades_df)}\n")


if __name__ == "__main__":
    main()