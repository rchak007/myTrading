"""
Historical Per-Ticker P&L Report
=================================

Uses YOUR EXISTING working cache system (no changes).
Prints every transaction as it processes.
"""

from datetime import date
import pandas as pd
import json

from portfolio_snapshot import make_client, get_positions_all_accounts
from analytics_core import (
    get_linked_accounts,
    get_transactions_cached_list,
)


def process_transactions(client, start_date="2020-01-01"):
    """Process all transactions and calculate per-ticker P&L"""
    
    accounts = get_linked_accounts(client)
    print(f"\nFound {len(accounts)} accounts\n")
    
    # Track everything per ticker
    ticker_data = {}  # symbol -> {buys: [], sells: [], dividends: 0, ...}
    
    for acct in accounts:
        acc_num = acct["account_number"]
        acc_hash = acct["hash"]
        
        print(f"{'='*80}")
        print(f"Account: {acc_num}")
        print(f"{'='*80}")
        
        # Get cached transactions - pass empty types to avoid filter
        txs = get_transactions_cached_list(client, acc_hash, start_date, date.today(), types=[])
        
        print(f"Total transactions: {len(txs)}\n")
        
        # Debug: show first transaction structure
        if txs and len(txs) > 0:
            print(f"DEBUG: First transaction keys: {list(txs[0].keys())}")
            print(f"DEBUG: First transaction type: {txs[0].get('type')}")
            if 'transactionItem' in txs[0]:
                print(f"DEBUG: transactionItem keys: {list(txs[0]['transactionItem'].keys())}")
            print()
        
        # Process each transaction
        for tx in txs:
            tx_type = tx.get("type", "")
            
            # Handle different transaction structures
            # Some have transactionItem, others have transferItems
            symbol = None
            qty = 0.0
            price = 0.0
            instruction = ""
            
            if "transactionItem" in tx:
                # Old structure
                item = tx["transactionItem"]
                instrument = item.get("instrument", {})
                symbol = instrument.get("symbol")
                qty = float(item.get("amount", 0) or 0)
                price = float(item.get("price", 0) or 0)
                instruction = item.get("instruction", "")
            
            elif "transferItems" in tx:
                # New structure - transferItems is a list
                transfer_items = tx.get("transferItems", [])
                if transfer_items and len(transfer_items) > 0:
                    item = transfer_items[0]  # Take first item
                    instrument = item.get("instrument", {})
                    symbol = instrument.get("symbol")
                    qty = float(item.get("amount", 0) or 0)
                    price = float(item.get("price", 0) or 0)
                    instruction = item.get("positionEffect", "")  # Different field name!
            
            if not symbol:
                continue
            
            # Initialize ticker data if needed
            if symbol not in ticker_data:
                ticker_data[symbol] = {
                    "buys": [],
                    "sells": [],
                    "dividends": 0.0,
                }
            
            # Get date
            trade_date = tx.get("tradeDate") or tx.get("transactionDate") or ""
            if isinstance(trade_date, str) and len(trade_date) >= 10:
                trade_date = trade_date[:10]
            
            # Handle TRADE and RECEIVE_AND_DELIVER transactions
            if tx_type == "TRADE" or tx_type == "RECEIVE_AND_DELIVER":
                # Check instruction first
                if "BUY" in instruction.upper() or "OPENING" in instruction.upper():
                    ticker_data[symbol]["buys"].append({
                        "date": trade_date,
                        "qty": abs(qty),
                        "price": price,
                    })
                    print(f"  BUY  {symbol:8s} {abs(qty):10.2f} @ ${price:10.2f}  ({trade_date})")
                
                elif "SELL" in instruction.upper() or "CLOSING" in instruction.upper():
                    ticker_data[symbol]["sells"].append({
                        "date": trade_date,
                        "qty": abs(qty),
                        "price": price,
                    })
                    print(f"  SELL {symbol:8s} {abs(qty):10.2f} @ ${price:10.2f}  ({trade_date})")
                
                # If no clear instruction, use qty sign
                elif qty > 0:
                    ticker_data[symbol]["buys"].append({
                        "date": trade_date,
                        "qty": abs(qty),
                        "price": price,
                    })
                    print(f"  BUY  {symbol:8s} {abs(qty):10.2f} @ ${price:10.2f}  ({trade_date})")
                
                elif qty < 0:
                    ticker_data[symbol]["sells"].append({
                        "date": trade_date,
                        "qty": abs(qty),
                        "price": price,
                    })
                    print(f"  SELL {symbol:8s} {abs(qty):10.2f} @ ${price:10.2f}  ({trade_date})")
            
            # Handle DIVIDEND transactions
            elif tx_type == "DIVIDEND_OR_INTEREST":
                amount = float(tx.get("netAmount", 0) or 0)
                ticker_data[symbol]["dividends"] += amount
                print(f"  DIV  {symbol:8s} ${amount:10.2f}  ({trade_date})")
        
        print()
    
    return ticker_data


def calculate_pl(ticker_data, current_positions):
    """Calculate P&L for each ticker using FIFO"""
    
    results = []
    
    for symbol, data in ticker_data.items():
        # Sort buys by date
        buys = sorted(data["buys"], key=lambda x: x["date"])
        sells = sorted(data["sells"], key=lambda x: x["date"])
        
        # FIFO calculation
        lots = [{"qty": b["qty"], "price": b["price"]} for b in buys]
        realized_pl = 0.0
        
        for sell in sells:
            remaining = sell["qty"]
            
            while remaining > 0 and lots:
                lot = lots[0]
                
                if lot["qty"] <= remaining:
                    # Sell entire lot
                    realized_pl += (sell["price"] - lot["price"]) * lot["qty"]
                    remaining -= lot["qty"]
                    lots.pop(0)
                else:
                    # Partial sell
                    realized_pl += (sell["price"] - lot["price"]) * remaining
                    lot["qty"] -= remaining
                    remaining = 0
        
        # Current holdings
        current_qty = sum(lot["qty"] for lot in lots)
        cost_basis = sum(lot["qty"] * lot["price"] for lot in lots)
        avg_price = cost_basis / current_qty if current_qty > 0 else 0.0
        
        # Get current market value
        pos = current_positions[current_positions["symbol"] == symbol]
        market_value = pos["market_value"].sum() if not pos.empty else 0.0
        
        # Calculate unrealized P&L
        unrealized_pl = market_value - cost_basis
        
        # Total P&L
        dividends = data["dividends"]
        total_pl = realized_pl + unrealized_pl + dividends
        total_pl_pct = (total_pl / cost_basis * 100) if cost_basis > 0 else 0.0
        
        results.append({
            "symbol": symbol,
            "qty": current_qty,
            "cost_basis": cost_basis,
            "avg_price": avg_price,
            "market_value": market_value,
            "unrealized_pl": unrealized_pl,
            "realized_pl": realized_pl,
            "dividends": dividends,
            "total_pl": total_pl,
            "total_pl_pct": total_pl_pct,
        })
    
    if not results:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=[
            "symbol", "qty", "cost_basis", "avg_price", "market_value",
            "unrealized_pl", "realized_pl", "dividends", "total_pl", "total_pl_pct"
        ])
    
    return pd.DataFrame(results).sort_values("total_pl", ascending=False)


def main():
    client = make_client()
    
    print("="*80)
    print("HISTORICAL PER-TICKER P&L REPORT")
    print("="*80)
    print("\nUsing your existing cache system (data/transactions/*.state.json)")
    print("Start date: 2020-01-01\n")
    
    # Process all transactions
    ticker_data = process_transactions(client, start_date="2020-01-01")
    
    print(f"\n{'='*80}")
    print(f"DEBUG: Found {len(ticker_data)} unique tickers")
    print(f"{'='*80}\n")
    
    if not ticker_data:
        print("❌ No ticker data found! Check transaction parsing.\n")
        return
    
    # Get current positions
    print("="*80)
    print("FETCHING CURRENT POSITIONS...")
    print("="*80 + "\n")
    
    positions_df, _ = get_positions_all_accounts(client)
    
    # Calculate P&L
    print("="*80)
    print("CALCULATING P&L...")
    print("="*80 + "\n")
    
    pl_df = calculate_pl(ticker_data, positions_df)
    
    if pl_df.empty:
        print("No trading data found.\n")
        return
    
    # Display results
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    pd.set_option("display.width", 200)
    
    print("="*80)
    print("GRAND TOTAL")
    print("="*80)
    
    print(f"Total Cost Basis:      ${pl_df['cost_basis'].sum():,.2f}")
    print(f"Total Market Value:    ${pl_df['market_value'].sum():,.2f}")
    print(f"Total Unrealized P&L:  ${pl_df['unrealized_pl'].sum():,.2f}")
    print(f"Total Realized P&L:    ${pl_df['realized_pl'].sum():,.2f}")
    print(f"Total Dividends:       ${pl_df['dividends'].sum():,.2f}")
    print(f"GRAND TOTAL P&L:       ${pl_df['total_pl'].sum():,.2f}")
    
    total_return = (pl_df['total_pl'].sum() / pl_df['cost_basis'].sum() * 100) if pl_df['cost_basis'].sum() > 0 else 0
    print(f"GRAND RETURN %:        {total_return:,.2f}%")
    
    print("\n" + "="*80)
    print("PER-TICKER P&L")
    print("="*80 + "\n")
    
    print(pl_df.to_string(index=False))
    
    # Save
    pl_df.to_csv("data/per_ticker_pl.csv", index=False)
    print("\n✓ Saved to: data/per_ticker_pl.csv\n")


if __name__ == "__main__":
    main()