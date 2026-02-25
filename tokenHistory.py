#!/usr/bin/env python3
"""
tokenHistory.py
---------------
Fetches complete ZEUS / USDC transaction history for a Solana wallet.
Shows swaps, deposits, withdrawals and calculates P&L.

Requirements:
    pip install requests tabulate python-dotenv

API key:
    Get a FREE Helius API key at https://helius.dev  (no credit card needed)
    Then either:
      - Set env var:  export HELIUS_API_KEY=your_key_here
      - Or create a .env file with:  HELIUS_API_KEY=your_key_here
"""

import os
import sys
import json
import time
import requests
from datetime import datetime, timezone
from tabulate import tabulate

# ── Try to load .env if present ──────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Config ───────────────────────────────────────────────────────────────────
HELIUS_API_KEY    = os.getenv("HELIUS_API_KEY", "YOUR_HELIUS_API_KEY_HERE")

WALLET_ADDRESS    = "EcBwr9rirS2MDjSUjNufp2VRqqzEAm7zNycN7C7ryxjG"
ZEUS_MINT         = "ZEUS1aR7aX8DFFJf5QjWj2ftDDdNTroMNGo8YoQm3Gq"
USDC_MINT         = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

HELIUS_BASE       = "https://api.helius.xyz/v0"
LIMIT             = 100  # transactions per page (max 100)

# ── Helpers ───────────────────────────────────────────────────────────────────

def ts_to_dt(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# ── Fetch all parsed transactions from Helius ─────────────────────────────────

def fetch_all_transactions(wallet: str) -> list:
    """Pages through all wallet transactions using Helius Enhanced Transactions API."""
    all_txs = []
    before_sig = None

    print(f"Fetching transactions for wallet: {wallet}")
    page = 0

    while True:
        page += 1
        url = f"{HELIUS_BASE}/addresses/{wallet}/transactions"
        params = {
            "api-key": HELIUS_API_KEY,
            "limit":   LIMIT,
        }
        if before_sig:
            params["before"] = before_sig

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"\nHTTP error: {e}")
            if resp.status_code == 401:
                print("→ Invalid or missing HELIUS_API_KEY. Get one free at https://helius.dev")
            sys.exit(1)
        except requests.exceptions.RequestException as e:
            print(f"\nNetwork error: {e}")
            sys.exit(1)

        data = resp.json()

        if not data:
            break

        all_txs.extend(data)
        print(f"  Page {page}: fetched {len(data)} txs (total so far: {len(all_txs)})")

        if len(data) < LIMIT:
            break   # last page

        before_sig = data[-1]["signature"]
        time.sleep(0.3)  # be polite to the API

    print(f"Total transactions fetched: {len(all_txs)}\n")
    return all_txs

# ── Parse each transaction for ZEUS / USDC movements ─────────────────────────

def parse_transactions(txs: list) -> list:
    """
    Extracts ZEUS and USDC token movements from each transaction.
    Returns a list of event dicts.
    """
    events = []

    for tx in txs:
        sig       = tx.get("signature", "")
        ts        = tx.get("timestamp", 0)
        tx_type   = tx.get("type", "UNKNOWN")          # e.g. SWAP, TRANSFER, etc.
        fee_payer = tx.get("feePayer", "")

        # ── tokenTransfers: direct token movements ────────────────────────────
        token_transfers = tx.get("tokenTransfers", []) or []

        zeus_delta = 0.0   # positive = received, negative = sent
        usdc_delta = 0.0

        for tt in token_transfers:
            mint      = tt.get("mint", "")
            from_acct = tt.get("fromUserAccount", "")
            to_acct   = tt.get("toUserAccount", "")

            # Helius Enhanced API returns tokenAmount as a float already
            # in human/UI units (e.g. 4345.0 ZEUS, not 4345000000 raw).
            # Do NOT divide by decimals again.
            amount = float(tt.get("tokenAmount", 0))

            if mint == ZEUS_MINT:
                if to_acct == WALLET_ADDRESS:
                    zeus_delta += amount
                elif from_acct == WALLET_ADDRESS:
                    zeus_delta -= amount

            elif mint == USDC_MINT:
                if to_acct == WALLET_ADDRESS:
                    usdc_delta += amount
                elif from_acct == WALLET_ADDRESS:
                    usdc_delta -= amount

        # Skip txs with no ZEUS or USDC involvement
        if zeus_delta == 0.0 and usdc_delta == 0.0:
            continue

        # ── Determine event type & implied price ──────────────────────────────
        #
        # Swap logic:
        #   Bought ZEUS  → zeus_delta > 0, usdc_delta < 0   → price = |usdc_delta| / zeus_delta
        #   Sold ZEUS    → zeus_delta < 0, usdc_delta > 0   → price = usdc_delta / |zeus_delta|
        #   ZEUS deposit → zeus_delta > 0, usdc_delta == 0
        #   USDC deposit → usdc_delta > 0, zeus_delta == 0

        implied_price = None

        if zeus_delta > 0 and usdc_delta < 0:
            event_label   = "BUY ZEUS"
            implied_price = abs(usdc_delta) / zeus_delta

        elif zeus_delta < 0 and usdc_delta > 0:
            event_label   = "SELL ZEUS"
            implied_price = usdc_delta / abs(zeus_delta)

        elif zeus_delta > 0 and usdc_delta == 0:
            event_label   = "ZEUS DEPOSIT"

        elif zeus_delta < 0 and usdc_delta == 0:
            event_label   = "ZEUS WITHDRAW"

        elif usdc_delta > 0 and zeus_delta == 0:
            event_label   = "USDC DEPOSIT"

        elif usdc_delta < 0 and zeus_delta == 0:
            event_label   = "USDC WITHDRAW"

        else:
            event_label   = tx_type   # fallback (e.g. complex multi-token swap)

        events.append({
            "datetime":      ts_to_dt(ts),
            "timestamp":     ts,
            "type":          event_label,
            "zeus_delta":    zeus_delta,
            "usdc_delta":    usdc_delta,
            "price_usd":     implied_price,
            "signature":     sig,
        })

    # Sort chronologically
    events.sort(key=lambda x: x["timestamp"])
    return events

# ── Build running balances & P&L ──────────────────────────────────────────────

def compute_pnl(events: list) -> dict:
    """
    Tracks running ZEUS & USDC balances.
    Computes realized P&L using FIFO cost basis for ZEUS.
    """
    zeus_balance  = 0.0
    usdc_balance  = 0.0
    cost_basis_q  = []   # FIFO queue of (qty, price_per_unit)
    realized_pnl  = 0.0
    total_zeus_bought = 0.0
    total_usdc_spent  = 0.0
    total_zeus_sold   = 0.0
    total_usdc_received = 0.0

    for ev in events:
        ev["zeus_balance"] = 0.0
        ev["usdc_balance"] = 0.0
        ev["realized_pnl"] = None

        zeus_balance += ev["zeus_delta"]
        usdc_balance += ev["usdc_delta"]

        if ev["type"] == "BUY ZEUS" and ev["price_usd"]:
            cost_basis_q.append((abs(ev["zeus_delta"]), ev["price_usd"]))
            total_zeus_bought += abs(ev["zeus_delta"])
            total_usdc_spent  += abs(ev["usdc_delta"])

        elif ev["type"] == "SELL ZEUS" and ev["price_usd"]:
            sell_qty   = abs(ev["zeus_delta"])
            sell_price = ev["price_usd"]
            total_zeus_sold     += sell_qty
            total_usdc_received += abs(ev["usdc_delta"])

            # FIFO cost basis
            remaining = sell_qty
            batch_pnl = 0.0
            while remaining > 0 and cost_basis_q:
                lot_qty, lot_price = cost_basis_q[0]
                used = min(remaining, lot_qty)
                batch_pnl += used * (sell_price - lot_price)
                remaining -= used
                if used == lot_qty:
                    cost_basis_q.pop(0)
                else:
                    cost_basis_q[0] = (lot_qty - used, lot_price)
            realized_pnl      += batch_pnl
            ev["realized_pnl"] = batch_pnl

        ev["zeus_balance"] = zeus_balance
        ev["usdc_balance"] = usdc_balance

    # Unrealized P&L requires current price — fetch from Jupiter
    current_price = fetch_current_zeus_price()
    unrealized_pnl = None
    if current_price:
        unrealized_pnl = zeus_balance * current_price
        # subtract remaining cost basis
        remaining_cost = sum(q * p for q, p in cost_basis_q)
        unrealized_pnl -= remaining_cost

    return {
        "events":             events,
        "final_zeus_balance": zeus_balance,
        "final_usdc_balance": usdc_balance,
        "total_zeus_bought":  total_zeus_bought,
        "total_usdc_spent":   total_usdc_spent,
        "total_zeus_sold":    total_zeus_sold,
        "total_usdc_received":total_usdc_received,
        "realized_pnl":       realized_pnl,
        "current_price":      current_price,
        "unrealized_pnl":     unrealized_pnl,
    }

# ── Fetch current ZEUS price from Jupiter ─────────────────────────────────────

def fetch_current_zeus_price() -> float | None:
    """Fetch current ZEUS price in USD, trying multiple sources."""

    # ── 1. Jupiter Price API v2 (no vsToken param needed) ────────────────────
    try:
        resp = requests.get(
            "https://api.jup.ag/price/v2",
            params={"ids": ZEUS_MINT},
            timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            price = data.get("data", {}).get(ZEUS_MINT, {}).get("price")
            if price:
                return float(price)
    except Exception:
        pass

    # ── 2. Jupiter Quote API (swap ZEUS → USDC for 1 ZEUS) ───────────────────
    try:
        resp = requests.get(
            "https://quote-api.jup.ag/v6/quote",
            params={
                "inputMint":  ZEUS_MINT,
                "outputMint": USDC_MINT,
                "amount":     1_000_000,   # 1 ZEUS in raw units (6 decimals)
                "slippageBps": 50,
            },
            timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            out_amount = data.get("outAmount")
            if out_amount:
                # outAmount is USDC raw units (6 decimals)
                return int(out_amount) / 1_000_000
    except Exception:
        pass

    # ── 3. CoinGecko (no key needed for basic endpoint) ──────────────────────
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/token_price/solana",
            params={
                "contract_addresses": ZEUS_MINT,
                "vs_currencies":      "usd",
            },
            timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            price = data.get(ZEUS_MINT.lower(), {}).get("usd")
            if price:
                return float(price)
    except Exception:
        pass

    print("  [Warning] Could not fetch current ZEUS price from any source.")
    return None

# ── Display ───────────────────────────────────────────────────────────────────

def display_results(result: dict):
    events = result["events"]

    if not events:
        print("No ZEUS or USDC activity found for this wallet.")
        return

    # ── Transaction table ─────────────────────────────────────────────────────
    rows = []
    for ev in events:
        price_str = f"${ev['price_usd']:.4f}" if ev["price_usd"] else "—"
        pnl_str   = f"${ev['realized_pnl']:+.2f}" if ev["realized_pnl"] is not None else "—"
        rows.append([
            ev["datetime"],
            ev["type"],
            f"{ev['zeus_delta']:+.4f}"  if ev["zeus_delta"] != 0 else "—",
            f"{ev['usdc_delta']:+.2f}"  if ev["usdc_delta"] != 0 else "—",
            price_str,
            f"{ev['zeus_balance']:.4f}",
            f"{ev['usdc_balance']:.2f}",
            pnl_str,
            ev["signature"][:12] + "...",
        ])

    headers = [
        "Date/Time (UTC)", "Type",
        "ZEUS Δ", "USDC Δ",
        "Price (USDC)",
        "ZEUS Bal", "USDC Bal",
        "Realized P&L",
        "Signature"
    ]

    print("=" * 130)
    print("  ZEUS / USDC WALLET HISTORY")
    print(f"  Wallet: {WALLET_ADDRESS}")
    print("=" * 130)
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Final ZEUS balance:       {result['final_zeus_balance']:.4f} ZEUS")
    print(f"  Final USDC balance:       {result['final_usdc_balance']:.2f} USDC")
    print(f"  Total ZEUS bought:        {result['total_zeus_bought']:.4f} ZEUS")
    print(f"  Total USDC spent buying:  ${result['total_usdc_spent']:.2f}")
    print(f"  Total ZEUS sold:          {result['total_zeus_sold']:.4f} ZEUS")
    print(f"  Total USDC from selling:  ${result['total_usdc_received']:.2f}")
    print(f"  Realized P&L (FIFO):      ${result['realized_pnl']:+.2f}")

    if result["current_price"]:
        print(f"\n  Current ZEUS price:       ${result['current_price']:.4f}")
    if result["unrealized_pnl"] is not None:
        print(f"  Unrealized P&L:           ${result['unrealized_pnl']:+.2f}")
        total = result["realized_pnl"] + result["unrealized_pnl"]
        print(f"  Total P&L (realized+unr): ${total:+.2f}")

    net_usdc = result["total_usdc_received"] - result["total_usdc_spent"]
    print(f"\n  Net USDC flow from trading: ${net_usdc:+.2f}")
    if net_usdc > 0:
        print("  → You've extracted more USDC than you put in (profitable trading!)")
    else:
        print("  → You've put in more USDC than extracted (net loss on closed trades)")

    print("=" * 60)

# ── Save JSON output ──────────────────────────────────────────────────────────

def save_json(result: dict, filename: str = "zeus_history.json"):
    # Make it serializable
    out = {k: v for k, v in result.items() if k != "events"}
    out["events"] = result["events"]
    with open(filename, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Full data saved to: {filename}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if HELIUS_API_KEY == "YOUR_HELIUS_API_KEY_HERE":
        print("ERROR: Please set your Helius API key.")
        print("  1. Sign up free at https://helius.dev")
        print("  2. Copy your API key")
        print("  3. Set env var:  export HELIUS_API_KEY=your_key")
        print("     or add to .env file: HELIUS_API_KEY=your_key")
        sys.exit(1)

    print("\n── ZEUS Token History Analyzer ─────────────────────────────\n")

    # Fetch
    txs = fetch_all_transactions(WALLET_ADDRESS)

    # Parse
    events = parse_transactions(txs)
    print(f"Events involving ZEUS or USDC: {len(events)}\n")

    # Compute P&L
    result = compute_pnl(events)

    # Display
    display_results(result)

    # Save
    save_json(result)


if __name__ == "__main__":
    main()