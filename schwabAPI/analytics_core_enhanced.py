import logging
from datetime import date, datetime, timedelta
from typing import List, Dict
from collections import defaultdict

import pandas as pd

from portfolio_snapshot import make_client, get_account_balances, get_positions_all_accounts

import os
import json
from pathlib import Path
import time
from requests.exceptions import ReadTimeout, RequestException, HTTPError

# Use existing cache infrastructure
CACHE_DIR = Path("data/transactions")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# =============================================================================
# CACHE INFRASTRUCTURE (keep existing functions)
# =============================================================================

def _ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _state_path(account_hash: str) -> Path:
    return CACHE_DIR / f"{account_hash}.state.json"

def _csv_path(account_hash: str) -> Path:
    return CACHE_DIR / f"{account_hash}.csv"

def load_last_cached_date(account_hash: str):
    p = _state_path(account_hash)
    if not p.exists():
        return None
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
        d = s.get("last_date")
        return datetime.strptime(d, "%Y-%m-%d").date() if d else None
    except Exception:
        return None

def save_last_cached_date(account_hash: str, last_date: date):
    p = _state_path(account_hash)
    p.write_text(json.dumps({"last_date": last_date.strftime("%Y-%m-%d")}, indent=2), encoding="utf-8")


# =============================================================================
# LINKED ACCOUNTS
# =============================================================================

def get_linked_accounts(client) -> List[Dict]:
    """Return list of linked accounts with accountNumber, hashValue, displayName"""
    resp = client.account_linked()
    data = resp.json()
    
    accounts = []
    for a in data:
        accounts.append({
            "account_number": a.get("accountNumber"),
            "hash": a.get("hashValue"),
            "display_name": a.get("displayName") or a.get("accountNumber"),
        })
    
    logger.info("Found %d linked accounts", len(accounts))
    return accounts


# =============================================================================
# TRANSACTION FETCHING (with retry logic)
# =============================================================================

def _fetch_transactions_chunk(client, account_hash, start_date, end_date, types=None):
    """Fetch a chunk of transactions with retry logic"""
    try:
        from urllib3.exceptions import ReadTimeoutError
        timeout_exceptions = (ReadTimeout, ReadTimeoutError)
    except Exception:
        timeout_exceptions = (ReadTimeout,)
    
    start_str = start_date.strftime("%Y-%m-%dT00:00:00.000Z")
    end_str = end_date.strftime("%Y-%m-%dT23:59:59.999Z")
    
    if types is None:
        types_list = [
            "TRADE",
            "DIVIDEND_OR_INTEREST",
            "CORPORATE_ACTION",
            "SECURITY_TRANSFER",
            "ACH_RECEIPT",
            "ACH_DISBURSEMENT",
            "CASH_RECEIPT",
            "CASH_DISBURSEMENT",
            "ELECTRONIC_FUND",
            "WIRE_IN",
            "WIRE_OUT",
            "FEE",
            "TAX",
            "ADJUSTMENT",
            "JOURNAL",
            "MEMORANDUM",
        ]
    elif isinstance(types, (list, tuple)):
        types_list = list(types)
    else:
        types_list = [types]
    
    max_tries = 8
    backoff = 2
    last_err = None
    
    for attempt in range(1, max_tries + 1):
        try:
            resp = client.transactions(account_hash, start_str, end_str, types_list)
            
            if resp.status_code != 200:
                logger.error(
                    "HTTP %s for hash=%s %s..%s types=%s",
                    resp.status_code, account_hash, start_str, end_str, types_list
                )
                return []
            
            data = resp.json() or []
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                if isinstance(data.get("transactions"), list):
                    return data["transactions"]
                if isinstance(data.get("items"), list):
                    return data["items"]
            return []
        
        except timeout_exceptions as e:
            last_err = e
            logger.warning(
                "ReadTimeout fetching tx (attempt %d/%d) hash=%s %s..%s : %s",
                attempt, max_tries, account_hash, start_str, end_str, repr(e)
            )
        
        except HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 400:
                logger.error("HTTP 400 (Bad Request) for hash=%s %s..%s", account_hash, start_str, end_str)
                return []
            last_err = e
            logger.warning(
                "HTTP error fetching tx (attempt %d/%d) hash=%s: %s",
                attempt, max_tries, account_hash, repr(e)
            )
        
        except RequestException as e:
            last_err = e
            logger.warning(
                "Request error fetching tx (attempt %d/%d) hash=%s: %s",
                attempt, max_tries, account_hash, repr(e)
            )
        
        if attempt < max_tries:
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
        else:
            raise last_err


def fetch_all_transactions(
    client,
    account_hash: str,
    start_date: date,
    end_date: date | None = None,
    types: List[str] | None = None,
) -> List[Dict]:
    """Walk forward in chunks from start_date to end_date"""
    if end_date is None:
        end_date = date.today()
    
    if start_date > end_date:
        return []
    
    try:
        from urllib3.exceptions import ReadTimeoutError
        timeout_exceptions = (ReadTimeout, ReadTimeoutError)
    except Exception:
        timeout_exceptions = (ReadTimeout,)
    
    all_tx: List[Dict] = []
    chunk_days = 60
    min_chunk_days = 7
    cur_start = start_date
    
    while cur_start <= end_date:
        cur_end = min(cur_start + timedelta(days=chunk_days - 1), end_date)
        
        try:
            chunk = _fetch_transactions_chunk(
                client=client,
                account_hash=account_hash,
                start_date=cur_start,
                end_date=cur_end,
                types=types,
            )
        
        except timeout_exceptions:
            if chunk_days > min_chunk_days:
                chunk_days = max(min_chunk_days, chunk_days // 2)
                logger.warning(
                    "Timeout: shrinking chunk_days to %d and retrying hash=%s starting %s",
                    chunk_days, account_hash, cur_start.strftime("%Y-%m-%d"),
                )
                continue
            raise
        
        if chunk:
            all_tx.extend(chunk)
        
        logger.info(
            "Fetched %d tx for hash=%s from %s to %s",
            len(chunk) if chunk else 0, account_hash,
            cur_start.strftime("%Y-%m-%d"), cur_end.strftime("%Y-%m-%d")
        )
        
        cur_start = cur_end + timedelta(days=1)
    
    return all_tx


# =============================================================================
# CACHE OPERATIONS
# =============================================================================

def normalize_txns_to_df(txs: list, account_hash: str) -> pd.DataFrame:
    """Convert raw transactions to DataFrame with essential fields"""
    rows = []
    for t in txs or []:
        trade_date = t.get("tradeDate") or t.get("transactionDate") or t.get("postDate")
        d = None
        if isinstance(trade_date, str) and len(trade_date) >= 10:
            d = trade_date[:10]
        
        rows.append({
            "account_hash": account_hash,
            "transaction_id": t.get("transactionId") or t.get("id"),
            "date": d,
            "type": t.get("type"),
            "sub_type": t.get("subType"),
            "description": t.get("description") or t.get("memo"),
            "amount": t.get("amount"),
            "raw_json": json.dumps(t, ensure_ascii=False),
        })
    
    return pd.DataFrame(rows)


from pandas.errors import EmptyDataError

def append_cache(account_hash: str, new_df: pd.DataFrame) -> pd.DataFrame:
    """Append new transactions to cache, deduplicating by transaction_id"""
    csvp = _csv_path(account_hash)
    
    if new_df is None or new_df.empty:
        if csvp.exists() and csvp.stat().st_size == 0:
            csvp.unlink(missing_ok=True)
        if csvp.exists():
            try:
                return pd.read_csv(csvp)
            except EmptyDataError:
                csvp.unlink(missing_ok=True)
        return pd.DataFrame()
    
    old = pd.DataFrame()
    if csvp.exists():
        try:
            old = pd.read_csv(csvp)
        except EmptyDataError:
            old = pd.DataFrame()
            csvp.unlink(missing_ok=True)
    
    combined = pd.concat([old, new_df], ignore_index=True)
    
    if "transaction_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["transaction_id"], keep="last")
    
    combined.to_csv(csvp, index=False)
    return combined


def _to_date(x):
    """Convert various date formats to date object"""
    if x is None:
        return None
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    if isinstance(x, str):
        return datetime.strptime(x[:10], "%Y-%m-%d").date()
    raise TypeError(f"Unsupported date type: {type(x)}")


def get_transactions_cached(client, account_hash: str, start_date, end_date, types=None) -> pd.DataFrame:
    """
    Smart caching: fetches only missing range, returns full cached DataFrame.
    
    - On first run: fetches from start_date to end_date
    - On subsequent runs: only fetches new transactions since last cache
    """
    _ensure_cache_dir()
    
    req_start = _to_date(start_date)
    req_end = _to_date(end_date)
    
    if req_start is None:
        req_start = date(2015, 1, 1)
    if req_end is None:
        req_end = date.today()
    
    last_cached = load_last_cached_date(account_hash)
    fetch_start = req_start if last_cached is None else max(req_start, last_cached + timedelta(days=1))
    
    csvp = _csv_path(account_hash)
    
    # If cache already covers requested end date, just return cache
    if last_cached is not None and last_cached >= req_end and csvp.exists():
        logger.info("Cache hit for %s (covers through %s)", account_hash, last_cached)
        return pd.read_csv(csvp)
    
    # Fetch only the missing part
    if fetch_start <= req_end:
        logger.info("Fetching new transactions for %s from %s to %s", 
                   account_hash, fetch_start, req_end)
        txs = fetch_all_transactions(client, account_hash, fetch_start, req_end, types)
        new_df = normalize_txns_to_df(txs, account_hash)
        combined = append_cache(account_hash, new_df)
        
        # Update last_cached based on combined data
        if "date" in combined.columns:
            valid_dates = combined["date"].dropna().astype(str)
            if len(valid_dates) > 0:
                max_date = max(valid_dates)
                save_last_cached_date(account_hash, datetime.strptime(max_date, "%Y-%m-%d").date())
        return combined
    
    # Nothing to fetch; return existing cache if any
    if csvp.exists():
        return pd.read_csv(csvp)
    
    return pd.DataFrame(columns=["account_hash", "transaction_id", "date", "type", 
                                  "sub_type", "description", "amount", "raw_json"])


def get_transactions_cached_list(client, account_hash, start_date, end_date, types=None):
    """Returns list of transaction dicts using cached data"""
    df = get_transactions_cached(client, account_hash, start_date, end_date, types=types)
    
    if df is None or df.empty:
        return []
    
    if "raw_json" not in df.columns:
        return df.to_dict("records")
    
    out = []
    for s in df["raw_json"].dropna().tolist():
        try:
            out.append(json.loads(s))
        except Exception:
            continue
    return out


# =============================================================================
# PER-TICKER P&L CALCULATION
# =============================================================================

def parse_trade_transaction(tx: dict) -> dict | None:
    """
    Extract trade details from a TRADE transaction.
    
    Returns dict with:
        symbol, quantity, price, amount, action (BUY/SELL), date
    """
    if tx.get("type") != "TRADE":
        return None
    
    instrument = tx.get("transactionItem", {}).get("instrument", {})
    symbol = instrument.get("symbol")
    
    if not symbol:
        return None
    
    qty = tx.get("transactionItem", {}).get("amount", 0.0)
    price = tx.get("transactionItem", {}).get("price", 0.0)
    amount = tx.get("netAmount", 0.0)  # Net amount (includes fees)
    
    # Determine if BUY or SELL
    instruction = tx.get("transactionItem", {}).get("instruction", "")
    if "BUY" in instruction.upper():
        action = "BUY"
    elif "SELL" in instruction.upper():
        action = "SELL"
    else:
        action = instruction or "UNKNOWN"
    
    trade_date = tx.get("tradeDate") or tx.get("transactionDate")
    if isinstance(trade_date, str) and len(trade_date) >= 10:
        trade_date = trade_date[:10]
    
    return {
        "symbol": symbol,
        "quantity": abs(float(qty)),
        "price": float(price),
        "amount": float(amount),
        "action": action,
        "date": trade_date,
        "transaction_id": tx.get("transactionId"),
        "description": tx.get("description", ""),
    }


def parse_dividend_transaction(tx: dict) -> dict | None:
    """
    Extract dividend details.
    
    Returns dict with:
        symbol, amount, date
    """
    if tx.get("type") != "DIVIDEND_OR_INTEREST":
        return None
    
    instrument = tx.get("transactionItem", {}).get("instrument", {})
    symbol = instrument.get("symbol")
    
    if not symbol:
        return None
    
    amount = tx.get("netAmount", 0.0)
    
    div_date = tx.get("transactionDate") or tx.get("postDate")
    if isinstance(div_date, str) and len(div_date) >= 10:
        div_date = div_date[:10]
    
    return {
        "symbol": symbol,
        "amount": float(amount),
        "date": div_date,
        "transaction_id": tx.get("transactionId"),
        "description": tx.get("description", ""),
    }


class PositionTracker:
    """
    Track cost basis and realized/unrealized P&L for each ticker using FIFO.
    """
    
    def __init__(self):
        # symbol -> list of (qty, price, date) tuples (FIFO queue)
        self.lots = defaultdict(list)
        
        # symbol -> total realized P&L from sells
        self.realized_pl = defaultdict(float)
        
        # symbol -> total dividends received
        self.dividends = defaultdict(float)
        
        # Trade history for debugging
        self.trade_history = []
    
    def add_buy(self, symbol: str, qty: float, price: float, date_str: str):
        """Add a buy transaction (creates a new lot)"""
        self.lots[symbol].append({
            "qty": qty,
            "price": price,
            "date": date_str,
        })
        self.trade_history.append({
            "symbol": symbol,
            "action": "BUY",
            "qty": qty,
            "price": price,
            "date": date_str,
        })
    
    def add_sell(self, symbol: str, qty: float, price: float, date_str: str):
        """Add a sell transaction (removes from lots using FIFO, records realized P&L)"""
        remaining_qty = qty
        realized = 0.0
        
        while remaining_qty > 0 and self.lots[symbol]:
            lot = self.lots[symbol][0]
            
            if lot["qty"] <= remaining_qty:
                # Sell entire lot
                realized += (price - lot["price"]) * lot["qty"]
                remaining_qty -= lot["qty"]
                self.lots[symbol].pop(0)
            else:
                # Partial lot sale
                realized += (price - lot["price"]) * remaining_qty
                lot["qty"] -= remaining_qty
                remaining_qty = 0
        
        if remaining_qty > 0:
            # Sold more than we own (short sale or error) - treat as selling at current price
            logger.warning(f"Short sale detected for {symbol}: selling {remaining_qty} more than owned")
            realized += price * remaining_qty
        
        self.realized_pl[symbol] += realized
        self.trade_history.append({
            "symbol": symbol,
            "action": "SELL",
            "qty": qty,
            "price": price,
            "date": date_str,
            "realized_pl": realized,
        })
    
    def add_dividend(self, symbol: str, amount: float, date_str: str):
        """Record dividend income"""
        self.dividends[symbol] += amount
    
    def get_current_position(self, symbol: str) -> dict:
        """Get current position details for a symbol"""
        lots = self.lots.get(symbol, [])
        
        if not lots:
            return {
                "symbol": symbol,
                "qty": 0.0,
                "cost_basis": 0.0,
                "avg_price": 0.0,
                "realized_pl": self.realized_pl.get(symbol, 0.0),
                "dividends": self.dividends.get(symbol, 0.0),
            }
        
        total_qty = sum(lot["qty"] for lot in lots)
        total_cost = sum(lot["qty"] * lot["price"] for lot in lots)
        avg_price = total_cost / total_qty if total_qty > 0 else 0.0
        
        return {
            "symbol": symbol,
            "qty": total_qty,
            "cost_basis": total_cost,
            "avg_price": avg_price,
            "realized_pl": self.realized_pl.get(symbol, 0.0),
            "dividends": self.dividends.get(symbol, 0.0),
        }
    
    def get_all_positions(self) -> pd.DataFrame:
        """Get all positions as DataFrame"""
        all_symbols = set(self.lots.keys()) | set(self.realized_pl.keys()) | set(self.dividends.keys())
        
        rows = []
        for symbol in all_symbols:
            pos = self.get_current_position(symbol)
            rows.append(pos)
        
        return pd.DataFrame(rows)


def build_per_ticker_pl(client, account_hash: str, account_number: str, 
                        start_date, end_date=None) -> PositionTracker:
    """
    Build per-ticker P&L from cached transactions.
    
    Returns PositionTracker with all trades processed.
    """
    txs = get_transactions_cached_list(client, account_hash, start_date, end_date)
    
    tracker = PositionTracker()
    
    for tx in txs:
        # Process trades
        trade = parse_trade_transaction(tx)
        if trade:
            if trade["action"] == "BUY":
                tracker.add_buy(
                    trade["symbol"],
                    trade["quantity"],
                    trade["price"],
                    trade["date"]
                )
            elif trade["action"] == "SELL":
                tracker.add_sell(
                    trade["symbol"],
                    trade["quantity"],
                    trade["price"],
                    trade["date"]
                )
        
        # Process dividends
        dividend = parse_dividend_transaction(tx)
        if dividend:
            tracker.add_dividend(
                dividend["symbol"],
                dividend["amount"],
                dividend["date"]
            )
    
    return tracker


def calculate_comprehensive_pl(client, start_date="2022-01-01", end_date=None):
    """
    Calculate comprehensive P&L across all accounts and all tickers.
    
    Returns:
        - per_ticker_df: P&L for each ticker (symbol level)
        - per_account_ticker_df: P&L per account per ticker
        - summary_stats: Grand totals and aggregations
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if end_date is None:
        end_date = date.today()
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    
    # Get all accounts
    accounts = get_linked_accounts(client)
    
    # Get current positions (for unrealized P&L)
    positions_df, _ = get_positions_all_accounts(client)
    
    # Track per-account-ticker data
    all_account_ticker_rows = []
    
    # Process each account
    for acct in accounts:
        acc_num = acct["account_number"]
        acc_hash = acct["hash"]
        
        logger.info("Processing account %s...", acc_num)
        
        # Build position tracker from historical transactions
        tracker = build_per_ticker_pl(client, acc_hash, acc_num, start_date, end_date)
        
        # Get historical positions
        hist_positions = tracker.get_all_positions()
        
        # Get current positions for this account
        acct_current = positions_df[positions_df["account_number"] == acc_num].copy()
        
        if not acct_current.empty:
            acct_current = acct_current.groupby("symbol", as_index=False).agg({
                "quantity": "sum",
                "market_value": "sum",
            })
        
        # Merge historical with current
        if not hist_positions.empty:
            for _, row in hist_positions.iterrows():
                symbol = row["symbol"]
                
                # Get current market value
                current = acct_current[acct_current["symbol"] == symbol]
                market_value = current["market_value"].iloc[0] if not current.empty else 0.0
                current_qty = current["quantity"].iloc[0] if not current.empty else 0.0
                
                # Calculate unrealized P&L
                unrealized_pl = market_value - row["cost_basis"]
                
                # Total P&L = realized + unrealized + dividends
                total_pl = row["realized_pl"] + unrealized_pl + row["dividends"]
                
                all_account_ticker_rows.append({
                    "account_number": acc_num,
                    "symbol": symbol,
                    "qty_held": row["qty"],
                    "qty_current": current_qty,
                    "cost_basis": row["cost_basis"],
                    "avg_price": row["avg_price"],
                    "market_value": market_value,
                    "unrealized_pl": unrealized_pl,
                    "realized_pl": row["realized_pl"],
                    "dividends": row["dividends"],
                    "total_pl": total_pl,
                })
    
    # Create DataFrames
    per_account_ticker_df = pd.DataFrame(all_account_ticker_rows)
    
    if per_account_ticker_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    
    # Aggregate by ticker across all accounts
    per_ticker_df = per_account_ticker_df.groupby("symbol", as_index=False).agg({
        "qty_held": "sum",
        "qty_current": "sum",
        "cost_basis": "sum",
        "market_value": "sum",
        "unrealized_pl": "sum",
        "realized_pl": "sum",
        "dividends": "sum",
        "total_pl": "sum",
    })
    
    # Recalculate avg_price for aggregated view
    per_ticker_df["avg_price"] = per_ticker_df["cost_basis"] / per_ticker_df["qty_held"].replace(0, pd.NA)
    
    # Calculate total P&L %
    per_ticker_df["total_pl_pct"] = (
        per_ticker_df["total_pl"] / per_ticker_df["cost_basis"].replace(0, pd.NA) * 100.0
    )
    
    # Summary statistics
    summary_stats = {
        "total_cost_basis": per_ticker_df["cost_basis"].sum(),
        "total_market_value": per_ticker_df["market_value"].sum(),
        "total_unrealized_pl": per_ticker_df["unrealized_pl"].sum(),
        "total_realized_pl": per_ticker_df["realized_pl"].sum(),
        "total_dividends": per_ticker_df["dividends"].sum(),
        "grand_total_pl": per_ticker_df["total_pl"].sum(),
    }
    
    # Calculate overall return %
    if summary_stats["total_cost_basis"] > 0:
        summary_stats["grand_total_pl_pct"] = (
            summary_stats["grand_total_pl"] / summary_stats["total_cost_basis"] * 100.0
        )
    else:
        summary_stats["grand_total_pl_pct"] = 0.0
    
    # Sort by total P&L descending
    per_ticker_df = per_ticker_df.sort_values("total_pl", ascending=False)
    per_account_ticker_df = per_account_ticker_df.sort_values(["account_number", "total_pl"], ascending=[True, False])
    
    return per_ticker_df, per_account_ticker_df, summary_stats