import logging
from datetime import date, datetime, timedelta
from typing import List, Dict

import pandas as pd

from portfolio_snapshot import make_client, get_account_balances

import os
import json
from pathlib import Path


import time
from requests.exceptions import ReadTimeout, RequestException

from requests.exceptions import HTTPError


CACHE_DIR = Path("data/transactions")



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# --- Helper: linked accounts --------------------------------------------------


def get_linked_accounts(client) -> List[Dict]:
    """
    Return a list of linked accounts with:
      - accountNumber
      - hashValue
      - displayName (if present)
    """
    resp = client.account_linked()
    data = resp.json()

    accounts = []
    for a in data:
        accounts.append(
            {
                "account_number": a.get("accountNumber"),
                "hash": a.get("hashValue"),
                "display_name": a.get("displayName") or a.get("accountNumber"),
            }
        )

    logger.info("Found %d linked accounts", len(accounts))
    return accounts


# --- Low-level transaction fetch ---------------------------------------------



def _fetch_transactions_chunk(client, account_hash, start_date, end_date, types=None):
    """
    Fetch a chunk of transactions for a single account using Schwabdev client's transactions().

    Returns: list[dict]
    Retries on timeouts / transient request errors with exponential backoff.
    """
    try:
        from urllib3.exceptions import ReadTimeoutError
        timeout_exceptions = (ReadTimeout, ReadTimeoutError)
    except Exception:
        timeout_exceptions = (ReadTimeout,)

    # start_str = start_date.strftime("%Y-%m-%d")
    # end_str = end_date.strftime("%Y-%m-%d")
    start_str = start_date.strftime("%Y-%m-%dT00:00:00.000Z")
    end_str   = end_date.strftime("%Y-%m-%dT23:59:59.999Z")


    if types is None:
        # types_list = [
        #     "TRADE",
        #     "DIVIDEND_OR_INTEREST",
        #     "ACH_RECEIPT",
        #     "ACH_DISBURSEMENT",
        #     "CASH_RECEIPT",
        #     "CASH_DISBURSEMENT",
        #     "ELECTRONIC_FUND",
        #     "WIRE_IN",
        #     "WIRE_OUT",
        #     "JOURNAL",
        #     "MEMORANDUM",
        # ]
        types_list = [
            # ===== Trading & Investment =====
            "TRADE",
            "DIVIDEND_OR_INTEREST",
            "CORPORATE_ACTION",

            # ===== Security movement =====
            "SECURITY_TRANSFER",

            # ===== Cash movement =====
            "ACH_RECEIPT",
            "ACH_DISBURSEMENT",
            "CASH_RECEIPT",
            "CASH_DISBURSEMENT",
            "ELECTRONIC_FUND",
            "WIRE_IN",
            "WIRE_OUT",

            # ===== Fees / adjustments =====
            "FEE",
            "TAX",
            "ADJUSTMENT",
            "JOURNAL",

            # ===== Catch-all =====
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
                    "HTTP %s for hash=%s %s..%s types=%s body=%s",
                    resp.status_code,
                    account_hash,
                    start_str,
                    end_str,
                    types_list,
                    getattr(resp, "text", ""),
                )
                return []

            # optional
            # resp.raise_for_status()

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
            body = getattr(getattr(e, "response", None), "text", "")

            if status == 400:
                logger.error(
                    "HTTP 400 (Bad Request) for hash=%s %s..%s types=%s body=%s",
                    account_hash, start_str, end_str, types_list, body
                )
                return []

            last_err = e
            logger.warning(
                "HTTP error fetching tx (attempt %d/%d) hash=%s %s..%s : %s",
                attempt, max_tries, account_hash, start_str, end_str, repr(e)
            )

        except RequestException as e:
            last_err = e
            logger.warning(
                "Request error fetching tx (attempt %d/%d) hash=%s %s..%s : %s",
                attempt, max_tries, account_hash, start_str, end_str, repr(e)
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
    """
    Walk forward in chunks from start_date until end_date (or today)
    and return a flat list of transactions.

    Adaptive chunk sizing:
      - Start 60 days
      - On timeout: shrink (60 -> 30 -> 15 -> 7)
      - Retries same window until success or min window fails
    """
    if end_date is None:
        end_date = date.today()

    if start_date > end_date:
        return []

    # Catch both requests + urllib3 timeout types
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

        # success
        if chunk:
            all_tx.extend(chunk)

        logger.info(
            "Fetched %d tx for hash=%s from %s to %s",
            len(chunk) if chunk else 0,
            account_hash,
            cur_start.strftime("%Y-%m-%d"),
            cur_end.strftime("%Y-%m-%d"),
        )

        # light throttle helps avoid API/network flakiness when pulling years of history
        time.sleep(0.15)

        cur_start = cur_end + timedelta(days=1)

    return all_tx



# --- Deposit / withdrawal analytics ------------------------------------------

# You can tweak these lists as we refine things
DEPOSIT_TYPES = {
    "ACH_RECEIPT",
    "CASH_RECEIPT",
    "ELECTRONIC_FUND",
    "WIRE_IN",
    "JOURNAL",  # can be internal transfers; we may need to refine later
}

WITHDRAWAL_TYPES = {
    "ACH_DISBURSEMENT",
    "CASH_DISBURSEMENT",
    "ELECTRONIC_FUND",
    "WIRE_OUT",
    "JOURNAL",  # internal transfers again
}


def build_deposit_withdrawal_df(
    client,
    start_date: date = date(2015, 1, 1),
    end_date: date | None = None,
) -> pd.DataFrame:
    """
    Build a DataFrame with all deposits / withdrawals for every linked account.

    Columns:
      - account_number
      - transaction_id
      - transaction_date
      - type
      - description
      - amount
      - signed_amount  (positive = deposit into account, negative = withdrawal)
    """
    accounts = get_linked_accounts(client)
    all_rows: List[Dict] = []

    if end_date is None:
        end_date = date.today()

    for acc in accounts:
        acc_num = acc["account_number"]
        acc_hash = acc["hash"]

        # txs = fetch_all_transactions(
        #     client, acc_hash, start_date=start_date, end_date=end_date, types=None
        # )
        # txs = get_transactions_cached_list(client, acc_hash, start_date=start_date, end_date=end_date, types=None)

        cash_types = sorted(DEPOSIT_TYPES.union(WITHDRAWAL_TYPES))
        txs = get_transactions_cached_list(
            client,
            acc_hash,
            start_date=start_date,
            end_date=end_date,
            types=cash_types,
        )


        # df_all = get_transactions_cached(client, acc_hash, start_date, end_date, types=None)  

        for t in txs:
            t_type = t.get("type")
            if t_type not in DEPOSIT_TYPES and t_type not in WITHDRAWAL_TYPES:
                continue

            # Schwab typically has netAmount for cash, fall back to amount if needed
            amt = t.get("netAmount") or t.get("amount")
            if amt is None:
                continue

            amt = float(amt)

            # Sign deposit/withdrawal
            if t_type in DEPOSIT_TYPES:
                signed = amt
            else:
                signed = -amt

            row = {
                "account_number": acc_num,
                "transaction_id": t.get("transactionId"),
                "transaction_date": t.get("transactionDate"),
                "type": t_type,
                "description": t.get("description"),
                "amount": amt,
                "signed_amount": signed,
            }
            all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"]).dt.date
    return df


def summarize_contributions_vs_equity(
    client,
    start_date: date = date(2015, 1, 1),
    end_date: date | None = None,
) -> pd.DataFrame:
    """
    High-level view:

    For each account:
      - net_contribution: sum of signed_amount (deposits - withdrawals)
      - equity_now: current equity (from account balances)
      - pnl_total: equity_now - net_contribution
      - pnl_pct: pnl_total / net_contribution (if net_contribution > 0)
    """
    if end_date is None:
        end_date = date.today()

    dw_df = build_deposit_withdrawal_df(client, start_date=start_date, end_date=end_date)
    if dw_df.empty:
        logger.warning("No deposits/withdrawals found in the given range.")
        return pd.DataFrame()

    contrib = (
        dw_df.groupby("account_number", as_index=False)
        .agg(net_contribution=("signed_amount", "sum"))
    )

    balances = get_account_balances(client)

    merged = balances.merge(contrib, on="account_number", how="left")
    merged["net_contribution"] = merged["net_contribution"].fillna(0.0)

    merged["pnl_total"] = merged["equity"] - merged["net_contribution"]

    # Avoid divide by zero
    merged["pnl_pct"] = None
    nonzero = merged["net_contribution"] != 0
    merged.loc[nonzero, "pnl_pct"] = (
        merged.loc[nonzero, "pnl_total"] / merged.loc[nonzero, "net_contribution"] * 100.0
    )

    return merged




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

def normalize_txns_to_df(txs: list, account_hash: str) -> pd.DataFrame:
    # keep raw json but pull useful fields too
    rows = []
    for t in txs or []:
        trade_date = t.get("tradeDate") or t.get("transactionDate") or t.get("postDate")
        # trade_date in API often includes time; keep first 10 chars if string
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
    df = pd.DataFrame(rows)
    return df

from pandas.errors import EmptyDataError

def append_cache(account_hash: str, new_df: pd.DataFrame) -> pd.DataFrame:
    # csvp = _cache_path(account_hash)
    csvp = _state_path(account_hash)


    # If API returned nothing, don't create/append to cache at all
    if new_df is None or new_df.empty:
        if csvp.exists() and csvp.stat().st_size == 0:
            # clean up bad/empty cache file
            csvp.unlink(missing_ok=True)
        # return existing cache if present, else empty df
        if csvp.exists():
            try:
                return pd.read_csv(csvp)
            except EmptyDataError:
                csvp.unlink(missing_ok=True)
        return pd.DataFrame()

    # Read old cache safely
    old = pd.DataFrame()
    if csvp.exists():
        try:
            old = pd.read_csv(csvp)
        except EmptyDataError:
            # empty file -> treat as no cache
            old = pd.DataFrame()
            csvp.unlink(missing_ok=True)

    # Combine + de-dupe (adjust keys as appropriate)
    combined = pd.concat([old, new_df], ignore_index=True)

    # (Optional) if you have a unique transaction id column:
    if "transactionId" in combined.columns:
        combined = combined.drop_duplicates(subset=["transactionId"])

    combined.to_csv(csvp, index=False)
    return combined



def _to_date(x):
    # Accepts datetime.date, datetime.datetime, or "YYYY-MM-DD"
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
    start_date/end_date: datetime.date, datetime.datetime, or 'YYYY-MM-DD'
    Fetches only missing range and caches to CSV.
    Returns the full cached DF for that account.
    """
    _ensure_cache_dir()

    req_start = _to_date(start_date)
    req_end   = _to_date(end_date)

    if req_start is None:
        req_start = date(2015, 1, 1)
    if req_end is None:
        req_end = date.today()

    last_cached = load_last_cached_date(account_hash)
    fetch_start = req_start if last_cached is None else max(req_start, last_cached + timedelta(days=1))

    csvp = _csv_path(account_hash)

    # If cache already covers requested end date, just return cache
    if last_cached is not None and last_cached >= req_end and csvp.exists():
        return pd.read_csv(csvp)

    # Fetch only the missing part
    if fetch_start <= req_end:
        txs = fetch_all_transactions(client, account_hash, fetch_start, req_end, types)
        new_df = normalize_txns_to_df(txs, account_hash)
        combined = append_cache(account_hash, new_df)

        # update last_cached based on combined data
        if "date" in combined.columns:
            valid_dates = combined["date"].dropna().astype(str)
            if len(valid_dates) > 0:
                max_date = max(valid_dates)
                save_last_cached_date(account_hash, datetime.strptime(max_date, "%Y-%m-%d").date())
        return combined

    # Nothing to fetch; return existing cache if any
    if csvp.exists():
        return pd.read_csv(csvp)

    return pd.DataFrame(columns=["account_hash","transaction_id","date","type","sub_type","description","amount","raw_json"])



def get_transactions_cached_list(client, account_hash, start_date, end_date, types=None):
    """
    Returns list-of-dict transactions (txs) but uses the on-disk cache under the hood.

    start_date/end_date can be datetime.date, datetime.datetime, or YYYY-MM-DD string.
    """
    df = get_transactions_cached(client, account_hash, start_date, end_date, types=types)

    if df is None or df.empty:
        return []

    if "raw_json" not in df.columns:
        # fallback: try to rebuild dicts from df columns (best-effort)
        return df.to_dict("records")

    out = []
    for s in df["raw_json"].dropna().tolist():
        try:
            out.append(json.loads(s))
        except Exception:
            # ignore malformed json rows
            continue
    return out
