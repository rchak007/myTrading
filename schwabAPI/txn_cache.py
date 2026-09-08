"""
txn_cache.py
------------
Pulls the COMPLETE Schwab transaction history for every linked account and
caches it to  data/transactions/<account_hash>.csv .

Fixes vs the old analytics_core cache:
  1. Schwab returns "activityId", not "transactionId".  The old normalizer
     wrote t.get("transactionId") -> every id was NaN, so de-dupe never ran.
  2. Schwab returns "netAmount", not "amount".  The old amount column was NaN.
  3. The old cache was type-scoped but the state file was not, so a run that
     fetched only cash types would mark the range "cached" and permanently
     hide TRADE rows.  We now ALWAYS fetch every type into one cache and let
     downstream code filter.
  4. Incremental runs re-fetch an OVERLAP_DAYS window (Schwab back-dates
     corrections) and de-dupe on activityId.

Cache layout per account:
    <hash>.csv          full raw dump, one row per transaction
    <hash>.state.json   {"last_date": "YYYY-MM-DD", "count": N}
"""

import json
import logging
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from pandas.errors import EmptyDataError
from requests.exceptions import ReadTimeout, RequestException

from schwab_client import CACHE_DIR, COLD_START, OVERLAP_DAYS

logger = logging.getLogger(__name__)

# Types the CURRENT Schwab API accepts, measured with probe_schwab_api.py.
# Re-run that probe if these ever start 400ing again.
#
# Dropped as invalid enum values by the current API:
#     CORPORATE_ACTION, SECURITY_TRANSFER, FEE, TAX, ADJUSTMENT
#
# That is less of a loss than it looks. Schwab now books splits, mergers and
# ACATS transfers under RECEIVE_AND_DELIVER, and per-trade fees ride inside a
# TRADE's transferItems (txn_parser._fees_of reads them there), not as
# standalone FEE rows. Still: after a cold start, confirm splits and transfers
# actually appear before trusting the P&L -- see the type histogram in the
# run log.
ALL_TYPES = [
    "TRADE",
    "DIVIDEND_OR_INTEREST",
    "RECEIVE_AND_DELIVER",
    "ACH_RECEIPT",
    "ACH_DISBURSEMENT",
    "CASH_RECEIPT",
    "CASH_DISBURSEMENT",
    "ELECTRONIC_FUND",
    "WIRE_IN",
    "WIRE_OUT",
    "JOURNAL",
    "MEMORANDUM",
]

# The endpoint's signature is `types: str`. Passing the list url-encodes into
# something Schwab rejects, so join it once here and reuse.
TYPES_PARAM = ",".join(ALL_TYPES)

CSV_COLS = [
    "account_hash",
    "activity_id",
    "date",
    "type",
    "sub_type",
    "description",
    "net_amount",
    "raw_json",
]

try:
    from urllib3.exceptions import ReadTimeoutError

    TIMEOUTS = (ReadTimeout, ReadTimeoutError)
except Exception:  # pragma: no cover
    TIMEOUTS = (ReadTimeout,)


# ---------------------------------------------------------------- accounts


def get_linked_accounts(client) -> List[Dict]:
    # renamed from account_linked() in current schwabdev
    resp = client.linked_accounts()
    accounts = []
    for a in resp.json():
        if not isinstance(a, dict):
            logger.warning("Unexpected account payload: %r", a)
            continue
        accounts.append(
            {
                "account_number": a.get("accountNumber"),
                "hash": a.get("hashValue"),
                "display_name": a.get("displayName") or a.get("accountNumber"),
            }
        )
    logger.info("Linked accounts: %d", len(accounts))
    return accounts


# ---------------------------------------------------------------- fetching


def _fetch_chunk(client, account_hash: str, start: date, end: date) -> List[Dict]:
    start_str = start.strftime("%Y-%m-%dT00:00:00.000Z")
    end_str = end.strftime("%Y-%m-%dT23:59:59.999Z")

    max_tries, backoff, last_err = 6, 2, None
    for attempt in range(1, max_tries + 1):
        try:
            resp = client.transactions(account_hash, start_str, end_str, TYPES_PARAM)
            if resp.status_code != 200:
                logger.error(
                    "HTTP %s hash=%s %s..%s body=%s",
                    resp.status_code,
                    account_hash[:8],
                    start,
                    end,
                    getattr(resp, "text", "")[:300],
                )
                return []
            data = resp.json() or []
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for k in ("transactions", "items"):
                    if isinstance(data.get(k), list):
                        return data[k]
            return []
        except TIMEOUTS as e:
            last_err = e
            logger.warning("timeout %d/%d %s %s..%s", attempt, max_tries, account_hash[:8], start, end)
        except RequestException as e:
            last_err = e
            logger.warning("request error %d/%d: %r", attempt, max_tries, e)

        if attempt < max_tries:
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)

    raise last_err  # type: ignore[misc]


def fetch_range(client, account_hash: str, start: date, end: date, chunk_days: int = 365) -> List[Dict]:
    """
    Walk start->end in chunks (default 1 year, the Schwab max).
    Shrinks the window on timeout instead of giving up.
    """
    out: List[Dict] = []
    cur = start
    while cur <= end:
        span = chunk_days
        while True:
            cur_end = min(cur + timedelta(days=span - 1), end)
            try:
                chunk = _fetch_chunk(client, account_hash, cur, cur_end)
                break
            except TIMEOUTS:
                if span <= 15:
                    raise
                span = max(15, span // 2)
                logger.warning("shrinking chunk to %d days for %s", span, account_hash[:8])

        logger.info("  %s  %s..%s -> %d txns", account_hash[:8], cur, cur_end, len(chunk))
        out.extend(chunk)
        time.sleep(0.15)
        cur = cur_end + timedelta(days=1)
    return out


# ---------------------------------------------------------------- cache io


def _csv_path(h: str):
    return CACHE_DIR / f"{h}.csv"


def _state_path(h: str):
    return CACHE_DIR / f"{h}.state.json"


def _load_state(h: str) -> Optional[date]:
    p = _state_path(h)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text()).get("last_date")
        return datetime.strptime(d, "%Y-%m-%d").date() if d else None
    except Exception:
        return None


def _save_state(h: str, last: date, count: int) -> None:
    _state_path(h).write_text(
        json.dumps({"last_date": last.strftime("%Y-%m-%d"), "count": count}, indent=2)
    )


def normalize(txns: List[Dict], account_hash: str) -> pd.DataFrame:
    rows = []
    for t in txns or []:
        raw_dt = t.get("tradeDate") or t.get("time") or t.get("settlementDate")
        d = raw_dt[:10] if isinstance(raw_dt, str) and len(raw_dt) >= 10 else None
        rows.append(
            {
                "account_hash": account_hash,
                # <-- the fix: Schwab calls it activityId
                "activity_id": t.get("activityId") or t.get("transactionId") or t.get("id"),
                "date": d,
                "type": t.get("type"),
                "sub_type": t.get("subType"),
                "description": t.get("description") or t.get("memo"),
                # <-- the fix: Schwab calls it netAmount
                "net_amount": t.get("netAmount"),
                "raw_json": json.dumps(t, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows, columns=CSV_COLS)


def _read_cache(h: str) -> pd.DataFrame:
    p = _csv_path(h)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=CSV_COLS)
    try:
        df = pd.read_csv(p)
    except EmptyDataError:
        return pd.DataFrame(columns=CSV_COLS)

    # Tolerate the OLD schema (transaction_id / amount) by re-deriving from raw_json.
    if "activity_id" not in df.columns and "raw_json" in df.columns:
        logger.info("migrating legacy cache schema for %s", h[:8])
        df = normalize([json.loads(s) for s in df["raw_json"].dropna()], h)
    return df


def merge_and_save(h: str, new_df: pd.DataFrame) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    old = _read_cache(h)
    combined = pd.concat([old, new_df], ignore_index=True) if not new_df.empty else old
    if combined.empty:
        return combined

    before = len(combined)
    # de-dupe on activityId; keep the LAST copy (Schwab corrections win)
    combined = combined.drop_duplicates(subset=["activity_id"], keep="last")
    combined = combined.sort_values("date", na_position="first").reset_index(drop=True)
    if before != len(combined):
        logger.info("  de-duped %d -> %d rows", before, len(combined))

    combined.to_csv(_csv_path(h), index=False)
    return combined


def sync_account(client, account_hash: str, cold_start: str = COLD_START) -> pd.DataFrame:
    """
    Bring one account's cache fully up to date.
    Cold start: pulls everything from `cold_start` to today.
    Warm start: re-pulls the last OVERLAP_DAYS and merges.
    """
    today = date.today()
    last = _load_state(account_hash)

    if last is None:
        start = datetime.strptime(cold_start, "%Y-%m-%d").date()
        logger.info("COLD START %s from %s", account_hash[:8], start)
    else:
        start = last - timedelta(days=OVERLAP_DAYS)
        logger.info("incremental %s from %s (last=%s, overlap=%dd)",
                    account_hash[:8], start, last, OVERLAP_DAYS)

    txns = fetch_range(client, account_hash, start, today)
    combined = merge_and_save(account_hash, normalize(txns, account_hash))

    if not combined.empty and combined["date"].notna().any():
        max_date = datetime.strptime(str(combined["date"].dropna().max()), "%Y-%m-%d").date()
        _save_state(account_hash, max_date, len(combined))
        logger.info("  %s cached rows=%d last_date=%s", account_hash[:8], len(combined), max_date)
    return combined


def sync_all(client) -> pd.DataFrame:
    """Sync every linked account; return the concatenated ledger with account_number."""
    frames = []
    for acc in get_linked_accounts(client):
        df = sync_account(client, acc["hash"])
        if df.empty:
            continue
        df = df.copy()
        df["account_number"] = acc["account_number"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CSV_COLS)


def load_all_cached() -> pd.DataFrame:
    """Read every cached account CSV off disk without touching the API."""
    frames = []
    if not CACHE_DIR.exists():
        return pd.DataFrame(columns=CSV_COLS)
    for p in sorted(CACHE_DIR.glob("*.csv")):
        df = _read_cache(p.stem)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CSV_COLS)