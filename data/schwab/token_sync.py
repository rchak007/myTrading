from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any, Iterable
from datetime import datetime, timezone, timedelta

from data.schwab.token_store import load_tokens_db, save_tokens_db


def _read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_issued_time(s: str) -> Optional[datetime]:
    """
    Accepts formats like '2026-01-08T01:43:20...' from your screenshot.
    Handles possible trailing 'Z' too.
    """
    try:
        s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _extract_tokens_from_obj(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Supports your schwabdev token file structure:
      {
        "access_token_issued": "...",
        "refresh_token_issued": "...",
        "token_dictionary": {
            "expires_in": 1800,
            "refresh_token": "...",
            "access_token": "...",
            ...
        }
      }
    Also supports already-flat dicts with access_token/refresh_token/expires_at.
    Returns a normalized dict:
      {"access_token": str, "refresh_token": str, "expires_at": epoch_seconds}
    """
    if not isinstance(obj, dict):
        return None

    # Case 1: already normalized
    if all(k in obj for k in ("access_token", "refresh_token", "expires_at")):
        return obj

    # Case 2: your structure with token_dictionary
    td = obj.get("token_dictionary")
    if isinstance(td, dict) and "access_token" in td and "refresh_token" in td:
        access_token = td["access_token"]
        refresh_token = td["refresh_token"]

        expires_in = td.get("expires_in")
        try:
            expires_in = int(expires_in) if expires_in is not None else None
        except Exception:
            expires_in = None

        issued = None
        if "access_token_issued" in obj:
            issued = _parse_issued_time(str(obj["access_token_issued"]))

        if issued is None:
            issued = datetime.now(timezone.utc)

        if expires_in is None:
            # conservative fallback: 30 minutes
            expires_in = 1800

        expires_at = int((issued + timedelta(seconds=expires_in)).timestamp())

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": expires_at,
        }

    # Otherwise: try searching nested values
    for v in obj.values():
        found = _extract_tokens_from_obj(v) if isinstance(v, dict) else None
        if found:
            return found

    return None


def sync_db_to_local_multi(user_id: str, token_paths: Iterable[Path]) -> bool:
    """
    DB -> write tokens to local paths in schwabdev-style wrapper format.
    """
    tokens = load_tokens_db(user_id)
    if not tokens:
        return False

    # Convert expires_at (epoch) into issued + expires_in
    expires_at = int(tokens["expires_at"])
    now = datetime.now(timezone.utc)
    expires_dt = datetime.fromtimestamp(expires_at, tz=timezone.utc)

    # If token already expired, still write it (schwabdev may refresh using refresh_token)
    expires_in = int(max((expires_dt - now).total_seconds(), 0))

    wrapper = {
        # Use "now" as issued time. Good enough; expires_in drives validity.
        "access_token_issued": now.isoformat(),
        "refresh_token_issued": now.isoformat(),
        "token_dictionary": {
            "expires_in": expires_in,
            "token_type": "Bearer",
            "scope": "api",  # harmless default
            "refresh_token": tokens["refresh_token"],
            "access_token": tokens["access_token"],
        },
    }

    for p in token_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(wrapper, indent=2), encoding="utf-8")

    return True



def sync_local_to_db_multi(user_id: str, token_paths: Iterable[Path]) -> bool:
    """
    Local -> DB: read whichever local token file exists and extract tokens.
    """
    for p in token_paths:
        obj = _read_json(p)
        if obj is None:
            continue

        tok = _extract_tokens_from_obj(obj)
        if tok and all(k in tok for k in ("access_token", "refresh_token", "expires_at")):
            save_tokens_db(tok, user_id)
            return True

    return False
