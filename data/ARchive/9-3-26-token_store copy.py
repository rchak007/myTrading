from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import streamlit as st
from supabase import create_client


def _sb():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


def load_tokens_db(user_id: str = "main") -> Optional[Dict[str, Any]]:
    sb = _sb()
    res = (
        sb.table("schwab_tokens")
        .select("access_token,refresh_token,expires_at")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if not res.data:
        return None

    row = res.data[0]
    dt = datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00"))
    return {
        "access_token": row["access_token"],
        "refresh_token": row["refresh_token"],
        "expires_at": int(dt.timestamp()),
    }

def delete_tokens_db(user_id: str = "main") -> None:
    sb = _sb()
    sb.table("schwab_tokens").delete().eq("user_id", user_id).execute()


def save_tokens_db(tokens: Dict[str, Any], user_id: str = "main") -> None:
    exp = tokens["expires_at"]
    if isinstance(exp, (int, float)):
        expires_iso = datetime.fromtimestamp(exp, tz=timezone.utc).isoformat()
    elif isinstance(exp, datetime):
        expires_iso = exp.astimezone(timezone.utc).isoformat()
    else:
        raise ValueError("expires_at must be epoch seconds or datetime")

    payload = {
        "user_id": user_id,
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "expires_at": expires_iso,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    sb = _sb()
    sb.table("schwab_tokens").upsert(payload).execute()
