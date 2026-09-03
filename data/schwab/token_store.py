"""Local-only stub. Supabase token sync disabled; tokens.json on disk is authoritative."""
from typing import Any, Dict, Optional


def load_tokens_db(user_id: str = "main") -> Optional[Dict[str, Any]]:
    return None


def delete_tokens_db(user_id: str = "main") -> None:
    return None


def save_tokens_db(tokens: Dict[str, Any], user_id: str = "main") -> None:
    return None