# core/registry.py
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict

from core.utils import get_secret


def load_asset_registry() -> Dict[str, Dict[str, Any]]:
    """
    Priority:
      1) ASSET_REGISTRY_FILE (json file path) from secrets/env
      2) ASSET_REGISTRY (inline json / python-literal) from secrets/env

    Returns a dict of dicts.
    """
    # 1) file-based
    file_path = (get_secret("ASSET_REGISTRY_FILE") or "").strip()
    if file_path:
        p = Path(file_path)
        if p.exists() and p.is_file():
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    return _normalize_registry(obj)
            except Exception:
                return {}

    # 2) inline
    raw = (get_secret("ASSET_REGISTRY") or "").strip()
    if not raw:
        return {}

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return _normalize_registry(obj)
    except Exception:
        pass

    try:
        obj = ast.literal_eval(raw)
        if isinstance(obj, dict):
            return _normalize_registry(obj)
    except Exception:
        return {}

    return {}


def _normalize_registry(reg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in reg.items():
        if isinstance(v, dict):
            out[str(k)] = v
    return out


def find_registry_entry_by_yahoo_ticker(registry: Dict[str, Dict[str, Any]], yahoo_ticker: str):
    yt = (yahoo_ticker or "").strip().lower()
    if not yt:
        return None
    for _, entry in registry.items():
        if str(entry.get("yahoo_ticker", "")).strip().lower() == yt:
            return entry
    return None
