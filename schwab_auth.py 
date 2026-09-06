#!/usr/bin/env python3
# =====================================================================
# schwab_auth.py — headless Schwab re-authorization for Pi 1
# ---------------------------------------------------------------------
# WHY THIS EXISTS
#   schwabdev's login flow prints a URL and then blocks on input(),
#   waiting for you to paste the redirect URL back into the SAME
#   process. That coupling is a library convenience, not a protocol
#   requirement. Schwab's authorization-code exchange is a stateless
#   POST: no PKCE verifier, no server-side session to carry. So the URL
#   can be generated on the Pi, opened on your phone, and the resulting
#   code handed back minutes later through any channel you like.
#
# THE 7-DAY CLOCK
#   The refresh token dies exactly 7 days after ISSUANCE. Exchanging it
#   for access tokens does not roll the window forward. But a fresh
#   authorization-code exchange mints a brand new refresh token with a
#   full 7 days starting at that moment. So re-authorizing on day 6
#   gives you 7 more days from day 6 — you are never forced to wait for
#   the failure. Pick a fixed slot (Sunday morning) and stay ahead of it.
#
# SAFETY
#   The live tokens.json is never overwritten until the NEW credentials
#   have been proven against a real Schwab endpoint. Failure leaves the
#   working file untouched. The previous file is kept as tokens.json.bak.
# =====================================================================
from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import requests

AUTH_BASE = "https://api.schwabapi.com/v1/oauth/authorize"
TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
VALIDATE_URL = "https://api.schwabapi.com/trader/v1/accounts/accountNumbers"

REFRESH_TTL = timedelta(days=7)          # hard, enforced by Schwab
RENEW_AT = timedelta(days=6)             # our own trigger, one day of slack

TOKEN_PATH = Path(os.getenv("SCHWAB_TOKENS",
                            Path.home() / "github" / "myTrading" / "tokens.json"))

# Keys that hold secrets. Used to scrub anything before it reaches a
# log line or a spreadsheet cell.
SECRET_KEYS = {"access_token", "refresh_token", "id_token", "code"}


# ---------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------
def creds() -> tuple[str, str, str]:
    key = os.getenv("app_key") or os.getenv("APP_KEY", "")
    secret = os.getenv("app_secret") or os.getenv("APP_SECRET", "")
    cb = os.getenv("callback_url") or os.getenv("CALLBACK_URL", "")
    missing = [n for n, v in (("app_key", key), ("app_secret", secret),
                              ("callback_url", cb)) if not v]
    if missing:
        raise RuntimeError(f"missing in .env: {', '.join(missing)}")
    return key, secret, cb


def authorize_url() -> str:
    key, _, cb = creds()
    return f"{AUTH_BASE}?client_id={key}&redirect_uri={cb}"


# ---------------------------------------------------------------------
# Token file: read, inspect, write
# ---------------------------------------------------------------------
def load_tokens() -> dict | None:
    if not TOKEN_PATH.exists():
        return None
    try:
        return json.loads(TOKEN_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _walk_set(node, updates: dict, stamped: str) -> None:
    """
    Substitute values in place wherever their key appears, at any depth.

    schwabdev's file shape has drifted between versions (top-level keys
    vs. a nested token_dictionary). Rather than hardcode one layout and
    silently produce a file the library cannot read, we preserve whatever
    structure is already on disk and only swap the leaves.
    """
    if isinstance(node, dict):
        for k, v in node.items():
            if k in updates and not isinstance(v, (dict, list)):
                node[k] = updates[k]
            elif "issued" in k.lower() and not isinstance(v, (dict, list)):
                node[k] = stamped
            else:
                _walk_set(v, updates, stamped)
    elif isinstance(node, list):
        for item in node:
            _walk_set(item, updates, stamped)


def build_token_file(payload: dict, previous: dict | None) -> dict:
    """New token file, shaped like the old one when there is an old one."""
    stamped = datetime.now().astimezone().isoformat()
    updates = {
        "access_token": payload["access_token"],
        "refresh_token": payload["refresh_token"],
        "id_token": payload.get("id_token", ""),
        "expires_in": payload.get("expires_in", 1800),
        "token_type": payload.get("token_type", "Bearer"),
        "scope": payload.get("scope", "api"),
    }

    if previous:
        out = copy.deepcopy(previous)
        _walk_set(out, updates, stamped)
        return out

    # No prior file to imitate — fall back to the schwabdev v2 layout.
    return {
        "access_token_issued": stamped,
        "refresh_token_issued": stamped,
        "token_dictionary": {k: v for k, v in updates.items()},
    }


def refresh_issued_at(tokens: dict | None) -> datetime | None:
    """First value under any key that looks like the refresh issue stamp."""
    if not tokens:
        return None

    found: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, str) and "refresh" in k.lower() and "issued" in k.lower():
                    found.append(v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for i in node:
                walk(i)

    walk(tokens)
    for raw in found:
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def status() -> dict:
    tokens = load_tokens()
    issued = refresh_issued_at(tokens)
    now = datetime.now(timezone.utc)

    st = {
        "path": str(TOKEN_PATH),
        "exists": TOKEN_PATH.exists(),
        "refresh_issued": issued.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z") if issued else None,
    }
    if not issued:
        st["state"] = "UNKNOWN" if tokens else "MISSING"
        return st

    expires = issued + REFRESH_TTL
    left = expires - now
    hours = left.total_seconds() / 3600
    st["expires"] = expires.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    st["hours_left"] = round(hours, 1)
    st["days_left"] = round(hours / 24, 2)
    st["state"] = ("EXPIRED" if hours <= 0
                   else "RENEW_NOW" if (now - issued) >= RENEW_AT
                   else "OK")
    return st


# ---------------------------------------------------------------------
# The exchange
# ---------------------------------------------------------------------
def extract_code(pasted: str) -> str:
    """
    Pull the authorization code out of whatever the phone produced.

    Accepts the full redirect URL, a bare 'code=...' fragment, or the
    raw code. Schwab codes end in '@', which arrives percent-encoded as
    %40; forgetting to decode it is the single most common failure here.
    """
    s = (pasted or "").strip().strip('"').strip("'")
    if not s:
        raise ValueError("nothing pasted")

    if "code=" in s:
        qs = parse_qs(urlparse(s).query) if "://" in s else parse_qs(s.lstrip("?"))
        vals = qs.get("code") or []
        if not vals:
            raise ValueError("found 'code=' but could not parse a value out of it")
        code = vals[0]
    else:
        code = s

    code = unquote(code)
    if not re.fullmatch(r"[A-Za-z0-9._\-~+/=@%]{20,}", code):
        raise ValueError("that does not look like a Schwab authorization code")
    return code


def exchange(code: str) -> dict:
    key, secret, cb = creds()
    basic = base64.b64encode(f"{key}:{secret}".encode()).decode()
    r = requests.post(
        TOKEN_URL,
        headers={"Authorization": f"Basic {basic}",
                 "Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "authorization_code", "code": code, "redirect_uri": cb},
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"token exchange failed ({r.status_code}): {r.text[:300]}")

    payload = r.json()
    for req in ("access_token", "refresh_token"):
        if req not in payload:
            raise RuntimeError(f"response missing {req}")
    return payload


def validate(access_token: str) -> str:
    """Prove the new token works BEFORE the live file is replaced."""
    r = requests.get(VALIDATE_URL,
                     headers={"Authorization": f"Bearer {access_token}"},
                     timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"validation call failed ({r.status_code}): {r.text[:200]}")
    try:
        return f"{len(r.json())} account(s) reachable"
    except ValueError:
        return "endpoint responded 200"


def install(pasted: str) -> str:
    """Full flow. Returns a human summary with no secrets in it."""
    code = extract_code(pasted)
    payload = exchange(code)
    detail = validate(payload["access_token"])          # fail here => file untouched

    previous = load_tokens()
    new_file = build_token_file(payload, previous)

    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = TOKEN_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(new_file, indent=4))
    os.chmod(tmp, 0o600)
    if TOKEN_PATH.exists():
        TOKEN_PATH.replace(TOKEN_PATH.with_suffix(".json.bak"))
    tmp.replace(TOKEN_PATH)

    expires = (datetime.now().astimezone() + REFRESH_TTL).strftime("%Y-%m-%d %H:%M %Z")
    shape = "preserved existing schema" if previous else "wrote default schwabdev schema"
    return (f"tokens.json updated ({shape})\n"
            f"validated: {detail}\n"
            f"refresh token good until {expires}\n"
            f"previous file kept at {TOKEN_PATH.with_suffix('.json.bak')}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Headless Schwab re-authorization")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--status", action="store_true", help="how long the refresh token has left")
    g.add_argument("--url", action="store_true", help="print the authorization URL")
    g.add_argument("--code", metavar="PASTED", help="redirect URL or bare code to exchange")
    args = ap.parse_args()

    if args.status:
        print(json.dumps(status(), indent=2))
        return 0
    if args.url:
        print(authorize_url())
        return 0

    try:
        print(install(args.code))
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())