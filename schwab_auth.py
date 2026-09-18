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
# WHERE THE TOKENS ACTUALLY LIVE  (fixed 2026-09-18)
#   Current schwabdev keeps tokens in a SQLite database at
#   ~/.schwabdev/tokens.db, one row in a table called `schwabdev`. It no
#   longer reads or writes tokens.json at all.
#
#   This module used to read AND write tokens.json, which meant two
#   things went wrong at once:
#     * --status reported on a file nothing uses, so it called a live
#       token dead (observed 2026-09-14: the JSON was four months stale
#       while Schwab calls succeeded normally);
#     * install() wrote the new credentials to that same ignored file,
#       so a re-auth would report success and change nothing.
#   Both now go through the database.
#
#   tokens.json is left alone as the fossil it is. Do not resurrect it.
#
# SAFETY
#   The live token row is never written until the NEW credentials have
#   been proven against a real Schwab endpoint. Failure leaves the
#   working row untouched. The database is copied to tokens.db.bak
#   before any write.
# =====================================================================
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import sqlite3
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

# The real store. Shared by every Schwab caller on the box, so a re-auth
# here fixes all of them at once.
TOKEN_DB = Path(os.getenv("SCHWABDEV_TOKENS_DB",
                          Path.home() / ".schwabdev" / "tokens.db")).expanduser()
TOKEN_TABLE = "schwabdev"

# Legacy path. Nothing reads it; kept only so --status can say so out loud
# when it is still lying around confusing people.
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
    """
    The live token row, flat, straight out of schwabdev's SQLite store.

    Columns: access_token_issued, refresh_token_issued, access_token,
    refresh_token, id_token, expires_in, token_type, scope. The *_issued
    stamps are ISO 8601 with a UTC offset.
    """
    if not TOKEN_DB.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{TOKEN_DB}?mode=ro", uri=True)
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute(f"SELECT * FROM {TOKEN_TABLE} LIMIT 1").fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    return dict(row) if row else None


def write_tokens(payload: dict) -> str:
    """
    Put freshly exchanged credentials into schwabdev's store.

    Both *_issued stamps are set to now, in UTC ISO 8601 — matching what
    schwabdev writes, and what refresh_issued_at() expects to parse. A
    code exchange mints a NEW refresh token, so its 7-day clock restarts
    here; that is the whole point of re-authorizing.

    Returns the path of the backup taken before the write.
    """
    stamped = datetime.now(timezone.utc).isoformat()
    row = {
        "access_token_issued": stamped,
        "refresh_token_issued": stamped,
        "access_token": payload["access_token"],
        "refresh_token": payload["refresh_token"],
        "id_token": payload.get("id_token", ""),
        "expires_in": int(payload.get("expires_in", 1800)),
        "token_type": payload.get("token_type", "Bearer"),
        "scope": payload.get("scope", "api"),
    }

    TOKEN_DB.parent.mkdir(parents=True, exist_ok=True)
    backup = TOKEN_DB.with_suffix(".db.bak")
    if TOKEN_DB.exists():
        shutil.copy2(TOKEN_DB, backup)
        os.chmod(backup, 0o600)

    conn = sqlite3.connect(TOKEN_DB)
    try:
        cols = ", ".join(f"{k} = ?" for k in row)
        cur = conn.execute(f"UPDATE {TOKEN_TABLE} SET {cols}", list(row.values()))
        if cur.rowcount == 0:
            # Empty table — first ever write.
            names = ", ".join(row)
            marks = ", ".join("?" * len(row))
            conn.execute(f"INSERT INTO {TOKEN_TABLE} ({names}) VALUES ({marks})",
                         list(row.values()))
        conn.commit()
    finally:
        conn.close()

    os.chmod(TOKEN_DB, 0o600)
    return str(backup)


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
        "path": str(TOKEN_DB),
        "exists": TOKEN_DB.exists(),
        "refresh_issued": issued.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z") if issued else None,
    }
    if TOKEN_PATH.exists():
        st["legacy_file"] = (f"{TOKEN_PATH} still exists and is IGNORED — "
                             "its date says nothing about whether auth works")
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
    detail = validate(payload["access_token"])          # fail here => store untouched

    backup = write_tokens(payload)

    expires = (datetime.now().astimezone() + REFRESH_TTL).strftime("%Y-%m-%d %H:%M %Z")
    return (f"{TOKEN_DB} updated — schwabdev will pick this up immediately\n"
            f"validated: {detail}\n"
            f"refresh token good until {expires}\n"
            f"previous database kept at {backup}")


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