#!/usr/bin/env python3
"""
token_watch.py
==============
Email a warning before the Schwab refresh token dies, with the tappable
re-authorization link already in the message.

    .venv/bin/python token_watch.py --dry-run     # print, send nothing
    .venv/bin/python token_watch.py               # send if inside the window
    .venv/bin/python token_watch.py --force       # send regardless

WHY
    The refresh token is a HARD 7-day cap from issue. Using it does not extend
    it (measured 2026-09-08). So it dies on a schedule, and it has twice died
    at an inconvenient moment — most recently mid-probe on 2026-09-23, forcing
    a re-auth over SSH.

    Re-authorizing EARLY does give a full fresh 7 days: `install()` stamps
    refresh_token_issued to now, and status() computes issued + 7d. So a
    two-minute job on day 5 or 6 removes the problem entirely.

WHAT IT SENDS
    The state, the exact expiry, days left, the safe windows to do it in, and
    the authorize URL itself — so the phone flow starts from the email rather
    than from an ops-sheet round trip.

ENV (nothing is read from .env automatically; the cron sources it)
    SMTP_HOST      default smtp.gmail.com
    SMTP_PORT      default 587
    SMTP_USER      the sending account
    SMTP_PASS      an APP PASSWORD, not the account password
    ALERT_TO       comma-separated recipients
    TOKEN_WARN_DAYS  default 2

Runs on Pi 1.
"""
from __future__ import annotations

import argparse
import os
import smtplib
import sys
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE_DIR = Path(os.getenv("REMOTE_OPS_STATE",
                           Path.home() / ".local" / "state" / "myTrading"))
SENT_MARKER = STATE_DIR / "token_watch_last_sent"

# Don't re-send inside this window. One nag a day is a reminder; one every
# cron tick is noise you learn to ignore, which defeats the purpose.
QUIET_HOURS = float(os.getenv("TOKEN_WARN_QUIET_HOURS", "20"))

# jobStocksSignals fires at :15 and :50 and runs ~9 minutes, so it owns
# :15-:24 and :50-:59 of every hour between 01:00 and 16:59 on weekdays.
# Re-auth issues a NEW refresh token and invalidates the old one, so a job
# refreshing at that moment fails.
SAFE_WINDOWS = (
    "Any weekday hour at :25-:49 or :00-:14  (the stocks job owns :15-:24 "
    "and :50-:59)\nAnything after 17:00 PT on a weekday\nAll weekend — the "
    "stocks cron does not fire Sat/Sun"
)


def recently_sent() -> bool:
    try:
        ts = datetime.fromisoformat(SENT_MARKER.read_text().strip())
    except Exception:
        return False
    return datetime.now(timezone.utc) - ts < timedelta(hours=QUIET_HOURS)


def mark_sent() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    SENT_MARKER.write_text(datetime.now(timezone.utc).isoformat())


def compose(st: dict, auth_url: str | None) -> tuple[str, str]:
    state = st.get("state", "UNKNOWN")
    days = st.get("days_left")
    expires = st.get("expires", "?")

    if state == "EXPIRED":
        subject = "🔴 Schwab token EXPIRED — nothing is trading"
    elif state == "MISSING":
        subject = "🔴 Schwab token MISSING on Pi 1"
    else:
        subject = f"⚠️ Schwab token expires in {days} day(s) — {expires}"

    body = [
        f"State        : {state}",
        f"Expires      : {expires}",
        f"Days left    : {days if days is not None else 'unknown'}",
        f"Token issued : {st.get('refresh_issued', 'unknown')}",
        "",
        "Re-authorizing EARLY gives a full fresh 7 days — the clock runs from",
        "when the token was issued, not from when it would have expired. So",
        "doing this now costs nothing and buys the whole window back.",
        "",
        "HOW  (about two minutes, from your phone)",
        "  1. Open the link below, sign in, approve.",
        "  2. The redirect to 127.0.0.1 WILL fail to load. That is expected.",
        "  3. Copy the ENTIRE address bar.",
        "  4. In myTrading-ops-pi1, insert a row at the top:",
        "         column A: auth_code",
        "         column B: <the whole pasted URL>",
        "     Fill column B FIRST, then column A — the verb is what arms the row.",
        "",
        "WHEN — avoid clashing with jobStocksSignals.py",
        SAFE_WINDOWS,
        "",
        "A re-auth issues a new refresh token and invalidates the old one, so a",
        "job that refreshes mid-flight fails. The job takes ~9 minutes.",
        "",
    ]
    if auth_url:
        body += ["AUTHORIZE LINK", auth_url, ""]
    else:
        body += ["(Could not build the authorize link — add an `auth_url` row",
                 " to the ops sheet instead.)", ""]
    body.append("— token_watch.py on Pi 1")
    return subject, "\n".join(body)


def send(subject: str, body: str, log=print) -> bool:
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "")
    pw = os.getenv("SMTP_PASS", "")
    to = [a.strip() for a in os.getenv("ALERT_TO", "").split(",") if a.strip()]

    missing = [n for n, v in (("SMTP_USER", user), ("SMTP_PASS", pw),
                              ("ALERT_TO", to)) if not v]
    if missing:
        log(f"cannot send — not set: {', '.join(missing)}")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = ", ".join(to)
    msg.set_content(body)

    with smtplib.SMTP(host, port, timeout=30) as s:
        s.starttls()
        s.login(user, pw)
        s.send_message(msg)
    log(f"sent to {', '.join(to)}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Warn by email before the Schwab token dies")
    ap.add_argument("--dry-run", action="store_true", help="print the email, send nothing")
    ap.add_argument("--force", action="store_true", help="send even if not near expiry")
    ap.add_argument("--days", type=float,
                    default=float(os.getenv("TOKEN_WARN_DAYS", "2")),
                    help="warn when fewer than this many days remain (default 2)")
    args = ap.parse_args()

    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    try:
        from dotenv import load_dotenv
        load_dotenv(HERE / ".env")
    except Exception:
        pass

    import schwab_auth
    st = schwab_auth.status()
    state = st.get("state", "UNKNOWN")
    days = st.get("days_left")

    print(f"token state: {state}"
          + (f", {days} day(s) left, expires {st.get('expires')}" if days is not None else ""))

    # UNKNOWN/MISSING are worth an email too: they mean the token store is not
    # readable, which is indistinguishable from expired as far as trading goes.
    due = (args.force
           or state in ("EXPIRED", "MISSING", "UNKNOWN", "RENEW_NOW")
           or (days is not None and days <= args.days))
    if not due:
        print(f"nothing to do — more than {args.days} day(s) left")
        return 0

    if not args.force and not args.dry_run and recently_sent():
        print(f"already warned within {QUIET_HOURS}h — staying quiet")
        return 0

    try:
        auth_url = schwab_auth.authorize_url()
    except Exception as e:
        print(f"could not build authorize url: {type(e).__name__}: {e}")
        auth_url = None

    subject, body = compose(st, auth_url)

    if args.dry_run:
        print("\n--- DRY RUN, nothing sent ---")
        print(f"Subject: {subject}\n")
        print(body)
        return 0

    if send(subject, body):
        mark_sent()
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
