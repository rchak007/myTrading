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

CREDENTIALS come from the market-tracker project, read DIRECTLY — no copy:

    ~/agents/market-tracker/.env
        GMAIL_ADDRESS
        GMAIL_APP_PASSWORD

Several paths are searched because the two Pis have different home
directories. Override with GMAIL_ENV_FILE.

Deliberately not copied here. One place to update when the password rotates,
one place that can go stale — see that project's EMAIL-SETUP.md, which is the
reference for everything below including the gotchas worked around in send().

ENV (myTrading's own .env; the cron sources it)
    ALERT_TO         comma-separated recipients
    TOKEN_WARN_DAYS  default 2

Runs on Pi 1.
"""
from __future__ import annotations

import argparse
import os
import smtplib
import sys
from datetime import datetime, timedelta, timezone
from datetime import time as dtime      # `import time` below shadows it
import time
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE_DIR = Path(os.getenv("REMOTE_OPS_STATE",
                           Path.home() / ".local" / "state" / "myTrading"))
SENT_MARKER = STATE_DIR / "token_watch_last_sent"

# Don't re-send inside this window. One nag a day is a reminder; one every
# cron tick is noise you learn to ignore, which defeats the purpose.
QUIET_HOURS = float(os.getenv("TOKEN_WARN_QUIET_HOURS", "20"))

# No safe-window rule any more: the auth_code verb takes the shared job lock
# before swapping tokens, so it waits for any running job by itself. Kept as a
# named constant because the instruction it replaced was in every email.
SAFE_WINDOWS = (
    "ANY TIME. The ops sheet takes the job lock before swapping tokens, so it\n"
    "waits for a running job rather than interrupting one. You do not need to\n"
    "check the clock."
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
        "WHEN",
        SAFE_WINDOWS,
        "",
    ]
    if auth_url:
        body += ["AUTHORIZE LINK", auth_url, ""]
    else:
        body += ["TO RE-AUTHORIZE — add a row at the top of myTrading-ops-pi1:",
                 "    column A: auth_url        (leave B empty)",
                 "Within a minute column I holds the tappable link.",
                 "Then a second row with auth_code and the pasted URL.", ""]
    body.append("— token_watch.py on Pi 1")
    return subject, "\n".join(body)


# The market-tracker project owns the Gmail credentials. Read them where they
# live rather than duplicating the secret into a second .env.
#
# Searched in order, because the two machines have different home directories
# and a path hardcoded to one of them fails silently on the other: Pi 2 is
# /home/chakravarti, Pi 1 is /home/rchak007. Path.home() covers both.
MAIL_ENV_CANDIDATES = [
    Path(os.getenv("GMAIL_ENV_FILE", "")) if os.getenv("GMAIL_ENV_FILE") else None,
    Path.home() / "agents" / "market-tracker" / ".env",
    Path("/home/chakravarti/agents/market-tracker/.env"),
    Path("/home/rchak007/agents/market-tracker/.env"),
    HERE / ".env",                      # last resort: this project's own
]
SMTP_HOST, SMTP_PORT = "smtp.gmail.com", 587


def mail_env_path() -> Path | None:
    """First candidate that exists AND actually carries the credentials.

    Existence alone is not enough — myTrading's own .env exists on both
    machines and does not contain GMAIL_*, so it must not win merely by being
    present.
    """
    for c in MAIL_ENV_CANDIDATES:
        if c and c.exists():
            try:
                if "GMAIL_APP_PASSWORD" in c.read_text():
                    return c
            except Exception:
                continue
    return None


def load_mail_env(path: Path | None = None) -> dict:
    """Credentials from market-tracker's .env.

    The FILE takes precedence over os.environ, not the reverse. A stale
    GMAIL_APP_PASSWORD exported in an interactive shell otherwise shadows the
    correct value and produces an SMTP 535 that looks exactly like a wrong
    password — documented as gotcha #2 in that project's EMAIL-SETUP.md, and it
    cost real time there.
    """
    path = path or mail_env_path()
    values = {}
    if path and path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            values[k.strip()] = v.strip().strip('"').strip("'")
    for k, v in os.environ.items():
        values.setdefault(k, v)
    return values


IMAGE_SUBTYPES = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg",
                  ".gif": "gif", ".webp": "webp"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024          # Gmail rejects well before 25MB total


def _read_images(paths, log) -> list[tuple[str, str, bytes]]:
    """(cid, subtype, data) for each readable image. Skips, never raises.

    An unreadable picture must not cost the email. The whole point of the
    attachment is that the text arrives.
    """
    out = []
    for i, p in enumerate(paths or []):
        p = Path(p)
        sub = IMAGE_SUBTYPES.get(p.suffix.lower())
        if sub is None:
            log(f"skipping {p.name}: not an image type I inline")
            continue
        try:
            data = p.read_bytes()
        except Exception as e:
            log(f"skipping {p.name}: {e}")
            continue
        if len(data) > MAX_IMAGE_BYTES:
            log(f"skipping {p.name}: {len(data)/1e6:.1f} MB is too big to mail")
            continue
        out.append((f"img{i}@mytrading", sub, data))
    return out


def _html_body(body: str, images: list[tuple[str, str, bytes]]) -> str:
    """The plain text, preserved verbatim, with the pictures under it.

    The text part stays authoritative — this is the same words, not a second
    version of them, so the two alternatives can never disagree.
    """
    import html as _html
    parts = ["<div style=\"font-family:ui-monospace,Menlo,Consolas,monospace;"
             "font-size:13px;white-space:pre-wrap\">",
             _html.escape(body),
             "</div>"]
    for cid, _sub, _data in images:
        parts.append(f'<div style="margin-top:18px">'
                     f'<img src="cid:{cid}" style="max-width:100%;height:auto">'
                     f'</div>')
    return "".join(parts)


def send(subject: str, body: str, log=print, images=None) -> bool:
    """Send, and NEVER raise. Returns whether every recipient got it.

    A failure here must not propagate: the caller only marks the warning as
    sent on success, so a refused send simply retries on the next run rather
    than being silently swallowed. For a token about to expire, a missed
    warning is the whole failure.

    `images` are paths shown INLINE at the end of the message, not handed over
    as files to open. A note you have to tap twice to see is a note you stop
    looking at — Chakravarti asked for the house rules to be glanceable.
    """
    env = load_mail_env()
    address = env.get("GMAIL_ADDRESS", "")
    # Google displays app passwords as four groups of four. The credential is
    # the 16 characters with spaces removed; sending it with spaces gives a
    # 535 indistinguishable from a wrong password.
    password = env.get("GMAIL_APP_PASSWORD", "").replace(" ", "")
    to = [a.strip() for a in os.getenv(
        "ALERT_TO", "rchak0071@gmail.com,geniusact@keep-empowering.com"
    ).split(",") if a.strip()]

    if not address or not password:
        tried = "\n  ".join(str(c) for c in MAIL_ENV_CANDIDATES if c)
        log(f"no GMAIL_ADDRESS / GMAIL_APP_PASSWORD found. Looked in:\n  {tried}"
            f"\nSet GMAIL_ENV_FILE to point at the file, or copy the two "
            f"values into this project's .env.")
        return False
    if not to:
        log("ALERT_TO is empty — nobody to tell")
        return False

    # Read each picture ONCE, before connecting — the bytes are reused for
    # every recipient, and a bad path should be reported before we hold an
    # open SMTP session.
    imgs = _read_images(images, log)

    try:
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15)
    except Exception as e:
        log(f"SMTP connect failed: {e}")
        return False

    ok = True
    with server:
        try:
            server.starttls()
            server.login(address, password)
        except Exception as e:
            # 535 is a genuinely wrong credential; 534 5.7.14 is a
            # suspicious-sign-in flag where the credential is fine and a normal
            # browser sign-in on this network clears it.
            log(f"SMTP login failed: {e}")
            return False

        for i, addr in enumerate(to):
            if i:
                time.sleep(2)          # pacing: a burst scores on Gmail's abuse heuristics
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = formataddr(("myTrading on Pi 1", address))
            msg["To"] = addr
            msg.set_content(body)
            if imgs:
                # alternative[ text/plain, related[ text/html, image... ] ].
                # The images hang off the HTML part, not off the message, or a
                # client showing plain text advertises attachments it cannot
                # place.
                msg.add_alternative(_html_body(body, imgs), subtype="html")
                html_part = msg.get_payload()[-1]
                for cid, sub, data in imgs:
                    html_part.add_related(data, maintype="image", subtype=sub,
                                          cid=f"<{cid}>")
            try:
                server.send_message(msg)
                log(f"sent to {addr}")
            except Exception as e:
                log(f"send to {addr} failed: {e}")
                ok = False
    return ok


# jobStocksSignals' cron: `15,50 1-16 * * 1-5`. Pi 1 is NOT expected to poll
# outside these, which is the whole reason this is here — a flat "3 hours"
# threshold cried wolf every Saturday and Sunday morning, and twice before
# anyone checked whether anything was actually wrong.
POLL_MINUTES = (15, 50)
POLL_HOURS = range(1, 17)          # 01:00–16:59 PT
POLL_WEEKDAYS = range(0, 5)        # Mon–Fri
# The job takes ~8-10 minutes. Allow it to finish before calling it late.
POLL_GRACE_MIN = int(os.getenv("POLL_GRACE_MIN", "40"))


def expected_last_poll(now: datetime) -> datetime | None:
    """The most recent moment the stocks job was SCHEDULED to start.

    Staleness has to be measured against the schedule, not against the clock.
    `LAST POLL` being 16 hours old at 08:30 on a Saturday is exactly correct —
    Friday's final slot is 16:50 and there is nothing until Monday.
    """
    day = now.date()
    for back in range(0, 10):
        d = day - timedelta(days=back)
        if d.weekday() not in POLL_WEEKDAYS:
            continue
        past = [datetime.combine(d, dtime(h, m), tzinfo=now.tzinfo)
                for h in POLL_HOURS for m in POLL_MINUTES
                if datetime.combine(d, dtime(h, m), tzinfo=now.tzinfo) <= now]
        if past:
            return max(past)
    return None


def poll_is_stale(last_poll: datetime | None, now: datetime | None = None) -> bool:
    """Has Pi 1 missed a run it was actually scheduled for?

    True only when a scheduled slot has come and gone, with time to finish,
    and the header still predates it.
    """
    if last_poll is None:
        return False
    now = now or datetime.now(last_poll.tzinfo)
    # Look back from `now - grace`, not from `now`. Asking for the newest slot
    # and then allowing grace lets an OLDER missed slot hide behind a recent
    # one that is still legitimately running: at 02:10 the 01:50 run is within
    # its grace, but 01:15 was skipped entirely and nothing noticed.
    due = expected_last_poll(now - timedelta(minutes=POLL_GRACE_MIN))
    return due is not None and last_poll < due


def read_from_sheet(log=print) -> tuple[dict, str | None, float | None]:
    """Token state and poll freshness, read from the Orders header.

    Lets PI 2 do the warning. Pi 1 holds the Schwab credentials and rewrites
    this header every cycle; Pi 2 holds the Gmail credentials and can read the
    sheet. Neither needs what the other has, and nothing has to be copied
    between them.

    Returns (status-like dict, error, hours since LAST POLL). The poll age is
    the bonus: a header that has stopped moving means Pi 1 is down, which is
    worth an email in its own right and is invisible from Pi 1 by definition.
    """
    try:
        from google.oauth2.service_account import Credentials
        import gspread
        creds = os.getenv("GSHEET_READER_CREDS", "")
        sid = os.getenv("GSHEET_ORDERS_ID", "")
        if not creds or not sid:
            return {}, "GSHEET_READER_CREDS / GSHEET_ORDERS_ID not set", None
        gc = gspread.authorize(Credentials.from_service_account_file(
            creds, scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]))
        ws = gc.open_by_key(sid).worksheet("Orders")
        cells = ws.get("A1:B6")
    except Exception as e:
        return {}, f"could not read the Orders header: {type(e).__name__}: {e}", None

    rows = {str(r[0]).strip().upper(): (r[1] if len(r) > 1 else "")
            for r in cells if r}
    tok = str(rows.get("SCHWAB TOKEN", "")).strip()
    poll = str(rows.get("LAST POLL", "")).strip()
    if not tok:
        return {}, "no SCHWAB TOKEN row in the header", None

    # "OK — expires 2026-09-30 23:21:22 PDT (1.52 days)"
    import re as _re
    st = {"state": tok.split()[0].strip("—").strip() or "UNKNOWN",
          "refresh_issued": "(from the sheet)"}
    m = _re.search(r"expires\s+([\d-]+\s+[\d:]+\s*\w*)", tok)
    if m:
        st["expires"] = m.group(1).strip()
    m = _re.search(r"\(([\d.]+)\s*days?\)", tok)
    if m:
        st["days_left"] = float(m.group(1))

    age = None
    try:
        from zoneinfo import ZoneInfo
        pt = ZoneInfo("America/Los_Angeles")
        when = datetime.strptime(poll[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=pt)
        age = (datetime.now(pt) - when).total_seconds() / 3600
        st["_last_poll"] = when
    except Exception:
        pass
    return st, None, age


def main() -> int:
    ap = argparse.ArgumentParser(description="Warn by email before the Schwab token dies")
    ap.add_argument("--from-sheet", action="store_true",
                    help="read the token state from the Orders header instead "
                         "of the local token store — lets Pi 2 do the warning")
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

    poll_age = None
    if args.from_sheet:
        st, err, poll_age = read_from_sheet()
        if err:
            print(f"⛔ {err}")
            return 1
    else:
        import schwab_auth
        st = schwab_auth.status()
    state = st.get("state", "UNKNOWN")
    days = st.get("days_left")

    print(f"token state: {state}"
          + (f", {days} day(s) left, expires {st.get('expires')}" if days is not None else ""))
    if poll_age is not None:
        print(f"Pi 1 last polled {poll_age:.1f}h ago")

    # UNKNOWN/MISSING are worth an email too: they mean the token store is not
    # readable, which is indistinguishable from expired as far as trading goes.
    # A header that has stopped moving means Pi 1 is down — worth an email in
    # its own right, and something Pi 1 could never tell you itself.
    # Measured against the SCHEDULE, not a flat hour count. Falls back to the
    # old threshold only when the timestamp could not be parsed at all.
    last_poll = (st or {}).get("_last_poll")
    stale = (poll_is_stale(last_poll) if last_poll is not None
             else (poll_age is not None
                   and poll_age > float(os.getenv("POLL_STALE_HOURS", "3"))))
    due = (args.force or stale
           or state in ("EXPIRED", "MISSING", "UNKNOWN", "RENEW_NOW")
           or (days is not None and days <= args.days))
    if not due:
        print(f"nothing to do — more than {args.days} day(s) left")
        return 0

    if not args.force and not args.dry_run and recently_sent():
        print(f"already warned within {QUIET_HOURS}h — staying quiet")
        return 0

    auth_url = None
    if not args.from_sheet:
        try:
            import schwab_auth
            auth_url = schwab_auth.authorize_url()
        except Exception as e:
            print(f"could not build authorize url: {type(e).__name__}: {e}")

    subject, body = compose(st, auth_url)
    if stale:
        subject = f"🔴 Pi 1 has not polled in {poll_age:.0f}h — {subject}"
        body = (f"LAST POLL on the Orders sheet is {poll_age:.1f} HOURS old.\n"
                f"Pi 1 writes that header every cycle, so it has stopped "
                f"running.\nNothing is trading, pricing or reporting.\n\n"
                + body)

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
