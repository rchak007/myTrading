"""
schwab_client.py
----------------
Single source of truth for the Schwab connection.

TOKEN STORAGE -- read this before "fixing" anything here.

Current schwabdev keeps tokens in a SQLite DB (~/.schwabdev/tokens.db), NOT in
a tokens.json. Its constructor takes `tokens_db`; the old `tokens_file` keyword
no longer exists and passing it raises TypeError. That is the whole reason this
module used to fail on Pi 1 while data/schwab/schwab_helper.py kept working --
the helper was updated for the new schwabdev, this file was not.

The DB is shared with every other Schwab caller on the box, so re-authenticating
anywhere re-authenticates everything. We pass its path EXPLICITLY rather than
relying on the default, because this module runs with cwd=schwabAPI/ and a
silently-wrong path makes schwabdev start a fresh OAuth flow instead of reusing
the working session.

~/github/myTrading/tokens.json is a LEGACY artifact from the old schwabdev. It
is no longer read by anything here. Do not resurrect it.

.env (app_key / app_secret / callback_url) is read from the parent folder,
with a local fallback.
"""

import os
import sys
from pathlib import Path


# schwabAPI/ -> myTrading/
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Where current schwabdev actually keeps tokens. Shared by every Schwab caller
# on this machine, so a re-auth anywhere fixes all of them at once.
TOKENS_DB = Path(
    os.getenv("SCHWABDEV_TOKENS_DB", str(Path.home() / ".schwabdev" / "tokens.db"))
).expanduser()

# Legacy path from the old schwabdev. Nothing reads it; kept only so
# stray_token_check() can point at it if it reappears and confuses someone.
TOKENS_FILE = ROOT / "tokens.json"

# Where transaction cache lives (kept inside schwabAPI, git-ignorable)
CACHE_DIR = HERE / "data" / "transactions"

# Where final reports go (already published by the Streamlit dashboard)
REPORT_DIR = Path(
    os.getenv(
        "PORTFOLIO_OUT_DIR",
        str(Path.home() / "github" / "jobMyTrading" / "outputs" / "portfolio"),
    )
)

# Cold-start date. Schwab history begins ~2020-03 for these accounts; we start
# earlier and let empty chunks fall through harmlessly.
COLD_START = os.getenv("SCHWAB_START_DATE", "2019-01-01")

# Re-fetch this many days back from last_date on every incremental run.
# Schwab back-dates / corrects settled activity, so never resume exactly at
# last_date + 1.
OVERLAP_DAYS = 10


def _load_env() -> None:
    import dotenv

    for candidate in (ROOT / ".env", HERE / ".env"):
        if candidate.exists():
            dotenv.load_dotenv(candidate, override=False)


def make_client():
    """
    Build a schwabdev client bound to the shared ~/.schwabdev/tokens.db.
    """
    import schwabdev

    _load_env()

    app_key = os.getenv("app_key")
    app_secret = os.getenv("app_secret")
    callback_url = os.getenv("callback_url")

    missing = [
        n
        for n, v in (
            ("app_key", app_key),
            ("app_secret", app_secret),
            ("callback_url", callback_url),
        )
        if not v
    ]
    if missing:
        raise RuntimeError(
            f"Missing {', '.join(missing)} in .env "
            f"(looked in {ROOT / '.env'} and {HERE / '.env'})"
        )

    if not TOKENS_DB.exists():
        raise RuntimeError(
            f"No schwabdev token store at {TOKENS_DB}.\n"
            "Authenticate once with any Schwab tool on this machine "
            "(e.g. `python3 test_schwab_one.py TSLA` from the repo root) — "
            "they all share this DB — then re-run."
        )

    # `tokens_db`, not `tokens_file` — the old keyword no longer exists.
    return schwabdev.Client(
        app_key,
        app_secret,
        callback_url,
        tokens_db=str(TOKENS_DB),
    )


def stray_token_check() -> None:
    """Warn if a legacy tokens.json is lying around pretending to matter."""
    for stray in (HERE / "tokens.json", TOKENS_FILE):
        if stray.exists():
            print(
                f"NOTE: legacy token file at {stray}\n"
                f"      Current schwabdev reads {TOKENS_DB} instead, so this file "
                f"      is ignored. Its date tells you nothing about whether auth works.",
                file=sys.stderr,
            )