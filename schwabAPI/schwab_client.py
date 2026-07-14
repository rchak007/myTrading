"""
schwab_client.py
----------------
Single source of truth for the Schwab connection.

IMPORTANT: There is exactly ONE tokens.json for the whole project, and it lives
in the PARENT folder (~/github/myTrading/tokens.json) -- the same file app.py
uses. schwabAPI/ must NEVER create its own tokens.json, otherwise the two copies
drift and the ~7-day Schwab re-auth silently breaks one of them.

.env (app_key / app_secret / callback_url) is also read from the parent folder,
with a local fallback.
"""

import os
import sys
from pathlib import Path


# schwabAPI/ -> myTrading/
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# THE one and only token file (Pi owns this)
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
    Build a schwabdev client bound to the SHARED parent tokens.json.
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

    if not TOKENS_FILE.exists():
        raise RuntimeError(
            f"No token file at {TOKENS_FILE}.\n"
            "Run the OAuth flow on the Pi (or scp a fresh tokens.json there) first. "
            "Do NOT create a second tokens.json inside schwabAPI/."
        )

    return schwabdev.Client(
        app_key,
        app_secret,
        callback_url,
        tokens_file=str(TOKENS_FILE),
    )


def stray_token_check() -> None:
    """Warn loudly if a second token file has appeared inside schwabAPI/."""
    stray = HERE / "tokens.json"
    if stray.exists():
        print(
            f"WARNING: stray token file found at {stray}\n"
            f"         The project uses {TOKENS_FILE} only. Delete the stray copy.",
            file=sys.stderr,
        )