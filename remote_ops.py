#!/usr/bin/env python3
# =====================================================================
# remote_ops.py — allowlisted remote command channel for Pi 1
# ---------------------------------------------------------------------
# Pi 1 polls an ops Google Sheet, runs ONLY verbs that appear in the
# VERBS allowlist below, and writes the result back into the same row
# so you can read it from your phone.
#
# SECURITY MODEL
#   - The sheet supplies a VERB NAME, never a command string.
#   - Every command is an argv list. shell=True appears nowhere.
#   - Arguments are keys into fixed dicts, never paths, so "../../.env"
#     cannot be smuggled through.
#   - Anything that moves funds, touches tokens.json, or restarts the
#     bot service is deliberately NOT here. Keep it that way.
#
# WHERE WORK GOES
#   NEW COMMANDS GO AT THE TOP, directly under the header. The poller walks
#   down from row 2 and STOPS at the first row that already has a Status —
#   everything below that is history. Insert rows above to queue work; never
#   clear a Status to "reuse" a row.
#
# SHEET LAYOUT (row 1 is the header, written by --init)
#   A Verb  B Args  C Status  D QueuedAt  E StartedAt  F FinishedAt
#   G Secs  H Exit  I Output  J Host
#
# You fill A and B. Pi 1 fills C..J. Never DELETE a row and never clear a
# Status — a stamped row is the boundary marker, and losing it makes the
# poller walk down into history. Archive by copying to another tab.
# =====================================================================
from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
HOME = Path.home()
REPO = Path(os.getenv("MYTRADING_REPO", HOME / "github" / "myTrading"))
JOBS_REPO = Path(os.getenv("JOBS_REPO", HOME / "github" / "jobMyTrading"))
BOTS_REPO = Path(os.getenv("BOTS_REPO", HOME / "github" / "botsMyTrading"))

STATE_DIR = Path(os.getenv("REMOTE_OPS_STATE", HOME / ".local" / "state" / "myTrading"))
AUDIT_LOG = STATE_DIR / "remote_ops_audit.log"
OUTPUT_DIR = STATE_DIR / "remote_ops_out"

CREDS_PATH = os.getenv("REMOTE_OPS_CREDS",
                       os.getenv("GSHEET_CREDS", "/etc/myTrading/gsheets.json"))

# Full read/write scope. gsheet_notes.py uses spreadsheets.readonly and
# must keep using it — this module is the only writer.
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

HOST = socket.gethostname()
TS_FMT = "%Y-%m-%d %H:%M:%S %Z"

# ---------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------
START_ROW = 2               # row 1 is the header; new work goes directly below
SCAN_WINDOW = 25            # rows from START_ROW examined each poll, stamped or not
MAX_ROWS_PER_RUN = 5        # a pasted wall of rows cannot stampede the box

# After handing out an authorize URL, watch for the code instead of sleeping
# until the next cron tick. Schwab's authorization code expires in seconds,
# and a once-a-minute poll is slower than that — see _watch_for_code.
AUTH_WATCH_SECS = int(os.getenv("REMOTE_OPS_AUTH_WATCH", "300"))
AUTH_WATCH_EVERY = int(os.getenv("REMOTE_OPS_AUTH_POLL", "4"))
SHEET_OUTPUT_MAX = 1500     # chars kept in column I; full text goes to disk
TAIL_LINES = 120

HEADER = ["Verb", "Args", "Status", "QueuedAt", "StartedAt",
          "FinishedAt", "Secs", "Exit", "Output", "Host"]

# Log keys you may pass to tail_log. The sheet supplies the KEY, this
# dict supplies the path. Add entries here, never in the sheet.
LOGS = {
    "stocks": REPO / "job_stocks.log",
    "crypto": REPO / "job_crypto.log",
    "45":     REPO / "job_45.log",
    "ops":    AUDIT_LOG,
}

# Directories you may pass to ls. Same rule: key in, path out.
DIRS = {
    "repo":   REPO,
    "jobs":   JOBS_REPO,
    "bots":   BOTS_REPO,
}

SERVICE = os.getenv("BOT_SERVICE", "mytrading-bot.service")


# ---------------------------------------------------------------------
# Verb allowlist
# ---------------------------------------------------------------------
# Each builder returns a LIST of argv lists, run in order. Output is
# concatenated. Exit code is the first non-zero one.
def _git_pull(_):
    return [["git", "-C", str(REPO), "pull", "--rebase", "--autostash"],
            ["git", "-C", str(REPO), "log", "--oneline", "-5"]]


def _tail_log(arg):
    if arg not in LOGS:
        raise ValueError(f"unknown log key {arg!r}; valid: {', '.join(sorted(LOGS))}")
    return [["tail", "-n", str(TAIL_LINES), str(LOGS[arg])]]


def _ls(arg):
    key = arg or "jobs"
    if key not in DIRS:
        raise ValueError(f"unknown dir key {key!r}; valid: {', '.join(sorted(DIRS))}")
    return [["ls", "-la", str(DIRS[key])]]


def _status(_):
    return [["systemctl", "is-active", SERVICE],
            ["systemctl", "show", SERVICE,
             "--property=ActiveState,SubState,ExecMainStartTimestamp,NRestarts"],
            ["uptime"],
            ["df", "-h", "/"]]


# Packages this channel may upgrade. A NAMED SET, not a free-form argument.
#
# 2026-10-20: Pi 1's yfinance went stale and every download failed with
# "'NoneType' object is not subscriptable" — all 53 tickers, from the first
# call. The signals and price jobs kept running and kept writing nothing, so
# the sheet sat 17 hours out of date while the market was open. Pi 2, on
# yfinance 1.7.0, fetched the same ticker without complaint.
#
# The fix was one pip command, and it needed SSH — which is precisely the
# dependency this channel exists to remove. Chakravarti was away from home
# and could not reach the box.
#
# WHY AN ALLOWLIST. Letting the sheet name any package turns a spreadsheet
# cell into arbitrary code execution on the machine holding the brokerage
# credentials: a typo'd or hostile name is a package install. These are the
# market-data libraries that actually rot, and nothing else.
PIP_OK = {"yfinance", "curl_cffi", "requests", "pandas", "gspread"}


def _pip_upgrade(arg):
    """Upgrade one allowlisted package in the repo venv, reporting both versions.

    Prints the version before and after so the sheet shows what changed —
    "already current" and "1.2.3 -> 1.7.0" are different answers, and after
    a silent-failure bug you want the difference stated rather than implied.
    """
    pkg = (arg or "").strip().lower()
    if pkg not in PIP_OK:
        raise ValueError(
            f"{pkg!r} is not upgradable from the sheet; allowed: "
            f"{', '.join(sorted(PIP_OK))}")
    pip = str(REPO / ".venv" / "bin" / "pip")
    return [[pip, "show", pkg],
            [pip, "install", "--upgrade", pkg],
            [pip, "show", pkg]]


VERBS = {
    # verb        builder      timeout(s)  needs_arg
    "ping":       (lambda a: [["date", "-Is"], ["echo", f"alive on {HOST}"]], 20, False),
    "git_pull":   (_git_pull, 240, False),
    "git_status": (lambda a: [["git", "-C", str(REPO), "status", "--short", "--branch"]], 60, False),
    "git_log":    (lambda a: [["git", "-C", str(REPO), "log", "--oneline", "-15"]], 60, False),
    "status":     (_status, 60, False),
    "disk":       (lambda a: [["df", "-h"]], 30, False),
    "uptime":     (lambda a: [["uptime"], ["free", "-h"]], 30, False),
    "ls":         (_ls, 30, False),
    "tail_log":   (_tail_log, 60, True),
    "cron_check": (lambda a: [["crontab", "-l"]], 30, False),
    "svc_log":    (lambda a: [["journalctl", "-u", SERVICE, "-n", "80",
                               "--no-pager", "--output=short-iso"]], 60, False),
    # 300s: pip on a Pi builds wheels and is slow. svc_log's 60s cap already
    # cost us a FAIL on a journalctl that simply needed longer.
    "pip_upgrade": (_pip_upgrade, 300, True),
}

# Verbs handled in-process rather than via subprocess. They may take a
# free-form argument because nothing here reaches a shell or an argv.
#   verb -> (needs_arg, sensitive)
#
# These were one boolean until `seed` arrived, which needs an argument but
# whose argument is NOT a secret — and which moves money, so the argument is
# exactly what you want in the audit log. Conflating "requires an argument"
# with "must be redacted" would have silently hidden every seed amount.
PY_VERBS = {
    "token_status": (False, False),
    "auth_url":     (False, False),
    "auth_code":    (True,  True),
    "seed":         (True,  False),
    "fence":        (True,  False),
    # The RELEASE side of seeding. Without these the ops sheet could only
    # ever add reserves, never undo one — so an over-fenced account could not
    # be fixed without SSH, which is the thing this channel exists to avoid.
    # Both only REDUCE a reserve: neither moves money nor places an order.
    "withdraw":     (True,  False),
    "close":        (True,  False),
    # Borrowing between tickers. Amounts belong in the audit log for the same
    # reason seed's does: they move money, and the argument IS the record.
    "borrow":        (True, False),
    "return_borrow": (True, False),
    "owe":           (True, False),
    "paid_back":     (True, False),
    "apply_fills":  (False, False),
    "reserves":     (False, False),
    # Rebuild the sheet tabs now instead of waiting for :15 / :50. Launches
    # detached and returns at once — see _run_signals for why it must not
    # block the poller. Places no orders and moves no money; it recomputes
    # signals and rewrites the read-only tabs.
    "verbs_doc":    (False, False),
    # Sticky notes. Free text, so the whole of column B after the ticker is
    # the note — never split into fixed fields.
    "note":          (True,  False),
    "note_replace":  (True,  False),
    "note_delete":   (True,  False),
    "notes":         (False, False),
    "run_signals":  (False, False),
}


# A fat-fingered amount on a phone keyboard is the realistic failure here, not
# a malicious one. Anything above this is far more likely a typo than an
# intention, and the CLI on Pi 1 remains available for a genuinely large seed.
SEED_MAX = float(os.getenv("REMOTE_OPS_SEED_MAX", "100000"))

SEED_USAGE = (
    "seed  ACCT TICKER AMOUNT [POLICY] [TARGET]\n"
    "  e.g.  seed  171 MU 7606.68 TOTAL_CAPITAL 7968.10\n"
    "        seed  885 NOC 5000\n"
    "POLICY is CASH_ONLY (default) or TOTAL_CAPITAL.\n"
    "TOTAL_CAPITAL requires TARGET — the ceiling for that account's position."
)


def _run_seed(arg: str) -> tuple[int, str]:
    """Fence cash to a ticker from the ops sheet, so seeding needs no SSH.

    The ledger stays authoritative: this calls the same cash_reserve.seed()
    the CLI does. The sheet carries the intent, never the balance.
    """
    import cash_reserve

    parts = arg.split()
    if len(parts) < 3:
        return 2, "need at least ACCT TICKER AMOUNT\n\n" + SEED_USAGE

    acct, ticker, raw_amount = parts[0], parts[1], parts[2]
    policy = parts[3].upper() if len(parts) > 3 else "CASH_ONLY"
    raw_target = parts[4] if len(parts) > 4 else None

    try:
        amount = float(raw_amount.replace(",", "").lstrip("$"))
    except ValueError:
        return 2, f"AMOUNT {raw_amount!r} is not a number\n\n" + SEED_USAGE
    if amount <= 0:
        return 2, "AMOUNT must be positive\n\n" + SEED_USAGE
    if amount > SEED_MAX:
        return 2, (f"AMOUNT ${amount:,.2f} exceeds the ops-sheet ceiling of "
                   f"${SEED_MAX:,.2f}. If that was deliberate, seed it from the "
                   f"CLI on Pi 1; if not, check for a stray digit.")

    if policy not in cash_reserve.POLICIES:
        return 2, (f"POLICY {policy!r} unknown — use one of "
                   f"{', '.join(sorted(cash_reserve.POLICIES))}\n\n" + SEED_USAGE)

    target = None
    if raw_target is not None:
        try:
            target = float(raw_target.replace(",", "").lstrip("$"))
        except ValueError:
            return 2, f"TARGET {raw_target!r} is not a number\n\n" + SEED_USAGE
    if policy == "TOTAL_CAPITAL" and target is None:
        return 2, ("TOTAL_CAPITAL needs a TARGET, or the reserve can never "
                   "bind.\n\n" + SEED_USAGE)

    out: list[str] = []
    bal = cash_reserve.seed(acct, ticker, amount, policy=policy,
                            target_capital=target, source="ops_sheet",
                            reason="seeded from ops sheet",
                            log=out.append)
    out.append(f"\nreserve balance for {acct}/{ticker.upper()} is now ${bal:,.2f}")
    out.append("to undo:  withdraw or close, from the CLI on Pi 1")
    return 0, "\n".join(out)


def run_pyverb(verb: str, arg: str) -> tuple[int, str]:
    """Dispatch an in-process verb. NEVER raises.

    Every exception becomes an exit code and a message, because the caller has
    already stamped the row RUNNING by the time this is invoked. An escaping
    exception killed the whole poll cycle and left that row looking handled —
    stuck on RUNNING, skipped forever, no error anywhere the user would see,
    and the poller dying again every 10 minutes.

    `auth_url` was the worst case: it is what you reach for when the token has
    expired, which is precisely when everything else is failing too.
    """
    # Belt for running by hand. The cron sources .env before invoking us, but
    # someone at a terminal will not, and schwab_auth reads credentials from
    # the environment only.
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO / ".env")
    except Exception:
        pass                              # absent dotenv is not fatal

    try:
        return _run_pyverb(verb, arg)
    except Exception as e:
        audit(event="pyverb_error", verb=verb, error=f"{type(e).__name__}: {e}")
        return 1, (f"{type(e).__name__}: {e}\n\n"
                   f"The verb failed but the poller is fine — this row is "
                   f"marked FAIL rather than left stranded on RUNNING.")


def _run_signals() -> tuple[int, str]:
    """Kick off jobStocksSignals and return AT ONCE.

    Chakravarti, 2026-10-14: *"many time when we update the excel sheet we
    need this."* The Dashboard, Positions, Cash and the reserve tables are
    only rebuilt at :15 and :50, Mon-Fri 01-16h, so an edit made at 17:00 is
    invisible until 01:15 the next morning.

    DETACHED, NOT WAITED ON. The job takes 9-10 minutes and run_verb() uses a
    blocking subprocess.run — holding the poller for ten minutes would stall
    every other verb behind it and overlap the next poll. setsid --fork
    returns immediately; the job's own logging is the progress report.

    flock -n, NOT -w. The scheduled run takes the same lock. Queueing behind
    it would mean a second full pass starting the moment the first ended, for
    no benefit — if it is already running, the rebuild he wants is already
    happening. Reported, not silently skipped.
    """
    import shutil
    import subprocess as sp

    lock = Path(os.getenv("MYTRADING_JOB_LOCK", "/tmp/jobmytrading.lock"))
    py = REPO / ".venv" / "bin" / "python"
    job = REPO / "jobStocksSignals.py"
    if not py.exists():
        return 1, f"no interpreter at {py}"
    if not job.exists():
        return 1, f"no {job.name} in {REPO}"
    for tool in ("setsid", "flock"):
        if not shutil.which(tool):
            return 1, f"{tool} is not installed — cannot start detached"

    # Is it already running? flock -n on a subshell that exits immediately
    # tells us without waiting and without disturbing the holder.
    busy = sp.run(["flock", "-n", str(lock), "true"],
                  capture_output=True).returncode != 0
    if busy:
        return 0, ("already running — the job lock is held, so a rebuild is "
                   "in progress right now.\n"
                   "Watch it with:   tail_log  stocks")

    sp.Popen(["setsid", "--fork", "flock", "-n", str(lock),
              str(py), str(job), "--no-push"],
             cwd=str(REPO), stdin=sp.DEVNULL,
             stdout=sp.DEVNULL, stderr=sp.DEVNULL, start_new_session=True)
    audit(event="signals_started", verb="run_signals")
    return 0, ("started jobStocksSignals in the background.\n\n"
               "It takes 9-10 minutes and rewrites Positions, Cash, Dashboard "
               "and the reserve tables.\n"
               "This row is DONE as soon as it LAUNCHED — it does not wait "
               "for the job to finish.\n\n"
               "Progress:   tail_log  stocks\n"
               "Finished when that log shows '⏱ Total runtime'.")


def _run_pyverb(verb: str, arg: str) -> tuple[int, str]:
    import schwab_auth

    if verb == "run_signals":
        return _run_signals()

    if verb == "verbs_doc":
        return _write_verbs_doc()

    if verb == "seed":
        return _run_seed(arg)

    if verb in ("withdraw", "close"):
        import cash_reserve
        parts = arg.split()
        need = 3 if verb == "withdraw" else 2
        if len(parts) < need:
            return 2, (f"need ACCT TICKER{' AMOUNT' if need == 3 else ''}\n"
                       f"  e.g.  {verb}  171 GOOG"
                       + (" 5000" if need == 3 else "") + "\n\n"
                       + ("Releases fenced cash back to the account's free "
                          "pool. The position is untouched."
                          if verb == "withdraw" else
                          "Zeroes the reserve and deactivates the row. "
                          "History is kept."))
        acct, ticker = parts[0], parts[1]
        out = []
        if verb == "withdraw":
            try:
                amount = float(parts[2].replace(",", "").lstrip("$"))
            except ValueError:
                return 2, f"{parts[2]!r} is not an amount"
            bal = cash_reserve.withdraw(acct, ticker, amount,
                                        source="ops_sheet", log=out.append)
        else:
            bal = cash_reserve.close(acct, ticker, source="ops_sheet",
                                     log=out.append)
        out.append(f"\nreserve balance for {acct}/{ticker.upper()} "
                   f"is now ${bal:,.2f}")
        return 0, "\n".join(out)

    # BORROWING, from the sheet. Four verbs, one shape: ACCT TICKER AMOUNT.
    #
    #   borrow         171 MSTR 1000   reserve down, cash freed, Borrowed up
    #   return_borrow  171 MSTR 1000   reserve back up, Borrowed down
    #   owe            171 ALAB 1000   memo only — no cash moves
    #   paid_back      171 ALAB 1000   clears the memo
    #
    # Every one refuses to overdraw, so a typo'd amount is rejected rather
    # than quietly leaving a reserve or a debt that never existed.
    if verb in ("borrow", "return_borrow", "owe", "paid_back"):
        import cash_reserve
        parts = arg.split()
        if len(parts) < 3:
            return 2, (
                f"need ACCT TICKER AMOUNT\n  e.g.  {verb}  171 MSTR 1000\n\n"
                + {"borrow": "Takes cash OUT of that ticker's reserve. The "
                             "reserve falls and Free_To_Deploy rises, so you "
                             "can spend it elsewhere. Borrowed records what is "
                             "owed back.",
                   "return_borrow": "Puts borrowed cash back. The reserve rises "
                                    "and Borrowed falls. Refuses more than is "
                                    "actually out on loan.",
                   "owe": "A NOTE on the ticker you bought: this position was "
                          "funded with borrowed money. No cash moves.",
                   "paid_back": "Clears some or all of what a ticker owes. No "
                                "cash moves."}[verb])
        acct, ticker = parts[0], parts[1]
        try:
            amount = float(parts[2].replace(",", "").lstrip("$"))
        except ValueError:
            return 2, f"{parts[2]!r} is not an amount"

        out = []
        fn = getattr(cash_reserve, verb)
        try:
            fn(acct, ticker, amount, source="ops_sheet", log=out.append)
        except ValueError as e:
            # The guard messages already say what is wrong and by how much, so
            # pass them through rather than wrapping them in something vaguer.
            return 1, str(e)

        key = (acct, ticker)
        out.append(
            f"\n{acct}/{ticker.upper()}: reserve "
            f"${cash_reserve.balance_of(*key):,.2f} · borrowed "
            f"${cash_reserve.outstanding(*key, 'Borrowed'):,.2f} · owed "
            f"${cash_reserve.outstanding(*key, 'Owed'):,.2f}")
        out.append("Run run_signals to refresh the sheet columns.")
        return 0, "\n".join(out)

    # STICKY NOTES. Free text, so the argument is NOT split beyond the first
    # token — everything after the ticker is the note, spaces and all.
    if verb in ("note", "note_replace", "note_delete", "notes"):
        import ticker_notes

        if verb == "notes":
            return 0, ticker_notes.summary()

        parts = (arg or "").strip().split(None, 1)
        if not parts:
            return 2, (f"need TICKER{' TEXT' if verb != 'note_delete' else ''}\n"
                       f"  e.g.  {verb}  PBLS"
                       + ("  20 shares I never sell, only trade the rest"
                          if verb != "note_delete" else ""))
        ticker = parts[0]
        text = parts[1] if len(parts) > 1 else ""

        out = []
        try:
            if verb == "note":
                ticker_notes.add(ticker, text, source="ops_sheet",
                                 log=out.append)
            elif verb == "note_replace":
                ticker_notes.replace(ticker, text, source="ops_sheet",
                                     log=out.append)
            else:
                ticker_notes.delete(ticker, source="ops_sheet", log=out.append)
        except ValueError as e:
            return 1, str(e)

        out.append("\nRun run_signals to show it on the Dashboard.")
        return 0, "\n".join(out)

    if verb == "fence":
        import cash_reserve
        parts = arg.split()
        if len(parts) < 2:
            return 2, ("need ACCT TICKER\n  e.g.  fence  171 MSTR\n\n"
                       "Registers a ticker you already own. No cash moves now; "
                       "when you later trim or sell it the proceeds are held "
                       "for that ticker instead of becoming free cash.")
        out = []
        bal = cash_reserve.fence(parts[0], parts[1], source="ops_sheet",
                                 log=out.append)
        out.append(f"\nreserve balance for {parts[0]}/{parts[1].upper()} "
                   f"is ${bal:,.2f}")
        return 0, "\n".join(out)

    if verb == "apply_fills":
        import cash_reserve
        import jobStocksSignals as job
        out = []
        fills = cash_reserve.fetch_fills(job.get_schwab_client(), days_back=7,
                                         log=out.append)
        ev = cash_reserve.apply_fills(fills, log=out.append)
        if ev is not None and not ev.empty:
            out.append("")
            out.append(ev[["Account", "Ticker", "Event", "Amount", "Reason"]]
                       .to_string(index=False))
        return 0, "\n".join(out)

    if verb == "reserves":
        import cash_reserve
        d = cash_reserve.build_reserves_table(log=lambda *a, **k: None)
        return 0, (d.to_string(index=False) if not d.empty
                   else "(no reserves configured)")

    if verb == "token_status":
        return 0, json.dumps(schwab_auth.status(), indent=2)

    if verb == "auth_url":
        # ORDER MATTERS, AND IT IS NOT THE OBVIOUS ONE.
        #
        # The code Schwab hands back lives about THIRTY SECONDS, counted from
        # the moment you approve. Typing the verb after approving spends most
        # of that budget switching apps. So: arm the row first, approve
        # second, and the only thing left to do on the clock is one paste.
        # Pi 1 is already watching every 4s, so it fires the instant B lands.
        return 0, (
            "⏱  THE CODE DIES ~30 SECONDS AFTER YOU TAP APPROVE.\n"
            "So do step 1 BEFORE you open the link.\n\n"
            "1. Add a row now: column A = auth_code, leave B EMPTY.\n"
            "   (Nothing runs until B is filled, so it can sit there.)\n"
            "2. Open the link below, sign in, approve.\n"
            "   The redirect to 127.0.0.1 WILL fail to load — that is expected.\n"
            "3. Copy the ENTIRE address bar and paste it into that row's B.\n"
            "   Pi 1 picks it up within 4 seconds.\n\n"
            "If it fails, do NOT re-paste the same URL — that code is spent.\n"
            "Run auth_url again for a fresh one.\n\n"
            + schwab_auth.authorize_url()
        )

    if verb == "auth_code":
        # Under the job lock, so this can be done at ANY time. Installing new
        # tokens invalidates the old refresh token; a job mid-refresh would
        # otherwise fail, which used to mean watching the clock before
        # re-authorising.
        # install() takes the job lock itself, so every caller gets it —
        # the ops sheet, the CLI, anything else.
        # SAY WHAT ARRIVED, BEFORE TRYING TO SPEND IT.
        #
        # 2026-10-19: the sheet path failed twice with "Authorization code is
        # invalid, expired or revoked" while the SAME install() called from
        # Pi 1's shell worked. The authorize URL was byte-identical and only
        # two minutes had passed, so neither the URL nor the clock explains
        # it — which leaves what the cell actually contained. A phone paste
        # into Google Sheets can arrive shortened, linkified, or with the
        # query trimmed, and nothing here would have noticed.
        #
        # Redacted to a shape, never the code: length, first/last few chars,
        # and whether the pieces a redirect URL must have are present. A good
        # one measured 76 characters ending '@'.
        shape = []
        try:
            raw = (arg or "").strip()
            shape.append(f"pasted {len(raw)} chars")
            shape.append("has 'code='" if "code=" in raw else "NO 'code=' — "
                         "the cell is not a redirect URL")
            code = schwab_auth.extract_code(raw)
            shape.append(f"code {len(code)} chars, "
                         f"{code[:4]}…{code[-3:]}")
            if len(code) < 40:
                return 1, ("CODE LOOKS TRUNCATED — " + " · ".join(shape) +
                           "\n\nA whole Schwab code is ~76 characters and ends "
                           "'@'. Paste the ENTIRE address bar, not a shortened "
                           "or linkified version of it.")
        except Exception as e:
            return 2, (f"could not read a code out of column B: {e}\n\n"
                       + " · ".join(shape))

        try:
            return 0, schwab_auth.install(arg) + "\n\n[" + " · ".join(shape) + "]"
        except Exception as e:
            # The shape goes on the FAILURE too — that is the case we are
            # trying to diagnose.
            return 1, f"{type(e).__name__}: {e}\n\n[{' · '.join(shape)}]"

    return 2, f"unhandled python verb {verb!r}"


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def now_str() -> str:
    return datetime.now().astimezone().strftime(TS_FMT)


def safe_arg(verb: str, arg: str) -> str:
    """What may be written to the audit log in place of the raw argument.

    Subprocess verbs take a KEY into a fixed dict, so logging them verbatim
    is safe and useful. In-process verbs are the only ones that accept a
    free-form string, so an argument to any of them may carry a secret —
    an OAuth redirect URL today, something else tomorrow.

    So the default for an in-process verb is REDACT, and a verb must opt out
    explicitly by declaring sensitive=False. A malformed or half-written
    PY_VERBS entry is treated as sensitive too: the failure mode of redacting
    something harmless is an unhelpful log line, while the failure mode of the
    reverse is a credential in a file we keep forever.
    """
    entry = PY_VERBS.get(verb)
    if not arg or entry is None:      # no arg, or a subprocess verb
        return arg
    try:
        sensitive = bool(entry[1])
    except (TypeError, IndexError):   # someone wrote a bare bool, or a 1-tuple
        sensitive = True
    return "(redacted)" if sensitive else arg


def ensure_dirs() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def audit(**fields) -> None:
    """One JSON object per line. Append-only, never rotated by us."""
    fields.setdefault("ts", now_str())
    fields.setdefault("host", HOST)
    ensure_dirs()
    with AUDIT_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(fields, default=str) + "\n")


def spill(row: int, verb: str, text: str) -> Path:
    """Full output to disk so column I can stay short and readable."""
    ensure_dirs()
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    path = OUTPUT_DIR / f"{stamp}_r{row}_{verb}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def shrink(text: str, spill_path: Path) -> str:
    """Keep the TAIL — for logs and git output that is the useful end."""
    if len(text) <= SHEET_OUTPUT_MAX:
        return text
    keep = text[-SHEET_OUTPUT_MAX:]
    return f"[truncated · full output: {spill_path}]\n…\n{keep}"


# ---------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------
def run_verb(verb: str, arg: str) -> tuple[int, str]:
    if verb in PY_VERBS:
        if PY_VERBS[verb][0] and not arg:
            return 2, f"verb '{verb}' requires an argument in column B"
        return run_pyverb(verb, arg)

    builder, timeout, needs_arg = VERBS[verb]

    if needs_arg and not arg:
        return 2, f"verb '{verb}' requires an argument in column B"

    try:
        cmds = builder(arg)
    except ValueError as e:
        return 2, str(e)

    chunks, rc = [], 0
    for argv in cmds:
        chunks.append("$ " + " ".join(argv))
        try:
            p = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                # A missing repo would otherwise make every verb die with a
                # confusing FileNotFoundError about the cwd, not the command.
                cwd=str(REPO if REPO.is_dir() else HOME),
                # No shell. No user-supplied strings in argv.
            )
            out = (p.stdout or "") + (p.stderr or "")
            chunks.append(out.rstrip())
            if p.returncode != 0 and rc == 0:
                rc = p.returncode
        except subprocess.TimeoutExpired:
            chunks.append(f"!! timed out after {timeout}s")
            rc = rc or 124
            break
        except FileNotFoundError as e:
            chunks.append(f"!! not found: {e}")
            rc = rc or 127
            break

    return rc, "\n".join(c for c in chunks if c).strip() or "(no output)"


# ---------------------------------------------------------------------
# Sheet plumbing
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# The reference tab
# ---------------------------------------------------------------------
# DOCS LIVE NEXT TO THE CODE, NOT IN THE SHEET.
#
# The ops tab's own header row already carried a cramped sentence listing the
# verbs, and it went stale the moment one was added — nothing made it update.
# Here, a verb without an entry shows up as "(undocumented)" in the generated
# tab, so the gap is visible rather than silent.
#
# (verb, argument, example, what it does)
VERB_DOCS = [
    ("ping", "—", "ping", "Is Pi 1 alive and reading the sheet? Returns the time and hostname."),
    ("status", "—", "status", "Bot service state, uptime and disk in one shot."),
    ("uptime", "—", "uptime", "Load average and free memory. Rising load with nothing running is worth a look."),
    ("disk", "—", "disk", "Free space on every filesystem."),
    ("git_pull", "—", "git_pull", "Pull the latest code, then show the last 5 commits. Run this after any change is pushed."),
    ("git_status", "—", "git_status", "Uncommitted changes and the current branch."),
    ("git_log", "—", "git_log", "The last 15 commits."),
    ("cron_check", "—", "cron_check", "The full crontab — what is scheduled and when."),
    ("svc_log", "—", "svc_log", "Last 80 journal lines for the bot service. Can time out if the box is busy."),
    ("ls", "repo | jobs | bots", "ls  jobs", "Directory listing for one of the three repos."),
    ("tail_log", "stocks | crypto | 45 | ops", "tail_log  stocks", "Tail a job log. Use this to watch run_signals progress."),
    ("pip_upgrade", "package name", "pip_upgrade  yfinance", "Upgrade one allowlisted library in the venv (yfinance, curl_cffi, requests, pandas, gspread). Shows the version before and after."),

    ("token_status", "—", "token_status", "Schwab token state and days remaining."),
    ("auth_url", "—", "auth_url", "Start a Schwab re-auth. ⏱ THE CODE DIES ~30s AFTER YOU TAP APPROVE — add the auth_code row with column B EMPTY first, then approve, then paste."),
    ("auth_code", "the whole pasted redirect URL", "auth_code  https://127.0.0.1:8000/?code=C0.b...@&session=...", "Finishes the re-auth. Paste the ENTIRE address bar. A re-pasted code is refused — get a fresh auth_url instead."),

    ("run_signals", "—", "run_signals", "Rebuild Positions, Cash, Dashboard and the reserve tables now instead of waiting for the next cron. Returns as soon as it LAUNCHES; watch with tail_log stocks."),
    ("reserves", "—", "reserves", "The whole reserve table: what is fenced, deployed and available per ticker."),
    ("apply_fills", "—", "apply_fills", "Pull recent Schwab fills into the reserve ledger. Idempotent — already-seen fills are skipped."),

    ("seed", "ACCT TICKER AMOUNT", "seed  171 MSTR 5000", "Fence cash for a ticker. It leaves Free_To_Deploy and is held for that name."),
    ("fence", "ACCT TICKER", "fence  171 MSTR", "Register a ticker you already own. No cash moves now; future sale proceeds are held for it instead of becoming free cash."),
    ("withdraw", "ACCT TICKER AMOUNT", "withdraw  171 MSTR 1000", "Release fenced cash back to the free pool. The position is untouched."),
    ("close", "ACCT TICKER", "close  171 MSTR", "Zero the reserve and deactivate the row. History is kept."),

    ("borrow", "ACCT TICKER AMOUNT", "borrow  171 MSTR 1000", "Take cash OUT of that ticker's reserve so you can spend it elsewhere. Seed_Reserved falls, Free_To_Deploy rises, Borrowed records what is owed back. Refuses more than the reserve holds."),
    ("return_borrow", "ACCT TICKER AMOUNT", "return_borrow  171 MSTR 1000", "Put borrowed cash back. Seed_Reserved rises, Borrowed falls. Refuses more than is actually out on loan."),
    ("owe", "ACCT TICKER AMOUNT", "owe  171 ALAB 1000", "A NOTE on the ticker you bought with borrowed money. NO cash moves — this only fills the Owed column."),
    ("paid_back", "ACCT TICKER AMOUNT", "paid_back  171 ALAB 1000", "Clear some or all of what a ticker owes. NO cash moves."),

    ("note", "TICKER then free text", "note  PBLS 20 shares I never sell, only trade the rest", "APPEND a dated line to that ticker's note. Today's date is stamped on the new line; older lines keep their own dates."),
    ("note_replace", "TICKER then free text", "note_replace  PBLS now 25 core shares", "Throw away the existing note and start again. The old text stays in the log — nothing is ever erased."),
    ("note_delete", "TICKER", "note_delete  PBLS", "Clear a ticker's note. It stays in the archive log and can be read back."),
    ("notes", "—", "notes", "Every live note, one ticker per block, with the date each line was written."),
    ("verbs_doc", "—", "verbs_doc", "Rebuild this tab from the code. Run it after a new verb is added."),
]

DOC_TAB = os.getenv("GSHEET_VERBS_TAB", "VERBS-howToUse")


def _write_verbs_doc() -> tuple[int, str]:
    """(Re)build the VERBS-howToUse tab from VERB_DOCS and the verb tables."""
    import gspread
    from google.oauth2.service_account import Credentials

    creds = Credentials.from_service_account_file(CREDS_PATH, scopes=SCOPES)
    book = gspread.authorize(creds).open_by_key(os.environ["GSHEET_OPS_ID"])

    known = set(VERBS) | set(PY_VERBS)
    documented = {v for v, *_ in VERB_DOCS}
    # A verb nobody documented still gets a row. Silence would mean the tab
    # looks complete while missing exactly the thing you came to look up.
    extra = [(v, "?", v, "(undocumented — add it to VERB_DOCS in remote_ops.py)")
             for v in sorted(known - documented)]
    gone = sorted(documented - known - {"verbs_doc"})

    rows = [["VERB", "COLUMN B", "EXAMPLE", "WHAT IT DOES"]]
    rows += [list(r) for r in VERB_DOCS if r[0] in known or r[0] == "verbs_doc"]
    rows += [list(r) for r in extra]
    rows.append([])
    rows.append(["HOW TO RUN", "", "",
                 "Type the verb in column A of the ops tab and its argument "
                 "(if any) in column B. Leave C-J empty — Pi 1 fills them in. "
                 "Rows are picked up within a minute."])
    rows.append(["", "", "",
                 f"Generated from remote_ops.py on {now_str()} · "
                 f"{len(known)} verb(s) live. Re-run verbs_doc after a change."])

    try:
        ws = book.worksheet(DOC_TAB)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = book.add_worksheet(title=DOC_TAB, rows=max(len(rows) + 10, 60),
                                cols=4)

    ws.resize(rows=max(len(rows) + 10, 60), cols=4)
    ws.update(values=rows, range_name=f"A1:D{len(rows)}")
    with contextlib.suppress(Exception):
        ws.format("A1:D1", {"backgroundColor": {"red": 0.20, "green": 0.25,
                                                "blue": 0.35},
                            "textFormat": {"bold": True,
                                           "foregroundColor": {"red": 1.0,
                                                               "green": 1.0,
                                                               "blue": 1.0}}})
        ws.format(f"A2:A{len(rows)}", {"textFormat": {"bold": True}})
        ws.columns_auto_resize(0, 3)
        ws.freeze(rows=1)

    note = f"wrote {len(rows)} row(s) to '{DOC_TAB}'"
    if extra:
        note += f"\n⚠️  {len(extra)} undocumented verb(s): " \
                + ", ".join(v for v, *_ in extra)
    if gone:
        note += f"\n⚠️  documented but no longer a verb: {', '.join(gone)}"
    return 0, note


def open_sheet():
    from google.oauth2.service_account import Credentials
    import gspread

    sid = os.getenv("GSHEET_OPS_ID", "")
    tab = os.getenv("GSHEET_OPS_TAB", "ops")
    if not sid:
        sys.exit("GSHEET_OPS_ID is not set (put it in the systemd unit or .env)")
    if not Path(CREDS_PATH).exists():
        sys.exit(f"service-account key not found at {CREDS_PATH}")

    creds = Credentials.from_service_account_file(CREDS_PATH, scopes=SCOPES)
    return gspread.authorize(creds).open_by_key(sid).worksheet(tab)


def write_back(ws, row: int, values: dict) -> bool:
    """One API call per row update: C..J in a single range write.

    Returns True if the write landed. The caller MUST check this when claiming
    a row: in top-scan mode a non-empty Status is the only thing marking a row
    as handled, so an unwritten claim means the row is re-run on the next poll.
    For `seed` that would fence the money twice.
    """
    order = ["Status", "QueuedAt", "StartedAt", "FinishedAt",
             "Secs", "Exit", "Output", "Host"]
    payload = [[str(values.get(k, "")) for k in order]]
    for attempt in range(3):
        try:
            ws.update(values=payload, range_name=f"C{row}:J{row}")
            return True
        except Exception as e:                       # transient 429/500
            if attempt == 2:
                audit(event="writeback_failed", row=row, error=str(e))
                return False
            time.sleep(2 ** attempt)
    return False


def nudge(ws) -> int:
    """Append an auth_url result row when the refresh token is aging out."""
    import schwab_auth
    st = schwab_auth.status()
    if st.get("state") not in ("RENEW_NOW", "EXPIRED", "MISSING", "UNKNOWN"):
        return 0
    ws.append_row(
        ["", "", f"ACTION: {st['state']}", now_str(), now_str(), now_str(), 0, "-",
         f"Schwab refresh token: {st.get('days_left', '?')} days left.\n"
         f"Tap to re-authorize, then add an auth_code row.\n\n"
         + schwab_auth.authorize_url(), HOST],
        value_input_option="RAW")
    audit(event="nudge", state=st.get("state"), days_left=st.get("days_left"))
    return 1


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Allowlisted remote ops channel for Pi 1")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would run; execute nothing, write nothing")
    ap.add_argument("--verbs", action="store_true", help="print the allowlist and exit")
    ap.add_argument("--init", action="store_true", help="write the header row and exit")
    ap.add_argument("--nudge", action="store_true",
                    help="append a re-auth prompt row if the refresh token is aging out")
    args = ap.parse_args()

    if args.verbs:
        for name, (_, timeout, needs) in sorted(VERBS.items()):
            note = " <arg required>" if needs else ""
            print(f"{name:<12} timeout={timeout}s{note}")
        for name, (needs, secret) in sorted(PY_VERBS.items()):
            print(f"{name:<12} in-process"
                  f"{' <arg required>' if needs else ''}"
                  f"{' <arg redacted in audit log>' if secret else ''}")
        print(f"\nlog keys: {', '.join(sorted(LOGS))}")
        print(f"dir keys: {', '.join(sorted(DIRS))}")
        return 0

    ws = open_sheet()

    if args.init:
        ws.update(values=[HEADER], range_name="A1:J1")
        print("header written to row 1")
        return 0

    if args.nudge:
        print("nudge row appended" if nudge(ws) else "token healthy; no nudge")
        return 0

    rows = ws.get_all_values()
    last_sheet_row = len(rows)

    # ---- Top-scan window, adopted 2026-09-22 -----------------------------
    # Examine the top SCAN_WINDOW rows every poll. A row that already carries
    # a Status is SKIPPED, not a stopping point — anything unstamped below it
    # still runs.
    #
    # Stopping at the first stamped row was the obvious reading of "history
    # sinks down", and it was wrong: with MAX_ROWS_PER_RUN = 5, queueing six
    # commands ran five, and the next poll then hit row 2's OK and stopped —
    # the sixth never ran at all. Same if a run died partway through a batch.
    #
    # This replaced a stored row-number cursor. Inserting a row shifted every
    # number below it, so the cursor silently pointed at the wrong row and new
    # commands above it were never scanned. Nothing points anywhere now.
    #
    # What keeps a row from running twice is that it is CLAIMED before its
    # verb runs, and an unclaimable row is not run at all — see the gate below.
    executed = 0
    for r in range(START_ROW, min(last_sheet_row, START_ROW + SCAN_WINDOW - 1) + 1):
        cells = rows[r - 1] + [""] * (len(HEADER) - len(rows[r - 1]))
        verb = cells[0].strip()
        arg = cells[1].strip()
        status = cells[2].strip()

        if status:
            # Already handled. Skip it and keep looking — a stamped row above
            # must never hide an unstamped one below.
            continue

        if not verb:
            # Blank row above the boundary: spacing between queued commands,
            # or a row typed but not filled in yet. Skip it and keep looking.
            continue

        # A verb whose argument has not been typed yet. Leave the row ALONE —
        # unstamped, so it runs on a later poll once you finish typing. The
        # poller used to stamp FAIL here, burning a row you were mid-way
        # through entering (seen 2026-09-21 on `seed 171 MSTR 7499.84`).
        needs_arg = (PY_VERBS[verb][0] if verb in PY_VERBS
                     else VERBS[verb][2] if verb in VERBS else False)
        if needs_arg and not arg:
            # Skip, do not stop: a row you are still typing must not hold up
            # the ones below it. It stays unstamped and runs on a later poll.
            # The cost is that a later row may run first — acceptable, since
            # the verbs that need an argument (seed, tail_log, auth_code) do
            # not depend on the ones that do not.
            audit(event="awaiting_arg", row=r, verb=verb)
            print(f"row {r}: {verb} is waiting for an argument in column B")
            continue

        if executed >= MAX_ROWS_PER_RUN:
            audit(event="batch_cap", stopped_at=r, cap=MAX_ROWS_PER_RUN)
            break

        if args.dry_run:
            known = verb in VERBS or verb in PY_VERBS
            verdict = "WOULD RUN" if known else "WOULD REJECT"
            print(f"row {r}: {verdict}  {verb} {safe_arg(verb, arg)}".rstrip())
            executed += 1
            continue

        queued = now_str()

        if verb not in VERBS and verb not in PY_VERBS:
            write_back(ws, r, {
                "Status": "REJECTED", "QueuedAt": queued, "StartedAt": queued,
                "FinishedAt": queued, "Secs": 0, "Exit": "-",
                "Output": f"'{verb}' is not in the allowlist. Valid: "
                          f"{', '.join(sorted(set(VERBS) | set(PY_VERBS)))}",
                "Host": HOST,
            })
            audit(event="rejected", row=r, verb=verb, args=arg)
            executed += 1
            continue

        # Claim the row first, so a slow verb shows RUNNING on your phone
        # and a crash mid-flight leaves a visible marker rather than a
        # silently re-runnable blank.
        started = now_str()
        claimed = write_back(ws, r, {"Status": "RUNNING", "QueuedAt": queued,
                                     "StartedAt": started, "Host": HOST})
        if not claimed:
            # Could not mark the row. Since a non-empty Status is the ONLY
            # record that a row has been handled, running the verb now would
            # leave it looking untouched and re-runnable — a second `seed`
            # fences the money twice. Stop the batch and retry next poll.
            audit(event="claim_failed", row=r, verb=verb)
            print(f"row {r}: could not claim the row, ran nothing — will retry")
            break
        audit(event="start", row=r, verb=verb, args=safe_arg(verb, arg))

        t0 = time.monotonic()
        rc, output = run_verb(verb, arg)
        secs = round(time.monotonic() - t0, 1)

        path = spill(r, verb, output)
        write_back(ws, r, {
            "Status": "OK" if rc == 0 else "FAIL",
            "QueuedAt": queued,
            "StartedAt": started,
            "FinishedAt": now_str(),
            "Secs": secs,
            "Exit": rc,
            "Output": shrink(output, path),
            "Host": HOST,
        })

        # The authorization code is single-use and short-lived, but leaving
        # it sitting in a spreadsheet cell is pointless risk.
        if verb == "auth_code":
            try:
                ws.update(values=[["(consumed)"]], range_name=f"B{r}")
            except Exception as e:
                audit(event="redact_failed", row=r, error=str(e))

        audit(event="done", row=r, verb=verb, args=safe_arg(verb, arg), exit=rc,
              secs=secs, spill=str(path))

        executed += 1

        # AFTER auth_url, STAY AWAKE AND WATCH FOR THE CODE.
        #
        # The poller runs once a minute, so a pasted auth_code waited up to 60
        # seconds before Pi 1 even looked at the row. Schwab's authorization
        # code does not live that long: two were lost to exactly this
        # (2026-10-13, 2026-10-19), both reported as "Authorization code is
        # invalid, expired or revoked" after an exchange that itself took
        # 0.8s. The paste was never the problem; the WAIT was.
        #
        # Having just handed out a URL, we know a code is coming. Watching for
        # a few seconds at a time collapses the gap from 60s to ~4s. Bounded,
        # and only ever entered right after auth_url.
        #
        # Safe to hold the process open: the cron invocation takes remote_ops'
        # own lock with -n, so the next minute's run simply skips rather than
        # doubling up.
        if verb == "auth_url" and rc == 0:
            _watch_for_code(ws, deadline=time.monotonic() + AUTH_WATCH_SECS)

    if executed == 0:
        audit(event="idle", sheet_rows=last_sheet_row)
    return 0


def _watch_for_code(ws, deadline: float) -> None:
    """Poll the sheet every few seconds for an armed auth_code row.

    Runs the verb the moment one appears rather than waiting for the next
    cron tick. Never raises: a failure here must not lose the auth_url result
    that has already been written back.
    """
    print(f"watching for auth_code for {AUTH_WATCH_SECS}s "
          f"(every {AUTH_WATCH_EVERY}s) — paste the URL and it goes straight "
          f"through")
    audit(event="auth_watch_start", secs=AUTH_WATCH_SECS)
    while time.monotonic() < deadline:
        time.sleep(AUTH_WATCH_EVERY)
        try:
            rows = ws.get_all_values()
        except Exception as e:
            audit(event="auth_watch_read_failed", error=str(e))
            continue
        for r in range(START_ROW,
                       min(len(rows), START_ROW + SCAN_WINDOW - 1) + 1):
            cells = rows[r - 1] + [""] * (len(HEADER) - len(rows[r - 1]))
            if cells[0].strip() != "auth_code" or cells[2].strip():
                continue                      # not it, or already handled
            arg = cells[1].strip()
            if not arg:
                continue                      # still being typed
            started = now_str()
            if not write_back(ws, r, {"Status": "RUNNING", "QueuedAt": started,
                                      "StartedAt": started, "Host": HOST}):
                audit(event="auth_watch_claim_failed", row=r)
                return
            t0 = time.monotonic()
            rc, output = run_verb("auth_code", arg)
            secs = round(time.monotonic() - t0, 1)
            path = spill(r, "auth_code", output)
            write_back(ws, r, {
                "Status": "OK" if rc == 0 else "FAIL", "QueuedAt": started,
                "StartedAt": started, "FinishedAt": now_str(), "Secs": secs,
                "Exit": rc, "Output": shrink(output, path), "Host": HOST})
            try:
                ws.update(values=[["(consumed)"]], range_name=f"B{r}")
            except Exception as e:
                audit(event="redact_failed", row=r, error=str(e))
            audit(event="auth_watch_done", row=r, exit=rc, secs=secs)
            print(f"row {r}: auth_code handled {secs}s after it appeared")
            return
    audit(event="auth_watch_timeout", secs=AUTH_WATCH_SECS)
    print("no auth_code appeared — back to the normal once-a-minute poll")


if __name__ == "__main__":
    sys.exit(main())