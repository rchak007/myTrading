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
    "reserves":     (False, False),
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


def _run_pyverb(verb: str, arg: str) -> tuple[int, str]:
    import schwab_auth

    if verb == "seed":
        return _run_seed(arg)

    if verb == "reserves":
        import cash_reserve
        d = cash_reserve.build_reserves_table(log=lambda *a, **k: None)
        return 0, (d.to_string(index=False) if not d.empty
                   else "(no reserves configured)")

    if verb == "token_status":
        return 0, json.dumps(schwab_auth.status(), indent=2)

    if verb == "auth_url":
        return 0, (
            "Open this on your phone, sign in, approve.\n"
            "The redirect to 127.0.0.1 WILL fail to load — that is expected.\n"
            "Copy the entire address bar, then add a row:\n"
            "    A: auth_code    B: <the whole pasted URL>\n\n"
            + schwab_auth.authorize_url()
        )

    if verb == "auth_code":
        try:
            return 0, schwab_auth.install(arg)
        except Exception as e:
            return 1, f"{type(e).__name__}: {e}"

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

    if executed == 0:
        audit(event="idle", sheet_rows=last_sheet_row)
    return 0


if __name__ == "__main__":
    sys.exit(main())