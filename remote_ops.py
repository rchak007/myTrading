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
# COLD START
#   With no state file, the first run ADOPTS the current bottom of the
#   sheet and executes nothing. That stops a fresh install from
#   replaying months of history. Use --catchup to override once.
#
# SHEET LAYOUT (row 1 is the header, written by --init)
#   A Verb  B Args  C Status  D QueuedAt  E StartedAt  F FinishedAt
#   G Secs  H Exit  I Output  J Host
#
# You fill A and B. Pi 1 fills C..J. Rows are APPEND-ONLY — never
# delete a row, or the row numbers shift and the cursor skips work.
# Archive by copying to another tab.
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
STATE_FILE = STATE_DIR / "remote_ops.state"
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


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def now_str() -> str:
    return datetime.now().astimezone().strftime(TS_FMT)


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


def read_cursor() -> int | None:
    if not STATE_FILE.exists():
        return None
    try:
        return int(STATE_FILE.read_text().strip())
    except (ValueError, OSError):
        return None


def write_cursor(row: int) -> None:
    ensure_dirs()
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(str(row))
    tmp.replace(STATE_FILE)          # atomic; no half-written cursor


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


def write_back(ws, row: int, values: dict) -> None:
    """One API call per row update: C..J in a single range write."""
    order = ["Status", "QueuedAt", "StartedAt", "FinishedAt",
             "Secs", "Exit", "Output", "Host"]
    payload = [[str(values.get(k, "")) for k in order]]
    for attempt in range(3):
        try:
            ws.update(values=payload, range_name=f"C{row}:J{row}")
            return
        except Exception as e:                       # transient 429/500
            if attempt == 2:
                audit(event="writeback_failed", row=row, error=str(e))
                return
            time.sleep(2 ** attempt)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Allowlisted remote ops channel for Pi 1")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would run; execute nothing, write nothing")
    ap.add_argument("--catchup", action="store_true",
                    help="on cold start, actually run the pending backlog")
    ap.add_argument("--verbs", action="store_true", help="print the allowlist and exit")
    ap.add_argument("--init", action="store_true", help="write the header row and exit")
    ap.add_argument("--reset-cursor", type=int, metavar="ROW",
                    help="force the cursor to ROW (rows at or below it are ignored)")
    args = ap.parse_args()

    if args.verbs:
        for name, (_, timeout, needs) in sorted(VERBS.items()):
            note = " <arg required>" if needs else ""
            print(f"{name:<12} timeout={timeout}s{note}")
        print(f"\nlog keys: {', '.join(sorted(LOGS))}")
        print(f"dir keys: {', '.join(sorted(DIRS))}")
        return 0

    if args.reset_cursor is not None:
        write_cursor(args.reset_cursor)
        audit(event="cursor_reset", row=args.reset_cursor)
        print(f"cursor set to row {args.reset_cursor}")
        return 0

    ws = open_sheet()

    if args.init:
        ws.update(values=[HEADER], range_name="A1:J1")
        print("header written to row 1")
        return 0

    rows = ws.get_all_values()
    last_sheet_row = len(rows)

    cursor = read_cursor()
    if cursor is None:
        if not args.catchup:
            write_cursor(last_sheet_row)
            audit(event="cold_start_adopt", row=last_sheet_row)
            print(f"cold start: adopted row {last_sheet_row}, executed nothing")
            return 0
        cursor = 1                                   # row 1 is the header

    executed = 0
    for r in range(max(cursor + 1, 2), last_sheet_row + 1):
        cells = rows[r - 1] + [""] * (len(HEADER) - len(rows[r - 1]))
        verb = cells[0].strip()
        arg = cells[1].strip()
        status = cells[2].strip()

        if not verb:                                 # blank spacer row
            write_cursor(r)
            continue

        if status:                                   # already handled
            write_cursor(r)
            continue

        if executed >= MAX_ROWS_PER_RUN:
            audit(event="batch_cap", stopped_at=r, cap=MAX_ROWS_PER_RUN)
            break                                    # cursor NOT advanced

        if args.dry_run:
            verdict = "WOULD RUN" if verb in VERBS else "WOULD REJECT"
            print(f"row {r}: {verdict}  {verb} {arg}".rstrip())
            executed += 1
            continue

        queued = now_str()

        if verb not in VERBS:
            write_back(ws, r, {
                "Status": "REJECTED", "QueuedAt": queued, "StartedAt": queued,
                "FinishedAt": queued, "Secs": 0, "Exit": "-",
                "Output": f"'{verb}' is not in the allowlist. Valid: "
                          f"{', '.join(sorted(VERBS))}",
                "Host": HOST,
            })
            audit(event="rejected", row=r, verb=verb, args=arg)
            write_cursor(r)
            executed += 1
            continue

        # Claim the row first, so a slow verb shows RUNNING on your phone
        # and a crash mid-flight leaves a visible marker rather than a
        # silently re-runnable blank.
        started = now_str()
        write_back(ws, r, {"Status": "RUNNING", "QueuedAt": queued,
                           "StartedAt": started, "Host": HOST})
        audit(event="start", row=r, verb=verb, args=arg)

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
        audit(event="done", row=r, verb=verb, args=arg, exit=rc,
              secs=secs, spill=str(path))

        write_cursor(r)
        executed += 1

    if executed == 0:
        audit(event="idle", cursor=read_cursor(), sheet_rows=last_sheet_row)
    return 0


if __name__ == "__main__":
    sys.exit(main())