#!/usr/bin/env python3
"""
pi2_ops.py — the ops sheet's verbs that must run on PI 2, not Pi 1
===================================================================

The ops sheet is Pi 1's channel: Pi 1 polls it and writes every result back.
A few things only make sense on Pi 2 — today, waking the Claude Code sessions
that live here in tmux. Chakravarti, 2026-11-26: *"this pi2 itself can run
instead of me getting my dell laptop and running on terminal doing ssh"*.

HOW IT WORKS, AND WHY THIS SHAPE
  Pi 2's sheet identity is a VIEWER — it can read the ops sheet but never
  write it (that is a deliberate boundary, see PROJECT_PLAN §3). So:

    1. You insert a row at the top:  A = wakeup_claude
    2. Pi 1 stamps it (QueuedAt etc.) with "handed to Pi 2".
    3. This script, every 2 min from Pi 2's cron, sees a stamped
       wakeup_claude row it has not handled, runs ~/start-claude-sessions.sh,
       and EMAILS the result — the sheet row cannot carry it.

  Pi 1's QueuedAt stamp is the request's identity: unique per row, written
  once, and present whether Pi 1 accepted the verb or (not yet pulled)
  rejected it — so this works even before Pi 1 has the new code.

SAFETY
  - The sheet supplies a verb NAME only. Column B is ignored; nothing from
    the sheet reaches a shell or an argv.
  - Only rows queued in the last FRESH_MINUTES count, so the first run never
    replays history, and a request left lying around cannot fire days later.
  - One run per request, recorded in a local state file before running.
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOME = Path.home()
STATE_DIR = HOME / ".local" / "state" / "myTrading"
STATE = STATE_DIR / "pi2_ops_done.json"
LOCK = STATE_DIR / "pi2_ops.lock"
LOG = STATE_DIR / "pi2_ops.log"

SCAN_ROWS = 25            # same window Pi 1 scans
FRESH_MINUTES = 30
KEEP = 200                # handled-request ids remembered

WAKE_SCRIPT = HOME / "start-claude-sessions.sh"

# verb -> (argv, timeout seconds). Fixed here; the sheet only names the key.
PI2_VERBS = {
    "wakeup_claude": ([str(WAKE_SCRIPT)], 300),
}


def log(msg: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line)
    with LOG.open("a") as f:
        f.write(line + "\n")


def _env() -> dict:
    env = {}
    try:
        for line in (HERE / ".env").read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.strip().split("=", 1)
                env[k.strip()] = v.strip().strip('"\'')
    except OSError:
        pass
    return env


def read_rows() -> list[dict]:
    import gspread
    env = _env()
    creds = os.path.expanduser(env.get("GSHEET_READER_CREDS")
                               or "~/.config/myTrading/gsheets-reader.json")
    gc = gspread.service_account(filename=creds)
    sh = gc.open_by_key(env["GSHEET_OPS_ID"])
    ws = sh.worksheet(env.get("GSHEET_OPS_TAB") or "ops")
    values = ws.get_values(f"A1:J{SCAN_ROWS + 1}")
    if not values:
        return []
    head = [h.strip() for h in values[0]]
    out = []
    for i, row in enumerate(values[1:], start=2):
        row = row + [""] * (len(head) - len(row))
        out.append({"row": i, **dict(zip(head, row))})
    return out


def _parse_ts(s: str):
    # Pi 1 writes "%Y-%m-%d %H:%M:%S %Z"; drop the zone name (both Pis are
    # on the same local time) rather than depend on strptime knowing it.
    try:
        return datetime.strptime(" ".join(s.split()[:2]), "%Y-%m-%d %H:%M:%S")
    except (ValueError, IndexError):
        return None


def load_done() -> list[str]:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return []


def save_done(done: list[str]) -> None:
    STATE.write_text(json.dumps(done[-KEEP:]))


def pending(rows: list[dict], done: list[str], now: datetime) -> list[dict]:
    """Stamped, fresh, Pi-2 verb rows not yet handled — oldest first."""
    hits = []
    for r in rows:
        verb = r.get("Verb", "").strip().lower()
        queued = r.get("QueuedAt", "").strip()
        if verb not in PI2_VERBS or not queued:
            continue
        key = f"{verb}|{queued}"
        ts = _parse_ts(queued)
        if key in done or ts is None:
            continue
        if now - ts > timedelta(minutes=FRESH_MINUTES):
            continue
        hits.append({**r, "_key": key, "_verb": verb})
    return list(reversed(hits))


def run(verb: str) -> tuple[int, str]:
    argv, timeout = PI2_VERBS[verb]
    env = dict(os.environ)
    # Cron's PATH has no ~/.local/bin, where `claude` lives. If this run is
    # the one that starts the tmux server, every pane inherits this PATH.
    env["PATH"] = f"{HOME}/.local/bin:{HOME}/bin:{env.get('PATH', '/usr/bin:/bin')}"
    try:
        p = subprocess.run(argv, capture_output=True, text=True,
                           timeout=timeout, env=env, cwd=HOME)
        return p.returncode, (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"
    except OSError as e:
        return 127, f"could not run {argv[0]}: {e}"


def main() -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0                      # a wake-up is still running
        try:
            rows = read_rows()
        except Exception as e:            # noqa: BLE001
            log(f"cannot read ops sheet: {type(e).__name__}: {e}")
            return 1

        done = load_done()
        todo = pending(rows, done, datetime.now())
        for r in todo:
            # Record FIRST: a crash mid-run must not re-fire it every 2 min.
            done.append(r["_key"])
            save_done(done)
            log(f"row {r['row']} {r['_verb']} (queued {r['QueuedAt']}) — running")
            rc, out = run(r["_verb"])
            log(f"row {r['row']} {r['_verb']} exit {rc}\n{out}")
            try:
                from token_watch import send
                status = "OK" if rc == 0 else f"exit {rc}"
                send(f"{r['_verb']} on Pi 2 — {status}",
                     f"Ops sheet row {r['row']}, queued {r['QueuedAt']}.\n"
                     f"Ran {PI2_VERBS[r['_verb']][0][0]} on Pi 2.\n\n{out}",
                     log=log)
            except Exception as e:        # noqa: BLE001
                log(f"email failed: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
