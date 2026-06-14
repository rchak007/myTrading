#!/usr/bin/env python3
"""
Standalone git pusher for the jobMyTrading repo.

Runs every 5 minutes from cron, under the SAME flock the signal jobs use
(/tmp/jobmytrading.lock) so it can never commit a half-written file. This is
the SINGLE git writer in the system now — the signal jobs only write files.

It self-heals a stale .git/index.lock, rebases on the remote before pushing,
and retries transient push failures. The log lives OUTSIDE the repo on purpose
so that logging never itself creates a change that needs pushing.
"""
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO      = Path.home() / "github" / "jobMyTrading"
LOG_FILE  = Path.home() / "gitpush.log"                 # outside the repo
FAIL_MARK = Path.home() / ".jobmytrading_push_failed"   # visibility marker
LOCK      = REPO / ".git" / "index.lock"
MAX_TRIES = 2


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with LOG_FILE.open("a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def run(cmd, cwd=REPO):
    p = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout.strip()


def git_running() -> bool:
    code, out = run(["pgrep", "-x", "git"], cwd=Path.home())
    return code == 0 and bool(out)


def clear_stale_lock() -> None:
    if not LOCK.exists():
        return
    if git_running():
        log("index.lock present and a git process is running — leaving it.")
        return
    # We hold the flock and no git process exists, so the lock is stale.
    try:
        age = time.time() - LOCK.stat().st_mtime
        LOCK.unlink()
        log(f"Removed stale index.lock (age {age:.0f}s).")
    except FileNotFoundError:
        pass


def has_changes() -> bool:
    code, out = run(["git", "status", "--porcelain"])
    if code != 0:
        log(f"git status failed:\n{out}")
        return False
    return bool(out)


def main() -> None:
    if not (REPO / ".git").exists():
        log("jobMyTrading is not a git repo — aborting.")
        return

    clear_stale_lock()

    if not has_changes():
        return  # nothing to do — stay quiet

    code, out = run(["git", "add", "-A"])
    if code != 0:
        log(f"git add failed:\n{out}")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    code, out = run(["git", "commit", "-m", f"auto snapshot {stamp}"])
    if code != 0:
        log(f"git commit failed:\n{out}")
        return

    for attempt in range(1, MAX_TRIES + 1):
        code, out = run(["git", "pull", "--rebase", "origin", "main"])
        if code != 0:
            log(f"pull --rebase failed (try {attempt}):\n{out}")
            run(["git", "rebase", "--abort"])
            time.sleep(3)
            continue
        code, out = run(["git", "push", "origin", "main"])
        if code == 0:
            if FAIL_MARK.exists():
                FAIL_MARK.unlink()
            log("Pushed to GitHub ✅")
            return
        log(f"push failed (try {attempt}):\n{out}")
        time.sleep(3)

    FAIL_MARK.write_text(stamp)
    log("Push failed after retries — wrote marker, will retry next tick.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {repr(e)}")
        raise