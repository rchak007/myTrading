#!/usr/bin/env python3
"""
Monthly rotation of the bot's output/log files into a NON-git archive.

Designed to be called FROM bot.py (in-process) at the top of the main loop,
so the bot — the sole writer of these files — is the only thing that ever
touches them. No cron, no external process, no race.

What it does on a month rollover
--------------------------------
1. Every file in the MIRROR dirs (the git-tracked jobMyTrading/outputs tree)
   is MOVED into <archive_root>, preserving the outputs[/bot] structure, with
   the just-finished month appended before the extension:
        bot_errors_WETH_4h.log   ->  bot_errors_WETH_4h-May2026.log
        bot_errors_MASTER.csv    ->  bot_errors_MASTER-May2026.csv
2. The matching PRIMARY files (the source the bot appends to) are deleted,
   so the next mirror copy starts small instead of refilling 29 MB.
3. State files (any name containing "state") are NEVER rotated — they hold
   live regime / last-bar info and MUST survive the month boundary, otherwise
   a restart right after rotation could lose a bot's position tracking.

Safety
------
- Fires at most once per month, guarded by a marker file.
- Never rotates state files or the marker itself.
- The caller wraps this in try/except: archiving must never break trading.
"""
import os
import shutil
from datetime import datetime
from pathlib import Path

STATE_HINT  = "state"               # filenames containing this are left alone
MARKER_NAME = ".last_archive_month"


def _is_rotatable(name: str) -> bool:
    low = name.lower()
    if low.startswith("."):
        return False
    if STATE_HINT in low:           # bot_state_*.json, jupbot_state.json
        return False
    return True


def _outputs_tail(d: Path) -> Path:
    """Path tail starting at 'outputs' (e.g. outputs, outputs/bot)."""
    parts = d.parts
    if "outputs" in parts:
        return Path(*parts[parts.index("outputs"):])
    return Path(d.name)


def _stamp(name: str, label: str) -> str:
    base, ext = os.path.splitext(name)
    return f"{base}-{label}{ext}"


def archive_if_new_month(*, mirror_dirs, primary_dirs, archive_root, logger=None):
    def say(msg):
        (logger.info if logger else print)(msg)

    archive_root = Path(os.path.expanduser(archive_root))
    archive_root.mkdir(parents=True, exist_ok=True)
    marker = archive_root / MARKER_NAME

    now_key = datetime.now().strftime("%Y-%m")

    # First ever run: just record the month, don't surprise-move anything.
    if not marker.exists():
        marker.write_text(now_key)
        say(f"[archive] initialised marker at {now_key} — no rotation on first run.")
        return

    last_key = marker.read_text().strip()
    if last_key == now_key:
        return  # same month — nothing to do (cheap path, runs every loop)

    # Roll over: label files with the month they actually belong to.
    try:
        label = datetime.strptime(last_key, "%Y-%m").strftime("%b%Y")   # May2026
    except ValueError:
        label = last_key

    say(f"[archive] month rollover {last_key} -> {now_key}; archiving as '{label}'.")
    archived_names = set()

    # 1. Move the git-tracked mirror files into the archive.
    for raw in mirror_dirs:
        d = Path(raw)
        if not d.is_dir():
            continue
        dest_dir = archive_root / _outputs_tail(d)
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            if not f.is_file() or not _is_rotatable(f.name):
                continue
            dest = dest_dir / _stamp(f.name, label)
            try:
                if dest.exists():
                    f.unlink()                 # twin already archived this run
                else:
                    shutil.move(str(f), str(dest))
                archived_names.add(f.name)
            except Exception as e:
                say(f"[archive] failed to move {f}: {e}")

    # 2. Reset the primary sources (any depth) so mirrors start small again.
    for raw in primary_dirs:
        d = Path(raw)
        if not d.is_dir():
            continue
        for f in d.rglob("*"):
            if f.is_file() and f.name in archived_names:
                try:
                    f.unlink()
                except Exception as e:
                    say(f"[archive] failed to reset primary {f}: {e}")

    marker.write_text(now_key)
    say(f"[archive] done — {len(archived_names)} file name(s) rotated to {archive_root}.")