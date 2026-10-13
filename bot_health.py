#!/usr/bin/env python3
"""
bot_health.py
=============
Are the crypto bots alive, and can they actually trade?

    .venv/bin/python bot_health.py             # report
    .venv/bin/python bot_health.py --email     # ...and mail it if anything is wrong
    .venv/bin/python bot_health.py --quiet     # only speak when something is wrong

RUNS ON PI 2, WHICH RUNS NO BOTS. That is the point. Pi 1 runs them and
pushes outputs/ to the botsMyTrading mirror; Pi 2 has read-only access to that
mirror AND the Gmail credentials. Same arrangement as the Orders sheet: the
machine doing the work cannot tell anyone, so the machine that can watches its
footprints instead. A checker running on Pi 1 would die with the thing it is
meant to be checking.

TWO DIFFERENT ALARMS, AND CONFLATING THEM WOULD BE THE BUG
    MIRROR STALE    Pi 1 has stopped pushing. Every heartbeat then looks
                    frozen whether the bots are fine or not, so this is
                    checked FIRST and reported on its own. Measured against
                    the wall clock.
    BOT STALE       The mirror is current but one bot's heartbeat is not —
                    that bot specifically is wedged. Measured against the
                    SNAPSHOT, never the wall clock, or every bot looks dead
                    the moment a push is late.

ALIVE IS NOT THE SAME AS WORKING
    On 2026-10-13 all 55 heartbeats were healthy and ten bots could not trade
    a thing: dry gas wallets on ethereum and bnb, two tickers yfinance has no
    data for, and one Base pair with no route. A heartbeat only proves the
    loop is turning. The error log is where "it ran and achieved nothing"
    shows up, so both are read.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.getenv("BOTS_MIRROR_REPO", Path.home() / "github" / "botsMyTrading"))
HB_DIR = "outputs/bot"
ERR_FILE = "outputs/bot_errors_MASTER.csv"

# A full loop is (bots x ~10s) + per-tick work; 55 bots measured ~33 min on
# 2026-10-13. Generous so an ordinary slow cycle is not an alarm.
BOT_STALE_MIN = float(os.getenv("BOT_STALE_MIN", "90"))
# The snapshot cron pushes periodically. Well past that and Pi 1 has stopped.
MIRROR_STALE_MIN = float(os.getenv("MIRROR_STALE_MIN", "180"))

# Failures grouped by what you would actually DO about them. The raw log is
# thousands of near-identical rows; the useful question is "which problems do
# I have", not "how many lines".
CAUSES = [
    ("DRY GAS WALLET", r"[Ii]nsufficient funds (to pay for gas|for gas)",
     "fund the chain's gas wallet — the bot retries forever and each attempt "
     "costs RPC credits"),
    ("NO PRICE DATA", r"No yfinance data",
     "the ticker does not exist upstream; this bot can never trade — fix the "
     "symbol or retire it"),
    ("NO LIQUIDITY", r"no liquidity available",
     "the router cannot price this pair"),
    ("RPC UNREACHABLE", r"Cannot connect to .* RPC",
     "the RPC endpoint is refusing — check the key and the quota"),
    ("TX TIMED OUT", r"is not in the chain after",
     "submitted but never mined; usually gas priced too low"),
]


def _git(*args, cwd=REPO) -> str:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                          text=True, timeout=300).stdout.strip()


def fetch(log=print) -> bool:
    """Refresh the mirror. Read-only remote; nothing local is modified."""
    if not (REPO / ".git").exists():
        log(f"⚠️  no mirror clone at {REPO}")
        return False
    try:
        subprocess.run(["git", "fetch", "--quiet", "origin"], cwd=str(REPO),
                       capture_output=True, text=True, timeout=300, check=True)
        return True
    except Exception as e:
        # Report it rather than silently reading a stale clone — that is how a
        # dead fleet looks healthy.
        log(f"⚠️  could not fetch the mirror: {type(e).__name__}: {e}")
        return False


def snapshot_time() -> datetime | None:
    raw = _git("log", "-1", "--format=%cd", "--date=format:%Y-%m-%d %H:%M:%S",
               "origin/main")
    try:
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _export(subdir: str) -> Path | None:
    """One `git archive` instead of a `git show` per file — 55 subprocesses
    was most of the runtime."""
    tmp = Path(tempfile.mkdtemp())
    try:
        tar = subprocess.run(["git", "archive", "origin/main", subdir],
                             cwd=str(REPO), capture_output=True, timeout=300)
        if tar.returncode != 0:
            return None
        subprocess.run(["tar", "-x", "-C", str(tmp)], input=tar.stdout,
                       capture_output=True, timeout=120)
        return tmp
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        return None


def heartbeats(log=print) -> list[dict]:
    """[{bot, chain, last, age_min}] newest-last, from the mirror."""
    root = _export(HB_DIR)
    if root is None:
        log("⚠️  could not read the heartbeat directory from the mirror")
        return []
    snap = snapshot_time()
    out = []
    try:
        for p in sorted((root / HB_DIR).glob("bot_heartbeat_*.log")):
            bot = p.name[len("bot_heartbeat_"):-len(".log")]
            try:
                lines = [ln for ln in p.read_text(errors="replace").splitlines()
                         if ln.strip()]
            except Exception:
                lines = []
            if not lines:
                out.append({"bot": bot, "chain": "", "last": None,
                            "age_min": None, "why": "heartbeat file is empty"})
                continue
            f = lines[-1].split(",")
            try:
                ts = datetime.strptime(f[1].strip(), "%Y-%m-%d %H:%M:%S")
            except Exception:
                out.append({"bot": bot, "chain": "", "last": None,
                            "age_min": None, "why": "unparseable timestamp"})
                continue
            out.append({"bot": bot, "chain": f[3].strip() if len(f) > 3 else "",
                        "last": ts, "why": "",
                        "age_min": ((snap - ts).total_seconds() / 60
                                    if snap else None)})
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return out


def errors_on(day: str, log=print) -> dict:
    """{cause: {(chain, bot): count}} for one YYYY-MM-DD."""
    root = _export(ERR_FILE)
    if root is None:
        log("⚠️  could not read the error log from the mirror")
        return {}
    grouped: dict = defaultdict(lambda: defaultdict(int))
    try:
        with (root / ERR_FILE).open(newline="", encoding="utf-8",
                                    errors="replace") as fh:
            for r in csv.reader(fh):
                if len(r) < 7 or not r[0].startswith(day):
                    continue
                bot, chain, msg = r[1], r[3], r[6]
                for name, pat, _fix in CAUSES:
                    if re.search(pat, msg):
                        grouped[name][(chain, bot)] += 1
                        break
                else:
                    grouped["OTHER"][(chain, bot)] += 1
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return grouped


def report(day: str | None = None, log=print) -> tuple[list[str], bool]:
    """(lines, anything_wrong)."""
    day = day or datetime.now().strftime("%Y-%m-%d")
    out, bad = [], False

    snap = snapshot_time()
    if snap is None:
        return ["⛔ could not read the mirror's snapshot time — "
                "treating as BROKEN rather than healthy"], True
    mirror_age = (datetime.now() - snap).total_seconds() / 60
    out.append(f"mirror last pushed {snap:%Y-%m-%d %H:%M} "
               f"({mirror_age:.0f} min ago)")
    if mirror_age > MIRROR_STALE_MIN:
        bad = True
        out.append(f"⛔ MIRROR STALE — Pi 1 has not pushed in "
                   f"{mirror_age / 60:.1f}h. Every heartbeat below is frozen "
                   f"at that moment, so NONE of it can be trusted. Check Pi 1 "
                   f"and the snapshot cron before reading on.")
        return out, bad

    hb = heartbeats(log=log)
    if not hb:
        return out + ["⛔ no heartbeats found in the mirror"], True
    stale = [h for h in hb if h["age_min"] is None or h["age_min"] > BOT_STALE_MIN]
    out.append(f"{len(hb)} bot(s) reporting · "
               f"{len(hb) - len(stale)} fresh · {len(stale)} stale")
    if stale:
        bad = True
        out.append("")
        out.append(f"⛔ STALE HEARTBEATS (> {BOT_STALE_MIN:.0f} min behind "
                   f"the snapshot) — these bots are not cycling:")
        for h in sorted(stale, key=lambda x: -(x["age_min"] or 1e9)):
            age = (f"{h['age_min']:.0f} min" if h["age_min"] is not None
                   else h["why"])
            out.append(f"    {h['bot']:<18}{h['chain']:<10}{age}")
    else:
        oldest = max(h["age_min"] for h in hb)
        out.append(f"    all within {oldest:.0f} min — one full loop, healthy")

    grouped = errors_on(day, log=log)
    out.append("")
    if not grouped:
        out.append(f"no errors logged on {day}")
        return out, bad

    total = sum(sum(v.values()) for v in grouped.values())
    out.append(f"ERRORS ON {day} — {total} line(s). Alive is not the same as "
               f"working:")
    fixes = {n: f for n, _p, f in CAUSES}
    for cause in sorted(grouped, key=lambda c: -sum(grouped[c].values())):
        per = grouped[cause]
        out.append("")
        out.append(f"  {cause}  ({sum(per.values())} line(s), "
                   f"{len(per)} bot(s))")
        if cause in fixes:
            out.append(f"    → {fixes[cause]}")
        for (chain, bot), n in sorted(per.items(), key=lambda kv: -kv[1])[:12]:
            out.append(f"      {bot:<18}{chain:<10}{n:>5}")
        if cause != "OTHER":
            bad = True
    return out, bad


def main() -> int:
    ap = argparse.ArgumentParser(description="Crypto bot health, from the mirror")
    ap.add_argument("--email", action="store_true",
                    help="mail the report when something is wrong")
    ap.add_argument("--quiet", action="store_true",
                    help="print nothing when everything is fine")
    ap.add_argument("--day", help="YYYY-MM-DD for the error scan (default today)")
    ap.add_argument("--no-fetch", action="store_true",
                    help="read the clone as-is, do not refresh it")
    args = ap.parse_args()

    if not args.no_fetch:
        fetch()
    lines, bad = report(args.day)
    body = "\n".join(lines)

    if not (args.quiet and not bad):
        print(body)

    if args.email and bad:
        sys.path.insert(0, str(HERE))
        from token_watch import send
        n = sum(1 for ln in lines if ln.startswith("⛔"))
        subject = ("🔴 Crypto bots: "
                   + ("mirror stale — Pi 1 may be down" if "MIRROR STALE" in body
                      else f"{n} problem area(s)" if n
                      else "bots alive but not trading"))
        send(subject, body + "\n\n— bot_health.py on Pi 2\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
