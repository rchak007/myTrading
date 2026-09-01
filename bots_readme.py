#!/usr/bin/env python3
"""
Generate README.md for the botsMyTrading repo from the bot heartbeat logs.

Standalone on purpose: run from cron immediately BEFORE the botsMyTrading git
push, so it never requires restarting mytrading-bot.service to take effect.

Reads every outputs/bot/bot_heartbeat_*.log, takes the last row of each, and
renders a status table plus a staleness roll-up.

Env overrides:
  BOTS_DIR          default ~/github/botsMyTrading
  BOT_HEARTBEAT_DIR default $BOTS_DIR/outputs/bot
  BOT_STALE_MINUTES default 150
"""
from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PST = ZoneInfo("America/Los_Angeles")

BOTS_DIR = Path(os.getenv("BOTS_DIR", str(Path.home() / "github" / "botsMyTrading")))
HB_DIR = Path(os.getenv("BOT_HEARTBEAT_DIR", str(BOTS_DIR / "outputs" / "bot")))
OUT_README = BOTS_DIR / "README.md"
STALE_MINUTES = int(os.getenv("BOT_STALE_MINUTES", "150"))


def last_row(path: Path) -> dict | None:
    """Return the final data row of a heartbeat log as a dict, or None."""
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    return rows[-1] if rows else None


def parse_ts(row: dict) -> datetime | None:
    raw = (row.get("time") or "").strip()
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S").replace(tzinfo=PST)
    except ValueError:
        return None


def collect() -> list[dict]:
    out = []
    for path in sorted(HB_DIR.glob("bot_heartbeat_*.log")):
        bot_id = path.stem.replace("bot_heartbeat_", "")
        row = last_row(path)
        if row is None:
            out.append({"bot_id": bot_id, "row": None, "ts": None, "age_min": None})
            continue
        ts = parse_ts(row)
        age = None
        if ts is not None:
            age = (datetime.now(PST) - ts).total_seconds() / 60.0
        out.append({"bot_id": bot_id, "row": row, "ts": ts, "age_min": age})
    return out


def fmt_price(raw: str) -> str:
    try:
        return f"{float(raw):,.4f}"
    except (TypeError, ValueError):
        return raw or "—"


def build(entries: list[dict]) -> str:
    now = datetime.now(PST).strftime("%Y-%m-%d %H:%M:%S %Z")

    total = len(entries)
    stale = [e for e in entries
             if e["age_min"] is None or e["age_min"] > STALE_MINUTES]
    live = total - len(stale)

    regimes: dict[str, int] = {}
    for e in entries:
        if e["row"]:
            key = (e["row"].get("regime") or "UNKNOWN").strip() or "UNKNOWN"
            regimes[key] = regimes.get(key, 0) + 1

    regime_lines = "\n".join(
        f"- **{k}:** {v}" for k, v in sorted(regimes.items(), key=lambda kv: -kv[1])
    ) or "- _no regime data_"

    lines = [
        "# Bot Status Snapshot",
        "",
        f"Last updated: **{now}**",
        "",
        "## Totals",
        f"- **Bots reporting:** {total}",
        f"- **Fresh (tick within {STALE_MINUTES} min):** {live}",
        f"- **Stale / no data:** {len(stale)}",
        "",
        "## Regime breakdown",
        regime_lines,
        "",
        "## Per-bot",
        "",
        "| Bot | Chain | Last tick (PT) | Age | Price | Final | Regime | Desired | Action |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for e in sorted(entries, key=lambda x: x["bot_id"]):
        bid = e["bot_id"]
        row = e["row"]
        if row is None:
            lines.append(f"| {bid} | — | ⚠️ no data | — | — | — | — | — | — |")
            continue

        age = e["age_min"]
        if age is None:
            age_txt = "⚠️ ?"
        elif age > STALE_MINUTES:
            age_txt = f"⚠️ {age:.0f}m"
        else:
            age_txt = f"{age:.0f}m"

        lines.append(
            f"| {row.get('bot_name') or bid} "
            f"| {row.get('blockchain') or '—'} "
            f"| {row.get('time') or '—'} "
            f"| {age_txt} "
            f"| {fmt_price(row.get('price'))} "
            f"| {row.get('final_signal') or '—'} "
            f"| {row.get('regime') or '—'} "
            f"| {row.get('desired_regime') or '—'} "
            f"| {row.get('action') or '—'} |"
        )

    if stale:
        names = ", ".join(sorted(e["bot_id"] for e in stale))
        lines += ["", "## ⚠️ Stale bots", "", names]

    lines += [
        "",
        "---",
        "",
        "Heartbeat logs live in `outputs/bot/`. "
        "Generated on the Raspberry Pi before each push.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    if not HB_DIR.exists():
        raise SystemExit(f"heartbeat dir not found: {HB_DIR}")
    entries = collect()
    OUT_README.parent.mkdir(parents=True, exist_ok=True)
    OUT_README.write_text(build(entries), encoding="utf-8")
    print(f"Wrote {OUT_README} ({len(entries)} bots)")


if __name__ == "__main__":
    main()