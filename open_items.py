#!/usr/bin/env python3
"""
open_items.py
=============
What is still open in PROJECT_PLAN.md, emailed.

    .venv/bin/python open_items.py              # print the digest
    .venv/bin/python open_items.py --email      # ...and send it
    .venv/bin/python open_items.py --full       # include the resolved ones

WHY THIS EXISTS. Chakravarti, 2026-10-14: *"do we have an email being sent out
on all open items for this project?? i need one please."* No — nothing read
PROJECT_PLAN.md. reminders.csv is a different thing: a short hand-kept list of
chores he types in. This is the engineering backlog, 81 headings and growing,
and it was only ever visible to whoever happened to open the file.

HOW IT DECIDES. The plan already marks every heading with an emoji, and has
done so consistently for months — this reads that convention rather than
imposing a new one:

    💡 OPEN / IDEA      open      something to build
    🐞 DEFECT           open      something broken
    ❓ OPEN QUESTION    open      a decision not yet made
    🔁 / 🟡 / ⏳        open      in flight or waiting
    ✅ RESOLVED         closed    kept as the record, hidden by default

FAILS LOUD. A plan it cannot parse, or one that suddenly has zero open items,
is reported as a problem rather than mailed as good news — "nothing is open"
is exactly what a broken parser says.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN = Path(os.getenv("PROJECT_PLAN",
                      HERE / "Documentation" / "PROJECT_PLAN.md"))

# marker -> (bucket, label). Order is display order.
MARKERS = {
    "🐞": ("open", "DEFECTS — something is broken"),
    "💡": ("open", "TO BUILD"),
    "❓": ("open", "DECISIONS NEEDED"),
    "🔁": ("open", "IN FLIGHT"),
    "🟡": ("open", "PARTLY DONE"),
    "⏳": ("open", "WAITING"),
    "🎯": ("open", "TARGET"),
    "✅": ("done", "RESOLVED"),
    "📒": ("note", "REFERENCE"),
    "📎": ("note", "REFERENCE"),
}
SECTION_RE = re.compile(r"^##\s+(.+?)\s*$")
ITEM_RE = re.compile(r"^###\s+(\S+)\s*(.*)$")


def parse(path: Path = PLAN) -> list[dict]:
    """[{marker, bucket, title, section, line}] for every ### heading."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"could not read {path}: {e}") from e

    out, section = [], ""
    for n, line in enumerate(text.splitlines(), 1):
        m = SECTION_RE.match(line)
        if m and not line.startswith("###"):
            section = m.group(1)
            continue
        m = ITEM_RE.match(line)
        if not m:
            continue
        marker, title = m.group(1), m.group(2).strip()
        bucket, _label = MARKERS.get(marker, (None, None))
        if bucket is None:
            # A heading with no marker is prose structure ("### How it works"),
            # not a tracked item. Skipped rather than guessed at.
            continue
        out.append({"marker": marker, "bucket": bucket, "title": title,
                    "section": section, "line": n})
    return out


def compose(items: list[dict], full: bool = False) -> tuple[str, str]:
    open_items = [i for i in items if i["bucket"] == "open"]
    done = [i for i in items if i["bucket"] == "done"]

    by_marker: dict[str, list[dict]] = {}
    for i in open_items:
        by_marker.setdefault(i["marker"], []).append(i)

    n_def = len(by_marker.get("🐞", []))
    subject = (f"📋 {len(open_items)} open item(s)"
               + (f" · {n_def} defect(s)" if n_def else "")
               + " — myTrading")

    body = [f"{len(open_items)} open, {len(done)} resolved, "
            f"in Documentation/PROJECT_PLAN.md", ""]
    for marker, (bucket, label) in MARKERS.items():
        group = by_marker.get(marker) if bucket == "open" else None
        if not group:
            continue
        body.append(f"{marker}  {label}  ({len(group)})")
        last_section = None
        for i in group:
            if i["section"] != last_section:
                body.append(f"    — {i['section']}")
                last_section = i["section"]
            body.append(f"      {i['title']}")
        body.append("")

    if full and done:
        body.append(f"✅  RESOLVED  ({len(done)})")
        for i in done:
            body.append(f"      {i['title']}")
        body.append("")

    body += ["Full detail, with the reasoning and the numbers behind each:",
             "    Documentation/PROJECT_PLAN.md",
             "",
             "Items are read from the ### headings' emoji markers. Marking one",
             "✅ RESOLVED is what takes it off this list.",
             "", "— open_items.py on Pi 2"]
    return subject, "\n".join(body)


def main() -> int:
    ap = argparse.ArgumentParser(description="Open items from PROJECT_PLAN.md")
    ap.add_argument("--email", action="store_true", help="send it")
    ap.add_argument("--full", action="store_true", help="list resolved ones too")
    args = ap.parse_args()

    try:
        items = parse()
    except Exception as e:
        print(f"⛔ {e}", file=sys.stderr)
        return 1

    n_open = sum(1 for i in items if i["bucket"] == "open")
    if not items or not n_open:
        # Zero open items is what a broken parser looks like, and this plan has
        # never been empty. Refuse to report it as good news.
        print(f"⛔ parsed {len(items)} heading(s) and {n_open} open item(s) "
              f"from {PLAN} — that is almost certainly a parse failure, not an "
              f"empty backlog. Nothing sent.", file=sys.stderr)
        return 1

    subject, body = compose(items, full=args.full)
    print(subject)
    print()
    print(body)

    if args.email:
        sys.path.insert(0, str(HERE))
        from token_watch import send
        send(subject, body + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
