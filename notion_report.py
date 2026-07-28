#!/usr/bin/env python3
"""
notion_report.py — Daily Notion task report (Route 2: direct Notion API walker)

Walks a Notion page's toggle/heading tree, skips Archive nodes, and surfaces:
  • keyword items (default: A1000)
  • items with due dates  → OVERDUE / DUE SOON
Optional --smart pass hands the items to Haiku for natural-language matching.

The deterministic walk costs $0 (no model, free Notion API). Haiku only runs
when you pass --smart, and by default goes through the `claude` CLI so it stays
on your Pro plan (same pattern as GeniusAct).

Config is read from real env vars. A `.env` file in the working directory (or
$DOTENV_PATH) is auto-loaded at startup — no python-dotenv needed. Real env vars
already set in the shell take precedence over .env. Keep .env out of git.

Env / .env keys:
  NOTION_TOKEN_LAUSD          Notion internal integration token (ntn_...)      [required]
  NOTION_ROOT_PAGE_ID   page id to walk (or pass --page)                 [required]
  KEYWORDS              comma-separated flags, default "A1000"
  DUE_SOON_DAYS         look-ahead window for "due soon", default 7
  SKIP_NODES            comma-separated node titles to prune, default "Archive"
  SMART_BACKEND         'cli' (default, Pro-covered) or 'api'
  CLAUDE_CLI_MODEL      model passed to `claude -p`, default "claude-haiku-4-5"
  ANTHROPIC_API_KEY     only needed when SMART_BACKEND=api
  TELEGRAM_BOT_TOKEN }  optional: deliver the report to Telegram
  TELEGRAM_CHAT_ID   }

Run:
  python3 notion_report.py --page <id>
  python3 notion_report.py --page <id> --smart "anything blocking the July payroll run"
  python3 notion_report.py --selftest          # offline demo, mock data, no network
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime

# ─── .env loader (stdlib, no python-dotenv) ──────────────────────────────────

def load_dotenv(path=None):
    """Load KEY=VALUE lines from a .env file into os.environ. Harmless if the
    file is absent. Existing env vars are NOT overwritten (shell wins over .env).
    Supports optional `export ` prefix, # comments, and quoted values."""
    path = path or os.environ.get("DOTENV_PATH", ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val

load_dotenv()  # must run before config below reads os.environ

# ─── Config ──────────────────────────────────────────────────────────────────

NOTION_VERSION = os.environ.get("NOTION_VERSION", "2022-06-28")
KEYWORDS = [k.strip().lower() for k in os.environ.get("KEYWORDS", "A1000").split(",") if k.strip()]
SKIP_NODES = {s.strip().lower() for s in os.environ.get("SKIP_NODES", "Archive").split(",") if s.strip()}
DUE_SOON_DAYS = int(os.environ.get("DUE_SOON_DAYS", "7"))
REQUEST_DELAY = float(os.environ.get("NOTION_REQUEST_DELAY", "0.34"))  # ~3 req/s ceiling

SMART_BACKEND = os.environ.get("SMART_BACKEND", "cli")
CLAUDE_CLI_MODEL = os.environ.get("CLAUDE_CLI_MODEL", "claude-haiku-4-5")
HAIKU_API_MODEL = os.environ.get("HAIKU_API_MODEL", "claude-haiku-4-5-20251001")

DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")
RICH_TEXT_TYPES = {
    "paragraph", "toggle", "heading_1", "heading_2", "heading_3",
    "bulleted_list_item", "numbered_list_item", "to_do", "quote", "callout",
}

# ─── HTTP helper (stdlib only) ───────────────────────────────────────────────

def http_json(method, url, headers, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code} {url}\n{detail}") from None

# ─── Notion client ───────────────────────────────────────────────────────────

class NotionClient:
    def __init__(self, token):
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }

    def get_children(self, block_id):
        """Yield every child block, following pagination."""
        cursor = None
        while True:
            url = f"https://api.notion.com/v1/blocks/{block_id}/children?page_size=100"
            if cursor:
                url += f"&start_cursor={cursor}"
            time.sleep(REQUEST_DELAY)
            payload = http_json("GET", url, self.headers)
            for block in payload.get("results", []):
                yield block
            if not payload.get("has_more"):
                break
            cursor = payload.get("next_cursor")


class MockClient:
    """Offline client for --selftest. Mirrors Notion block JSON shape."""
    def __init__(self, tree):
        self.tree = tree

    def get_children(self, block_id):
        for block in self.tree.get(block_id, []):
            yield block

# ─── Block parsing ───────────────────────────────────────────────────────────

def block_text(block):
    btype = block.get("type", "")
    if btype == "child_page":
        return block.get("child_page", {}).get("title", "").strip()
    payload = block.get(btype, {})
    rich = payload.get("rich_text", [])
    text = "".join(rt.get("plain_text", "") for rt in rich).strip()
    return text


def is_checked(block):
    btype = block.get("type", "")
    if btype == "to_do":
        return bool(block.get("to_do", {}).get("checked"))
    return None


def node_is_skipped(text):
    return text.strip().lower() in SKIP_NODES

# ─── Tree walk ───────────────────────────────────────────────────────────────

def walk(client, block_id, path, out):
    """Depth-first. Every text-bearing block is a candidate item AND, if it has
    children, a node we descend into. Skipped nodes prune their whole subtree."""
    for block in client.get_children(block_id):
        text = block_text(block)
        if text and node_is_skipped(text):
            continue  # prune Archive (and everything under it)
        if text:
            out.append({
                "path": list(path),
                "text": text,
                "type": block.get("type", ""),
                "checked": is_checked(block),
            })
        if block.get("has_children"):
            child_path = path + [text] if text else path
            walk(client, block["id"], child_path, out)

# ─── Deterministic analysis (free) ───────────────────────────────────────────

def parse_due(text):
    m = DATE_RE.search(text)
    if not m:
        return None
    mm, dd, yy = (int(g) for g in m.groups())
    if yy < 100:
        yy += 2000
    try:
        return date(yy, mm, dd)
    except ValueError:
        return None


def analyze(items, today=None):
    today = today or date.today()
    soon = today.toordinal() + DUE_SOON_DAYS
    for it in items:
        low = it["text"].lower()
        it["keywords"] = [k for k in KEYWORDS if k in low]
        it["due"] = parse_due(it["text"])
    overdue, due_soon, keyword = [], [], []
    for it in items:
        d = it["due"]
        if d and not it["checked"]:
            if d < today:
                overdue.append(it)
            elif d.toordinal() <= soon:
                due_soon.append(it)
        if it["keywords"] and not it["checked"]:
            keyword.append(it)
    overdue.sort(key=lambda i: i["due"])
    due_soon.sort(key=lambda i: i["due"])
    return overdue, due_soon, keyword

# ─── Smart pass (Haiku, opt-in) ──────────────────────────────────────────────

def smart_filter(items, criteria):
    """Ask Haiku which items match a natural-language criteria. Returns
    list of (item, reason). Never raises — a smart-pass hiccup must not break
    the deterministic report."""
    if not items:
        return []
    numbered = "\n".join(f"{i}. {it['text']}" for i, it in enumerate(items))
    prompt = (
        "You are filtering a task list. Return ONLY a JSON array (no prose, no "
        "markdown fences) of objects {\"i\": <index>, \"why\": \"<≤8 words>\"} "
        f"for tasks matching this criteria: \"{criteria}\".\n\nTASKS:\n{numbered}"
    )
    try:
        raw = _smart_call(prompt)
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        hits = json.loads(raw)
        out = []
        for h in hits:
            idx = h.get("i")
            if isinstance(idx, int) and 0 <= idx < len(items):
                out.append((items[idx], h.get("why", "")))
        return out
    except Exception as e:
        print(f"[smart] skipped ({SMART_BACKEND}): {e}", file=sys.stderr)
        return []


def _smart_call(prompt):
    if SMART_BACKEND == "api":
        key = os.environ["ANTHROPIC_API_KEY"]
        payload = http_json(
            "POST", "https://api.anthropic.com/v1/messages",
            {
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            {
                "model": HAIKU_API_MODEL,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        return "".join(b.get("text", "") for b in payload.get("content", []))
    # default: 'cli' — routes through Claude Code, covered by Pro.
    # NOTE: verify the exact flags against your working GeniusAct invocation;
    # some CLI versions pin the model differently or drop --model.
    proc = subprocess.run(
        ["claude", "-p", "--model", CLAUDE_CLI_MODEL, prompt],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "claude CLI failed")
    return proc.stdout

# ─── Report ──────────────────────────────────────────────────────────────────

def crumb(item):
    return " › ".join(item["path"]) or "(root)"


def build_report(items, smart_hits=None, criteria=None, today=None):
    today = today or date.today()
    overdue, due_soon, keyword = analyze(items, today)
    lines = [f"=== Notion Daily Report — {today.isoformat()} ===",
             f"scanned {len(items)} items · skipped: {', '.join(sorted(SKIP_NODES)) or 'none'}", ""]

    def fmt(it, extra=""):
        return f"  • {it['text']}\n      [{crumb(it)}]{extra}"

    lines.append(f"🔴 OVERDUE ({len(overdue)})")
    for it in overdue:
        days = (today - it["due"]).days
        lines.append(fmt(it, f"  (due {it['due']}, {days}d overdue)"))
    if not overdue:
        lines.append("  — none")
    lines.append("")

    lines.append(f"🟡 DUE SOON · next {DUE_SOON_DAYS}d ({len(due_soon)})")
    for it in due_soon:
        days = (it["due"] - today).days
        lines.append(fmt(it, f"  (due {it['due']}, in {days}d)"))
    if not due_soon:
        lines.append("  — none")
    lines.append("")

    label = "/".join(k.upper() for k in KEYWORDS)
    lines.append(f"⭐ {label} ITEMS ({len(keyword)})")
    for it in keyword:
        extra = f"  (due {it['due']})" if it["due"] else ""
        lines.append(fmt(it, extra))
    if not keyword:
        lines.append("  — none")

    if smart_hits is not None:
        lines += ["", '🤖 SMART MATCHES — "{}" ({})'.format(criteria, len(smart_hits))]
        for it, why in smart_hits:
            lines.append(fmt(it, f"  ({why})" if why else ""))
        if not smart_hits:
            lines.append("  — none")

    return "\n".join(lines)

# ─── Telegram (optional) ─────────────────────────────────────────────────────

def send_telegram(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for chunk_start in range(0, len(text), 3900):
        http_json("POST", url, {"Content-Type": "application/json"},
                  {"chat_id": chat, "text": text[chunk_start:chunk_start + 3900]})
    return True

# ─── Self-test (offline) ─────────────────────────────────────────────────────

def _b(bid, btype, text, has_children=False, toggleable=False, checked=None):
    block = {"id": bid, "type": btype, "has_children": has_children}
    if btype == "child_page":
        block["child_page"] = {"title": text}
        return block
    payload = {"rich_text": [{"plain_text": text}] if text else []}
    if btype.startswith("heading"):
        payload["is_toggleable"] = toggleable
    if btype == "to_do":
        payload["checked"] = bool(checked)
    block[btype] = payload
    return block


def selftest():
    tree = {
        "root": [
            _b("u", "heading_2", "Union contracts", has_children=True, toggleable=True),
            _b("redwood", "toggle", "Redwood -"),
        ],
        "u": [
            _b("a1sheet", "toggle", "A1000 Schema co-ordination sheet"),
            _b("prem", "toggle", "Premiums number changes"),
            _b("pymod", "toggle", "6/21/26 - PY Mod 39_Create Parental Leave Unpaid Absence Code PLUP"),
            _b("r11949", "toggle", "R11949 - Teamsters - Units D, S, A, and H Implementation", has_children=True),
            _b("r11959", "toggle", "R11959 - SEIU Implementation Memo and MOUs", has_children=True),
            _b("r11969", "toggle", "R11969 - UTLA 2025-27 Salary Raise other", has_children=True),
            _b("budget", "toggle", "Budget Memo to Establish Volunteer Athletic Coach with an Honorarium Classification (unclassified) - email Angela 6/22/26"),
            _b("r11975", "toggle", "R11975 - Unit S A H - Vacation accruals"),
            _b("archive", "toggle", "Archive", has_children=True),
        ],
        "r11949": [_b("r11949c", "paragraph", "A1000 -review again to make sure all is covered for changes.")],
        "r11959": [_b("unitg", "toggle", "Unit G - A1000")],
        "r11969": [_b("parental", "toggle", "Parental Leave changes A1000 - Due 7/1/26")],
        "archive": [_b("secret", "toggle", "A1000 OLD archived item - Due 1/1/26 - should NOT appear")],
    }
    items = []
    walk(MockClient(tree), "root", [], items)
    print(build_report(items))
    # sanity assertions
    texts = [i["text"] for i in items]
    assert not any("should NOT appear" in t for t in texts), "Archive was not pruned!"
    assert any("A1000 Schema" in t for t in texts), "A1000 item missing!"
    print("\n[selftest] OK — Archive pruned, A1000 + due dates detected.")

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Daily Notion task report (Route 2).")
    ap.add_argument("--page", default=os.environ.get("NOTION_ROOT_PAGE_ID"),
                    help="root page/block id to walk")
    ap.add_argument("--smart", metavar="CRITERIA",
                    help="natural-language criteria for the Haiku pass")
    ap.add_argument("--telegram", action="store_true",
                    help="also push the report to Telegram (needs env vars)")
    ap.add_argument("--selftest", action="store_true", help="offline demo, no network")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    token = os.environ.get("NOTION_TOKEN_LAUSD")
    if not token or not args.page:
        ap.error("NOTION_TOKEN_LAUSD and --page (or NOTION_ROOT_PAGE_ID) are required")

    items = []
    walk(NotionClient(token), args.page, [], items)

    smart_hits = None
    if args.smart:
        smart_hits = smart_filter(items, args.smart)

    report = build_report(items, smart_hits, args.smart)
    print(report)
    if args.telegram and send_telegram(report):
        print("\n[telegram] sent.", file=sys.stderr)


if __name__ == "__main__":
    main()