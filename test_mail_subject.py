#!/usr/bin/env python3
"""
test_mail_subject.py
====================
Every email this project sends is tagged, and the tag goes on exactly once.

    .venv/bin/python test_mail_subject.py

Chakravarti 2026-10-10: *"for any and all email you send always in title
prepend - from myTradingPi - so that i know its from this project"*.

"ANY AND ALL" IS THE REQUIREMENT, AND IT IS THE HARD PART
    The tag is applied inside token_watch.send(), not at the call sites, so a
    caller written next month inherits it without anyone remembering to. This
    file asserts that arrangement — that send() is where it happens — because
    the day someone "tidies" it out to the callers is the day a new alert goes
    out untagged and nobody notices.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import token_watch as TW                                   # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


P = TW.SUBJECT_PREFIX

print("\n── the tag ──")
check("the tag says what he asked for", "from myTradingPi" in P, P)
check("a plain subject gets it",
      TW.with_prefix("hello") == f"{P} hello", TW.with_prefix("hello"))
check("the original subject survives intact",
      TW.with_prefix("hello").endswith("hello"))

print("\n── exactly once ──")
once = TW.with_prefix("🔴 Schwab token EXPIRED")
check("an already-tagged subject is not tagged twice",
      TW.with_prefix(once) == once, TW.with_prefix(once))
check("...and a third pass still does nothing",
      TW.with_prefix(TW.with_prefix(once)) == once)
check("the tag appears exactly one time", once.count(P) == 1, once)

print("\n── the real subjects this project builds ──")
# The emoji must stay readable: it is the severity marker, and burying it
# behind the tag is fine but losing it is not.
for s in ("\U0001F4CC Still to look at: trendline cheatcode",
          "\U0001F534 Schwab token EXPIRED — nothing is trading",
          "\U0001F534 Pi 1 has not polled in 16h — ⚠️ Schwab token expires",
          "\U0001F3AF OPTIONS / PRINCIPLES — 2 to act on"):
    out = TW.with_prefix(s)
    check(f"{s[:38]}…", out.startswith(P) and out.endswith(s), out)

print("\n── degenerate input ──")
check("empty is just the tag, not a crash", TW.with_prefix("") == P)
check("None is just the tag, not 'None'",
      TW.with_prefix(None) == P, TW.with_prefix(None))
check("surrounding whitespace is cleaned",
      TW.with_prefix("  hi  ") == f"{P} hi", repr(TW.with_prefix("  hi  ")))

print("\n── it is applied in the MAILER, not the callers ──")
src = inspect.getsource(TW.send)
check("send() builds the Subject header through with_prefix",
      'msg["Subject"] = with_prefix(subject)' in src,
      "send() sets Subject without the prefixer — a new caller would go "
      "out untagged")
# reminders.py is the only module outside token_watch that mails. It must NOT
# pre-tag: send() does it, and double-tagging is only prevented by the guard
# above rather than by anything reminders.py knows.
rem = Path(__file__).resolve().parent / "reminders.py"
if rem.exists():
    text = rem.read_text(encoding="utf-8")
    body = "\n".join(ln for ln in text.splitlines()
                     if "with_prefix" not in ln)
    check("reminders.py does not hard-code the tag into its subjects",
          "from myTradingPi" not in body)

print(f"\n{'ALL PASS' if not FAILED else str(len(FAILED)) + ' FAILED: ' + ', '.join(FAILED)}")
raise SystemExit(1 if FAILED else 0)
