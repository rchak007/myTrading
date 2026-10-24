#!/usr/bin/env python3
"""
test_notes.py
=============
Per-ticker sticky notes.

The property worth protecting is that NOTHING IS EVER ERASED. Replace and
delete are rows in an append-only log, not edits — so "if note is deleted its
ok, maybe archive log can have it" has to stay true no matter what the fold
does on top.
"""
import tempfile
from pathlib import Path

import ticker_notes as tn

QUIET = lambda *a, **k: None


def _fresh():
    return Path(tempfile.mkdtemp()) / "notes.csv"


def test_add_then_append_keeps_both_dates():
    P = _fresh()
    tn.add("PBLS", "20 shares I never sell", path=P, log=QUIET)
    tn.add("PBLS", "trimmed 5 above 3.20", path=P, log=QUIET)
    frags = tn.current(P)["PBLS"]
    assert len(frags) == 2
    # Each fragment keeps the date it was written. Appending must not restamp
    # an observation made months earlier.
    assert all(len(d) == 10 and d.count("-") == 2 for d, _ in frags)
    assert "never sell" in tn.render("PBLS", P)
    assert "trimmed 5" in tn.render("PBLS", P)


def test_ticker_is_case_insensitive():
    P = _fresh()
    tn.add("pbls", "lower", path=P, log=QUIET)
    tn.add("PBLS", "upper", path=P, log=QUIET)
    assert len(tn.current(P)["PBLS"]) == 2, "one ticker, not two"


def test_replace_drops_the_fold_but_not_the_log():
    P = _fresh()
    tn.add("PBLS", "original text", path=P, log=QUIET)
    tn.replace("PBLS", "brand new", path=P, log=QUIET)

    assert tn.render("PBLS", P).endswith("brand new")
    assert "original" not in tn.render("PBLS", P)
    # THE ARCHIVE. The old words are still on disk.
    assert any("original text" in r["Text"] for r in tn.history("PBLS", P))


def test_delete_clears_the_note_and_keeps_the_history():
    P = _fresh()
    tn.add("PBLS", "something worth keeping", path=P, log=QUIET)
    tn.delete("PBLS", path=P, log=QUIET)

    assert "PBLS" not in tn.current(P)
    assert tn.render("PBLS", P) == ""
    rows = tn.history("PBLS", P)
    assert len(rows) == 2 and rows[-1]["Action"] == "DELETE"
    assert any("worth keeping" in r["Text"] for r in rows), "archive lost it"


def test_note_can_start_again_after_a_delete():
    P = _fresh()
    tn.add("PBLS", "first life", path=P, log=QUIET)
    tn.delete("PBLS", path=P, log=QUIET)
    tn.add("PBLS", "second life", path=P, log=QUIET)

    frags = tn.current(P)["PBLS"]
    assert len(frags) == 1 and frags[0][1] == "second life", \
        "a delete must not leak into the next note"


def test_tickers_do_not_bleed_into_each_other():
    P = _fresh()
    tn.add("PBLS", "pbls note", path=P, log=QUIET)
    tn.add("MSTR", "mstr note", path=P, log=QUIET)
    tn.delete("PBLS", path=P, log=QUIET)
    assert "MSTR" in tn.current(P) and "PBLS" not in tn.current(P)


def test_bad_input_is_refused():
    P = _fresh()
    tn.add("PBLS", "ok", path=P, log=QUIET)
    for fn, args in ((tn.add, ("PBLS", "")),
                     (tn.add, ("PBLS", "   ")),
                     (tn.add, ("", "text")),
                     (tn.add, ("PBLS", "x" * (tn.MAX_TEXT + 1))),
                     (tn.delete, ("NOSUCH",))):
        try:
            fn(*args, path=P, log=QUIET)
        except ValueError:
            continue
        raise AssertionError(f"{fn.__name__}{args} was NOT refused")


def test_missing_file_is_empty_not_an_error():
    """A dashboard rebuild before the first note was ever written must not
    fail — it should simply render no notes."""
    P = Path(tempfile.mkdtemp()) / "never-created.csv"
    assert tn.current(P) == {}
    assert tn.render("PBLS", P) == ""
    assert tn.summary(P) == "no notes"


def test_the_verbs_exist_and_are_documented():
    import remote_ops as r
    for v in ("note", "note_replace", "note_delete", "notes"):
        assert v in r.PY_VERBS, f"{v} not registered"
    documented = {n for n, *_ in r.VERB_DOCS}
    assert {"note", "note_replace", "note_delete", "notes"} <= documented


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
