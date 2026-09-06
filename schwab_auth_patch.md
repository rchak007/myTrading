# Patch: Schwab auth verbs into `remote_ops.py`

Three new verbs — `token_status`, `auth_url`, `auth_code`.

These run **in-process**, not through `subprocess`. That matters: `auth_code`
takes a free-form string (the redirect URL), which breaks the keys-only rule
everywhere else in the file. It is safe here precisely because the string never
reaches an argv or a shell — it is parsed by `urlparse`, validated against a
character class, and the extracted code goes into a `requests` POST body. Keep
it that way. Do not add a free-string verb that shells out.

---

## 1. New dict, after the `VERBS` block

```python
# Verbs handled in-process rather than via subprocess. They may take a
# free-form argument because nothing here reaches a shell or an argv.
#   verb -> needs_arg
PY_VERBS = {
    "token_status": False,
    "auth_url":     False,
    "auth_code":    True,
}


def run_pyverb(verb: str, arg: str) -> tuple[int, str]:
    import schwab_auth

    if verb == "token_status":
        return 0, json.dumps(schwab_auth.status(), indent=2)

    if verb == "auth_url":
        return 0, (
            "Open this on your phone, sign in, approve.\n"
            "The redirect to 127.0.0.1 WILL fail to load — that is expected.\n"
            "Copy the entire address bar, then add a row:\n"
            "    A: auth_code    B: <the whole pasted URL>\n\n"
            + schwab_auth.authorize_url()
        )

    if verb == "auth_code":
        try:
            return 0, schwab_auth.install(arg)
        except Exception as e:
            return 1, f"{type(e).__name__}: {e}"

    return 2, f"unhandled python verb {verb!r}"
```

## 2. Dispatch — at the very top of `run_verb`

Insert before `builder, timeout, needs_arg = VERBS[verb]`:

```python
    if verb in PY_VERBS:
        if PY_VERBS[verb] and not arg:
            return 2, f"verb '{verb}' requires an argument in column B"
        return run_pyverb(verb, arg)
```

## 3. Allowlist check — in `main`

Replace:

```python
        if verb not in VERBS:
```

with:

```python
        if verb not in VERBS and verb not in PY_VERBS:
```

and in the same block change the valid-verb list to:

```python
                          f"{', '.join(sorted(set(VERBS) | set(PY_VERBS)))}",
```

## 4. Burn the code cell after use

Right after the final `write_back(...)` call in the loop, before `audit(event="done"...)`:

```python
        # The authorization code is single-use and short-lived, but leaving
        # it sitting in a spreadsheet cell is pointless risk.
        if verb == "auth_code":
            try:
                ws.update(values=[["(consumed)"]], range_name=f"B{r}")
            except Exception as e:
                audit(event="redact_failed", row=r, error=str(e))
```

## 5. `--verbs` listing

After the `VERBS` loop in the `args.verbs` block:

```python
        for name, needs in sorted(PY_VERBS.items()):
            print(f"{name:<12} in-process{' <arg required>' if needs else ''}")
```

---

## The day-6 nudge (optional, this is the elegant part)

Have Pi 1 ask *you* for a re-auth, with the tappable link already waiting in
the sheet. Add to `remote_ops.py`:

```python
def nudge(ws) -> int:
    """Append an auth_url result row when the refresh token is aging out."""
    import schwab_auth
    st = schwab_auth.status()
    if st.get("state") not in ("RENEW_NOW", "EXPIRED", "MISSING", "UNKNOWN"):
        return 0
    ws.append_row(
        ["", "", f"ACTION: {st['state']}", now_str(), now_str(), now_str(), 0, "-",
         f"Schwab refresh token: {st.get('days_left', '?')} days left.\n"
         f"Tap to re-authorize, then add an auth_code row.\n\n"
         + schwab_auth.authorize_url(), HOST],
        value_input_option="RAW")
    audit(event="nudge", state=st.get("state"), days_left=st.get("days_left"))
    return 1
```

Wire it to a flag (`--nudge`) and a daily cron at, say, 07:30. Leaving column
A blank means the cursor walks past the row without trying to execute it.

---

## Setup

Add to `~/github/myTrading/.env` if not already there — `schwab_auth.py` reads
the same lowercase names `portfolio_snapshot.py` uses:

```
app_key=...
app_secret=...
callback_url=https://127.0.0.1:8182
```

`callback_url` must match the registered app **exactly**, port included.

Verify on the Pi before touching the sheet:

```bash
cd ~/github/myTrading
set -a; . ./.env; set +a
.venv/bin/python schwab_auth.py --status
.venv/bin/python schwab_auth.py --url
```

`--status` reports `OK`, `RENEW_NOW`, `EXPIRED`, or `MISSING`, with hours and
days remaining computed from the `refresh_token_issued` stamp in your existing
`tokens.json`.

## Phone workflow

1. Row: `token_status` → see days left
2. Row: `auth_url` → tap the link in column I
3. Sign in, approve. The 127.0.0.1 page fails to load — expected.
4. Tap the address bar, **Select All**, copy. You need the whole thing, `code=`
   and all.
5. Row: `auth_code` in A, the pasted URL in B
6. Column I comes back with the validated expiry date

Mobile Safari and Chrome both keep the failed URL in the address bar. Chrome
hides the query string until you tap in — tap, then select all.

## What protects you

- The new tokens are proven against `trader/v1/accounts/accountNumbers`
  before the live file is touched. A bad exchange leaves the working
  `tokens.json` exactly as it was.
- Previous file retained as `tokens.json.bak`; new file written `0600` via
  atomic rename.
- The existing file's JSON shape is preserved leaf-by-leaf, so a schwabdev
  version change cannot leave you with a structurally valid but unreadable
  token file.
- The authorization code is worthless without `app_key` + `app_secret`, which
  never leave Pi 1. It is also single-use and short-lived — and step 4 above
  overwrites the cell anyway.