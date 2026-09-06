# remote_ops — Pi 1 setup

## 1. The ops sheet

Create a **new spreadsheet**, separate from your trade journal. Reason: the
journal is shared Viewer-only with the service account and `gsheet_notes.py`
uses `spreadsheets.readonly`. This one needs **Editor**, and you don't want
write access on the journal.

- Share it with the service-account email in `/etc/myTrading/gsheets.json`
  (`client_email`) as **Editor**
- Name the tab `ops`
- Grab the sheet ID from the URL

Row 1 header (written for you by `--init`):

| A | B | C | D | E | F | G | H | I | J |
|---|---|---|---|---|---|---|---|---|---|
| Verb | Args | Status | QueuedAt | StartedAt | FinishedAt | Secs | Exit | Output | Host |

You fill **A and B only**. Pi 1 fills C through J.

**Rows are append-only.** Never delete one — row numbers shift and the cursor
skips work. Archive by copying to another tab.

## 2. Install

```bash
cd ~/github/myTrading
git pull                     # after you push remote_ops.py from Pi 2
mkdir -p ~/.local/state/myTrading
```

## 3. Environment

Add to `~/github/myTrading/.env` (never committed):

```
GSHEET_OPS_ID=<the ops sheet id>
GSHEET_OPS_TAB=ops
```

## 4. Initialise

```bash
cd ~/github/myTrading
set -a; . ./.env; set +a
.venv/bin/python remote_ops.py --verbs          # confirm the allowlist
.venv/bin/python remote_ops.py --init           # write the header row
.venv/bin/python remote_ops.py                  # cold start: adopts, runs nothing
```

Third command prints `cold start: adopted row N, executed nothing`. That is
correct — it stops a fresh install replaying old rows. Now type `ping` into A2
and run it again; you should get `OK` back in the sheet within seconds.

## 5. systemd

`/etc/systemd/system/remote-ops.service`:

```ini
[Unit]
Description=myTrading remote ops channel (allowlisted)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=pi
WorkingDirectory=/home/pi/github/myTrading
EnvironmentFile=/home/pi/github/myTrading/.env
ExecStart=/usr/bin/flock -n /tmp/remoteops.lock /home/pi/github/myTrading/.venv/bin/python /home/pi/github/myTrading/remote_ops.py
TimeoutStartSec=600

# It only ever runs allowlisted read-mostly commands, but cheap hardening costs nothing
NoNewPrivileges=true
PrivateTmp=false
ProtectSystem=full
ProtectHome=false
```

`/etc/systemd/system/remote-ops.timer`:

```ini
[Unit]
Description=Poll the ops sheet every 2 minutes

[Timer]
OnBootSec=3min
OnUnitActiveSec=2min
AccuracySec=15s
Persistent=false

[Install]
WantedBy=timers.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now remote-ops.timer
```

Note `User=pi`, not root. And `flock -n` so a slow `git_pull` never stacks
runs — same pattern as `gitpush.py`.

## 6. Verify

```bash
systemctl list-timers remote-ops.timer
journalctl -u remote-ops.service -n 30 --no-pager
tail -n 20 ~/.local/state/myTrading/remote_ops_audit.log
cat ~/.local/state/myTrading/remote_ops.state       # cursor row
```

## 7. Verbs

| Verb | Args | Does |
|---|---|---|
| `ping` | — | liveness + host + Pi clock |
| `git_pull` | — | `pull --rebase --autostash` on `myTrading`, then last 5 commits |
| `git_status` | — | `status --short --branch` |
| `git_log` | — | last 15 commits |
| `status` | — | bot service state, uptime, root disk |
| `svc_log` | — | last 80 journal lines for the bot service |
| `tail_log` | `stocks` `crypto` `45` `ops` | last 120 lines |
| `ls` | `repo` `jobs` `bots` | directory listing |
| `disk` | — | `df -h` |
| `uptime` | — | uptime + memory |
| `cron_check` | — | full crontab |

`Args` takes a **key**, never a path. Add paths to the `LOGS` / `DIRS` dicts
in the script, then `git pull` on Pi 1 to pick them up.

## 8. Files it writes

| Path | What |
|---|---|
| `~/.local/state/myTrading/remote_ops.state` | cursor (last processed row), atomic write |
| `~/.local/state/myTrading/remote_ops_audit.log` | append-only JSON lines, one per event |
| `~/.local/state/myTrading/remote_ops_out/` | full untruncated output per run |

Column I holds the last 1500 chars. The full text lands in
`remote_ops_out/` — and since `ops` is a `tail_log` key, you can read the
audit log through the channel itself.

## 9. Guards in place

- Verb allowlist; no `shell=True` anywhere; argv lists only
- Args are dict keys, so `../../.env` cannot be smuggled in
- Cursor + non-empty `Status` both gate re-execution
- `MAX_ROWS_PER_RUN = 5`, cursor not advanced when the cap trips
- Cold start adopts rather than replays
- Per-verb timeouts; `RUNNING` claimed before execution so a crash is visible
- `--dry-run` reads and reports without executing or writing

## 10. Deliberately absent

Nothing here restarts `mytrading-bot.service`, touches `tokens.json`, runs
`cash_reserve.py`, or executes any bot. If you later want a verb with real
consequences, add HMAC first: a secret on Pi 1 only, and the row must carry a
hash of `row|verb|args`. Tedious to produce from a phone — which is the point.