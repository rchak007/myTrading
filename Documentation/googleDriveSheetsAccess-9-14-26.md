# Giving a machine access to Google Drive / Sheets

Written 2026-09-14 from the AI-TPM backup setup. Reusable for any project. Covers
both file sync (what we did here) and reading/updating Sheets, because they need
different auth and the wrong choice wastes an hour.

---

## 1. Decide which method you need — this is the whole game

| What you want to do | Use | Why |
|---|---|---|
| Sync files to/from Drive (backup, upload CSVs) | rclone + OAuth as you | Uploads create new files, which needs storage quota |
| Read or edit an existing Sheet (cells, tabs) | Service account, sheet shared with it | Editing an existing file needs no quota |
| Read files someone shared with you | Service account | Reading needs no quota |
| Create new files in a personal Drive | OAuth as you — a service account **CANNOT** | See the quota trap below |

### 🔴 The quota trap

A service account **cannot upload new files to a personal Google Drive**. It has
no storage quota of its own, and consumer Drive has no Shared Drives to put them
in. Uploads fail with *"Service Accounts do not have storage quota."*

But a service account **can** read, and **can** edit files that already exist and
have been shared with it. So:

- creating/uploading → must be **you** (OAuth)
- reading/editing → **service account** is cleaner and safer

This is why the AI-TPM backup authenticates as the user, but the planned
read-only dashboard will use a service account.

---

## 2. Method A — rclone, for syncing files

What we used for the AI-TPM dataset backup.

### 2.1 🔴 Choose the scope deliberately

This is the single most important decision, and it is easy to get wrong.

| Scope | What the token can do |
|---|---|
| `drive` | Read and write your **entire** Drive. Everything. |
| `drive.file` | Only files **this app created**. Cannot list, see or read anything else — ever. |

⚠️ **`root_folder_id` is NOT a security boundary.** It constrains where rclone
*looks*, not what the credential can *reach*. With `scope=drive`, the token
sitting on your machine can read your whole Drive — a different tool, a leak, or
a mistake would have full access.

> "The tool only touches folder X" ≠ "the credential can only reach folder X."

Use `drive.file` unless you have a specific reason not to.

**Consequence of `drive.file`:** the app cannot see folders you created by hand —
they're invisible to it. It must create and own its own folder. Expect it to
appear at your Drive root; you can move it afterwards (see §2.5).

### 2.2 Create the remote

```bash
rclone config create gdrive drive scope=drive.file --non-interactive
```

This prints a JSON prompt. Answer "not local" (headless):

```bash
rclone config update gdrive --continue \
  --state "*oauth-islocal,teamdrive,," --result "false"
```

It responds with an authorize command containing a base64 blob, e.g.

```bash
rclone authorize "drive" "eyJzY29wZSI6ImRyaXZlLmZpbGUifQ"
```

That blob carries the scope. Decode it to check:
`echo '<blob>' | base64 -d` → `{"scope":"drive.file"}`

### 2.3 🔴 Authorize — where people lose the restriction

The machine needs a browser for this step.

**DANGER:** older rclone versions reject the blob with
`Invalid number of arguments: 2`. If you then fall back to plain
`rclone authorize "drive"`, it defaults to **FULL drive scope** — you silently
get the unrestricted token you were trying to avoid, with no warning. We nearly
did exactly this.

Always run the version that accepts the blob.

If the target machine is headless (a Pi, a server), tunnel from a machine with a
browser:

```bash
ssh -L 53682:localhost:53682 user@host
# then on the remote host:
rclone authorize "drive" "<the blob>"
```

It prints a URL. Open it in your local browser — the tunnel lets Google's
redirect reach the remote host's rclone. The token prints in your terminal.

**Sanity check on the consent screen:** it should say *"See, edit, create and
delete only the specific Google Drive files you use with this app"* — **not**
*"See and download all your Google Drive files."*

Harmless noise you can ignore afterwards:
`channel 3: open failed: connect failed: Connection refused` — that's the tunnel
complaining after rclone already got the code and shut down.

### 2.4 Finish the config

```bash
rclone config update gdrive --continue \
  --state "*oauth-authorize,teamdrive,," --result '<PASTE THE {...} TOKEN>'

# then answer the Shared Drive question:
rclone config update gdrive --continue \
  --state "teamdrive_ok" --result "false"
```

Single quotes around the token — it's JSON.

Run these yourself rather than pasting the token into a chat or a log. It is a
live credential.

### 2.5 Pin the folder by ID so it survives being moved

rclone will have created its folder at the Drive root. Once you move it where you
actually want it, rclone would look for the *name* at the root, find nothing, and
silently create a second empty folder — then happily sync into that while your
real backup sits orphaned.

Get the ID and pin it:

```bash
rclone lsjson gdrive: --dirs-only     # gives Name and ID
rclone config update gdrive root_folder_id <THE_ID> --non-interactive
```

Drive IDs are permanent. Move, rename, nest it — rclone still finds it.

### 2.6 Verify the restriction actually holds

Do not skip this.

```bash
rclone lsd gdrive:     # should be EMPTY (or only the app's own folder)
rclone ls  gdrive:     # should be EMPTY
rclone about gdrive:   # should WORK — proves the connection is live
```

Empty listings + working `about` = the credential is connected but blind to
everything it did not create. **That is the proof.**

### 2.7 Use it

```bash
rclone sync ~/local-dir gdrive: --progress      # push (mirrors: deletes too)
rclone copy ~/local-dir gdrive: --progress      # push (never deletes)
rclone sync gdrive: ~/local-dir --progress      # restore
rclone check ~/local-dir gdrive:                # verify by checksum
```

Always `--dry-run` first. See `sync_to_drive.sh` in that repo for a wrapper with
confirmation prompts and a `--pull` restore mode.

---

## 3. Method B — service account, for reading/updating Sheets

Use this when a script needs to read or edit an existing Sheet. Cleaner than
OAuth: no browser, no token refresh, no expiry, and access is limited to exactly
what you share with it.

### 3.1 Create it

1. `console.cloud.google.com` → create (or pick) a project
2. **APIs & Services → Library** → enable **Google Sheets API**, and **Google
   Drive API** if you also need to list/download files
3. **APIs & Services → Credentials → Create credentials → Service account**
4. Open the service account → **Keys → Add key → JSON** → download it

The JSON contains an email like
`something@your-project.iam.gserviceaccount.com`. **That email is the identity.**

> **A key can only be downloaded once.** Google keeps the public half; the
> private half is handed to you at creation and never again. There is no
> re-download button. A key listed in the console that you cannot match to a
> file on disk is unusable — create a new one and delete the orphaned row,
> because each row is a credential that exists somewhere you cannot account for.

> **Check the email inside the file, not the filename.** Without ever printing
> the secret:
> ```bash
> grep -o '"client_email"[^,]*' key.json
> ```
> On this project the reader key turned out to be `mytrading-reader@…`, not
> `reader@…` as assumed. The share dialog accepts a nonexistent principal
> silently, so the mistake surfaces later as an opaque `PermissionError`.

### 3.2 Share the sheet with it

In Google Sheets: **Share** → paste the service-account email →

- **Viewer** if the script only reads
- **Editor** if it needs to write

That share is the entire extent of its access. It cannot see anything else in
your Drive. This is the cleanest scoping available — no folder IDs, no scope
flags, just "I shared this one thing with this one identity."

### 3.3 Use it from Python

```bash
pip install gspread google-auth
```

```python
import gspread
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]        # read+write
# SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]  # read only

creds = Credentials.from_service_account_file("key.json", scopes=SCOPES)
gc = gspread.authorize(creds)

sh = gc.open_by_key("<SPREADSHEET_ID>")     # the /d/<ID>/ part of the URL
ws = sh.worksheet("Sheet1")

rows = ws.get_all_records()                 # read -> list of dicts
ws.update_acell("B2", "hello")              # write one cell
ws.update("A1:C3", [[1,2,3],[4,5,6],[7,8,9]])   # write a range
```

**Open by key, not by name** — names are ambiguous and change.

### 3.4 Keep the key out of the repo

```bash
echo "*.json"       >> .gitignore
echo "service-account*.json" >> .gitignore
```

Better still, point at it by environment variable so the path is not in code:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp/my-project-sa.json
```

For Streamlit Cloud, paste the JSON into **Secrets** — encrypted, never in the
repo, so the code can be public while the access is not.

### 3.5 This project's two identities

| | Pi 1 (`rchak007pi`) | Pi 2 (`raspberrypi2`) |
|---|---|---|
| Identity | `mytrading-ops@mytrading-sheets…` | `mytrading-reader@mytrading-sheets…` |
| Role on both sheets | **Editor** | **Viewer** |
| Key | `/etc/myTrading/gsheets-ops.json` | `~/.config/myTrading/gsheets-reader.json` |
| Env var | `REMOTE_OPS_CREDS` | `GSHEET_READER_CREDS` |
| Mode | `600`, owner `rchak007` | `600`, owner `chakravarti` |

Pi 1 runs everything and needs write. Pi 2 only reads, so it gets an identity
that **cannot** write — the Drive share is the real boundary, not the client
code. Keep it that way: Pi 2 is the machine where untested code runs.

Getting a key onto Pi 2 (from the laptop, over Tailscale):

```bash
scp ~/myDev/myTrading/READER-mytrading-sheets-XXXX.json \
    chakravarti@100.77.66.80:~/.config/myTrading/gsheets-reader.json
```

Then on Pi 2 — `scp` does not reliably carry permissions:

```bash
chmod 600 ~/.config/myTrading/gsheets-reader.json
```

**Raspberry Pi OS blocks `pip install --user`** (PEP 668, "externally managed
environment"). Use a venv rather than `--break-system-packages`:

```bash
python3 -m venv .venv && .venv/bin/pip install gspread google-auth
```

Verify `.venv/` is gitignored before creating it inside a public repo.

---

## 4. Checklist for a new project

- [ ] Decide: syncing files, or reading/editing Sheets? (§1)
- [ ] Uploading to a personal Drive? → must be OAuth, **not** a service account
- [ ] Using rclone? → `scope=drive.file`, never `drive` + `root_folder_id`
- [ ] Headless machine? → SSH tunnel on port 53682
- [ ] Old rclone on the browser machine? → do **not** fall back to the no-blob
      form, it silently grants full scope
- [ ] After auth: `rclone lsd` and `rclone ls` return nothing
- [ ] Folder moved? → pin `root_folder_id` by ID
- [ ] Service account? → share the specific file/folder with its email
- [ ] Key file gitignored / in Secrets / behind an env var
- [ ] Token or key never pasted into a chat, log or screenshot

---

## 5. Things that cost us time on this project

| Symptom | Cause | Fix |
|---|---|---|
| `Invalid number of arguments: 2` | old rclone on the browser machine | use the newer one; do not drop the blob — that grants full scope |
| Folder appeared at Drive root | `drive.file` cannot see hand-made folders | let it create its own, then move it and pin by ID |
| Would have created a duplicate empty folder | rclone looks up by name at root | `root_folder_id` |
| `channel 3: open failed` | SSH tunnel after rclone already finished | harmless, ignore |
| (avoided) full-Drive token | `scope=drive` + `root_folder_id` reads as scoped but is not | `scope=drive.file` |
| (would have hit) upload quota error | service account on a personal Drive | OAuth as the user |
