# =====================================================================
# jobMyTrading — CSV Viewer + Todo Viewer
# ---------------------------------------------------------------------
# Two modes (sidebar toggle):
#   📊 Trading CSVs  — the generic CSV browser. Reads CSVs off THIS repo
#                      on disk. Any CSV carrying a Ticker column gets a
#                      selectable table: pick a row and a modal opens with
#                      every matching row from the myTrading Google Sheet
#                      (see gsheet_notes.py).
#   🌳 Todos         — read-only tree view of the three OpenClaw agent
#                      JSONs, which live in a SEPARATE private repo
#                      (rchak007/todo-data) written by the Pi2 watcher.
#                      Fetched at runtime via a read-only GitHub token.
#
# Why two repos: jobTrading runs on the Pi, OpenClaw runs on Pi2. Each
# pushes to its OWN repo so the two machines never collide on one branch.
# This app (the one private Streamlit slot) reads both.
#
# Run locally:
#     streamlit run dashboard.py
#
# Deploy on Streamlit Community Cloud:
#     rchak007/jobMyTrading  →  dashboard.py  (main branch)
# =====================================================================
from __future__ import annotations

import base64
import io
import json
import os
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Per-ticker trade-journal popup, backed by the myTrading Google Sheet.
# Optional: if gsheet_notes.py is absent (or gspread / creds are not
# configured) the CSV viewer falls back to plain tables, unchanged.
try:
    import gsheet_notes
except Exception:
    gsheet_notes = None

PST = ZoneInfo("America/Los_Angeles")

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="jobMyTrading",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Repo root = folder this script lives in (works on Streamlit Cloud
# because the repo is checked out and dashboard.py sits at the root).
REPO_ROOT = Path(__file__).resolve().parent


# =====================================================================
# ============== TRADING CSV VIEWER (original, unchanged) =============
# =====================================================================

def list_csv_files(root: Path) -> list[Path]:
    """Recursively list every .csv under root, sorted (root files first)."""
    skip_dirs = {".git", ".streamlit", "__pycache__", ".venv", "venv", "node_modules"}
    files: list[Path] = []
    for p in root.rglob("*.csv"):
        if any(part in skip_dirs for part in p.parts):
            continue
        files.append(p)
    files.sort(key=lambda p: (len(p.relative_to(root).parts), str(p).lower()))
    return files


def fmt_mtime(p: Path) -> str:
    try:
        return datetime.fromtimestamp(p.stat().st_mtime, tz=PST).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return "—"


@st.cache_data(show_spinner=False)
def load_csv_path(path_str: str, mtime: float) -> pd.DataFrame:
    """Cache by (path, mtime) so a re-pushed file invalidates the cache."""
    df = pd.read_csv(path_str, dtype=str, keep_default_na=False, na_values=[""])
    return _coerce_types(df)


@st.cache_data(show_spinner=False)
def load_csv_bytes(data: bytes, name: str) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, na_values=[""])
    return _coerce_types(df)


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort: try numeric then datetime on each column."""
    for c in df.columns:
        s = df[c]
        as_num = pd.to_numeric(s, errors="coerce")
        non_null = s.notna().sum()
        if non_null > 0 and as_num.notna().sum() / non_null >= 0.9:
            df[c] = as_num
            continue
        name_l = c.lower()
        looks_temporal = any(k in name_l for k in
                             ("date", "time", "updated", "timestamp", "ts"))
        if looks_temporal:
            as_dt = pd.to_datetime(s, errors="coerce", utc=False)
            if non_null > 0 and as_dt.notna().sum() / non_null >= 0.9:
                df[c] = as_dt
                continue
    return df


def filter_dataframe(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """Per-column filters; returns filtered df."""
    if df.empty:
        return df
    with st.expander("🔎 Column filters", expanded=False):
        cols_to_filter = st.multiselect(
            "Pick columns to filter on",
            options=list(df.columns),
            default=[],
            key=f"{key_prefix}_pick",
            help="Choose any column to add a filter widget for it.",
        )
        if not cols_to_filter:
            return df
        out = df.copy()
        grid = st.columns(2)
        for i, c in enumerate(cols_to_filter):
            with grid[i % 2]:
                out = _apply_one_filter(out, df, c, key_prefix)
        st.caption(f"Showing **{len(out):,}** of {len(df):,} rows.")
        return out


def _apply_one_filter(out: pd.DataFrame, full: pd.DataFrame, col: str,
                      key_prefix: str) -> pd.DataFrame:
    """Render the right widget for the column dtype and apply it."""
    s_full = full[col]
    widget_key = f"{key_prefix}_flt_{col}"

    if pd.api.types.is_numeric_dtype(s_full):
        s_num = pd.to_numeric(s_full, errors="coerce")
        if s_num.notna().sum() == 0:
            st.caption(f"`{col}`: no numeric values")
            return out
        lo, hi = float(s_num.min()), float(s_num.max())
        if lo == hi:
            st.caption(f"`{col}`: single value = {lo}")
            return out
        rng = st.slider(
            f"{col} (range)",
            min_value=lo, max_value=hi, value=(lo, hi),
            key=widget_key,
        )
        col_num = pd.to_numeric(out[col], errors="coerce")
        return out[col_num.between(rng[0], rng[1]) | col_num.isna()]

    if pd.api.types.is_datetime64_any_dtype(s_full):
        s_dt = pd.to_datetime(s_full, errors="coerce")
        valid = s_dt.dropna()
        if valid.empty:
            st.caption(f"`{col}`: no valid dates")
            return out
        dmin, dmax = valid.min().date(), valid.max().date()
        if dmin == dmax:
            st.caption(f"`{col}`: single date = {dmin}")
            return out
        picked = st.date_input(
            f"{col} (date range)",
            value=(dmin, dmax),
            min_value=dmin, max_value=dmax,
            key=widget_key,
        )
        if isinstance(picked, tuple) and len(picked) == 2:
            start, end = picked
            col_dt = pd.to_datetime(out[col], errors="coerce")
            mask = (col_dt.dt.date >= start) & (col_dt.dt.date <= end)
            return out[mask | col_dt.isna()]
        return out

    s_str = s_full.astype(str)
    nunique = s_str.nunique(dropna=True)
    if 0 < nunique <= 50:
        vals = sorted(s_str.dropna().unique().tolist())
        sel = st.multiselect(
            f"{col} ({nunique} unique)",
            options=vals, default=[],
            key=widget_key,
        )
        if sel:
            return out[out[col].astype(str).isin(sel)]
        return out

    needle = st.text_input(
        f"{col} contains…",
        value="", key=widget_key,
        placeholder="case-insensitive substring",
    )
    if needle:
        return out[out[col].astype(str).str.contains(needle, case=False, na=False)]
    return out


def sort_dataframe(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    if df.empty:
        return df
    with st.expander("↕️ Sort", expanded=False):
        sort_cols = st.multiselect(
            "Sort by (in order)",
            options=list(df.columns),
            default=[],
            key=f"{key_prefix}_sortcols",
            help="Pick one or more columns; the first is the primary sort key.",
        )
        if not sort_cols:
            return df
        directions = []
        cols_grid = st.columns(min(4, len(sort_cols)))
        for i, c in enumerate(sort_cols):
            with cols_grid[i % len(cols_grid)]:
                d = st.radio(
                    f"{c}", options=["↑ Asc", "↓ Desc"],
                    horizontal=True, key=f"{key_prefix}_dir_{c}",
                )
                directions.append(d == "↑ Asc")
        try:
            return df.sort_values(by=sort_cols, ascending=directions,
                                  kind="mergesort", na_position="last")
        except Exception as e:
            st.warning(f"Sort failed: {e}")
            return df


DEFAULT_STACK = [
    "macro.csv",
    "stocks_signals.csv",
    "stocks_orders.csv",
    "cash.csv",
    "beth_funds.csv",
    "io_fund.csv",
    "investanswers.csv",
    "chitra_tickers.csv",
    "crypto_signals.csv",
]


def build_column_config(view: pd.DataFrame) -> dict:
    """
    Pin the Ticker column and give the money/quantity columns enough room
    for ~10 digits plus decimals. Built defensively: older Streamlit builds
    that lack `pinned=` fall back to an unpinned column rather than crashing
    the whole table.
    """
    cfg: dict = {}
    money_cols = {"VALUE", "EST_VALUE", "LIMIT_PRICE", "STOP_PRICE",
                  "CURRENT PRICE", "LAST CLOSE", "SUPERTREND"}
    qty_cols   = {"QTY", "FILLED_QTY", "REMAINING_QTY"}

    for col in view.columns:
        cu = str(col).strip().upper()
        try:
            if cu == "TICKER":
                try:
                    cfg[col] = st.column_config.TextColumn(
                        col, pinned=True, width="small",
                    )
                except TypeError:          # Streamlit < 1.39: no pinned kwarg
                    cfg[col] = st.column_config.TextColumn(col, width="small")
            elif cu in money_cols:
                cfg[col] = st.column_config.NumberColumn(
                    col, format="%.2f", width="medium",
                )
            elif cu in qty_cols:
                cfg[col] = st.column_config.NumberColumn(
                    col, format="%.4f", width="medium",
                )
        except Exception:
            continue                       # never let styling break the table
    return cfg


def render_csv_block(label: str, df: pd.DataFrame, mtime: str, key_prefix: str):
    """Render one CSV's full interactive block."""
    c1, c2, c3 = st.columns([4, 2, 2])
    c1.markdown(f"**File:** `{label}`")
    c2.markdown(f"**Last modified:** {mtime}")
    c3.markdown(f"**Shape:** {len(df):,} rows × {df.shape[1]} cols")

    all_cols = list(df.columns)
    shown_cols = st.multiselect(
        "Columns to display",
        options=all_cols, default=all_cols,
        key=f"{key_prefix}_cols",
        help="Uncheck to hide columns from the table below.",
    )
    view = df[shown_cols] if shown_cols else df
    view = filter_dataframe(view, key_prefix=key_prefix)
    view = sort_dataframe(view, key_prefix=key_prefix)

    # Ticker-bearing CSVs get a selectable table wired to the journal
    # popup. macro.csv has no ticker column, so it renders as before.
    ticker_col = gsheet_notes.find_ticker_col(view) if gsheet_notes else None
    use_popup = ticker_col is not None and gsheet_notes.available()

    if use_popup:
        ev = st.dataframe(
            view, use_container_width=True, height=480,
            key=f"{key_prefix}_tbl",
            column_config=build_column_config(view),
            on_select="rerun", selection_mode="single-row",
        )
        picked = list(ev.selection.rows) if ev and ev.selection else []
        if picked:
            tk = str(view.iloc[picked[0]][ticker_col])
            # The selection survives the rerun that closes the dialog, so
            # only auto-open when the pick actually changed; otherwise
            # offer a button to reopen the same ticker.
            seen_key = f"{key_prefix}_seen"
            if st.session_state.get(seen_key) != (tk, picked[0]):
                st.session_state[seen_key] = (tk, picked[0])
                gsheet_notes.show_ticker_notes(tk, extra=render_open_orders)
            elif st.button(f"📓 Journal for {tk.upper()}",
                           key=f"{key_prefix}_reopen"):
                gsheet_notes.show_ticker_notes(tk, extra=render_open_orders)
        else:
            st.caption("Select a row to open its trade journal.")
    else:
        st.dataframe(
            view, use_container_width=True, height=480,
            column_config=build_column_config(view),
        )

    st.download_button(
        "📥 Download current view as CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name=f"view_{Path(label).stem or 'data'}.csv",
        mime="text/csv",
        key=f"{key_prefix}_dl",
    )


ORDERS_CSV_NAME = "stocks_orders.csv"


def load_open_orders(ticker: str) -> pd.DataFrame:
    """
    Open Schwab orders for one ticker, read from stocks_orders.csv.
    Reuses the cached load_csv_path loader. Returns an empty frame if the
    file is absent (job hasn't run yet) or the ticker has no open orders.
    """
    path = REPO_ROOT / ORDERS_CSV_NAME
    if not path.exists():
        return pd.DataFrame()
    try:
        df = load_csv_path(str(path), path.stat().st_mtime)
    except Exception:
        return pd.DataFrame()
    if df.empty or "Ticker" not in df.columns:
        return pd.DataFrame()
    return df[df["Ticker"].astype(str).str.upper() == str(ticker).strip().upper()]


def render_open_orders(ticker: str) -> None:
    """
    Render this ticker's open orders. Safe to call from inside a dialog:
    it only writes Streamlit output, never opens one.
    """
    orders = load_open_orders(ticker)
    st.markdown("#### 📋 Open orders")

    if orders.empty:
        st.caption("No open orders for this ticker.")
        return

    cols = [c for c in ["Side", "Order_Type", "Status", "QTY", "Limit_Price",
                        "Stop_Price", "Est_Value", "Duration", "Entered_Time",
                        "Account"] if c in orders.columns]
    st.dataframe(
        orders[cols] if cols else orders,
        use_container_width=True, hide_index=True,
        column_config=build_column_config(orders),
    )

    path = REPO_ROOT / ORDERS_CSV_NAME
    try:
        st.caption(f"{len(orders)} open order(s) · as of {fmt_mtime(path)}")
    except Exception:
        st.caption(f"{len(orders)} open order(s)")


def render_csv_mode():
    """The original CSV viewer, driven from the sidebar."""
    st.title("📊 jobMyTrading CSV Viewer")

    source = st.sidebar.radio(
        "Source", ["Repo files", "Upload a CSV"], horizontal=False, key="csv_source",
    )

    if source == "Repo files":
        files = list_csv_files(REPO_ROOT)
        if not files:
            st.sidebar.error("No .csv files found in this repo.")
            st.stop()

        labels = [str(p.relative_to(REPO_ROOT)) for p in files]
        label_set = set(labels)
        default_present = [name for name in DEFAULT_STACK if name in label_set]

        picked = st.sidebar.multiselect(
            f"CSVs to show ({len(files)} found)",
            options=labels,
            default=default_present,
            help="Add or remove CSVs. They render top-to-bottom in the order chosen.",
        )

        missing_defaults = [name for name in DEFAULT_STACK if name not in label_set]
        if missing_defaults:
            st.sidebar.caption("⚠️ Not in repo: " + ", ".join(missing_defaults))

        if not picked:
            st.info("Pick one or more CSVs from the sidebar to get started.")
            st.stop()

        for i, label in enumerate(picked):
            chosen = REPO_ROOT / label
            st.markdown(f"## 📄 {label}")
            if not chosen.exists():
                st.warning(f"`{label}` not found — skipping.")
                st.divider()
                continue
            try:
                df = load_csv_path(str(chosen), chosen.stat().st_mtime)
                render_csv_block(label, df, fmt_mtime(chosen), key_prefix=f"f{i}")
            except Exception as e:
                st.error(f"Failed to load `{label}`: {e}")
            st.divider()

    else:  # Upload
        up = st.sidebar.file_uploader("Drop a CSV", type=["csv"])
        if up is None:
            st.info("Upload a CSV from the sidebar to get started.")
            st.stop()
        try:
            df = load_csv_bytes(up.getvalue(), up.name)
            st.markdown(f"## 📄 {up.name}")
            render_csv_block(up.name, df, "(uploaded)", key_prefix="upload")
        except Exception as e:
            st.error(f"Failed to read upload: {e}")


# =====================================================================
# ===================== TODO VIEWER (new) =============================
# =====================================================================
#
# Todo JSONs live in a SEPARATE private repo (rchak007/todo-data), written
# only by the Pi2 watcher — so there are no two-writer git collisions with
# the trading CSVs in this repo. This tab fetches them at runtime via the
# GitHub Contents API using a READ-ONLY fine-grained token.
#
# Set in Streamlit -> App settings -> Secrets:
#   todo_github_token  = "github_pat_..."     # read-only, todo-data only
#   todo_github_repo   = "rchak007/todo-data"
#   todo_github_branch = "main"
# =====================================================================

# Selector label -> filename in the todo-data repo (synced by the Pi2 watcher).
TODO_AGENTS = {
    "🗂️ Main": "todos.json",
    "🏠 Real Estate": "properties.json",
    "🕉️ Self-Realization": "eternalquest-todos.json",
}

TODO_TOKEN = st.secrets.get("todo_github_token", "")
TODO_REPO = st.secrets.get("todo_github_repo", "")
TODO_BRANCH = st.secrets.get("todo_github_branch", "main")


@st.cache_data(ttl=60, show_spinner=False)
def load_todo_json(repo_file: str):
    """Fetch a todo JSON from the private todo-data repo via the GitHub API."""
    url = f"https://api.github.com/repos/{TODO_REPO}/contents/{repo_file}"
    headers = {
        "Authorization": f"Bearer {TODO_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    resp = requests.get(url, headers=headers, params={"ref": TODO_BRANCH}, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    raw = base64.b64decode(payload["content"]).decode("utf-8")
    data = json.loads(raw)
    if isinstance(data, dict) and "nodes" not in data and "categories" in data:
        data = {"nodes": data["categories"]}
    # Some files carry a SECOND, list-style tree under a top-level "children"
    # key (a different writer's schema). Fold it into nodes{} so nothing hides.
    if isinstance(data, dict) and isinstance(data.get("children"), list):
        data.setdefault("nodes", {})
        _merge_list_children(data["nodes"], data["children"])
    return data


def _merge_list_children(nodes: dict, children_list: list):
    """Fold a list-style [{name, children[], items[]}] tree into the dict nodes{}."""
    for entry in children_list or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name:
            continue
        if name not in nodes:
            nodes[name] = {"children": {}, "items": []}
        node = nodes[name]
        node.setdefault("children", {})
        node.setdefault("items", [])
        next_id = max([it.get("id", 0) for it in node["items"]], default=0)
        for it in entry.get("items", []):
            next_id += 1
            node["items"].append({
                "id": it.get("id", next_id),
                "text": it.get("text", ""),
                "done": it.get("done", False),
            })
        if entry.get("children"):
            _merge_list_children(node["children"], entry["children"])


def _is_structured(node) -> bool:
    """A node using the main 'items[]/children{}' schema."""
    return isinstance(node, dict) and ("items" in node or "children" in node)


def _count_items(node) -> tuple[int, int]:
    """Count (total_tasks, done_tasks) across either data shape."""
    total = done = 0
    if _is_structured(node):
        for it in node.get("items", []):
            total += 1
            if it.get("done"):
                done += 1
        for child in (node.get("children") or {}).values():
            t, d = _count_items(child)
            total += t
            done += d
    elif isinstance(node, dict):
        # Implicit shape: each key is a folder (non-empty dict) or a task (leaf).
        for v in node.values():
            if isinstance(v, dict) and len(v) > 0:
                t, d = _count_items(v)
                total += t
                done += d
            else:
                total += 1  # leaf task; no done-state in this shape
    return total, done


def _total_stats(nodes: dict):
    total = done = folders = 0
    for node in nodes.values():
        folders += 1
        t, d = _count_items(node)
        total += t
        done += d
    return total, done, total - done, folders


def _render_node(name: str, node, depth: int = 0):
    total, done = _count_items(node)
    badge = f"  `{done}/{total}`" if total else ""

    with st.expander(f"📁 **{name}**{badge}", expanded=(depth == 0)):
        if _is_structured(node):
            # Main shape: explicit items with done flags.
            for it in node.get("items", []):
                check = "✅" if it.get("done") else "⬜"
                line = f"{check} `#{it.get('id', '')}` {it.get('text', '')}"
                st.markdown(f"~~{line}~~" if it.get("done") else line)
            for child_name in sorted(node.get("children", {}).keys()):
                _render_node(child_name, node["children"][child_name], depth + 1)
        elif isinstance(node, dict):
            # Implicit shape: keys are folders (non-empty dict) or tasks (leaf).
            folders = {k: v for k, v in node.items()
                       if isinstance(v, dict) and len(v) > 0}
            tasks = [k for k, v in node.items() if k not in folders]
            for task in tasks:
                st.markdown(f"⬜ {task}")
            for child_name in sorted(folders.keys()):
                _render_node(child_name, folders[child_name], depth + 1)
        else:
            st.markdown(f"⬜ {node}")


def _render_tree(data: dict):
    nodes = data.get("nodes", {})
    total, done, pending, folders = _total_stats(nodes)
    m1, m2, m3 = st.columns(3)
    m1.metric("Open", pending)
    m2.metric("Done", done)
    m3.metric("Folders", folders)
    st.divider()
    if not nodes:
        st.info("No items yet.")
        return
    for name in sorted(nodes.keys()):
        _render_node(name, nodes[name])


def _render_generic(data):
    """Fallback for files that aren't the node/children/items tree."""
    st.info("This file isn't in the standard tree format — showing raw structure.")
    if isinstance(data, list):
        st.caption(f"{len(data)} records")
        for i, rec in enumerate(data):
            label = ""
            if isinstance(rec, dict):
                label = rec.get("name") or rec.get("address") or rec.get("title") or f"Record {i + 1}"
            with st.expander(str(label) or f"Record {i + 1}"):
                st.json(rec)
    elif isinstance(data, dict):
        for key, val in data.items():
            with st.expander(str(key)):
                st.json(val)
    else:
        st.json(data)


def render_todo_mode():
    hdr_l, hdr_r = st.columns([4, 1])
    with hdr_l:
        st.title("🌳 Todos")
        st.caption("Read-only · edits via Telegram bots or the home web UI")
    with hdr_r:
        st.write("")  # vertical nudge to align button with title
        refresh = st.button("🔄 Refresh", key="todo_refresh_top",
                            use_container_width=True,
                            help="Fetch the latest from todo-data now")

    if not TODO_TOKEN or not TODO_REPO:
        st.error(
            "Todo secrets missing. In App settings → Secrets add "
            "`todo_github_token` and `todo_github_repo`."
        )
        st.stop()

    choice = st.sidebar.radio(
        "Agent", list(TODO_AGENTS.keys()), key="todo_agent",
    )
    # Sidebar reload mirrors the top button.
    if st.sidebar.button("🔄 Reload todos", key="todo_refresh_side"):
        refresh = True

    if refresh:
        # Clear ONLY the todo fetch cache — leave the trading CSV caches alone.
        load_todo_json.clear()
        st.rerun()

    repo_file = TODO_AGENTS[choice]

    try:
        data = load_todo_json(repo_file)
    except requests.HTTPError as e:
        code = e.response.status_code
        if code == 404:
            st.warning(
                f"`{repo_file}` isn't in todo-data yet — the Pi2 watcher may not "
                f"have synced this agent. Other tabs still work."
            )
        else:
            st.error(f"GitHub fetch failed ({code}). Check the todo token/repo.")
        st.stop()
    except Exception as e:
        st.error(f"Could not load `{repo_file}`: {e}")
        st.stop()

    st.caption(f"Showing **{choice}** · loaded {datetime.now(tz=PST):%Y-%m-%d %H:%M:%S %Z}")

    if isinstance(data, dict) and "nodes" in data:
        _render_tree(data)
    else:
        _render_generic(data)


# =====================================================================
# ============================ ROUTER =================================
# =====================================================================

st.sidebar.title("jobMyTrading")
mode = st.sidebar.radio(
    "View",
    ["📊 Trading CSVs", "🌳 Todos"],
    key="app_mode",
)
st.sidebar.caption(f"Repo root:\n`{REPO_ROOT}`")

# Quick visibility into whether the journal popup is live — saves a
# round of guessing when secrets are missing on a fresh deploy.
if gsheet_notes is None:
    st.sidebar.caption("📓 Journal: module not found")
elif gsheet_notes.available():
    st.sidebar.caption(f"📓 Journal: on · tab `{gsheet_notes.tab_name()}`")
else:
    st.sidebar.caption("📓 Journal: off · check gsheets secrets")

st.sidebar.divider()

if mode == "📊 Trading CSVs":
    render_csv_mode()
else:
    render_todo_mode()