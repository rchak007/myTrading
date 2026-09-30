# Documentation

**Start here.** Two entry points depending on what you want:

| | |
|---|---|
| **[HOW-IT-WORKS.md](HOW-IT-WORKS.md)** | What the system is for, the daily rhythm, the sheets, how orders and seed money work. Non-technical. |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | Every module, the data flow, state files, locking, and the failure modes this system has actually had. |

---

## Reference

| doc | covers |
|---|---|
| **[OPERATIONS.md](OPERATIONS.md)** | The cron schedule, every log file, lock rules, `health_check.py`, `smoke_test.py`, and a symptom→log table |
| **[PROJECT_PLAN.md](PROJECT_PLAN.md)** | **Living document.** Open items, known defects, decisions waiting on Chakravarti. Ask "what's still open?" and this is the answer |

## Subsystems

| doc | covers |
|---|---|
| [ordersSheetDesign-9-18-26.md](ordersSheetDesign-9-18-26.md) | The orders sheet and the execution engine. **Current and authoritative** for everything order-related |
| [orderExecutionDesign-9-7-26.md](orderExecutionDesign-9-7-26.md) | The original hardened-engine design. **Partly superseded** — its HMAC scheme was dropped; its ledger, state machine and kill switch were built as written |
| [remoteOpsGuide-9-7-26.md](remoteOpsGuide-9-7-26.md) | Driving Pi 1 from a phone. Verb allowlist, row rules, Schwab re-auth |
| [CASH_RESERVE_HANDOFF.md](CASH_RESERVE_HANDOFF.md) | Per-ticker cash reserves: the ledger, the policies, the gate |
| [googleDriveSheetsAccess-9-14-26.md](googleDriveSheetsAccess-9-14-26.md) | Service accounts, Drive shares, and the traps in both |
| [WALL-STREET-LEVEL-STOCK-ANALYSIS.md](WALL-STREET-LEVEL-STOCK-ANALYSIS.md) | The analysis framework behind the signals |
| [integrationsGuide-3-30-26.md](integrationsGuide-3-30-26.md) | Notion, Streamlit and other outbound integrations |

---

## Conventions

**Dated filenames** (`-9-18-26`) are documents written at a point in time and
revised in place as the thing they describe changes. Undated ones
(`OPERATIONS`, `PROJECT_PLAN`, and these two overviews) are living documents
edited continuously.

**`PROJECT_PLAN.md` distinguishes three kinds of entry** because they need
different responses: 💡 idea, 🐞 defect, ❓ a decision only Chakravarti can
make. Finished items move to §8 rather than being deleted — the history of what
was fixed is worth as much as the list of what is left.

**Superseded sections are marked and kept**, usually in a `<details>` block.
Knowing what was tried and rejected is how the same wrong turn gets avoided
twice.
