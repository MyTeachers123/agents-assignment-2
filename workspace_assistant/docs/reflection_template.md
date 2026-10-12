# Assignment 2 Reflection

**Name:** Ying Chen
**Option:** A - Calendar
**Date:** 2026-10-11

---

## Tool Design Decisions

### Tools Implemented
1. **get_current_datetime**: Returns today's date, current time and weekday in the user's time zone, so the agent can turn "tomorrow" or "next Monday" into an exact date instead of guessing.
2. **list_upcoming_events**: Lists events in a date range, with an optional keyword filter. It also provides the `event_id` needed for rescheduling.
3. **find_available_slots**: Finds free slots of a given length across one or more days, within working hours, for the user and optional attendees (using the free/busy API).
4. **check_conflicts**: Checks whether a proposed time overlaps existing events, and whether attendees are busy.
5. **create_event**: Creates an event and emails invitations to attendees. It checks for conflicts first and refuses to double-book unless the user explicitly agrees.
6. **reschedule_event**: Moves an existing event, keeping its original duration if no new end time is given. It also checks conflicts and notifies attendees.

Part 2 adds the GitHub MCP server (`McpToolset`, 26 tools). The bonus adds `search_github_tools`, which loads GitHub tools on demand.

### Why These Tools?
They cover the full scheduling loop a real user goes through:

- **Read:** what do I have? (`list_upcoming_events`)
- **Plan:** when am I free, and does this time clash? (`find_available_slots`, `check_conflicts`)
- **Write:** book it, or move it (`create_event`, `reschedule_event`)

`get_current_datetime` was added after realizing that most calendar requests use relative dates, and an LLM does not reliably know today's date. Read tools and write tools are kept separate, so the agent can always look before it changes anything. The write tools also run their own conflict check, so the safety rule does not depend only on the prompt.

### Description Strategy
- Every docstring starts with one line saying what the tool does, followed by a **"Use this when..."** sentence with real example phrases ("What do I have tomorrow?", "Find a free hour next week"). These are the words users actually type, so the model can match intent to tool.
- Overlapping tools say when **not** to use them. For example, `find_available_slots` says "Do NOT use it to create the event", and `create_event` says "ONLY after the user has confirmed".
- Arguments state the exact format (`YYYY-MM-DD HH:MM`, 24-hour, user's time zone), defaults and valid ranges. This prevented most malformed calls.
- Names follow verb_noun (`find_available_slots`, `reschedule_event`).
- The system instruction repeats the routing as a short "which tool for which request" list. It also adds behavior rules: confirm before writes, handle `conflict` and `error` statuses, and ask one clarifying question when a request is ambiguous.

---

## Challenges Encountered

### Challenge 1: The model in the starter code was retired
- **Problem:** In the Week 2 lab, every agent call failed with `404 NOT_FOUND: models/gemini-2.5-flash is no longer available to new users`.
- **Solution:** I switched to `gemini-3.8-flash`. In this project the model name comes from `MODEL_NAME` in `.env` (via `Settings`), so a future retirement only needs a config change, not a code change.

### Challenge 2: `defer_loading` does not exist in the installed ADK
- **Problem:** The scaffold suggests `McpToolset(..., defer_loading=True)`, but the installed google-adk `McpToolset` has no such parameter (it raises `TypeError`). My first working version also never showed newly found tools in the same turn: the search succeeded, but the model still could not call the tool.
- **Solution:**
  - I wrote `GitHubMcpToolset(McpToolset)`, which accepts `defer_loading=True`. It installs a dynamic `tool_filter` predicate that exposes only the always-loaded tools plus the tools `search_github_tools` has activated for that session.
  - Reading the ADK source showed that toolsets cache their tool list per invocation (`_use_invocation_cache`). I disable that cache for the deferred toolset (ADK's own `SkillToolset` does the same).
  - With that change, a tool found by search is callable on the very next model step. I verified this end-to-end with a mock MCP server: step 1 calls `search_github_tools`, step 2 sees `create_issue` and calls it.

### Challenge 3: The provided CLI failed with "Session not found"
- **Problem:** `python main.py "..."` returned `Error: Session not found: session`. Newer ADK runners no longer create a session implicitly.
- **Solution:** I added one line to `main.py`: `runner.auto_create_session = True`.

### Challenge 4: A failing MCP server could take the whole agent down
- **Problem:** If Node.js is missing, the token is wrong, or `npx` cannot download the package, the GitHub toolset fails while tools are being listed.
- **Solution:**
  - `GitHubMcpToolset.get_tools()` catches the failure, logs it, keeps the error in `last_error`, and returns no GitHub tools. The calendar keeps working, and the instruction tells the agent to say "GitHub is not connected".
  - Without a token, the agent is simply built calendar-only, so `create_agent()` never crashes.
  - I also raised the MCP startup timeout from 5s to 60s, because the first `npx` run downloads the package.

---

## Error Handling Approach

**No tool ever raises.** Every tool returns a structured dict:

- `{"status": "success", ...}`
- `{"status": "conflict", "conflicts": [...]}` for write tools
- `{"status": "error", "message": ..., "hint": ...}`

The `message` explains the problem in plain language, and the `hint` says how to fix it. The system instruction tells the agent to relay both and to never claim success unless the status is `success`.

**Anticipated errors:**

- **Bad input (caught before any API call):** unparseable dates or times, end before start, start in the past, invalid emails, empty titles, out-of-range durations or result counts, and search windows longer than 31 days.
- **Auth:** missing `credentials.json` (points to the setup guide) and an expired or revoked token (delete `token.json` and sign in again).
- **API errors, mapped by HTTP status:**
  - 400: invalid request
  - 401/403: authorization or rate limit
  - 404: event not found, so list events first
  - 409: concurrent edit
  - 410: already deleted
  - 429: wait a minute
  - 5xx: temporary outage
- **Partial information:** if an attendee's free/busy is not shared, the result lists them under `unchecked_attendees` instead of silently treating them as free.
- **Scheduling safety:** write tools return `conflict` instead of double-booking, and the agent must offer alternatives. All-day, "free" and declined events do not count as conflicts.
- **GitHub/MCP:** server start failures degrade gracefully (Challenge 4), and `search_github_tools` returns the list of available tool names when nothing matches.

**Testing:** `tests/test_calendar_offline.py` contains 12 unit tests that run against an in-memory fake Calendar client, so validation, conflict detection, slot search and error messages are tested without a Google account. Both course checks pass:

- `tests.test_tools`: 8/8
- `tests.test_mcp`: all checks pass

---

## Bonus: Tool Search with `defer_loading`

**Implementation:**

- `get_github_mcp_toolset_deferred()` creates a `GitHubMcpToolset` with `defer_loading=True`. Only `search_repositories` and `list_issues` are always loaded, since they are the most common requests.
- `search_github_tools(query)` ranks the full GitHub catalog by keyword overlap with tool names and descriptions, with simple synonyms (repo→repository, PR→pull, README→file). It activates the top 3 matches for the session.
- `create_agent_with_tool_search()` builds the agent with the calendar tools, the deferred toolset and `search_github_tools`, plus an instruction that says to search before using any other GitHub tool.

**Comparison:** measured with `python -m scripts.compare_context_tokens`. The script captures the tool declarations sent on the first model call for the query "List my GitHub repositories", using the real `@modelcontextprotocol/server-github` server (26 tools). Token counts are estimated as JSON characters / 4.

| Mode | Tools Loaded | GitHub tool declarations | All tool declarations |
|------|--------------|--------------------------|-----------------------|
| Without defer_loading | 32 (6 calendar + 26 GitHub) | ~6,478 tokens | ~8,558 tokens |
| With defer_loading | 9 (6 calendar + 2 GitHub + search) | ~601 tokens | ~2,680 tokens |
| **Savings** | **72% fewer tools** | **~91%** | **~69%** |

The GitHub portion drops by about 91%, close to the ~80% the assignment expects. The total drops less, because the six calendar tools (with long, descriptive docstrings) are always loaded.

The trade-off is an extra model step whenever a non-preloaded GitHub tool is needed (search, then call). That step costs about 100–150 tokens of results, which is still far below the ~5,900 tokens saved on every single request.

---

## Ideas for Improvement

1. **Semantic tool search:** replace keyword overlap with embeddings over tool descriptions. Queries like "what changed recently" would then find `list_commits` without synonym lists. I would also "unload" tools that have not been used for several turns.
2. **Public tour booking:** a real use I have for these tools is a "Schedule a tour" flow on a childcare listing page, replacing a third-party booking link. Each listing has its own allowed tour window ("non-holiday Tue/Wed/Thu 17:00–19:00"), and both the parent and the provider get the invitation. For a public page I would not expose the LLM agent directly, to avoid prompt injection, spam and cost. Instead I would reuse the validation logic as a fixed API: validate the chosen time against the parsed schedule, then send an `.ics` invitation, with the LLM used only offline to parse free-text schedules into JSON.
3. **Recurring events and time zones per attendee:** support "every Monday" and show slots in each attendee's own time zone. I would also add `delete_event` behind a stricter confirmation step.

---

## Key Learnings

The biggest lesson is that **tool design is prompt design**. The model only sees names, docstrings and type hints, so a sentence like "Use this when the user asks 'Am I free at 3pm?'" or "Do NOT use this to create the event" changed tool selection more than any wording in the system instruction. Structured return values matter just as much. Because every tool reports `success`, `conflict` or `error` with a hint, the agent can explain failures and recover instead of hallucinating success. Safety rules such as "never double-book" belong in the tool's code, with the prompt only as a second layer.

Second, **context is a budget**. Connecting one MCP server added 26 tool schemas (about 6,500 tokens) to every request, even when the user only asked about their calendar. The tool-search pattern cut the GitHub portion by about 91% at the cost of one extra step when a rare tool is needed. That trade-off, paying a little latency for a much smaller context, will matter more as agents connect to more servers.

Finally, **agent frameworks move fast**, so I need to read the source, not just the examples. In one week the starter model was retired, the runner stopped creating sessions automatically, and the suggested `defer_loading` argument did not exist. Each fix came from reading the actual ADK code (for example, the per-invocation tool cache) and testing against a mock MCP server, rather than trusting the scaffold comments. Keeping the model name in configuration and building tools that fail gracefully made those changes easy to absorb.
