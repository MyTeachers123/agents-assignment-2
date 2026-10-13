"""
Google Workspace Assistant - Main Agent Definition

Part 1: Calendar tools (Option A) + system instruction
Part 2: GitHub McpToolset
Bonus:  create_agent_with_tool_search() -- GitHub tools loaded on demand
"""

import os
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from config.settings import Settings
from tools.calendar_tools import calendar_tools
from tools.mcp_tools import (
    ALWAYS_LOADED_GITHUB_TOOLS,
    get_github_mcp_toolset_deferred,
    mcp_tools,
    search_github_tools,
)


CALENDAR_INSTRUCTION = """You are a friendly, efficient Google Workspace assistant that manages the
user's Google Calendar. All times are in the user's own time zone (get_current_datetime
tells you which one).

## Calendar tools - which one to use
- get_current_datetime: call FIRST whenever the user uses a relative date
  ("today", "tomorrow", "next Monday", "next week") so you can compute the exact
  YYYY-MM-DD date. Never guess today's date.
- list_upcoming_events: "What's on my calendar?", "What meetings do I have
  tomorrow?". Also use it to look up an event_id before rescheduling.
- find_available_slots: "When am I free?", "Find an hour for a meeting next week".
  If the result has more_available=true, say these are the earliest options and
  offer to look for later ones.
- check_conflicts: "Am I free Tuesday at 3pm?", "Does this clash with anything?".
- create_event: schedule a new event - only after the user confirms.
- reschedule_event: move an existing event to a new time - only after the user confirms.

## Rules
1. Convert natural language to tool arguments: dates as YYYY-MM-DD, times as
   24-hour "YYYY-MM-DD HH:MM" (e.g. "Monday at 2pm" -> "2026-10-12 14:00").
   Default meeting length is 60 minutes unless the user says otherwise.
2. ALWAYS confirm before changing the calendar. Before create_event or
   reschedule_event, restate title, date, start-end time and guests in one line
   and ask "Shall I go ahead?". Act only after the user says yes.
3. If a tool returns status "conflict", tell the user which events overlap and
   offer alternatives (use find_available_slots). Only pass allow_conflicts=True
   if the user explicitly says to double-book.
4. If a tool returns status "error", explain the problem in plain words, include
   the 'hint' if there is one, and suggest a next step. Never invent events,
   times or event ids; never claim an action succeeded unless status is "success".
5. If a request is ambiguous (missing date, time, or which event), ask one short
   clarifying question instead of guessing.
6. Keep answers short. List events as bullets: "Mon Oct 12, 2:00-3:00 PM - Team sync".
"""

GITHUB_INSTRUCTION = """
## GitHub (via the GitHub MCP server)
You can also work with the user's GitHub account:
- "List my GitHub repositories" -> search_repositories with query "user:{username}".
  {username_note}
- "Show open issues in owner/repo" -> list_issues with owner, repo and state "open".
- "Create an issue in owner/repo about X" -> create_issue with a clear title and a
  2-3 sentence body. Confirm the title and repository with the user first.
- "Show the README of owner/repo" -> get_file_contents with path "README.md".
Repositories are written as owner/repo. If the user gives only a repo name, assume
the owner is the user's GitHub username. If a GitHub tool fails (bad credentials,
not found, rate limit), say what went wrong and how to fix it. If no GitHub tools
are available, tell the user GitHub is not connected (missing token or Node.js).
"""

TOOL_SEARCH_INSTRUCTION = """
## Finding GitHub tools (tool search)
To save context, only these GitHub tools are loaded up front: {always_loaded}.
For any other GitHub action (create issue, read files/README, pull requests,
commits, branches, comments, ...) FIRST call search_github_tools with a short
query such as "create issue" or "file contents". The matching tools are loaded
for your next step; then call the best one. Do not tell the user about this
internal search step.
"""


def _github_instruction() -> str:
    username = os.getenv("GITHUB_USERNAME", "").strip()
    return GITHUB_INSTRUCTION.format(
        username=username or "<username>",
        username_note=(
            f"The user's GitHub username is {username}."
            if username else
            "If you don't know the user's GitHub username, ask for it once."
        ),
    )


def create_agent() -> LlmAgent:
    """Create the Workspace Assistant agent."""
    settings = Settings()

    # Part 1: calendar tools. Part 2: GitHub McpToolset (if a token is configured).
    tools = list(calendar_tools) + list(mcp_tools)
    github_enabled = any(isinstance(t, McpToolset) for t in tools)

    instruction = CALENDAR_INSTRUCTION
    if github_enabled:
        instruction += _github_instruction()

    return LlmAgent(
        name="workspace_assistant",
        model=settings.model_name,
        description="Manages Google Calendar and GitHub through natural language.",
        instruction=instruction,
        tools=tools,
    )


def create_agent_with_tool_search() -> LlmAgent:
    """BONUS: Create agent with defer_loading for tool search.

    Only search_github_tools + ALWAYS_LOADED_GITHUB_TOOLS are sent to the model
    up front; other GitHub tools are loaded when search_github_tools finds them.
    """
    settings = Settings()

    tools = list(calendar_tools)
    instruction = CALENDAR_INSTRUCTION
    try:
        tools += [get_github_mcp_toolset_deferred(), search_github_tools]
        instruction += _github_instruction() + TOOL_SEARCH_INSTRUCTION.format(
            always_loaded=", ".join(ALWAYS_LOADED_GITHUB_TOOLS)
        )
    except ValueError as e:  # no GitHub token: calendar-only agent
        print(f"Warning: GitHub disabled ({e})")

    return LlmAgent(
        name="workspace_assistant",
        model=settings.model_name,
        description="Manages Google Calendar and GitHub (tools loaded on demand).",
        instruction=instruction,
        tools=tools,
    )
