"""
Part 2: GitHub MCP Integration (+ Bonus: tool search with defer_loading)

Required: get_github_mcp_toolset()            -- direct configuration in Python
Optional: get_github_mcp_toolset_from_config() -- config/mcp_servers.json
Bonus:    get_github_mcp_toolset_deferred() + search_github_tools()

Design notes
------------
* GitHubMcpToolset is a thin McpToolset subclass that adds two things:
  1. Resilience: if the MCP server cannot start (no Node.js, bad token,
     network down) it logs the error and exposes no GitHub tools, instead of
     crashing the whole agent. Calendar tools keep working.
  2. defer_loading: when True, the GitHub tools are NOT sent to the model up
     front. Only ALWAYS_LOADED_GITHUB_TOOLS are visible; every other tool is
     hidden until search_github_tools() finds it and records its name in the
     session state. ADK re-resolves toolsets before every model call, so a
     tool found by search becomes callable on the very next step.
"""

import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.tool_context import ToolContext
from mcp import StdioServerParameters

load_dotenv()

logger = logging.getLogger(__name__)

# Path to MCP server configuration (file-based option)
MCP_CONFIG_PATH = Path(__file__).parent.parent / "config" / "mcp_servers.json"

GITHUB_MCP_PACKAGE = "@modelcontextprotocol/server-github"
# First run downloads the npm package, so allow more than the 5s default.
MCP_STARTUP_TIMEOUT_SECONDS = float(os.getenv("MCP_STARTUP_TIMEOUT", "60"))

# Bonus: the 1-2 GitHub tools that are always loaded (most frequent requests).
ALWAYS_LOADED_GITHUB_TOOLS = ["search_repositories", "list_issues"]
# Session-state key holding the names of tools activated by search_github_tools.
LOADED_TOOLS_STATE_KEY = "github_tools_loaded"


def _get_github_token() -> str:
    token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "").strip()
    if not token or token.startswith(("ghp_your", "ghp_xxx")):
        raise ValueError(
            "GITHUB_PERSONAL_ACCESS_TOKEN not set in .env "
            "(see docs/github_mcp_setup.md)"
        )
    return token


def _github_server_params(token: str) -> StdioServerParameters:
    return StdioServerParameters(
        command="npx",  # the MCP client resolves npx.cmd on Windows
        args=["-y", GITHUB_MCP_PACKAGE],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": token},
    )


def _safe_errlog():
    """Where the MCP server's stderr goes.

    Jupyter/Colab replace sys.stderr with an object that has no real file
    descriptor, and starting the server subprocess then fails with 'fileno'.
    In that case write the server's log to a temp file instead.
    """
    try:
        sys.stderr.fileno()
        return sys.stderr
    except (AttributeError, OSError, ValueError):
        return open(os.path.join(tempfile.gettempdir(), "github_mcp_server.log"), "a")


class GitHubMcpToolset(McpToolset):
    """McpToolset with graceful failure and optional defer_loading."""

    def __init__(
        self,
        *,
        defer_loading: bool = False,
        always_loaded: Optional[list[str]] = None,
        **kwargs,
    ):
        if defer_loading:
            kwargs["tool_filter"] = self._deferred_tool_filter
        kwargs.setdefault("errlog", _safe_errlog())
        super().__init__(**kwargs)
        self.defer_loading = defer_loading
        self.always_loaded = set(always_loaded or ALWAYS_LOADED_GITHUB_TOOLS)
        self.last_error: Optional[str] = None
        # session id -> tool names activated by search (visible immediately,
        # before the session-state delta is committed at the end of the step)
        self._activated: dict[str, set[str]] = {}
        if defer_loading:
            # ADK caches a toolset's tools for the whole invocation; turn that
            # off so tools activated by search appear on the next model step.
            self._use_invocation_cache = False

    def activate_tools(self, session_id: str, names: list[str]) -> None:
        self._activated.setdefault(session_id, set()).update(names)

    def _deferred_tool_filter(
        self, tool: BaseTool, readonly_context: Optional[ReadonlyContext] = None
    ) -> bool:
        # No context = an internal catalog lookup (search_github_tools): show all.
        if readonly_context is None:
            return True
        loaded = set(readonly_context.state.get(LOADED_TOOLS_STATE_KEY, []) or [])
        loaded |= self._activated.get(readonly_context.session.id, set())
        return tool.name in self.always_loaded or tool.name in loaded

    async def get_tools(self, readonly_context: Optional[ReadonlyContext] = None):
        try:
            tools = await super().get_tools(readonly_context)
            self.last_error = None
            return tools
        except Exception as e:  # server failed to start / auth failed
            self.last_error = f"{type(e).__name__}: {e}"
            logger.error("GitHub MCP server unavailable: %s", self.last_error)
            return []


# =============================================================================
# REQUIRED: Direct Configuration
# =============================================================================
def get_github_mcp_toolset() -> McpToolset:
    """Create an McpToolset connected to the GitHub MCP server (all tools loaded)."""
    token = _get_github_token()
    return GitHubMcpToolset(
        connection_params=StdioConnectionParams(
            server_params=_github_server_params(token),
            timeout=MCP_STARTUP_TIMEOUT_SECONDS,
        )
    )


# =============================================================================
# OPTIONAL: File-based Configuration
# =============================================================================
def load_mcp_config() -> dict:
    """Load MCP server configuration from JSON file."""
    if not MCP_CONFIG_PATH.exists():
        raise FileNotFoundError(f"MCP config not found: {MCP_CONFIG_PATH}")

    with open(MCP_CONFIG_PATH) as f:
        config = json.load(f)

    # Replace environment variable placeholders
    github_config = config.get("mcpServers", {}).get("github", {})
    env = github_config.get("env", {})
    for key, value in env.items():
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            env[key] = os.getenv(env_var, "")

    return config


def get_github_mcp_toolset_from_config() -> McpToolset:
    """Create the GitHub McpToolset from config/mcp_servers.json."""
    config = load_mcp_config()
    github = config.get("mcpServers", {}).get("github")
    if not github:
        raise ValueError("No 'github' server defined in config/mcp_servers.json")

    token = github.get("env", {}).get("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not token:
        raise ValueError("GITHUB_PERSONAL_ACCESS_TOKEN not set in .env")

    server_params = StdioServerParameters(
        command=github["command"],
        args=github["args"],
        env=github["env"],
    )
    return GitHubMcpToolset(
        connection_params=StdioConnectionParams(
            server_params=server_params, timeout=MCP_STARTUP_TIMEOUT_SECONDS
        )
    )


# =============================================================================
# BONUS (+25 points) - Tool Search Pattern
# =============================================================================
# Without defer_loading the model receives every GitHub tool schema (26 tools)
# on every request. With defer_loading it receives search_github_tools + the
# 2 always-loaded tools, and pulls in other tools only when needed.

_deferred_toolset: Optional[GitHubMcpToolset] = None

# Common user words -> words used in GitHub MCP tool names/descriptions.
_SYNONYMS = {
    "repo": "repository", "repos": "repository", "pr": "pull", "prs": "pull",
    "readme": "file", "contents": "file", "content": "file", "comment": "comment",
    "bug": "issue", "ticket": "issue", "new": "create",
    "make": "create", "edit": "update", "change": "update",
    "find": "search", "lookup": "search", "show": "get", "read": "get",
    "branches": "branch", "commits": "commit", "merge": "merge",
}


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = set()
    for w in words:
        w = w[:-1] if len(w) > 3 and w.endswith("s") else w
        out.add(_SYNONYMS.get(w, w))
    return out


def get_github_mcp_toolset_deferred() -> McpToolset:
    """Create McpToolset with defer_loading for on-demand tool discovery."""
    global _deferred_toolset
    token = _get_github_token()

    _deferred_toolset = GitHubMcpToolset(
        connection_params=StdioConnectionParams(
            server_params=_github_server_params(token),
            timeout=MCP_STARTUP_TIMEOUT_SECONDS,
        ),
        defer_loading=True,  # Key: tools loaded on-demand
        always_loaded=ALWAYS_LOADED_GITHUB_TOOLS,
    )
    return _deferred_toolset


async def search_github_tools(query: str, tool_context: ToolContext) -> dict:
    """Search the GitHub tool catalog by keyword and load the matching tools.

    Use this BEFORE any GitHub action that you don't already have a tool for,
    e.g. creating an issue, reading a file/README, pull requests, commits,
    branches or comments. Matching tools become available to call on your
    next step.

    Args:
        query: What you want to do, e.g. "create issue", "file contents",
            "pull request", "list commits".

    Returns:
        dict with 'status', 'loaded_tools' (name + short description) and
        'message'; or 'status': 'error' with 'available_tools' to try instead.
    """
    try:
        if not query or not query.strip():
            return {"status": "error", "message": "Please give a search keyword, e.g. 'create issue'."}
        if _deferred_toolset is None:
            return {"status": "error",
                    "message": "GitHub tool search is not configured (no GitHub token?)."}

        catalog = await _deferred_toolset.get_tools(None)  # full catalog, no filter
        if not catalog:
            return {"status": "error",
                    "message": "GitHub tools are unavailable right now: "
                               f"{_deferred_toolset.last_error or 'MCP server not reachable'}."}

        wanted = _tokens(query)
        scored = []
        for tool in catalog:
            name_hits = len(wanted & _tokens(tool.name.replace("_", " ")))
            desc_hits = len(wanted & _tokens(tool.description or ""))
            score = name_hits * 3 + desc_hits
            if score:
                scored.append((score, tool))
        scored.sort(key=lambda x: (-x[0], x[1].name))
        matches = [t for _, t in scored[:3]]  # keep the context small

        if not matches:
            return {
                "status": "error",
                "message": f"No GitHub tool matches '{query}'. Try other words.",
                "available_tools": sorted(t.name for t in catalog),
            }

        loaded = set(tool_context.state.get(LOADED_TOOLS_STATE_KEY, []) or [])
        loaded.update(t.name for t in matches)
        tool_context.state[LOADED_TOOLS_STATE_KEY] = sorted(loaded)  # persists in session
        _deferred_toolset.activate_tools(tool_context.session.id, sorted(loaded))

        return {
            "status": "success",
            "loaded_tools": [
                {"name": t.name, "description": (t.description or "")[:120]} for t in matches
            ],
            "message": "These tools are now loaded. Call the best one directly.",
        }
    except Exception as e:
        return {"status": "error", "message": f"Tool search failed: {type(e).__name__}: {e}"}


# =============================================================================
# Default toolset list used by agent.py
# =============================================================================
def _build_mcp_tools() -> list:
    try:
        return [get_github_mcp_toolset()]
    except ValueError as e:
        logger.warning("GitHub MCP disabled: %s", e)
        return []


mcp_tools = _build_mcp_tools()
