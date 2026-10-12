"""
Bonus: measure context bloat WITHOUT vs WITH defer_loading.

Runs the same query through create_agent() (all GitHub tools loaded) and
create_agent_with_tool_search() (tool search, defer_loading=True) and reports
for the FIRST model call of each run:

  * how many tools were declared to the model,
  * the size of the tool declarations (characters and ~tokens, chars/4),
  * the real prompt token count reported by Gemini (usage_metadata).

Usage (from workspace_assistant/, with .env containing GOOGLE_API_KEY and
GITHUB_PERSONAL_ACCESS_TOKEN):

    python -m scripts.compare_context_tokens
    python -m scripts.compare_context_tokens "Show open issues in octocat/Hello-World"
"""

import asyncio
import json
import sys

from google.adk.runners import InMemoryRunner
from google.genai import types

from agent import create_agent, create_agent_with_tool_search
from tools.calendar_tools import calendar_tools

CALENDAR_NAMES = {f.__name__ for f in calendar_tools}

DEFAULT_QUERY = "List my GitHub repositories"


async def measure(agent, query: str) -> dict:
    first_call: dict = {}

    def capture(callback_context, llm_request):
        if not first_call:
            decls = []
            for tool in llm_request.config.tools or []:
                for fd in getattr(tool, "function_declarations", None) or []:
                    decls.append(fd.model_dump(exclude_none=True, mode="json"))
            chars = len(json.dumps(decls))
            gh_chars = len(json.dumps([d for d in decls if d.get("name") not in CALENDAR_NAMES]))
            first_call.update(
                tools=sorted(llm_request.tools_dict.keys()),
                decl_chars=chars,
                decl_tokens_est=chars // 4,
                github_decl_tokens_est=gh_chars // 4,
            )
        return None  # continue to the model

    agent.before_model_callback = capture
    runner = InMemoryRunner(agent=agent)
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="compare"
    )
    prompt_tokens, total_prompt_tokens, answer = None, 0, ""
    async for event in runner.run_async(
        user_id="compare",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=query)]),
    ):
        usage = getattr(event, "usage_metadata", None)
        if usage and usage.prompt_token_count:
            if prompt_tokens is None:
                prompt_tokens = usage.prompt_token_count
            total_prompt_tokens += usage.prompt_token_count
        if event.content and event.content.parts:
            answer += "".join(p.text or "" for p in event.content.parts)

    for tool in agent.tools:  # close MCP server subprocesses
        if hasattr(tool, "close"):
            await tool.close()

    first_call.update(
        first_prompt_tokens=prompt_tokens,
        total_prompt_tokens=total_prompt_tokens,
        answer=answer.strip()[:300],
    )
    return first_call


def _pct(a, b):
    return f"{(1 - b / a) * 100:.0f}%" if a and b is not None else "n/a"


async def main():
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    print(f"Query: {query}\n")
    full = await measure(create_agent(), query)
    lazy = await measure(create_agent_with_tool_search(), query)

    rows = [
        ("Tools declared (1st call)", len(full["tools"]), len(lazy["tools"])),
        ("All tool declarations, ~tokens", full["decl_tokens_est"], lazy["decl_tokens_est"]),
        ("GitHub tool declarations, ~tokens", full["github_decl_tokens_est"], lazy["github_decl_tokens_est"]),
        ("Prompt tokens, 1st call (Gemini)", full["first_prompt_tokens"], lazy["first_prompt_tokens"]),
        ("Prompt tokens, whole query (Gemini)", full["total_prompt_tokens"], lazy["total_prompt_tokens"]),
    ]
    print(f"| {'Metric':<36} | {'Without defer_loading':>21} | {'With defer_loading':>18} | {'Reduction':>9} |")
    print(f"|{'-' * 38}|{'-' * 23}|{'-' * 20}|{'-' * 11}|")
    for name, a, b in rows:
        print(f"| {name:<36} | {str(a):>21} | {str(b):>18} | {_pct(a, b):>9} |")

    print("\nTools WITHOUT defer_loading:", ", ".join(full["tools"]))
    print("Tools WITH defer_loading:   ", ", ".join(lazy["tools"]))
    print("\nAnswer (without):", full["answer"])
    print("Answer (with):   ", lazy["answer"])


if __name__ == "__main__":
    asyncio.run(main())
