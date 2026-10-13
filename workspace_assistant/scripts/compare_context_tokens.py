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
import os
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

    await _close_tools(agent)

    first_call.update(
        first_prompt_tokens=prompt_tokens,
        total_prompt_tokens=total_prompt_tokens,
        answer=answer.strip()[:300],
    )
    return first_call


RETRY_CODES = {429, 500, 503}
ATTEMPTS_PER_MODEL = int(os.getenv("COMPARE_ATTEMPTS_PER_MODEL", "3"))
SKIP_WORDS = ("image", "tts", "audio", "live", "embedding", "thinking-exp", "preview-native")


class GeminiBusy(Exception):
    pass


def _api_error_code(exc: BaseException):
    """HTTP status of a Gemini API error, even if ADK wrapped it."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        code = getattr(exc, "code", None)
        if isinstance(code, int):
            return code
        exc = exc.__cause__ or exc.__context__
    return None


def _candidate_models(first: str) -> list[str]:
    """The configured model first, then other Flash models this API key can use."""
    models = [first]
    try:
        from google import genai
        client = genai.Client()  # keep a reference: the pager needs an open client
        for m in list(client.models.list()):
            name = (m.name or "").removeprefix("models/")
            if ("flash" in name and name not in models
                    and "generateContent" in (m.supported_actions or [])
                    and not any(w in name for w in SKIP_WORDS)):
                models.append(name)
    except Exception as e:
        print(f"  (could not list other models: {type(e).__name__})")
    return models


async def measure_with_retry(make_agent, label: str, query: str, model: str) -> dict:
    """Gemini's free tier often answers 503 (busy) or 429 (rate limit).
    Retry with a growing wait; raise GeminiBusy if this model stays busy."""
    for attempt in range(1, ATTEMPTS_PER_MODEL + 1):
        agent = make_agent()
        agent.model = model
        try:
            return await measure(agent, query)
        except Exception as e:
            code = _api_error_code(e)
            await _close_tools(agent)
            if code not in RETRY_CODES:
                raise
            if attempt == ATTEMPTS_PER_MODEL or (code == 429 and "PerDay" in str(e)):
                raise GeminiBusy(code) from e
            wait = 20 * attempt
            print(f"  [{label}] {model} returned {code} (attempt {attempt}/{ATTEMPTS_PER_MODEL}); "
                  f"retrying in {wait}s...", flush=True)
            await asyncio.sleep(wait)


async def _close_tools(agent):
    for tool in agent.tools:  # close MCP server subprocesses
        if hasattr(tool, "close"):
            try:
                await tool.close()
            except Exception:
                pass


def _pct(a, b):
    return f"{(1 - b / a) * 100:.0f}%" if a and b is not None else "n/a"


async def main():
    import importlib.metadata as md
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    print(f"Query: {query}", flush=True)
    full = lazy = None
    for model in _candidate_models(os.getenv("MODEL_NAME", "gemini-3.8-flash")):
        try:
            print(f"Trying model {model}: WITHOUT defer_loading...", flush=True)
            full = await measure_with_retry(create_agent, "without", query, model)
            print(f"Trying model {model}: WITH defer_loading...", flush=True)
            lazy = await measure_with_retry(create_agent_with_tool_search, "with", query, model)
            break
        except GeminiBusy as e:
            print(f"  {model} is busy or out of quota ({e}); trying the next model.\n", flush=True)
    if full is None or lazy is None:
        print("All Gemini models were busy or out of quota. Please try again later.")
        sys.exit(1)

    print(f"\nQuery: {query}")
    print(f"Model: {model} | google-adk {md.version('google-adk')} | mcp {md.version('mcp')}\n")

    rows = [
        ("Tools declared (1st call)", len(full["tools"]), len(lazy["tools"])),
        ("All tool declarations, ~tokens (chars/4)", full["decl_tokens_est"], lazy["decl_tokens_est"]),
        ("GitHub tool declarations, ~tokens (chars/4)", full["github_decl_tokens_est"], lazy["github_decl_tokens_est"]),
        ("**Prompt tokens, 1st call (Gemini)**", full["first_prompt_tokens"], lazy["first_prompt_tokens"]),
        ("**Prompt tokens, whole query (Gemini)**", full["total_prompt_tokens"], lazy["total_prompt_tokens"]),
    ]
    table = ["| Metric | Without defer_loading | With defer_loading | Reduction |",
             "|---|---:|---:|---:|"]
    table += [f"| {n} | {a} | {b} | {_pct(a, b)} |" for n, a, b in rows]
    print("\n".join(table))

    print("\nTools WITHOUT defer_loading:", ", ".join(full["tools"]))
    print("Tools WITH defer_loading:   ", ", ".join(lazy["tools"]))
    print("\nAnswer (without):", full["answer"])
    print("Answer (with):   ", lazy["answer"])

    import datetime
    block = "\n".join([
        f"Measured on {datetime.date.today().isoformat()} with `python -m scripts.compare_context_tokens`.",
        "",
        f"- **Query:** \"{query}\"",
        f"- **Model:** `{model}` | **google-adk** {md.version('google-adk')} | **mcp** {md.version('mcp')}",
        "- **Prompt tokens** are Gemini's own `usage_metadata.prompt_token_count`. "
        "\"Whole query\" adds up every model call needed to answer the query.",
        "",
        *table,
    ])
    if _write_into_reflection(block):
        print("\nReflection updated: docs/reflection_template.md")


START_MARK = "<!-- TOKEN_RESULTS:START -->"
END_MARK = "<!-- TOKEN_RESULTS:END -->"


def _write_into_reflection(block: str) -> bool:
    """Put the measured table between the markers in the reflection."""
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "docs" / "reflection_template.md"
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if START_MARK not in text or END_MARK not in text:
        return False
    head, rest = text.split(START_MARK, 1)
    _, tail = rest.split(END_MARK, 1)
    path.write_text(f"{head}{START_MARK}\n{block}\n{END_MARK}{tail}", encoding="utf-8")
    return True


if __name__ == "__main__":
    asyncio.run(main())
