"""Neurax MCP server over stdio: a thin layer over memory.py.

Run with: uv run neurax-server  (needs NEURAX_USER_EMAIL)

Never write to stdout here: the stdio transport uses it for protocol messages.
Tool docstrings are the descriptions agents read; keep them in sync with
docs/ARCHITECTURE.md section 3.
"""

import functools
import json
import logging
import sys
from collections.abc import Callable
from contextlib import redirect_stdout
from typing import Any, Literal, ParamSpec
from uuid import UUID

import psycopg
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from neurax import config, embeddings, memory

log = logging.getLogger("neurax.server")

MemoryType = Literal["decision", "constraint", "fact", "task"]
P = ParamSpec("P")

CONFLICT_HINT = "If the new memory replaces any of these, call supersede(old_id, new_id)."

mcp = MCPServer(
    "neurax",
    instructions=(
        "Neurax is decision memory for this user's projects. It keeps the current decisions, "
        "constraints, facts and tasks, and the history of what replaced what.\n"
        "1. Start of session: call get_project_context(project) and treat the result as the "
        "project's current state.\n"
        "2. During work: call search_context when a past decision matters.\n"
        "3. End of session: call save_context once per decision, constraint, fact or task. "
        "Review possible_conflicts in each response and call supersede(old_id, new_id) where "
        "the new memory replaces an old one."
    ),
)

_user_id: UUID | None = None  # set once in main() from NEURAX_USER_EMAIL, never from tool args


def _user() -> UUID:
    assert _user_id is not None, "server not started via main()"
    return _user_id


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def tool(fn: Callable[P, str]) -> Callable[P, str]:
    """Register fn as an MCP tool (docstring = description) with readable errors."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> str:
        try:
            return fn(*args, **kwargs)
        except memory.NeuraxError as e:
            # Anticipated: the agent gets the message and can act on it.
            raise ToolError(str(e)) from None
        except psycopg.OperationalError:
            log.exception("database error")
            raise ToolError(
                "database unreachable; is Postgres running? (docker compose up -d)"
            ) from None
        # Anything else: the SDK logs the traceback to stderr and the agent only sees
        # "Error executing tool <name>", so no internals leak.

    mcp.tool(structured_output=False)(wrapper)
    return wrapper


# --------------------------------------------------------------------------- tools


@tool
def list_projects() -> str:
    """List your Neurax projects (slug, name, description, number of current memories).

    Call this when you don't know the project slug, or when another tool says a project
    was not found. Use the slug as the `project` argument of the other tools.
    """
    return _json(memory.list_projects(_user()))


@tool
def create_project(slug: str, name: str, description: str | None = None) -> str:
    """Create a project so decisions can be saved for it.

    Call this once per codebase, the first time you work on it (check list_projects
    first). slug: lowercase letters, digits and '-', e.g. "neurax"; usually the repo name.
    """
    return _json(memory.create_project(_user(), slug, name, description))


@tool
def get_project_context(project: str, limit: int = 30) -> str:
    """Get the project's current state: call this at the START of every session.

    Returns pinned memories first, then current constraints and decisions (newest first),
    then open tasks. Only current memories: anything that was replaced is left out, so
    you can treat every item as true now. Facts that aren't pinned are not included;
    find them with search_context.
    """
    return _json(memory.get_project_context(_user(), project, limit))


@tool
def save_context(
    project: str,
    content: str,
    type: MemoryType,
    topic: str,
    rationale: str | None = None,
    supersedes_id: str | None = None,
    source_tool: str = "unknown",
) -> str:
    """Save one decision, constraint, fact or task: call this at the END of a session.

    Call once per item, and keep each memory to ONE statement: split a paragraph with
    three decisions into three calls.
    - type: decision (a choice made), constraint (a rule that must hold), fact (true
      about the project), task (open work).
    - topic: short grouping key, e.g. "database" or "auth/session-store". Reuse the
      topic of the memory this one relates to, so changes stay in one history.
    - rationale: why. Strongly encouraged for decisions.
    - supersedes_id: id of the memory this one replaces (e.g. the old decision).
    - source_tool: your client name, e.g. "claude-code".

    The response lists possible_conflicts: current memories on the same topic or with
    similar meaning. If the new memory replaces any of them, call supersede(old_id,
    new_id). If identical content already exists, the existing memory is returned with
    duplicate=true and nothing is saved or superseded.
    """
    result = memory.save(
        _user(), project, content, type, topic, rationale, supersedes_id, source_tool
    )
    if result["possible_conflicts"]:
        result["hint"] = CONFLICT_HINT
    return _json(result)


@tool
def supersede(old_id: str, new_id: str) -> str:
    """Mark memory old_id as replaced by memory new_id (both current, same project).

    Call this after save_context when possible_conflicts shows an older memory that the
    new one replaces. The old memory stays in history but no longer appears as current;
    a pin moves to the new memory.
    """
    return _json(memory.supersede(_user(), old_id, new_id))


@tool
def search_context(
    project: str,
    query: str,
    limit: int = 8,
    include_history: bool = False,
    types: list[MemoryType] | None = None,
) -> str:
    """Search the project's memories by meaning and keywords.

    Call this during a session when a past decision, constraint or fact matters for
    what you're doing ("what did we decide about auth?"). Returns current memories only;
    set include_history=true to also see replaced ones (current ones rank higher).
    types filters by kind, e.g. ["decision", "constraint"].
    """
    return _json(memory.search(_user(), project, query, limit, include_history, types))


@tool
def decision_history(project: str, topic: str) -> str:
    """Show how a topic evolved: every memory in it, oldest to newest.

    Call this when you need to know how the project ended up with its current decision
    ("why don't we use Redis any more?"). Each item has its status and supersedes_id,
    the memory it replaced.
    """
    return _json(memory.history(_user(), project, topic))


@tool
def pin(memory_id: str) -> str:
    """Pin a current memory so it always comes first in get_project_context.

    Use for the few things every session must see (core constraints, the main stack).
    Replaced memories can't be pinned; pin their successor.
    """
    return _json(memory.pin(_user(), memory_id))


@tool
def unpin(memory_id: str) -> str:
    """Remove a pin set with pin(). The memory stays current."""
    return _json(memory.unpin(_user(), memory_id))


# --------------------------------------------------------------------------- startup


def main() -> None:
    global _user_id
    # force=True: replace any handler the SDK installed; all logs go to stderr.
    logging.basicConfig(
        level=config.LOG_LEVEL,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    if not config.USER_EMAIL:
        log.error("NEURAX_USER_EMAIL is not set; pass it in the server's env (see README)")
        sys.exit(1)
    try:
        _user_id = memory.ensure_user(config.USER_EMAIL)
    except (memory.NeuraxError, psycopg.OperationalError) as e:
        log.error("cannot start: %s", e)
        sys.exit(1)

    log.info("loading embedding model %s (cache: %s)", config.EMBED_MODEL, config.MODEL_CACHE)
    try:
        # Model download/loading may print; nothing may reach stdout (the protocol channel).
        with redirect_stdout(sys.stderr):
            embeddings.embed(["warmup"])
    except Exception:
        # Saves still work without vectors (keyword search), so start anyway.
        log.warning("embedding model failed to load; continuing without vectors", exc_info=True)

    log.info("neurax MCP server ready for %s", config.USER_EMAIL)
    mcp.run("stdio")


if __name__ == "__main__":
    main()
