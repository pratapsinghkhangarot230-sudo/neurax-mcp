"""Neurax demo: the Redis -> Postgres -> MySQL story through a real MCP client.

    uv run python examples/demo.py

Saves the same three decisions into two projects over MCP: one never supersedes (flat
memory), one supersedes like an agent following Neurax's possible_conflicts. Then asks both
the same question. Uses its own database (neurax_demo, created and migrated if missing) and
a fresh user per run, so your real memories are never touched.
"""

import os

DEMO_DATABASE_URL = os.environ.get(
    "NEURAX_DEMO_DATABASE_URL", "postgresql://neurax:neurax@127.0.0.1:5432/neurax_demo"
)
# Point neurax at the demo database before any neurax module reads config.
os.environ["DATABASE_URL"] = DEMO_DATABASE_URL

import asyncio  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import uuid  # noqa: E402
from typing import Any  # noqa: E402

from mcp import Client  # noqa: E402
from mcp.client.stdio import StdioServerParameters  # noqa: E402

from neurax import config, migrate  # noqa: E402

TOPIC = "database"
STORY = [  # (content, rationale), one per "session"
    ("Use Redis as the main database", "fast key-value store for the prototype"),
    ("Switch the main database from Redis to Postgres", "we need joins and transactions"),
    ("Switch the main database from Postgres to MySQL", "the host only offers managed MySQL"),
]
QUESTION = "which database do we use?"


async def run() -> None:
    env = {
        "DATABASE_URL": DEMO_DATABASE_URL,
        "NEURAX_USER_EMAIL": f"demo-{uuid.uuid4().hex[:8]}@neurax.local",
        "NEURAX_MODEL_CACHE": config.MODEL_CACHE,
        "NEURAX_LOG_LEVEL": "WARNING",  # keep server logs out of the recording
    }
    params = StdioServerParameters(command=sys.executable, args=["-m", "neurax.server"], env=env)
    async with Client(params) as client:

        async def call(tool: str, /, **args: Any) -> Any:
            result = await client.call_tool(tool, args)
            text: str = result.content[0].text  # type: ignore[union-attr]
            if result.is_error:
                sys.exit(f"{tool} failed: {text}")
            return json.loads(text)

        for slug, name in (("flat", "Flat memory"), ("neurax", "Neurax")):
            await call("create_project", slug=slug, name=name)

        print(f'Three sessions, three decisions about the main database (topic "{TOPIC}"):\n')
        for i, (content, rationale) in enumerate(STORY, 1):
            print(f"  {i}. {content}")
            common = {"content": content, "type": "decision", "topic": TOPIC}
            await call("save_context", project="flat", rationale=rationale, **common)
            saved = await call("save_context", project="neurax", rationale=rationale, **common)
            # What the agent does with possible_conflicts: replace the old decision.
            for old in saved["possible_conflicts"]:
                if old["type"] == "decision" and old["topic"] == TOPIC:
                    await call("supersede", old_id=old["id"], new_id=saved["memory"]["id"])
                    print(f'       Neurax flags "{old["content"]}" -> supersede')

        print(f'\nA new session asks: "{QUESTION}"\n')
        for slug, title in (
            ("flat", "BEFORE  flat memory (same store, nothing superseded)"),
            ("neurax", "AFTER   Neurax"),
        ):
            found = await call("search_context", project=slug, query=QUESTION)
            print(title)
            for m in found:
                print(f"  [{m['status']}] {m['content']}")
            verdict = (
                f"{len(found)} answers, all current: the agent has to guess."
                if len(found) > 1
                else "one current answer."
            )
            print(f"  -> {verdict}\n")

        print(f'How it got there (decision_history "{TOPIC}"):')
        for m in await call("decision_history", project="neurax", topic=TOPIC):
            print(f"  {m['status']:<10}  {m['content']}  ({m['rationale']})")


def main() -> None:
    migrate.create_database()
    with contextlib.redirect_stdout(io.StringIO()):  # "applied/skipped" lines, not demo output
        migrate.migrate()
    print("\nNeurax demo: Redis -> Postgres -> MySQL\n")
    asyncio.run(run())


if __name__ == "__main__":
    main()
