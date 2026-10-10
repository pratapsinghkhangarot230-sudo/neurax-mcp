"""End to end: real MCP clients talk to `python -m neurax.server` over stdio (real Postgres)."""

import asyncio
import json
import os
import subprocess
import sys
import uuid
from typing import Any

from mcp import Client
from mcp.client.stdio import StdioServerParameters

from neurax import config

SERVER_CMD = [sys.executable, "-m", "neurax.server"]
TOOLS = {
    "list_projects",
    "create_project",
    "get_project_context",
    "save_context",
    "supersede",
    "search_context",
    "decision_history",
    "pin",
    "unpin",
}


def _env(**extra: str) -> dict[str, str]:
    """Server env: the test database, a fresh user, the shared model cache."""
    return {
        "DATABASE_URL": config.DATABASE_URL,
        "NEURAX_USER_EMAIL": f"mcp-{uuid.uuid4().hex}@example.com",
        "NEURAX_MODEL_CACHE": config.MODEL_CACHE,
        **extra,
    }


async def _session_flow() -> None:
    params = StdioServerParameters(command=SERVER_CMD[0], args=SERVER_CMD[1:], env=_env())
    async with Client(params) as client:

        async def call(tool: str, /, **args: Any) -> Any:
            result = await client.call_tool(tool, args)
            assert not result.is_error, result.content
            return json.loads(result.content[0].text)  # type: ignore[union-attr]

        async def call_error(tool: str, /, **args: Any) -> str:
            result = await client.call_tool(tool, args)
            assert result.is_error
            text: str = result.content[0].text  # type: ignore[union-attr]
            assert "Traceback" not in text
            return text

        assert {t.name for t in (await client.list_tools()).tools} == TOOLS

        await call("create_project", slug="neurax", name="Neurax")
        assert [p["slug"] for p in await call("list_projects")] == ["neurax"]

        redis = await call(
            "save_context",
            project="neurax",
            content="Use Redis as the main datastore",
            type="decision",
            topic="database",
            source_tool="pytest",
        )
        assert redis["duplicate"] is False and redis["possible_conflicts"] == []
        assert "hint" not in redis

        pg = await call(
            "save_context",
            project="neurax",
            content="Use Postgres + pgvector as the main datastore",
            type="decision",
            topic="database",
            rationale="one datastore for rows and vectors",
            supersedes_id=redis["memory"]["id"],
        )
        assert pg["memory"]["supersedes_id"] == redis["memory"]["id"]

        # A same-topic save surfaces the current decision as a possible conflict.
        task = await call(
            "save_context",
            project="neurax",
            content="Benchmark pgvector HNSW recall",
            type="task",
            topic="database",
        )
        assert [c["id"] for c in task["possible_conflicts"]] == [pg["memory"]["id"]]
        assert task["hint"].startswith("If the new memory replaces")

        ctx = await call("get_project_context", project="neurax")
        assert [m["content"] for m in ctx["memories"]] == [
            "Use Postgres + pgvector as the main datastore",
            "Benchmark pgvector HNSW recall",
        ]

        found = await call("search_context", project="neurax", query="main datastore")
        assert redis["memory"]["id"] not in [m["id"] for m in found]
        everything = await call(
            "search_context", project="neurax", query="main datastore", include_history=True
        )
        assert redis["memory"]["id"] in [m["id"] for m in everything]
        assert "embedding" not in everything[0]

        hist = await call("decision_history", project="neurax", topic="database")
        assert [m["status"] for m in hist] == ["superseded", "active", "active"]

        assert (await call("pin", memory_id=pg["memory"]["id"]))["pinned"] is True
        assert (await call("unpin", memory_id=pg["memory"]["id"]))["pinned"] is False

        # Errors are readable messages an agent can act on.
        assert "not found; call list_projects" in await call_error(
            "get_project_context", project="nope"
        )
        assert "already superseded" in await call_error(
            "supersede", old_id=redis["memory"]["id"], new_id=task["memory"]["id"]
        )
        assert "invalid memory id" in await call_error("pin", memory_id="abc")
        await call_error(
            "save_context", project="neurax", content="x", type="opinion", topic="database"
        )


def test_mcp_session_end_to_end() -> None:
    asyncio.run(_session_flow())


def _rpc(id: int | None, method: str, params: dict[str, Any] | None = None) -> bytes:
    msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if id is not None:
        msg["id"] = id
    return (json.dumps(msg) + "\n").encode()


def test_stdout_carries_only_json_rpc() -> None:
    """Every byte the server writes to stdout, startup included, must be JSON-RPC."""
    proc = subprocess.Popen(
        SERVER_CMD,
        env={**os.environ, **_env()},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    requests = _rpc(
        1,
        "initialize",
        {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "raw-test", "version": "0"},
        },
    )
    requests += _rpc(None, "notifications/initialized") + _rpc(2, "tools/list")
    stdout, stderr = proc.communicate(requests, timeout=120)

    lines = [line for line in stdout.decode().splitlines() if line.strip()]
    messages = [json.loads(line) for line in lines]  # fails on any non-JSON output
    assert all(m.get("jsonrpc") == "2.0" for m in messages)
    by_id = {m.get("id"): m for m in messages}
    assert by_id[1]["result"]["serverInfo"]["name"] == "neurax"
    assert {t["name"] for t in by_id[2]["result"]["tools"]} == TOOLS
    assert b"neurax MCP server ready" in stderr


def test_server_refuses_to_start_without_user_email() -> None:
    env = {k: v for k, v in os.environ.items() if k != "NEURAX_USER_EMAIL"}
    env["DATABASE_URL"] = config.DATABASE_URL
    proc = subprocess.run(
        SERVER_CMD, env=env, capture_output=True, timeout=60, stdin=subprocess.DEVNULL
    )
    assert proc.returncode == 1
    assert proc.stdout == b""
    assert b"NEURAX_USER_EMAIL is not set" in proc.stderr
