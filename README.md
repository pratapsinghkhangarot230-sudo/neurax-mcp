# Neurax

Decision memory for AI coding agents. Neurax records what was decided in a project, why,
and what has since changed, and serves the **current** state to any MCP client. A changed
decision is never edited: it is a new memory that supersedes the old one, so agents see
current decisions by default and can ask for history.

## Setup (Windows, PowerShell)

Needs [uv](https://docs.astral.sh/uv/) and Docker Desktop (running).

```powershell
git clone <repo-url> C:\path\to\neurax
cd C:\path\to\neurax
uv sync
docker compose up -d --wait
uv run python -m neurax.migrate
uv run python -m neurax.health
```

`neurax.health` should print `OK: connected` and `OK: vector extension ... installed`.

Download the embedding model once (about 70 MB, into `%LOCALAPPDATA%\neurax\models`), so
the server's first start doesn't spend its startup time downloading:

```powershell
uv run python -c "from neurax import embeddings; embeddings.embed(['warmup'])"
```

## Connect to Claude Code

Register the server once for your user, so every project can use it. Replace the email and
the path:

```powershell
claude mcp add --scope user --env NEURAX_USER_EMAIL=you@example.com --transport stdio neurax -- uv --directory C:\path\to\neurax run python -m neurax.server
```

- `NEURAX_USER_EMAIL` is your identity in Neurax; all your projects and memories belong to it.
- `uv --directory` runs the server from the Neurax folder, whatever project Claude Code is in.
- The defaults connect to the Docker Postgres on `127.0.0.1:5432`. To use another database,
  add `--env DATABASE_URL=postgresql://...` next to the other `--env`, before `--transport`.

Check it:

```powershell
claude mcp list
```

`neurax` should show as connected. Inside Claude Code, `/mcp` lists its 9 tools. To remove
it: `claude mcp remove neurax --scope user`.

The server must be able to reach Postgres, so keep Docker Desktop running
(`docker compose up -d` after a reboot).

## Use it in your projects

Paste this into your project's `CLAUDE.md` (or `.cursorrules`, `AGENTS.md`), and set the
project slug:

```markdown
## Project memory (Neurax)

This project's decisions live in Neurax under the project slug `my-project`.

- **Start of every session:** call `get_project_context("my-project")` and treat the result
  as the current state of the project. If the project doesn't exist, call `list_projects`,
  then `create_project` if needed.
- **During work:** when a past decision matters, call `search_context` before assuming.
- **End of every session:** call `save_context` once per decision, constraint, fact or task
  that came up. One statement per call, with a rationale for decisions, a short topic
  (e.g. `database`, `auth/session-store`) and `source_tool: "claude-code"`.
- **After each save:** review `possible_conflicts`. If the new memory replaces one of them,
  call `supersede(old_id, new_id)`. Never re-save an old decision to "update" it.
```

## Tools

| Tool | When agents call it |
|---|---|
| `get_project_context` | Start of a session: pinned items, current constraints and decisions, open tasks |
| `save_context` | End of a session, once per item; returns `possible_conflicts` |
| `supersede` | When a new memory replaces an old one |
| `search_context` | During work, to find a past decision (`include_history` for replaced ones) |
| `decision_history` | "How did we end up here?" for a topic |
| `pin` / `unpin` | Keep a few memories at the top of every session's context |
| `list_projects` / `create_project` | Find or create the project slug |

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Development

```powershell
uv run pytest          # integration tests against the neurax_test database
uv run ruff check; uv run ruff format --check; uv run mypy src tests
```
