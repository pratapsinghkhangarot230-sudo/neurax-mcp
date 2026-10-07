# Neurax: instructions for Claude Code

Neurax is **decision memory for AI coding agents**. It records what was decided in a project, why, and what has since changed, and serves the *current* state to any MCP client (Claude Code, Cursor, Claude Desktop, Codex).

**The wedge:** generic memory tools store flat facts, so outdated decisions keep being treated as true. Neurax never edits a decision; a changed decision is a new memory that **supersedes** the old one. Agents see current decisions by default and can ask for history.

Open source, local-first, multi-user from day one (a hosted team version comes later). The product owner is Pratap: he makes architecture and scope calls, and you implement.

Read `docs/ARCHITECTURE.md` before any non-trivial change. It holds the tool contracts, ranking formula and invariants in detail.

---

## 1. Locked v1 scope

Do not expand this without Pratap's explicit approval in chat.

| Area | v1 | Not in v1 → `PARKING_LOT.md` |
|---|---|---|
| Storage | Postgres 16 + pgvector | Graph DB, Redis, any second datastore |
| Interface | MCP server (stdio first, then streamable HTTP) | REST API, SDK, web UI, IDE/browser extensions |
| Capture | Client model summarizes at end of session and calls `save_context` | Live capture, transcript ingestion, LLM extraction inside Neurax |
| Versioning | `status` + `supersedes_id`, via `neurax_supersede()` | Branching histories, merge, undo |
| Contradictions | On save, return similar active memories in the same project; the calling model decides | Automatic contradiction resolution, LLM judges inside Neurax |
| Retrieval | One scored SQL query: full-text + vector + recency + pin | Learned ranking, rerankers, context composer, compression |
| Embeddings | Local `BAAI/bge-small-en-v1.5` (384 dims) | Paid embedding APIs |

If a task seems to need something in the right column: stop, add it to `PARKING_LOT.md` with one line on why, and tell Pratap.

## 2. Stack

- Python 3.11+, managed with `uv` (`pyproject.toml`, lockfile committed)
- Official MCP Python SDK (`mcp`)
- `psycopg` 3 with `psycopg_pool`. **Plain SQL, no ORM.**
- `fastembed` for embeddings (ONNX, no torch)
- `pytest` for tests, `ruff` for lint and format, `mypy` (non-strict to start)
- Docker Compose with the `pgvector/pgvector:pg16` image

Ask before adding any dependency not on this list.

## 3. Repository layout (target)

```
neurax/
├── CLAUDE.md
├── PARKING_LOT.md
├── docs/ARCHITECTURE.md
├── docker-compose.yml
├── .env.example
├── pyproject.toml
├── db/
│   ├── migrations/        # 0001_init.sql, 0002_..., applied in order, never edited after applying
│   └── tests/             # SQL invariant tests
├── src/neurax/
│   ├── config.py          # settings from env vars only
│   ├── db.py              # connection pool, transaction helper
│   ├── migrate.py         # applies db/migrations/*.sql, records them in schema_migrations
│   ├── embeddings.py      # embed(texts) -> list[list[float]]; model loaded once
│   ├── memory.py          # save, supersede, search, history, pin: the core logic
│   ├── ranking.py         # the hybrid search SQL and weights
│   └── server.py          # MCP server: thin layer over memory.py
└── tests/                 # pytest; integration tests hit a real Postgres
```

Keep `server.py` thin. Business logic lives in `memory.py`, so it can be tested without MCP.

## 4. Database rules

- **`db/migrations/0001_init.sql` is the source of truth.** Read it before writing any query.
- **Never edit a migration that has been applied.** Schema changes go in a new numbered file, and you ask Pratap before creating one.
- **Every query on `projects` or `memories` filters by `user_id`.** No exceptions, including internal helpers. This is the tenancy guarantee.
- **Never `UPDATE` content, rationale, topic or type.** The DB trigger rejects it anyway. A change means a new memory plus `neurax_supersede()`.
- **Superseding always goes through `SELECT neurax_supersede(user_id, old_id, new_id)`.** Never set `status` or `supersedes_id` by hand.
- Parameterized queries only (`%s` placeholders). **Never build SQL with f-strings or string concatenation.**
- Each query gets a short comment saying what it does and why.
- Multi-step writes (save + supersede) happen in **one transaction**.
- Topics are normalized in Python before insert: lowercase, spaces → `-`, matching `^[a-z0-9][a-z0-9._/-]{0,79}$`.
- Embeddings: store `embedding` and `embedding_model` together. If embedding fails, still save the memory with both NULL and log a warning. Keyword search must keep working.
- For filtered vector search, set `SET LOCAL hnsw.iterative_scan = relaxed_order` (pgvector ≥ 0.8) inside the transaction, so per-project filters don't cut results short.

## 5. MCP server rules

- **Never write to stdout in the server.** stdio transport uses stdout for protocol messages, and a stray `print` breaks the connection. Log to stderr with `logging`.
- Tool descriptions are product copy that agents read to decide when to call a tool. Write them carefully, say when to use the tool, and keep them in sync with `docs/ARCHITECTURE.md`.
- Validate all tool inputs, and return clear error messages an agent can act on (e.g. "project 'foo' not found; call list_projects"). Never return raw stack traces.
- Tool outputs are compact JSON: ids, content, rationale, topic, type, status, created_at. Never return embeddings.
- The user identity comes from config (`NEURAX_USER_EMAIL`) in local mode. Never from tool arguments.

## 6. Code standards

- Type hints on all functions. Small functions. Names over comments, but comment the *why* of anything non-obvious.
- Config only from environment variables via `config.py`. Never hardcode credentials. Never commit `.env`.
- Load the embedding model once per process, not per call.
- Errors: raise specific exceptions in `memory.py` (`ProjectNotFound`, `MemoryNotFound`, `SupersedeConflict`), and map them to tool errors in `server.py`.
- `ruff check`, `ruff format` and `pytest` must pass before a step counts as done.

## 7. Testing

- Integration tests use a real Postgres (the Docker one, separate database `neurax_test`). Don't mock the database.
- Run `db/tests/test_invariants.sql` after any migration change. It must print `ALL INVARIANT TESTS PASSED`.
- Every bug fix comes with a test that fails without the fix.
- Ranking changes must keep the decision-drift tests green: after A is superseded by B, a search on that topic returns B first, and A does not appear unless `include_history=true`.

## 8. How to work

- **One phase at a time** (section 10). Don't start the next phase until Pratap confirms the current one.
- **Plan first** for anything touching more than ~2 files: list the files and the approach in a few lines, then wait for a go-ahead.
- **Small diffs.** Never rewrite large parts of the codebase in one pass, even if asked to "clean up everything". Propose a sequence instead.
- **Show proof:** a step is done only after you've run it and shown the output (tests, the health check, a sample tool call).
- **Explain briefly** what you built and any decision you made, in plain language. Pratap reviews and needs to be able to explain every part of the code.
- If something is ambiguous, ask **one** focused question rather than guessing big.
- Flag scope creep, including your own. If you notice yourself building something not in the current phase, stop.
- Git: small commits, conventional messages (`feat:`, `fix:`, `chore:`, `test:`, `docs:`). Never force-push. Never commit secrets.

## 9. Never

- Never add a graph DB, ORM, web framework or UI in v1.
- Never edit applied migrations, or run destructive commands (`DROP`, `TRUNCATE`, `docker compose down -v`, `rm -rf`) without explicit approval.
- Never write a query on user data without a `user_id` filter.
- Never update a memory's content or hand-edit `status` / `supersedes_id`.
- Never print to stdout inside the MCP server.
- Never call a paid API (LLM or embeddings) from Neurax in v1.

## 10. Phases

Each phase ends with a short demo and Pratap's sign-off.

- [ ] **0. Foundation:** `docker-compose.yml` (pgvector/pgvector:pg16, volume, healthcheck), `.env.example`, `pyproject.toml`, `migrate.py` applying `db/migrations/` in order, a health check script confirming DB connection and the `vector` extension, and `test_invariants.sql` passing.
- [ ] **1. Core logic:** `memory.py` with `ensure_user`, `create_project`, `save`, `supersede`, `search` (keyword + vector, simple version), `history`, `pin`/`unpin`; `embeddings.py`; pytest coverage including the full Redis → Postgres → MySQL supersede chain.
- [ ] **2. MCP server:** the tools in `docs/ARCHITECTURE.md`, stdio transport, connected to Claude Code, with setup steps in the README. Pratap uses it daily on Neurax itself from here.
- [ ] **3. Ranking + contradiction check:** the hybrid scoring query in `ranking.py`, with weights in config; `save_context` returns similar active memories in the same topic or above the similarity threshold; decision-drift tests.
- [ ] **4. Benchmark:** `bench/`: a small public decision-drift dataset (projects whose decisions change over time) and a script comparing Neurax against flat memory (same store, no supersession). Report accuracy of "answers with the current decision".
- [ ] **5. Launch prep:** one-command install, README with a GIF demo, license (Apache-2.0), examples.
