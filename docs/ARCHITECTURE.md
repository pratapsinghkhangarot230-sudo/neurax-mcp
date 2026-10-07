# Neurax architecture (v1)

The detailed spec behind `CLAUDE.md`. If code and this document disagree, stop and ask Pratap which one is right, then update the other.

## 1. Concepts

| Term | Meaning |
|---|---|
| **User** | Owner of all data. In local mode there is one user, identified by `NEURAX_USER_EMAIL`. |
| **Project** | A codebase or product the agent works on, addressed by `slug` (e.g. `neurax`). |
| **Memory** | One atomic statement: a `decision`, `constraint`, `fact` or `task`. |
| **Topic** | Normalized grouping key (`database`, `auth/session-store`). Supersession happens within a topic. |
| **Active / superseded** | Active memories are current truth. Superseded ones are history, kept but hidden by default. |
| **Supersede** | Replacing an active memory with a newer one. Linear: each memory is superseded at most once. |

### Memory types

- `decision`: a choice made, ideally with rationale. *"Use Postgres + pgvector for storage."*
- `constraint`: a rule that must hold. *"No paid APIs in v1."*
- `fact`: a true statement about the project. *"Deployed on Fly.io, region bom."*
- `task`: an open piece of work. *"Add rate limiting to save_context."*

One memory = one statement. If an agent sends a paragraph with three decisions, the tool description asks it to split them.

## 2. Invariants (enforced by the database)

All of these are enforced in `db/migrations/0001_init.sql` and covered by `db/tests/test_invariants.sql`.

1. A memory's project belongs to the same user (composite foreign key).
2. Content, rationale, topic and type are immutable after insert (trigger).
3. Superseded memories are read-only, except for re-embedding (trigger).
4. A superseded memory can never become active again (trigger).
5. Each memory is superseded at most once, so history is a straight line (unique index).
6. Supersession only happens within one project and one user (composite self-FK plus `neurax_supersede`).
7. No two active memories in a project have identical content (unique index on `content_hash`).
8. A superseded memory is never pinned; the pin moves to its successor (check + function).
9. `embedding` and `embedding_model` are set together (check).

## 3. MCP tools

All tools act as the configured user. The user is never a tool argument. Outputs are JSON. Memory objects in outputs look like:

```json
{"id": "uuid", "type": "decision", "topic": "database",
 "content": "Use Postgres + pgvector", "rationale": "one datastore",
 "status": "active", "pinned": false, "source_tool": "claude-code",
 "created_at": "2026-10-07T19:20:00Z", "supersedes_id": null}
```

### `list_projects()`
Returns the user's non-archived projects: slug, name, description, active memory count.

### `create_project(slug, name, description?)`
Creates a project. Errors if the slug exists. The slug must match `^[a-z0-9][a-z0-9-]{0,62}$`.

### `get_project_context(project, limit=30)`
**Agents call this at the start of a session.** Returns pinned memories first, then active `constraint` and `decision` memories newest first, then open `task`s, up to `limit`. Superseded memories are never included. This is the "current state of the project" in one call.

### `save_context(project, content, type, topic, rationale?, supersedes_id?, source_tool?)`
Saves one memory. **Agents call this at the end of a session, once per decision, constraint, fact or task.**
- If `supersedes_id` is given, save and supersede happen in one transaction.
- If identical active content already exists in the project, return the existing memory with `"duplicate": true` instead of erroring.
- **Contradiction check:** the response includes `possible_conflicts`, up to 5 *other* active memories in the same project that share the topic **or** have cosine similarity ≥ `NEURAX_CONFLICT_THRESHOLD` (default 0.80). The tool result tells the agent: *"If the new memory replaces any of these, call supersede(old_id, new_id)."* Neurax does not decide; the calling model does.

Response: `{"memory": {...}, "duplicate": false, "possible_conflicts": [{...}]}`

### `supersede(old_id, new_id)`
Calls `neurax_supersede`. Errors are passed through as readable messages ("already superseded", "different projects", "memory not found").

### `search_context(project, query, limit=8, include_history=false, types?)`
Hybrid search (section 4). Active only unless `include_history=true`. Optional `types` filter, e.g. `["decision", "constraint"]`.

### `decision_history(project, topic)`
Every memory in the topic, oldest → newest, with status and which memory replaced which. Built by following `supersedes_id` links. Answers "how did we end up here?"

### `pin(memory_id)` / `unpin(memory_id)`
Pinned memories always appear first in `get_project_context` and get a ranking boost. Only active memories can be pinned.

## 4. Ranking (Phase 3)

Score for each candidate memory:

```
score = W_VEC    * vec_sim                       -- 1 - cosine distance, 0..1
      + W_TEXT   * text_rank                     -- ts_rank_cd(search_tsv, query, 32), 0..1
      + W_RECENT * 0.5 ^ (age_days / HALF_LIFE)  -- 1.0 today, 0.5 after HALF_LIFE days
      + W_PIN    * pinned
```

Defaults (env-configurable): `W_VEC=0.55`, `W_TEXT=0.25`, `W_RECENT=0.10`, `W_PIN=0.10`, `HALF_LIFE=30` days.

Candidate set: union of the top 50 by vector distance and the top 50 by full-text match, both filtered by `user_id`, `project_id` and (by default) `status = 'active'`. Memories without an embedding get `vec_sim = 0` but can still match on text.

Reference SQL shape:

```sql
-- Hybrid search: gather candidates from vector and keyword search, then score them together.
WITH q AS (
    SELECT %(embedding)s::vector AS emb, websearch_to_tsquery('english', %(query)s) AS tsq
),
vec AS (   -- nearest neighbours by cosine distance
    SELECT m.id FROM memories m, q
     WHERE m.user_id = %(user_id)s AND m.project_id = %(project_id)s
       AND m.status = 'active' AND m.embedding IS NOT NULL
     ORDER BY m.embedding <=> q.emb
     LIMIT 50
),
txt AS (   -- keyword matches
    SELECT m.id FROM memories m, q
     WHERE m.user_id = %(user_id)s AND m.project_id = %(project_id)s
       AND m.status = 'active' AND m.search_tsv @@ q.tsq
     ORDER BY ts_rank_cd(m.search_tsv, q.tsq, 32) DESC
     LIMIT 50
)
SELECT m.*,
       %(w_vec)s    * coalesce(1 - (m.embedding <=> q.emb), 0)
     + %(w_text)s   * ts_rank_cd(m.search_tsv, q.tsq, 32)
     + %(w_recent)s * power(0.5, extract(epoch FROM now() - m.created_at) / 86400 / %(half_life)s)
     + %(w_pin)s    * m.pinned::int AS score
  FROM memories m, q
 WHERE m.id IN (SELECT id FROM vec UNION SELECT id FROM txt)
 ORDER BY score DESC
 LIMIT %(limit)s;
```

With `include_history=true`, drop the `status` filter and add a small penalty for superseded rows so current decisions still rank first.

## 5. Session flow (what the README will tell users)

1. **Start of session:** the agent calls `get_project_context(project)`.
2. **During work:** the agent calls `search_context` when it needs a past decision.
3. **End of session:** the agent summarizes what was decided and calls `save_context` once per item. It reviews `possible_conflicts` and calls `supersede` where a new decision replaces an old one.

Users add a short instruction block to their own project's `CLAUDE.md` / `.cursorrules` telling the agent to follow this flow. Shipping that snippet is part of Phase 2.

## 6. Configuration

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql://neurax:neurax@localhost:5432/neurax` | Postgres connection |
| `NEURAX_USER_EMAIL` | (required) | Local-mode user identity, created on first run |
| `NEURAX_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Must output 384 dims to match the schema |
| `NEURAX_CONFLICT_THRESHOLD` | `0.80` | Cosine similarity for `possible_conflicts` |
| `NEURAX_W_VEC` / `_W_TEXT` / `_W_RECENT` / `_W_PIN` | `0.55 / 0.25 / 0.10 / 0.10` | Ranking weights |
| `NEURAX_HALF_LIFE_DAYS` | `30` | Recency decay |
| `NEURAX_LOG_LEVEL` | `INFO` | Logs go to stderr |
