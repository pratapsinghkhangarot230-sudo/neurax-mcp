"""Core memory logic: save, supersede, search, history, pin.

Every query on projects or memories filters by user_id: that is the tenancy guarantee.
Memories are never edited; a changed decision is a new memory plus neurax_supersede().
"""

import logging
import re
from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from psycopg import errors

from neurax import config, embeddings, ranking
from neurax.db import Conn, transaction

log = logging.getLogger(__name__)

TYPES = ("decision", "constraint", "fact", "task")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
TOPIC_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]{0,79}$")
DUPLICATE_SUPERSEDE_NOTE = (
    "an identical active memory already exists; returned it and nothing was superseded"
)


class NeuraxError(Exception):
    """Base for errors the MCP layer turns into readable tool errors."""


class ProjectNotFound(NeuraxError):
    pass


class MemoryNotFound(NeuraxError):
    pass


class SupersedeConflict(NeuraxError):
    pass


class InvalidInput(NeuraxError):
    pass


# --------------------------------------------------------------------------- helpers


def normalize_topic(topic: str) -> str:
    """Lowercase, whitespace -> '-', then check the same pattern the DB enforces."""
    normalized = re.sub(r"\s+", "-", topic.strip().lower())
    if not TOPIC_RE.fullmatch(normalized):
        raise InvalidInput(
            f"invalid topic {topic!r}: use lowercase letters, digits and . _ / - "
            "(max 80 chars), e.g. 'database' or 'auth/session-store'"
        )
    return normalized


def _uuid(value: UUID | str) -> UUID:
    try:
        return value if isinstance(value, UUID) else UUID(value)
    except ValueError:
        raise InvalidInput(f"invalid memory id {value!r}; expected a UUID") from None


def _json(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float):
        return round(value, 4)
    return value


def _out(row: dict[str, Any]) -> dict[str, Any]:
    return {k: _json(v) for k, v in row.items()}


def _embed_one(text: str) -> tuple[str | None, str | None]:
    """(vector, model) in pgvector text form, or (None, None) if embedding fails."""
    try:
        return embeddings.to_pgvector(embeddings.embed([text])[0]), config.EMBED_MODEL
    except Exception:
        # Never lose a memory over an embedding problem: keyword search still works.
        log.warning("embedding failed; saving without a vector", exc_info=True)
        return None, None


def _project_id(conn: Conn, user_id: UUID, slug: str) -> UUID:
    # Resolve the agent-facing slug to the user's own, non-archived project.
    row = conn.execute(
        "SELECT id FROM projects WHERE user_id = %s AND slug = %s AND archived_at IS NULL",
        (user_id, slug),
    ).fetchone()
    if row is None:
        raise ProjectNotFound(f"project {slug!r} not found; call list_projects")
    return row["id"]


def _get(conn: Conn, user_id: UUID, memory_id: UUID) -> dict[str, Any]:
    # One memory in the caller-facing shape (never the embedding).
    row = conn.execute(
        """SELECT id, type, topic, content, rationale, status, pinned, source_tool,
                  created_at, supersedes_id
             FROM memories WHERE id = %s AND user_id = %s""",
        (memory_id, user_id),
    ).fetchone()
    if row is None:
        raise MemoryNotFound(f"memory {memory_id} not found")
    return _out(row)


def _supersede(conn: Conn, user_id: UUID, old_id: UUID, new_id: UUID) -> None:
    try:
        # The only supported way to supersede: locks both rows, checks project/status, moves pin.
        conn.execute("SELECT neurax_supersede(%s, %s, %s)", (user_id, old_id, new_id))
    except errors.NoDataFound:
        raise MemoryNotFound(f"memory {old_id} or {new_id} not found") from None
    except errors.CheckViolation as e:
        raise SupersedeConflict(e.diag.message_primary or str(e)) from None


def _conflicts(
    conn: Conn,
    user_id: UUID,
    project_id: UUID,
    new_id: UUID,
    topic: str,
    vector: str | None,
) -> list[dict[str, Any]]:
    # Other active memories the new one may contradict: same topic, or semantically close.
    # Same-topic first, then most similar. Without a vector only the topic match applies.
    rows = conn.execute(
        """SELECT id, type, topic, content, rationale, status, pinned, source_tool,
                  created_at, supersedes_id,
                  1 - (embedding <=> %(emb)s::vector) AS similarity
             FROM memories
            WHERE user_id = %(user_id)s AND project_id = %(project_id)s
              AND status = 'active' AND id <> %(new_id)s
              AND (topic = %(topic)s
                   OR 1 - (embedding <=> %(emb)s::vector) >= %(threshold)s)
            ORDER BY topic = %(topic)s DESC, similarity DESC NULLS LAST
            LIMIT 5""",
        {
            "user_id": user_id,
            "project_id": project_id,
            "new_id": new_id,
            "topic": topic,
            "emb": vector,
            "threshold": config.CONFLICT_THRESHOLD,
        },
    ).fetchall()
    return [_out(r) for r in rows]


# --------------------------------------------------------------------------- users & projects


def ensure_user(email: str) -> UUID:
    """Return the user's id, creating the user on first use. Email is case-insensitive."""
    try:
        with transaction() as conn:
            # Create if missing; users_email_lower_uq makes this idempotent.
            conn.execute(
                "INSERT INTO users (email) VALUES (%s) ON CONFLICT ((lower(email))) DO NOTHING",
                (email,),
            )
            # Users table has no user_id; look up by the same case-insensitive key.
            row = conn.execute(
                "SELECT id FROM users WHERE lower(email) = lower(%s)", (email,)
            ).fetchone()
    except errors.CheckViolation:
        raise InvalidInput(f"invalid email {email!r}") from None
    assert row is not None  # just inserted or already existed
    return row["id"]


def create_project(
    user_id: UUID, slug: str, name: str, description: str | None = None
) -> dict[str, Any]:
    if not SLUG_RE.fullmatch(slug):
        raise InvalidInput(
            f"invalid project slug {slug!r}: lowercase letters, digits and '-', max 63 chars"
        )
    try:
        with transaction() as conn:
            # New project for this user; projects_user_slug_uq rejects duplicates.
            row = conn.execute(
                """INSERT INTO projects (user_id, slug, name, description)
                   VALUES (%s, %s, %s, %s)
                   RETURNING slug, name, description, created_at""",
                (user_id, slug, name, description),
            ).fetchone()
    except errors.UniqueViolation:
        raise InvalidInput(f"project {slug!r} already exists") from None
    assert row is not None  # INSERT ... RETURNING always returns the row
    return _out(row)


def list_projects(user_id: UUID) -> list[dict[str, Any]]:
    with transaction() as conn:
        # The user's live projects with how many current (active) memories each has.
        rows = conn.execute(
            """SELECT p.slug, p.name, p.description,
                      count(m.id) FILTER (WHERE m.status = 'active') AS active_memories
                 FROM projects p
                 LEFT JOIN memories m ON m.project_id = p.id AND m.user_id = %(user_id)s
                WHERE p.user_id = %(user_id)s AND p.archived_at IS NULL
                GROUP BY p.id
                ORDER BY p.slug""",
            {"user_id": user_id},
        ).fetchall()
    return [_out(r) for r in rows]


# --------------------------------------------------------------------------- memories


def save(
    user_id: UUID,
    project: str,
    content: str,
    type: str,
    topic: str,
    rationale: str | None = None,
    supersedes_id: UUID | str | None = None,
    source_tool: str = "unknown",
) -> dict[str, Any]:
    """Save one memory, optionally superseding an older one in the same transaction.

    Returns {"memory", "duplicate", "possible_conflicts"} (+ "note" when a duplicate
    save asked to supersede, which is then skipped).
    """
    if type not in TYPES:
        raise InvalidInput(f"invalid type {type!r}; use one of {', '.join(TYPES)}")
    topic = normalize_topic(topic)
    content = content.strip()
    if not content:
        raise InvalidInput("content is empty")
    old_id = _uuid(supersedes_id) if supersedes_id is not None else None

    # Embed outside the transaction: it is the slow part. Topic gives short statements
    # context and is stable across versions, which helps the conflict check.
    vector, model = _embed_one(f"{topic}: {content}")

    try:
        with transaction() as conn:
            project_id = _project_id(conn, user_id, project)
            # Insert unless an identical active memory exists (memories_active_content_uq).
            # ON CONFLICT keeps this correct when two agents save the same thing at once.
            row = conn.execute(
                """INSERT INTO memories (user_id, project_id, type, topic, content, rationale,
                                         source_tool, embedding, embedding_model)
                   VALUES (%(user_id)s, %(project_id)s, %(type)s, %(topic)s, %(content)s,
                           %(rationale)s, %(source_tool)s, %(emb)s::vector, %(model)s)
                   ON CONFLICT (project_id, content_hash) WHERE status = 'active' DO NOTHING
                   RETURNING id""",
                {
                    "user_id": user_id,
                    "project_id": project_id,
                    "type": type,
                    "topic": topic,
                    "content": content,
                    "rationale": rationale,
                    "source_tool": source_tool,
                    "emb": vector,
                    "model": model,
                },
            ).fetchone()

            if row is None:
                # The existing active memory with the same normalized content.
                dup = conn.execute(
                    """SELECT id FROM memories
                        WHERE user_id = %s AND project_id = %s AND status = 'active'
                          AND content_hash = md5(lower(btrim(%s)))""",
                    (user_id, project_id, content),
                ).fetchone()
                assert dup is not None  # the row the INSERT conflicted with
                result: dict[str, Any] = {
                    "memory": _get(conn, user_id, dup["id"]),
                    "duplicate": True,
                    "possible_conflicts": [],
                }
                if old_id is not None:
                    result["note"] = DUPLICATE_SUPERSEDE_NOTE
                return result

            new_id: UUID = row["id"]
            if old_id is not None:
                _supersede(conn, user_id, old_id, new_id)
            return {
                "memory": _get(conn, user_id, new_id),
                "duplicate": False,
                "possible_conflicts": _conflicts(conn, user_id, project_id, new_id, topic, vector),
            }
    except errors.CheckViolation as e:
        # Length limits etc. enforced by the schema.
        raise InvalidInput(e.diag.message_primary or str(e)) from None


def supersede(user_id: UUID, old_id: UUID | str, new_id: UUID | str) -> dict[str, Any]:
    """Mark old_id as replaced by new_id (both must be active, same project)."""
    old, new = _uuid(old_id), _uuid(new_id)
    with transaction() as conn:
        _supersede(conn, user_id, old, new)
        return {"superseded": _get(conn, user_id, old), "current": _get(conn, user_id, new)}


def search(
    user_id: UUID,
    project: str,
    query: str,
    limit: int = 8,
    include_history: bool = False,
    types: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Hybrid search (ranking.py). Active only unless include_history; then current
    memories rank above superseded ones via a small penalty."""
    if not 1 <= limit <= 50:
        raise InvalidInput("limit must be between 1 and 50")
    if types is not None and (bad := set(types) - set(TYPES)):
        raise InvalidInput(f"invalid types {sorted(bad)}; use any of {', '.join(TYPES)}")
    vector, _ = _embed_one(query)

    with transaction() as conn:
        project_id = _project_id(conn, user_id, project)
        rows = ranking.search(
            conn, user_id, project_id, vector, query, limit, include_history, types
        )
    return [_out(r) for r in rows]


def history(user_id: UUID, project: str, topic: str) -> list[dict[str, Any]]:
    """Every memory in the topic, oldest first, with status and supersedes_id links."""
    topic = normalize_topic(topic)
    with transaction() as conn:
        project_id = _project_id(conn, user_id, project)
        # The whole topic including superseded rows, so the chain reads top to bottom.
        rows = conn.execute(
            """SELECT id, type, topic, content, rationale, status, pinned, source_tool,
                      created_at, supersedes_id
                 FROM memories
                WHERE user_id = %s AND project_id = %s AND topic = %s
                ORDER BY created_at, id""",
            (user_id, project_id, topic),
        ).fetchall()
    return [_out(r) for r in rows]


def get_project_context(user_id: UUID, project: str, limit: int = 30) -> dict[str, Any]:
    """The project's current state: pinned first, then constraints and decisions, then tasks.

    Never includes superseded memories. Unpinned facts are left out (use search for them).
    """
    if not 1 <= limit <= 100:
        raise InvalidInput("limit must be between 1 and 100")
    with transaction() as conn:
        # Project header; also the not-found check.
        proj = conn.execute(
            """SELECT id, slug, name, description FROM projects
                WHERE user_id = %s AND slug = %s AND archived_at IS NULL""",
            (user_id, project),
        ).fetchone()
        if proj is None:
            raise ProjectNotFound(f"project {project!r} not found; call list_projects")
        # Current memories in session-start priority order, each group newest first.
        rows = conn.execute(
            """SELECT id, type, topic, content, rationale, status, pinned, source_tool,
                      created_at, supersedes_id
                 FROM memories
                WHERE user_id = %s AND project_id = %s AND status = 'active'
                  AND (pinned OR type IN ('constraint', 'decision', 'task'))
                ORDER BY pinned DESC, type = 'task', created_at DESC, id
                LIMIT %s""",
            (user_id, proj["id"], limit),
        ).fetchall()
    return {
        "project": {k: proj[k] for k in ("slug", "name", "description")},
        "memories": [_out(r) for r in rows],
    }


def pin(user_id: UUID, memory_id: UUID | str) -> dict[str, Any]:
    return _set_pinned(user_id, memory_id, True)


def unpin(user_id: UUID, memory_id: UUID | str) -> dict[str, Any]:
    return _set_pinned(user_id, memory_id, False)


def _set_pinned(user_id: UUID, memory_id: UUID | str, pinned: bool) -> dict[str, Any]:
    mid = _uuid(memory_id)
    with transaction() as conn:
        # Only current memories can change pin; superseded rows are read-only.
        row = conn.execute(
            """UPDATE memories SET pinned = %s
                WHERE id = %s AND user_id = %s AND status = 'active'
            RETURNING id""",
            (pinned, mid, user_id),
        ).fetchone()
        if row is None:
            # Tell apart "doesn't exist (for this user)" from "superseded".
            exists = conn.execute(
                "SELECT 1 FROM memories WHERE id = %s AND user_id = %s", (mid, user_id)
            ).fetchone()
            if exists is None:
                raise MemoryNotFound(f"memory {mid} not found")
            raise InvalidInput(f"memory {mid} is superseded; pin its successor instead")
        return _get(conn, user_id, mid)
