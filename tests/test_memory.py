"""Integration tests for memory.py against a real Postgres (neurax_test)."""

import uuid
from typing import Any
from uuid import UUID

import pytest

from neurax import embeddings, memory
from neurax.db import transaction
from neurax.memory import InvalidInput, MemoryNotFound, ProjectNotFound, SupersedeConflict


def save(
    user_id: UUID,
    project: str,
    content: str,
    topic: str = "database",
    type: str = "decision",
    **kw: Any,
) -> dict[str, Any]:
    return memory.save(user_id, project, content, type, topic, **kw)


# --------------------------------------------------------------------------- users & projects


def test_ensure_user_is_idempotent_and_case_insensitive() -> None:
    email = f"Case-{uuid.uuid4().hex}@Example.com"
    assert memory.ensure_user(email) == memory.ensure_user(email.lower())


def test_create_and_list_projects(user_id: UUID) -> None:
    memory.create_project(user_id, "alpha", "Alpha", "first")
    memory.create_project(user_id, "beta", "Beta")
    save(user_id, "alpha", "Use Postgres")
    assert memory.list_projects(user_id) == [
        {"slug": "alpha", "name": "Alpha", "description": "first", "active_memories": 1},
        {"slug": "beta", "name": "Beta", "description": None, "active_memories": 0},
    ]


def test_create_project_rejects_bad_and_duplicate_slugs(user_id: UUID, project: str) -> None:
    with pytest.raises(InvalidInput, match="invalid project slug"):
        memory.create_project(user_id, "Bad Slug", "x")
    with pytest.raises(InvalidInput, match="already exists"):
        memory.create_project(user_id, project, "again")


def test_unknown_project(user_id: UUID) -> None:
    with pytest.raises(ProjectNotFound, match="call list_projects"):
        save(user_id, "nope", "Use Postgres")


# --------------------------------------------------------------------------- save


def test_save_normalizes_topic_and_stores_embedding(user_id: UUID, project: str) -> None:
    result = save(user_id, project, "  Use Postgres  ", topic=" Auth Session Store ")
    m = result["memory"]
    assert result["duplicate"] is False
    assert m["topic"] == "auth-session-store"
    assert m["content"] == "Use Postgres"
    assert "embedding" not in m
    with transaction() as conn:
        row = conn.execute(
            "SELECT vector_dims(embedding) AS dims, embedding_model FROM memories "
            "WHERE id = %s AND user_id = %s",
            (m["id"], user_id),
        ).fetchone()
    assert row == {"dims": 384, "embedding_model": "BAAI/bge-small-en-v1.5"}


def test_save_rejects_invalid_input(user_id: UUID, project: str) -> None:
    with pytest.raises(InvalidInput, match="invalid type"):
        save(user_id, project, "x", type="opinion")
    with pytest.raises(InvalidInput, match="invalid topic"):
        save(user_id, project, "x", topic="!!")
    with pytest.raises(InvalidInput, match="invalid memory id"):
        save(user_id, project, "x", supersedes_id="not-a-uuid")


def test_save_duplicate_returns_existing(user_id: UUID, project: str) -> None:
    first = save(user_id, project, "Use Postgres")["memory"]
    again = save(user_id, project, "use postgres ")
    assert again["duplicate"] is True
    assert again["memory"]["id"] == first["id"]
    assert "note" not in again


def test_save_duplicate_with_supersedes_id_does_not_supersede(user_id: UUID, project: str) -> None:
    pg = save(user_id, project, "Use Postgres")["memory"]
    mysql = save(user_id, project, "Use MySQL")["memory"]
    result = save(user_id, project, "Use Postgres", supersedes_id=mysql["id"])
    assert result["duplicate"] is True
    assert result["memory"]["id"] == pg["id"]
    assert "nothing was superseded" in result["note"]
    assert [m["status"] for m in memory.history(user_id, project, "database")] == [
        "active",
        "active",
    ]


def test_save_without_embedding_still_saves_and_keyword_search_works(
    user_id: UUID, project: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(embeddings, "embed", broken)
    m = save(user_id, project, "Use pgvector for similarity search")["memory"]
    assert [r["id"] for r in memory.search(user_id, project, "pgvector")] == [m["id"]]


def test_failed_supersede_rolls_back_the_new_memory(user_id: UUID, project: str) -> None:
    with pytest.raises(MemoryNotFound):
        save(user_id, project, "Use MySQL", supersedes_id=uuid.uuid4())
    assert memory.history(user_id, project, "database") == []


def test_possible_conflicts_include_same_topic(user_id: UUID, project: str) -> None:
    redis = save(user_id, project, "Use Redis for caching", topic="cache")["memory"]
    save(user_id, project, "Deploy on Fly.io", topic="hosting", type="fact")
    result = save(user_id, project, "Use Memcached for caching", topic="cache")
    conflicts = result["possible_conflicts"]
    assert conflicts[0]["id"] == redis["id"]
    assert result["memory"]["id"] not in [c["id"] for c in conflicts]


# --------------------------------------------------------------------------- supersede chain


def test_redis_postgres_mysql_supersede_chain(user_id: UUID, project: str) -> None:
    redis = save(user_id, project, "Use Redis as the main datastore")["memory"]
    memory.pin(user_id, redis["id"])

    pg = save(
        user_id,
        project,
        "Use Postgres + pgvector as the main datastore",
        rationale="one datastore for rows and vectors",
        supersedes_id=redis["id"],
    )["memory"]
    assert pg["supersedes_id"] == redis["id"]
    assert pg["pinned"] is True  # pin moved to the successor

    result = memory.supersede(
        user_id, pg["id"], save(user_id, project, "Use MySQL as the main datastore")["memory"]["id"]
    )
    mysql = result["current"]
    assert result["superseded"]["status"] == "superseded"

    chain = memory.history(user_id, project, "database")
    assert [m["content"] for m in chain] == [
        "Use Redis as the main datastore",
        "Use Postgres + pgvector as the main datastore",
        "Use MySQL as the main datastore",
    ]
    assert [m["status"] for m in chain] == ["superseded", "superseded", "active"]
    assert [m["supersedes_id"] for m in chain] == [None, redis["id"], pg["id"]]
    assert [m["pinned"] for m in chain] == [False, False, True]

    # History is linear: a superseded memory can't be superseded again.
    with pytest.raises(SupersedeConflict, match="already superseded"):
        memory.supersede(user_id, redis["id"], mysql["id"])

    # Decision drift: current decision only, unless history is asked for.
    current = memory.search(user_id, project, "main datastore")
    assert [m["id"] for m in current] == [mysql["id"]]
    everything = memory.search(user_id, project, "main datastore", include_history=True)
    assert everything[0]["id"] == mysql["id"]
    assert {m["id"] for m in everything} == {redis["id"], pg["id"], mysql["id"]}


def test_supersede_errors(user_id: UUID, project: str) -> None:
    a = save(user_id, project, "Use Postgres")["memory"]
    with pytest.raises(SupersedeConflict, match="cannot supersede itself"):
        memory.supersede(user_id, a["id"], a["id"])
    with pytest.raises(MemoryNotFound):
        memory.supersede(user_id, a["id"], uuid.uuid4())


# --------------------------------------------------------------------------- search


def test_search_keyword_and_types_filter(user_id: UUID, project: str) -> None:
    decision = save(user_id, project, "Use Postgres + pgvector")["memory"]
    constraint = save(user_id, project, "No paid APIs in v1", topic="budget", type="constraint")[
        "memory"
    ]
    assert memory.search(user_id, project, "pgvector")[0]["id"] == decision["id"]
    only = memory.search(user_id, project, "pgvector", types=["constraint"])
    assert [m["id"] for m in only] == [constraint["id"]]
    with pytest.raises(InvalidInput, match="invalid types"):
        memory.search(user_id, project, "x", types=["opinion"])


# --------------------------------------------------------------------------- pin


def test_pin_unpin_and_superseded_cannot_be_pinned(user_id: UUID, project: str) -> None:
    old = save(user_id, project, "Use Redis")["memory"]
    assert memory.pin(user_id, old["id"])["pinned"] is True
    assert memory.unpin(user_id, old["id"])["pinned"] is False
    save(user_id, project, "Use Postgres", supersedes_id=old["id"])
    with pytest.raises(InvalidInput, match="pin its successor"):
        memory.pin(user_id, old["id"])
    with pytest.raises(MemoryNotFound):
        memory.pin(user_id, uuid.uuid4())


# --------------------------------------------------------------------------- tenancy


def test_other_user_cannot_see_or_touch_memories(user_id: UUID, project: str) -> None:
    mine = save(user_id, project, "Use Postgres")["memory"]
    other = memory.ensure_user(f"other-{uuid.uuid4().hex}@example.com")
    with pytest.raises(ProjectNotFound):
        memory.search(other, project, "Postgres")
    memory.create_project(other, project, "Their demo")
    theirs = save(other, project, "Use MySQL")["memory"]
    assert mine["id"] not in [m["id"] for m in memory.search(other, project, "Postgres")]
    with pytest.raises(MemoryNotFound):
        memory.supersede(other, mine["id"], theirs["id"])
    with pytest.raises(MemoryNotFound):
        memory.pin(other, mine["id"])
