"""Phase 3: weighted hybrid ranking, decision drift, and conflict-threshold calibration."""

from typing import Any
from uuid import UUID

import pytest

from neurax import config, memory
from neurax.db import transaction


def save(user_id: UUID, project: str, content: str, topic: str, **kw: Any) -> dict[str, Any]:
    return memory.save(user_id, project, content, kw.pop("type", "decision"), topic, **kw)["memory"]


def ids(results: list[dict[str, Any]]) -> list[str]:
    return [m["id"] for m in results]


# --------------------------------------------------------------------------- decision drift


def test_superseded_decision_is_hidden_and_ranks_below_its_successor(
    user_id: UUID, project: str
) -> None:
    save(user_id, project, "Deploy on Fly.io, region bom", "hosting", type="fact")
    a = save(user_id, project, "Use Redis as the session store", "auth/session-store")
    b = save(
        user_id,
        project,
        "Use Postgres as the session store",
        "auth/session-store",
        supersedes_id=a["id"],
    )

    current = memory.search(user_id, project, "auth session store")
    assert current[0]["id"] == b["id"]
    assert a["id"] not in ids(current)

    everything = memory.search(user_id, project, "auth session store", include_history=True)
    assert a["id"] in ids(everything)
    assert ids(everything).index(b["id"]) < ids(everything).index(a["id"])


# --------------------------------------------------------------------------- pin and weights


def test_pin_boost_lifts_an_equal_match(user_id: UUID, project: str) -> None:
    save(user_id, project, "Use Postgres for the job queue", "queue")
    save(user_id, project, "Use Postgres for the session store", "sessions")
    second = memory.search(user_id, project, "Postgres")[1]
    memory.pin(user_id, second["id"])
    assert memory.search(user_id, project, "Postgres")[0]["id"] == second["id"]


def test_weights_come_from_config(
    user_id: UUID, project: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    strong = save(user_id, project, "Use Redis as the session store", "sessions")
    weak = save(user_id, project, "No paid APIs in v1", "budget", type="constraint")
    memory.pin(user_id, weak["id"])
    assert memory.search(user_id, project, "Redis session store")[0]["id"] == strong["id"]

    monkeypatch.setattr(config, "W_PIN", 1.0)
    assert memory.search(user_id, project, "Redis session store")[0]["id"] == weak["id"]


# --------------------------------------------------------------------------- conflict threshold

# Real cross-topic conflicts: same question, different topic names. Measured with
# bge-small-en-v1.5 (topic-prefixed text): 0.706-0.760. The default threshold is 0.74, the
# best F2 in the Phase 4 benchmark (bench/RESULTS.md): it trades ~3% cross-topic recall for
# twice the precision, so the two pairs below 0.74 are kept as expected misses.
EXPECTED_MISS = pytest.mark.xfail(
    strict=True, reason="similarity below 0.74; accepted miss, see bench/RESULTS.md"
)
CROSS_TOPIC_CONFLICTS = [
    pytest.param(
        ("database", "Storage is Postgres 16 with pgvector"),
        ("storage", "Use MongoDB for storage"),
        marks=EXPECTED_MISS,
    ),
    (("performance", "Use Redis as the cache layer"), ("cache", "Use Memcached for caching")),
    (
        ("sessions", "Keep user sessions in Redis"),
        ("auth/session-store", "Store sessions in Postgres"),
    ),
    pytest.param(
        ("ui", "The UI is built with React"),
        ("frontend", "Use Vue for the web UI"),
        marks=EXPECTED_MISS,
    ),
]


@pytest.mark.parametrize(("old", "new"), CROSS_TOPIC_CONFLICTS)
def test_cross_topic_conflict_is_flagged(
    user_id: UUID, project: str, old: tuple[str, str], new: tuple[str, str]
) -> None:
    existing = save(user_id, project, old[1], old[0])
    result = memory.save(user_id, project, new[1], "decision", new[0])
    with transaction() as conn:
        # Cosine similarity of the two stored embeddings, for calibration output.
        row = conn.execute(
            """SELECT 1 - (a.embedding <=> b.embedding) AS sim
                 FROM memories a, memories b
                WHERE a.id = %s AND b.id = %s AND a.user_id = %s AND b.user_id = %s""",
            (existing["id"], result["memory"]["id"], user_id, user_id),
        ).fetchone()
    assert row is not None
    print(f"\n{row['sim']:.3f}  {old}  vs  {new}")
    assert existing["id"] in ids(result["possible_conflicts"])
