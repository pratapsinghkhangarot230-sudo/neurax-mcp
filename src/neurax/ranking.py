"""Hybrid search: one scored SQL query over vector + keyword candidates (ARCHITECTURE §4).

score = W_VEC * vec_sim + W_TEXT * text_rank + W_RECENT * 0.5^(age_days / HALF_LIFE)
      + W_PIN * pinned - SUPERSEDED_PENALTY * superseded
"""

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from neurax import config
from neurax.db import Conn

# Superseded rows only appear with include_history; this keeps current decisions above
# their predecessors without hiding an old memory that matches much better.
# ponytail: constant, move to config if the Phase 4 benchmark shows it needs tuning.
SUPERSEDED_PENALTY = 0.15

# Candidates: top 50 by cosine distance UNION top 50 keyword matches, both filtered by
# user, project, status and type; then every candidate gets the weighted score.
SEARCH_SQL = """
WITH q AS (
    SELECT %(emb)s::vector AS emb, websearch_to_tsquery('english', %(query)s) AS tsq
),
vec AS (
    SELECT m.id FROM memories m, q
     WHERE m.user_id = %(user_id)s AND m.project_id = %(project_id)s
       AND (%(include_history)s OR m.status = 'active')
       AND (%(types)s::text[] IS NULL OR m.type = ANY(%(types)s::text[]))
       AND m.embedding IS NOT NULL AND q.emb IS NOT NULL
     ORDER BY m.embedding <=> q.emb
     LIMIT 50
),
txt AS (
    SELECT m.id FROM memories m, q
     WHERE m.user_id = %(user_id)s AND m.project_id = %(project_id)s
       AND (%(include_history)s OR m.status = 'active')
       AND (%(types)s::text[] IS NULL OR m.type = ANY(%(types)s::text[]))
       AND m.search_tsv @@ q.tsq
     ORDER BY ts_rank_cd(m.search_tsv, q.tsq, 32) DESC
     LIMIT 50
)
SELECT m.id, m.type, m.topic, m.content, m.rationale, m.status, m.pinned,
       m.source_tool, m.created_at, m.supersedes_id,
       %(w_vec)s    * coalesce(1 - (m.embedding <=> q.emb), 0)
     + %(w_text)s   * ts_rank_cd(m.search_tsv, q.tsq, 32)
     + %(w_recent)s * power(0.5, extract(epoch FROM now() - m.created_at) / 86400
                                 / %(half_life)s)
     + %(w_pin)s    * m.pinned::int
     - %(penalty)s  * (m.status = 'superseded')::int AS score
  FROM memories m, q
 WHERE m.user_id = %(user_id)s
   AND m.id IN (SELECT id FROM vec UNION SELECT id FROM txt)
 ORDER BY score DESC, m.created_at DESC
 LIMIT %(limit)s
"""


def search(
    conn: Conn,
    user_id: UUID,
    project_id: UUID,
    vector: str | None,
    query: str,
    limit: int,
    include_history: bool,
    types: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """Ranked memories for one project. vector=None means keyword-only (vec_sim = 0)."""
    # Filtered HNSW scans can stop short; iterative scan keeps going until enough match.
    conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
    return conn.execute(
        SEARCH_SQL,
        {
            "user_id": user_id,
            "project_id": project_id,
            "emb": vector,
            "query": query,
            "include_history": include_history,
            "types": list(types) if types is not None else None,  # text[] needs a list
            "limit": limit,
            # Read at call time, so tests can monkeypatch config.
            "w_vec": config.W_VEC,
            "w_text": config.W_TEXT,
            "w_recent": config.W_RECENT,
            "w_pin": config.W_PIN,
            "half_life": config.HALF_LIFE_DAYS,
            "penalty": SUPERSEDED_PENALTY,
        },
    ).fetchall()
