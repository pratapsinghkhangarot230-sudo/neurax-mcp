"""Decision-drift benchmark: Neurax vs flat memory, retrieval only (no LLM calls).

    uv run python -m bench.run

Prints the results and writes bench/RESULTS.md. Uses its own database (neurax_bench,
created if missing) and a fresh user per run, so nothing is ever deleted.
"""

import os

BENCH_DATABASE_URL = os.environ.get(
    "NEURAX_BENCH_DATABASE_URL", "postgresql://neurax:neurax@127.0.0.1:5432/neurax_bench"
)
# Point neurax at the bench database before any neurax module reads config.
os.environ["DATABASE_URL"] = BENCH_DATABASE_URL

import uuid  # noqa: E402
from datetime import date, datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402
from uuid import UUID  # noqa: E402

import psycopg  # noqa: E402
from psycopg import sql  # noqa: E402

from bench import dataset  # noqa: E402
from neurax import config, memory, migrate  # noqa: E402
from neurax.db import get_pool, transaction  # noqa: E402

RESULTS_PATH = Path(__file__).parent / "RESULTS.md"
SYSTEMS = ("Neurax", "Neurax (conflict-gated)", "Flat", "Flat + recency")
TOP_K = 5  # results fetched per question; flat + recency re-sorts these by date
THRESHOLDS = [round(0.60 + 0.02 * i, 2) for i in range(11)]
CATEGORIES = ("unchanged", "same topic", "new topic")
# Genuine data bugs fixed after the dataset commit (63b75f0), as "commit: what".
DATA_FIXES: list[str] = []

Candidates = list[dict[str, Any]]  # one save: every other current memory, see candidates()


# --------------------------------------------------------------------------- setup


def create_database() -> None:
    dbname = str(psycopg.conninfo.conninfo_to_dict(BENCH_DATABASE_URL)["dbname"])
    admin_url = psycopg.conninfo.make_conninfo(BENCH_DATABASE_URL, dbname="postgres")
    with psycopg.connect(admin_url, autocommit=True) as conn:
        # Create the bench database once; runs are isolated by user, never dropped.
        if conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone():
            return
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))


def backdate(user_id: UUID, memory_id: str, age_days: int) -> None:
    with transaction() as conn:
        # Bench only: give the memory its timeline age so the recency weight acts as in
        # real use. The guard trigger allows this on active rows, so call before superseding.
        conn.execute(
            """UPDATE memories SET created_at = now() - make_interval(days => %s)
                WHERE id = %s AND user_id = %s""",
            (age_days, memory_id, user_id),
        )


def candidates(user_id: UUID, new_id: str) -> Candidates:
    with transaction() as conn:
        # Every other current memory in the new one's project, with what _conflicts filters
        # on (same topic, cosine similarity), so the threshold sweep can replay it offline.
        return conn.execute(
            """SELECT o.id::text AS id, o.topic = n.topic AS same_topic,
                      1 - (o.embedding <=> n.embedding) AS similarity
                 FROM memories n
                 JOIN memories o ON o.user_id = n.user_id AND o.project_id = n.project_id
                WHERE n.id = %s AND n.user_id = %s
                  AND o.status = 'active' AND o.id <> n.id""",
            (new_id, user_id),
        ).fetchall()


def build(
    user_id: UUID, p: dataset.Project, mode: str
) -> tuple[str, dict[str, str], list[Candidates], list[str]]:
    """Save the timeline into its own Neurax project.

    mode "neurax" supersedes every replaced event, "gated" only when the old memory shows
    up in possible_conflicts (an agent can only replace what it is shown), "flat" never.
    Returns (slug, event id -> memory id, candidates per save (neurax only), missed links).
    """
    slug = f"{p['slug']}-{mode}"
    memory.create_project(user_id, slug, p["name"])
    last_day = p["events"][-1]["day"]
    ids: dict[str, str] = {}
    saves: list[Candidates] = []
    missed: list[str] = []
    for e in p["events"]:
        r = memory.save(
            user_id, slug, e["content"], e["type"], e["topic"],
            rationale=e.get("rationale"), source_tool="bench",
        )  # fmt: skip
        assert not r["duplicate"], f"{slug} {e['id']} deduplicated"
        new = ids[e["id"]] = r["memory"]["id"]
        backdate(user_id, new, last_day - e["day"])
        old = e.get("supersedes")
        if mode == "neurax":
            cands = candidates(user_id, new)
            for c in cands:
                c["replaced"] = old is not None and c["id"] == ids[old]
            saves.append(cands)
        if old is None or mode == "flat":
            continue
        if mode == "gated" and ids[old] not in {c["id"] for c in r["possible_conflicts"]}:
            missed.append(f"{p['slug']} {old} -> {e['id']}")
            continue
        memory.supersede(user_id, ids[old], new)
    return slug, ids, saves, missed


# --------------------------------------------------------------------------- metrics


def ranked(user_id: UUID, slug: str, query: str, by_recency: bool) -> list[str]:
    results = memory.search(user_id, slug, query, limit=TOP_K)
    if by_recency:
        results.sort(key=lambda m: datetime.fromisoformat(m["created_at"]), reverse=True)
    return [m["id"] for m in results]


def flagged(cands: Candidates, threshold: float) -> Candidates:
    """What possible_conflicts returns at this threshold: the rule and order of _conflicts."""
    hits = [c for c in cands if c["same_topic"] or (c["similarity"] or 0) >= threshold]
    hits.sort(key=lambda c: (not c["same_topic"], -(c["similarity"] or 0)))
    return hits[:5]


def conflict_scores(
    saves: list[Candidates], threshold: float, cross_only: bool = False
) -> tuple[float, float, float]:
    """(precision, recall, F2) of possible_conflicts against the dataset's supersede links."""
    tp = fp = positives = 0
    for cands in saves:
        for c in flagged(cands, threshold):
            if not (cross_only and c["same_topic"]):
                tp += c["replaced"]
                fp += not c["replaced"]
        positives += sum(c["replaced"] and not (cross_only and c["same_topic"]) for c in cands)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else 0.0
    # F2 weights recall over precision: a missed conflict leaves an outdated decision live,
    # a false one only costs the agent a few tokens to read.
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    return precision, recall, f2


def pct(hits: list[bool]) -> str:
    return f"{100 * sum(hits) / len(hits):.0f}%" if hits else "-"


def results_table(rows: list[dict[str, Any]]) -> str:
    """Rows are per (system, question): top1, stale, category."""
    n = sum(1 for r in rows if r["system"] == SYSTEMS[0])
    changed = sum(1 for r in rows if r["system"] == SYSTEMS[0] and r["category"] != "unchanged")
    counts = {
        c: sum(r["category"] == c for r in rows if r["system"] == SYSTEMS[0]) for c in CATEGORIES
    }
    out = [
        f"| System | Top-1 (n={n}) | Stale in top 3 (n={changed} changed) | "
        + " | ".join(f"Top-1 {c} (n={counts[c]})" for c in CATEGORIES)
        + " |",
        "|---|---|---|" + "---|" * len(CATEGORIES),
    ]
    for s in SYSTEMS:
        mine = [r for r in rows if r["system"] == s]
        cells = [
            pct([r["top1"] for r in mine]),
            pct([r["stale"] for r in mine if r["category"] != "unchanged"]),
            *(pct([r["top1"] for r in mine if r["category"] == c]) for c in CATEGORIES),
        ]
        out.append(f"| {s} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def sweep_table(saves: list[Candidates]) -> str:
    out = [
        "| Threshold | Precision | Recall | F2 | Cross-topic precision | Cross-topic recall |",
        "|---|---|---|---|---|---|",
    ]
    for t in THRESHOLDS:
        p, r, f2 = conflict_scores(saves, t)
        cp, cr, _ = conflict_scores(saves, t, cross_only=True)
        out.append(f"| {t:.2f} | {p:.2f} | {r:.2f} | {f2:.2f} | {cp:.2f} | {cr:.2f} |")
    return "\n".join(out)


# --------------------------------------------------------------------------- run


def main() -> None:
    create_database()
    migrate.migrate()
    projects = dataset.load()
    user_id = memory.ensure_user(f"bench-{uuid.uuid4().hex}@example.com")

    rows: list[dict[str, Any]] = []
    saves: dict[str, list[Candidates]] = {s: [] for s in dataset.SPLITS}
    missed: list[str] = []
    for p in projects:
        nx_slug, nx_ids, nx_saves, _ = build(user_id, p, "neurax")
        gated_slug, gated_ids, _, gated_missed = build(user_id, p, "gated")
        flat_slug, flat_ids, _, _ = build(user_id, p, "flat")
        saves[p["split"]] += nx_saves
        missed += gated_missed
        runs = [
            (SYSTEMS[0], nx_slug, nx_ids, False),
            (SYSTEMS[1], gated_slug, gated_ids, False),
            (SYSTEMS[2], flat_slug, flat_ids, False),
            (SYSTEMS[3], flat_slug, flat_ids, True),
        ]
        for q in p["questions"]:
            outdated = dataset.predecessors(p, q["answer"])
            for system, slug, ids, by_recency in runs:
                top = ranked(user_id, slug, q["q"], by_recency)
                rows.append({
                    "split": p["split"],
                    "system": system,
                    "category": dataset.category(p, q["answer"]),
                    "top1": bool(top) and top[0] == ids[q["answer"]],
                    "stale": any(ids[o] in top[:3] for o in outdated),
                })  # fmt: skip

    # Recommend from calibration only: best F2, ties to the higher (quieter) threshold.
    best = max(THRESHOLDS, key=lambda t: (conflict_scores(saves["calibration"], t)[2], t))
    current = config.CONFLICT_THRESHOLD

    def split_line(split: str) -> str:
        names = [p["slug"] for p in projects if p["split"] == split]
        return f"- **{split}** ({len(names)} projects): {', '.join(names)}"

    def heldout_check(t: float) -> str:
        p_, r, f2 = conflict_scores(saves["heldout"], t)
        cp, cr, _ = conflict_scores(saves["heldout"], t, cross_only=True)
        return f"| {t:.2f} | {p_:.2f} | {r:.2f} | {f2:.2f} | {cp:.2f} | {cr:.2f} |"

    report = f"""# Decision-drift benchmark results

Generated by `uv run python -m bench.run` on {date.today().isoformat()}. Retrieval only: no LLM
calls, so the numbers are deterministic for a given embedding model and dataset.

## Setup

- Dataset: `bench/data/`, {len(projects)} hand-written projects,
  {sum(len(p["events"]) for p in projects)} events,
  {sum(len(p["questions"]) for p in projects)} questions.
  Each question's answer is the current version of a decision.
- Split, fixed in the data files before the first run:
{split_line("calibration")}
{split_line("heldout")}
- The threshold sweep and recommendation use **calibration** only.
  The headline table is **held-out**.
- Embeddings: `{config.EMBED_MODEL}`. Ranking weights: vec {config.W_VEC}, text {config.W_TEXT},
  recency {config.W_RECENT}, pin {config.W_PIN}, half-life {config.HALF_LIFE_DAYS:g} days.
- Each event is backdated to its timeline day (the latest event is 0 days old), so recency
  acts as it would in real use.

### Systems (same Postgres, same `memory.save` / `memory.search`, same embeddings)

- **Neurax**: every change the dataset declares is applied with `supersede`; search returns
  current memories only.
- **Neurax (conflict-gated)**: a change is applied only if the old memory appears in the new
  one's `possible_conflicts` at the current threshold ({current:.2f}). This is what an agent that
  only replaces what it is shown would do.
- **Flat**: every event saved as active, never superseded, same hybrid search.
- **Flat + recency**: Flat's top {TOP_K} results, re-sorted newest first.

### Metrics

- **Top-1**: the first result is the question's current decision.
- **Stale in top 3**: an outdated version of that decision appears in the top 3. Counted over
  questions whose decision changed (an unchanged decision has no outdated version).
- Question categories: **unchanged**, **same topic** (every version shares a topic),
  **new topic** (at least one change was saved under a different topic).

## Results: held-out (headline)

{results_table([r for r in rows if r["split"] == "heldout"])}

## Results: all 20 projects (reference)

{results_table(rows)}

Supersede links the conflict-gated run missed at {current:.2f} (old memory not in
`possible_conflicts`): {len(missed)}{"" if not missed else ": " + "; ".join(missed)}.

## Conflict threshold (calibration projects)

Positives are the dataset's supersede links. For each save, `possible_conflicts` is replayed
offline with the same rule as `memory._conflicts` (same topic, or similarity >= threshold;
same-topic first; at most 5). Same-topic memories are always included, so the threshold only
changes cross-topic results; the cross-topic columns isolate that.

{sweep_table(saves["calibration"])}

**Recommended threshold: {best:.2f}** (highest F2 on calibration; F2 weights recall twice as
much as precision). Current default: {current:.2f}. Not changed; needs Pratap's approval.

Held-out check of both values:

| Threshold | Precision | Recall | F2 | Cross-topic precision | Cross-topic recall |
|---|---|---|---|---|---|
{heldout_check(current)}
{heldout_check(best)}

## Data fixes after the dataset commit

{chr(10).join(f"- {f}" for f in DATA_FIXES) or "None."}
"""
    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(report)
    get_pool().close()


if __name__ == "__main__":
    main()
