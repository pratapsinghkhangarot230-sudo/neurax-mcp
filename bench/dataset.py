"""The decision-drift dataset: one JSON file per project in bench/data/.

Each project has a timeline of events (memories, in save order) where an event may
`supersede` an earlier one, and questions whose `answer` is the current event.
"""

import json
from pathlib import Path
from typing import Any

from neurax.memory import SLUG_RE, TOPIC_RE, TYPES

DATA_DIR = Path(__file__).parent / "data"
SPLITS = ("calibration", "heldout")

Project = dict[str, Any]


def load() -> list[Project]:
    """All projects in file order; raises ValueError on the first invalid item."""
    projects = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(DATA_DIR.glob("*.json"))]
    for p in projects:
        check(p)
    return projects


def check(p: Project) -> None:
    def fail(msg: str) -> None:
        raise ValueError(f"{p.get('slug', '?')}: {msg}")

    # "-gated" is the longest suffix run.py appends to the slug.
    if not SLUG_RE.fullmatch(p["slug"] + "-gated"):
        fail("invalid slug")
    if p["split"] not in SPLITS:
        fail(f"split must be one of {SPLITS}")
    seen: set[str] = set()
    replaced: set[str] = set()
    contents: set[str] = set()
    last_day = 0
    for e in p["events"]:
        if e["id"] in seen:
            fail(f"duplicate event id {e['id']}")
        if e["type"] not in TYPES or not TOPIC_RE.fullmatch(e["topic"]):
            fail(f"{e['id']}: bad type or topic")
        if e["day"] < last_day:
            fail(f"{e['id']}: events must be in day order")
        if (old := e.get("supersedes")) is not None:
            if old not in seen:
                fail(f"{e['id']} supersedes {old}, which is not an earlier event")
            if old in replaced:
                fail(f"{old} is superseded twice")
            replaced.add(old)
        # Identical active content would be deduplicated by the store (flat keeps all active).
        if (key := e["content"].strip().lower()) in contents:
            fail(f"{e['id']}: duplicate content")
        seen.add(e["id"])
        contents.add(key)
        last_day = e["day"]
    for q in p["questions"]:
        if q["answer"] not in seen:
            fail(f"question {q['q']!r}: unknown answer {q['answer']}")
        if q["answer"] in replaced:
            fail(f"question {q['q']!r}: answer {q['answer']} is outdated")


def predecessors(p: Project, event_id: str) -> list[str]:
    """The events event_id replaced, newest first: the outdated versions of its decision."""
    by_id = {e["id"]: e for e in p["events"]}
    out = []
    while (old := by_id[event_id].get("supersedes")) is not None:
        out.append(old)
        event_id = old
    return out


def category(p: Project, answer: str) -> str:
    """unchanged, same topic (every version shares a topic) or new topic (some change moved it)."""
    chain = [answer, *predecessors(p, answer)]
    if len(chain) == 1:
        return "unchanged"
    topics = {e["topic"] for e in p["events"] if e["id"] in chain}
    return "same topic" if len(topics) == 1 else "new topic"
