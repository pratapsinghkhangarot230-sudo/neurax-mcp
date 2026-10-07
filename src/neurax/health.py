"""Health check: DB reachable and the vector extension installed.

Run with: uv run python -m neurax.health  (exit code 0 = healthy)
"""

import sys

import psycopg

from neurax import config


def main() -> int:
    try:
        with psycopg.connect(config.DATABASE_URL, connect_timeout=5) as conn:
            # Server version, to prove the connection works.
            pg_version = conn.execute("SHOW server_version").fetchone()[0]  # type: ignore[index]
            # Installed (not just available) pgvector extension.
            row = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
    except psycopg.Error as e:
        print(f"FAIL: cannot connect to database: {e}", file=sys.stderr)
        return 1

    print(f"OK: connected, PostgreSQL {pg_version}")
    if row is None:
        print("FAIL: vector extension not installed; run: uv run python -m neurax.migrate")
        return 1
    print(f"OK: vector extension {row[0]} installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
