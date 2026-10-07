"""Apply db/migrations/*.sql in order, skipping versions already in schema_migrations.

Run with: uv run python -m neurax.migrate
"""

import sys
from pathlib import Path

import psycopg

from neurax import config

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "db" / "migrations"


def applied_versions(conn: psycopg.Connection) -> set[str]:
    # On a fresh database schema_migrations does not exist yet (0001 creates it).
    if conn.execute("SELECT to_regclass('public.schema_migrations')").fetchone()[0] is None:  # type: ignore[index]
        return set()
    # Every version that has already run.
    return {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}


def migrate() -> None:
    # Autocommit: each migration file manages its own BEGIN/COMMIT.
    with psycopg.connect(config.DATABASE_URL, autocommit=True) as conn:
        done = applied_versions(conn)
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            version = path.stem
            if version in done:
                print(f"skipped {version} (already applied)")
                continue
            # No params, so psycopg sends the file as-is (simple query protocol, multi-statement).
            conn.execute(path.read_text(encoding="utf-8"))  # type: ignore[arg-type]
            # Record the version; 0001_init records itself, so ignore the duplicate.
            conn.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s) "
                "ON CONFLICT (version) DO NOTHING",
                (version,),
            )
            print(f"applied {version}")


if __name__ == "__main__":
    try:
        migrate()
    except psycopg.Error as e:
        print(f"migration failed: {e}", file=sys.stderr)
        sys.exit(1)
