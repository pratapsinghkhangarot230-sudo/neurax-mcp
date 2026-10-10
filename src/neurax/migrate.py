"""Apply db/migrations/*.sql in order, skipping versions already in schema_migrations.

Run with: uv run neurax-migrate
"""

import sys
from pathlib import Path

import psycopg
from psycopg import sql

from neurax import config

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "db" / "migrations"


def applied_versions(conn: psycopg.Connection) -> set[str]:
    # On a fresh database schema_migrations does not exist yet (0001 creates it).
    if conn.execute("SELECT to_regclass('public.schema_migrations')").fetchone()[0] is None:  # type: ignore[index]
        return set()
    # Every version that has already run.
    return {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}


def create_database() -> None:
    """Create the database DATABASE_URL points at, if missing (never drops anything)."""
    dbname = str(psycopg.conninfo.conninfo_to_dict(config.DATABASE_URL)["dbname"])
    admin_url = psycopg.conninfo.make_conninfo(config.DATABASE_URL, dbname="postgres")
    with psycopg.connect(admin_url, autocommit=True) as conn:
        # Does the database already exist?
        if conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone():
            return
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))


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


def main() -> None:
    try:
        migrate()
    except psycopg.Error as e:
        print(f"migration failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
